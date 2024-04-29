"""Full training pipeline using data parallelism."""

import os
import functools
from typing import Any
from absl import logging
import jax
from jax import numpy as jnp
from flax.training import train_state
from flax import linen as nn, jax_utils, struct
import gin
from tqdm import tqdm
import wandb

import dataloader
import utils


SCRATCH = os.getenv("SCRATCH", "~/scratch")


@gin.configurable
class Trainer:
    """Implements a JAX training loop."""

    def __init__(
            self, 
            seed: int = 0, 
            num_steps: int = gin.REQUIRED, 
            model_definition: nn.Module = gin.REQUIRED,
            create_dataset_fn: dataloader.CustomLoaderFn = gin.REQUIRED, 
            create_preprocess_fn: Any = gin.REQUIRED,
            create_optimizer_fn: Any = gin.REQUIRED, 
            create_platuea_fn: Any | None = gin.REQUIRED, 
            create_loss_fn: Any = gin.REQUIRED,
            use_wandb: bool = True,
    ):
        self.seed = seed
        self.num_steps = num_steps
        self.model_definition = model_definition
        self.create_dataset_fn = create_dataset_fn
        self.create_preprocess_fn = create_preprocess_fn
        self.create_optimizer_fn = create_optimizer_fn
        self.create_platuea_fn = create_platuea_fn
        self.create_loss_fn = create_loss_fn
        
        config_dict = gin.config_str().split('\n')
        config_dict = {line.split(' ')[0]: line.split(' ')[-1] for line in config_dict if line}
        wandb_dir = os.path.join(SCRATCH, 'wandb')
        if use_wandb:
            wandb.init(project="LRA SSM", job_type='model_training', config=config_dict, dir=wandb_dir)
        else:
            wandb.init(mode='offline', config=config_dict, dir='wandb_offline', dir=wandb_dir)


    def create_training_state(self, key, dummy_inputs):
        model = self.batched_model_definition(training=True)
        key, key_params, key_dropout = jax.random.split(key, 3)
        init_rngs = {"params": key_params, "dropout": key_dropout}
        # p_init = jax.pmap(model.init, axis_name='i')
        # init_rngs = utils.broadcast_to_local_devices(init_rngs)
        # dummy_inputs = utils.broadcast_to_local_devices(dummy_inputs)
        variables = model.init(init_rngs, dummy_inputs)
        params = variables["params"]
        batch_stats = variables.get("batch_stats", None)
        fn_is_complex = lambda x: x.dtype in [jnp.complex64, jnp.complex128]
        param_sizes = utils.map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
        logging.info(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
        if batch_stats is not None:
            class TrainStateWithBatchStats(utils.TrainStateX):
                batch_stats: Any
            training_sate = TrainStateWithBatchStats.create(
                apply_fn=model.apply,
                params=params,
                batch_stats=batch_stats,
                tx=self.optimizer,
                plateau_tx=self.plateau)
        else:
            training_sate = train_state.TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=self.optimizer,
                plateau_tx=self.plateau)
        return jax_utils.replicate(training_sate)


    def train_epoch(self, key):
        train_metrics = []
        for batch in tqdm(self.trainloader):
            # preprocess the data
            inputs, targets = self.preprocess_fn(batch)
            # create training state if nonexistent
            if self.training_state is None:
                key, key_init = jax.random.split(key)
                self.training_state = self.create_training_state(key=key_init, dummy_inputs=inputs)
            
            key, key_step = jax.random.split(key)
            key_step = jax_utils.replicate(key_step)
            inputs, targets = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (self.num_local_devices, -1) + x.shape[1:]), (inputs, targets))
            self.training_state, metrics = self.train_step(key_step, self.training_state, inputs, targets)
            train_metrics.append(metrics)

        train_metrics = jax.tree_util.tree_map(lambda *x: jnp.stack(x).mean(), *train_metrics)
        # TODO(mahanfathi): log metrics properly
        logging.info("Training Metrics: %r", train_metrics)
        wandb.log({"training/"+key: val for key, val in train_metrics.items()})
    

    @functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def train_step(self, key, training_state, inputs, targets):
        key, key_dropout = jax.random.split(key)
        bs_dict = {}
        bs_list = []
        if hasattr(training_state, "batch_stats"):
            bs_dict = {"batch_stats": training_state.batch_stats}
            bs_list = ["batch_stats"]
        pred_fn = lambda params: training_state.apply_fn(
            {**{"params": params}, **bs_dict}, 
            inputs, 
            rngs={"dropout": key_dropout}, 
            mutable=["intermediates"]+bs_list)
        def loss_fn(params):
            preds, mutes = pred_fn(params)
            loss, metrics = self.loss_fn(preds, targets)
            return loss, (mutes, metrics)
        (loss, (mutes, metrics)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(training_state.params)
        metrics = jax.lax.pmean(metrics, axis_name="batch") # average metrics across the batch
        grads = jax.lax.pmean(grads, axis_name="batch")
        if hasattr(training_state, "batch_stats"):
            new_batch_stats = jax.lax.pmean(mutes["batch_stats"], axis_name='batch')
            training_state = training_state.replace(batch_stats=new_batch_stats)
        training_state = training_state.apply_gradients(grads=grads)
        return training_state, metrics


    def evaluate_epoch(self, key, data_loader, scope="eval"):
        del key
        eval_metrics = []
        eval_model = self.batched_model_definition(training=False)
        bs_dict = {}
        @functools.partial(jax.pmap, axis_name="batch")
        def eval_fn(training_state, inputs, targets):
            bs_dict = {}
            if hasattr(self.training_state, "batch_stats"):
                bs_dict = {"batch_stats": training_state.batch_stats}
            preds = eval_model.apply({**{"params": training_state.params}, **bs_dict}, inputs)
            return jax.lax.pmean(self.loss_fn(preds, targets), axis_name="batch")
        for batch in tqdm(data_loader):
            inputs, targets = self.preprocess_fn(batch)
            inputs, targets = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (self.num_local_devices, -1) + x.shape[1:]), (inputs, targets))
            loss, metrics = eval_fn(self.training_state, inputs, targets)
            eval_metrics.append(metrics)
        eval_metrics = jax.tree_util.tree_map(lambda *x: jnp.stack(x).mean(), *eval_metrics)
        logging.info("Evaluation Metrics: %r", eval_metrics)
        wandb.log({f"{scope}/{key}": val for key, val in eval_metrics.items()})
        return eval_metrics


    @functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0,), in_axes=(None, 0, None))
    def handle_plateau(self, training_state, metric):
        return training_state.monitor_plateau(metric=metric)


    def train(self):
        """Train the model."""

        self.num_local_devices = jax.local_device_count()
        logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
        logging.info("JAX local devices: %r", jax.local_devices())

        # get dataloaders and some basic data properties
        self.trainloader, self.valloader, self.testloader, \
            aux_dataloaders, n_classes, input_len, in_dim, train_size = self.create_dataset_fn()

        # create preprocess function
        self.preprocess_fn = self.create_preprocess_fn(input_len=input_len, in_dim=in_dim)

        # create loss function
        self.loss_fn = self.create_loss_fn()

        # create optimizer
        self.optimizer = self.create_optimizer_fn()

        # create lr scaler at val plateau
        self.plateau = self.create_platuea_fn() if self.create_platuea_fn is not None else None

        # update model definition
        self.model_definition = functools.partial(self.model_definition, d_output=n_classes, input_len=input_len)

        # call vmap to parallelize across a batch of input sequences
        self.batched_model_definition = nn.vmap(
            self.model_definition, in_axes=(0,), out_axes=0,
            variable_axes={"params": None, "dropout": None, "batch_stats": None, "cache": 0, "prime": None},
            split_rngs={"params": False, "dropout": True}, axis_name="minibatch",
        )

        # PRNG key initialization
        key = jax.random.PRNGKey(self.seed)

        # get number of epochs
        num_epochs = self.num_steps // train_size + 1

        logging.info("STARTED TRAINING FOR %r EPOCHS.", num_epochs)

        self.training_state = None
        for epoch_num in range(num_epochs):
            key = jax.random.fold_in(key, epoch_num)
            key_train, key_val, key_test = jax.random.split(key, 3)
            # train for one epoch
            logging.info("Training epoch %d", epoch_num)
            self.train_epoch(key_train)
            # evaluate the model
            logging.info("Evaluating epoch %d", epoch_num)
            val_metrics = self.evaluate_epoch(key_val, self.valloader, scope="val")
            self.training_state = self.handle_plateau(self.training_state, val_metrics["loss"])
            _ = self.evaluate_epoch(key_test, self.testloader, scope="test")
