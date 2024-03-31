"""Full training pipeline using data parallelism."""

import functools
from typing import Any
from absl import logging
import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
import gin

import dataloader
import utils


@gin.configurable
class Trainer(object):
    """Implements a JAX training loop."""
    seed: int = 0
    num_steps: int = gin.REQUIRED
    model_definition: nn.Module = gin.REQUIRED
    create_dataset_fn: dataloader.CustomLoaderFn = gin.REQUIRED
    create_preprocess_fn: Any = gin.REQUIRED
    create_optimizer_fn: Any = gin.REQUIRED
    create_loss_fn: Any = gin.REQUIRED


    def create_training_state(self, key, dummy_inputs):
        model = self.batched_model_definition(training=True)
        p_init = jax.pmap(model.init, axis_name='i')
        key, key_params, key_dropout = jax.random.split(key, 3)
        init_rngs = {"params": key_params, "dropout": key_dropout}
        init_rngs = utils.broadcast_to_local_devices(init_rngs)
        if isinstance(dummy_inputs, tuple):
            dummy_inputs = dummy_inputs[0]
        params = p_init(init_rngs, dummy_inputs)
        return flax.training.train_state.TrainState.create(
            apply_fn=model.apply,
            params=params['params'],
            tx=self.optimizer,
        )


    def train_epoch(self, key):
        train_metrics = []
        for batch in self.trainloader:
            # preprocess the data
            inputs, targets = self.preprocess_fn(batch)
            # create training state if nonexistent
            if self.training_state is None:
                key, key_init = jax.random.split(key)
                self.training_state = self._create_training_state(key=key_init, dummy_inputs=inputs)
            
            key, key_step = jax.random.split(key)
            key_step = utils.broadcast_to_local_devices(key_step)
            self.training_state, metrics = self.train_step(key_step, self.training_state, inputs, targets)
            train_metrics.append(metrics)
        train_metrics = jax.tree_util.tree_map(lambda *x: jnp.stack(x).mean(), *train_metrics)
        # TODO(mahanfathi): log metrics properly
        logging.info("Training Metrics: %r", train_metrics)
    

    @functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def train_step(self, key, training_state, inputs, targets):
        key, key_dropout = jax.random.split(key)
        (loss, metrics), grads = jax.value_and_grad(
            lambda params: self.loss_fn(
                training_state.apply_fn(inputs, params=params, dropout_rngs=key_dropout), 
                targets
            ),
            has_aux=True,
        )(training_state.params)
        metrics = jax.lax.pmean(metrics, axis_name='batch') # average metrics across the batch
        grads = jax.lax.pmean(grads, axis_name='batch')
        training_state = training_state.apply_gradients(grads=grads)
        return training_state, metrics


    def evaluate_epoch(self):
        eval_metrics = []
        eval_model = self.batched_model_definition(training=False)
        @jax.pmap
        def eval_fn(params, inputs, targets):
            return jax.lax.pmean(self.loss_fn(eval_model.apply(inputs, params=params), targets))
        for batch in self.testloader:
            inputs, targets = self.preprocess_fn(batch)
            loss, metrics = eval_fn(self.training_state.params, inputs, targets)
            eval_metrics.append(metrics)
        eval_metrics = jax.tree_util.tree_map(lambda *x: jnp.stack(x).mean(), *eval_metrics)
        logging.info("Evaluation Metrics: %r", eval_metrics)


    def train(self):

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

        # update model definition
        self.model_definition = functools.partial(self.model_definition, d_output=n_classes, input_len=input_len)

        # call vmap to parallelize across a batch of input sequences
        self.batched_model_definition = nn.vmap(
            self.model_definition, in_axes=(0,), out_axes=0,
            variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
            split_rngs={"params": False, "dropout": True}, axis_name='batch',
        )

        # model = batched_model_definition(training=True)

        # PRNG key initialization
        key = jax.random.PRNGKey(self.seed)

        # get number of epochs
        num_epochs = self.num_steps // train_size + 1

        self.training_state = None
        for epoch_num in range(num_epochs):
            key = jax.random.fold_in(key, epoch_num)
            key, key_train, key_eval = jax.random.split(key, 3)
            # train for one epoch
            logging.info("Training epoch %d", epoch_num)
            self.train_epoch(key_train)
            # evaluate the model
            logging.info("Evaluating epoch %d", epoch_num)
            self.evaluate_epoch(key_eval)
