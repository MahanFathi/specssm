import jax
import jax.numpy as jnp
import optax
import gin

import utils


@gin.configurable
class WarmupCosineDecay:
    """Cosine decay with linear warmup."""
    def __init__(
        self,
        lr: float,
        start_val: float = gin.REQUIRED,
        min_lr: float = gin.REQUIRED,
        num_steps: int = gin.REQUIRED,
        warmup_steps: int = gin.REQUIRED,
    ) -> None:
        """Initialize a cosine decay schedule with warmup.

        Args:
            start_val: The value to start at.
            min_lr: The minimum value to decay to.
            lr: The peak value to reach.
            num_steps: The total number of steps to decay over.
            warmup_steps: The number of steps to warmup for.
        """
        self.start_val = start_val
        self.min_lr = min_lr
        self.lr = lr
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps

    def __call__(self, itr) -> jax.Array:
        """Get learning rate for a given step.

        Args:
            itr: The current step.

        Returns:
            The learning rate for the given step.
        """
        warmup_val = (self.lr - self.start_val) * (
            itr / self.warmup_steps
        ) + self.start_val

        cos_itr = (itr - self.warmup_steps) / (self.num_steps - self.warmup_steps)
        cos = 1 + jnp.cos(jnp.pi * cos_itr)
        cos_val = 0.5 * (self.lr - self.min_lr) * cos + self.min_lr

        # Select warmup_val if itr < warmup, else cosine val
        values = jnp.array([warmup_val, cos_val])
        index = jnp.sum(jnp.array(self.warmup_steps) < itr)

        return jnp.take(values, index)


@gin.configurable
def create_specssm_optimizer(
    learning_rate: float = 5e-4,
    weight_decay: float = 0.1,
    m_y_learning_rate: float = 5e-5,
    m_y_weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Build AdamW optimizer with linear warmup and cosine decay.

    Args:
        num_steps: The total number of steps to decay over.
        warmup_steps: The number of steps to warmup for.
        learning_rate: The peak learning rate.
        weight_decay: The weight decay to use.
        m_y_learning_rate: The peak learning rate for m_y parameters.
        m_y_weight_decay: The weight decay to use for m_y parameters.

    Returns:
        An optax.GradientTransformation.
    """
    optimizers = {
        'default': optax.adamw(
            learning_rate=WarmupCosineDecay(
                lr=learning_rate),
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            eps_root=0.0,
            weight_decay=weight_decay,
        ),
        'm_y': optax.adamw(
            learning_rate=WarmupCosineDecay(
                lr=m_y_learning_rate),
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            eps_root=0.0,
            weight_decay=m_y_weight_decay,
        ),
    }

    label_fn = utils.map_nested_fn(
        lambda k, _: 'm_y' if k.startswith('m_y') else 'default'
    )

    return optax.multi_transform(optimizers, label_fn)


@gin.configurable
def create_s5_optimizer(
    num_steps: int,
    num_warmup_steps: int,
    learning_rate: optax.ScalarOrSchedule = 5e-4,
    weight_decay: float = 0.1,
    ssm_learning_rate: optax.ScalarOrSchedule = 5e-4,
    opt_config = "standard",
    dt_global = False,
):
    # TODO(mahanfathi): implement schedule for learning rate
    """Create optimizer for S5 model."""
    if opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
        if dt_global:
            ssm_fn = utils.map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = utils.map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_learning_rate),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = utils.map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in ["B"] else "regular")
            )

        else:
            ssm_fn = utils.map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_learning_rate,
                                                              weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_learning_rate),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply 
           faster global learning rate to B also.
        """
        print("configuring optimization with B in AdamW setup with learning_rate")
        if dt_global:
            ssm_fn = utils.map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = utils.map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_learning_rate),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = utils.map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = utils.map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_learning_rate),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

        # tx is the base optimizer, we now proceed to add the learning rate schedule
        schedule = optax.join_schedules(
            schedules=[
              optax.linear_schedule(.1, 1., num_warmup_steps),
              optax.cosine_decay_schedule(1., num_steps - num_warmup_steps)
            ],
            boundaries=[
              num_warmup_steps,
            ]
        )

        tx = optax.chain(
          tx,
          # optax.scale_by_schedule(schedule),
          optax.scale_by_learning_rate(learning_rate=schedule, flip_sign=False),
        )


    return tx