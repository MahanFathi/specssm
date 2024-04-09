"""General-purpose utilities."""

from typing import Any, Callable

import jax
import chex
import optax

from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT


class TrainStateX(struct.PyTreeNode):
    """   
        Extended train state for the common case with 
        a single Optax optimizer
        &
        a plateau optimizer.

        Initial implementation is stolen from the flax library.
    """

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    plateau_tx: optax.GradientTransformationExtraArgs | None = struct.field(pytree_node=False)
    plateau_state: optax.OptState | None = struct.field(pytree_node=True)

    def monitor_plateau(self, *, metric):
        if self.plateau_state is not None:
            _, plateau_state = self.plateau_tx.update(
                  updates=self.params, state=self.plateau_state, value=metric)
            return self.replace(plateau_state=plateau_state)
        return self

    def apply_gradients(self, *, grads, **kwargs):
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads['params']
            params_with_opt = self.params['params']
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
           grads_with_opt, self.opt_state, params_with_opt
        )

        # Apply plateau transformations
        if self.plateau_state is not None:
            updates = optax.tree_utils.tree_scalar_mul(self.plateau_state.scale, updates)

        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        # As implied by the OWG name, the gradients are used directly to update the
        # parameters.
        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
              'params': new_params_with_opt,
              OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, plateau_tx, **kwargs):
        # We exclude OWG params when present because they do not need opt states.
        params_with_opt = (
            params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        )
        opt_state = tx.init(params_with_opt)
        plateau_state = plateau_tx.init(params_with_opt) if plateau_tx else None
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            plateau_tx=plateau_tx,
            plateau_state=plateau_state,
            **kwargs,
        )


def broadcast_to_local_devices(pytree: chex.ArrayTree) -> chex.ArrayTree:
    """Broadcasts a Pytree to all local devices.

    Args:
      pytree: The Pytree to broadcast.

    Returns:
      A Pytree with the same structure as `pytree`, but with values broadcasted
      to all local devices.
    """
    devices = jax.local_devices()
    return jax.tree_util.tree_map(
       lambda v: jax.device_put_sharded(len(devices) * [v], devices), pytree
    )


def map_nested_fn(
    fn: Callable[[str, jax.Array], jax.Array],
) -> Callable[[chex.ArrayTree], chex.ArrayTree]:
    """Recursively apply `fn` to the key-value pairs of a nested dict.

    Example from optax.multi_transform for defining custom schedulers.

    Args:
        fn: local function applied to leaves mapping (k, v) to string key

    Returns:
        function mapping parameter names to key
  """

    def map_fn(nested_dict):
        return {
            k: map_fn(v) if isinstance(v, dict) else fn(k, v)
            for k, v in nested_dict.items()
        }

    return map_fn