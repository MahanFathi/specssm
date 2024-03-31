"""General-purpose utilities."""

from typing import Callable

import chex
import jax

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