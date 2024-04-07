import jax
from jax import numpy as jnp

import numpy as np


def position_encoding(num_positions: int, input_dim: int, max_wavelength: float = 0):
    """Returns a position encoding of shape (num_positions, input_dim).

    Positions are encoded as sin/cos pairs at geometrically increasing
    wavelengths.

    The length of a half-wave (peak to trough) increases geometrically from 1 to
    max_wavelength.  (Technically, it's slightly less; the last sin/cos pair has
    a wavelength of max_wavelength**((d-1)/d), where d = input_dim/2.)

    NOTE: unlike prior published position encodings, we multiply the position of
    each token by pi to convert from fractions of a wave (position/wavelength)
    to radians.  Thus, the highest frequency wave alternates between -1 and 1 on
    every token, whereas in prior published work the highest frequency alternates
    between -1 and 1 every pi tokens.  The max_wavelength is also effectively
    1/pi times as long, so a prior published factor of 10,000
    (e.g. https://arxiv.org/abs/1706.03762) would equate to a max_wavelength
    of 31,416.

    This encoding also does not alternate between sin/cos values, but puts all of
    the cos values on one side, and the sin values on the other.  That makes it
    easier to split the sin,cos values to construct or apply a rotation matrix.

    The default value for max_wavelength is 2 * num_positions.

    Args:
      num_positions:  The number of positions.
      input_dim:      The dimension of the position vector.
      max_wavelength: The maximum length of a half-wave (peak to trough)

    Returns:
      Numpy matrix of shape (num_positions, input_dim) containing the encodings.
      Position encodings are packed as concat(cos_values, sin_values, axis=1).
    """

    if max_wavelength == 0:
        max_wavelength = 2 * num_positions
    assert max_wavelength > 1

    assert (input_dim % 2) == 0
    idim2 = input_dim // 2

    # t ranges from 0 <= t < 1
    t = np.arange(0, idim2, dtype=np.float32) / idim2

    # wavelength (columns)
    # The length of a half-wave (trough to peak) increases geometrically
    # 1 <= wavelength < max_wavelength
    wavelength = float(max_wavelength)**t
    wavelength = np.reshape(wavelength, (1, idim2))  # broadcast over rows

    # k is the position in the sequence (rows)
    k = np.arange(0, num_positions, dtype=np.float32)
    k = np.reshape(k, (num_positions, 1))  # broadcast over columns

    # For each position (row) compute an angle (column) at various wavelengths.
    # NOTE: unlike prior published work, we multiply by pi to convert to radians.
    pi_f = np.array(np.pi, dtype=np.float32)
    angles = pi_f * k / wavelength  # shape (num_positions, idim2)
    posx = np.cos(angles, dtype=np.float32)
    posy = np.sin(angles, dtype=np.float32)
    return np.concatenate([posx, posy], axis=1)  # shape (num_positions, idim)


def rotate_x(x: jnp.ndarray, max_wavelength: float = 0):
    """Rotate keys and queries by the relative distance between query and key.

    Implements rotary position embeddings (RoPE) https://arxiv.org/abs/2104.09864.

    Args:
      keys: array of shape (batch_size, num_keys, num_heads, head_size)
      queries: aray of shape (batch_size, num_queries, num_heads, head_size)
      max_wavelength: The maximum length of a half-wave (peak to trough)

    Returns:
      (keys, queries) after rotation.
    """

    (input_len, input_dim) = x.shape

    # Get position encodings, which can be used to do a rotation.
    pos = position_encoding(input_len, input_dim, max_wavelength=max_wavelength)
    # Split position encoding into separate sin/cos values in order to
    # construct a rotation matrix.
    (cosa, sina) = np.split(pos, 2, axis=-1)
    cosa = jnp.asarray(cosa, dtype=x.dtype)  # convert from numpy -> jax
    sina = jnp.asarray(sina, dtype=x.dtype)  # convert from numpy -> jax

    # Split keys/queries into real & imaginary (i.e. x & y) parts.
    (xx, xy) = jnp.split(x, 2, axis=-1)
    # Apply rotation matrix.
    xx_rot = (xx * cosa) - (xy * sina)
    xy_rot = (xx * sina) + (xy * cosa)
    # Concatenate back into keys/queries.
    return jnp.concatenate([xx_rot, xy_rot], axis=-1)