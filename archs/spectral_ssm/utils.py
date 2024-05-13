"""Utility functions for spectral SSM."""

import functools
import jax
from jax import numpy as jnp
import numpy as np

Array = jax.Array

def get_hankel_matrix(
    n: int,
) -> Array:
    """Generate a spectral Hankel matrix.

    Args:
        n: Number of rows in square spectral Hankel matrix.

    Returns:
        A spectral Hankel matrix of shape [n, n].
    """
    z = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            z[i - 1, j - 1] = 2 / ((i + j) ** 3 - (i + j))
    return z


def get_top_hankel_eigh(
    n: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get top k eigenvalues and eigenvectors of spectral Hankel matrix.

    Args:
        n: Number of rows in square spectral Hankel matrix.
        k: Number of eigenvalues to return.

    Returns:
        A tuple of eigenvalues of shape [k,] and eigenvectors of shape [l, k].
    """
    eig_vals, eig_vecs = np.linalg.eigh(get_hankel_matrix(n))
    return eig_vals[-k:], eig_vecs[:, -k:]


@jax.jit
def shift(
    x: Array,
) -> Array:
    """Shift time axis by one to align x_{t-1} and x_t.

    Args:
        x: An array of shape [l, d].

    Returns:
        An array of shape [l, d] where index [0, :] is all zeros and [i, :] is equal
        to x[i - 1, :] for i > 0.
    """
    return jnp.pad(x, ((1, 0), (0, 0)), mode='constant')[:-1, :]


@jax.jit
def conv(
    v: Array,
    u: Array,
) -> Array:
    """Compute convolution to project input sequences into the spectral basis.

    Args:
        v: Top k eigenvectors of shape [l, k].
        u: Input of shape [l, d_in].

    Returns:
        A matrix of shape [l, k, d_in]
    """
    # Convolve two vectors of length l (x.shape[0]) and truncate to the l oldest
    # values.
    tr_conv = lambda x, y: jax.scipy.signal.convolve(x, y, method='fft')[
        : x.shape[0]
    ]

    # Convolve each sequence of length l in v with each sequence in u.
    mvconv = jax.vmap(tr_conv, in_axes=(1, None), out_axes=1)
    mmconv = jax.vmap(mvconv, in_axes=(None, 1), out_axes=-1)

    return mmconv(v, u)


@jax.jit
def compute_y_t(
    m_y: Array,
    deltas: Array,
) -> Array:
    """Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y: A matrix of shape [d_out, k, d_out] that acts as windowed transition
        matrix for the linear dynamical system evolving y_t.
        deltas: A matrix of shape [l, d_out].

    Returns:
        A matrix of shape [l, d_out].
    """
    d_out, k, _ = m_y.shape

    def scan_op(carry, x):
        output = jnp.tensordot(m_y, carry, axes=2) + x
        carry = jnp.roll(carry, 1, axis=0)
        carry = carry.at[0].set(output)
        return carry, output

    _, ys = jax.lax.scan(scan_op, jnp.zeros((k, d_out)), deltas)

    return ys


@jax.jit
def compute_ar_x_preds(
    w: Array,
    x: Array,
) -> Array:
    """Compute the auto-regressive component of spectral SSM.

    Args:
        w: An array of shape [d_out, d_in, k].
        x: A single input sequence of shape [l, d_in].

    Returns:
        ar_x_preds: An output of shape [l, d_out].
    """
    d_out, _, k = w.shape
    l = x.shape[0]

    # Contract over `d_in`.
    o = jnp.einsum('oik,li->klo', w, x)

    # For each `i` in `k`, roll the `(l, d_out)` by `i` steps.
    o = jax.vmap(functools.partial(jnp.roll, axis=0))(o, jnp.arange(k))

    # Create a mask that zeros out nothing at `k=0`, the first `(l, d_out)` at
    # `k=1`, the first two `(l, dout)`s at `k=2`, etc.
    m = jnp.triu(jnp.ones((k, l)))
    m = jnp.expand_dims(m, axis=-1)
    m = jnp.tile(m, (1, 1, d_out))

    # Mask and sum along `k`.
    return jnp.sum(o * m, axis=0)


@functools.partial(jax.jit, static_argnums=(2,))
def compute_x_tilde(
    inputs: Array,
    eigh: tuple[Array, Array],
    shift_for_ar: bool = True,
) -> Array:
    """Project input sequence into spectral basis.

    Args:
        inputs: A single input sequence of shape [l, d_in].
        eigh: A tuple of eigenvalues [k] and circulant eigenvecs [l, k].

    Returns:
        x_tilde: An output of shape [l, k * d_in].
    """
    eig_vals, eig_vecs = eigh

    l = inputs.shape[0]
    x_tilde = conv(eig_vecs, inputs)

    # Broadcast an element-wise multiple along the k-sized axis.
    x_tilde *= jnp.expand_dims(eig_vals, axis=(0, 2)) ** 0.25
    x_tilde = x_tilde.reshape((l, -1))

    if shift_for_ar:
        # This shift is introduced as the rest is handled by the AR part.
        return shift(shift(x_tilde))
    else:
        return x_tilde
