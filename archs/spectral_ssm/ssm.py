"""Spectral temporal unit (STU) block."""

import functools

import jax
import jax.numpy as jnp

from flax import linen as nn

import gin

import archs.spectral_ssm.utils as stu_utils

PRNGKey = jax.Array
Array = jax.Array

@functools.partial(jax.jit, static_argnums=(3, ))
def apply_stu(
    params: tuple[Array, Array, Array],
    inputs: Array,
    eigh: tuple[Array, Array],
    pure_spec: bool = False,
) -> Array:
    """Apply STU.

    Args:
        params: A tuple of parameters of shapes [d_out, d_out], [d_in, d_out, k_u],
        [d_in * k, d_out] and [d_in * k, d_out]
        inputs: Input matrix of shape [l, d_in].
        eigh: A tuple of eigenvalues [k] and circulant eigenvecs [k, l].
        pure_spec: If True, only spectral component is used.

    Returns:
        A sequence of y_ts of shape [l, d_out].
    """
    m_y, m_u, m_phi = params

    x_tilde = stu_utils.compute_x_tilde(inputs, eigh, not pure_spec)

    # Compute deltas from the spectral filters, which are of shape [l, d_out].
    delta_phi = x_tilde @ m_phi

    if pure_spec:
        return delta_phi

    # Compute deltas from AR on x part
    delta_ar_u = stu_utils.compute_ar_x_preds(m_u, inputs)

    # Compute y_ts, which are of shape [l, d_out].
    return stu_utils.compute_y_t(m_y, delta_phi + delta_ar_u)


@gin.configurable
class STU(nn.Module):
    """Simple STU Layer.

        d_out: Output dimension.
        input_len: Input sequence length.
        num_eigh: Tuple of eigen values and vecs sized (k,) and (l, k)
        auto_reg_k_u: Auto-regressive depth on the input sequence,
        auto_reg_k_y: Auto-regressive depth on the output sequence,
        learnable_m_y: m_y matrix learnable,
        pure_spec: If True, only spectral component is used.
    """

    input_len: int
    d_model: int = gin.REQUIRED
    num_eigh: int = gin.REQUIRED
    auto_reg_k_u: int = gin.REQUIRED
    auto_reg_k_y: int = gin.REQUIRED
    learnable_m_y: bool = gin.REQUIRED
    pure_spec: bool = False

    def setup(self):
        """Initialize STU layer."""

        self.eigh = stu_utils.get_top_hankel_eigh(self.input_len, self.num_eigh)
        self.m_phi = self.param('m_phi', nn.initializers.zeros_init(), (self.d_model * self.num_eigh, self.d_model))
        if self.pure_spec:
            self.m_y = None
            self.m_u = None
        else:
            if self.learnable_m_y:
                self.m_y = self.param('m_y', nn.initializers.zeros_init(), (self.d_model, self.auto_reg_k_y, self.d_model))
            else:
                self.m_y = jnp.zeros([self.d_model, self.auto_reg_k_y, self.d_model])

            m_x_var = 1.0 / (float(self.d_model) ** 0.5)
            self.m_u = m_x_var * self.param('m_u', nn.initializers.truncated_normal(), (self.d_model, self.d_model, self.auto_reg_k_u))

    def __call__(
        self,
        inputs: Array,
    ) -> Array:
        """Forward pass.

        Args:
            inputs: Assumed to be of shape (L, H) where L is sequence length, 
                    and H is number of features (channels) in the input.
                    We use nn.vmap later to expand to (B, L, H) where B is batch size.

        Returns:
            `Array` of preactivations.
        """
        d_in = inputs.shape[-1]

        params = (self.m_y, self.m_u, self.m_phi)

        return apply_stu(params, inputs, self.eigh, self.pure_spec)


# deprecated
@gin.configurable
class SpectralSSM(nn.Module):
    """General model architecture.
    
        d_model: Dimension of the embedding.
        d_target: Dimension of the target.
        num_layers: Number of layers.
        dropout: Dropout rate.
        input_len: Input sequence length.
        num_eigh: Number of eigen values and vecs.
        auto_reg_k_u: Auto-regressive depth on the input sequence.
        auto_reg_k_y: Auto-regressive depth on the output sequence.
        learnable_m_y: m_y matrix learnable.
    """

    input_len: int
    d_model: int = gin.REQUIRED
    d_target: int = gin.REQUIRED
    num_layers: int = gin.REQUIRED
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        is_training: bool = True,
    ) -> Array:
        """Forward pass of classification pipeline.

        Args:
            inputs: (batch, length, source dim).
            is_training: True for training mode.

        Returns:
            Outputs of general architecture
        """
        # Embedding layer.
        x = nn.Dense(self.d_model)(inputs)

        for _ in range(self.num_layers):
            # Saving input to layer for residual.
            z = x

            # Construct pre-layer batch norm.
            x = nn.BatchNorm(   # TODO(mahanfathi): batch norm is probably not a good idea here 
                use_bias=True,
                use_scale=True,
                momentum=0.9,
            )(x, use_running_average=not is_training)

            x = STU(d_model=self.d_model)(x)

            # GeLU + dropout.
            x = jax.nn.gelu(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not is_training)
            x = nn.Dense(2 * self.d_model)(x)
            x = jax.nn.glu(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not is_training)

            # Residual connection.
            x = x + z

        # Projection
        x = jnp.mean(x, axis=1, keepdims=True)
        x = nn.Dense(self.d_target)(x)

        return x
