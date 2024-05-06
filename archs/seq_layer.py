import jax
from jax import numpy as jnp
from flax import linen as nn
import gin

from archs.position import rotate_x


@gin.configurable
class SequenceLayer(nn.Module):
    """ Defines a single SSM layer, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            input_len   (int32):    the length of the input sequence
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            training    (bool):     whether in training mode or not
            ssm         (nn.Module): the SSM to be used (i.e. S5 or STU ssm)
            dropout     (float32):  dropout rate
            activation  (string):   Type of activation function to use
            prenorm     (bool):     apply prenorm if true or postnorm if false
            # batchnorm   (bool):     apply batchnorm if true or layernorm if false
            # bn_momentum (float32):  the batchnorm momentum if batchnorm is used
    """
    input_len: int
    d_model: int = gin.REQUIRED
    training: bool = True
    ssm: nn.Module = gin.REQUIRED
    dropout: float = 0.1
    activation: str = "half_glu1"
    prenorm: bool = False
    positional_embedding: str | None = None
    batchnorm: bool = False
    bn_momentum: float = 0.90

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(input_len=self.input_len)

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name="minibatch")
        else:
            self.norm = nn.LayerNorm()
        
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

        if self.positional_embedding in ["independent"]:
            self._pos_emb = nn.Embed(
                num_embeddings=self.input_len,
                features=self.d_model,
            )
            self.pos_emb = lambda _: self._pos_emb(jnp.arange(self.input_len))

        elif self.positional_embedding in ["rotary"]:
            self.pos_emb = lambda x: rotate_x(x, max_wavelength=10_000)
            

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.positional_embedding:
            if self.positional_embedding in ["independent"]:
                x += self.pos_emb(x)
            elif self.positional_embedding in ["rotary"]:
                x = self.pos_emb(x)
            else:
                raise NotImplementedError(
                    "Positional embedding: {} not implemented".format(self.positional_embedding))
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)

        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)
        return x
