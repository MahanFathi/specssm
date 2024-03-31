import jax
from flax import linen as nn
import gin


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
    activation: str = "gelu"
    prenorm: bool = False
    # batchnorm: bool = False
    # bn_momentum: float = 0.90

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(input_len=self.input_len)

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        # TODO(mahanfathi): support batchnorm in training_state
        # if self.batchnorm:
        #     self.norm = nn.BatchNorm(use_running_average=not self.training,
        #                              momentum=self.bn_momentum, axis_name='batch')
        # else:
        #     self.norm = nn.LayerNorm()
        
        self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
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
