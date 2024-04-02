"""S5 SSM module."""

import jax
from jax import numpy as jnp
from flax import linen as nn

import gin

from .utils import make_DPLR_HiPPO, init_CV, init_VinvB, init_log_steps, trunc_standard_normal


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, d_model)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P, d_model)
    """
    Identity = jnp.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, d_model)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P, d_model)
    """
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """ Compute the Lxd_model output of discretized SSM given an Lxd_model input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, d_model)
            C_tilde    (complex64): output matrix                        (d_model, P)
            input_sequence (float32): input sequence of features         (L, d_model)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, d_model)
    """
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (Lambda_elements, Bu_elements),
                                          reverse=True)
        xs = jnp.concatenate((xs, xs2), axis=-1)

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


@gin.configurable
class S5(nn.Module):

    input_len: int = None
    d_model: int = gin.REQUIRED
    ssm_size_base: int = gin.REQUIRED
    num_blocks: int = gin.REQUIRED

    dt_min: float = 0.001
    dt_max: float = 0.1
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0
    discretization: str = "zoh"
    C_init: str = "trunc_standard_normal"

    """ The S5 SSM
        Args:
            input_len       (int32):    unused!
            d_model         (int32):    Dimensionality of input/output
            ssm_size_base   (int32):    State base size
            num_blocks      (int32):    Number of blocks to split the state matrix into
            dt_min:         (float32):  Minimum value to draw timescale values from when 
                                        initializing log_step
            dt_max:         (float32):  Maximum value to draw timescale values from when 
                                        initializing log_step
            conj_sym        (bool):     Whether conjugate symmetry is enforced
            clip_eigs       (bool):     Whether to enforce left-half plane condition, i.e.
                                        constrain real part of eigenvalues to be negative. 
                                        True recommended for autoregressive task/unbounded sequence lengths
                                        Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional   (bool):     Whether model is bidirectional, if True, uses two C matrices
            discretization: (string):   Specifies discretization method, options: [zoh, bilinear]
            step_rescale:   (float32):  Allows for uniformly changing the timescale parameter, e.g. after training 
                                        on a different resolution for the speech commands benchmark
            C_init          (string):   Specifies how C is initialized, 
                                        options: [trunc_standard_normal, lecun_normal, complex_normal]
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        # input_len is unused
        del self.input_len

        ssm_size = self.ssm_size_base
        block_size = int(ssm_size / self.num_blocks)

        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

        if self.conj_sym:
            block_size = block_size // 2
            ssm_size = ssm_size // 2
        
        self.P = ssm_size

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * jnp.ones((self.num_blocks, block_size))).ravel()
        V = jax.scipy.linalg.block_diag(*([V] * self.num_blocks))
        Vinv = jax.scipy.linalg.block_diag(*([Vc] * self.num_blocks))


        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: Lambda.real, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: Lambda.imag, (None,))
        if self.clip_eigs:
            self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = jax.nn.initializers.lecun_normal()
        B_shape = (local_P, self.d_model)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(
                                B_init, rng, shape, Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.d_model, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = jax.nn.initializers.lecun_normal()
            C_shape = (self.d_model, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = jax.nn.initializers.normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.d_model, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.d_model, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param("C1",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)
                self.C2 = self.param("C2",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = jnp.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param("C",
                                    lambda rng, shape: init_CV(C_init, rng, shape, V),
                                    C_shape)

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", jax.nn.initializers.normal(stddev=1.0), (self.d_model,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * jnp.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, input_sequence):
        """
        Compute the Lxd_model output of the S5 SSM given an Lxd_model input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        ys = apply_ssm(self.Lambda_bar,
                       self.B_bar,
                       self.C_tilde,
                       input_sequence,
                       self.conj_sym,
                       self.bidirectional)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du
