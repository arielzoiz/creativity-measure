# log p(y | x, gamma) = log N(y; gamma*x, gamma*I).
import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributions import MultivariateNormal


def log_p_Y_given_X(
    y: Float[Tensor, "... d"],
    x: Float[Tensor, "... d"],
    gamma: Float[Tensor, ""],
) -> Float[Tensor, "..."]:
    """log p(y | x, gamma) under Y = gamma*x + sqrt(gamma)*W, W ~ N(0, I)."""
    d = x.shape[-1]
    cov = float(gamma) * torch.eye(d, device=x.device, dtype=x.dtype)
    return MultivariateNormal(float(gamma) * x, cov).log_prob(y)


def simulate_brownian(
    gammas: Float[Tensor, "N_gamma"],
    num_eps: int,
    d: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "N_gamma N_eps 1 d"]:
    """Simulate num_eps Wiener paths W on the gamma grid:
    W[0] ~ N(0, g0 I) and increments dW ~ N(0, dgamma I), so Var(W_g) = g at every grid point.
    (A standard Wiener process at the first grid point g0 > 0 is not zero.)"""
    num_gamma = gammas.shape[0]
    dgamma = gammas[1:] - gammas[:-1]
    generator = torch.Generator(device=device).manual_seed(seed)
    # Brownian increments: dW ~ N(0, dgamma * I), so scale standard normals by sqrt(dgamma)
    dW = torch.randn(num_gamma - 1, num_eps, 1, d,
                     device=device, dtype=dtype, generator=generator) * dgamma.sqrt().view(-1, 1, 1, 1)
    # Initial value W[0] ~ N(0, g0 I); offset the whole path by it so Var(W_g) = g exactly.
    W0 = torch.randn(1, num_eps, 1, d,
                     device=device, dtype=dtype, generator=generator) * gammas[0].sqrt()
    W = torch.zeros(num_gamma, num_eps, 1, d, device=device, dtype=dtype)
    W[0] = W0
    W[1:] = W0 + torch.cumsum(dW, dim=0)     # cumulative sum of increments, shifted by W0
    return W
