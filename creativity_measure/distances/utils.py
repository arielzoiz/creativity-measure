# Shared helpers for the IEM distances.
import math

import torch
from jaxtyping import Float
from torch import Tensor


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


def log_uniform_gammas(
    gamma_lo: float,
    gamma_hi: float,
    num_gamma: int,
    seed: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Float[Tensor, "G"], Float[Tensor, "G"]]:
    """Caller-side: G i.i.d. log-uniform draws on [gamma_lo, gamma_hi] and their importance weights.

    gamma_g = gamma_lo * (gamma_hi / gamma_lo)^u_g with u_g ~ U(0, 1), i.e. density 1/(gamma * L) with
    L = ln(gamma_hi / gamma_lo). The weight w_g = gamma_g * L / G is the inverse density over G, so
    sum_g w_g h(gamma_g) is an unbiased Monte-Carlo estimate of the integral of h over [gamma_lo, gamma_hi].

    Drawn ONCE per run (a frozen bank): the reward stays a deterministic function of x. Returned sorted by
    gamma, which only makes diagnostics readable -- the estimate is order-invariant.
    """
    L = math.log(gamma_hi / gamma_lo)
    generator = torch.Generator(device=device).manual_seed(seed)
    u = torch.rand(num_gamma, device=device, dtype=dtype, generator=generator)
    gammas = (gamma_lo * torch.exp(L * u)).sort().values
    return gammas, gammas * (L / num_gamma)


def simulate_iid_noise(
    gammas: Float[Tensor, "G"],
    num_eps: int,
    d: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "G N_eps 1 d"]:
    """The i.i.d. counterpart of `simulate_brownian`: W_g = sqrt(gamma_g) * eps, eps ~ N(0, I) i.i.d. over (g, eps).

    For f = identity the IEM integrand at each gamma depends only on the MARGINAL law W_gamma ~ N(0, gamma I),
    never on how W is correlated across gammas, so dropping the Brownian coupling leaves the expectation
    unchanged. Same (G, N_eps, 1, d) layout as `simulate_brownian`, so downstream code indexes W[g] alike.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(gammas.shape[0], num_eps, 1, d, device=device, dtype=dtype, generator=generator)
    return gammas.to(device=device, dtype=dtype).sqrt().view(-1, 1, 1, 1) * eps
