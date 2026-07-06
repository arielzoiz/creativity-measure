"""2D toy generator: an autograd EDM denoiser derived directly from an analytical ``Density`` (no model).

Mirrors how the IEM distances obtain a score when no learned ``score_fn`` is given: the denoiser
``D(x_sigma, sigma) = E[X | x_sigma]`` comes from autograd through ``density.log_p_Y`` via
``scores.marginal_score``. Used by the Phase-2 pCN calibration on 2D toys (``phase2_pcn_calibration.ipynb``).
"""

from collections.abc import Callable

from jaxtyping import Float
from torch import Tensor

from creativity_measure.density import Density
from creativity_measure.distances.edm_adapter import Denoiser, sigma_to_gamma
from creativity_measure.scores import marginal_score

from .base import heun_prob_flow


def density_denoiser(density: Density) -> Denoiser:
    """Autograd EDM denoiser ``D(x_sigma, sigma) = E[X | x_sigma]`` for a 2D ``Density`` (no learned model).

    Uses the validated adapter relation ``D(x_sigma, sigma) = x_sigma + grad_y log p_Y(gamma*x_sigma, gamma)`` with ``gamma = 1/sigma^2``;
    the marginal score comes from ``scores.marginal_score`` (autograd through ``density.log_p_Y``, ``score_fn=None``). For ``X~N(0,I)``
    this reduces to ``D = x_sigma / (1 + sigma^2)``.
    """
    def denoiser(x_sigma: Float[Tensor, "B d"], sigma: Float[Tensor, "B"]) -> Float[Tensor, "B d"]:
        s = sigma.reshape(-1)[0]                   # scalar: the ODE uses one sigma across the batch
        gamma = sigma_to_gamma(s)
        y = gamma * x_sigma
        return x_sigma + marginal_score(y, gamma, density)
    return denoiser


def density_generator(
    density: Density,
    *,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    n_steps: int = 64,
) -> Callable[[Float[Tensor, "B d"]], Float[Tensor, "B d"]]:
    """Build the 2D autograd generator ``G(z) = heun_prob_flow(z, density_denoiser(density), ...)``.

    ``sigma_max`` must exceed the data scale so ``p_{sigma_max} ~= N(0, sigma_max^2 I)`` (validated by the
    generator test; tune per density).
    """
    denoiser = density_denoiser(density)

    def g(z: Float[Tensor, "B d"]) -> Float[Tensor, "B d"]:
        return heun_prob_flow(z, denoiser, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, n_steps=n_steps)
    return g
