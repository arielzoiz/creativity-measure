"""Deterministic EDM probability-flow ODE generator  G(z): N(0,I) latent -> x ~ p.

Used by the Phase-2 latent-space pCN kernel (``creativity_measure/smc.py``): a particle is carried as a
latent ``z`` and mapped to data via ``x = G(z)``. Because ``z ~ N(0,I) => x = G(z) ~ p``, the pCN proposal
``z' = sqrt(1-s^2) z + s*xi`` is reversible w.r.t. the Gaussian prior and the base measure cancels, so the
acceptance ratio needs only ``f``.

``G`` is built from a denoiser ``D(x_sigma, sigma) = E[X | X + sigma*eps = x_sigma]`` (EDM/Karras
convention), exactly mirroring how the IEM distances obtain a score:

  * **2D / toy** — ``density_denoiser(density)`` derives ``D`` by autograd through ``density.log_p_Y``
    (no learned model), the same fallback the IEM distances use when no ``score_fn`` is given.
  * **pixel** — ``edm_generator(denoiser, ...)`` takes a pre-learned EDM denoiser directly.

Everything the SMC sees stays FLAT ``(B, d)`` with ``d = prod(data shape)``; image ``(C, H, W)`` handling
is an internal detail of the denoiser wrapper (``img_shape``), mirroring ``edm_adapter.edm_score_fn``.

``G`` MUST be deterministic (fixed solver, fixed ``n_steps``, no injected noise); otherwise the pCN prior
cancellation — hence the acceptance formula — is invalid. It is used forward-only (gradient-free): no
backprop through the ODE.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure.density import Density
from creativity_measure.distances.edm_adapter import Denoiser
from creativity_measure.distances.global_iem import marginal_score


def heun_prob_flow(
    z: Float[Tensor, "B d"],
    denoiser: Denoiser,
    *,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    n_steps: int = 64,
) -> Float[Tensor, "B d"]:
    """Deterministic EDM/Karras probability-flow ODE: latent ``z ~ N(0,I)`` -> ``x ~ p``.

    Heun (2nd-order) integration of ``dx/dsigma = (x - D(x, sigma)) / sigma`` down a Karras sigma-schedule
    from ``sigma_max`` to 0, starting at ``x = sigma_max * z``. No stochasticity (deterministic map).
    Elementwise in ``x``, so a flat ``(B, d)`` layout serves both 2D (d=2) and pixels (d=C*H*W).
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    b = z.shape[0]
    i = torch.arange(n_steps, device=z.device, dtype=z.dtype)
    inv = 1.0 / rho
    # Karras schedule (decreasing); append a final sigma = 0 so the last step lands on the data manifold.
    sigmas = (sigma_max ** inv + i / (n_steps - 1) * (sigma_min ** inv - sigma_max ** inv)) ** rho
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

    def d_eval(x: Tensor, s: Tensor) -> Tensor:
        return (x - denoiser(x, s.expand(b))) / s

    x = z * sigma_max
    for k in range(n_steps):
        s0, s1 = sigmas[k], sigmas[k + 1]
        d0 = d_eval(x, s0)
        x_next = x + (s1 - s0) * d0
        if float(s1) != 0.0:                       # Heun correction (skip at the final sigma = 0)
            d1 = d_eval(x_next, s1)
            x_next = x + (s1 - s0) * 0.5 * (d0 + d1)
        x = x_next
    return x


def density_denoiser(density: Density) -> Denoiser:
    """Autograd EDM denoiser ``D(x_sigma, sigma) = E[X | x_sigma]`` for a 2D ``Density`` (no learned model).

    Uses the validated adapter relation ``D(x_sigma, sigma) = x_sigma + grad_y log p_Y(gamma*x_sigma, gamma)``
    with ``gamma = 1/sigma^2`` (cf. ``edm_adapter.edm_score_fn``); the marginal score comes from
    ``global_iem.marginal_score`` (autograd through ``density.log_p_Y``, ``score_fn=None``). For ``X~N(0,I)``
    this reduces to ``D = x_sigma / (1 + sigma^2)``.
    """
    def denoiser(x_sigma: Float[Tensor, "B d"], sigma: Float[Tensor, "B"]) -> Float[Tensor, "B d"]:
        s = sigma.reshape(-1)[0]                   # scalar: the ODE uses one sigma across the batch
        gamma = 1.0 / (s * s)
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


def edm_generator(
    denoiser: Denoiser,
    *,
    img_shape: tuple[int, ...] | None = None,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    n_steps: int = 64,
) -> Callable[[Float[Tensor, "B d"]], Float[Tensor, "B d"]]:
    """Build the pixel generator from a pre-learned EDM denoiser ``D(x_sigma, sigma) = E[X | x_sigma]``.

    The SMC keeps latents/particles flat ``(B, d=C*H*W)``; ``img_shape`` reshapes flat <-> image around the
    denoiser, exactly as ``edm_adapter.edm_score_fn`` does. This is the builder a future image run uses with
    its checkpoint; no change to ``smc.py`` / ``PCNKernel`` is needed to switch from 2D to pixels.
    """
    def wrapped(x: Float[Tensor, "B d"], sigma: Float[Tensor, "B"]) -> Float[Tensor, "B d"]:
        if img_shape is None:
            return denoiser(x, sigma)
        out = denoiser(x.reshape(x.shape[0], *img_shape), sigma)
        return out.reshape(x.shape)

    def g(z: Float[Tensor, "B d"]) -> Float[Tensor, "B d"]:
        return heun_prob_flow(z, wrapped, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, n_steps=n_steps)
    return g
