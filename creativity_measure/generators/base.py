"""Universal EDM probability-flow ODE generator machinery: latent ``z ~ N(0,I)`` -> ``x ~ p``.

This is the model-agnostic core every factory in ``creativity_measure.generators`` funnels into.
``heun_prob_flow`` integrates the deterministic EDM/Karras prob-flow ODE; ``edm_generator`` wraps a
denoiser ``D(x_sigma, sigma) = E[X | x_sigma]`` into the flat ``(B, d)`` generator interface the SMC
consumes (see ``creativity_measure/adaptive_tempering_smc.py`` ``PCNKernel``). ``eps_to_edm_denoiser`` adapts a VP
epsilon-predictor into that EDM denoiser convention.

``G`` MUST be deterministic (fixed solver, fixed ``n_steps``, no injected noise); otherwise the pCN prior
cancellation -- hence the SMC acceptance formula -- is invalid. It is used forward-only (gradient-free).
"""

from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure.distances.edm_adapter import Denoiser


def karras_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    rho: float,
    n_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "steps_plus_1"]:
    """The decreasing Karras sigma-schedule with a final ``sigma = 0`` appended (length ``n_steps + 1``).

    ``sigma_i = (sigma_max^(1/rho) + i/(n_steps-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho`` for
    ``i = 0..n_steps-1``, then a trailing 0 so the last ODE step lands on the data manifold. Shared by the
    prob-flow generator (``heun_prob_flow``) and the trajectory-space sampler (``diffusion_smc``).
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    i = torch.arange(n_steps, device=device, dtype=dtype)
    inv = 1.0 / rho
    sigmas = (sigma_max ** inv + i / (n_steps - 1) * (sigma_min ** inv - sigma_max ** inv)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def edm_ode_step(
    x: Float[Tensor, "B d"],
    denoiser: Denoiser,
    s0: Float[Tensor, ""],
    s1: Float[Tensor, ""],
) -> Float[Tensor, "B d"]:
    """One deterministic Heun (2nd-order) step of ``dx/dsigma = (x - D(x, sigma)) / sigma`` from ``s0`` to ``s1``.

    Euler predictor plus a Heun corrector, skipping the corrector when ``s1 == 0`` (the final on-manifold
    step). ``s0`` / ``s1`` are scalar sigma tensors; the denoiser is called with sigma broadcast to the batch.
    """
    b = x.shape[0]

    def d_eval(xx: Tensor, s: Tensor) -> Tensor:
        return (xx - denoiser(xx, s.expand(b))) / s

    d0 = d_eval(x, s0)
    x_next = x + (s1 - s0) * d0
    if float(s1) != 0.0:                           # Heun correction (skip at the final sigma = 0)
        d1 = d_eval(x_next, s1)
        x_next = x + (s1 - s0) * 0.5 * (d0 + d1)
    return x_next


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
    sigmas = karras_sigma_schedule(sigma_min, sigma_max, rho, n_steps, device=z.device, dtype=z.dtype)
    x = z * sigma_max
    for k in range(n_steps):
        x = edm_ode_step(x, denoiser, sigmas[k], sigmas[k + 1])
    return x


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
    denoiser, exactly as ``edm_adapter.edm_score_fn`` does. This is the builder every model factory in this
    package returns; no change to ``adaptive_tempering_smc.py`` / ``PCNKernel`` is needed to switch models.
    """
    def wrapped(x: Float[Tensor, "B d"], sigma: Float[Tensor, "B"]) -> Float[Tensor, "B d"]:
        if img_shape is None:
            return denoiser(x, sigma)
        out = denoiser(x.reshape(x.shape[0], *img_shape), sigma)
        return out.reshape(x.shape)

    def g(z: Float[Tensor, "B d"]) -> Float[Tensor, "B d"]:
        return heun_prob_flow(z, wrapped, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, n_steps=n_steps)
    return g


# --- VP epsilon-predictor -> EDM denoiser adapter (Stable Diffusion path) ---------------------

# A VP epsilon model: (x_in, timestep) -> predicted noise eps, same shape as x_in.  ``timestep`` is a
# (possibly fractional) index into the model's discrete noise schedule.
EpsFn = Callable[[Tensor, Tensor], Tensor]


def _sigma_to_t(sigma: Tensor, model_log_sigmas: Tensor) -> Tensor:
    """k-diffusion ``sigma -> fractional timestep`` against an ascending log-sigma schedule.

    ``model_log_sigmas`` is ``log(sigma_i)`` for the model's discrete noise levels, ascending in ``i``.
    Returns a float index (linearly interpolated in log-sigma) suitable to feed a VP epsilon model.
    """
    log_sigma = sigma.log().reshape(-1)
    dists = log_sigma - model_log_sigmas[:, None]
    n = model_log_sigmas.shape[0]
    low_idx = (dists >= 0).to(torch.long).cumsum(dim=0).argmax(dim=0).clamp(max=n - 2)
    high_idx = low_idx + 1
    low = model_log_sigmas[low_idx]
    high = model_log_sigmas[high_idx]
    w = ((low - log_sigma) / (low - high)).clamp(0.0, 1.0)
    return (1.0 - w) * low_idx.to(sigma.dtype) + w * high_idx.to(sigma.dtype)


def eps_to_edm_denoiser(eps_fn: EpsFn, model_sigmas: Tensor) -> Denoiser:
    """Wrap a VP epsilon-predictor into an EDM denoiser ``D(x_sigma, sigma) = E[X | x_sigma]``.

    Standard k-diffusion ``DiscreteEpsDDPMDenoiser`` conversion: scale the input by
    ``c_in = 1/sqrt(sigma^2 + 1)``, map ``sigma`` to the model's (fractional) discrete timestep via the
    ascending schedule ``model_sigmas`` (e.g. ``sqrt((1 - alphas_cumprod)/alphas_cumprod)``), predict the
    noise, and recover the signal estimate as ``D = x_sigma - sigma * eps``.

    Args:
        eps_fn:       callable ``(x_in, timestep) -> eps`` (the SD UNet with text conditioning baked in).
        model_sigmas: 1D tensor of the scheduler's discrete sigmas, ascending in index.
    """
    model_log_sigmas = model_sigmas.log()

    def denoiser(x_sigma: Float[Tensor, "B ..."], sigma: Float[Tensor, "B"]) -> Float[Tensor, "B ..."]:
        s = sigma.reshape(-1)[0]                                       # ODE uses one sigma across the batch
        broadcast = (-1,) + (1,) * (x_sigma.dim() - 1)
        c_in = 1.0 / (s * s + 1.0).sqrt()
        t = _sigma_to_t(s.reshape(1), model_log_sigmas.to(s.device, s.dtype))
        eps = eps_fn((c_in * x_sigma), t.expand(x_sigma.shape[0]))
        return x_sigma - s.reshape(broadcast) * eps

    return denoiser
