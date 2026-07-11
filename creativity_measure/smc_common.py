"""Particle-weight helpers shared by the SMC samplers.

Kernel- and schedule-agnostic primitives lifted out of ``smc.py`` so both the data-space
adaptive-tempering sampler (``smc.py``) and the trajectory-space twisted-diffusion sampler
(``diffusion_smc.py``) reuse one implementation of ESS and systematic resampling. Keeping them
here avoids a dependency edge from ``diffusion_smc`` onto ``smc`` (and its tempering machinery).

Both are driven from a caller-owned ``torch.Generator`` where randomness is involved, so runs stay
reentrant / reproducible without touching global torch RNG state.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor


def _ess_from_logw(logw: Float[Tensor, "N"]) -> float:
    """Effective sample size from unnormalized log-weights, in log-space.

    ESS = (sum w)^2 / sum w^2 = exp(2*logsumexp(logw) - logsumexp(2*logw)).
    Uniform log-weights -> N; a one-hot weight -> 1.
    """
    a = torch.logsumexp(logw, dim=0)
    b = torch.logsumexp(2.0 * logw, dim=0)
    return float(torch.exp(2.0 * a - b))


def _systematic_resample(
    weights: Float[Tensor, "N"],
    generator: torch.Generator,
) -> Int[Tensor, "N"]:
    """Systematic resampling: return N parent indices with E[count_i] = N * w_i (deterministic given gen)."""
    n = weights.shape[0]
    w = weights / weights.sum()
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0
    u = torch.rand((), generator=generator, device=weights.device, dtype=weights.dtype)
    positions = (torch.arange(n, device=weights.device, dtype=weights.dtype) + u) / n
    idx = torch.searchsorted(cdf, positions)
    return idx.clamp_max_(n - 1)
