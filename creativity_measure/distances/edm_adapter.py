#
# Adapter: wrap an EDM-style denoiser into a gamma-convention ScoreFn.
#
# A denoiser D(y_sigma, sigma) returns the MMSE estimate E[X | y_sigma] for the signal-space
# observation y_sigma = x + sigma*eps (EDM / k-diffusion convention). This repo works in the
# paper's gamma-convention Y = gamma*x + sqrt(gamma)*W, related by
#     y_sigma = y / gamma,    sigma = 1 / sqrt(gamma).
# Conditioning on Y = gamma*y_sigma is the same as on y_sigma, so E[X|Y=y] = D(y/gamma, 1/sqrt(gamma)),
# and by Tweedie the marginal score in the gamma-convention is
#     grad_y log p_Y(y, gamma) = E[X|y] - y/gamma = D(y/gamma, 1/sqrt(gamma)) - y/gamma.
# Only this adapter ever sees sigma; the metric core uses gamma.

from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import ScoreFn

# (y_sigma: (B, ...), sigma: (B,)) -> x_pred = E[X | y_sigma], same shape as y_sigma.
Denoiser = Callable[[Tensor, Tensor], Tensor]


def gamma_to_sigma(gamma: Tensor) -> Tensor:
    """EDM noise scale from the gamma-convention precision:  sigma = 1 / sqrt(gamma)."""
    return gamma.rsqrt()


def sigma_to_gamma(sigma: Tensor) -> Tensor:
    """gamma-convention precision from the EDM noise scale:  gamma = 1 / sigma^2."""
    return 1.0 / (sigma * sigma)


def chunked_denoiser(denoiser: Denoiser, max_rows: int) -> Denoiser:
    """Bound a denoiser's batch width to ``max_rows``, splitting wider calls into sequential blocks.

    Since we use reference-score caching, R (reference set size) is a batch dimension rather than a
    loop counter, so raising R raises the width of a single forward pass. Thefore, use of high R is
    expected to cause OOMs on GPUs with limited VRAM.

    Rows of a denoiser batch are independent (no cross-batch attention), so splitting is
    mathematically identical -- this is purely a VRAM ceiling, like ``global_iem``'s ``r_chunk``.

    Wrap ONCE and pass the result to BOTH ``edm_generator`` and ``edm_score_fn``: that covers the
    generator path and every score path, including the reference ``score_bank`` build, which is the
    widest call in a run.

    ``max_rows`` must be FIXED for a run. ``Reward`` requires f to be a deterministic function of x.
    """
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")

    def chunked(x: Tensor, sigma: Tensor) -> Tensor:
        b = x.shape[0]
        if b <= max_rows:
            return denoiser(x, sigma)
        # sigma arrives as (B,) from both callers (edm_score_fn expands it; edm_ode_step passes
        # s.expand(b)), but tolerate a scalar so any Denoiser honouring the protocol can be wrapped.
        batched = isinstance(sigma, Tensor) and sigma.ndim >= 1 and sigma.shape[0] == b
        return torch.cat(
            [denoiser(x[i:i + max_rows], sigma[i:i + max_rows] if batched else sigma)
             for i in range(0, b, max_rows)],
            dim=0,
        )

    return chunked


def edm_score_fn(denoiser: Denoiser, img_shape: tuple[int, ...] | None = None) -> ScoreFn:
    """Wrap an EDM-style denoiser D(y_sigma, sigma) = E[X | y_sigma] into a marginal-score ScoreFn.

    The returned score_fn(y, gamma) = grad_y log p_Y(y, gamma) in this repo's gamma-convention (gamma scalar, or (B,)
    for one level per row), suitable as the score_fn= argument of GlobalIEMDistance / GeneralizedGlobalIEMDistance.

    Args:
        denoiser:  callable (y_sigma, sigma) -> x_pred; works on flat (B, d) inputs, or on
                   (B, *img_shape) when img_shape is given.
        img_shape: per-sample shape (C, H, W) the denoiser expects; None keeps the flat (B, d) layout.

    The caller must restrict the gamma-grid to the denoiser's valid sigma-range, i.e.
    gamma in [1/sigma_max^2, 1/sigma_min^2].
    """
    def score_fn(y: Float[Tensor, "B d"], gamma: Float[Tensor, ""] | Float[Tensor, "B"]) -> Float[Tensor, "B d"]:
        sigma = gamma_to_sigma(gamma)                          # 1 / sqrt(gamma)
        # gamma is a scalar (one level for the whole batch) or (B,) (one level per row, used by the fused i.i.d.
        # bank). A scalar reshapes to (1, 1, ...) so the division below is elementwise-identical to y / gamma.
        y_sigma = y / gamma.reshape(-1, *([1] * (y.ndim - 1)))   # signal-space obs  x + sigma*eps
        x_in = y_sigma if img_shape is None else y_sigma.reshape(y.shape[0], *img_shape)
        sigma_b = sigma.reshape(-1).expand(y.shape[0])         # (B,)
        x_pred = denoiser(x_in, sigma_b).reshape(y.shape)      # E[X | y_sigma]
        return x_pred - y_sigma                                # E[X|y] - y/gamma = marginal score
    return score_fn
