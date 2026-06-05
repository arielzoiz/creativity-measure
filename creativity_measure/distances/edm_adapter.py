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

from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import ScoreFn

# (y_sigma: (B, ...), sigma: (B,)) -> x_pred = E[X | y_sigma], same shape as y_sigma.
Denoiser = Callable[[Tensor, Tensor], Tensor]


def edm_score_fn(denoiser: Denoiser, img_shape: tuple[int, ...] | None = None) -> ScoreFn:
    """Wrap an EDM-style denoiser D(y_sigma, sigma) = E[X | y_sigma] into a marginal-score ScoreFn.

    The returned score_fn(y, gamma) = grad_y log p_Y(y, gamma) in this repo's gamma-convention,
    suitable as the score_fn= argument of GlobalIEMDistance / GeneralizedGlobalIEMDistance.

    Args:
        denoiser:  callable (y_sigma, sigma) -> x_pred; works on flat (B, d) inputs, or on
                   (B, *img_shape) when img_shape is given.
        img_shape: per-sample shape (C, H, W) the denoiser expects; None keeps the flat (B, d) layout.

    The caller must restrict the gamma-grid to the denoiser's valid sigma-range, i.e.
    gamma in [1/sigma_max^2, 1/sigma_min^2].
    """
    def score_fn(y: Float[Tensor, "B d"], gamma: Float[Tensor, ""]) -> Float[Tensor, "B d"]:
        sigma = gamma.rsqrt()                                  # 1 / sqrt(gamma)
        y_sigma = y / gamma                                    # signal-space obs  x + sigma*eps
        x_in = y_sigma if img_shape is None else y_sigma.reshape(y.shape[0], *img_shape)
        sigma_b = sigma.reshape(1).expand(y.shape[0])          # (B,)
        x_pred = denoiser(x_in, sigma_b).reshape(y.shape)      # E[X | y_sigma]
        return x_pred - y_sigma                                # E[X|y] - y/gamma = marginal score
    return score_fn
