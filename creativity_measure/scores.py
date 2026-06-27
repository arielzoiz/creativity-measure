# ``marginal_score`` is the score of the (Gaussian-blurred) marginal ``p_{Y_gamma}``.

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import ScoreFn
from creativity_measure.density import Density


def marginal_score(
    y: Float[Tensor, "B d"],
    gamma: Float[Tensor, ""],
    density: Density | None,
    score_fn: ScoreFn | None = None,
) -> Float[Tensor, "B d"]:
    """∇_y log p_Yg(y) — the score of the (blurred) marginal density at y.

    If score_fn is given it supplies the score directly (a pre-learned model);
    otherwise the score is obtained by autograd through density.log_p_Y.
    """
    if score_fn is not None:
        return score_fn(y, gamma)
    if density is None or density.log_p_Y is None:
        raise RuntimeError("Provide either score_fn or a Density exposing log_p_Y")
    y = y.detach().clone().requires_grad_(True)   # isolate y so we can take d/dy at this point
    # .sum() lets one grad call return per-row gradients (rows are independent)
    g = torch.autograd.grad(density.log_p_Y(y, gamma).sum(), y)[0]
    return g.detach()
