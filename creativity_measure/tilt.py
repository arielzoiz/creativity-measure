from jaxtyping import Float
from torch import Tensor

from creativity_measure.density import Density
from creativity_measure.distances.base import Distance


def expected_distance(
    distance: Distance,
    X: Float[Tensor, "B d"],
    x_refs: Float[Tensor, "R d"],
) -> Float[Tensor, "B"]:
    """E_{x'~p}[ D(x', X) ] approximated by the mean over the supplied refs."""
    return distance.pairwise(X, x_refs).mean(dim=1)


def tilted_log_density(
    X: Float[Tensor, "B d"],
    density: Density,
    distance: Distance,
    x_refs: Float[Tensor, "R d"],
    lam: float,
) -> Float[Tensor, "B"]:
    """
    Unnormalized log of  q_lambda(x) ∝ p(x) * exp(lambda * E_{x'~p}[D(x', x)]).

    Returns log p(X) + lambda * E_{x'~p}[D(x', X)].
    (The normalizer Z_lambda is omitted; use grid_normalize for 2D, or
    ignore it for sampling where it cancels.)
    """
    score = expected_distance(distance, X, x_refs)        # (B,)
    return density.log_p_X(X) + lam * score


def grid_normalize(
    log_q_unnorm: Float[Tensor, "..."],
    cell_area: float,
) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, ""]]:
    """
    Normalize an unnormalized log-density evaluated on a regular grid.

    Args:
        log_q_unnorm: (G,) or (gn, gn) tensor of log q (unnormalized)
        cell_area:    dx*dy of one grid cell

    Returns:
        log_q:  normalized log-density (same shape)
        q:      normalized density      (same shape)
        Z:      scalar Z_lambda ≈ ∫ exp(log_q_unnorm) dx
    """
    shape = log_q_unnorm.shape
    flat = log_q_unnorm.reshape(-1)
    flat = flat - flat[~flat.isnan()].max()      # max-stabilize for exp()
    q_un = flat.exp()
    Z = q_un.sum() * cell_area
    q = (q_un / Z).reshape(shape)
    return q.clamp_min(1e-300).log(), q, Z
