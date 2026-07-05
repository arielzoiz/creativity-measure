from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure.density import Density
from creativity_measure.distances.base import Distance


def expected_distance(
    distance: Distance,
    X: Float[Tensor, "B d"],
    x_refs: Float[Tensor, "R d"],
    weights: Float[Tensor, "R"] | None = None,
) -> Float[Tensor, "B"]:
    """E_{x'~p}[ D(x', X) ] approximated by the (weighted) mean over the supplied refs.

    f(X) = sum_r w_r D(X, x'_r) / sum_r w_r.

    weights: per-reference weights (e.g. RefSelector.weights). None => uniform, i.e. the plain mean over refs.
        Weighted selectors (WeightedFPSRefs) must pass their weights here, otherwise the the non-uniform
        weighting is silently dropped.
    """
    pw = distance.pairwise(X, x_refs)                     # (B, R)
    if weights is None:
        return pw.mean(dim=1)
    w = weights.to(pw)                                    # match dtype/device
    return (pw * w).sum(dim=1) / w.sum()


@dataclass(frozen=True)
class Reward:
    """The frozen tilt reward  f(X) = sum_r w_r D(X, x'_r) / sum_r w_r, bundled as one object.

    Holds the three pieces that define f: the `distance`, the frozen reference set `x_refs`, and
    the per-reference `weights` (None => uniform).
    
    Frozen so f is fixed once selected - the determinism the SMC/MCMC theory assumes (see `smc.py`).
    `x_refs` also pins the run's device/dtype.
    """

    distance: Distance
    x_refs: Float[Tensor, "R d"]
    weights: Float[Tensor, "R"] | None = None

    def __call__(self, X: Float[Tensor, "B d"]) -> Float[Tensor, "B"]:
        return expected_distance(self.distance, X, self.x_refs, weights=self.weights)


def reference_pair_mean(
    distance: Distance,
    x_refs: Float[Tensor, "R d"],
    weights: Float[Tensor, "R"] | None = None,
) -> Float[Tensor, ""]:
    """E_{x',x''~p}[ D(x', x'') ] estimated over the OFF-DIAGONAL pairs of the frozen refs.

    Reuses the reference set as the pair sample (no extra draws). With a squared distance injected
    (SquaredGlobalIEMDistance) this is E[D_IEM^2]. Off-diagonal only (the r==r self-pairs, D=0, are dropped
    so they don't bias the mean down). weights: per-reference
    weights (None => uniform, plain off-diagonal mean); applied on BOTH indices for weighted selectors.
    """
    R = x_refs.shape[0]
    if R < 2:
        raise ValueError(f"reference_pair_mean needs >= 2 references, got {R}")
    M = distance.pairwise(x_refs, x_refs)                     # (R, R); D^2 when distance is squared
    eye = torch.eye(R, dtype=torch.bool, device=M.device)
    if weights is None:
        return M[~eye].mean()
    w = weights.to(M)                                         # match dtype/device
    W2 = (w[:, None] * w[None, :]).masked_fill(eye, 0)        # zero the diagonal pair-weights
    return (M * W2).sum() / W2.sum()


@dataclass(frozen=True)
class NormalizedExpectedDistanceReward(Reward):
    """The normalized tilt reward  f(X) = E_{x'~p}[D(X, x')] / E_{x',x''~p}[D(x', x'')].

    Same (distance, x_refs, weights) bundle as `Reward`; only the reduction differs: the expected
    distance is divided by the scalar constant `reference_pair_mean(distance, x_refs, weights)`.
    With `SquaredGlobalIEMDistance` injected this is the squared-IEM reward. The denominator
    reuses the frozen refs and is computed ONCE (cached in __post_init__; base is frozen so we set
    it via object.__setattr__), keeping f deterministic as the SMC/MCMC theory assumes.
    """

    _denom: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        denom = float(reference_pair_mean(self.distance, self.x_refs, self.weights))
        object.__setattr__(self, "_denom", denom)

    def __call__(self, X: Float[Tensor, "B d"]) -> Float[Tensor, "B"]:
        num = expected_distance(self.distance, X, self.x_refs, weights=self.weights)
        return num / self._denom


def tilted_log_density(
    X: Float[Tensor, "B d"],
    density: Density,
    reward: Reward,
    lam: float,
) -> Float[Tensor, "B"]:
    """
    TOY / LOW-DIM (2D) ONLY. Needs a tractable base log-density `density.log_p_X` and is meant to be fed
    to `grid_normalize`, so it applies only to the toy/analytic setting.

    Unnormalized log of  q_lambda(x) ∝ p(x) * exp(lambda * E_{x'~p}[D(x', x)]).

    Returns log p(X) + lambda * reward(X), where `reward` carries the (distance, refs, weights) that define f.
    The normalizer Z_lambda is omitted; recover it with grid_normalize, or ignore it for sampling where it cancels.
    """
    return density.log_p_X(X) + lam * reward(X)


def grid_normalize(
    log_q_unnorm: Float[Tensor, "..."],
    cell_area: float,
) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, ""]]:
    """
    TOY / LOW-DIM (2D) ONLY. Normalize an unnormalized log-density evaluated on a regular grid.
    Normalizes by summing exp(log q) over an enumerated regular grid -
    only exists in the toy setting, used on the output of `tilted_log_density`.    

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
    flat = flat - flat[~flat.isnan()].max()
    q_un = flat.exp()
    Z = q_un.sum() * cell_area
    q = (q_un / Z).reshape(shape)
    return q.clamp_min(1e-300).log(), q, Z
