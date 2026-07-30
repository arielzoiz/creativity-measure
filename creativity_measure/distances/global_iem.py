#
# Global IEM pairwise distance (Ohayon et al., ICLR 2026, Def. 1, f = identity):
#   D_IEM^2(x1,x2) = ∫_0^∞ E_W[ || ∇log p_Yg(g x1 + W) - ∇log p_Yg(g x2 + W) ||^2 ] dg
#   then D_IEM = sqrt(D_IEM^2).
# Direct transcription: differentiates the marginal log-density log p_Yg w.r.t. y at each
# of the two noisy points (shared Brownian path W), with no conditional-score term.
#
# COST. The Brownian bank W is built once and shared by every point, so the marginal score at a point depends only on (that point, gamma, W) —
# never on what it is compared against. Scores are therefore evaluated ONCE per point per gamma interval and the pairwise differences are formed
# by broadcasting, costing
#       (N_gamma - 1) * N_eps * (R + B)   score rows per `pairwise` call,
# and only  (N_gamma - 1) * N_eps * B  once the reference bank is cached (`cache_refs=True`), i.e. independent of R.

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import ScoreFn
from creativity_measure.density import Density
from creativity_measure.distances.base import Distance
from creativity_measure.distances.utils import simulate_brownian
from creativity_measure.scores import marginal_score

# Chunks to no pass the memory budget. `r_chunk` is chosen automatically.
# Purely a memory ceiling: `r_chunk` has no effect on the returned values (see `iem_sq_integral`).
DIFF_BYTES_BUDGET: int = 128 * 2**20    # 128 MB


def iem_sq_increments_one_to_many(
    x_ref: Float[Tensor, "1 d"],
    X: Float[Tensor, "G d"],
    W: Float[Tensor, "N_gamma N_eps 1 d"],
    gammas: Float[Tensor, "N_gamma"],
    density: Density | None,
    score_fn: ScoreFn | None = None,
) -> Float[Tensor, "N_gamma_minus_1 N_eps G"]:
    """
    IEM^2 increments (Def. 1) for ONE reference vs the whole batch X, shared Brownian path W.
    x_ref: (1, d), X: (G, d), W: (N_gamma, N_eps, 1, d)
    score_fn: optional pre-learned marginal score; falls back to autograd through density.
    Returns:
        score_diff_sq_increments: (N_gamma-1, N_eps, G)  summed over gamma -> IEM^2

    NOT used in practice — it re-evaluates the batch scores for every reference it is called with.
    `GlobalIEMDistance.pairwise` uses `iem_sq_integral`, which evaluates each point once and chaches the results.
    Kept as the independent single-reference oracle that `tests/test_global_iem.py` checks the batched implementation against.
    """
    num_gamma, num_eps = W.shape[0], W.shape[1]
    G = X.shape[0]
    d = X.shape[1]
    dgamma = gammas[1:] - gammas[:-1]   # gamma step sizes (integration widths)

    # Noisy observation paths y_g = g*x + W, shared W across both points
    y1_path = gammas.view(-1, 1, 1, 1) * x_ref.view(1, 1, 1, d) + W     # (N_gamma, N_eps, 1, d)
    y2_path = gammas.view(-1, 1, 1, 1) * X.view(1, 1, G, d) + W         # (N_gamma, N_eps, G, d)

    increment_list = []
    for i in range(num_gamma - 1):  # one step per integration interval
        gamma = gammas[i]
        y1 = y1_path[i].reshape(num_eps, d)
        y2 = y2_path[i].reshape(num_eps * G, d)
        s1 = marginal_score(y1, gamma, density, score_fn).view(num_eps, 1, d)   # ∇log p_Yg(g x1 + W)
        s2 = marginal_score(y2, gamma, density, score_fn).view(num_eps, G, d)   # ∇log p_Yg(g x2 + W)
        score_diff = s1 - s2                                          # (N_eps, G, d)
        increment = score_diff.pow(2).sum(-1) * dgamma[i]             # ||score diff||^2 dg  (N_eps, G)
        increment_list.append(increment)

    return torch.stack(increment_list, dim=0)   # (N_gamma-1, N_eps, G)


def _scores_at_gamma(
    points: Float[Tensor, "P d"],
    gamma: Float[Tensor, ""],
    W_i: Float[Tensor, "N_eps 1 d"],
    density: Density | None,
    score_fn: ScoreFn | None,
) -> Float[Tensor, "N_eps P d"]:
    """∇log p_Yg at y = gamma*points + W_i, for every Brownian path in W_i.

    One score call on a batch of N_eps*P rows.
    """
    num_eps, P, d = W_i.shape[0], points.shape[0], points.shape[1]
    y = gamma * points.view(1, P, d) + W_i                    # (N_eps, P, d)
    return marginal_score(y.reshape(num_eps * P, d), gamma, density, score_fn).view(num_eps, P, d)


def score_bank(
    points: Float[Tensor, "P d"],
    W: Float[Tensor, "N_gamma N_eps 1 d"],
    gammas: Float[Tensor, "N_gamma"],
    density: Density | None,
    score_fn: ScoreFn | None = None,
) -> Float[Tensor, "N_gamma_minus_1 N_eps P d"]:
    """Marginal scores at `points` for every gamma interval and Brownian path.

    This is the cacheable object: it depends only on (points, gammas, W) — never on whatever the
    points are later compared against — so for a frozen reference set it can be built once and
    reused for the whole run. Costs (N_gamma-1) * N_eps * P score rows.
    """
    num_gamma = W.shape[0]
    return torch.stack(
        [_scores_at_gamma(points, gammas[i], W[i], density, score_fn) for i in range(num_gamma - 1)],
        dim=0,
    )


def auto_r_chunk(
    num_eps: int, B: int, d: int, itemsize: int, budget: int = DIFF_BYTES_BUDGET
) -> int:
    """Largest reference block whose (N_eps, B, r_chunk, d) difference tensor fits in `budget`."""
    bytes_per_ref = num_eps * B * d * itemsize
    return max(1, budget // max(bytes_per_ref, 1))


def iem_sq_integral(
    x_refs: Float[Tensor, "R d"],
    X: Float[Tensor, "B d"],
    W: Float[Tensor, "N_gamma N_eps 1 d"],
    gammas: Float[Tensor, "N_gamma"],
    density: Density | None,
    score_fn: ScoreFn | None = None,
    *,
    ref_scores: Float[Tensor, "N_gamma_minus_1 N_eps R d"] | None = None,
    batch_scores: Float[Tensor, "N_gamma_minus_1 N_eps B d"] | None = None,
    r_chunk: int | None = None,
    verbose: bool = False,
) -> Float[Tensor, "N_eps B R"]:
    """IEM^2 per Brownian path for the full batch x reference grid, integrated over gamma.

    ∫ || ∇log p_Yg(g x_r + W) - ∇log p_Yg(g x_b + W) ||^2 dg  for every (path, b, r),
    by the left-endpoint rule over the N_gamma-1 intervals of `gammas`. Average over the N_eps axis to get E_W[IEM^2].

    Every point's score is evaluated exactly once per gamma interval; the pairwise differences are pure broadcasting.
    `ref_scores` / `batch_scores` accept a prebuilt `score_bank` to skip the corresponding evaluation entirely.

    r_chunk: reference-block size for the difference tensor; None picks it from
        `DIFF_BYTES_BUDGET`. Memory knob only — the returned values are byte-for-byte identical for
        every r_chunk (see the loop comment below).
    """
    num_gamma, num_eps = W.shape[0], W.shape[1]
    R, d = x_refs.shape
    B = X.shape[0]
    dgamma = gammas[1:] - gammas[:-1]   # gamma step sizes (integration widths)
    if r_chunk is None:
        r_chunk = auto_r_chunk(num_eps, B, d, X.element_size())

    total = torch.zeros(num_eps, B, R, device=X.device, dtype=X.dtype)
    for i in range(num_gamma - 1):  # one step per integration interval
        gamma = gammas[i]
        s_ref = (ref_scores[i] if ref_scores is not None
                 else _scores_at_gamma(x_refs, gamma, W[i], density, score_fn))    # (N_eps, R, d)
        s_bat = (batch_scores[i] if batch_scores is not None
                 else _scores_at_gamma(X, gamma, W[i], density, score_fn))         # (N_eps, B, d)

        
        # The loop below computes the following, but splits it to chunks over r blocks:
        #     diff   = s_ref.unsqueeze(1) - s_bat.unsqueeze(2)      # (N_eps, B, R, d)
        #     total += diff.pow(2).sum(-1) * dgamma[i]              # (N_eps, B, R)
        # As a part of the calculation of the marginal score difference,
        # ||∇log p_Yg(g·x_r + W) - ∇log p_Yg(g·x_b + W)||^2 · dgamma, accumulated across the gamma grid.
        # Chunking over r instead of calculating ||∇log p_Yg(g·x_r + W) - ∇log p_Yg(g·x_b + W)||^2 · dgamma directly.
        # Byte-for-byte identical to the unchunked math, since the sum over r is associative and commutative. 
        for r0 in range(0, R, r_chunk):
            r1 = min(r0 + r_chunk, R)
            diff = s_ref[:, r0:r1, :].unsqueeze(1) - s_bat.unsqueeze(2)   # (N_eps, B, r1-r0, d)
            total[:, :, r0:r1] += diff.pow(2).sum(-1) * dgamma[i]         # (N_eps, B, r1-r0)

        if verbose and (i + 1) % max(1, (num_gamma - 1) // 4) == 0:
            print(f'  gamma step {i+1}/{num_gamma-1}')

    return total     # (N_eps, B, R)


class GlobalIEMDistance(Distance):
    """
    Global IEM distance D_IEM(x, x') (Def. 1, f = identity), direct marginal-score formulation.
    Builds one Brownian path bank and evaluates every point against the whole X batch at once.

    Args:
        density:  Density exposing log_p_Y (optional if score_fn is given)
        gammas:   integration grid, e.g. logspace(-10, 10, 200, base=2)
        num_eps:  Brownian path samples (variance reduction)
        seed:     RNG seed for the Brownian path bank
        score_fn: pre-learned marginal score ∇log p_Y(y, gamma); when supplied it
                  replaces autograd through density.log_p_Y.
        cache_refs: keep the reference `score_bank` between calls, making the per-call cost independent of R. 
                  A VRAM knob, not a behaviour switch: cached and uncached results are byte-identical,
                  (N_gamma-1) * N_eps * R * d (251 MB at R=64, d=65536, fp32); set False if tight.
        r_chunk:  reference-block size for the pairwise difference; None derives it from
                  `DIFF_BYTES_BUDGET`. Memory knob only — no numerical effect.
    """

    def __init__(
        self,
        density: Density | None,
        gammas: Float[Tensor, "N_gamma"],
        num_eps: int = 50,
        seed: int = 123,
        verbose: bool = False,
        score_fn: ScoreFn | None = None,
        cache_refs: bool = True,
        r_chunk: int | None = None,
    ):
        if density is None and score_fn is None:
            raise ValueError("Provide either density (with log_p_Y) or score_fn")
        self.density = density
        self.gammas = gammas
        self.num_eps = num_eps
        self.seed = seed
        self.verbose = verbose
        self.score_fn = score_fn
        self.cache_refs = cache_refs
        self.r_chunk = r_chunk
        # Single entry: (key, the x_refs tensor itself, its score bank). A run has one frozen reference set, so there is nothing to evict.
        self._ref_cache: tuple[tuple, Tensor, Tensor] | None = None     # (key, x_refs, score_bank)

    def _points_key(self, x: Tensor) -> tuple:
        """Identity of a point set, for the reference-bank cache.

        `Reward` freezes `x_refs`, so address identity is a sound key;
        `self.gammas` is included so swapping the integration grid on a live instance invalidates the bank.
        """
        return (x.data_ptr(), tuple(x.shape), x.dtype, str(x.device),
                self.num_eps, self.seed, self.gammas.data_ptr(), tuple(self.gammas.shape))

    def _ref_bank(
        self,
        x_refs: Float[Tensor, "R d"],
        W: Float[Tensor, "N_gamma N_eps 1 d"],
        gammas: Float[Tensor, "N_gamma"],
    ) -> Float[Tensor, "N_gamma_minus_1 N_eps R d"]:
        """The reference `score_bank`, rebuilt only when the reference set changes."""
        key = self._points_key(x_refs)
        if self._ref_cache is not None and self._ref_cache[0] == key:
            # cache hit: the reference set is unchanged, the bank is still valid. return the cached bank
            return self._ref_cache[2]
        # cache miss: rebuild the bank and store it for future calls
        bank = score_bank(x_refs, W, gammas, self.density, self.score_fn)
        self._ref_cache = (key, x_refs, bank)
        return bank

    def pairwise(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
    ) -> Float[Tensor, "B R"]:
        """Returns D_IEM(X[b], x_refs[r]) for all b, r."""
        device, dtype = X.device, X.dtype
        d = X.shape[1]
        gammas = self.gammas.to(device=device, dtype=dtype)
        W = simulate_brownian(gammas, self.num_eps, d, self.seed, device, dtype)   # built once, shared by all points
        ref_scores = self._ref_bank(x_refs, W, gammas) if self.cache_refs else None
        # uses ref_scores if available instead of recalculating, in the case of pairwise(x_refs, x_refs)
        batch_scores = ref_scores if (ref_scores is not None
                                      and self._points_key(X) == self._points_key(x_refs)) else None
        iem_sq = iem_sq_integral(                          # (N_eps, B, R): ∫dg -> IEM^2 per path
            x_refs, X, W, gammas, self.density, self.score_fn,
            ref_scores=ref_scores, batch_scores=batch_scores,
            r_chunk=self.r_chunk, verbose=self.verbose,
        )
        iem_sq_mean = iem_sq.mean(0).clamp_min(0)          # (B, R): E_W[IEM^2]
        return self._finalize(iem_sq_mean)                 # (B, R): -> D_IEM (or D_IEM^2 in subclass)

    def _finalize(
        self, iem_sq_mean: Float[Tensor, "B R"]
    ) -> Float[Tensor, "B R"]:
        """Map E_W[IEM^2] -> the reported pairwise matrix. Default: sqrt -> D_IEM.
        Overridden by SquaredGlobalIEMDistance to return D_IEM^2 (skip the sqrt)."""
        return iem_sq_mean.sqrt()


class SquaredGlobalIEMDistance(GlobalIEMDistance):
    """Global IEM squared distance D_IEM^2(x, x') = the pre-sqrt integral E_W[IEM^2].

    Same __init__/pairwise as GlobalIEMDistance; only the final reduction differs (no sqrt), so
    `pairwise` returns D_IEM^2 directly. Injectable into any RefSelector to build the squared-distance
    tilt reward (see NormalizedExpectedDistanceReward in tilt.py).
    """

    def _finalize(
        self, iem_sq_mean: Float[Tensor, "B R"]
    ) -> Float[Tensor, "B R"]:
        return iem_sq_mean  # D_IEM^2
