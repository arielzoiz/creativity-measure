# Farthest-point (max-min / k-center) reference selection in the IEM metric.
#
# next = argmax_pool ( min_chosen D_IEM ): greedily add the worst-covered pool point. Covers the
# manifold's IEM geometry with few references; captures rare modes random under-weights.
# Incremental: keep min_dist, compute only ONE new column per step -> pool x R IEM evals, never pool x pool.
#
# BIAS WARNING (NOT unbiased for E_{x'~p}[.]): plain FPS over-represents diverse/rare regions vs p, so the
# uniform-weight mean estimates a MODE-BALANCED average, not the p-weighted E_{x'~p}[D_IEM]. If f is DEFINED
# as E_{x'~p}[.], use RandomRefs (or inverse-density-weighted FPS, which would restore unbiasedness --
# not implemented).
# Selection MUST use the IEM `distance`, never pixel/Euclidean geometry.

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import SampleableDensity
from creativity_measure.distances.base import Distance
from .base import RefSelector


class FPSRefs(RefSelector):
    """Greedy max-min FPS references in the IEM metric (biased: mode-balanced; see module docstring)."""

    def __init__(self, p: SampleableDensity, distance: Distance, pool_size: int = 1000, start: int | None = 0, **kw):
        super().__init__(p, distance=distance, **kw)
        self.pool_size = pool_size
        self.start = start
        self._pool: Tensor | None = None
        self._order: Tensor | None = None
        self._chosen: list[int] | None = None      # picks so far, in FPS order (a strict prefix)
        self._min_dist: Tensor | None = None        # running min_chosen D_IEM over the pool
        self._rho_hist: list[float] = []

    def _max_available_size(self) -> int:
        return self.pool_size

    def _auto_r_effective_draws(self) -> int:
        return 1                                   # deterministic given the pool: all draws identical

    def _ensure_order(self, n: int) -> None:
        # FPS order is a strict prefix, so EXTEND the greedy run up to n -- never restart from the seed.
        # Keeps _pool/_chosen/_min_dist as state so an auto-R sweep (R=1,2,4,..) builds the order ONCE.
        assert self.distance is not None       # FPS always has a distance (required by __init__)
        if self._order is not None and self._order.numel() >= n:
            return                                                 # already long enough -> caller slices
        if self._pool is None:                                     # first build: sample pool + seed pick
            pool = self.p.sample(self.pool_size, seed=self.seed)
            start = self.start
            if start is None:                                      # outlier seed without an O(N^2) row-sum
                gen = None if self.seed is None else torch.Generator().manual_seed(self.seed)
                anchor = int(torch.randint(0, pool.shape[0], (1,), generator=gen).item())
                start = int(self.distance.pairwise(pool, pool[anchor:anchor + 1]).squeeze(1).argmax())
            self._pool = pool
            self._chosen = [start]
            self._min_dist = self.distance.pairwise(pool, pool[start:start + 1]).squeeze(1)  # dist to first ref
            self._rho_hist = [self._min_dist.max().item()]
        # First build is now guaranteed done -> these three are populated together. Bind to locals
        pool, chosen, min_dist = self._pool, self._chosen, self._min_dist
        assert pool is not None and chosen is not None and min_dist is not None
        N = pool.shape[0]
        if n > N:
            raise ValueError(f"R={n} exceeds pool_size={N}; enlarge pool_size")
        while len(chosen) < n:                                     # extend, continuing the running min
            nxt = int(min_dist.argmax())                          # max-min: worst-covered point
            chosen.append(nxt)
            new_col = self.distance.pairwise(pool, pool[nxt:nxt + 1]).squeeze(1)  # the ONLY IEM call this step
            min_dist = torch.minimum(min_dist, new_col)           # running min over chosen (no recompute)
            self._rho_hist.append(min_dist.max().item())
        self._min_dist = min_dist                                 # rebound by torch.minimum; write back
        self._order = torch.tensor(chosen)

    def _refs_for_size(self, n: int, draw: int) -> Float[Tensor, "n d"]:
        # Deterministic: the FPS order is a prefix, so extend only if needed and slice. `draw` is ignored.
        self._ensure_order(n)
        assert self._pool is not None and self._order is not None   # populated by _ensure_order above
        return self._pool[self._order[:n]]

    @property
    def covering_radius_history(self) -> list[float]:
        """rho_R after each pick: max_pool min_chosen D_IEM. Diagnostic of coverage, not a stopping rule."""
        return self._rho_hist