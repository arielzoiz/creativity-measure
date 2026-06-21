# Voronoi-weighted farthest-point reference selection in the IEM metric.
#
# Same greedy max-min coverage as FPSRefs, but each selected reference x'_m is weighted by the p-mass of
# its IEM-Voronoi cell, so the weighted mean f(x) = sum_m w_m D(x,x'_m) / sum_m w_m estimates the
# p-WEIGHTED expectation E_{x'~p}[D(x,.)] -- recovering what plain (mode-balanced) FPS biases away.
#
# w_m = (# estimation points z~p whose NEAREST reference, in IEM, is m) / M_est. Assignment uses the same
# `distance` as selection (never pixels). CONSISTENT / asymptotically unbiased: exact as R->inf (cells
# shrink); at finite R it carries a within-cell quantization bias (D approximated by its value at the
# reference, not the cell average). This is a different, smaller bias than plain FPS's mode-balancing.
#
# Incremental cost: the assignment distance block D_est = pairwise(estimation_set, refs) grows ONE column
# per new reference (IEM distances never change when refs are added) and is cached; weights are the cheap
# argmin->bincount over the current columns (no IEM recompute). M_est scales with R (constant points/cell).
#
# Auto-R caveat: chosen R is OPTIMISTIC. The nested R-vs-2R test shares R reference terms (bias A) -- this
# is intrinsic to deterministic FPS and is NOT removable. We DO remove the weight-noise correlation (bias B)
# by estimating the R-side and 2R-side weights from two INDEPENDENT estimation sets in the sweep. Treat the
# auto-chosen R as a lower bound; consider a margin before an expensive (e.g. image-space) SMC run.

from typing import cast

import torch
from jaxtyping import Float
from torch import Tensor
from scipy.stats import weightedtau

from creativity_measure._types import SampleableDensity
from creativity_measure.distances.base import Distance
from .fps import FPSRefs


class WeightedFPSRefs(FPSRefs):
    """
    FPS references with Voronoi (inverse-density) weights -> consistent estimator of E_{x'~p}[D_IEM].

    Args (beyond FPSRefs):
        points_per_cell: target estimation points per Voronoi cell (sets M_est = points_per_cell * R).
                         Holds weight quality ~constant as R grows; ~1/sqrt(points_per_cell) rel. error/cell.
        est_floor:       minimum M_est (guards small R: enough points to cover p's modes / rare cells).
    """

    def __init__(
        self,
        p: SampleableDensity,
        distance: Distance,
        pool_size: int = 1000,
        start: int | None = 0,
        points_per_cell: int = 64,
        est_floor: int = 1000,
        **kw,
    ):
        super().__init__(p, distance=distance, pool_size=pool_size, start=start, **kw)
        self.points_per_cell = points_per_cell
        self.est_floor = est_floor
        # cached estimation sets + their incremental D_est blocks (one col per reference).
        # role "A"/"B" = the two INDEPENDENT sets used to de-correlate the R-/2R-sides in the auto-R sweep.
        self._est: dict[str, Tensor] = {}                 # role -> (M_est, d) estimation points ~ p
        self._dest: dict[str, Tensor] = {}                # role -> (M_est, n_cols) cached IEM distances

    # ---- estimation sets + incremental assignment block ----------------------------------------

    def _m_est(self, R_max: int) -> int:
        return max(self.est_floor, self.points_per_cell * R_max)

    def _est_seed(self, role: str) -> int | None:
        # distinct from refs (pool), probes (seed+9973), and the other role; None stays None (irreproducible).
        if self.seed is None:
            return None
        return self.seed + (12289 if role == "A" else 24571)

    def _ensure_est(self, role: str, R_max: int) -> None:
        # Draw the role's estimation set ONCE (sized for the sweep's max R), so D_est only grows columns.
        if role not in self._est:
            self._est[role] = self.p.sample(self._m_est(R_max), seed=self._est_seed(role))
            self._dest[role] = self._est[role].new_empty((self._est[role].shape[0], 0))

    def _dest_cols(self, role: str, n: int) -> Float[Tensor, "M n"]:
        # Ensure the cached D_est for `role` has >= n columns (= dist to the first n FPS refs), growing
        # ONE column per missing reference. IEM distances never change when refs are added -> append-only.
        self._ensure_order(n)                                          # FPS order extended to >= n (prefix)
        assert self._pool is not None and self._order is not None and self.distance is not None
        est = self._est[role]
        dest = self._dest[role]
        have = dest.shape[1]
        if have < n:
            new_refs = self._pool[self._order[have:n]]                 # the references not yet in D_est
            new_cols = self.distance.pairwise(est, new_refs)           # (M_est, n-have): the ONLY IEM calls
            self._dest[role] = torch.cat([dest, new_cols], dim=1)
        return self._dest[role][:, :n]

    def _voronoi_weights(self, role: str, R: int) -> Float[Tensor, "R"]:
        # w_m = fraction of estimation points whose nearest (IEM) reference among the first R is m.
        dcols = self._dest_cols(role, R)                              # (M_est, R) cached distances
        nearest = dcols.argmin(dim=1)                                 # (M_est,) assigned cell per point
        counts = torch.bincount(nearest, minlength=R).to(dcols.dtype)
        return counts / counts.sum()                                  # sums to 1; empty cells -> weight 0

    # ---- reductions: weighted mean everywhere f is computed ------------------------------------

    def _weighted_mean(self, pw: Float[Tensor, "K n"], w: Float[Tensor, "n"]) -> Float[Tensor, "K"]:
        w = w.to(pw)
        return (pw * w).sum(dim=1) / w.sum()

    def _weights_for(self, R: int) -> Float[Tensor, "R"]:
        # FROZEN deployment weights for the chosen R, from a SINGLE estimation set (role "A"), sized to R.
        # Cached with self._refs by base.select() -> f (refs AND weights) is fixed for the whole SMC run.
        self._ensure_est("A", R)
        return self._voronoi_weights("A", R)

    def _f_from_pairwise(self, pw: Float[Tensor, "K n"], n: int, draw: int) -> Float[Tensor, "K"]:
        # Used by expected_distance() (n == chosen R). Reduce with the frozen deployment weights ("A").
        return self._weighted_mean(pw, self._voronoi_weights("A", n))

    def _tau_from_block(self, block: Float[Tensor, "K twoMaxR"], R: int, draw: int) -> float:
        # Slice the precomputed (probes x 2*max_r) block for refs; weights from independent sets A (R-side)
        # and B (2R-side) -> removes shared-weight-noise inflation (bias B). Reference nesting (A) remains.
        assert weightedtau is not None
        R_max = self._max_swept_R()
        self._ensure_est("A", R_max)
        self._ensure_est("B", R_max)
        wR = self._voronoi_weights("A", R)                          # R-side weights  <- set A
        w2R = self._voronoi_weights("B", 2 * R)                     # 2R-side weights <- set B (independent)
        fR = self._weighted_mean(block[:, :R], wR)
        f2R = self._weighted_mean(block[:, :2 * R], w2R)
        return cast(float, weightedtau(fR.numpy(), f2R.numpy())[0])

    def _max_swept_R(self) -> int:
        # Largest R the auto-R sweep will reach (so estimation sets / D_est are sized once for the whole sweep).
        cap = self._max_available_size()
        return max((r for r in self.auto_r_grid if 2 * r <= cap), default=self.auto_r_grid[0])