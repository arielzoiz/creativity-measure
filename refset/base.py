# refset: reference-set selectors for the creativity measure.
# A selector chooses R reference points x'_1..x'_R (from a Density p) used to estimate the tilt reward
#     f(x) = E_{x'~p}[ D_IEM(x, x') ]   ≈   sum_m w_m D_IEM(x, x'_m) / sum_m w_m.
# Unbiased for E_{x'~p}[.] iff refs ~ p with uniform weights (RandomRefs) or reweighted by inverse
# selection density (inverse-density-weighted FPS -- not implemented). Plain coverage selection
# (FPSRefs) is biased (mode-balanced).
#
# R may be omitted in select(): the selector then auto-chooses R by a weighted-τ LEVEL target (rank
# stability of f over probes ~ p). Auto-R needs a `distance` (to compute f) and internally-sampled probes.
from abc import ABC, abstractmethod
from typing import cast

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure.distances.base import Distance

try:
    from scipy.stats import weightedtau           # only needed for auto-R
except ImportError:                               # keep import soft so R-supplied use needs no scipy
    weightedtau = None


class RefSelector(ABC):
    """
    Base class for reference selectors.

    Construction:
        p:          sampler/density with .sample(n, seed=...) -> (n, d).
        distance:   a Distance with .pairwise(X, Y) -> (|X|, |Y|) IEM distances.
                    Required for FPS-style selection, for auto-R (R omitted), and for expected_distance().
                    The SAME distance both selects and scores references, so f is internally consistent.
        seed:       RNG seed for reference / pool draws.

    Auto-R controls (used only when select() is called without R):
        auto_r_grid:  candidate R values (doubling ladder) for the τ sweep.
        tau_target:   accept the smallest R whose weighted-τ(f_R, f_2R) >= this. τ is a unitless rank
                      correlation, so the bar transfers across dimensions; in high d, calibrate it to the
                      curve's achievable ceiling rather than assuming a fixed value is reachable. Inspect
                      `tau_history` after an auto-R call to see that ceiling.
        probe_size:   number of probes ~ p for the τ test (disjoint from references).
        auto_r_draws: draws to average over per R for STOCHASTIC selectors (deterministic ones loop once;
                      see _auto_r_effective_draws).
        fallback:     what to do if no R reaches tau_target. "raise" (default) errors -- right for a
                      notebook; "best" returns the SMALLEST R achieving the max observed τ (ties toward
                      fewer refs), so an auto-R call inside a sampler loop degrades instead of aborting.
    """

    def __init__(
        self,
        p,
        distance: Distance | None = None,
        seed: int | None = None,
        auto_r_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        tau_target: float = 0.97,
        probe_size: int = 300,
        auto_r_draws: int = 20,
        fallback: str = "raise",
    ):
        if fallback not in ("raise", "best"):
            raise ValueError(f"fallback must be 'raise' or 'best', got {fallback!r}")
        self.p = p
        self.distance: Distance | None = distance
        self.seed = seed
        self.auto_r_grid = tuple(sorted(set(auto_r_grid)))   # ascending: _auto_select_R's break relies on it
        self.tau_target = tau_target
        self.probe_size = probe_size
        self.auto_r_draws = auto_r_draws
        self.fallback = fallback
        self._tau_history: list[tuple[int, float]] = []      # (R, mean-τ) per swept R; see tau_history
        self._refs: Float[Tensor, "R d"] | None = None
        self._weights: Float[Tensor, "R"] | None = None

    # ---- subclass hooks -------------------------------------------------------------------------

    @abstractmethod
    def _refs_for_size(self, n: int, draw: int) -> Float[Tensor, "n d"]:
        """
        Return n references. `draw` indexes repeated draws: STOCHASTIC selectors should vary with it
        (so the auto-R sweep averages real reference variance); DETERMINISTIC ones may ignore it.
        Used both by select() and by the auto-R τ sweep.
        """
        ...

    def _auto_r_effective_draws(self) -> int:
        """
        How many draws the auto-R sweep loops per R. Stochastic selectors average over independent draws
        (default auto_r_draws); DETERMINISTIC selectors override to 1 (all draws identical -> no redundancy).
        """
        return self.auto_r_draws

    def _f_from_pairwise(self, pw: Float[Tensor, "K n"], n: int, draw: int) -> Float[Tensor, "K"]:
        """
        Reduce a (K, n) distance block to f over the K rows. Default = plain mean (uniform weights).
        WEIGHTED selectors override to apply per-reference weights. `n`/`draw` let weighted selectors fetch
        the matching weights for this prefix/draw. Used by BOTH the auto-R sweep and expected_distance(),
        so the reduction is identical everywhere f is computed.
        """
        return pw.mean(dim=1)
    
    def _weights_for(self, R: int) -> Float[Tensor, "R"] | None:
        """
        Per-reference weights for the FINAL (deployed/frozen) reference set of size R, cached by select()
        alongside self._refs. None => uniform (1/R). WEIGHTED selectors override; the returned weights are
        what every later expected_distance() call uses (frozen for SMC: f depends on refs AND weights).
        """
        return None

    def _max_available_size(self) -> int:
        """Upper bound on R the selector can produce (pool-limited selectors override)."""
        return self.auto_r_grid[-1] * 2

    # ---- public API -----------------------------------------------------------------------------

    def select(self, R: int | None = None, *, force: bool = False) -> Float[Tensor, "R d"]:
        """
        Return references. If R is given, use it; if R is None, auto-choose via the τ-level stopping rule.
        Weighted subclasses override to also populate self._weights for the chosen R.
        """
        if R is not None and R < 1:
            raise ValueError(f"R must be >= 1, got {R}")
        if self._refs is not None and not force:
            if R is None or R == self._refs.shape[0]:
                return self._refs                     # reuse cached refs (and the _weights set alongside)
            # R differs from cache -> fall through and reselect at the new R
        if R is None:
            R = self._auto_select_R()
        self._refs = self._refs_for_size(R, draw=0)
        self._weights = self._weights_for(R)          # uniform by default; weighted subclasses override
        return self._refs

    @property
    def weights(self) -> Float[Tensor, "R"] | None:
        """Per-reference weights for f's weighted mean. None means uniform (1/R)."""
        return self._weights

    @property
    def tau_history(self) -> list[tuple[int, float]]:
        """(R, mean-τ) for each R swept in the last auto-R call. Use it to read off the achievable τ
        ceiling and calibrate tau_target (esp. in high d). Empty until select(R=None) runs."""
        return self._tau_history

    def expected_distance(self, X: Float[Tensor, "B d"], R: int | None = None) -> Float[Tensor, "B"]:
        """
        f(X) = sum_m w_m D_IEM(X, x'_m) / sum_m w_m, the (possibly weighted) estimate of E_{x'~p}[D_IEM].
        Selects references (auto-R if R is None) and scores them with the selector's own `distance`,
        reduced via _f_from_pairwise (so uniform/weighted reduction matches the auto-R sweep exactly).
        """
        if self.distance is None:
            raise ValueError("needs a `distance`; pass distance= at construction.")
        refs = self.select(R)
        pw = self.distance.pairwise(X, refs)                       # (B, R)
        return self._f_from_pairwise(pw, refs.shape[0], draw=0)

    # ---- shared auto-R: weighted-τ level target on nested R-vs-2R --------------------------------

    def _auto_select_R(self) -> int:
        if self.distance is None:
            raise ValueError("needs a `distance`; pass distance= at construction.")
        if weightedtau is None:
            raise ImportError("Auto-R needs scipy (scipy.stats.weightedtau); install scipy or pass R.")
        probe_seed = None if self.seed is None else self.seed + 9973
        probes = self.p.sample(self.probe_size, seed=probe_seed)
        n_draws = self._auto_r_effective_draws()
        max_avail = self._max_available_size()
        max_r = max((r for r in self.auto_r_grid if 2 * r <= max_avail), default=0)
        self._tau_history = []
        for R in self.auto_r_grid:
            if 2 * R > max_avail:
                break
            taus = [self._tau_at_R(probes, R, d) for d in range(n_draws)]
            tau = float(torch.tensor(taus).mean())
            self._tau_history.append((R, tau))
            if tau >= self.tau_target:
                return R
        # No R met the target. Either error (notebook) or degrade to the best observed R (sampler loop).
        if self.fallback == "best":
            valid = [(R, t) for R, t in self._tau_history if t == t]   # drop NaN (t != t)
            if valid:
                best_tau = max(t for _, t in valid)
                return min(R for R, t in valid if t == best_tau)       # smallest R at the τ ceiling
        raise RuntimeError(
            f"weighted-τ never reached tau_target={self.tau_target} within R<={max_r} "
            f"(observed ceiling τ≈{max((t for _, t in self._tau_history), default=float('nan')):.3f}); "
            f"enlarge the pool / auto_r_grid, lower tau_target, or pass fallback='best'.")
    
    def _tau_at_R(self, probes: Float[Tensor, "K d"], R: int, draw: int) -> float:
        """
        One weighted-τ(f_R, f_2R) for the auto-R sweep. Default: nested test on a single reference set
        (first R vs first 2R of the same draw). NOTE: the 2R set is a SUPERSET of the R set, so f_R and
        f_2R share R terms and τ is biased optimistic -> chosen R may be smaller than an independent test
        would give. Nesting is intrinsic for deterministic FPS; RandomRefs varies `draw` to average real
        reference variance. Weighted selectors override to also de-correlate weight noise across R/2R.
        """
        if self.distance is None:
            raise ValueError("needs a `distance`; pass distance= at construction.")
        assert weightedtau is not None    # guaranteed by _auto_select_R's guard (narrows the soft import)
        refs2 = self._refs_for_size(2 * R, draw=draw)              # nested: first R are the R-set
        pw = self.distance.pairwise(probes, refs2)                 # (K, 2R)
        fR = self._f_from_pairwise(pw[:, :R], R, draw)             # f from first R refs
        f2R = self._f_from_pairwise(pw, 2 * R, draw)              # f from all 2R refs
        return cast(float, weightedtau(fR.numpy(), f2R.numpy())[0])   # scipy stub types [0] as object