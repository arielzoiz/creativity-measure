#
# The shared-Brownian squared global IEM distance (`global_iem.py`) PLUS the two methods the
# inference-time-gradient samplers need:
#
#   expected()            -- the closed-form weighted mean over references
#   expected_gamma_chunk()-- a partial sum over the integration axis, so the backward can be split
#
# WHY THIS FILE EXISTS. `SquaredGlobalIEMDistance` is perfectly differentiable in x given a
# differentiable `score_fn`; what it lacks is a route whose AUTOGRAD GRAPH fits beside a grad-enabled
# forward through a 12 B transformer. Without `expected()`, `tilt.expected_distance` routes through
# `pairwise`, which materialises and retains (N_eps, B, R, d) score-difference tensors -- measured
# 43.5 / 174 / 348 MB at R = 8 / 32 / 64 (d = 16384), i.e. LINEAR in R, against a flat 3.1 MB for the
# closed-form route. `r_chunk` does not help: it accumulates in place into one `total`, so every
# chunk's graph stays live. See notebooks/iid_iem_flux_check/ROADMAP.md Phase 2 for the memory ceiling
# this is measured against (a grad-enabled FLUX forward alone OOM'd at 44.51 of 44.53 GiB, batch 2).
#
# WHY A SUBCLASS IN A NEW FILE, not an edit to `global_iem.py`. `tilt.expected_distance` dispatches on
# `isinstance(distance, ExpectedDistance)`, a runtime_checkable Protocol testing only for the presence
# of an `expected` method. Giving `SquaredGlobalIEMDistance` that method would silently reroute EVERY
# existing Algorithm 1-3 caller onto a different float reduction order, perturbing f at the ~1e-7 level
# in runs already used as yardsticks for lambda_s. Additive only, exactly as `iid_global_iem.py` was.
#
# WHY THE i.i.d. HELPERS APPLY VERBATIM. `ref_score_stats` / `iid_iem_sq_expected` contain nothing
# i.i.d.-specific: they consume a (K, N_eps, R, d) bank and a per-sample weight vector. The identity
# they rest on,
#       sum_r rho_r ||s_x - s_r||^2 = ||s_x - s_bar||^2 + sum_r rho_r ||s_r - s_bar||^2,
# is over the REFERENCE axis alone and says nothing about how gamma is coupled across levels. So the
# Brownian bank slots straight in with dgamma playing the role gamma_weights plays there. Verified
# exact: rel err 1.4e-07 in float32 and 2.0e-16 in float64 against `pairwise().mean()` -- an algebraic
# identity, not an approximation -- and f bitwise-identical under a full gamma partition.
#
# THE OFF-BY-ONE, which the i.i.d. class does not have. A grid of N_gamma POINTS defines N_gamma - 1
# INTERVALS, and `score_bank` loops `range(num_gamma - 1)` internally. So a chunk over intervals
# [g_lo, g_hi) must be handed `gammas[g_lo:g_hi+1]` and `W[g_lo:g_hi+1]` -- one extra point as that
# loop's sentinel -- while `dgamma` / `s_bar` / `spread` are sliced [g_lo:g_hi]. Getting it wrong drops
# or double-counts intervals with NO error raised. `n_gamma_chunks` exists so `_reward_grad` partitions
# the interval axis rather than the point axis; see `samplers/flow_guided_common.py`.
#
# COST. Unchanged from `GlobalIEMDistance`: (N_gamma-1) * N_eps * (R + B) score rows cold,
# (N_gamma-1) * N_eps * B once the reference bank is cached. `expected()` removes R from the GRAPH and
# from the (B, R, d) temporary, not from the score-row count.

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import ScoreFn
from creativity_measure.density import Density
from creativity_measure.distances.global_iem import SquaredGlobalIEMDistance, score_bank
from creativity_measure.distances.iid_global_iem import iid_iem_sq_expected, ref_score_stats
from creativity_measure.distances.utils import simulate_brownian


class ExpectedSquaredGlobalIEMDistance(SquaredGlobalIEMDistance):
    """D_IEM^2 on a shared Brownian bank, with the closed-form mean over references.

    Same __init__, same `pairwise`, same numbers as `SquaredGlobalIEMDistance` -- this class only ADDS
    `expected` / `expected_gamma_chunk` (and detaches the reference bank, which the base does not).
    That makes it an `ExpectedDistance`, so `tilt.expected_distance` -- hence `Reward` /
    `NormalizedExpectedDistanceReward` -- takes the closed-form path and never forms the (B, R) matrix
    nor a (B, R, d) difference.

    Args: exactly `GlobalIEMDistance`'s. `gammas` is an integration GRID (e.g.
        ``torch.logspace(log2(g_lo), log2(g_hi), 11, base=2)``), not a sample of noise levels -- the
        left-endpoint rule over its N_gamma - 1 intervals is what `pairwise` already implements.

    Gradients: the reference bank is detached and cached; the X-side scores are not, so d f / d X exists
    whenever `score_fn` is differentiable in y. The autograd-through-`density` path detaches y
    (`scores.marginal_score`), so it has no X-gradient -- a differentiable `score_fn` is required.

    Only possible for the SQUARED distance: a mean of square roots does not factor, which is why
    `GlobalIEMDistance` (unsquared) has no counterpart here.
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
        super().__init__(density, gammas, num_eps=num_eps, seed=seed, verbose=verbose,
                         score_fn=score_fn, cache_refs=cache_refs, r_chunk=r_chunk)
        if gammas.ndim != 1 or gammas.shape[0] < 2:
            raise ValueError(f"gammas must be a 1-D grid of at least 2 points, got {tuple(gammas.shape)}")
        # (key, refs, ref_weights, s_bar, spread). Holds the tensors so their addresses -- part of the
        # key -- cannot be recycled while the entry lives, same reason as the base's _ref_cache.
        self._ref_stats: tuple[tuple, Tensor, Tensor | None, Tensor, Tensor] | None = None

    @property
    def n_gamma_chunks(self) -> int:
        """Length of the axis `expected_gamma_chunk` indexes: INTERVALS, not grid points.

        `samplers/flow_guided_common.py`'s `_reward_grad` reads this to partition the integration axis.
        It defaults there to ``gammas.shape[0]``, which is right for the i.i.d. class (whose gamma axis
        IS points) and off by one here -- with g_chunk=1 that default would emit a final
        (N_gamma-1, N_gamma) pair that is not a valid interval.
        """
        return self.gammas.shape[0] - 1

    def _ref_bank(
        self,
        x_refs: Float[Tensor, "R d"],
        W: Float[Tensor, "N_gamma N_eps 1 d"],
        gammas: Float[Tensor, "N_gamma"],
    ) -> Float[Tensor, "N_gamma_minus_1 N_eps R d"]:
        """The reference `score_bank`, rebuilt only when the reference set changes. Detached: refs are
        constants (`Reward` freezes them), so a graph through them would be retained for the whole run
        for nothing. Mirrors `IIDGlobalIEMDistance._ref_bank`; values are unaffected."""
        key = self._points_key(x_refs)
        if self._ref_cache is not None and self._ref_cache[0] == key:
            return self._ref_cache[2]
        bank = score_bank(x_refs.detach(), W, gammas, self.density, self.score_fn).detach()
        self._ref_cache = (key, x_refs, bank)
        return bank

    def _ref_scores(
        self, x_refs: Tensor, W: Tensor, gammas: Tensor
    ) -> Float[Tensor, "N_gamma_minus_1 N_eps R d"]:
        if self.cache_refs:
            return self._ref_bank(x_refs, W, gammas)
        return score_bank(x_refs.detach(), W, gammas, self.density, self.score_fn).detach()

    def _stats(
        self, x_refs: Tensor, ref_weights: Tensor | None, ref_scores: Tensor
    ) -> tuple[Tensor, Tensor]:
        """(s_bar, spread) of the reference bank, cached on (ref set, weights identity).

        The weights are assumed frozen, as `Reward` freezes them; an in-place edit of a weights tensor
        is not detected. Mirrors `SquaredIIDGlobalIEMDistance._stats`.
        """
        key = self._points_key(x_refs) + (None if ref_weights is None
                                          else (ref_weights.data_ptr(), tuple(ref_weights.shape)),)
        if self.cache_refs and self._ref_stats is not None and self._ref_stats[0] == key:
            return self._ref_stats[3], self._ref_stats[4]
        s_bar, spread = ref_score_stats(ref_scores, ref_weights)
        if self.cache_refs:
            self._ref_stats = (key, x_refs, ref_weights, s_bar, spread)
        return s_bar, spread

    def _batch_scores(
        self,
        X: Tensor,
        x_refs: Tensor,
        ref_scores: Tensor,
        W: Tensor,
        gammas: Tensor,
    ) -> Float[Tensor, "N_gamma_minus_1 N_eps B d"]:
        """X-side scores. Reuses the reference bank when X IS the (grad-free) reference set itself;
        otherwise evaluated with the graph intact so d/dX flows through a differentiable `score_fn`.
        The `not X.requires_grad` guard is what stops the alias swallowing a gradient request."""
        if (self.cache_refs and not X.requires_grad
                and self._points_key(X) == self._points_key(x_refs)):
            return ref_scores
        return score_bank(X, W, gammas, self.density, self.score_fn)

    def _parts(
        self, X: Tensor, x_refs: Tensor, weights: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """(gammas, W, dgamma, ref_scores, s_bar, spread) for the FULL grid, on X's device/dtype.

        `W` is always the complete Brownian bank, regenerated from `seed` and sliced by the caller --
        never redrawn per chunk. That is required, not merely convenient: `Reward` demands f be a
        deterministic function of x (CLAUDE.md invariant 1), and the Brownian increments are a
        sequential cumsum, so a chunk's rows only match the full bank's if the full bank is built first.
        The reference side is likewise computed for the full grid and sliced afterward -- it is cached,
        and its cost is O(R), which is not what chunking needs to bound.
        """
        device, dtype = X.device, X.dtype
        gammas = self.gammas.to(device=device, dtype=dtype)
        W = simulate_brownian(gammas, self.num_eps, X.shape[1], self.seed, device, dtype)
        dgamma = gammas[1:] - gammas[:-1]                   # (N_gamma-1,) left-endpoint rule widths
        ref_scores = self._ref_scores(x_refs, W, gammas)
        s_bar, spread = self._stats(x_refs, weights, ref_scores)
        return gammas, W, dgamma, ref_scores, s_bar, spread

    def expected(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
        weights: Float[Tensor, "R"] | None = None,
    ) -> Float[Tensor, "B"]:
        """sum_r rho_r D_IEM^2(X[b], x_refs[r]) with rho = weights / sum(weights) (uniform if None).

        Equal to `pairwise(X, x_refs)`'s weighted row mean up to floating-point reduction order
        (measured 1.4e-07 in float32, 2.0e-16 in float64), and differentiable in X.
        """
        gammas, W, dgamma, ref_scores, s_bar, spread = self._parts(X, x_refs, weights)
        batch_scores = self._batch_scores(X, x_refs, ref_scores, W, gammas)
        return iid_iem_sq_expected(batch_scores, s_bar, spread, dgamma)

    def expected_gamma_chunk(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
        g_lo: int,
        g_hi: int,
        weights: Float[Tensor, "R"] | None = None,
    ) -> Float[Tensor, "B"]:
        """Partial sum over the gamma INTERVALS ``[g_lo, g_hi)`` of ``expected(X, x_refs, weights)``.

        Summing over a full partition of ``[0, n_gamma_chunks)`` reproduces ``expected()`` exactly (up
        to reduction order; f measured bitwise, gradient 1.1e-07 in float32 and 0.0 at a single chunk).
        A memory knob only, exactly like `r_chunk`: peak retained backward memory scales as
        ``g_chunk / n_gamma_chunks`` (measured 0.86 / 1.67 / 4.11 / 8.17 / 16.30 MB at g_chunk =
        1 / 2 / 5 / 10 / 20 of 20 intervals), because only the batch side's score computation is
        restricted to the chunk.

        NOTE the slicing asymmetry, and see this module's header: `score_bank` loops
        ``range(num_gamma - 1)``, so it needs ``gammas``/``W`` sliced ``[g_lo:g_hi+1]`` -- the extra
        point is that loop's sentinel -- while the per-interval quantities ``dgamma``/``s_bar``/
        ``spread`` are sliced ``[g_lo:g_hi]``.
        """
        n_int = self.gammas.shape[0] - 1
        if not (0 <= g_lo < g_hi <= n_int):
            raise ValueError(
                f"g_lo={g_lo}, g_hi={g_hi} out of range for {n_int} gamma intervals "
                f"(gammas has {self.gammas.shape[0]} points)"
            )
        gammas, W, dgamma, ref_scores, s_bar, spread = self._parts(X, x_refs, weights)

        gammas_c = gammas[g_lo:g_hi + 1]                    # g_hi-g_lo+1 POINTS -> g_hi-g_lo intervals
        W_c = W[g_lo:g_hi + 1]
        if (self.cache_refs and not X.requires_grad
                and self._points_key(X) == self._points_key(x_refs)):
            batch_scores_c = ref_scores[g_lo:g_hi]
        else:
            batch_scores_c = score_bank(X, W_c, gammas_c, self.density, self.score_fn)
        return iid_iem_sq_expected(batch_scores_c, s_bar[g_lo:g_hi], spread[g_lo:g_hi],
                                   dgamma[g_lo:g_hi])
