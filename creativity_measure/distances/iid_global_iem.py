#
# Global IEM distance (Ohayon et al., ICLR 2026, Def. 1, f = identity) by i.i.d. Monte Carlo:
#   D_IEM^2(x1,x2) = ∫ E_W[ || ∇log p_Yg(g x1 + W) - ∇log p_Yg(g x2 + W) ||^2 ] dg
#                  ≈ sum_g w_g * mean_eps || s(g x1 + sqrt(g) eps_ge) - s(g x2 + sqrt(g) eps_ge) ||^2
# with (g, w_g) supplied by the caller (see `utils.log_uniform_gammas`) and eps_ge ~ N(0, I) i.i.d.
#
# Why this is the same quantity as `global_iem.py`: at each gamma the integrand depends only on the MARGINAL law
# W_g ~ N(0, g I), never on how W is coupled across gammas. The Brownian bank couples them (and needs a sequential
# cumsum + a left-endpoint rule over a grid); here every (g, eps) sample is independent, so the whole estimate is
# one batch with no sequential dependency, and its autograd graph is a sum of K = G * N_eps independent terms.
#
# FROZEN BANK. (g, eps) are drawn once from `seed` and reused by every call, so f is a deterministic function of x
# (CLAUDE.md invariant 1) and the reference scores are computed ONCE and cached, exactly as `GlobalIEMDistance` does.
#
# COST.  G * N_eps * (R + B) score rows on a cold `pairwise` call, G * N_eps * B once the reference bank is cached.
#
# Everything here is additive: `GlobalIEMDistance` / `SquaredGlobalIEMDistance` are subclassed, never edited.

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import ScoreFn
from creativity_measure.density import Density
from creativity_measure.distances.global_iem import (
    GlobalIEMDistance,
    SquaredGlobalIEMDistance,
    _scores_at_gamma,
    auto_r_chunk,
)
from creativity_measure.distances.utils import simulate_iid_noise


def iid_score_bank(
    points: Float[Tensor, "P d"],
    gammas: Float[Tensor, "G"],
    W: Float[Tensor, "G N_eps 1 d"],
    density: Density | None,
    score_fn: ScoreFn | None = None,
    *,
    batched_gamma: bool = False,
) -> Float[Tensor, "G N_eps P d"]:
    """Marginal scores at `points` for every i.i.d. (gamma, eps) sample: ∇log p_Yg(g * points + W[g, e]).

    Depends only on (points, gammas, W), never on what the points are compared against, so a frozen reference set's
    bank is built once and cached (see `IIDGlobalIEMDistance`).

    batched_gamma=False: one score call per gamma on N_eps * P rows (scalar gamma per call). The calls are independent
        of each other; the loop is only there because scalar-gamma score functions (a `Density`'s autograd score, or
        `flow_map_denoiser`) take one level per call.
    batched_gamma=True: ONE score call on G * N_eps * P rows with a per-row gamma vector. Needs a score_fn that
        accepts a (rows,) gamma, e.g. `edm_score_fn`.
    """
    G, num_eps = W.shape[0], W.shape[1]
    P, d = points.shape
    if not batched_gamma:
        return torch.stack(
            [_scores_at_gamma(points, gammas[g], W[g], density, score_fn) for g in range(G)], dim=0)
    if score_fn is None:
        raise ValueError("batched_gamma=True needs a score_fn that accepts a per-row gamma")
    y = gammas.view(G, 1, 1, 1) * points.view(1, 1, P, d) + W                       # (G, N_eps, P, d)
    gamma_rows = gammas.view(G, 1, 1).expand(G, num_eps, P).reshape(-1)             # (G * N_eps * P,)
    return score_fn(y.reshape(-1, d), gamma_rows).view(G, num_eps, P, d)


def _sample_weights(gamma_weights: Float[Tensor, "G"], num_eps: int) -> Float[Tensor, "K"]:
    """Per-sample quadrature weight omega_k = w_g / N_eps for the flattened K = G * N_eps samples (g-major)."""
    return gamma_weights.repeat_interleave(num_eps) / num_eps


def iid_iem_sq_pairwise(
    ref_scores: Float[Tensor, "G N_eps R d"],
    batch_scores: Float[Tensor, "G N_eps B d"],
    gamma_weights: Float[Tensor, "G"],
    r_chunk: int | None = None,
) -> Float[Tensor, "B R"]:
    """MC estimate of D_IEM^2 for every (batch, reference) pair: sum_k omega_k ||s_ref,k - s_bat,k||^2.

    The (K, B, r_chunk, d) difference is formed one reference block at a time and the blocks are joined with
    `torch.cat` (no in-place writes, so this stays differentiable in `batch_scores`). `r_chunk` is a memory knob only;
    each column depends on its own reference alone, so every chunking returns identical values.
    """
    G, num_eps, R, d = ref_scores.shape
    B = batch_scores.shape[2]
    K = G * num_eps
    s_ref = ref_scores.reshape(K, R, d)
    s_bat = batch_scores.reshape(K, B, d)
    omega = _sample_weights(gamma_weights, num_eps).to(s_bat)
    if r_chunk is None:
        r_chunk = auto_r_chunk(K, B, d, s_bat.element_size())
    cols = []
    for r0 in range(0, R, r_chunk):
        r1 = min(r0 + r_chunk, R)
        diff = s_ref[:, r0:r1, :].unsqueeze(1) - s_bat.unsqueeze(2)     # (K, B, r1-r0, d)
        cols.append(torch.einsum("k,kbr->br", omega, diff.pow(2).sum(-1)))
    return torch.cat(cols, dim=1)                                       # (B, R)


def ref_score_stats(
    ref_scores: Float[Tensor, "G N_eps R d"],
    ref_weights: Float[Tensor, "R"] | None = None,
) -> tuple[Float[Tensor, "G N_eps d"], Float[Tensor, "G N_eps"]]:
    """The two x-independent moments of the reference scores that `iid_iem_sq_expected` needs.

    s_bar  = sum_r rho_r s_r                    (weighted mean score per sample)
    spread = sum_r rho_r ||s_r - s_bar||^2      (weighted variance of the ref scores)
    with rho = ref_weights / sum(ref_weights) (uniform if None). Built one gamma at a time to bound the temporary.
    """
    G, _, R, _ = ref_scores.shape
    if ref_weights is None:
        rho = torch.full((R,), 1.0 / R, device=ref_scores.device, dtype=ref_scores.dtype)
    else:
        rho = ref_weights.to(ref_scores)
        rho = rho / rho.sum()
    s_bar_list, spread_list = [], []
    for g in range(G):
        s_g = ref_scores[g]                                            # (N_eps, R, d)
        s_bar_g = torch.einsum("r,erd->ed", rho, s_g)                  # (N_eps, d)
        spread_g = torch.einsum("r,er->e", rho, (s_g - s_bar_g.unsqueeze(1)).pow(2).sum(-1))
        s_bar_list.append(s_bar_g)
        spread_list.append(spread_g)
    return torch.stack(s_bar_list, dim=0), torch.stack(spread_list, dim=0)


def iid_iem_sq_expected(
    batch_scores: Float[Tensor, "G N_eps B d"],
    s_bar: Float[Tensor, "G N_eps d"],
    spread: Float[Tensor, "G N_eps"],
    gamma_weights: Float[Tensor, "G"],
) -> Float[Tensor, "B"]:
    """Weighted mean over refs of the MC D_IEM^2, in closed form: sum_r rho_r D^2(x_b, x_r).

    Bias-variance identity, exact for weights rho that sum to 1:
        sum_r rho_r ||s_x - s_r||^2 = ||s_x - s_bar||^2 + sum_r rho_r ||s_r - s_bar||^2.
    So the reference set enters only through (s_bar, spread), both x-independent and cached; the per-x work and the
    autograd graph are O(K * B * d), independent of R, and no (B, R, d) tensor exists. Both terms are non-negative,
    so there is no cancellation.
    """
    term = (batch_scores - s_bar.unsqueeze(2)).pow(2).sum(-1) + spread.unsqueeze(-1)   # (G, N_eps, B)
    return torch.einsum("g,gb->b", gamma_weights.to(term), term.mean(1))


class IIDGlobalIEMDistance(GlobalIEMDistance):
    """Global IEM distance D_IEM(x, x') by a frozen i.i.d. Monte-Carlo bank (see the module header).

    Args:
        density, score_fn, cache_refs, r_chunk, verbose: as `GlobalIEMDistance`.
        gammas:        (G,) the caller's MC noise levels (NOT a grid), e.g. from `utils.log_uniform_gammas`.
        gamma_weights: (G,) their quadrature weights, so that sum_g w_g h(gamma_g) ~ ∫ h dgamma. The class does not
                       care how they were drawn (log-uniform, stratified, midpoints, even the old grid with dgamma).
        num_eps:       eps draws per gamma; K = G * num_eps samples in all.
        seed:          RNG seed for eps (the bank is frozen: the same eps on every call).
        batched_gamma: score the whole bank in ONE call with a per-row gamma vector (needs a score_fn that accepts it,
                       e.g. `edm_score_fn`; NOT `flow_map_denoiser`, which assumes one sigma per call). Numerically
                       equal to the looped path up to batch-size effects; a speed/VRAM choice, not a behaviour switch.

    Gradients: the reference bank is detached and cached; the X-side scores are NOT, so d f / d X exists whenever
    `score_fn` is differentiable in y (the autograd-through-`density` path detaches y, so it has no X-gradient).
    """

    def __init__(
        self,
        density: Density | None,
        gammas: Float[Tensor, "G"],
        gamma_weights: Tensor,   # (G,), checked below: a plain Tensor so a length mismatch raises our ValueError, not a type error
        num_eps: int = 1,
        seed: int = 123,
        verbose: bool = False,
        score_fn: ScoreFn | None = None,
        cache_refs: bool = True,
        r_chunk: int | None = None,
        batched_gamma: bool = False,
    ):
        super().__init__(density, gammas, num_eps=num_eps, seed=seed, verbose=verbose,
                         score_fn=score_fn, cache_refs=cache_refs, r_chunk=r_chunk)
        if gammas.ndim != 1 or gamma_weights.shape != gammas.shape:
            raise ValueError(f"gammas and gamma_weights must be 1-D of equal length, got "
                             f"{tuple(gammas.shape)} and {tuple(gamma_weights.shape)}")
        if batched_gamma and score_fn is None:
            raise ValueError("batched_gamma=True needs a score_fn that accepts a per-row gamma")
        self.gamma_weights = gamma_weights
        self.batched_gamma = batched_gamma
        # (key, refs, ref_weights, s_bar, spread): moments of the cached ref bank for `expected`. Holds the tensors so
        # their addresses (part of the key) cannot be recycled while the entry lives.
        self._ref_stats: tuple[tuple, Tensor, Tensor | None, Tensor, Tensor] | None = None

    def _points_key(self, x: Tensor) -> tuple:
        return super()._points_key(x) + (self.batched_gamma,)

    def _noise(self, d: int, device: torch.device, dtype: torch.dtype) -> Float[Tensor, "G N_eps 1 d"]:
        """The frozen eps bank, regenerated from `seed` on each call (like `simulate_brownian`): deterministic and cheap."""
        return simulate_iid_noise(self.gammas.to(device=device, dtype=dtype), self.num_eps, d, self.seed, device, dtype)

    def _bank(
        self, points: Tensor, W: Tensor, gammas: Tensor
    ) -> Float[Tensor, "G N_eps P d"]:
        return iid_score_bank(points, gammas, W, self.density, self.score_fn, batched_gamma=self.batched_gamma)

    def _ref_bank(  # type: ignore[override]  # W's leading axis is G here, not N_gamma; same role and layout
        self,
        x_refs: Float[Tensor, "R d"],
        W: Float[Tensor, "G N_eps 1 d"],
        gammas: Float[Tensor, "G"],
    ) -> Float[Tensor, "G N_eps R d"]:
        """The reference bank, rebuilt only when the reference set changes. Detached: refs are constants."""
        key = self._points_key(x_refs)
        if self._ref_cache is not None and self._ref_cache[0] == key:
            return self._ref_cache[2]
        bank = self._bank(x_refs.detach(), W, gammas).detach()
        self._ref_cache = (key, x_refs, bank)
        return bank

    def _ref_scores(
        self, x_refs: Tensor, W: Tensor, gammas: Tensor
    ) -> Float[Tensor, "G N_eps R d"]:
        return self._ref_bank(x_refs, W, gammas) if self.cache_refs else self._bank(x_refs.detach(), W, gammas).detach()

    def _batch_scores(
        self, X: Tensor, x_refs: Tensor, ref_scores: Tensor, W: Tensor, gammas: Tensor
    ) -> Float[Tensor, "G N_eps B d"]:
        """X-side scores. Reuses the ref bank when X is the (grad-free) reference set itself; otherwise evaluated
        with the graph intact so d/dX flows through a differentiable score_fn."""
        if (self.cache_refs and not X.requires_grad
                and self._points_key(X) == self._points_key(x_refs)):
            return ref_scores
        return self._bank(X, W, gammas)

    def pairwise(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
    ) -> Float[Tensor, "B R"]:
        """Returns D_IEM(X[b], x_refs[r]) for all b, r (D_IEM^2 in the Squared subclass)."""
        device, dtype = X.device, X.dtype
        gammas = self.gammas.to(device=device, dtype=dtype)
        W = self._noise(X.shape[1], device, dtype)
        ref_scores = self._ref_scores(x_refs, W, gammas)
        batch_scores = self._batch_scores(X, x_refs, ref_scores, W, gammas)
        iem_sq = iid_iem_sq_pairwise(ref_scores, batch_scores, self.gamma_weights.to(device=device, dtype=dtype),
                                     self.r_chunk)
        return self._finalize(iem_sq.clamp_min(0))


class SquaredIIDGlobalIEMDistance(IIDGlobalIEMDistance, SquaredGlobalIEMDistance):
    """D_IEM^2 by the frozen i.i.d. bank. MRO picks `SquaredGlobalIEMDistance._finalize` (no sqrt).

    Also provides `expected`, the closed-form weighted mean over references, so `tilt.expected_distance` (hence
    `Reward` / `NormalizedExpectedDistanceReward`) never forms the (B, R) matrix nor a (B, R, d) difference. This is
    only possible for the SQUARED distance: a mean of square roots does not factor.
    """

    def _stats(
        self, x_refs: Tensor, ref_weights: Tensor | None, ref_scores: Tensor
    ) -> tuple[Tensor, Tensor]:
        """(s_bar, spread) of the ref bank, cached on (ref set, weights identity). The weights are assumed frozen, as
        `Reward` freezes them; an in-place edit of a weights tensor is not detected."""
        key = self._points_key(x_refs) + (None if ref_weights is None
                                          else (ref_weights.data_ptr(), tuple(ref_weights.shape)),)
        if self.cache_refs and self._ref_stats is not None and self._ref_stats[0] == key:
            return self._ref_stats[3], self._ref_stats[4]
        s_bar, spread = ref_score_stats(ref_scores, ref_weights)
        if self.cache_refs:
            self._ref_stats = (key, x_refs, ref_weights, s_bar, spread)
        return s_bar, spread

    def expected(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
        weights: Float[Tensor, "R"] | None = None,
    ) -> Float[Tensor, "B"]:
        """sum_r rho_r D_IEM^2(X[b], x_refs[r]) with rho = weights / sum(weights) (uniform if None). Differentiable in X
        (see the class docstring of `IIDGlobalIEMDistance`)."""
        device, dtype = X.device, X.dtype
        gammas = self.gammas.to(device=device, dtype=dtype)
        W = self._noise(X.shape[1], device, dtype)
        ref_scores = self._ref_scores(x_refs, W, gammas)
        s_bar, spread = self._stats(x_refs, weights, ref_scores)
        batch_scores = self._batch_scores(X, x_refs, ref_scores, W, gammas)
        return iid_iem_sq_expected(batch_scores, s_bar, spread, self.gamma_weights.to(device=device, dtype=dtype))

    def expected_gamma_chunk(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
        g_lo: int,
        g_hi: int,
        weights: Float[Tensor, "R"] | None = None,
    ) -> Float[Tensor, "B"]:
        """Partial weighted sum over gamma indices ``[g_lo, g_hi)`` of ``expected(X, x_refs, weights)``.

        Summing this over a full partition of ``[0, G)`` reproduces ``expected()`` exactly (up to
        floating-point reduction order): the total is ``sum_g w_g * mean_e term[g,e,b]``
        (``iid_iem_sq_expected``), and each gamma's term depends only on ``gammas[g]``/``W[g]`` -- a
        memory knob, exactly like the existing ``r_chunk``, never a behaviour change. Added for
        ``creativity_measure/flow_guided.py``'s OOM fallback (ROADMAP.md Phase 3 step 1b): score rows
        are ``G * N_eps * B``, so at ``B = 1`` chunking the batch axis buys nothing -- this chunks the
        MC (gamma) axis instead, which is where the memory actually is.

        Uses the SAME frozen ``W`` as ``expected()`` (builds the full-``G`` noise via ``self._noise``,
        matching the seeded generator's deterministic output, THEN slices) -- never re-draws, so the
        eps rows for gammas ``[g_lo, g_hi)`` are bit-identical to what ``expected()`` would use for the
        same indices. This is required, not just convenient: ``Reward`` demands ``f`` be a deterministic
        function of ``x`` (invariant 1), so an OOM-fallback path that quietly redrew noise would change
        what ``f`` measures without changing its code.

        The reference side (bank/stats) is computed for the FULL ``G`` and sliced afterward, never
        recomputed per chunk: it is cached (``_ref_scores``/``_stats`` short-circuit on an unchanged
        ``x_refs``) and its cost is O(R), not O(B) -- not what chunking needs to bound. Only the batch
        side's score computation (``self._bank(X, W_c, gammas_c)``, one real transformer forward per
        gamma in the chunk) is restricted to ``[g_lo, g_hi)``; that is the actual memory saving, and
        slicing a fully-computed ``batch_scores`` after the fact (instead of computing only the chunk)
        would defeat the whole point.
        """
        G = self.gammas.shape[0]
        if not (0 <= g_lo < g_hi <= G):
            raise ValueError(f"g_lo={g_lo}, g_hi={g_hi} out of range for G={G}")
        device, dtype = X.device, X.dtype
        gammas_full = self.gammas.to(device=device, dtype=dtype)
        W_full = self._noise(X.shape[1], device, dtype)

        ref_scores_full = self._ref_scores(x_refs, W_full, gammas_full)
        s_bar_full, spread_full = self._stats(x_refs, weights, ref_scores_full)

        gammas_c = gammas_full[g_lo:g_hi]
        W_c = W_full[g_lo:g_hi]
        gweights_c = self.gamma_weights.to(device=device, dtype=dtype)[g_lo:g_hi]
        s_bar_c, spread_c = s_bar_full[g_lo:g_hi], spread_full[g_lo:g_hi]

        if (self.cache_refs and not X.requires_grad
                and self._points_key(X) == self._points_key(x_refs)):
            batch_scores_c = ref_scores_full[g_lo:g_hi]
        else:
            batch_scores_c = self._bank(X, W_c, gammas_c)
        return iid_iem_sq_expected(batch_scores_c, s_bar_c, spread_c, gweights_c)
