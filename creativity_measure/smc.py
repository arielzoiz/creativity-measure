"""Gradient-free Sequential Monte Carlo sampler for  q_lambda(x) ∝ p(x) · exp(lambda · f(x)).

This module replaces the 2D-only `tilt.grid_normalize` path with a dimension-agnostic
**Sequential Monte Carlo (SMC)** sampler. It needs only

  (a) sampling from the base density `p` (``Density.sample`` / ``sample_fn``), and
  (b) evaluating the reward ``f`` via ``Distance.pairwise`` (through the weight-aware
      ``tilt.expected_distance``).

No ``log p`` and no gradients are required: the base-density factors cancel exactly in the
independence-Metropolis acceptance ratio (proposals are drawn from ``p`` itself). The same
``smc_sample`` is intended to later drive the pixel/EDM path unchanged — the caller simply
passes a pixel ``Density`` (``sample_fn`` from an EDM prior) and a
``GlobalIEMDistance(score_fn=edm_score_fn(...))`` (see
``creativity_measure/distances/edm_adapter.py`` for wrapping an EDM denoiser into a
``score_fn``). Phase 2 (a latent-space pCN kernel) is a separate module/plan.

Notes
-----
* ``tilt.grid_normalize`` / ``plotting.make_grid`` / ``plotting.plot_field`` are **not** used
  here (they are 2D-only). SMC only ever touches ``f``.
* **Target distribution.** ``f`` is frozen once: references ``x_refs`` and their per-reference
  ``weights`` come from a selector and stay fixed, and the ``Distance``'s Brownian seed is fixed
  at construction (``GlobalIEMDistance(seed=...)`` defaults to a fixed seed). Freezing makes
  ``f`` a deterministic function of ``x``, which is what makes the SMC/MCMC theory apply exactly.
* **Randomness / determinism.** ``Density.sample`` only accepts an integer ``seed`` (it seeds the
  global torch RNG); it cannot take a ``torch.Generator``. So for reproducibility we seed the
  global RNG once at init via ``p.sample(N, seed=seed)`` — every later ``p.sample(N)`` proposal
  then follows deterministically — and use a dedicated ``torch.Generator(seed)`` for the parts we
  control directly (Metropolis-acceptance draws and systematic resampling). Same ``seed`` ->
  identical ``SMCResult.X``, provided the caller fixed the ``Distance`` Brownian seed upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from creativity_measure.density import Density
from creativity_measure.distances.base import Distance
from creativity_measure.tilt import expected_distance


@dataclass
class SMCResult:
    """Output of :func:`smc_sample`.

    Attributes:
        X:           final particles, (N, d).
        logw:        final unnormalized log-weights, (N,). ``softmax(logw)`` gives the
                     self-normalized q̂_lambda importance weights (all-equal if the run
                     ended on a resample or with ``final_resample=True``).
        betas:       adaptive temperature schedule actually used (one entry per level).
        ess_history: ESS at each level (post-reweight, pre-resample).
        acc_history: mean independence-MH acceptance rate at each level.
    """

    X: Float[Tensor, "N d"]
    logw: Float[Tensor, "N"]
    betas: list[float]
    ess_history: list[float]
    acc_history: list[float]


def _ess_from_logw(logw: Float[Tensor, "N"]) -> float:
    """Effective sample size from unnormalized log-weights, computed in log-space.

    ESS = (sum w)^2 / sum w^2 = exp(2*logsumexp(logw) - logsumexp(2*logw)).
    Uniform log-weights -> N; a one-hot weight -> 1.
    """
    a = torch.logsumexp(logw, dim=0)
    b = torch.logsumexp(2.0 * logw, dim=0)
    return float(torch.exp(2.0 * a - b))


def _systematic_resample(
    weights: Float[Tensor, "N"],
    generator: torch.Generator,
) -> Int[Tensor, "N"]:
    """Systematic resampling: return N parent indices with E[count_i] = N * w_i.

    A single uniform draw ``u ~ U(0,1)`` fixes N evenly-spaced positions ``(i + u)/N``; each is
    mapped through the inverse CDF of ``weights``. Deterministic given ``generator``.
    """
    n = weights.shape[0]
    w = weights / weights.sum()
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0  # guard against rounding so the last position always lands
    u = torch.rand((), generator=generator, device=weights.device, dtype=weights.dtype)
    positions = (torch.arange(n, device=weights.device, dtype=weights.dtype) + u) / n
    idx = torch.searchsorted(cdf, positions)
    return idx.clamp_max_(n - 1)


def _next_dbeta(
    logw: Float[Tensor, "N"],
    fX: Float[Tensor, "N"],
    lam: float,
    beta: float,
    ess_target_count: float,
) -> float:
    """Adaptive temperature step: largest ``dβ ∈ (0, 1-β]`` keeping ESS at the target.

    The incremental log-weight at step ``dβ`` is ``dβ·lam·fX`` added to ``logw``; ESS is monotone
    decreasing in ``dβ``. We bisect for the ``dβ`` where ``ESS(logw + dβ·lam·fX) == ess_target_count``.
    If even ``dβ = 1-β`` keeps ESS above the target (e.g. ``lam == 0``), take ``dβ = 1-β`` (final level).
    """
    hi = 1.0 - beta

    def ess_at(db: float) -> float:
        return _ess_from_logw(logw + db * lam * fX)

    if ess_at(hi) >= ess_target_count:
        return hi
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ess_at(mid) >= ess_target_count:
            lo = mid
        else:
            hi = mid
    return lo


def smc_sample(
    p: Density,
    distance: Distance,
    x_refs: Float[Tensor, "R d"],
    weights: Float[Tensor, "R"] | None,
    lam: float,
    n_particles: int,
    *,
    ess_target: float = 0.5,
    n_mcmc: int = 3,
    final_resample: bool = False,
    seed: int | None = None,
) -> SMCResult:
    """Sample from  q̂_lambda(x) ∝ p(x) · exp(lambda · f(x))  via adaptive-tempering SMC.

    Gradient-free and ``log p``-free: proposals come from ``p`` (so its density cancels in the
    independence-MH ratio) and the reward ``f(x) = Σ_r w_r D(x, x'_r) / Σ_r w_r`` is the
    weight-aware ``tilt.expected_distance``.

    Args:
        p:            base density; **must** provide ``sample_fn`` (used for the initial particles
                      and every MH proposal). ``log_p_X`` is not required.
        distance:     reward distance ``D``, evaluated through ``.pairwise``.
        x_refs:       frozen references, (R, d).
        weights:      frozen per-reference weights, (R,); ``None`` => uniform. Pass a weighted
                      selector's ``.weights`` here, else its Voronoi weighting is silently dropped.
        lam:          tilt strength ``lambda``.
        n_particles:  number of SMC particles ``N``.
        ess_target:   target ESS as a fraction of ``N`` — both the conditional-ESS target for the
                      adaptive temperature and the resampling threshold.
        n_mcmc:       independence-MH rejuvenation moves per level.
        final_resample: if ``True``, resample once at the end so the returned particles are
                      equal-weight (``logw`` reset to zeros).
        seed:         seeds the global torch RNG (for ``p.sample``) and the internal generator
                      (for MH/resampling). ``None`` => nondeterministic.

    Returns:
        :class:`SMCResult`.
    """
    def f(X: Float[Tensor, "B d"]) -> Float[Tensor, "B"]:
        return expected_distance(distance, X, x_refs, weights=weights)

    # ---- init (β = 0, π_0 = p) ----------------------------------------------------------------
    X = p.sample(n_particles, seed=seed)               # seeds global RNG when seed is not None
    fX = f(X)
    logw = torch.zeros(n_particles, device=X.device, dtype=X.dtype)
    beta = 0.0
    ess_target_count = ess_target * n_particles

    gen = torch.Generator(device=X.device)
    if seed is not None:
        gen.manual_seed(seed)

    betas: list[float] = []
    ess_history: list[float] = []
    acc_history: list[float] = []

    while beta < 1.0:
        # (a) adaptive temperature
        dbeta = _next_dbeta(logw, fX, lam, beta, ess_target_count)
        final_level = dbeta >= (1.0 - beta) - 1e-12
        beta = min(beta + dbeta, 1.0)

        # (b) reweight
        logw = logw + dbeta * lam * fX
        ess = _ess_from_logw(logw)
        betas.append(beta)
        ess_history.append(ess)

        # (c) resample to reset weights. A non-final tempering level sits *exactly* at the
        # ESS target by construction (that is how dβ was chosen), so it must resample —
        # otherwise the carried weights keep ESS pinned at the target and the next level's
        # dβ collapses to 0, stalling the schedule. The final level (β just reached 1) only
        # resamples if it is genuinely below the threshold.
        if (not final_level) or ess < ess_target_count:
            idx = _systematic_resample(torch.softmax(logw, dim=0), gen)
            X, fX = X[idx], fX[idx]
            logw = torch.zeros_like(logw)

        # (d) rejuvenate: independence-MH invariant to π_β (proposals ~ p, p-factors cancel)
        accs: list[float] = []
        for _ in range(n_mcmc):
            Xp = p.sample(n_particles)
            fXp = f(Xp)
            log_a = (beta * lam) * (fXp - fX)          # log of acceptance ratio a
            u = torch.rand(n_particles, generator=gen, device=X.device, dtype=X.dtype)
            acc = u < torch.exp(log_a.clamp_max(0.0))  # accept w.p. min(1, a)
            X = torch.where(acc.unsqueeze(-1), Xp, X)
            fX = torch.where(acc, fXp, fX)
            accs.append(float(acc.to(fX.dtype).mean()))
        acc_history.append(sum(accs) / len(accs) if accs else 0.0)

        if final_level:
            break

    if final_resample:
        idx = _systematic_resample(torch.softmax(logw, dim=0), gen)
        X, fX = X[idx], fX[idx]
        logw = torch.zeros_like(logw)

    return SMCResult(
        X=X,
        logw=logw,
        betas=betas,
        ess_history=ess_history,
        acc_history=acc_history,
    )
