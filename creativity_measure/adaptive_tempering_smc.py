"""Sequential Monte Carlo sampler for  q_lambda(x) ∝ p(x) · exp(lambda · f(x)).

Dimension-agnostic SMC with a **pluggable rejuvenation kernel**:

* `IndependenceKernel` — global independence-MH: proposes fresh draws ``x' ~ p``; the
    base density cancels in the acceptance ratio (no ``log p``, no gradients). Efficient when ``q ≈ p``.
* `PCNKernel` — a *local*, prior-preserving move in the diffusion model's latent Gaussian
    space. Carries each particle as a latent ``z`` with ``x = G(z)`` (deterministic EDM prob-flow ODE, see
    ``creativity_measure/generators/``); a pCN proposal ``z' = sqrt(1-s^2) z + s·xi`` is reversible w.r.t. ``N(0,I)``,
    so the Gaussian prior cancels and acceptance is ``min(1, exp[beta·lambda·(f(G(z'))-f(G(z)))])``.
    Mixes well when ``q`` is pushed off the data manifold (higher ``lambda``), where independence-MH collapses.

The tempering / ESS / resampling / frozen-refs machinery is shared by both kernels; ``f`` is the weight-aware
``tilt.expected_distance`` evaluated in data space against frozen references (identical for both kernels).

Notes
-----
* **Target distribution** ``f`` is frozen once: references ``x_refs`` and ``weights`` come from a selector
    and stay fixed; the ``Distance``'s Brownian seed is fixed at construction. This makes ``f`` a deterministic
    function of ``x`` — a required assumption for SMC/MCMC.
* **Determinism** All randomness is driven from a single ``torch.Generator(seed)`` (pCN draws / MH-accept / resampling,
    *and* `IndependenceKernel`'s ``x' ~ p`` proposals, which thread that same generator through ``Density.sample``). No
    global torch RNG state is touched, so kernel runs are reentrant / thread-safe; same ``seed`` -> identical
    ``AdaptiveTemperingSMCResult.X`` provided the caller fixed the ``Distance`` Brownian seed upstream.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure.density import Density
from creativity_measure.smc_common import _ess_from_logw, _systematic_resample
from creativity_measure.tilt import Reward

RewardFn = Callable[[Tensor], Tensor]   # f(X) -> (N,)

# ``_ess_from_logw`` / ``_systematic_resample`` live in ``smc_common`` (shared with ``diffusion_smc``);
# re-exported here so existing ``creativity_measure.adaptive_tempering_smc`` imports keep resolving.
__all__ = ["_ess_from_logw", "_systematic_resample"]


@dataclass
class LevelSnapshot:
    """The particle cloud at one tempering level, captured after rejuvenation.

    Level ``k`` targets  π_{β_k} ∝ p·exp(β_k·λ·f) = q_{λ_eff}  with ``lam_eff = β_k · λ``. So a *single*
    run at ``λ`` carries a whole λ-sweep: each snapshot is a valid sample set for its own effective tilt,
    against the identical reward, references and base density (which separate per-λ runs never quite match).

    Caveat: consecutive snapshots share ancestry — a snapshot is not an independent draw from q_{lam_eff},
    and neighbouring levels are strongly correlated. ``ancestors`` makes that lineage explicit rather than
    hiding it.

    Attributes:
        beta:      tempering level β ∈ [0, 1].
        lam_eff:   ``beta * lam`` — the tilt strength this cloud actually targets.
        X:         (N, d) particles after the level's rejuvenation sweeps.
        fX:        (N,) reward at ``X`` (already computed by the sweeps — free to keep).
        logw:      (N,) unnormalized log-weights; zeros unless the level skipped resampling.
        ancestors: (N,) long — for particle ``i``, the index of its parent in the *previous* snapshot's
                   ``X``. ``arange(N)`` on levels that skipped resampling. Walk it backwards to trace one
                   particle's lineage across λ_eff.
    """

    beta: float
    lam_eff: float
    X: Float[Tensor, "N d"]
    fX: Float[Tensor, "N"]
    logw: Float[Tensor, "N"]
    ancestors: Tensor


@dataclass
class AdaptiveTemperingSMCResult:
    """Output of :func:`adaptive_tempering_smc_sample`.

    Attributes:
        X:          final particles, (N, d).
        logw:       final unnormalized log-weights, (N,). ``softmax(logw)`` gives the self-normalized
                    q̂_lambda weights (all-equal if the run ended on a resample or with ``final_resample``).
        betas:      adaptive temperature schedule actually used (one entry per level).
        ess_history: ESS at each level (post-reweight, pre-resample).
        acc_history: mean rejuvenation acceptance rate at each level.
        n_mcmc_history: number of rejuvenation sweeps actually run at each level (constant in
                    fixed-``n_mcmc`` mode; adaptive per level otherwise).
        stop_reason_history: why rejuvenation stopped at each level — ``"decorr"`` (decorrelation
                    threshold), ``"cap"`` (hit ``max_n_mcmc``), or ``"fixed"`` (non-adaptive ``n_mcmc`` path).
        levels:     per-level `LevelSnapshot`s when the run was given ``keep_levels=True``; ``None`` otherwise.
                    **Indexing:** ``levels[0]`` is the β=0 initial cloud (π_0 = p, ``lam_eff=0``) — the free
                    untilted baseline — so ``len(levels) == len(betas) + 1`` and ``levels[k]`` corresponds to
                    ``betas[k-1]`` for k ≥ 1. With ``final_resample=True``, ``X`` above is resampled once more
                    and so differs from ``levels[-1].X``.
    """

    X: Float[Tensor, "N d"]
    logw: Float[Tensor, "N"]
    betas: list[float]
    ess_history: list[float]
    acc_history: list[float]
    n_mcmc_history: list[int]
    stop_reason_history: list[str]
    levels: list[LevelSnapshot] | None = None


# ---------------------------------------------------------------------------------------------------
# Kernel-agnostic helpers
# ---------------------------------------------------------------------------------------------------

def _next_dbeta(
    logw: Float[Tensor, "N"],
    fX: Float[Tensor, "N"],
    lam: float,
    beta: float,
    ess_target_count: float,
) -> float:
    """Largest ``dβ ∈ (0, 1-β]`` keeping ESS at the target (bisection; ESS is monotone-decreasing in dβ).

    Returns ``1-β`` when even the full remaining step holds ESS ≥ target; otherwise the
    returned dβ puts ESS *at* the target by construction.
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


MAX_N_MCMC: int = 30
MIN_N_MCMC: int = 4

@dataclass(frozen=True)
class RejuvenationStop:
    """Stopping policy for adaptive-length rejuvenation (used when ``adaptive_tempering_smc_sample(n_mcmc=None)``).

    Each tempering level keeps applying π_β-invariant sweeps until the resampled duplicates have
    **decorrelated** — measured by the kernel's own geometry-aware ``decorrelation`` — subject to a
    ``min_n_mcmc`` floor and the ``max_n_mcmc`` cap.

    Attributes:
        min_n_mcmc:      always run at least this many sweeps before the decorrelation stop can fire.
        max_n_mcmc:      hard cap on sweeps per level.
        decorr_threshold: stop once ``kernel.decorrelation`` (1=still correlated, 0=decorrelated) < this.
    """

    min_n_mcmc: int = MIN_N_MCMC
    max_n_mcmc: int = MAX_N_MCMC
    decorr_threshold: float = 0.2


# ---------------------------------------------------------------------------------------------------
# Pluggable rejuvenation kernels
# ---------------------------------------------------------------------------------------------------

@dataclass
class _State:
    """Mutable particle state carried through the SMC loop."""
    X: Tensor                                   # (N, d) particles in data space
    fX: Tensor                                  # (N,)   reward f(X)
    logw: Tensor                                # (N,)   unnormalized log-weights
    aux: dict[str, Tensor] = field(default_factory=dict)   # kernel-specific (e.g. latent Z for pCN)


class Kernel(ABC):
    """A rejuvenation kernel: how particles are initialized and moved (invariant to π_β)."""

    default_ess_target: float = 0.5
    default_n_mcmc: int = 3

    @abstractmethod
    def init(
        self, n_particles: int, f: RewardFn, *,
        generator: torch.Generator, device: torch.device, dtype: torch.dtype,
    ) -> _State:
        """Draw initial particles (π_0 = p), set fX and logw=0."""

    @abstractmethod
    def reorder(self, state: _State, idx: Tensor) -> None:
        """Reindex X and any aux state on resampling (the loop reindexes logw)."""

    @abstractmethod
    def step(
        self, state: _State, *,
        beta: float, lam: float, f: RewardFn, generator: torch.Generator,
    ) -> float:
        """Apply **one** π_β-invariant move in place; return that sweep's mean acceptance rate.

        Invariant means: if the cloud is marginally π_β before the move, it still is after.
        Sweeps buy diversity — they break up the duplicates left by resampling — never a change of target,
        so any number of them compose without biasing the level.
        """

    @abstractmethod
    def snapshot_baseline(self, state: _State) -> dict[str, Tensor]:
        """Clone the state that `decorrelation` compares against (the post-resample positions)."""

    @abstractmethod
    def decorrelation(self, state: _State, baseline: dict[str, Tensor]) -> float:
        """Scalar in ``[0, 1]``: 1 = still fully correlated with ``baseline``, 0 = fully decorrelated."""

    def rejuvenate(
        self, state: _State, *,
        beta: float, lam: float, f: RewardFn, n_mcmc: int, generator: torch.Generator,
    ) -> float:
        """Apply ``n_mcmc`` π_β-invariant moves in place; return the mean acceptance rate."""
        accept_rates = [
            self.step(state, beta=beta, lam=lam, f=f, generator=generator)
            for _ in range(n_mcmc)
        ]
        return sum(accept_rates) / len(accept_rates) if accept_rates else 0.0


class IndependenceKernel(Kernel):
    """global independence-MH: proposals ``x' ~ p`` (base density cancels)."""

    def __init__(self, p: Density):
        self.p = p

    def init(self, n_particles, f, *, generator, device, dtype):
        X = self.p.sample(n_particles, generator=generator)   # threaded RNG (no global state)
        fX = f(X)
        logw = torch.zeros(n_particles, device=X.device, dtype=X.dtype)
        return _State(X=X, fX=fX, logw=logw)

    def reorder(self, state, idx):
        state.X = state.X[idx]
        state.fX = state.fX[idx]

    def step(self, state, *, beta, lam, f, generator):
        n = state.X.shape[0]
        Xp = self.p.sample(n, generator=generator)
        fXp = f(Xp)
        # Metropolis-Hastings. Proposals are drawn from p itself, so the base density cancels and only
        # the tilt survives: log a = βλ(f(x') - f(x)) — no log p, no ∇f.
        log_a = (beta * lam) * (fXp - state.fX)
        u = torch.rand(n, generator=generator, device=state.X.device, dtype=state.X.dtype)
        accepted = u < torch.exp(log_a.clamp_max(0.0))   # clamp_max(0) is the min(1, ·) of the MH rule
        # On rejection, keeping the old (X, fX).
        # That, plus the cancellation above, is what makes the sweep π_β-invariant.
        state.X = torch.where(accepted.unsqueeze(-1), Xp, state.X)
        state.fX = torch.where(accepted, fXp, state.fX)
        return float(accepted.to(state.fX.dtype).mean())

    def snapshot_baseline(self, state):
        return {"X": state.X.clone()}

    def decorrelation(self, state, baseline):
        # Global proposals: one accepted x'~p fully replaces the state, so the exact decorrelation signal
        # is the fraction of particles still identical to their resampled parent (rejected -> bitwise equal).
        moved = (state.X != baseline["X"]).flatten(1).any(dim=1)
        return float((~moved).to(state.fX.dtype).mean())


class PCNKernel(Kernel):
    """latent-space pCN: local, prior-preserving moves via a generator ``G(z) -> x``.

    The unitless defaults below were locked by the qualitative calibration in ``phase2_pcn_calibration.ipynb``
    (a 2D toy with ``grid_normalize`` as ground truth). Raise ``ess_target`` toward 0.9 for strong tilts.

    Args:
        generator_fn: deterministic map ``G: (N, latent_dim) -> (N, d)`` with ``G(N(0,I)) ~ p``
                      (``generators.density_generator`` for 2D toy example, ``generators.edm_generator`` / ``build_edm_pixel_generator`` for pixel / latent space).
        latent_dim:   flat latent dimensionality.
        s0:           initial pCN step size in (0, 1); only the warm-up, the step then adapts.
        target_acc:   acceptance the step size adapts toward; ~0.23 is the high-d optimum.
        adapt_rate:   log-step adaptation rate.
    """

    default_ess_target = 0.7        # calibrated: a finer schedule is needed for off-manifold tilts
    default_n_mcmc = 4

    def __init__(
        self,
        generator_fn: Callable[[Tensor], Tensor],
        latent_dim: int,
        *,
        s0: float = 0.4,
        target_acc: float = 0.23,
        adapt_rate: float = 0.1,
    ):
        self.G = generator_fn
        self.latent_dim = latent_dim
        self.s = float(s0)
        self.target_acc = target_acc
        self.adapt_rate = adapt_rate

    def init(self, n_particles, f, *, generator, device, dtype):
        Z = torch.randn(n_particles, self.latent_dim, generator=generator, device=device, dtype=dtype)
        X = self.G(Z)
        fX = f(X)
        logw = torch.zeros(n_particles, device=device, dtype=dtype)
        return _State(X=X, fX=fX, logw=logw, aux={"Z": Z})

    def reorder(self, state, idx):
        state.X = state.X[idx]
        state.fX = state.fX[idx]
        state.aux["Z"] = state.aux["Z"][idx]

    def step(self, state, *, beta, lam, f, generator):
        Z = state.aux["Z"]
        n = Z.shape[0]
        xi = torch.randn(Z.shape, generator=generator, device=Z.device, dtype=Z.dtype)
        Zp = math.sqrt(1.0 - self.s * self.s) * Z + self.s * xi
        Xp = self.G(Zp)
        fXp = f(Xp)
        # Metropolis-Hastings. The pCN proposal is reversible w.r.t. the latent prior N(0,I),
        # so the prior cancels and only the tilt survives: log a = βλ(f(x') - f(x)). No log p, no ∇f.
        log_a = (beta * lam) * (fXp - state.fX)
        u = torch.rand(n, generator=generator, device=Z.device, dtype=Z.dtype)
        accepted = u < torch.exp(log_a.clamp_max(0.0))   # clamp_max(0) is the min(1, ·) of the MH rule
        accept_mask = accepted.unsqueeze(-1)   # (N, 1): a particle moves as a unit, never component-wise
        # Rejection keeps the old (Z, X, fX).
        # That, plus reversibility, is what makes the sweep π_β-invariant.
        state.aux["Z"] = torch.where(accept_mask, Zp, Z)
        state.X = torch.where(accept_mask, Xp, state.X)
        state.fX = torch.where(accepted, fXp, state.fX)

        accept_rate = float(accepted.to(state.fX.dtype).mean())
        # Robbins-Monro: nudge log-step toward the target acceptance, clamp s in (1e-3, 0.999).
        new_log_s = math.log(self.s) + self.adapt_rate * (accept_rate - self.target_acc)
        self.s = min(0.999, max(1e-3, math.exp(new_log_s)))
        return accept_rate

    def snapshot_baseline(self, state):
        return {"Z": state.aux["Z"].clone()}

    def decorrelation(self, state, baseline):
        # Local moves accumulate: measure how far the latent z_t has drifted from its post-resample value, z_0,
        # via a dimension-robust normalized Expected Squared Jumping Distance (ESJD) autocorrelation metric:
        # 1 - E‖z_t-z_0‖² / (E‖z_0‖² + E‖z_t‖²). This is 1 at the baseline and -> 0 at independence in ANY dimension
        # (the cross term vanishes regardless of d), so one threshold transfers across latent sizes — unlike |cos|,
        # whose floor grows as d shrinks.
        z0 = baseline["Z"].flatten(1)
        zt = state.aux["Z"].flatten(1)
        num = (zt - z0).square().sum()
        den = z0.square().sum() + zt.square().sum() + 1e-12
        return float((1.0 - num / den).clamp(0.0, 1.0))


# ---------------------------------------------------------------------------------------------------
# Shared SMC loop
# ---------------------------------------------------------------------------------------------------

def _resample_and_reset(kernel: Kernel, state: _State, generator: torch.Generator) -> Tensor:
    """Collapse the weighted cloud into an equally-weighted one: systematic resample, reindex, logw←0.

    Returns the resample index (parent of each particle) so callers can record ancestry.
    """
    idx = _systematic_resample(torch.softmax(state.logw, dim=0), generator)
    kernel.reorder(state, idx)
    state.logw = torch.zeros_like(state.logw)
    return idx


def _snapshot(
    state: _State, *, beta: float, lam: float, ancestors: Tensor,
    device: torch.device | str | None,
) -> LevelSnapshot:
    """Clone the current cloud into a `LevelSnapshot` (never a view into live state)."""
    def keep(t: Tensor) -> Tensor:
        t = t.detach().clone()
        return t if device is None else t.to(device)

    return LevelSnapshot(
        beta=beta, lam_eff=beta * lam,
        X=keep(state.X), fX=keep(state.fX), logw=keep(state.logw), ancestors=keep(ancestors),
    )


def _rejuvenate_adaptive(
    kernel: Kernel,
    state: _State,
    *,
    beta: float,
    lam: float,
    f: RewardFn,
    generator: torch.Generator,
    stop: RejuvenationStop,
) -> tuple[float, int, str]:
    """Sweep until the resampled duplicates decorrelate; return (mean_acc, n_sweeps, reason).

    Stop (``reason="decorr"``): ``kernel.decorrelation`` < ``decorr_threshold`` (geometry-aware), gated by
    the ``min_n_mcmc`` floor. If it does not fire within ``max_n_mcmc`` sweeps, ``reason="cap"`` — this is
    the correct behavior for a stuck (low-acceptance) level: keep spending the full budget on transport.

    Each sweep is π_β-invariant (see ``Kernel.step``), so the cloud's law is unchanged however many run;
    what varies is only how far the duplicates left by resampling have spread. The *number* of sweeps,
    though, is read off the cloud itself, making the composite kernel adaptive rather than exactly
    invariant — the same approximation already accepted for the ESS-driven β schedule, and justified
    asymptotically in N rather than per run.
    """
    baseline = kernel.snapshot_baseline(state)
    accept_rates: list[float] = []
    reason = "cap"
    for t in range(stop.max_n_mcmc):
        accept_rates.append(kernel.step(state, beta=beta, lam=lam, f=f, generator=generator))
        if t < stop.min_n_mcmc - 1:
            continue
        if kernel.decorrelation(state, baseline) < stop.decorr_threshold:
            reason = "decorr"
            break
    mean_accept_rate = sum(accept_rates) / len(accept_rates) if accept_rates else 0.0
    return mean_accept_rate, len(accept_rates), reason


def _run_smc(
    kernel: Kernel,
    f: RewardFn,
    lam: float,
    n_particles: int,
    *,
    ess_target: float,
    n_mcmc: int | None,
    stop: RejuvenationStop,
    final_resample: bool,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    keep_levels: bool = False,
    snapshot_device: torch.device | str | None = "cpu",
) -> AdaptiveTemperingSMCResult:
    state = kernel.init(n_particles, f, generator=generator, device=device, dtype=dtype)
    beta = 0.0
    ess_target_count = ess_target * n_particles

    betas: list[float] = []
    ess_history: list[float] = []
    acc_history: list[float] = []
    n_mcmc_history: list[int] = []
    stop_reason_history: list[str] = []

    # Snapshotting only reads the state (deep copies), so it never touches ``generator``.
    # A run with ``keep_levels=True`` is bit-identical to one without.
    no_resample = torch.arange(n_particles, device=device)
    levels: list[LevelSnapshot] | None = None
    if keep_levels:
        # levels[0]: the untilted β=0 cloud (π_0 = p), i.e. the free lam_eff=0 baseline row.
        levels = [_snapshot(state, beta=0.0, lam=lam, ancestors=no_resample, device=snapshot_device)]

    while beta < 1.0:
        # (a) adaptive temperature
        #   dβ hitting its 1-β ceiling means this level lands on β=1
        #   (1e-12 / min() absorb bisection float slop).
        dbeta = _next_dbeta(state.logw, state.fX, lam, beta, ess_target_count)
        final_level = dbeta >= (1.0 - beta) - 1e-12
        beta = min(beta + dbeta, 1.0)

        # (b) reweight
        #   π_β ∝ p·exp(βλf), so the incremental weight is π_{β+dβ}/π_β = exp(dβ·λ·f(x)).
        #   p cancels, making the update exact and gradient-free. state.fX matches state.X (kernel.step refreshes it, reorder permutes it).
        state.logw = state.logw + dbeta * lam * state.fX
        ess = _ess_from_logw(state.logw)
        betas.append(beta)
        ess_history.append(ess)

        # (c) resample
        #   collapse weights back into particles (duplicate heavy, drop light) so (d) rejuvenates an equally-weighted cloud.
        #   Non-final levels resample unconditionally.
        #   only the final level can sit well above target, so only it gets a real test against the ESS target.
        ancestors = no_resample
        if (not final_level) or ess < ess_target_count:
            ancestors = _resample_and_reset(kernel, state, generator)

        # (d) rejuvenate with the chosen kernel (invariant to π_β); adaptive length unless n_mcmc is fixed
        if n_mcmc is None:
            accept_rate, n_used, reason = _rejuvenate_adaptive(
                kernel, state, beta=beta, lam=lam, f=f, generator=generator, stop=stop
            )
        else:
            accept_rate = kernel.rejuvenate(state, beta=beta, lam=lam, f=f, n_mcmc=n_mcmc, generator=generator)
            n_used, reason = n_mcmc, "fixed"
        acc_history.append(accept_rate)
        n_mcmc_history.append(n_used)
        stop_reason_history.append(reason)

        # (e) snapshot: sweeps done, β about to advance — the cloud is a valid sample of q_{β·λ} right here.
        if levels is not None:
            levels.append(_snapshot(state, beta=beta, lam=lam, ancestors=ancestors, device=snapshot_device))

        if final_level:
            break

    if final_resample:
        _resample_and_reset(kernel, state, generator)

    return AdaptiveTemperingSMCResult(
        X=state.X,
        logw=state.logw,
        betas=betas,
        ess_history=ess_history,
        acc_history=acc_history,
        n_mcmc_history=n_mcmc_history,
        stop_reason_history=stop_reason_history,
        levels=levels,
    )


def adaptive_tempering_smc_sample(
    reward: Reward,
    lam: float,
    n_particles: int,
    *,
    kernel: Kernel,
    ess_target: float | None = None,
    n_mcmc: int | None = None,
    stop: RejuvenationStop | None = None,
    max_n_mcmc: int | None = None,
    final_resample: bool = False,
    keep_levels: bool = False,
    snapshot_device: torch.device | str | None = "cpu",
    seed: int | None = None,
) -> AdaptiveTemperingSMCResult:
    """Sample from  q̂_lambda(x) ∝ p(x) · exp(lambda · f(x))  via adaptive-tempering SMC.

    The reward ``f(x) = Σ_r w_r D(x, x'_r) / Σ_r w_r`` is the frozen `~creativity_measure.tilt.Reward`
    (distance + references + weights) bundled by a ``refset`` selector.
    ``f`` is identical for every kernel.

    Choosing parameters:
    ``lam`` - set ``lambda = m * lambda_0`` with the data-derived scale ``lambda_0 = std(log p at refs) / std(f at refs)`` and a
                    unitless multiplier ``m`` (how far off the manifold you want).
    ``reward`` — from a ``refset`` selector via ``selector.reward()`` (auto-R picks ``R`` by a
                    unitless τ-rule; ``RandomRefs`` => uniform weights, ``WeightedFPSRefs`` => Voronoi).
    ``kernel`` - `PCNKernel` for strong / off-manifold tilts;
                    `IndependenceKernel`(p) is fine when ``q ≈ p`` (mild tilt). The base density ``p`` lives in `IndependenceKernel`.
    ``n_particles``— as large as compute allows; check ``AdaptiveTemperingSMCResult.ess_history[-1]`` is adequate.
    ``ess_target`` / ``n_mcmc`` — Tune via the ESS- and acceptance-vs-β histories. Leaving ``n_mcmc=None``
                    runs **adaptive** rejuvenation: each level sweeps until the resampled duplicates
                    decorrelate (per-kernel metric), bounded by ``MAX_N_MCMC`` (override the cap per run
                    with ``max_n_mcmc=``); inspect ``AdaptiveTemperingSMCResult.n_mcmc_history`` for the per-level effort.
                    In high dimensions decorrelation rarely fires, so the cap sets the per-level budget —
                    lower ``max_n_mcmc`` to bound cost.

    Args:
        reward:       frozen reward ``f`` (distance, references, weights); its ``x_refs`` also fixes the run's device/dtype.
        lam:          tilt strength ``lambda`` (see "Choosing parameters" above).
        n_particles:  number of SMC particles ``N``.
        kernel:       rejuvenation kernel (required): `IndependenceKernel`(p) or `PCNKernel`(G, d).
        ess_target:   target ESS as a fraction of ``N`` (tempering target + resampling threshold).
                      ``None`` => the kernel's default (Independence 0.5; pCN 0.7). Raise toward 0.9
                      for strong tilts — the dominant recovery lever.
        n_mcmc:       rejuvenation moves per level. ``None`` (default) => **adaptive** length per level
                      (see ``stop``); pass an ``int`` to force a fixed number of sweeps (legacy behavior).
        stop:         `RejuvenationStop` policy for the adaptive path (min/max sweeps, decorrelation
                      threshold). ``None`` => defaults. Ignored when ``n_mcmc`` is an int.
        max_n_mcmc:   convenience cap on adaptive sweeps per level — shorthand for
                      ``stop=RejuvenationStop(max_n_mcmc=...)`` with all other stop fields at their
                      defaults. Mutually exclusive with ``stop``. ``None`` => use ``stop`` (or the
                      ``MAX_N_MCMC`` default). Ignored when ``n_mcmc`` is an int.
        final_resample: if ``True``, resample once at the end so the returned particles are equal-weight.
        keep_levels:  if ``True``, also return the particle cloud at *every* tempering level (see `LevelSnapshot`),
                      captured after that level's rejuvenation sweeps and before β advances.
                      Since level β targets q_{β·λ}, one run at ``lam`` then yields a whole λ-sweep for free —
                      no extra reward evaluations, no extra randomness (the results are bit-identical to a
                      ``keep_levels=False`` run with the same ``seed``).
        snapshot_device: where ``keep_levels`` snapshots are stored. Defaults to ``"cpu"`` so a long high-dimensional run
                      does not accumulate levels in VRAM alongside the generator; ``None`` keeps them on the compute device.
        seed:         seeds the single internal ``torch.Generator`` that drives every kernel (no global RNG).

    Returns: `AdaptiveTemperingSMCResult`.
    """
    x_refs = reward.x_refs

    if ess_target is None:
        ess_target = kernel.default_ess_target
    if stop is not None and max_n_mcmc is not None:
        raise ValueError("pass either `max_n_mcmc` (convenience) or a full `stop` policy, not both")
    if stop is None:
        stop = RejuvenationStop(max_n_mcmc=max_n_mcmc) if max_n_mcmc is not None else RejuvenationStop()

    gen = torch.Generator(device=x_refs.device)
    if seed is not None:
        gen.manual_seed(seed)

    return _run_smc(
        kernel, reward, lam, n_particles,
        ess_target=ess_target, n_mcmc=n_mcmc, stop=stop, final_resample=final_resample,
        generator=gen, device=x_refs.device, dtype=x_refs.dtype,
        keep_levels=keep_levels, snapshot_device=snapshot_device,
    )
