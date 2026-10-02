"""Phase 5 of notebooks/iid_iem_flux_check/ROADMAP.md: a PREDICTOR-CORRECTOR (Langevin) sampler.

Phase 3 (`flow_guided.py`) steers the model's own flow-matching ODE with `lam * grad_x r(x_hat_0)` in a
single deterministic pass. It works, but it ceilings: the fine lambda scan (jobs 958150/958630) found a
creative-but-recognizable window at `lam in [0.4, 3.54]`, and past `lam ~ 3.93` the guidance overpowers
the velocity field and locks onto a fixed off-manifold attractor (`f` runs away to 271 at `lam = 5.5` on
visually destroyed images). A single Euler pass has no mechanism to RE-EQUILIBRATE onto the model's own
marginal after each nudge, so every step's off-manifold error compounds.

This module adds that mechanism. After each ODE step lands at `t`, it runs `corrector_steps` steps of
the Unadjusted Langevin Algorithm at that FIXED noise level, targeting

    q_t(x) ~ p_t(x) * exp(lam * r(x_hat_0(x_t))),

whose drift is the model's own marginal score PLUS the reward gradient. The score term actively pulls the
particle back onto `p_t` while the reward term pushes it up `r` -- the structural consistency Phase 3
loses at high `lam`. Phase 4 (standalone Langevin at `t = 1`) is deliberately skipped as structurally
uninformative.

`corrector_steps=0` reduces this sampler to `flow_guided_sample` BITWISE (both share the one
`guided_euler_step` implementation in `flow_guided_common.py`, and the generator is consumed identically),
which is what makes a three-way sweep of {PC-guided-predictor, PC-unguided-predictor, Phase 3} an
apples-to-apples comparison inside one job on one GPU.

TIME CONVENTION: every t is diffusers-native -- `t = 1` pure noise, `t = 0` clean data -- the convention
`_types.VelocityFn` declares. The OPPOSITE polarity to `generators/flux_flowmap.py`'s own convention. See
`flow_guided.py`'s docstring for this repo's history of silent errors here.


THE MATH
--------
Interpolant: `x_t = (1-t) x_0 + t eps`, hence `v = eps - x_0` and `x_t = x_0 + t v`.

1. Denoised target (exact):      `x_hat_0 = x_t - t * v_theta(x_t, t)`
2. Velocity -> score (exact):    `s_theta(x_t, t) = -(x_t + (1-t) v_theta(x_t, t)) / t`

   Derivation: `eps_hat = E[eps | x_t] = x_t + (1-t) v_theta`, and `p(x_t | x_0) = N((1-t) x_0, t^2 I)`,
   so `grad log p_t(x_t) = -E[eps | x_t] / t`. Sanity checks that are asserted in the tests: at `t = 1`
   this collapses to `-x_t` (the score of `N(0, I)`) for ANY `v`; and for Gaussian data `N(0, sd^2 I)`
   with the optimal `v` it reduces to `-x_t / ((1-t)^2 sd^2 + t^2)`, the true marginal score.

3. **THE IDENTITY-JACOBIAN APPROXIMATION -- the one heavy assumption in this file.**
   `r` cannot be meaningfully evaluated on a noisy `x_t` (the IEM reward compares score fields of clean
   latents), so it is evaluated on the clean estimate `x_hat_0` and the gradient is transported back as

       grad_{x_t} r(x_t)  ~=  grad_{x_hat_0} r(x_hat_0),     i.e.  d(x_hat_0)/d(x_t)  treated as  I.

   This is NOT a small correction being dropped. The true Jacobian is `I - t * dv_theta/dx_t`, a full
   `d x d` operator encoding the network's own sensitivity, and it is being replaced by the identity;
   the approximation is worst exactly where `t` is large (early, high-noise steps) and where the
   transformer is most nonlinear. It is the standard universal-guidance approximation and it is what
   `flow_guided.py`'s default path already does. `exact_jacobian=True` drops it entirely at the cost of a
   backward through `v_theta` itself; both modes are available here and apply to predictor and corrector
   alike. Everything downstream of this line -- the normalization, the step size, the whole corrector --
   inherits whatever bias this approximation carries.

4. Normalized reward direction and total drift:

       g_tilde_t = grad_{x_hat_0} r / (||grad_{x_hat_0} r||_2 + eps0) * ||s_theta(x_t, t)||_2
       g_total   = s_theta(x_t, t) + lam * g_tilde_t

   The reward gradient's MAGNITUDE is discarded and replaced by the intrinsic prior score's, so `lam` is
   a dimensionless mixing weight: `lam = 1` puts the drift at 45 degrees between "stay on the manifold"
   and "ascend the reward", independent of the reward's arbitrary gradient scale.

5. Dynamic SNR step size (Song et al. 2021, see `SNR_SONG_2021`). With `eta_reference="total"` (default):

       eta_t = 2 * ( snr * ||z||_2 / (||s_theta(x_t,t) + lam * g_tilde_t||_2 + eps0) )^2

   With `eta_reference="score"` (the ablation):

       eta_t = 2 * ( snr * ||z||_2 / (||s_theta(x_t,t)||_2 + eps0) )^2

   CONSEQUENCE OF THE DEFAULT, by design: since `||g_tilde|| = ||s_theta||` exactly,
   `||g_total|| <= (1 + lam) ||s_theta||`, so the drift displacement `eta*||g_total||` and the noise
   `sqrt(2 eta)*||z||` BOTH scale as `~1/(1 + lam)`. Raising `lam` rotates the drift toward the reward
   while ANNEALING the step size -- that is the mechanism preventing numerical collapse at large `lam`,
   and it also means `lam` and `snr` are coupled: to hold the corrector's displacement fixed while
   raising `lam`, raise `snr` by roughly `(1 + lam)`. `eta_reference="score"` decouples them.

6. Corrector loop (ULA). For `j = 0 .. corrector_steps-1` at fixed `t`, with `z^(j) ~ N(0, I)`:

       x^(j+1) = x^(j) + eta_t^(j) * g_total(x^(j), t) + sqrt(2 eta_t^(j)) * z^(j)

   `eta` is recomputed every `j` (it depends on `x^(j)` through both `s_theta` and `grad r`).
   **This is UNADJUSTED Langevin: there is no Metropolis accept/reject**, so the stationary distribution
   is `q_t` only in the `eta -> 0` limit and carries an `O(eta)` discretization bias at any finite step.
   That is the accepted trade (an MH ratio would need `log p_t`, which is exactly what this repo does not
   have -- only its gradient), and `snr` is the knob that controls it. Consequently CLAUDE.md's standing
   rule applies with full force here: always subtract a `lam = 0` control, and note that this sampler has
   TWO distinct ones -- `corrector_steps=0` (pure Phase 3 base ODE) and `lam=0, corrector_steps>0`
   (base ODE plus pure Langevin on `p_t`, which still moves the cloud).

   Following Song's reference implementation, the SAME `z` supplies both `||z||` in `eta` and the
   injected noise. At `d = 65536` the relative spread of `||z||` is `1/sqrt(2d) ~ 0.3%`, so the induced
   dependence is numerically irrelevant; norms here are per-sample rather than Song's batch-mean, which
   coincides with his at the production `n_samples = 1`.

NUMERICS: `s_theta` carries a `1/t` factor, so the corrector is skipped below `corrector_t_min` -- the
schedule's final node is exactly `t = 0`, where the score is undefined. All norms and `eta` are computed
in float32 regardless of the latents' dtype: production runs `bfloat16` at `d = 65536`, where
`||z||^2 ~ 65536` retains only ~3 significant digits and the reduction itself is lossy.

COST: `n_steps * (1 + corrector_steps)` velocity evaluations, and as many reward backwards as there are guided units.
At Phase 3's production setting (`n_steps=10`, `exact_jacobian=True`, `n_samples=1`, measured 57 s per
guided unit on an L40S), `corrector_steps=2` makes a run 3x Phase 3's. VRAM is unchanged: the
`1 + corrector_steps` grad-enabled
transformer graphs per ODE step are built and freed strictly sequentially (`retain_graph=False`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from jaxtyping import Float
from torch import Generator, Tensor

from creativity_measure._types import VelocityFn
from creativity_measure.samplers.flow_guided_common import (
    _EPS,
    _reward_grad,
    _shifted_schedule,
    clip_grad_percentile,
    guided_euler_step,
    normalize_to,
    require_frozen_module,
)
from creativity_measure.tilt import Reward

# Song et al. 2021, "Score-Based Generative Modeling through Stochastic Differential Equations"
# (arXiv:2011.13456), Appendix G -- the `signal-to-noise ratio` of the predictor-corrector sampler, and
# the default of the reference `score_sde` `LangevinCorrector` implementation.
SNR_SONG_2021 = 0.16


def denoised_from_velocity(
    x_t: Float[Tensor, "B d"], v: Float[Tensor, "B d"], t: float
) -> Float[Tensor, "B d"]:
    """``x_hat_0 = x_t - t * v_theta(x_t, t)`` -- the clean-latent estimate. Exact, no approximation.

    Same algebra as ``generators/flux.py``'s ``flux_edm_denoiser`` under the ``t = sigma/(1+sigma)``
    reparametrization, and the same expression ``guided_euler_step`` computes inline.
    """
    return x_t - t * v


def velocity_to_score(
    x_t: Float[Tensor, "B d"], v: Float[Tensor, "B d"], t: float, *, eps: float = _EPS
) -> Float[Tensor, "B d"]:
    """``s_theta(x_t, t) = -(x_t + (1-t) v_theta(x_t, t)) / t`` -- the marginal score of ``p_t``.

    EXACT for the rectified-flow interpolant ``x_t = (1-t) x_0 + t eps`` (derivation in the module
    docstring), not an approximation: ``eps_hat = x_t + (1-t) v`` is the conditional mean of the noise and
    ``grad log p_t(x_t) = -eps_hat / t``.

    Computed at float32 or better (``promote_types`` with float32, so bfloat16/float16 are upcast and
    float64 is preserved): the ``1/t`` factor reaches 100x at the default ``corrector_t_min = 1e-2``,
    which bfloat16's ~8-bit mantissa cannot carry. ``t`` must be positive; ``eps`` only guards against a
    caller passing an exact zero, it does not make ``t = 0`` meaningful (the score genuinely diverges
    there, which is why the corrector has a ``corrector_t_min`` floor).

    NOTE on conditioning: ``eps_hat = x_t + (1-t) v`` is a difference of two same-order quantities, so it
    loses relative precision wherever the true ``eps_hat`` happens to be small compared to ``x_t`` -- i.e.
    wherever ``t * ||s_theta||`` is small. On real latents ``eps_hat`` stays O(sqrt(d)) at every ``t``
    (it is an estimate of unit-variance noise) so this never bites in production, but a synthetic ``p``
    can be constructed where it does; such cases need float64 to separate the formula from the rounding.
    """
    dt = torch.promote_types(x_t.dtype, torch.float32)
    return -(x_t.to(dt) + (1.0 - t) * v.to(dt)) / (t + eps)


def ula_step_size(
    ref_norm: Float[Tensor, "B 1"],
    z_norm: Float[Tensor, "B 1"],
    snr: float,
    *,
    eps: float = _EPS,
) -> Float[Tensor, "B 1"]:
    """``eta = 2 * (snr * ||z|| / (ref_norm + eps))^2`` -- Song et al. 2021's dynamic Langevin step size.

    ``ref_norm`` is ``||g_total||`` under ``eta_reference="total"`` (the default) and ``||s_theta||``
    under ``"score"``; see the module docstring for the ``1/(1+lam)`` annealing that the former implies.

    Because ``eta`` is ADAPTIVE it overshoots the target variance by ``(1 + snr^2)(1 + 2/d)^2``: the first
    factor is ULA's own ``O(eta)`` bias (set by ``snr``, independent of ``d``), the second is that ``eta``
    depends on ``||z||^2`` and ``1/||x||^2``, whose fluctuations do not average out -- the noise energy
    goes as ``E||z||^4 = d(d+2)``, and stationarity pins ``E[1/||x||^2]`` rather than ``E[||x||^2]``, so
    Jensen adds a second gap. **Both ``(1 + 2/d)`` factors go to 1 as ``d`` grows**, so this is a
    small-``d`` concern only -- at the production ``d = 65536`` it is a ``6e-5`` effect and nothing to
    correct for. Derivation and the ablation confirming it: ROADMAP.md Phase 5.
    """
    return 2.0 * (snr * z_norm / (ref_norm + eps)) ** 2


@dataclass
class CorrectorSnapshot:
    """Per-ODE-step record of the Langevin corrector. One entry per corrector sub-step ``j`` in each list.

    Diagnostics are not optional (CLAUDE.md): in high dimensions they are the only way to tell whether a
    run worked. The pair to read first is ``rel_displacement`` (is the corrector moving the particle at
    all, or has the ``1/(1+lam)`` annealing shut it down?) against ``f_hat0`` (is it moving uphill?).

    Attributes:
        t:    the noise level this corrector ran at (the ODE step's arrival node).
        ran:  False when the corrector was skipped (``corrector_steps=0``, or ``t`` outside
              ``[corrector_t_min, corrector_t_max]``); all lists are then empty.
        eta:              the step size actually used, mean over batch.
        score_norm:       ``||s_theta||``, the intrinsic prior-score scale the reward is normalized to.
        grad_norm:        ``||grad r||`` BEFORE normalization (nan when ``lam_corrector == 0``, where the
                          reward gradient is never computed).
        drift_norm:       ``||g_total||``, i.e. what actually drives the step.
        noise_frac:       ``sqrt(2 eta)||z|| / (eta ||g_total||)``, the noise-to-drift displacement ratio.
                          Under ``eta_reference="total"`` this is EXACTLY ``1/snr`` by construction,
                          independent of everything else -- a free wiring assert (and ``1/0.16 = 6.25``,
                          i.e. ULA at this snr is noise-dominated by ~6x, as a sampler should be).
        rel_displacement: ``||x^(j+1) - x^(j)|| / ||x^(j)||`` -- how far one corrector step actually moves.
        f_hat0:           ``r(x_hat_0)`` evaluated at ``x^(j)`` (nan when ``lam_corrector == 0``). Note
                          CLAUDE.md: ``f`` read at intermediate ``t`` is inflated; only ``t = 0`` is
                          artifact-free, so read this as a within-run trend, never as a level.
        oom_fallback:     whether the gamma-chunked OOM fallback fired on this sub-step.
    """

    t: float
    ran: bool = False
    eta: list[float] = field(default_factory=list)
    score_norm: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    drift_norm: list[float] = field(default_factory=list)
    noise_frac: list[float] = field(default_factory=list)
    rel_displacement: list[float] = field(default_factory=list)
    f_hat0: list[float] = field(default_factory=list)
    oom_fallback: list[bool] = field(default_factory=list)


@dataclass
class FlowGuidedPCResult:
    """Output of :func:`flow_guided_pc_sample`.

    The predictor fields are field-for-field identical to ``flow_guided.FlowGuidedResult``'s, so Phase 3
    plotting and result-dict code works against either without translation.

    Attributes:
        X:           terminal latents at the schedule's final node (``t = 0``, the artifact-free case).
        t_history:   the schedule's t at the START of each ODE step (length ``n_steps``).
        guided_history:       whether the PREDICTOR's guidance branch was taken (needs
                              ``predictor_guided``, ``lam != 0`` and ``t`` in the window).
        grad_norm_history:    predictor's mean-over-batch ``||grad r||`` before scaling (nan if unguided).
        v_norm_history:       predictor's mean-over-batch ``||v_theta||``.
        applied_norm_history: predictor's mean-over-batch ``||lam * g_t||`` (0 if unguided).
        f_hat0_history:       predictor's mean-over-batch ``r(x_hat_0)`` (nan if unguided).
        oom_fallback_history / static_fallback_history: predictor's fallback flags.
        x_norm_history:       ``||x||/sqrt(d)`` after each full predictor+corrector step. The cheapest
                              off-manifold alarm there is: Phase 3's high-lam failure inflates the latent
                              norm, so compare this against the ``lam = 0`` control's trace before
                              decoding anything.
        corrector_history:    one :class:`CorrectorSnapshot` per ODE step (always ``n_steps`` long; the
                              snapshot carries ``ran=False`` where the corrector was skipped).
        n_velocity_evals:     total ``velocity_fn`` calls -- ``n_steps * (1 + M_effective)``.
        n_reward_grads:       total reward backwards. Together these are the compute accounting Phase 6
                              (quality vs cost) needs, and the cheapest check that the sampler did what
                              its kwargs said.
    """

    X: Float[Tensor, "B d"]
    t_history: list[float] = field(default_factory=list)
    guided_history: list[bool] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    v_norm_history: list[float] = field(default_factory=list)
    applied_norm_history: list[float] = field(default_factory=list)
    f_hat0_history: list[float] = field(default_factory=list)
    oom_fallback_history: list[bool] = field(default_factory=list)
    static_fallback_history: list[bool] = field(default_factory=list)
    x_norm_history: list[float] = field(default_factory=list)
    corrector_history: list[CorrectorSnapshot] = field(default_factory=list)
    n_velocity_evals: int = 0
    n_reward_grads: int = 0


def _corrector_drift(
    x: Float[Tensor, "B d"],
    t: float,
    *,
    reward: Reward,
    lam_corrector: float,
    velocity_fn: VelocityFn,
    exact_jacobian: bool,
    grad_clip_percentile: float | None,
    g_chunk: int | None,
) -> tuple[Float[Tensor, "B d"], Float[Tensor, "B 1"], float, float, bool]:
    """``(g_total, s_norm, grad_norm_mean, f_hat0_mean, fell_back)`` at ``(x, t)``.

    ``g_total = s_theta + lam_corrector * g_tilde`` per the module docstring's step 4. ``s_theta`` is
    ALWAYS detached -- it is drift, never part of the reward's autograd graph; the only graph built here
    is the one from the reward back to ``x_hat_0`` (approximate) or ``x_t`` (exact), and it is freed by
    ``_reward_grad``'s ``retain_graph=False`` before this function returns.

    ``lam_corrector == 0`` short-circuits the reward entirely (drift is the bare score, one velocity
    evaluation, no backward), which is what makes the pure-Langevin control cheap.
    """
    if lam_corrector == 0.0:
        with torch.no_grad():
            v = velocity_fn(x, t)
            s = velocity_to_score(x, v, t)
        return s, s.norm(dim=1, keepdim=True), float("nan"), float("nan"), False

    if exact_jacobian:
        x_req = x.detach().requires_grad_(True)
        v = velocity_fn(x_req, t)                         # grad ENABLED: graph starts at x_req
        x_hat0 = denoised_from_velocity(x_req, v, t)
        g, fell_back = _reward_grad(reward, x_hat0, x_req, create_graph=False, g_chunk=g_chunk)
        s = velocity_to_score(x_req.detach(), v.detach(), t)
        x_hat0 = x_hat0.detach()
    else:
        with torch.no_grad():
            v = velocity_fn(x, t)
            s = velocity_to_score(x, v, t)
        x_hat0 = denoised_from_velocity(x, v, t).detach().requires_grad_(True)
        g, fell_back = _reward_grad(reward, x_hat0, x_hat0, create_graph=False, g_chunk=g_chunk)
        x_hat0 = x_hat0.detach()

    if grad_clip_percentile is not None:
        g = clip_grad_percentile(g, grad_clip_percentile)

    # Accumulate in s's dtype -- float32 or better, pinned by velocity_to_score, never the latents'
    # bfloat16: at d = 65536 a bf16 norm keeps ~3 significant digits and eta depends on its SQUARE.
    g = g.to(s.dtype)
    grad_norm = g.norm(dim=1, keepdim=True)
    s_norm = s.norm(dim=1, keepdim=True)
    g_tilde = normalize_to(g, s_norm, source_norm=grad_norm)
    g_total = s + lam_corrector * g_tilde

    with torch.no_grad():
        f_hat0 = float(reward(x_hat0).mean())
    return g_total, s_norm, float(grad_norm.mean()), f_hat0, fell_back


def _ula_step(
    x: Float[Tensor, "B d"],
    t: float,
    *,
    reward: Reward,
    lam_corrector: float,
    velocity_fn: VelocityFn,
    snr: float,
    eta_reference: Literal["total", "score"],
    exact_jacobian: bool,
    grad_clip_percentile: float | None,
    g_chunk: int | None,
    generator: Generator,
    snapshot: CorrectorSnapshot,
) -> Float[Tensor, "B d"]:
    """One Unadjusted Langevin step at fixed ``t``, appending its diagnostics to ``snapshot``.

    ``x^(j+1) = x^(j) + eta * g_total + sqrt(2 eta) z``, with ``eta`` from :func:`ula_step_size`. The
    returned tensor is detached -- memory isolation, as in ``guided_euler_step``; this matters more here
    because a corrector runs ``corrector_steps`` of these back to back inside one ODE step.
    """
    g_total, s_norm, grad_norm, f_hat0, fell_back = _corrector_drift(
        x, t, reward=reward, lam_corrector=lam_corrector, velocity_fn=velocity_fn,
        exact_jacobian=exact_jacobian, grad_clip_percentile=grad_clip_percentile, g_chunk=g_chunk,
    )

    # Noise in the drift's accumulation dtype (float32 or better, never the latents' bfloat16): at
    # d = 65536 a bf16 ||z||^2 ~ 65536 keeps ~3 significant digits, and eta depends on its SQUARE.
    z = torch.randn(x.shape, generator=generator, device=x.device, dtype=g_total.dtype)
    z_norm = z.norm(dim=1, keepdim=True)
    drift_norm = g_total.norm(dim=1, keepdim=True)
    ref_norm = drift_norm if eta_reference == "total" else s_norm
    eta = ula_step_size(ref_norm, z_norm, snr)

    x_next = (x + eta * g_total + (2.0 * eta).sqrt() * z).detach()

    snapshot.eta.append(float(eta.mean()))
    snapshot.score_norm.append(float(s_norm.mean()))
    snapshot.grad_norm.append(grad_norm)
    snapshot.drift_norm.append(float(drift_norm.mean()))
    snapshot.noise_frac.append(
        float(((2.0 * eta).sqrt() * z_norm / (eta * drift_norm + _EPS)).mean())
    )
    snapshot.rel_displacement.append(
        float(((x_next - x).float().norm(dim=1) / (x.float().norm(dim=1) + _EPS)).mean())
    )
    snapshot.f_hat0.append(f_hat0)
    snapshot.oom_fallback.append(fell_back)
    return x_next


def flow_guided_pc_sample(
    reward: Reward,
    lam: float,
    n_samples: int,
    *,
    velocity_fn: VelocityFn,
    n_steps: int = 28,
    shift: float = 3.0,
    t_start: float = 1.0,
    t_end: float = 0.0,
    predictor_guided: bool = True,
    corrector_steps: int = 2,
    lam_corrector: float | None = None,
    snr: float = SNR_SONG_2021,
    eta_reference: Literal["total", "score"] = "total",
    corrector_t_max: float = 1.0,
    corrector_t_min: float = 1e-2,
    exact_jacobian: bool = False,
    grad_scaling: Literal["velocity", "static"] = "velocity",
    static_scale: float = 1.0,
    grad_clip_percentile: float | None = None,
    min_v_norm: float = 1e-4,
    g_chunk: int | None = None,
    z0: Float[Tensor, "B d"] | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> FlowGuidedPCResult:
    """Sample ``q_lambda`` by alternating guided Euler ODE steps with Langevin correction at fixed ``t``.

    Backend-agnostic: needs only a ``velocity_fn`` conforming to the diffusers-native ``VelocityFn``
    convention. The full derivation -- score reparametrization, the identity-Jacobian approximation, the
    normalization, both ``eta`` formulas, and the ULA bias -- is in the module docstring; read it before
    changing any of the arithmetic here.

    Args:
        reward:      the frozen tilt ``f`` (invariant 1); ``reward.x_refs`` pins device/dtype/``d``. The
                     reference bank is built once before the time loop, not per step.
        lam:         guidance strength for the PREDICTOR, and the default for the corrector. Dimensionless:
                     the reward gradient is renormalized to ``||v_theta||`` (predictor) or ``||s_theta||``
                     (corrector), so ``lam = 1`` means "equal weight". Phase 3's creative window on FLUX
                     was ``lam in [0.4, 3.54]``.
        n_samples:   number of independent trajectories (a batch, not SMC particles -- there is no
                     resampling here, so there is no ESS and no lineage to collapse).
        velocity_fn: ``(x_t, t) -> v_theta``, diffusers-native t. For ``exact_jacobian=True`` it must
                     carry a frozen ``.module`` (``_types.GuidableVelocityFn``), checked up front.
        n_steps, shift: Euler step count and resolution-shift parameter (FLUX.1-dev's own inference
                     defaults; override for another backend's recommended schedule).
        t_start, t_end: the PREDICTOR's guidance window; guidance is active at a step iff
                     ``predictor_guided and lam != 0 and t_end <= t <= t_start``.
        predictor_guided: **the headline ablation.** True (default) makes the predictor Phase 3's guided
                     step, so this sampler is a strict superset of Phase 3 and the corrector's effect is
                     measured as a pure addition. False makes the predictor the model's plain ODE, so
                     100% of the tilt comes from the Langevin corrector -- the cleaner isolation of
                     "can Langevin alone beat Phase 3's ceiling", at the cost of no longer nesting
                     Phase 3. Both are intended to be run and compared.
        corrector_steps: the number of ULA steps after each ODE step (NOT to be confused with this repo's
                     ``m = lam/lam_s`` tilt strength, nor with Algorithm 3's particle count ``M``). ``0`` disables the corrector
                     and reduces this function to ``flow_guided_sample`` BITWISE. Cost is linear in
                     ``1 + corrector_steps``.
        lam_corrector: the corrector's ``lam``; ``None`` (default) means "same as ``lam``". Setting it to
                     ``0`` with ``lam != 0`` gives a guided predictor with a pure-``p_t`` Langevin
                     corrector (reward-free re-equilibration), which is a meaningful ablation in its own
                     right and costs no reward backwards.
        snr:         Langevin signal-to-noise ratio; see :data:`SNR_SONG_2021`. Larger means bigger steps
                     and more discretization bias; the noise-to-drift displacement ratio is exactly
                     ``1/snr``.
        eta_reference: ``"total"`` (default, as specified) divides by ``||s_theta + lam*g_tilde||``, so
                     raising ``lam`` anneals the step as ``~1/(1+lam)``. ``"score"`` divides by
                     ``||s_theta||`` alone, making ``eta`` independent of ``lam``. See the module
                     docstring for both formulas and why the coupling is deliberate.
        corrector_t_max, corrector_t_min: the corrector runs at an ODE step's ARRIVAL node ``t`` iff
                     ``corrector_t_min <= t <= corrector_t_max``. ``corrector_t_min`` must be positive:
                     ``s_theta`` carries a ``1/t`` factor and the schedule's final node is exactly 0, so
                     the corrector is an interior-node operation by construction.
        exact_jacobian: False (default) treats ``d(x_hat_0)/d(x_t)`` as the identity in BOTH the
                     predictor and the corrector; True backpropagates through ``v_theta`` in both.
        grad_scaling, static_scale, grad_clip_percentile, min_v_norm: predictor-side knobs, forwarded
                     unchanged to ``guided_euler_step`` -- identical semantics to ``flow_guided_sample``.
                     ``grad_clip_percentile`` additionally applies to the corrector's gradient.
        g_chunk:     gamma-chunk size for the OOM fallback (default 1, i.e. one gamma at a time).
        z0:          initial noise at ``t = t_start``; if None, drawn ``N(0, I)`` from the run's own
                     generator (flow matching's noise endpoint IS standard normal).
        seed:        seeds the single generator this run uses (0 if None); never touches global RNG
                     (invariant 3). ``z0`` is drawn first, then all corrector noise in order -- so at
                     ``corrector_steps=0`` the generator is consumed exactly as ``flow_guided_sample``
                     consumes it, which is what makes the bitwise reduction hold.

    Returns:
        FlowGuidedPCResult -- read ``corrector_history[*].rel_displacement`` and ``x_norm_history``
        before deciding whether a run is worth decoding.
    """
    if corrector_steps < 0:
        raise ValueError(f"corrector_steps must be >= 0, got {corrector_steps}")
    if snr <= 0.0:
        raise ValueError(f"snr must be > 0, got {snr}")
    if eta_reference not in ("total", "score"):
        raise ValueError(f"eta_reference must be 'total' or 'score', got {eta_reference!r}")
    if corrector_steps > 0 and corrector_t_min <= 0.0:
        raise ValueError(
            f"corrector_t_min must be > 0, got {corrector_t_min}: the marginal score "
            "s_theta = -(x_t + (1-t) v)/t diverges as t -> 0, and the schedule's final node is exactly 0. "
            "The Langevin corrector is an interior-node operation by construction."
        )
    if exact_jacobian:
        require_frozen_module(velocity_fn, flag="exact_jacobian=True")

    lam_c = lam if lam_corrector is None else lam_corrector

    x_refs = reward.x_refs
    device, dtype, d = x_refs.device, x_refs.dtype, x_refs.shape[1]
    gen = torch.Generator(device=device).manual_seed(0 if seed is None else seed)
    x = torch.randn(n_samples, d, generator=gen, device=device, dtype=dtype) if z0 is None else z0

    # Force the reference bank build ONCE, before the time loop, so the O(R) cost (CLAUDE.md: ~19 min at
    # R=64 on an L40S) is paid outside every per-step timing -- exactly as flow_guided_sample does.
    with torch.no_grad():
        reward(x_refs[:1])

    schedule = _shifted_schedule(n_steps, shift, device=device, dtype=torch.float32)
    res = FlowGuidedPCResult(X=x)
    sqrt_d = float(d) ** 0.5

    for i in range(n_steps):
        t_from, t_to = float(schedule[i]), float(schedule[i + 1])
        guided = predictor_guided and lam != 0.0 and t_end <= t_from <= t_start

        # ---- Predictor: Phase 3's step verbatim (the shared implementation) --------
        x, rec = guided_euler_step(
            x, t_from=t_from, t_to=t_to, reward=reward, lam=lam, velocity_fn=velocity_fn,
            guided=guided, exact_jacobian=exact_jacobian, grad_scaling=grad_scaling,
            static_scale=static_scale, grad_clip_percentile=grad_clip_percentile,
            min_v_norm=min_v_norm, g_chunk=g_chunk,
        )
        res.n_velocity_evals += 1
        res.n_reward_grads += int(rec.guided)

        res.t_history.append(t_from)
        res.guided_history.append(rec.guided)
        res.grad_norm_history.append(rec.grad_norm)
        res.v_norm_history.append(rec.v_norm)
        res.applied_norm_history.append(rec.applied_norm)
        res.f_hat0_history.append(rec.f_hat0)
        res.oom_fallback_history.append(rec.oom_fallback)
        res.static_fallback_history.append(rec.static_fallback)

        # ---- Corrector: corrector_steps of ULA at the arrival node t_to --------------------------------------
        snapshot = CorrectorSnapshot(t=t_to)
        if corrector_steps > 0 and corrector_t_min <= t_to <= corrector_t_max:
            snapshot.ran = True
            for _ in range(corrector_steps):
                x = _ula_step(
                    x, t_to, reward=reward, lam_corrector=lam_c, velocity_fn=velocity_fn, snr=snr,
                    eta_reference=eta_reference, exact_jacobian=exact_jacobian,
                    grad_clip_percentile=grad_clip_percentile, g_chunk=g_chunk, generator=gen,
                    snapshot=snapshot,
                )
                res.n_velocity_evals += 1
                res.n_reward_grads += int(lam_c != 0.0)
        res.corrector_history.append(snapshot)

        res.x_norm_history.append(float(x.float().norm(dim=1).mean()) / sqrt_d)

        if verbose:
            tail = (
                f" corr(eta={snapshot.eta[-1]:.3g} disp={snapshot.rel_displacement[-1]:.3g} "
                f"f={snapshot.f_hat0[-1]:.4g})"
                if snapshot.ran else " corr(skipped)"
            )
            print(f"[step {i}] t={t_from:.4f}->{t_to:.4f} "
                  f"pred(guided={rec.guided} f={rec.f_hat0:.4g} v={rec.v_norm:.4g})"
                  f"{tail} |x|/sqrt(d)={res.x_norm_history[-1]:.4g}", flush=True)

    res.X = x
    return res
