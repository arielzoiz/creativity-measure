"""SMC weighted flow maps for  q_lambda(x) ∝ p(x) · exp(lambda · f(x))  -- Algorithm 3.

A **single forward pass** from noise to data: ``n_steps`` transitions over ``t ∈ [0, 1]``, no tempering
ladder, no rejuvenation, no acceptance rate to collapse. Where ``adaptive_tempering_smc.py`` pays a full
ladder of pCN sweeps per temperature level (and never finishes at FLUX scale), this pays one lookahead
per step. The lookahead that estimates the value function is one network call per candidate, so the
network is nearly free and the run is dominated entirely by reward evaluations.

The loop walks ``t: 0 -> 1`` (noise -> data). At each step:

1. one **base transition** ``x_t -> x_{t_next}`` -- a `TransitionStep`, `ddpm_step` inside
   ``stoch_window`` and `flow_map_step` outside it. Together with ``ts``, ``schedule`` and
   ``stoch_window`` this is what defines the base process ``p``;
2. a **weighted lookahead**: renoise ``x_{t_next}`` to ``t' < t_next`` with ``K`` noise draws, then map
   each renoised state to a clean candidate ``z^k = flow_map.map(x_{t'}^k, t', 1)`` in one network call.
   The ``K`` candidates are ``K`` plausible clean images that ``x_{t_next}`` could become -- we cannot
   evaluate ``f`` on ``x_{t_next}`` itself, which is noisy, and ``f`` is defined on clean images;
3. the potential ``V_t = log( (1/K) sum_k exp(v_k) )``, the Monte-Carlo estimate of the value function
   ``V_t(x_t) = log E_{z~p_{1|t}(·|x_t)}[exp(r(z))]``;
4. ``U += V_next - V_prev`` and ESS-triggered resampling.

**Where lambda enters, and why only there.** The potential telescopes: with ``V_0 = 0`` the particle
system targets ``p(x_1)·exp(V_N(x_1) - V_0)``. At ``t = 1`` the lookahead posterior collapses to a point
mass, so ``V_N = lambda·f(x_1)`` exactly -- and that is what makes the target ``q_lambda``. Therefore
``lambda`` multiplies ``f`` only, never ``L`` or ``S`` (those are lookahead-proposal corrections, not
rewards); the terminal update is mandatory and always raw; and every *intermediate* ``V_t`` is a
**twist** -- it changes variance and where resampling spends effort, not the target. That is what
licenses the normalization option in ``use_full_normalized_v``.

Notes
-----
* **Model-agnostic.** The sampler never mentions FLUX: swapping in another distillation means supplying
  a different `FlowMap` and ``score_fn`` in the notebook, nothing more. Swapping to a VP/EDM-scheduled
  model additionally means supplying a different `Schedule` -- a real extension point, since every
  coefficient below is derived from ``alpha``/``sigma`` rather than hard-coded.
* **The reward is frozen**, exactly as for the other samplers: ``x_refs``, ``weights`` and the
  ``Distance``'s Brownian seed are fixed before sampling, so ``f`` is a deterministic function of ``x``.
  ``reward.x_refs`` also pins ``d``, device and dtype for the whole run, so no shape argument is needed.
* **Determinism.** Three ``torch.Generator``s seeded from ``seed``, ``seed + 1`` and ``seed + 2``: base-step
  noise, lookahead noise, resampling. ``adaptive_tempering_smc.py`` threads a *single* generator, but
  here the streams must be independent, or the number of lookahead draws -- or of resampling events --
  would shift the base trajectory and the ``lambda = 0`` reproduction check could not pass. (The plan
  called for two streams; resampling gets its own for the same reason, and because whether a step
  resamples at ``lambda = 0`` otherwise hinges on float slop in ``ESS == M``.) No global RNG is touched.
* **Conventions.** Plan time: ``t = 0`` is noise, ``t = 1`` is data, ``x_t = alpha_t·z + sigma_t·eps``.
  FLUX/``diffusers`` runs the other way (1 = noise); that flip lives in the `FlowMap` implementation
  (``generators/flux_flowmap.py``), never here.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from jaxtyping import Float, Int
from torch import Tensor

from creativity_measure._types import FlowMap, Schedule, ScoreFn, TransitionStep
from creativity_measure.samplers.smc_common import _ess_from_logw, _systematic_resample
from creativity_measure.tilt import Reward

__all__ = [
    "flowmap_smc_sample",
    "FlowMapSMCResult",
    "StepSnapshot",
    "StepCallback",
    "BaseSchedule",
    "LinearSchedule",
    "SCHEDULES",
    "ddpm_step",
    "flow_map_step",
]

# Numerical guards. These fire at the schedule ends, where the uniform grid actually lives (first
# step t = 0.0625, last t = 0.9375), so they are load-bearing rather than defensive decoration.
T_PRIME_MAX: float = 0.9999      # t' is clamped below this; g(t) -> 0 as t -> 1
MIN_RENOISE_VAR: float = 1e-8    # renoise variance floor (§3C)
MIN_ZSCORE_STD: float = 1e-8     # Z-score sigma floor: it collapses to 0 when the K candidates coincide
SCORE_CLIP: float = 1e6          # symmetric clip on score values and on the alpha ratio
_TINY: float = 1e-12


# ---------------------------------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------------------------------

def _g(schedule: Schedule, t: float) -> float:
    """``g(t) = sigma(t)^2 / alpha(t)^2`` -- the (inverse) SNR the renoise level is chosen in.

    Derived from the protocol's ``alpha``/``sigma`` rather than being a `Schedule` member, so a
    notebook-supplied schedule only has to provide the two functions and its own ``t_of_snr``. Keeping
    ``g`` out of the protocol is also what makes a mismatched inverse impossible to introduce by
    accident: there is exactly one definition of ``g`` in the package, and `BaseSchedule` inverts it.
    """
    a, s = schedule.alpha(t), schedule.sigma(t)
    return (s * s) / max(a * a, _TINY)


class BaseSchedule:
    """Interpolant ``x_t = alpha(t)*x_1 + sigma(t)*eps`` with a **numeric** SNR inverse.

    A new schedule only has to supply ``alpha`` and ``sigma``: ``g(t) = sigma(t)^2 / alpha(t)^2`` is
    strictly decreasing whenever ``alpha`` increases and ``sigma`` decreases, so bisection on ``(0, 1)``
    inverts it for *any* such schedule. That is what makes a differently-noised model droppable in.
    Subclasses with a closed form (see `LinearSchedule`) override ``t_of_snr``.
    """

    name: str = "base"

    def alpha(self, t: float) -> float:
        raise NotImplementedError

    def sigma(self, t: float) -> float:
        raise NotImplementedError

    def g(self, t: float) -> float:
        """``sigma(t)^2 / alpha(t)^2`` -- see the module-level `_g`, which this delegates to."""
        return _g(self, t)

    def t_of_snr(self, y: float, *, n_iter: int = 80) -> float:
        """``g^-1(y)`` by bisection on ``(0, 1)``; ``g`` is strictly decreasing in ``t``."""
        lo, hi = _TINY, 1.0 - _TINY
        if y >= _g(self, lo):
            return lo
        if y <= _g(self, hi):
            return hi
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            # g decreasing: g(mid) > y means mid is still too noisy, so move the *low* end up.
            if _g(self, mid) > y:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


class LinearSchedule(BaseSchedule):
    """The rectified-flow / linear interpolant: ``alpha_t = t``, ``sigma_t = 1 - t``.

    ``g(t) = ((1-t)/t)^2``, so the SNR inverse is closed form: ``t_of_snr(y) = 1/(1 + sqrt(y))``.
    This is what FLUX.1-dev and its flow-map distillations use.
    """

    name: str = "linear"

    def alpha(self, t: float) -> float:
        return float(t)

    def sigma(self, t: float) -> float:
        return 1.0 - float(t)

    def t_of_snr(self, y: float, *, n_iter: int = 80) -> float:
        return 1.0 / (1.0 + math.sqrt(max(y, 0.0)))


# The one place a fixed menu genuinely helps: a reference file records the base process by name, and
# this maps the string back on load. It stays out of the sampler's API.
SCHEDULES: dict[str, Schedule] = {"linear": LinearSchedule()}


# ---------------------------------------------------------------------------------------------------
# Base transitions (`TransitionStep`)
# ---------------------------------------------------------------------------------------------------

def _rho(schedule: Schedule, t: float, t_next: float) -> tuple[float, float]:
    """``(rho, 1 - rho^2)`` for the DDPM transition ``t -> t_next``, in float64.

    ``rho = alpha_t·sigma_{t'} / (alpha_{t'}·sigma_t)`` is the correlation between the noise components
    of ``x_t`` and ``x_{t'}`` given ``x_1``, read off the *noising* kernel ``q_{t|t'}``.

    ``1 - rho^2`` is computed as a difference of squares rather than literally: as ``h -> 0`` we have
    ``rho -> 1``, and ``1 - rho*rho`` then loses every significant digit.
    In the difference-of-squares form the numerator is exactly ``h`` under `LinearSchedule`, so no cancellation occurs at all.
    """
    a_t, s_t = schedule.alpha(t), schedule.sigma(t)
    a_next, s_next = schedule.alpha(t_next), schedule.sigma(t_next)
    den = a_next * s_t
    if abs(den) < _TINY:                      # only reachable at sigma_t = 0, i.e. t = 1, never a source
        raise ValueError(f"ddpm_step has a pole at sigma_t = 0 (t = {t}); t = 1 is never a step source")
    rho = (a_t * s_next) / den
    one_minus_rho2 = max((den * den - (a_t * s_next) ** 2) / (den * den), 0.0)
    return rho, one_minus_rho2


def ddpm_step(
    x: Float[Tensor, "B d"],
    t: float,
    t_next: float,
    *,
    flow_map: FlowMap,
    schedule: Schedule,
    generator: torch.Generator,
) -> Float[Tensor, "B d"]:
    """One stochastic base transition ``x_t -> x_{t_next}``, ``t_next > t`` (toward data).

    Obtained by Gaussian-conditioning the joint ``(X_t, X_{t'}) | X_1 = z`` implied by the noising
    kernel ``q_{t|t'}``, with ``z_hat = flow_map.map(x, t, 1)`` (one network call). Writing
    ``rho = alpha_t·sigma_{t'} / (alpha_{t'}·sigma_t)``::

        x_{t'} = rho·(sigma_{t'}/sigma_t)·x_t  +  alpha_{t'}(1-rho^2)·z_hat  +  sigma_{t'}·sqrt(1-rho^2)·eps

    identical to the ``c_x``/``c_z``/``sigma_{t'|t}`` form but factored through ``rho`` (see `_rho`).

    **No special cases.** At ``t = 0``: ``rho = 0`` -> ``x_{t'} = alpha_{t'}·z_hat + sigma_{t'}·eps``, a pure draw.
    At ``t' = 1``: ``sigma_{t'} = 0`` -> ``rho = 0`` -> ``x_1 = z_hat`` deterministically, the correct terminal denoise,
    and exactly where the mandatory ``V_N = lambda·f(x_1)`` is evaluated.
    The only pole is ``sigma_t = 0`` at ``t = 1``, which is never a *source*.

    ``z_hat`` is the ODE endpoint, i.e. a *sample*, standing in for a draw from ``p(x_1|x_t)``; the step
    is exact when it is one, and approximate for a deterministic map. This is why the head of the
    trajectory is left outside ``stoch_window`` by default -- there are no resampled clones to separate
    there, so the approximation buys nothing.

    This step is the SMC contribution: guidance methods advance on a deterministic Euler step because a
    single trajectory has no clone problem. Resampling creates one, and this solves it.
    """
    rho, one_minus_rho2 = _rho(schedule, t, t_next)
    s_t, a_next, s_next = schedule.sigma(t), schedule.alpha(t_next), schedule.sigma(t_next)

    c_x = rho * (s_next / s_t)
    c_z = a_next * one_minus_rho2
    sigma_cond = s_next * math.sqrt(one_minus_rho2)

    z_hat = flow_map.map(x, t, 1.0)
    out = c_x * x + c_z * z_hat
    if sigma_cond > 0.0:
        eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        out = out + sigma_cond * eps
    return out


def flow_map_step(
    x: Float[Tensor, "B d"],
    t: float,
    t_next: float,
    *,
    flow_map: FlowMap,
    schedule: Schedule,
    generator: torch.Generator,
) -> Float[Tensor, "B d"]:
    """One deterministic base transition: the flow map's own jump ``x_{t'} = map(x, t, t')``.

    One network call, no noise, ``generator`` unused (the `TransitionStep` signature is shared).
    """
    del schedule, generator
    return flow_map.map(x, t, t_next)


# ---------------------------------------------------------------------------------------------------
# Private helpers -- all ``(t, t_next)``- and `Schedule`-based, individually tested
# ---------------------------------------------------------------------------------------------------

def _uniform_ts(n_steps: int) -> list[float]:
    """``linspace(0, 1, n_steps + 1)``: ``h = 1/N``, ``t_n = n·h``.

    Deliberately NOT FLUX's mu-shifted sampling schedule: sigma is derived from ``t`` at each iteration
    rather than being scheduled independently, so there is one quantity and one place it is computed.
    """
    return [n / n_steps for n in range(n_steps + 1)]


def _t_prime(t: float, eta: float, schedule: Schedule) -> float:
    """The renoise level ``t' < t`` with ``g(t') = eta·g(t)``, clamped to ``T_PRIME_MAX``.

    Under `LinearSchedule` this is the spec's ``t' = 1/(1 + sqrt(eta·((1-t)/t)^2))``. ``eta > 1`` means
    more noise, hence ``t' < t``; larger ``eta`` pushes the lookahead further back and makes the ``K``
    clean candidates more diverse.
    """
    t_p = schedule.t_of_snr(eta * _g(schedule, t))
    return min(t_p, T_PRIME_MAX)


def _renoise(
    x: Float[Tensor, "B d"],
    t: float,
    t_prime: float,
    schedule: Schedule,
    eps: Float[Tensor, "B d"],
) -> Float[Tensor, "B d"]:
    """``x_{t'} = (alpha_{t'}/alpha_t)·x_t + sqrt(sigma_{t'}^2 - (alpha_{t'}/alpha_t)^2·sigma_t^2)·eps``.

    The noising kernel ``q_{t'|t}`` for ``t' < t``, so the variance is positive; it is still floored at
    ``MIN_RENOISE_VAR`` and the ratio symmetrically clipped, because the grid lives at the schedule ends.
    """
    ratio = schedule.alpha(t_prime) / max(schedule.alpha(t), _TINY)
    ratio = max(-SCORE_CLIP, min(SCORE_CLIP, ratio))
    var = schedule.sigma(t_prime) ** 2 - (ratio * schedule.sigma(t)) ** 2
    return ratio * x + math.sqrt(max(var, MIN_RENOISE_VAR)) * eps


def _score_at(
    x: Float[Tensor, "B d"],
    t: float,
    score_fn: ScoreFn,
    schedule: Schedule,
) -> Float[Tensor, "B d"]:
    """``s_t(x) = grad_x log p_t(x)``, routed through the repo's gamma-convention `ScoreFn`.

    The repo's score is ``score_fn(y, gamma) = grad_y log p_Y(y, gamma)`` for ``y = gamma·z + sqrt(gamma)·eps``.
    Setting ``c = alpha_t / sigma_t^2`` and ``gamma_t = (alpha_t/sigma_t)^2`` gives
    ``c·x_t = gamma_t·z + sqrt(gamma_t)·eps`` exactly, so one scalar factor does both jobs::

        s_t(x) = c · score_fn(c·x, gamma_t)

    This is *algebraically* the Tweedie form ``(alpha_t·E[x_1|x_t] - x)/sigma_t^2``, but routing it
    through ``score_fn`` means the score model that defines ``f``'s geometry and the one in ``S_k`` are
    the same object on the same code path -- one place to be wrong instead of two.
    """
    a, s = schedule.alpha(t), schedule.sigma(t)
    c = a / max(s * s, _TINY)
    gamma = (a / max(s, _TINY)) ** 2
    gamma_t = torch.as_tensor(gamma, device=x.device, dtype=x.dtype)
    out = c * score_fn(c * x, gamma_t)
    return torch.nan_to_num(out, nan=0.0, posinf=SCORE_CLIP, neginf=-SCORE_CLIP).clamp(
        -SCORE_CLIP, SCORE_CLIP
    )


def _log_likelihood(
    x_t: Float[Tensor, "B d"],
    z: Float[Tensor, "B d"],
    t: float,
    schedule: Schedule,
) -> Float[Tensor, "B"]:
    """``L = -||x_t - alpha_t·z||^2 / (2·sigma_t^2)`` -- ``log p(x_t | x_1 = z)`` up to a constant."""
    a, s = schedule.alpha(t), schedule.sigma(t)
    return -(x_t - a * z).pow(2).sum(dim=-1) / (2.0 * max(s * s, _TINY))


def _score_correction(
    s_at_x: Float[Tensor, "B d"],
    s_at_x_prime: Float[Tensor, "B d"],
    x: Float[Tensor, "B d"],
    x_prime: Float[Tensor, "B d"],
) -> Float[Tensor, "B"]:
    """``S = 1/2·(s_{t'}(x_t) + s_{t'}(x_{t'}))·(x_{t'} - x_t)``.

    Trapezoid rule for the line integral ``(x_{t'} - x_t)^T ∫_0^1 grad log p_{t'}(x_t + u(x_{t'} - x_t)) du``.
    NOTE: **both** endpoints are evaluated at level ``t'``. A cheaper variant evaluates the first at
    level ``t`` to reuse the base step's velocity, but that is a second approximation on top of the
    quadrature, valid only when ``t' ~ t``; at eta = 1.5 the levels are not close (t = 0.5 -> t' = 0.4495).
    Doing it properly costs M extra rows per step against a reward-dominated step -- do not "optimize"
    this away.
    """
    return (0.5 * (s_at_x + s_at_x_prime) * (x_prime - x)).sum(dim=-1)


def _zscore(v: Float[Tensor, "M K"]) -> Float[Tensor, "M K"]:
    """Z-score across the ``K`` lookahead samples of each particle (per ``m``, over ``k``), ddof = 1.

    ``sigma`` is floored at ``MIN_ZSCORE_STD``: it collapses to 0 whenever the K candidates coincide,
    which is possible at small ``eta``, where ``t' ~ t`` and the renoise barely moves the state.
    """
    k = v.shape[1]
    mu = v.mean(dim=1, keepdim=True)
    centered = v - mu
    if k > 1:
        var = centered.pow(2).sum(dim=1, keepdim=True) / (k - 1)
    else:
        var = torch.zeros_like(mu)
    return centered / var.sqrt().clamp_min(MIN_ZSCORE_STD)


def _soft_value(v_hat: Float[Tensor, "M K"]) -> Float[Tensor, "M"]:
    """``V^m = log( (1/K) sum_k exp(v_k) )`` = ``logsumexp_k(v_k) - log K``.

    Always ``logsumexp``, never a running sum of ``exp``: ``lambda·f`` reaches ~3 at the calibrated
    lambda, and the normalized branch can reach ``eta·lambda_normalized`` scale.
    """
    return torch.logsumexp(v_hat, dim=1) - math.log(v_hat.shape[1])


def _resample(
    x: Float[Tensor, "M d"],
    V_prev: Float[Tensor, "M"],
    U: Float[Tensor, "M"],
    ancestors: Int[Tensor, "M"],
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Int[Tensor, "M"]]:
    """Draw parents ∝ softmax(U), reindex ``x``/``V_prev``/``ancestors``, zero the potential.

    Systematic (not multinomial) resampling, per ``smc_common``. Besides the lower variance, this makes
    the ``lambda = 0`` case *exact*: on uniform weights systematic resampling returns the identity
    permutation, so an untilted run reproduces the base sampler particle-for-particle.
    """
    idx = _systematic_resample(torch.softmax(U, dim=0), generator)
    return x[idx], V_prev[idx], torch.zeros_like(U), ancestors[idx], idx


def _antithetic_noise(
    m: int, k: int, d: int, *, antithetic: bool,
    generator: torch.Generator, device: torch.device, dtype: torch.dtype,
) -> Float[Tensor, "M K d"]:
    """``K`` lookahead noise draws per particle, optionally antithetically paired (``eps_{2j+1} = -eps_{2j}``).

    Free variance reduction, worth having at K = 4. An odd ``K`` leaves the last draw unpaired.
    """
    if not antithetic:
        return torch.randn((m, k, d), generator=generator, device=device, dtype=dtype)
    half = (k + 1) // 2
    base = torch.randn((m, half, d), generator=generator, device=device, dtype=dtype)
    return torch.stack([base, -base], dim=2).reshape(m, 2 * half, d)[:, :k]


def _in_window(t: float, window: tuple[float, float]) -> bool:
    """Half-open-ish membership with a float-slop tolerance, so a grid point on the edge counts."""
    return (window[0] - 1e-9) <= t <= (window[1] + 1e-9)


def _check_window(name: str, window: tuple[float, float], flags: list[bool]) -> None:
    """A window must be a nonempty sub-interval of ``[0, 1]`` selecting a *contiguous* run of steps.

    ``t = 0`` IS degenerate -- ``g(0) = sigma^2/alpha^2 = inf``, so ``t'`` is undefined, and
    ``_score_at``'s ``c = alpha_t/sigma_t^2`` is 0 with ``gamma = 0`` (``sigma_EDM = inf`` inside
    ``edm_score_fn``). But a lower bound of 0 is nonetheless safe, and the windows are allowed to start
    there, because **neither window can ever put ``t = 0`` in a degenerate position**:

    * ``guid_window`` is evaluated on each step's *target* ``ts[n+1]``, which is the level the lookahead
      runs at. On any valid grid that is at least ``ts[1] > 0``, so ``t = 0`` never becomes a lookahead
      level however wide the window is. (At ``N = 16`` the earliest is ``t = 0.0625``, where
      ``t' = 0.0516``, ``c = 0.057``, ``gamma = 0.0030`` and the renoise variance is 0.30 -- all
      comfortably conditioned.) The caller-facing invariant is asserted below.
    * ``stoch_window`` is evaluated on each step's *source* ``ts[n]``, which can be 0 -- but the only
      thing it selects there is a `TransitionStep`, and ``ddpm_step`` is explicitly non-singular at
      ``t = 0`` (``rho = 0``, giving the pure draw ``alpha_{t'} z_hat + sigma_{t'} eps``). Its one pole
      is ``sigma_t = 0`` at ``t = 1``, which is never a source.

    Symmetrically ``t = 1`` is only ever a step *target*, never a source.

    Contiguity matters for guidance: a *deterministic* step between two guided ones would carry
    resampled clones into a lookahead that scores them identically, wasting the step. An *interval*
    window on an increasing grid always selects a consecutive run, so the guard below is unreachable
    through today's API -- it is here so a future non-interval window spec cannot silently drop a step.
    (The mandatory terminal update is deliberately outside this check: with a ``(0.1, 0.9)`` window the
    guided steps are the run up to ``t = 0.875`` *plus* ``t = 1``, and that gap is by design.)
    """
    lo, hi = window
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= lo <= hi <= 1, got {window}")
    on = [i for i, flag in enumerate(flags) if flag]
    if on and on != list(range(on[0], on[-1] + 1)):
        raise ValueError(f"{name}={window} selects a non-contiguous set of steps: {on}")


# ---------------------------------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------------------------------

@dataclass
class StepSnapshot:
    """The particle cloud after one transition, kept when ``keep_steps=True``.

    Captured *after* the step's potential update and resampling, so ``X`` at ``t`` is the cloud the run
    actually carried forward. Intermediate states can be decoded with the existing FLUX notebooks'
    ``decode_levels.py``.

    **MIND THE ORDER.** ``X``, ``U`` and ``ancestors`` are POST-resample; ``V`` is the potential as
    computed, i.e. PRE-resample. On a step that resampled they are therefore in *different* particle
    orders, and joining them per-particle without going through ``resample_idx`` is silently wrong.
    ``U_pre`` and ``resample_idx`` are recorded precisely so no reader has to reconstruct either:
    ``U`` is all zeros after a resample, so the weights the decision was taken on are otherwise gone.
    """

    t: float
    X: Float[Tensor, "M d"]
    U: Float[Tensor, "M"]
    V: Float[Tensor, "M"]
    ancestors: Int[Tensor, "M"]
    U_pre: Float[Tensor, "M"] | None = None
    resample_idx: Int[Tensor, "M"] | None = None
    f_proj: Float[Tensor, "M"] | None = None


@dataclass
class FlowMapSMCResult:
    """Output of :func:`flowmap_smc_sample`.

    Attributes:
        X:          final particles at ``t = 1``, ``(M, d)``.
        logw:       final unnormalized log-weights ``U``, ``(M,)``, *including* the mandatory terminal
                    update ``V_N = lambda·f(x_1)``. ``softmax(logw)`` gives the self-normalized
                    q̂_lambda weights; all-equal when the run ended on a resample or with
                    ``final_resample=True``.
        ess_history:       ESS after each step's potential update, before any resampling. Length ``N``.
        resampled_history: whether each step actually resampled. Length ``N``.
        uniq_history:      distinct ancestors surviving after each step, as a fraction of ``M``.
                    **Read ``uniq_history[-1]`` first when tuning the window**: with a ``(0.1, 0.9)``
                    guidance window it reports whether the tail froze resampled duplicates into the
                    returned images. With ``ess_history`` this is the degeneracy signal that decides how
                    far lambda can be pushed -- novelty ``E_q[f]`` rises forever and is *not* the signal.
        V_history:         per-step potential ``V`` per particle, each ``(M,)``. Length ``N``.
        t_history:         the time each step landed on (``ts[n+1]``). Length ``N``.
        t_prime_history:   the renoise level ``t'`` used by each step's lookahead; ``nan`` on unguided
                    steps and on the terminal step (which has no lookahead). Length ``N``.
        guided_history:    whether each step ran the lookahead / reward. Length ``N``.
        f_mean/f_std/f_min/f_max: reward statistics over that step's lookahead candidates (over the
                    final particles at the terminal step); ``nan`` on unguided steps. Length ``N`` each.
        l_mean/l_std, s_mean/s_std: the same for the raw ``L`` (log-likelihood) and ``S`` (score
                    correction) terms, so the twist can be inspected -- at K = 4 it is a live question
                    whether they move the weights at all. ``nan`` outside the guided, non-terminal steps.
        U_pre_history:     the accumulated potential ``U`` **before** that step's resampling, ``(M,)``
                    per step. This is the quantity the resampling decision is actually taken on, and
                    the correct weight for a self-normalized estimate at step ``n``. It is destroyed
                    in the returned state -- ``_resample`` zeroes ``U`` -- so it is recorded here
                    rather than left for a reader to reconstruct from ``V`` differences.
        resample_idx_history: the **parent map** applied at each step, ``(M,)``, or ``None`` where the
                    step did not resample. Not recoverable from ``ancestors``, which is the composition
                    of every resampling so far; see `_resample`.
        r_k_history:       per-step lookahead rewards ``(M, K)`` when ``record_r_k=True``, else
                    ``None`` per step. ``None`` on unguided and terminal steps (K = 1 there). This is
                    what lets any ``K' <= K`` be reconstructed offline by subsetting.
        f_proj_history:    ``f`` of each particle's projected clean endpoint, ``(M,)``, when
                    ``project_endpoint=True``; ``None`` per step otherwise. Recorded in
                    **pre-resample** particle order, so it aligns with ``U_pre_history``.
        eqf_history:       ``softmax(U_pre) . f_proj`` -- the self-normalized estimate of
                    ``E_q_n[f]`` at that step. ``nan`` when ``project_endpoint=False``.
        steps:      per-step `StepSnapshot`s when the run was given ``keep_steps=True``; ``None`` otherwise.
    """

    X: Float[Tensor, "M d"]
    logw: Float[Tensor, "M"]
    ess_history: list[float] = field(default_factory=list)
    resampled_history: list[bool] = field(default_factory=list)
    uniq_history: list[float] = field(default_factory=list)
    V_history: list[Tensor] = field(default_factory=list)
    t_history: list[float] = field(default_factory=list)
    t_prime_history: list[float] = field(default_factory=list)
    guided_history: list[bool] = field(default_factory=list)
    f_mean: list[float] = field(default_factory=list)
    f_std: list[float] = field(default_factory=list)
    f_min: list[float] = field(default_factory=list)
    f_max: list[float] = field(default_factory=list)
    l_mean: list[float] = field(default_factory=list)
    l_std: list[float] = field(default_factory=list)
    s_mean: list[float] = field(default_factory=list)
    s_std: list[float] = field(default_factory=list)
    U_pre_history: list[Tensor] = field(default_factory=list)
    resample_idx_history: list[Tensor | None] = field(default_factory=list)
    r_k_history: list[Tensor | None] = field(default_factory=list)
    f_proj_history: list[Tensor | None] = field(default_factory=list)
    eqf_history: list[float] = field(default_factory=list)
    steps: list[StepSnapshot] | None = None


def _stats(v: Tensor) -> tuple[float, float]:
    return float(v.mean()), (float(v.std()) if v.numel() > 1 else 0.0)


# Called after each step's bookkeeping with ``(step_index, snapshot, partial_result)``. Read-only by
# contract: it consumes no randomness and must not mutate what it is handed, so a run with a callback
# is bit-identical to one without. This is the hook a notebook checkpoints and enforces a wall-clock
# deadline from -- the loop is one long library call and cannot be interrupted from outside.
StepCallback = Callable[[int, StepSnapshot, FlowMapSMCResult], None]


# ---------------------------------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------------------------------

def flowmap_smc_sample(
    reward: Reward,
    lam: float,
    n_particles: int,
    *,
    flow_map: FlowMap,
    score_fn: ScoreFn,
    schedule: Schedule | None = None,
    ts: list[float] | None = None,
    n_steps: int = 16,
    mc_samples: int = 4,
    eta: float = 1.5,
    guid_window: tuple[float, float] = (0.1, 1.0),
    stoch_window: tuple[float, float] = (0.1, 1.0),
    inside_step: TransitionStep = ddpm_step,
    outside_step: TransitionStep = flow_map_step,
    use_full_normalized_v: bool = False,
    lam_normalized: float = 1.0,
    ess_threshold: float = 1.0,
    antithetic: bool = True,
    final_resample: bool = False,
    record_r_k: bool = False,
    project_endpoint: bool = False,
    keep_steps: bool = False,
    snapshot_device: torch.device | str | None = "cpu",
    on_step: "StepCallback | None" = None,
    seed: int | None = None,
    verbose: bool = False,
) -> FlowMapSMCResult:
    """Sample ``q_lambda ∝ p · exp(lambda·f)`` in one forward pass over a flow map's trajectory.

    Args:
        reward:     the frozen tilt ``f``. Called on flat ``(B, d)`` particles, matching the ``Reward``
                    contract; ``reward.x_refs`` pins ``d``, device and dtype for the whole run.
        lam:        inverse temperature of the target. ``lam = 0`` reduces the run to the base process.
        n_particles: number of particles ``M``.
        flow_map:   the two generative operations (see `FlowMap`). A **structural type, not a backend
                    object** -- the same spirit as ``PCNKernel(generator_fn=...)``. Everything that
                    defines ``p`` (checkpoint, prompt, guidance scale, prompt embeddings) is constructed
                    in the notebook and closed over. **The library never sees a prompt.**
        score_fn:   the marginal score, in this repo's gamma convention. Must be the **same object**
                    handed to the reward's ``Distance``, so ``f``'s geometry and ``S_k`` share one code
                    path. Build it once as
                    ``edm_score_fn(chunked_denoiser(flow_map_denoiser(flow_map, schedule), rows), img_shape)``.
        schedule:   the interpolant (see `Schedule`). ``None`` => `LinearSchedule`.
        ts:         explicit time grid, length ``n_steps + 1``, strictly increasing from 0 to 1.
                    ``None`` => ``linspace(0, 1, n_steps + 1)``.
        n_steps:    number of transitions ``N`` when ``ts`` is not given.
        mc_samples: lookahead draws ``K`` per particle per guided step. Larger ``K`` gives a less noisy
                    value estimate; it is also the direct cost multiplier on the reward, which is ~97%
                    of a FLUX-scale run.
        eta:        SNR factor for the renoise level, ``g(t') = eta·g(t)``. ``eta > 1``.
        guid_window: **where the reward runs.** A step is guided when the time it lands on,
                    ``ts[n+1]``, is inside the window. Set empirically. ``(0.0, 1.0)`` guides every
                    step and is safe: the lookahead level is the *target* time, so it is never 0
                    (see `_check_window`).
        stoch_window: **which transitions are stochastic**, by the time each step *starts* from,
                    ``ts[n]``. Together with ``ts``, ``schedule`` and the two `TransitionStep`s this
                    defines the base process ``p`` -- so unlike ``guid_window``, changing it
                    invalidates any reference set drawn under the old value.
        inside_step / outside_step: the transitions used inside / outside ``stoch_window``.
        use_full_normalized_v: ``False`` (default) => ``v_k = lambda·R_k``. ``True`` => each of the three
                    raw scalars is independently Z-scored across the K lookahead samples of that
                    particle and combined as ``v_k = eta·(lam_normalized·R̂_k + L̂_k + Ŝ_k)``.
        lam_normalized: the O(1) knob multiplying the Z-scored reward. **Not** ``lam``, and they must
                    never share a value: Z-scoring is scale-invariant, so ``lam`` applied before it
                    would cancel; applied after, it multiplies a unit-variance quantity, and the
                    calibrated ``lam ~ O(100)`` would drive ``logsumexp`` into a hard ``max_k`` and
                    annihilate ``L̂``/``Ŝ``.
        ess_threshold: resample when ``ESS < ess_threshold·M``. ``1.0`` means resample at every step.
                    Lowering it is one of the two real levers against duplicate outputs (the other is
                    ``M``) -- ``stoch_window`` is not.
        antithetic: pair the lookahead noise draws (``eps_{2j+1} = -eps_{2j}``).
        final_resample: resample once more after the terminal update, yielding an equally-weighted
                    ensemble. Off by default: ``logw`` carries the weight.
        record_r_k: keep each guided step's lookahead rewards ``(M, K)`` in ``r_k_history``. Costs
                    ``M*K`` floats per step and no compute. What it buys is the whole ``K' <= K``
                    axis offline: ``V(K')`` for any smaller ``K'`` is a subset average of the same
                    numbers, so one run at ``K`` answers for every ``K'`` below it.
        project_endpoint: at every step, **before** resampling, push each particle to a clean endpoint
                    ``map(x, t_next, 1)`` and evaluate ``f`` on it, recording ``f_proj_history`` and
                    the self-normalized ``eqf_history``. This is the honest per-step estimate of
                    ``E_q_n[f]``: the alternative (``f_mean`` over the ``M*K`` lookahead candidates) is
                    unweighted and taken at the renoised level ``t'``, so it is a biased proxy.
                    **Pre-resample by design** -- the self-normalized SMC estimate uses the weights
                    before resampling, and resampling only adds variance. Costs ``M`` reward rows per
                    step, except the terminal step, which reuses the reward it already computed.
                    ``map`` and ``reward`` are both deterministic and consume no generator, and
                    nothing here touches ``U``, so the trajectory is bit-identical either way.
        keep_steps: also return the cloud at every ``t`` (see `StepSnapshot`), so intermediate states
                    can be decoded. Snapshotting only reads state, so the run stays bit-identical.
        snapshot_device: where ``keep_steps`` snapshots are stored. ``"cpu"`` by default so a long
                    high-dimensional run does not accumulate them in VRAM; ``None`` keeps them in place.
        on_step:    optional read-only callback ``(step_index, snapshot, partial_result)`` run after each
                    step's bookkeeping (see `StepCallback`). The loop is one long call, so this is the
                    only place a notebook can checkpoint or enforce a wall-clock deadline -- raise from
                    it to stop the run and keep whatever has already been written. It consumes no
                    randomness, so passing one leaves the run bit-identical. A snapshot is built for the
                    callback whether or not ``keep_steps`` is set.
        seed:       seeds three independent generators: ``seed`` (base steps), ``seed + 1`` (lookahead)
                    and ``seed + 2`` (resampling). ``None`` => 0.
        verbose:    print per-step ESS / reward statistics.

    Returns:
        :class:`FlowMapSMCResult` with the final particles, their log-weights and per-step diagnostics.

    Note:
        The fully principled lookahead weight is the *unnormalized*
        ``v_k = lambda·R_k + L_k + S_k + 1/2·||eps_k||^2``, where the fourth term is
        ``-log q_{t'|t}(x_{t'}|x_t)`` up to a k-independent constant. This is true because the lookahead
        is an importance-sampling estimate of the reward of the clean version of each particle: together
        the four terms form the importance weight ``p(x_t|z)·p_{t'}(x_{t'}) / q_{t'|t}(x_{t'}|x_t)`` that
        corrects the renoise-denoise proposal to the true lookahead posterior. We deliberately use the
        three-term Z-scored form instead (``use_full_normalized_v=True``) or the reward alone
        (the default). Recorded so the choice stays legible and the omitted term is easy to restore
        for an ablation.
    """
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")
    if mc_samples < 1:
        raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")
    if eta <= 0.0:
        raise ValueError(f"eta must be > 0, got {eta}")

    schedule = LinearSchedule() if schedule is None else schedule

    if ts is None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        ts = _uniform_ts(n_steps)
    ts = [float(v) for v in ts]
    if len(ts) < 2:
        raise ValueError(f"ts needs at least 2 entries, got {len(ts)}")
    if abs(ts[0]) > 1e-12 or abs(ts[-1] - 1.0) > 1e-12:
        raise ValueError(f"ts must run from 0 to 1, got [{ts[0]}, ..., {ts[-1]}]")
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("ts must be strictly increasing")
    n_steps = len(ts) - 1

    # Windows are read at different ends of a step ON PURPOSE: `stoch_window` selects the transition,
    # which is an interval and is keyed on where it starts; `guid_window` selects where V is evaluated,
    # which is a point and is keyed on where the step lands.
    stoch_flags = [_in_window(ts[n], stoch_window) for n in range(n_steps)]
    guid_flags = [_in_window(ts[n + 1], guid_window) for n in range(n_steps)]
    _check_window("stoch_window", stoch_window, stoch_flags)
    _check_window("guid_window", guid_window, guid_flags)
    # The invariant that actually matters (see `_check_window`): every guided step's lookahead runs at
    # its TARGET time, and that level must be > 0 or `_t_prime` / `_score_at` are undefined there. This
    # holds structurally for a grid starting at 0 and strictly increasing, so it is a guard against a
    # future non-standard grid rather than against a bad `guid_window`.
    if any(flag and ts[n + 1] <= 0.0 for n, flag in enumerate(guid_flags)):
        raise ValueError("a guided step lands on t = 0, where the lookahead level is undefined")

    device, dtype = reward.x_refs.device, reward.x_refs.dtype
    d = int(reward.x_refs.shape[1])

    # Three INDEPENDENT streams. Sharing one would let the number of lookahead draws -- or of
    # resampling events -- shift the base trajectory, and the lambda = 0 reproduction check could not
    # pass. Whether a step resamples at lambda = 0 hinges on float slop in `ESS == M`, so the resampling
    # stream in particular must not be the base one.
    base_seed = 0 if seed is None else int(seed)
    gen_base = torch.Generator(device=device).manual_seed(base_seed)
    gen_look = torch.Generator(device=device).manual_seed(base_seed + 1)
    gen_resample = torch.Generator(device=device).manual_seed(base_seed + 2)

    M, K = n_particles, mc_samples
    x = torch.randn((M, d), generator=gen_base, device=device, dtype=dtype)
    U = torch.zeros(M, device=device, dtype=dtype)
    V_prev = torch.zeros(M, device=device, dtype=dtype)
    ancestors = torch.arange(M, device=device)

    result = FlowMapSMCResult(X=x, logw=U, steps=[] if keep_steps else None)

    def keep(t_: Tensor) -> Tensor:
        t_ = t_.detach().clone()
        return t_ if snapshot_device is None else t_.to(snapshot_device)

    for n in range(n_steps):
        t, t_next = ts[n], ts[n + 1]
        is_final = n == n_steps - 1

        # --- (1) base transition. `p` depends on stoch_window, NEVER on guid_window ----------------
        step_fn = inside_step if stoch_flags[n] else outside_step
        x = step_fn(x, t, t_next, flow_map=flow_map, schedule=schedule, generator=gen_base)

        # --- (2-3) lookahead and potential ---------------------------------------------------------
        # The terminal update is independent of the window and ALWAYS runs: at t = 1 the lookahead
        # posterior is a point mass, so V_N = lambda·f(x_1) exactly -- one reward call on M particles
        # (not M·K), no Monte Carlo, no normalization (K = 1 there, so a Z-score is undefined anyway).
        # This is what pins the target to q_lambda; without it a (0.1, 0.9) window would end on
        # V_{0.875} and push that cloud to t = 1 through the untilted transition, which is biased.
        guided = bool(guid_flags[n]) or is_final
        t_p = float("nan")
        f_stats: tuple[float, float, float, float] = (math.nan, math.nan, math.nan, math.nan)
        l_stats: tuple[float, float] = (math.nan, math.nan)
        s_stats: tuple[float, float] = (math.nan, math.nan)
        r_k_kept: Tensor | None = None
        f_proj: Tensor | None = None

        if guided and is_final:
            f = reward(x)
            V_next = lam * f
            # At t = 1 the particle IS the clean sample, so the projection is this same reward --
            # reuse it rather than paying M more rows on `map(x, 1, 1)`, which is also ill-posed.
            if project_endpoint:
                f_proj = f.detach().clone()
            f_stats = (float(f.mean()), float(f.std()) if f.numel() > 1 else 0.0,
                       float(f.min()), float(f.max()))
        elif guided:
            t_p = _t_prime(t_next, eta, schedule)
            eps = _antithetic_noise(M, K, d, antithetic=antithetic, generator=gen_look,
                                    device=device, dtype=dtype)
            x_rep = x.unsqueeze(1).expand(M, K, d).reshape(M * K, d)
            x_tp = _renoise(x_rep, t_next, t_p, schedule, eps.reshape(M * K, d))
            z = flow_map.map(x_tp, t_p, 1.0)                     # M·K clean candidates

            r_k = reward(z)                                       # the frozen reward, M·K rows
            l_k = _log_likelihood(x_rep, z, t_next, schedule)
            # BOTH endpoints at level t' -- see `_score_correction`.
            s_k = _score_correction(
                _score_at(x, t_p, score_fn, schedule).unsqueeze(1).expand(M, K, d).reshape(M * K, d),
                _score_at(x_tp, t_p, score_fn, schedule),
                x_rep, x_tp,
            )

            if use_full_normalized_v:
                v_hat = eta * (lam_normalized * _zscore(r_k.view(M, K))
                               + _zscore(l_k.view(M, K)) + _zscore(s_k.view(M, K)))
            else:
                v_hat = lam * r_k.view(M, K)
            V_next = _soft_value(v_hat)

            f_stats = (float(r_k.mean()), float(r_k.std()) if r_k.numel() > 1 else 0.0,
                       float(r_k.min()), float(r_k.max()))
            l_stats, s_stats = _stats(l_k), _stats(s_k)
            if record_r_k:
                r_k_kept = r_k.view(M, K).detach().clone()
        else:
            # Outside the guidance window U and V are carried forward unchanged: the K lookaheads, the
            # reward and the potential update are all skipped. The transition still happened above.
            V_next = V_prev

        U = U + V_next - V_prev
        V_prev = V_next

        # --- (3b) projected endpoint, BEFORE resampling ---------------------------------------------
        # The self-normalized estimate of E_q_n[f] uses the weights the step arrived with; resampling
        # is variance-adding, so measuring after it throws information away. Deterministic and
        # U-free, so the trajectory is unaffected -- see `project_endpoint`.
        if project_endpoint and f_proj is None:
            f_proj = reward(flow_map.map(x, t_next, 1.0)).detach().clone()
        U_pre = U.detach().clone()
        eqf = (float((torch.softmax(U_pre, dim=0) * f_proj).sum())
               if f_proj is not None else math.nan)

        # --- (4) ESS-triggered resampling ----------------------------------------------------------
        ess = _ess_from_logw(U)
        do_resample = (ess < ess_threshold * M) and (not is_final or final_resample)
        resample_idx: Tensor | None = None
        if do_resample:
            x, V_prev, U, ancestors, resample_idx = _resample(
                x, V_prev, U, ancestors, gen_resample
            )

        result.U_pre_history.append(U_pre)
        result.resample_idx_history.append(
            None if resample_idx is None else resample_idx.detach().clone()
        )
        result.r_k_history.append(r_k_kept)
        result.f_proj_history.append(f_proj)
        result.eqf_history.append(eqf)
        result.ess_history.append(ess)
        result.resampled_history.append(do_resample)
        result.uniq_history.append(len(torch.unique(ancestors)) / M)
        result.V_history.append(V_next.detach().clone())
        result.t_history.append(t_next)
        result.t_prime_history.append(t_p)
        result.guided_history.append(guided)
        result.f_mean.append(f_stats[0]); result.f_std.append(f_stats[1])
        result.f_min.append(f_stats[2]); result.f_max.append(f_stats[3])
        result.l_mean.append(l_stats[0]); result.l_std.append(l_stats[1])
        result.s_mean.append(s_stats[0]); result.s_std.append(s_stats[1])
        if result.steps is not None or on_step is not None:
            snap = StepSnapshot(
                t=t_next, X=keep(x), U=keep(U), V=keep(V_next), ancestors=keep(ancestors),
                U_pre=keep(U_pre),
                resample_idx=None if resample_idx is None else keep(resample_idx),
                f_proj=None if f_proj is None else keep(f_proj),
            )
            if result.steps is not None:
                result.steps.append(snap)
            if on_step is not None:
                # Kept last so `result`'s per-step lists are already complete for this step. Anything
                # raised here propagates: that is how a deadline stops the run.
                on_step(n, snap, result)

        if verbose:
            kind = "stoch" if stoch_flags[n] else "det  "
            twist = f"guided tp={t_p:.4f}" if guided else "unguided       "
            print(
                f"  step {n + 1}/{n_steps}  t={t:.4f}->{t_next:.4f}  {kind}  {twist}  "
                f"ESS/M={ess / M:.2f}  resampled={do_resample}  "
                f"uniq/M={result.uniq_history[-1]:.2f}  f mean={f_stats[0]:.4f}"
            )

    result.X = x
    result.logw = U
    return result
