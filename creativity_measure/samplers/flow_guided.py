"""Phase 3 of notebooks/iid_iem_flux_check/ROADMAP.md: direct test-time guidance on a flow-matching model.

At each step of the model's own native flow-matching ODE (noise -> image), nudge the velocity prediction
by the reward's gradient at the current denoised estimate, so a SINGLE guided forward pass reaches the
creative tilt with no secondary sampling loop -- the tilted proposal that Algorithm 1 (pCN, ceilings at
m_eff ~ 2.3) and Algorithm 3 (tilts by reweighting untilted draws, ESS/M ~ exp(-m^2)) cannot buy.

This sampler is backend-agnostic: it imports nothing from `generators/`, only the `VelocityFn` contract
(`creativity_measure._types`). FLUX.1-dev is currently its only backend, via `generators/flux.py`'s
`flux_velocity_fn`/`flux_edm_denoiser`/`build_flux_guidance` -- any other model exposing the same
native-t velocity convention (e.g. SD3/SD3.5, which share FLUX's `FlowMatchEulerDiscreteScheduler`
family) can plug in the same way; see CLAUDE.md's "Adding a New Flow-Matching Backend" section.

TIME CONVENTION (read this before touching any t in this file): every t here is diffusers-native --
t = 1 is pure noise, t = 0 is clean data -- the convention `VelocityFn` itself declares (`_types.py`),
which every backend's velocity function must match. This is the OPPOSITE
polarity to `generators/flux_flowmap.py`'s own stated "this repo" convention (t=0 noise, t=1 data),
which applies only to that module's flow map. This repo has a history of exactly this class of silent
sign error (see flux_flowmap.py's docstring); do not mix the two.

Two Jacobian modes for grad_{x_t} r, both single ordinary backwards (create_graph=False everywhere --
nothing in the reward chain contains a nested autograd.grad, so no double-backward is ever needed here):

  exact_jacobian=False (default, "approximate")   v_theta under no_grad; x_hat_0 detached; grad taken
                                                    w.r.t. x_hat_0 directly (treats d(x_hat_0)/d(x_t) = I).
  exact_jacobian=True  ("exact")                   v_theta computed WITH grad enabled (weights frozen,
                                                    only x_t requires grad); grad taken w.r.t. x_t, through
                                                    the whole x_t -> v_theta -> x_hat_0 -> reward graph.

Sign, derived once here (this repo's own history shows this class of error is silent, see above): the
Euler step is ``x_next = x - dt*v_guided``. To move x_next along +g (ascending r), we need
``v_guided = v - lam*g_t``, giving ``x_next = x - dt*v + dt*lam*g_t``.

The step itself, the gradient helper, the schedule and the freeze guard now live in
``flow_guided_common.py`` (non-public, like ``smc_common.py``), because ``flow_guided_pc.py`` -- Phase 5's
predictor-corrector Langevin sampler -- uses the SAME Euler step as its predictor. ``_EPS``,
``_GammaChunkable``, ``_shifted_schedule`` and ``_reward_grad`` are re-exported here under their original
names, so existing importers keep working.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import VelocityFn
from creativity_measure.samplers.flow_guided_common import (
    _EPS as _EPS,
    _GammaChunkable as _GammaChunkable,
    _reward_grad as _reward_grad,
    _shifted_schedule as _shifted_schedule,
    guided_euler_step,
    require_frozen_module,
)
from creativity_measure.tilt import Reward


@dataclass
class FlowGuidedResult:
    """Output of :func:`flow_guided_sample`.

    Attributes:
        X:     terminal latents at ``t = t_end`` (``t_end = 0`` is the artifact-free, clean-image case;
               CLAUDE.md: E_q[f] read at intermediate t is inflated).
        t_history:            the schedule's t at the START of each step (length n_steps).
        guided_history:       whether the guidance branch was taken this step (lam != 0 and t in window).
        grad_norm_history:    mean-over-batch ||grad r|| BEFORE velocity-relative scaling, per guided step
                              (nan on unguided steps).
        v_norm_history:       mean-over-batch ||v_theta|| at each step (what "velocity" scaling targets).
        transport_v_norm_history: mean-over-batch norm of the SEPARATE transport field, when
                              ``transport_velocity_fn`` was supplied (nan otherwise). Read against
                              ``v_norm_history`` to see how far the transport field departs from the one
                              the gradient was scaled to -- e.g. how much CFG at ``w`` inflates the
                              velocity norm.
        applied_norm_history: mean-over-batch ||lam * g_t||, i.e. what actually perturbed the velocity
                              (0 on unguided steps).
        f_hat0_history:       mean-over-batch r(x_hat_0) at each guided step -- "did guidance do anything"
                              without needing to decode/look at the image (cf. diamond_smc's f_mean); nan
                              on unguided steps.
        oom_fallback_history: whether the gamma-chunked OOM fallback fired, per step (False if unguided).
        static_fallback_history: whether ||v_theta|| < min_v_norm forced the static-scale fallback.
    """

    X: Float[Tensor, "B d"]
    t_history: list[float] = field(default_factory=list)
    guided_history: list[bool] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    v_norm_history: list[float] = field(default_factory=list)
    transport_v_norm_history: list[float] = field(default_factory=list)
    applied_norm_history: list[float] = field(default_factory=list)
    f_hat0_history: list[float] = field(default_factory=list)
    oom_fallback_history: list[bool] = field(default_factory=list)
    static_fallback_history: list[bool] = field(default_factory=list)


def flow_guided_sample(
    reward: Reward,
    lam: float,
    n_samples: int,
    *,
    velocity_fn: VelocityFn,
    n_steps: int = 28,
    shift: float = 3.0,
    t_start: float = 1.0,
    t_end: float = 0.0,
    exact_jacobian: bool = False,
    grad_scaling: Literal["velocity", "static"] = "velocity",
    static_scale: float = 1.0,
    grad_clip_percentile: float | None = None,
    min_v_norm: float = 1e-4,
    g_chunk: int | None = None,
    transport_velocity_fn: VelocityFn | None = None,
    z0: Float[Tensor, "B d"] | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> FlowGuidedResult:
    """Sample by guiding a flow-matching model's own ODE with ``lam * grad_x r(x)`` at each step.

    Backend-agnostic: needs only a ``velocity_fn`` conforming to the diffusers-native ``VelocityFn``
    convention (module docstring). ``n_steps=28, shift=3.0`` below are FLUX.1-dev's own inference
    defaults (its only backend today, via ``generators/flux.py``) -- override both for another backend's
    recommended schedule (e.g. SD3 has its own ``shift``).

    Args:
        reward:      the frozen tilt f (invariant 1); ``reward.x_refs`` pins device/dtype/d for the run.
                     The reference bank is built once, before the time loop (see below), not per step.
        lam:         guidance strength. ``lam = 0`` must reproduce the unguided trajectory EXACTLY
                     (bitwise) -- asserted in tests/test_flow_guided.py, never checked here at runtime
                     (computing an unguided step as well would double inference cost for no production
                     benefit).
        n_samples:   number of independent trajectories (the batch), not SMC particles -- there is no
                     resampling here, only one deterministic-given-z0 trajectory per sample.
        velocity_fn: ``(x_t, t) -> v_theta``, diffusers-native t (see module docstring). From
                     ``generators.flux.flux_velocity_fn`` (or another backend's equivalent builder); for
                     ``exact_jacobian=True`` it must have been built frozen, with a ``.module`` attribute
                     (checked below via ``velocity_fn.module`` -- formalized as `_types.GuidableVelocityFn`).
        n_steps, shift: the Euler step count and the resolution-shift parameter (see
                     ``_shifted_schedule``); ``t_start=1.0, t_end=0.0`` is the unguided base schedule.
        t_start, t_end: guidance-window bounds in t; guidance is active at a step iff
                     ``lam != 0 and t_end <= t <= t_start``. ``(1.0, 0.0)`` guides every step.
        exact_jacobian: False (default) treats d(x_hat_0)/d(x_t) as the identity (the standard
                     universal-guidance approximation); True backpropagates through v_theta itself.
        grad_scaling: "velocity" (default) rescales grad r to ||v_theta|| per sample, so guidance never
                     overpowers the model's structural constraints; "static" uses ``static_scale``
                     unconditionally.
        static_scale: the scale used by ``grad_scaling="static"``, and the fallback whenever
                     ``||v_theta|| < min_v_norm`` under ``grad_scaling="velocity"`` (recorded in
                     ``static_fallback_history`` either way the fallback fires).
        grad_clip_percentile: if set, clip grad r elementwise at this percentile (by absolute value)
                     before normalizing -- a numerical-stability knob, not a behaviour switch.
        min_v_norm:  below this, "velocity" scaling is numerically unsafe (division by ~0) and falls back
                     to ``static_scale``.
        g_chunk:     gamma-chunk size for the OOM fallback (default 1, i.e. one gamma at a time -- the
                     most conservative, matching Phase 2's per-sample precedent).
        transport_velocity_fn: optional second velocity field used ONLY for the Euler transport, while
                     ``velocity_fn`` keeps supplying ``x_hat_0``, the reward, the gradient and the
                     ``grad_scaling="velocity"`` reference norm. ``None`` (default) is the single-field
                     behaviour, bitwise. Its purpose, and why the asymmetry is deliberate rather than an
                     oversight, is documented on ``guided_euler_step``; the first user is manual CFG
                     (``generators/cfg.py``'s ``cfg_velocity_fn``), where this is what keeps ``f``
                     comparable across the CFG weight ``w`` and stops ``w`` from silently rescaling the
                     applied gradient. The transport field is evaluated under ``no_grad`` and never
                     differentiated, so ``exact_jacobian=True`` still backpropagates one network graph.
        z0:          initial noise at t=1; if None, drawn ``N(0, I)`` from the run's own generator (flow
                     matching's noise endpoint IS standard normal -- no sigma_max rescale needed, unlike
                     the EDM path's ``x = z * sigma_max``).
        seed:        seeds the single generator this run uses (0 if None); never touches global RNG.

    Returns:
        FlowGuidedResult -- diagnostics are not optional (CLAUDE.md): read ``f_hat0_history`` and
        ``applied_norm_history`` together to tell whether guidance did anything, before ever decoding
        an image.
    """
    if exact_jacobian:
        require_frozen_module(velocity_fn, flag="exact_jacobian=True")

    x_refs = reward.x_refs
    device, dtype, d = x_refs.device, x_refs.dtype, x_refs.shape[1]
    gen = torch.Generator(device=device).manual_seed(0 if seed is None else seed)
    x = torch.randn(n_samples, d, generator=gen, device=device, dtype=dtype) if z0 is None else z0

    # Force the reference bank build ONCE, before the time loop, so the O(R) cost (CLAUDE.md: ~19 min at
    # R=64 on an L40S) is paid outside every per-step timing. `_ref_cache`/`_ref_stats` short-circuit on
    # an unchanged x_refs afterward (iid_global_iem.py) -- no special handling beyond not reconstructing
    # `reward` between steps.
    with torch.no_grad():
        reward(x_refs[:1])

    schedule = _shifted_schedule(n_steps, shift, device=device, dtype=torch.float32)
    res = FlowGuidedResult(X=x)

    for i in range(n_steps):
        t_from, t_to = float(schedule[i]), float(schedule[i + 1])
        guided = lam != 0.0 and t_end <= t_from <= t_start

        x, rec = guided_euler_step(
            x, t_from=t_from, t_to=t_to, reward=reward, lam=lam, velocity_fn=velocity_fn,
            guided=guided, exact_jacobian=exact_jacobian, grad_scaling=grad_scaling,
            static_scale=static_scale, grad_clip_percentile=grad_clip_percentile,
            min_v_norm=min_v_norm, g_chunk=g_chunk, transport_velocity_fn=transport_velocity_fn,
        )

        res.t_history.append(t_from)
        res.guided_history.append(rec.guided)
        res.grad_norm_history.append(rec.grad_norm)
        res.v_norm_history.append(rec.v_norm)
        res.transport_v_norm_history.append(rec.transport_v_norm)
        res.applied_norm_history.append(rec.applied_norm)
        res.f_hat0_history.append(rec.f_hat0)
        res.oom_fallback_history.append(rec.oom_fallback)
        res.static_fallback_history.append(rec.static_fallback)
        if verbose and rec.guided:
            print(f"[step {i}] t={t_from:.4f} guided grad_norm={rec.grad_norm:.4g} "
                  f"v_norm={rec.v_norm:.4g} f_hat0={rec.f_hat0:.4g}")

    res.X = x
    return res
