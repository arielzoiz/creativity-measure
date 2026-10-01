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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import VelocityFn
from creativity_measure.tilt import Reward

_EPS = 1e-12


@runtime_checkable
class _GammaChunkable(Protocol):
    """Structural capability a Distance needs for the OOM fallback to chunk the MC (gamma) axis instead
    of the batch axis -- see `_reward_grad`'s docstring for why the batch axis is the wrong one to chunk.

    Added to `SquaredIIDGlobalIEMDistance` in notebooks/iid_iem_flux_check/ROADMAP.md Phase 3 step 1b,
    deliberately AFTER this module and its non-fallback tests, so this file can be written and verified
    without editing the reward class while a Phase-1 GPU job (frozen bank, additive-only change) is live.
    Until that method exists, `isinstance(distance, _GammaChunkable)` is simply False and the fallback
    raises NotImplementedError -- the approximate/exact gradient paths themselves need none of this.
    """
    gammas: Tensor

    def expected_gamma_chunk(
        self, X: Float[Tensor, "B d"], x_refs: Float[Tensor, "R d"], g_lo: int, g_hi: int,
    ) -> Float[Tensor, "B"]: ...


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
    applied_norm_history: list[float] = field(default_factory=list)
    f_hat0_history: list[float] = field(default_factory=list)
    oom_fallback_history: list[bool] = field(default_factory=list)
    static_fallback_history: list[bool] = field(default_factory=list)


def _shifted_schedule(n_steps: int, shift: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Resolution-shifted flow-matching schedule, diffusers-native t (1 = noise, 0 = data).

    Mirrors ``FlowMatchEulerDiscreteScheduler`` (FLUX.1-dev's default, also used by SD3/SD3.5): linear
    sigmas in ``[1/n_steps, 1]`` (descending), then the shift ``sigma <- shift*sigma / (1 + (shift-1)*sigma)``
    (shift=1 is the identity, i.e. unshifted -- the right choice for a backend with no resolution shift),
    with a trailing 0 appended -- same "N steps from N+1 nodes, final node 0" shape as this repo's own
    ``generators.base.karras_sigma_schedule``, just in flow-matching t instead of EDM sigma.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    sigmas = torch.linspace(1.0, 1.0 / n_steps, n_steps, device=device, dtype=dtype)
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _reward_grad(
    reward: Reward,
    reward_input: Float[Tensor, "B d"],
    grad_target: Float[Tensor, "B d"],
    *,
    create_graph: bool = False,
    g_chunk: int | None = None,
    force_fallback: bool = False,
) -> tuple[Float[Tensor, "B d"], bool]:
    """``grad_{grad_target} reward(reward_input).sum()``, with a gamma-chunked fallback on OOM.

    ``reward_input`` and ``grad_target`` are the SAME tensor on the approximate path (grad w.r.t.
    ``x_hat_0`` itself) and DIFFERENT tensors on the exact path (``reward_input = x_hat_0``,
    ``grad_target = x_t``/``x_req``, since the graph connecting them must be walked). Returns
    ``(grad, fell_back)``.

    ``create_graph=False`` and ``retain_graph=False`` (``True`` only for non-final fallback chunks, which
    all share ``grad_target``'s subgraph) are always passed explicitly, never left to default: this frees
    intermediate forward activations the instant the gradient is computed, keeping VRAM flat across a
    28-50 step trajectory instead of ballooning.

    The fallback chunks the MC (gamma) axis, NOT the batch axis: score rows are G * N_eps * B
    (`iid_global_iem.iid_score_bank`), so at B=1 -- the standard case for a 12 B model, one sample per
    trajectory -- a batch-dimension loop provides ZERO memory relief, while a gamma-chunk loop cuts peak
    concurrently-live score activations by g_chunk/G. Exact up to floating-point reduction order: the
    reward decomposes as an exact weighted sum over independent per-gamma terms
    (`iid_iem_sq_expected`), never a nested autograd.grad, so chunked accumulation reproduces the batched
    gradient (Phase 2 measured 1.86e-09 for the analogous per-sample split).
    """
    if not force_fallback:
        try:
            loss = reward(reward_input).sum()
            (grad,) = torch.autograd.grad(loss, grad_target, create_graph=create_graph, retain_graph=False)
            return grad, False
        except torch.cuda.OutOfMemoryError:
            if grad_target.device.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

    distance = reward.distance
    if not isinstance(distance, _GammaChunkable):
        raise NotImplementedError(
            "reward ran out of memory and its Distance does not support the gamma-chunked OOM fallback "
            f"(needs expected_gamma_chunk; got {type(distance).__name__}). See "
            "notebooks/iid_iem_flux_check/ROADMAP.md Phase 3 step 1b."
        )
    if reward.weights is not None:
        raise NotImplementedError("gamma-chunked OOM fallback does not yet support non-uniform weights")
    denom = getattr(reward, "_denom", 1.0)
    G = distance.gammas.shape[0]
    chunk = g_chunk if g_chunk is not None else 1
    bounds = list(range(0, G, chunk)) + [G]
    total_grad = torch.zeros_like(grad_target)
    for i in range(len(bounds) - 1):
        g_lo, g_hi = bounds[i], bounds[i + 1]
        is_last = g_hi >= G
        partial = distance.expected_gamma_chunk(reward_input, reward.x_refs, g_lo, g_hi)
        loss = (partial / denom).sum()
        (g,) = torch.autograd.grad(loss, grad_target, create_graph=create_graph, retain_graph=not is_last)
        total_grad = total_grad + g
        del loss, g
        if grad_target.device.type == "cuda":
            torch.cuda.empty_cache()
    return total_grad, True


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
        module = getattr(velocity_fn, "module", None)
        if module is None:
            raise ValueError(
                "exact_jacobian=True needs velocity_fn.module (e.g. from "
                "generators.flux.flux_velocity_fn, or any backend builder conforming to "
                "_types.GuidableVelocityFn) to verify the model is frozen before building an autograd "
                "graph through it."
            )
        if any(p.requires_grad for p in module.parameters()) or module.training:
            raise ValueError(
                "exact_jacobian=True requires velocity_fn.module to be frozen (requires_grad_(False)) "
                "and in eval() -- build it with flux_velocity_fn(..., differentiable=True) (or another "
                "backend's equivalent). Skipping this check would let autograd allocate gradient buffers "
                "for every model parameter and OOM on the first backward."
            )

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
        dt = t_from - t_to
        guided = lam != 0.0 and t_end <= t_from <= t_start

        if not guided:
            with torch.no_grad():
                v = velocity_fn(x, t_from)
            res.t_history.append(t_from)
            res.guided_history.append(False)
            res.grad_norm_history.append(float("nan"))
            res.v_norm_history.append(float(v.norm(dim=1).mean()))
            res.applied_norm_history.append(0.0)
            res.f_hat0_history.append(float("nan"))
            res.oom_fallback_history.append(False)
            res.static_fallback_history.append(False)
            x = (x - dt * v).detach()
            continue

        if exact_jacobian:
            x_req = x.detach().requires_grad_(True)
            v = velocity_fn(x_req, t_from)                        # grad ENABLED: graph starts at x_req
            x_hat0 = x_req - t_from * v
            g, fell_back = _reward_grad(reward, x_hat0, x_req, create_graph=False, g_chunk=g_chunk)
            v = v.detach()
            x_hat0 = x_hat0.detach()
        else:
            with torch.no_grad():
                v = velocity_fn(x, t_from)
            x_hat0 = (x - t_from * v).detach().requires_grad_(True)
            g, fell_back = _reward_grad(reward, x_hat0, x_hat0, create_graph=False, g_chunk=g_chunk)
            x_hat0 = x_hat0.detach()

        if grad_clip_percentile is not None:
            k = max(1, int(g.shape[1] * grad_clip_percentile / 100.0))
            thresh = g.abs().kthvalue(k, dim=1).values.clamp_min(_EPS)
            g = g.clamp(min=-thresh.unsqueeze(1), max=thresh.unsqueeze(1))

        grad_norm = g.norm(dim=1, keepdim=True)
        v_norm = v.norm(dim=1, keepdim=True)
        static_fallback = bool((v_norm < min_v_norm).any())
        if grad_scaling == "velocity":
            scale = torch.where(v_norm < min_v_norm, torch.full_like(v_norm, static_scale), v_norm)
            g_t = g / (grad_norm + _EPS) * scale
        else:
            g_t = g / (grad_norm + _EPS) * static_scale

        v_guided = v - lam * g_t
        x_next = (x - dt * v_guided).detach()  # memory isolation: never chains in VRAM across steps

        res.t_history.append(t_from)
        res.guided_history.append(True)
        res.grad_norm_history.append(float(grad_norm.mean()))
        res.v_norm_history.append(float(v_norm.mean()))
        res.applied_norm_history.append(float((lam * g_t).norm(dim=1).mean()))
        with torch.no_grad():
            res.f_hat0_history.append(float(reward(x_hat0).mean()))
        res.oom_fallback_history.append(fell_back)
        res.static_fallback_history.append(static_fallback)
        if verbose:
            print(f"[step {i}] t={t_from:.4f} guided grad_norm={res.grad_norm_history[-1]:.4g} "
                  f"v_norm={res.v_norm_history[-1]:.4g} f_hat0={res.f_hat0_history[-1]:.4g}")

        x = x_next

    res.X = x
    return res
