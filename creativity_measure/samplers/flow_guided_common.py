"""Shared infrastructure for the inference-time-gradient samplers (`flow_guided`, `flow_guided_pc`).

Not part of this package's public surface -- same status as `smc_common.py` for the SMC family, and for
the same reason: these are implementation details two sibling samplers must agree on *exactly*, not an
API anyone outside `samplers/` should reach for.

Everything here was originally inline in `flow_guided.py` (Phase 3 of
`notebooks/iid_iem_flux_check/ROADMAP.md`) and was lifted out unchanged when `flow_guided_pc.py`
(Phase 5, predictor-corrector Langevin) needed the *same* Euler step as its predictor. The move is
arithmetic-preserving by construction -- every expression below keeps its original operand order and
`.detach()` placement -- and
`tests/test_flow_guided.py::test_lambda_zero_matches_an_unguided_reference_loop_bitwise` is the guard
that it stayed that way.

TIME CONVENTION: every t here is diffusers-native -- t = 1 is pure noise, t = 0 is clean data -- the
convention `_types.VelocityFn` declares. See `flow_guided.py`'s module docstring for the full history of
this repo's sign/polarity traps; do not mix it with `generators/flux_flowmap.py`'s opposite convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor, nn

from creativity_measure.tilt import Reward
from creativity_measure._types import VelocityFn

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
    28-50 step trajectory instead of ballooning. ``flow_guided_pc`` leans on this harder still -- it
    builds ``1 + corrector_steps`` such graphs per ODE step, strictly sequentially.

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


def require_frozen_module(velocity_fn: VelocityFn, *, flag: str = "exact_jacobian=True") -> nn.Module:
    """Verify ``velocity_fn`` exposes a frozen, eval-mode ``.module`` before any graph is built through it.

    Every caller that backpropagates through the network itself must run this FIRST. Skipping it lets
    autograd allocate gradient buffers for every parameter of a 12 B model and OOM on the first backward
    (job 957136 measured a single grad-enabled FLUX forward at batch=2 consuming the entire remaining
    ~20 GB). ``flag`` names the kwarg that requested the exact path, so the message points at the right
    knob for whichever sampler called in; both messages keep the word "frozen", which
    ``tests/test_flow_guided.py`` matches on.
    """
    module = getattr(velocity_fn, "module", None)
    if module is None:
        raise ValueError(
            f"{flag} needs velocity_fn.module (e.g. from "
            "generators.flux.flux_velocity_fn, or any backend builder conforming to "
            "_types.GuidableVelocityFn) to verify the model is frozen before building an autograd "
            "graph through it."
        )
    if any(p.requires_grad for p in module.parameters()) or module.training:
        raise ValueError(
            f"{flag} requires velocity_fn.module to be frozen (requires_grad_(False)) "
            "and in eval() -- build it with flux_velocity_fn(..., differentiable=True) (or another "
            "backend's equivalent). Skipping this check would let autograd allocate gradient buffers "
            "for every model parameter and OOM on the first backward."
        )
    return module


def clip_grad_percentile(
    g: Float[Tensor, "B d"], grad_clip_percentile: float
) -> Float[Tensor, "B d"]:
    """Clip ``g`` elementwise at its per-sample ``grad_clip_percentile``-th absolute value.

    A numerical-stability knob, not a behaviour switch: it bounds outlier coordinates before the
    direction is renormalized, and leaves the direction itself essentially unchanged otherwise.
    """
    k = max(1, int(g.shape[1] * grad_clip_percentile / 100.0))
    thresh = g.abs().kthvalue(k, dim=1).values.clamp_min(_EPS)
    return g.clamp(min=-thresh.unsqueeze(1), max=thresh.unsqueeze(1))


def normalize_to(
    g: Float[Tensor, "B d"],
    target_norm: Float[Tensor, "B 1"],
    *,
    source_norm: Float[Tensor, "B 1"] | None = None,
    eps: float = _EPS,
) -> Float[Tensor, "B d"]:
    """``g / (||g|| + eps) * target_norm`` -- keep ``g``'s direction, take ``target_norm``'s magnitude.

    The reward's gradient magnitude is deliberately DISCARDED: ``f`` is a normalized IEM distance whose
    gradient scale carries no calibrated meaning against a velocity or a score, so every consumer picks
    an explicit reference scale instead (``||v_theta||`` for the ODE predictor, ``||s_theta||`` for the
    Langevin corrector) and lets ``lam`` be a dimensionless mixing weight against it.

    ``source_norm`` lets a caller that already holds ``||g||`` (it is a recorded diagnostic in both
    samplers) pass it in rather than pay a second reduction over ``d = 65536``.
    """
    n = g.norm(dim=1, keepdim=True) if source_norm is None else source_norm
    return g / (n + eps) * target_norm


@dataclass
class EulerStepRecord:
    """Per-step diagnostics from :func:`guided_euler_step`.

    Field-for-field the per-step slice of ``flow_guided.FlowGuidedResult``'s parallel lists, so both
    samplers can append it straight into their own result objects without translation.

    ``transport_v_norm`` defaults to nan so ``flow_guided_pc`` -- which never passes a
    ``transport_velocity_fn`` -- is unaffected by the field existing.
    """

    guided: bool
    grad_norm: float
    v_norm: float
    applied_norm: float
    f_hat0: float
    oom_fallback: bool
    static_fallback: bool
    transport_v_norm: float = float("nan")


def guided_euler_step(
    x: Float[Tensor, "B d"],
    *,
    t_from: float,
    t_to: float,
    reward: Reward,
    lam: float,
    velocity_fn: VelocityFn,
    guided: bool,
    exact_jacobian: bool = False,
    grad_scaling: Literal["velocity", "static"] = "velocity",
    static_scale: float = 1.0,
    grad_clip_percentile: float | None = None,
    min_v_norm: float = 1e-4,
    g_chunk: int | None = None,
    transport_velocity_fn: VelocityFn | None = None,
) -> tuple[Float[Tensor, "B d"], EulerStepRecord]:
    """One Euler step ``t_from -> t_to`` of the model's flow-matching ODE, optionally reward-guided.

    This is THE single implementation of Phase 3's step: ``flow_guided_sample`` is a loop over it, and
    ``flow_guided_pc_sample`` uses it verbatim as its predictor, which is what makes
    ``corrector_steps=0`` reduce to Phase 3 bitwise.

    Sign, derived once (this repo's own history shows this class of error is silent -- see
    ``flow_guided.py``'s docstring): the Euler step is ``x_next = x - dt*v_guided``. To move ``x_next``
    along ``+g`` (ascending ``r``), we need ``v_guided = v - lam*g_t``, giving
    ``x_next = x - dt*v + dt*lam*g_t``.

    THE IDENTITY-JACOBIAN APPROXIMATION. ``r`` cannot be meaningfully evaluated on a noisy ``x_t``, so it
    is evaluated on the clean estimate ``x_hat_0 = x_t - t*v_theta(x_t, t)``. With
    ``exact_jacobian=False`` (the default, and the standard universal-guidance approximation) the
    gradient is transported back as

        grad_{x_t} r(x_t)  ~=  grad_{x_hat_0} r(x_hat_0),     i.e. d(x_hat_0)/d(x_t) treated as I.

    This is a HEAVY assumption: the true Jacobian is ``I - t * dv_theta/dx_t``, a full ``d x d`` operator
    of the network's own sensitivity, and it is being replaced by the identity. ``exact_jacobian=True``
    drops the approximation entirely at the cost of a backward through ``v_theta`` itself (requires a
    frozen ``velocity_fn.module``; see :func:`require_frozen_module`).

    Args:
        x:          current latents at ``t_from``.
        t_from, t_to: the step's endpoints in diffusers-native t; ``dt = t_from - t_to``.
        reward:     the frozen tilt ``f`` (invariant 1).
        lam:        guidance strength; ignored when ``guided`` is False.
        velocity_fn: ``(x_t, t) -> v_theta``, diffusers-native t.
        guided:     whether to take the guided branch. The caller owns the windowing/``lam != 0``
                    decision, because the two samplers gate it differently (``flow_guided_pc`` also
                    consults ``predictor_guided``).
        exact_jacobian: see above.
        grad_scaling: "velocity" rescales ``grad r`` to ``||v_theta||`` per sample so guidance never
                    overpowers the model's structural constraints; "static" uses ``static_scale``.
        static_scale: the "static" scale, and the fallback whenever ``||v_theta|| < min_v_norm``.
        grad_clip_percentile: see :func:`clip_grad_percentile`.
        min_v_norm: below this, "velocity" scaling is a division by ~0 and falls back to ``static_scale``.
        g_chunk:    gamma-chunk size for :func:`_reward_grad`'s OOM fallback.
        transport_velocity_fn: optional SECOND velocity field used ONLY for the Euler transport, while
                    ``velocity_fn`` keeps supplying ``x_hat_0``, the reward, the gradient and the
                    ``grad_scaling="velocity"`` reference norm. ``None`` (default) is exactly today's
                    arithmetic -- one field for everything -- so ``flow_guided_pc``'s bitwise reduction to
                    ``flow_guided`` is untouched by this parameter existing.

                    THE SPLIT IS THE POINT, not a convenience. Its first user is manual CFG
                    (``generators/cfg.py``): the transport becomes ``v_CFG`` while ``x_hat_0`` stays
                    ``x - t*v_cond``, which buys two things. (i) The reward keeps being evaluated on the
                    model's TRUE conditional denoised estimate, not on a ``w``-extrapolated one that no
                    reference bank covers -- so ``f`` stays comparable across ``w``. (ii) The perturbation
                    ``lam * g_t`` is scaled against ``||v_cond||``, NOT ``||v_CFG||``. That matters: CFG
                    extrapolation inflates the velocity norm, so scaling to the transport field would
                    silently scale the applied gradient UP with ``w`` and confound "a stiffer base field
                    resists the gradient" with "the gradient got bigger". Here ``w`` moves the base field
                    and nothing else.

                    It also makes the CFG path CHEAPER than it looks: the transport field is evaluated
                    under ``no_grad`` and never enters an autograd graph, so ``exact_jacobian=True``
                    still builds and backpropagates exactly ONE network graph, not two.

                    Passing the SAME callable as ``velocity_fn`` is detected by identity and skipped, so
                    a ``w = 1`` CFG wrapper (which :func:`generators.cfg.cfg_velocity_fn` returns as the
                    conditional function itself) costs one forward and is bitwise the unsplit path.

    Returns:
        ``(x_next, EulerStepRecord)``. ``x_next`` is always detached -- memory isolation: the graph must
        never chain in VRAM across steps.
    """
    dt = t_from - t_to
    # Written as an if/else (not a conditional expression) so `transport_fn` narrows to VelocityFn for
    # the type checker on both branches.
    if transport_velocity_fn is None or transport_velocity_fn is velocity_fn:
        transport_fn, split_transport = velocity_fn, False
    else:
        transport_fn, split_transport = transport_velocity_fn, True

    if not guided:
        # Only the transport field is needed here: with no gradient there is no x_hat_0 and no reward, so
        # the conditional field has no second role to play and evaluating it would be a wasted forward.
        # `v_norm` therefore records the field actually integrated, which is what it means on this branch.
        with torch.no_grad():
            v = transport_fn(x, t_from)
        v_norm_unguided = float(v.norm(dim=1).mean())
        record = EulerStepRecord(
            guided=False,
            grad_norm=float("nan"),
            v_norm=v_norm_unguided,
            applied_norm=0.0,
            f_hat0=float("nan"),
            oom_fallback=False,
            static_fallback=False,
            transport_v_norm=v_norm_unguided if split_transport else float("nan"),
        )
        return (x - dt * v).detach(), record

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
        g = clip_grad_percentile(g, grad_clip_percentile)

    grad_norm = g.norm(dim=1, keepdim=True)
    v_norm = v.norm(dim=1, keepdim=True)
    static_fallback = bool((v_norm < min_v_norm).any())
    if grad_scaling == "velocity":
        scale = torch.where(v_norm < min_v_norm, torch.full_like(v_norm, static_scale), v_norm)
        g_t = g / (grad_norm + _EPS) * scale
    else:
        g_t = g / (grad_norm + _EPS) * static_scale

    # The transport field, if a separate one was supplied. Evaluated AFTER the gradient (so the backward
    # has already released its activations) and under no_grad: it is never differentiated, which is why a
    # split transport does not add a second autograd graph. `g_t` above is already scaled against
    # ||velocity_fn||, deliberately -- see the `transport_velocity_fn` docstring.
    transport_v_norm = float("nan")
    if split_transport:
        with torch.no_grad():
            v = transport_fn(x, t_from)
        transport_v_norm = float(v.norm(dim=1).mean())

    v_guided = v - lam * g_t
    x_next = (x - dt * v_guided).detach()  # memory isolation: never chains in VRAM across steps

    with torch.no_grad():
        f_hat0 = float(reward(x_hat0).mean())
    record = EulerStepRecord(
        guided=True,
        grad_norm=float(grad_norm.mean()),
        v_norm=float(v_norm.mean()),
        applied_norm=float((lam * g_t).norm(dim=1).mean()),
        f_hat0=f_hat0,
        oom_fallback=fell_back,
        static_fallback=static_fallback,
        transport_v_norm=transport_v_norm,
    )
    return x_next, record
