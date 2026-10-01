"""Algorithm 3 with a **resampling window** -- selection is switched off early in the trajectory.

Everything about the sampler is `flowmap_smc`: the schedules, both base transitions, the renoise, the
score correction, the value aggregation, the resampling primitive, the terminal update, the result
type. This module changes exactly one predicate -- *whether a step is allowed to resample* -- and adds
``resample_window`` to say where.

WHY
---
Two independent measurements point at the same stretch of trajectory:

1. **The potential's ordering is wrong early.** With estimator noise removed (both potentials at
   ``K = 2048`` on the toy), the renoise lookahead's ranking agrees with the true value function at
   Spearman **0.70 / 0.82 / 0.95** for ``t <= 0.25`` at ``lambda = 16 / 8 / 4`` -- and **0.99+** after.
   Early on, the value function genuinely cannot know much: it is an expectation over a future that is
   still mostly noise. (`notebooks/flowmap_smc_replay/a0b_lookahead_bias_gate.py`.)
2. **That is exactly where the cloud dies.** The FLUX runs collapse at **step 3** (``t = 0.1875``),
   inside that window, and ``uniq/M`` is monotone non-increasing -- systematic resampling only ever
   destroys lineages and nothing regenerates them. Every failed cell of the K sweep failed this way.

So the ensemble spends its diversity first, on the least trustworthy ordering it will ever compute.
Deferring resampling past ``t = 0.25`` keeps the lineages until the potential is worth acting on.

**This is a schedule change, not a target change.** Weights are never discarded: on a step that does
not resample, ``U`` simply keeps accumulating, and between resamples ``U_n = V_n - V_{last}`` is a
LONG-BASELINE difference in which the intermediate potentials telescope away. That is strictly more
information than the single-step increment ``_resample`` forces selection onto by zeroing ``U``. In the
limit ``resample_window`` empty this is exact importance sampling -- which is the arm that rises
monotonically with lambda (1.0116 / 1.0183 / 1.0203, job 782816) where SMC falls. This module
interpolates between that arm and today's.

WHY A LOOP COPY AND NOT A PATCH
-------------------------------
`flowmap_smc_max` and `flowmap_smc_mean` swap a module global because they change one *function*.
The resampling decision is not a function -- it is the inline predicate
``do_resample = (ess < ess_threshold * M) and (not is_final or final_resample)``. The tempting patch is
to wrap `_ess_from_logw` (it is called exactly once per step, so a call counter maps 1:1 to the step
index) and return ``M`` inside the deferred region. **Do not**: the same value is what
``ess_history`` records, so a faked ESS silently corrupts the run's central diagnostic. A copied loop
is honest and, pinned as below, safe.

**The copy is pinned by a bit-identity test.** At ``resample_window=(0.0, 1.0)`` this function must
reproduce `flowmap_smc_sample` exactly -- ``X``, ``logw`` and every diagnostic list -- at a real tilt,
with resampling on (``tests/test_flowmap_smc_defer.py``). That converts the duplication from a drift
risk into an invariant: whoever changes `flowmap_smc`'s bookkeeping gets a failing test rather than two
quietly diverging samplers.

Every helper below is IMPORTED, never re-defined. The reference files record the base process by *name*
(``BASE_PROCESS["inside_step"] = ddpm_step.__name__``), so a copied `ddpm_step` would pass
``_assert_same_p`` while being a different function -- silently invalidating a reference set against
the exact check written to catch that.

One import surface, as for the other variants::

    from creativity_measure.samplers.flowmap_smc_defer import (
        LinearSchedule, ddpm_step, flow_map_step, flowmap_smc_defer_sample,
    )

Not exported from ``creativity_measure/__init__.py``.
"""

import math

import torch
from jaxtyping import Float, Int
from torch import Tensor

from creativity_measure._types import FlowMap, Schedule, ScoreFn, TransitionStep
from creativity_measure.samplers.flowmap_smc import (
    SCHEDULES,
    BaseSchedule,
    FlowMapSMCResult,
    LinearSchedule,
    StepCallback,
    StepSnapshot,
    _antithetic_noise,
    _check_window,
    _in_window,
    _log_likelihood,
    _renoise,
    _resample,
    _score_at,
    _score_correction,
    _soft_value,
    _stats,
    _t_prime,
    _uniform_ts,
    _zscore,
    ddpm_step,
    flow_map_step,
)
from creativity_measure.samplers.smc_common import _ess_from_logw
from creativity_measure.tilt import Reward

__all__ = [
    "flowmap_smc_defer_sample",
    # re-exported from `flowmap_smc` -- SAME OBJECTS, not copies.
    "SCHEDULES",
    "BaseSchedule",
    "FlowMapSMCResult",
    "LinearSchedule",
    "StepCallback",
    "StepSnapshot",
    "ddpm_step",
    "flow_map_step",
]


def flowmap_smc_defer_sample(
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
    resample_window: tuple[float, float] = (0.0, 1.0),
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
    """`flowmap_smc.flowmap_smc_sample` with resampling restricted to ``resample_window``.

    Every argument other than ``resample_window`` has exactly the meaning it has in
    `flowmap_smc.flowmap_smc_sample`; see that docstring, which is the reference for all of them.

    Args:
        resample_window: **where selection is allowed**, keyed on the time each step lands on
            (``ts[n+1]``), the same convention as ``guid_window``. ``(0.0, 1.0)`` is today's behaviour
            and is asserted bit-identical to the base sampler. ``(0.25, 1.0)`` defers selection past
            the stretch where the potential's ordering is measurably untrustworthy. Outside the window
            the ESS is still computed and recorded -- only the resampling is suppressed, so ``U`` keeps
            accumulating and the diagnostics stay honest.

    Returns:
        :class:`flowmap_smc.FlowMapSMCResult` -- same fields, same meanings. ``resampled_history`` is
        ``False`` on every step outside the window, and ``resample_idx_history`` is ``None`` there.

    Note:
        The terminal step is governed by ``final_resample`` as before, and additionally by the window:
        a terminal resample happens only if both allow it. Deferring resampling raises ``sd(logw)`` at
        ``t = 1`` -- that is the point, since the weight is then carrying what selection would have
        discarded -- so read ``logw`` (or pass ``final_resample=True``) rather than assuming the
        returned ensemble is equally weighted.
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

    stoch_flags = [_in_window(ts[n], stoch_window) for n in range(n_steps)]
    guid_flags = [_in_window(ts[n + 1], guid_window) for n in range(n_steps)]
    # Keyed on the step's TARGET time, like `guid_window`: resampling happens after the step has
    # landed, so the time that matters is where it landed.
    resample_flags = [_in_window(ts[n + 1], resample_window) for n in range(n_steps)]
    _check_window("stoch_window", stoch_window, stoch_flags)
    _check_window("guid_window", guid_window, guid_flags)
    _check_window("resample_window", resample_window, resample_flags)
    if any(flag and ts[n + 1] <= 0.0 for n, flag in enumerate(guid_flags)):
        raise ValueError("a guided step lands on t = 0, where the lookahead level is undefined")

    device, dtype = reward.x_refs.device, reward.x_refs.dtype
    d = int(reward.x_refs.shape[1])

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
            V_next = V_prev

        U = U + V_next - V_prev
        V_prev = V_next

        # --- (3b) projected endpoint, BEFORE resampling ---------------------------------------------
        if project_endpoint and f_proj is None:
            f_proj = reward(flow_map.map(x, t_next, 1.0)).detach().clone()
        U_pre = U.detach().clone()
        eqf = (float((torch.softmax(U_pre, dim=0) * f_proj).sum())
               if f_proj is not None else math.nan)

        # --- (4) ESS-triggered resampling, GATED BY THE WINDOW ---------------------------------------
        # The ESS is computed and recorded either way: suppressing the *record* as well would make a
        # deferred run unreadable exactly where the decision to defer has to be justified.
        ess = _ess_from_logw(U)
        do_resample = (
            resample_flags[n]
            and (ess < ess_threshold * M)
            and (not is_final or final_resample)
        )
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
                on_step(n, snap, result)

        if verbose:
            kind = "stoch" if stoch_flags[n] else "det  "
            twist = f"guided tp={t_p:.4f}" if guided else "unguided       "
            gate = "" if resample_flags[n] else "  [resampling deferred]"
            print(
                f"  step {n + 1}/{n_steps}  t={t:.4f}->{t_next:.4f}  {kind}  {twist}  "
                f"ESS/M={ess / M:.2f}  resampled={do_resample}  "
                f"uniq/M={result.uniq_history[-1]:.2f}  f mean={f_stats[0]:.4f}{gate}"
            )

    result.X = x
    result.logw = U
    return result
