"""Manual (double-forward) classifier-free guidance as a ``VelocityFn`` combinator.

Backend-agnostic: this module imports only the ``VelocityFn`` contract (``creativity_measure/_types.py``)
and composes two of them. It lives under ``generators/`` because it is velocity-field plumbing, not a
sampling algorithm -- nothing in ``samplers/`` needs to know CFG exists (see CLAUDE.md, "Adding a New
Flow-Matching Backend": a new velocity contract is a different kind of contribution than a new sampler).

WHAT THIS IS FOR. FLUX.1-dev is guidance-DISTILLED: it takes a ``guidance`` scalar as a model *input*
(this repo's baseline is ``GUIDANCE = 1.5``, baked into every reference bank, reward and base velocity
on the FLUX path) and needs no second forward pass. This module adds the *traditional* two-forward CFG
extrapolation ON TOP of that embedded baseline,

    v_CFG(x_t, t) = v(x_t, t | null) + w * (v(x_t, t | c) - v(x_t, t | null)),

to test whether the sharper, higher-contrast field it produces is a *stiffer* structure that better
resists the off-manifold degradation of a reward gradient. ``w = 1`` collapses the null terms and is the
single-forward baseline.

TWO CAVEATS, both load-bearing for how a result here should be read:

1. ``w = 1`` is NOT bitwise the baseline if computed as written: ``v_u + 1.0*(v_c - v_u) != v_c`` in
   floating point. :func:`cfg_velocity_fn` therefore returns ``velocity_fn_cond`` ITSELF at ``w == 1.0``,
   so the reduction is exact and free (one forward, not two) -- the same standard
   ``flow_guided_pc_sample(corrector_steps=0)`` is held to.
2. FLUX.1-dev's empty-prompt branch is not a trained unconditional model -- guidance distillation removed
   the need for one, so there is no guarantee it behaves like a proper null. Large ``w`` on a distilled
   checkpoint is known to over-saturate. ``t_window`` exists for exactly this: it restricts the
   extrapolation to an interval of t and falls back to the plain conditional velocity outside it, which
   is the usual workaround when "true CFG" is applied to a distilled flow model.

TIME CONVENTION: diffusers-native t -- ``t = 1`` pure noise, ``t = 0`` clean data -- as ``VelocityFn``
declares. The OPPOSITE polarity to ``generators/flux_flowmap.py``'s own convention; this repo has a
history of silent sign errors exactly here (see ``samplers/flow_guided.py``'s docstring).
"""
from __future__ import annotations

from typing import cast

from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import VelocityFn


def cfg_velocity_fn(
    velocity_fn_cond: VelocityFn,
    velocity_fn_uncond: VelocityFn,
    w: float,
    *,
    t_window: tuple[float, float] = (1.0, 0.0),
) -> VelocityFn:
    """Compose a conditional and an unconditional ``VelocityFn`` into the CFG-extrapolated field.

    The two networks are evaluated STRICTLY SEQUENTIALLY, never as a concatenated batch. On a 12 B model
    at ``d = 65536`` the batch-concatenated form doubles peak activation memory for no throughput gain
    (the GPU is already saturated by one row), and the whole FLUX path in this repo runs at batch 1.

    Args:
        velocity_fn_cond:   ``(x_t, t) -> v`` under the real prompt. Its ``.module`` (if any) is
                            forwarded onto the returned closure, so the result still satisfies
                            ``GuidableVelocityFn`` and can be handed to an ``exact_jacobian=True`` path.
        velocity_fn_uncond: the same, under the null/empty prompt. MUST be built against the same
                            checkpoint and the same embedded ``guidance`` baseline as ``..._cond``, or
                            the difference term mixes two different fields.
        w:                  the CFG weight. ``w == 1.0`` returns ``velocity_fn_cond`` unchanged (see the
                            module docstring) -- an exact, single-forward reduction to the baseline.
        t_window:           ``(t_hi, t_lo)``; the extrapolation is applied iff ``t_lo <= t <= t_hi``,
                            and the plain conditional velocity is returned outside it. Defaults to the
                            whole trajectory. Bounds are given high-first to match
                            ``flow_guided_sample``'s own ``(t_start, t_end)`` ordering, since t runs
                            DOWNWARD from 1 to 0.

    Returns:
        A ``VelocityFn`` (in practice a ``GuidableVelocityFn`` whenever ``velocity_fn_cond`` was one).
    """
    if w == 1.0:
        return velocity_fn_cond

    t_hi, t_lo = t_window
    if t_hi < t_lo:
        raise ValueError(
            f"t_window must be (t_hi, t_lo) with t_hi >= t_lo -- t runs downward from 1 to 0; got {t_window}"
        )

    def velocity(x_t: Float[Tensor, "B d"], t: float) -> Float[Tensor, "B d"]:
        if not (t_lo <= t <= t_hi):
            return velocity_fn_cond(x_t, t)
        v_uncond = velocity_fn_uncond(x_t, t)      # sequential, NOT batched -- see docstring
        v_cond = velocity_fn_cond(x_t, t)
        return v_uncond + w * (v_cond - v_uncond)

    module = getattr(velocity_fn_cond, "module", None)
    if module is not None:
        velocity.module = module  # type: ignore[attr-defined]  # keeps GuidableVelocityFn conformance
    return cast(VelocityFn, velocity)
