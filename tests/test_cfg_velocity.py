"""Tests for manual CFG (`creativity_measure/generators/cfg.py`) and the split-transport Euler step.

CPU-only. The CFG combinator itself is pure arithmetic over two `VelocityFn`s, so most of this file uses
ANALYTIC velocity functions rather than a transformer -- that is what makes the extrapolation algebra and
the `w = 1` reduction checkable exactly instead of to a tolerance. The end-to-end wiring through
`flow_guided_sample` then runs on the same tiny real FluxTransformer2DModel as
`tests/test_flow_guided.py`, since the thing worth testing there is the plumbing, not the arithmetic.

The load-bearing claims, each with a test below:
  1. `w = 1` returns the conditional function ITSELF, so the baseline reduction is exact and single-forward.
  2. `transport_velocity_fn=None` leaves `guided_euler_step` bitwise unchanged (the guard on
     `flow_guided_pc`'s own bitwise reduction, which passes no transport field).
  3. A split transport changes WHERE the step lands but not the gradient: `x_hat_0`, the reward and the
     applied-perturbation norm come from `velocity_fn` alone. This is the property that keeps `f`
     comparable across `w` and stops `w` from rescaling the guidance -- see cfg.py's module docstring.
"""
from typing import Any

import pytest
import torch
from torch import Tensor

from creativity_measure.generators.cfg import cfg_velocity_fn
from creativity_measure.samplers.flow_guided_common import guided_euler_step
from creativity_measure.tilt import Reward

diffusers = pytest.importorskip("diffusers")

from tests.test_flow_guided import (  # noqa: E402  # reuse the tiny-model recipe verbatim
    C, D, H, W, _build_reward, _build_velocity_fn, _tiny_transformer,
)
from creativity_measure.samplers.flow_guided import flow_guided_sample  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# Analytic velocity fields -- exact arithmetic, no model
# ---------------------------------------------------------------------------------------------------

def _const_velocity(value: float):
    """``(x_t, t) -> value * ones_like(x_t)``, plus a call counter so "how many forwards" is testable."""
    def velocity(x_t: Tensor, t: float) -> Tensor:
        velocity.calls += 1          # type: ignore[attr-defined]
        velocity.ts.append(t)        # type: ignore[attr-defined]
        return torch.full_like(x_t, value)
    velocity.calls = 0               # type: ignore[attr-defined]
    velocity.ts = []                 # type: ignore[attr-defined]
    return velocity


def test_w_one_returns_the_conditional_function_itself() -> None:
    """Not "equal to" -- the SAME OBJECT, which is what makes the reduction exact and single-forward.

    Computed as written, `v_u + 1.0*(v_c - v_u)` is not bitwise `v_c` in floating point, and
    `guided_euler_step` detects the identity to skip the redundant forward entirely.
    """
    v_cond, v_uncond = _const_velocity(2.0), _const_velocity(5.0)
    assert cfg_velocity_fn(v_cond, v_uncond, 1.0) is v_cond
    cfg = cfg_velocity_fn(v_cond, v_uncond, 1.0)
    x = torch.randn(2, 8)
    assert torch.equal(cfg(x, 0.5), v_cond(x, 0.5))
    assert v_uncond.calls == 0, "the unconditional branch must never be evaluated at w=1"  # type: ignore[attr-defined]


@pytest.mark.parametrize("w", [0.0, 1.5, 2.0, 3.0, -1.0])
def test_extrapolation_matches_the_closed_form(w: float) -> None:
    v_cond, v_uncond = _const_velocity(2.0), _const_velocity(5.0)
    cfg = cfg_velocity_fn(v_cond, v_uncond, w)
    x = torch.randn(3, 8)
    expected = torch.full_like(x, 5.0 + w * (2.0 - 5.0))
    assert torch.allclose(cfg(x, 0.7), expected, atol=0.0, rtol=0.0)


def test_both_fields_are_evaluated_once_per_call_uncond_first() -> None:
    """Sequential, one row each -- never a concatenated batch (peak activation memory on a 12 B model)."""
    order: list[str] = []

    def v_cond(x_t: Tensor, t: float) -> Tensor:
        order.append("cond")
        return torch.full_like(x_t, 2.0)

    def v_uncond(x_t: Tensor, t: float) -> Tensor:
        order.append("uncond")
        return torch.full_like(x_t, 5.0)

    cfg = cfg_velocity_fn(v_cond, v_uncond, 2.0)
    cfg(torch.randn(1, 8), 0.5)
    assert order == ["uncond", "cond"], "exactly one call each, unconditional first"


def test_t_window_falls_back_to_the_conditional_field_outside_it() -> None:
    v_cond, v_uncond = _const_velocity(2.0), _const_velocity(5.0)
    cfg = cfg_velocity_fn(v_cond, v_uncond, 3.0, t_window=(0.8, 0.4))
    x = torch.randn(1, 8)
    inside = torch.full_like(x, 5.0 + 3.0 * (2.0 - 5.0))
    assert torch.equal(cfg(x, 0.8), inside)             # inclusive at both bounds
    assert torch.equal(cfg(x, 0.4), inside)
    assert torch.equal(cfg(x, 0.6), inside)
    assert torch.equal(cfg(x, 0.9), torch.full_like(x, 2.0))
    assert torch.equal(cfg(x, 0.2), torch.full_like(x, 2.0))


def test_inverted_t_window_is_rejected() -> None:
    v_cond, v_uncond = _const_velocity(2.0), _const_velocity(5.0)
    with pytest.raises(ValueError, match="t_hi >= t_lo"):
        cfg_velocity_fn(v_cond, v_uncond, 2.0, t_window=(0.2, 0.8))


def test_module_attribute_is_forwarded_so_exact_jacobian_can_still_verify_the_freeze() -> None:
    transformer = _tiny_transformer()
    transformer.requires_grad_(False)
    v_cond = _build_velocity_fn(transformer, differentiable=True)
    v_uncond = _build_velocity_fn(transformer, differentiable=True)
    cfg = cfg_velocity_fn(v_cond, v_uncond, 2.0)
    assert getattr(cfg, "module", None) is transformer


# ---------------------------------------------------------------------------------------------------
# The split-transport Euler step
# ---------------------------------------------------------------------------------------------------

def _euler_kwargs(reward: Reward, velocity_fn: Any, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        t_from=0.8, t_to=0.5, reward=reward, lam=1.0, velocity_fn=velocity_fn, guided=True,
        exact_jacobian=False,
    )
    base.update(kw)
    return base


@pytest.mark.parametrize("guided", [True, False])
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_transport_none_is_bitwise_the_unsplit_step(guided: bool, lam: float) -> None:
    """The guard on `flow_guided_pc`'s bitwise reduction: it never passes a transport field, so adding
    the parameter must not perturb a single float on the default path."""
    transformer = _tiny_transformer()
    v = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    x = torch.randn(2, D, generator=torch.Generator().manual_seed(3))
    kw = _euler_kwargs(reward, v, lam=lam, guided=guided)
    a, rec_a = guided_euler_step(x, **kw)
    b, rec_b = guided_euler_step(x, **kw, transport_velocity_fn=None)
    assert torch.equal(a, b)
    # `==` on floats would compare nan to nan and fail: f_hat0/grad_norm are nan by design on an
    # unguided step, and "both nan" is exactly the agreement being asserted.
    for name in ("v_norm", "applied_norm", "f_hat0", "grad_norm", "transport_v_norm"):
        va, vb = getattr(rec_a, name), getattr(rec_b, name)
        assert va == vb or (va != va and vb != vb), f"{name}: {va} vs {vb}"


def test_passing_the_same_callable_as_transport_is_detected_and_skipped() -> None:
    """A `w = 1` CFG wrapper IS the conditional function, so this is the path it takes: one forward,
    bitwise identical to not splitting at all."""
    transformer = _tiny_transformer()
    v = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    x = torch.randn(2, D, generator=torch.Generator().manual_seed(4))
    kw = _euler_kwargs(reward, v)
    a, _ = guided_euler_step(x, **kw)
    b, rec_b = guided_euler_step(x, **kw, transport_velocity_fn=v)
    assert torch.equal(a, b)
    assert rec_b.transport_v_norm != rec_b.transport_v_norm, "identity-skip must leave transport_v_norm nan"


def test_split_transport_moves_the_step_but_not_the_gradient() -> None:
    """THE central property of the design (cfg.py's docstring): `x_hat_0`, the reward and the applied
    perturbation all come from `velocity_fn`; only the transported base field changes.

    Verified by constructing the split step's result from the unsplit one in closed form. With
    `x_next = x - dt*(v_transport - lam*g_t)`, swapping only the transport field must give exactly
    `x_next_split - x_next_plain = -dt*(v_transport - v_cond)`, and every recorded gradient-side
    diagnostic must be unchanged.
    """
    transformer = _tiny_transformer()
    v_cond = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    x = torch.randn(2, D, generator=torch.Generator().manual_seed(5))
    dt = 0.8 - 0.5

    v_transport = _const_velocity(0.25)
    kw = _euler_kwargs(reward, v_cond)
    plain, rec_plain = guided_euler_step(x, **kw)
    split, rec_split = guided_euler_step(x, **kw, transport_velocity_fn=v_transport)

    with torch.no_grad():
        v_c = v_cond(x, 0.8)
        v_tr = torch.full_like(x, 0.25)
    assert torch.allclose(split - plain, -dt * (v_tr - v_c), atol=1e-5)

    # gradient-side diagnostics identical: the perturbation was scaled to ||v_cond||, not ||v_transport||
    assert rec_split.v_norm == rec_plain.v_norm
    assert rec_split.applied_norm == rec_plain.applied_norm
    assert rec_split.grad_norm == rec_plain.grad_norm
    assert rec_split.f_hat0 == rec_plain.f_hat0
    assert rec_split.transport_v_norm == pytest.approx(float(v_tr.norm(dim=1).mean()))


def test_unguided_split_step_transports_along_the_transport_field_only() -> None:
    """At lam = 0 there is no gradient and no x_hat_0, so the conditional field has no second role --
    the step is a plain (CFG) sample and only one forward is spent."""
    transformer = _tiny_transformer()
    v_cond = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    x = torch.randn(1, D, generator=torch.Generator().manual_seed(6))
    v_transport = _const_velocity(0.25)
    out, rec = guided_euler_step(
        x, **_euler_kwargs(reward, v_cond, guided=False, lam=0.0), transport_velocity_fn=v_transport,
    )
    assert torch.allclose(out, x - (0.8 - 0.5) * torch.full_like(x, 0.25))
    assert v_transport.calls == 1  # type: ignore[attr-defined]
    assert rec.v_norm == rec.transport_v_norm


# ---------------------------------------------------------------------------------------------------
# End-to-end through flow_guided_sample
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("exact_jacobian", [False, True])
def test_flow_guided_sample_threads_the_transport_field(exact_jacobian: bool) -> None:
    transformer = _tiny_transformer()
    v_cond = _build_velocity_fn(transformer, differentiable=True)
    v_uncond = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    kw: dict[str, Any] = dict(
        velocity_fn=v_cond, n_steps=3, shift=1.0, exact_jacobian=exact_jacobian, seed=0,
    )
    plain = flow_guided_sample(reward, 1.0, 2, **kw)
    cfg = flow_guided_sample(
        reward, 1.0, 2, transport_velocity_fn=cfg_velocity_fn(v_cond, v_uncond, 2.0), **kw,
    )
    assert cfg.X.shape == plain.X.shape
    assert torch.isfinite(cfg.X).all()
    # Same transformer for both branches, so v_cond == v_uncond and v_CFG == v_cond identically: the CFG
    # run must land in the SAME place. That is the cleanest available check that the split wiring is
    # arithmetically faithful rather than merely plausible -- a real null prompt is a GPU-scale question.
    assert torch.allclose(cfg.X, plain.X, atol=1e-5)
    assert len(cfg.transport_v_norm_history) == 3
    assert all(n == n for n in cfg.transport_v_norm_history), "transport norms must be recorded, not nan"
    assert all(n != n for n in plain.transport_v_norm_history), "unsplit runs record nan here"


def test_w_one_wrapper_reduces_flow_guided_sample_bitwise() -> None:
    """End-to-end version of claim 1: w=1 is the Phase 3 baseline exactly, on the identity-skip path."""
    transformer = _tiny_transformer()
    v_cond = _build_velocity_fn(transformer, differentiable=True)
    v_uncond = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    kw: dict[str, Any] = dict(
        velocity_fn=v_cond, n_steps=3, shift=1.0, exact_jacobian=False, seed=0,
    )
    plain = flow_guided_sample(reward, 1.0, 2, **kw)
    w1 = flow_guided_sample(
        reward, 1.0, 2, transport_velocity_fn=cfg_velocity_fn(v_cond, v_uncond, 1.0), **kw,
    )
    assert torch.equal(w1.X, plain.X)
