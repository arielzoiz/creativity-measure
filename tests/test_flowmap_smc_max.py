"""Tests for the hard-max lookahead aggregation (creativity_measure/flowmap_smc_max.py).

The analytic Gaussian models, the toy reward and the ``_run`` harness are imported from
`test_flowmap_smc` rather than restated: this module changes exactly one function, and the tests
should differ in exactly the same place. (``tests/`` has no ``__init__.py``, so pytest's default
prepend import mode puts it on ``sys.path`` and the sibling module imports by name.)

Three things are load-bearing here:

* **the target is untouched** -- the terminal ``V_N = lambda·f(x_1)`` must stay raw, because that is
  what pins ``q_lambda``; the aggregator is only allowed to move *intermediate* twists;
* **the two exact reductions** -- ``mc_samples = 1`` and ``lam = 0`` must reproduce `flowmap_smc`
  bit-for-bit, which is what proves the override is wired to the aggregation and to nothing else;
* **the patch is always restored**, including when the sampler raises, or a notebook's deadline
  callback would silently leave the base sampler mis-aggregating for the rest of the session.
"""

import math

import pytest
import torch

from creativity_measure.flowmap_smc import LinearSchedule, _soft_value, flowmap_smc_sample
from creativity_measure import flowmap_smc
from creativity_measure.flowmap_smc_max import (
    _max_value,
    _override_value_agg,
    flowmap_smc_max_sample,
)
from test_flowmap_smc import GaussianFlowMap, _gaussian_score_fn, _reward, dtype

FULL_WINDOW = (0.0, 1.0)


def _run_max(reward, lam: float, **kw):
    """`flowmap_smc_max_sample` with the same toy defaults `test_flowmap_smc._run` uses."""
    schedule = kw.pop("schedule", LinearSchedule())
    flow_map = kw.pop("flow_map", None) or GaussianFlowMap(schedule)
    kw.setdefault("score_fn", _gaussian_score_fn())
    kw.setdefault("n_steps", 16)
    kw.setdefault("mc_samples", 4)
    kw.setdefault("seed", 7)
    return flowmap_smc_max_sample(
        reward, lam, kw.pop("n_particles", 8),
        flow_map=flow_map, schedule=schedule, **kw,
    )


def _run_soft(reward, lam: float, **kw):
    """The unpatched sampler, same defaults -- the paired baseline for every comparison below."""
    schedule = kw.pop("schedule", LinearSchedule())
    flow_map = kw.pop("flow_map", None) or GaussianFlowMap(schedule)
    kw.setdefault("score_fn", _gaussian_score_fn())
    kw.setdefault("n_steps", 16)
    kw.setdefault("mc_samples", 4)
    kw.setdefault("seed", 7)
    return flowmap_smc_sample(
        reward, lam, kw.pop("n_particles", 8),
        flow_map=flow_map, schedule=schedule, **kw,
    )


# ---------------------------------------------------------------------------------------------------
# 1. The aggregator itself
# ---------------------------------------------------------------------------------------------------

def test_max_value_is_the_hard_max_and_dominates_the_soft_value():
    """``max_k v_k`` exactly, and ``>= logmeanexp_k v_k`` elementwise -- a strictly sharper twist."""
    gen = torch.Generator().manual_seed(0)
    v = torch.randn((5, 7), generator=gen, dtype=dtype)

    assert torch.equal(_max_value(v), v.max(dim=1).values)
    assert torch.all(_max_value(v) >= _soft_value(v))
    # Strict wherever the K candidates are not all equal, which they are not for a random draw.
    assert torch.all(_max_value(v) > _soft_value(v))


def test_the_two_aggregations_agree_at_k_one():
    """A mean of one exponential is that exponential, so ``logmeanexp`` IS the max at ``K = 1``."""
    gen = torch.Generator().manual_seed(1)
    v = torch.randn((6, 1), generator=gen, dtype=dtype)
    assert torch.allclose(_max_value(v), _soft_value(v), atol=0.0, rtol=0.0)


def test_max_is_the_zero_temperature_limit_of_the_soft_value():
    """``(1/tau)·logmeanexp(tau·v) -> max v - log(K)/tau``: the two are one family, not two mechanisms.

    The residual ``log(K)/tau`` is `_soft_value`'s ``- log K``, which the max does not carry. It is a
    constant across particles and therefore inert -- see the module docstring -- but it is a real
    offset, so the limit is asserted with it rather than absorbed into a loose tolerance.
    """
    gen = torch.Generator().manual_seed(2)
    v = torch.randn((4, 8), generator=gen, dtype=dtype)
    for tau in (50.0, 400.0):
        tempered = _soft_value(tau * v) / tau
        assert torch.allclose(tempered + math.log(v.shape[1]) / tau, _max_value(v), atol=1e-9)


# ---------------------------------------------------------------------------------------------------
# 2. The override is scoped
# ---------------------------------------------------------------------------------------------------

def test_the_patch_is_restored_after_a_normal_call():
    reward = _reward()
    before = flowmap_smc._soft_value
    _run_max(reward, lam=1.5, n_particles=4, n_steps=4)
    assert flowmap_smc._soft_value is before


def test_the_patch_is_restored_when_the_sampler_raises():
    """A notebook's deadline callback raises out of the sampler; the base one must not stay patched."""
    reward = _reward()
    before = flowmap_smc._soft_value

    def boom(i, snap, partial):
        raise RuntimeError("deadline")

    with pytest.raises(RuntimeError, match="deadline"):
        _run_max(reward, lam=1.5, n_particles=4, n_steps=4, on_step=boom)
    assert flowmap_smc._soft_value is before


def test_the_override_context_manager_restores_on_its_own():
    before = flowmap_smc._soft_value
    with _override_value_agg(_max_value):
        assert flowmap_smc._soft_value is _max_value
    assert flowmap_smc._soft_value is before


# ---------------------------------------------------------------------------------------------------
# 3. Exact reductions -- these prove the override touches the aggregation and nothing else
# ---------------------------------------------------------------------------------------------------

def test_k_one_reproduces_the_soft_sampler_exactly():
    reward = _reward()
    kw = dict(n_particles=8, mc_samples=1, guid_window=FULL_WINDOW, seed=11)
    assert torch.equal(_run_max(reward, lam=2.0, **kw).X, _run_soft(reward, lam=2.0, **kw).X)


def test_lambda_zero_reproduces_the_soft_sampler_exactly():
    """At ``lam = 0`` every ``v_k`` is 0, so both aggregations give 0 and this is the base process."""
    reward = _reward()
    kw = dict(n_particles=8, mc_samples=4, guid_window=FULL_WINDOW, seed=11)
    res_max, res_soft = _run_max(reward, lam=0.0, **kw), _run_soft(reward, lam=0.0, **kw)
    assert torch.equal(res_max.X, res_soft.X)
    assert torch.equal(res_max.logw, res_soft.logw)


# ---------------------------------------------------------------------------------------------------
# 4. The target is unchanged: the terminal update never goes through the aggregator
# ---------------------------------------------------------------------------------------------------

def test_terminal_potential_is_still_raw_lambda_f():
    """``V_N = lambda·f(x_1)`` -- the mandatory update that pins ``q_lambda``, on a K = 1 code path."""
    reward, lam = _reward(), 2.5
    res = _run_max(reward, lam=lam, n_particles=6, guid_window=FULL_WINDOW)
    assert torch.allclose(res.V_history[-1], lam * reward(res.X))


def test_intermediate_potential_is_lambda_times_the_per_particle_max():
    """``V_history / lam`` is exactly ``max_k r_k`` on a guided step -- the diagnostic claim made in
    the module docstring, so no separate recording is needed to read the max back out."""
    reward, lam = _reward(), 2.5
    res = _run_max(reward, lam=lam, n_particles=6, mc_samples=4, guid_window=FULL_WINDOW,
                   record_r_k=True)
    n = next(i for i, r in enumerate(res.r_k_history) if r is not None)
    r_k = res.r_k_history[n]
    assert r_k is not None
    assert torch.allclose(res.V_history[n], lam * r_k.max(dim=1).values)


# ---------------------------------------------------------------------------------------------------
# 5. The first guided step: a paired comparison before the two runs can diverge
# ---------------------------------------------------------------------------------------------------

def test_the_first_guided_potential_dominates_the_soft_one():
    """Up to the first guided step the two runs are identical (same seeds, no resampling on a flat
    ``U``), so their first potentials are computed on the SAME particles and the inequality is a
    paired comparison rather than a statement about two different clouds. After that they diverge --
    which is the point, and is why nothing later is compared this way."""
    reward, lam = _reward(), 3.0
    kw = dict(n_particles=8, mc_samples=8, guid_window=FULL_WINDOW, seed=5)
    res_max, res_soft = _run_max(reward, lam=lam, **kw), _run_soft(reward, lam=lam, **kw)

    assert res_max.guided_history[0] and res_soft.guided_history[0]
    assert torch.all(res_max.V_history[0] >= res_soft.V_history[0])
    assert torch.any(res_max.V_history[0] > res_soft.V_history[0])


# ---------------------------------------------------------------------------------------------------
# 6. Determinism, and the diagnostics still arrive
# ---------------------------------------------------------------------------------------------------

def test_same_seed_same_result():
    reward = _reward()
    kw = dict(n_particles=6, guid_window=FULL_WINDOW, seed=3)
    assert torch.equal(_run_max(reward, lam=2.0, **kw).X, _run_max(reward, lam=2.0, **kw).X)


def test_the_module_is_a_complete_notebook_surface():
    """A notebook must be able to import everything it needs from `flowmap_smc_max` alone, and the
    base-process names must be the SAME OBJECTS as `flowmap_smc`'s -- not copies.

    The reference files key ``p`` on ``ddpm_step.__name__``, so a duplicate would pass
    ``_assert_same_p`` while defining a different process. Identity is the only check that catches it.
    """
    import creativity_measure.flowmap_smc_max as fmm

    # Exactly the flowmap names the k-sweep notebook imports, plus the sampler.
    for name in ("LinearSchedule", "ddpm_step", "flow_map_step", "flowmap_smc_max_sample"):
        assert hasattr(fmm, name), name
        assert name in fmm.__all__, name

    assert fmm.ddpm_step is flowmap_smc.ddpm_step
    assert fmm.flow_map_step is flowmap_smc.flow_map_step
    assert fmm.LinearSchedule is flowmap_smc.LinearSchedule
    assert fmm.FlowMapSMCResult is flowmap_smc.FlowMapSMCResult


def test_diagnostics_are_complete():
    reward = _reward()
    res = _run_max(reward, lam=2.0, n_particles=6, n_steps=8, guid_window=FULL_WINDOW,
                   record_r_k=True, project_endpoint=True)
    n = 8
    for name in ("ess_history", "resampled_history", "uniq_history", "V_history", "t_history",
                 "guided_history", "eqf_history", "U_pre_history", "r_k_history",
                 "f_proj_history"):
        assert len(getattr(res, name)) == n, name
    assert all(v == v for v in res.eqf_history)  # not nan: project_endpoint is on
