"""Tests for the plain-mean lookahead aggregation (creativity_measure/flowmap_smc_mean.py).

Mirrors `test_flowmap_smc_max` deliberately: the two modules change the same one function at opposite
ends of the ``tau`` interpolation, so they should be tested in the same places and any asymmetry
between the files is a real asymmetry in the code.

Load-bearing here:

* **the target is untouched** -- the terminal ``V_N = lambda·f(x_1)`` stays raw whatever the
  aggregator does, because that is what pins ``q_lambda``;
* **the two exact reductions** (``mc_samples = 1``, ``lam = 0``) reproduce `flowmap_smc` bit-for-bit,
  proving the override is wired to the aggregation and nothing else;
* **the Jensen ordering** ``mean <= soft <= max``, which is the whole premise of the module: the gap
  between mean and soft is exactly the convexity term it exists to delete.
"""

import math

import pytest
import torch

from creativity_measure.samplers import flowmap_smc
from creativity_measure.samplers.flowmap_smc import LinearSchedule, _soft_value, flowmap_smc_sample
from creativity_measure.samplers.flowmap_smc_max import _max_value
from creativity_measure.samplers.flowmap_smc_mean import (
    _mean_value,
    flowmap_smc_mean_sample,
)
from test_flowmap_smc import GaussianFlowMap, _gaussian_score_fn, _reward, dtype

FULL_WINDOW = (0.0, 1.0)


def _run_mean(reward, lam: float, **kw):
    """`flowmap_smc_mean_sample` with the same toy defaults `test_flowmap_smc._run` uses."""
    schedule = kw.pop("schedule", LinearSchedule())
    flow_map = kw.pop("flow_map", None) or GaussianFlowMap(schedule)
    kw.setdefault("score_fn", _gaussian_score_fn())
    kw.setdefault("n_steps", 16)
    kw.setdefault("mc_samples", 4)
    kw.setdefault("seed", 7)
    return flowmap_smc_mean_sample(
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
# The aggregator itself
# ---------------------------------------------------------------------------------------------------

def test_mean_value_is_the_arithmetic_mean():
    v = torch.tensor([[0.0, 1.0, 2.0, 5.0], [-1.0, -1.0, -1.0, -1.0]], dtype=dtype)
    assert torch.allclose(_mean_value(v), torch.tensor([2.0, -1.0], dtype=dtype))


def test_jensen_ordering_mean_le_soft_le_max():
    """``mean <= soft <= max`` elementwise -- the premise of the whole module.

    The mean-to-soft gap is the convexity term; the soft-to-max gap is what `flowmap_smc_max` spends.
    Equality throughout at ``K = 1`` is covered by `test_k1_reduces_to_the_soft_sampler`.
    """
    gen = torch.Generator().manual_seed(0)
    v = torch.randn((64, 8), generator=gen, dtype=dtype) * 2.0
    mean, soft, mx = _mean_value(v), _soft_value(v), _max_value(v)
    assert torch.all(mean <= soft + 1e-12)
    assert torch.all(soft <= mx + 1e-12)
    assert float((soft - mean).mean()) > 0.05        # a real gap, not float slop


def test_mean_soft_gap_is_the_convexity_term():
    """``soft - mean -> Var_k(v)/2`` in the small-spread limit -- the second-order expansion.

    This is the identity the module's case rests on, and the same one measured on FLUX in
    `notebooks/flowmap_smc_replay/A0_RESULTS.md` (ratio 0.79-1.10 against the recorded drain).
    """
    gen = torch.Generator().manual_seed(1)
    v = 0.02 * torch.randn((512, 16), generator=gen, dtype=dtype)
    gap = _soft_value(v) - _mean_value(v)
    predicted = v.var(dim=1, unbiased=False) / 2.0
    assert torch.allclose(gap, predicted, rtol=0.05, atol=1e-8)


# ---------------------------------------------------------------------------------------------------
# The two exact reductions -- these prove the override touches the aggregation and nothing else
# ---------------------------------------------------------------------------------------------------

def test_k1_reduces_to_the_soft_sampler():
    """At ``K = 1`` mean and logmeanexp are both the identity, so the runs must be bit-identical."""
    reward = _reward()
    a = _run_mean(reward, lam=3.0, mc_samples=1, guid_window=FULL_WINDOW, record_r_k=True)
    b = _run_soft(reward, lam=3.0, mc_samples=1, guid_window=FULL_WINDOW, record_r_k=True)
    assert torch.equal(a.X, b.X)
    assert torch.equal(a.logw, b.logw)
    assert a.uniq_history == b.uniq_history


def test_lambda_zero_reduces_to_the_base_process():
    """At ``lam = 0`` every ``v_k`` is 0, both aggregations give 0, and the run is the base process."""
    reward = _reward()
    a = _run_mean(reward, lam=0.0, guid_window=FULL_WINDOW)
    b = _run_soft(reward, lam=0.0, guid_window=FULL_WINDOW)
    assert torch.equal(a.X, b.X)
    assert torch.equal(a.logw, b.logw)


def test_mean_and_soft_differ_when_they_should():
    """With ``K > 1`` and a real tilt the two must actually diverge, or the test above proves nothing."""
    reward = _reward()
    a = _run_mean(reward, lam=8.0, mc_samples=8, guid_window=FULL_WINDOW)
    b = _run_soft(reward, lam=8.0, mc_samples=8, guid_window=FULL_WINDOW)
    assert not torch.equal(a.X, b.X)


# ---------------------------------------------------------------------------------------------------
# The target is pinned by the terminal update, not by the aggregator
# ---------------------------------------------------------------------------------------------------

def test_terminal_potential_is_raw_lambda_f_under_the_mean():
    """The last step runs at ``K = 1`` on a separate branch, so ``V_N`` must be exactly ``lam·f(x_1)``.

    This is what makes ANY aggregation leave the target at ``q_lambda``; if the aggregator ever leaked
    into the terminal update, the mean would target a different distribution than `flowmap_smc`.
    """
    reward, lam = _reward(), 5.0
    res = _run_mean(reward, lam=lam, guid_window=FULL_WINDOW)
    assert torch.allclose(res.V_history[-1], lam * reward(res.X), rtol=1e-10, atol=1e-12)


def test_the_patch_is_restored_even_when_the_sampler_raises():
    """A notebook's deadline callback raises out of the sampler; the base module must be left clean."""
    reward = _reward()

    def boom(i, snap, partial):
        raise RuntimeError("deadline")

    original = flowmap_smc._soft_value
    with pytest.raises(RuntimeError, match="deadline"):
        _run_mean(reward, lam=3.0, guid_window=FULL_WINDOW, on_step=boom)
    assert flowmap_smc._soft_value is original


def test_the_patch_is_restored_after_a_normal_run():
    reward = _reward()
    original = flowmap_smc._soft_value
    _run_mean(reward, lam=3.0, guid_window=FULL_WINDOW)
    assert flowmap_smc._soft_value is original


# ---------------------------------------------------------------------------------------------------
# The reason the module exists: the estimator error must fall like 1/K again
# ---------------------------------------------------------------------------------------------------

def test_mean_estimator_error_falls_like_one_over_k_and_the_soft_one_does_not():
    """``sigma^2(K) ~ K^-p``: ``p = 1`` for the mean, materially below 1 for the soft value.

    This is the module's headline claim, measured on FLUX (`a1_mean_aggregation.py`: p = 0.90-1.21 for
    the mean against 0.32-0.82 for the soft value) and reproduced here analytically. The regime that
    matters is a LARGE candidate spread ``a = lam*sd_k``, since ``a(t -> 0) = m`` under any lookahead
    whose candidates approach prior draws -- so the test is run at ``a ~ 2``, not at ``a ~ 0``.
    """
    gen = torch.Generator().manual_seed(3)
    n_rep, a = 4000, 2.0
    base = a * torch.randn((n_rep, 32), generator=gen, dtype=dtype)

    def sigma2(fn, k: int) -> float:
        # variance across independent K-column blocks of the same draws
        blocks = torch.stack([fn(base[:, j * k:(j + 1) * k]) for j in range(32 // k)])
        return float(blocks.var(dim=0, unbiased=True).mean())

    def exponent(fn) -> float:
        ks = [2, 4, 8, 16]
        xs = [math.log(k) for k in ks]
        ys = [math.log(sigma2(fn, k)) for k in ks]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return -num / sum((x - mx) ** 2 for x in xs)

    p_mean, p_soft = exponent(_mean_value), exponent(_soft_value)
    assert p_mean == pytest.approx(1.0, abs=0.1), f"the mean must be a 1/K estimator, got {p_mean}"
    assert p_soft < 0.85, f"the soft value should saturate at a = {a}, got p = {p_soft}"
    assert p_mean > p_soft + 0.15
