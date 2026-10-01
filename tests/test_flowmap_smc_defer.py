"""Tests for the resampling window (creativity_measure/flowmap_smc_defer.py).

`flowmap_smc_defer` carries a COPY of the step loop, because the resampling decision is an inline
predicate rather than a swappable function (see that module's docstring on why faking `_ess_from_logw`
is not acceptable: the same value is what `ess_history` records).

**The first test is the reason the copy is safe.** At ``resample_window=(0.0, 1.0)`` the copy must
reproduce `flowmap_smc_sample` bit-for-bit -- X, logw and every diagnostic list -- at a real tilt with
resampling actually firing. That turns the duplication into a pinned invariant: anyone who changes
`flowmap_smc`'s bookkeeping gets a failing test here instead of two samplers that quietly diverge.
Everything else in this file tests the one behaviour that is genuinely new.
"""

from typing import Any

import pytest
import torch

from creativity_measure.samplers.flowmap_smc import LinearSchedule, flowmap_smc_sample
from creativity_measure.samplers.flowmap_smc_defer import flowmap_smc_defer_sample
from creativity_measure.tilt import Reward
from test_flowmap_smc import GaussianFlowMap, _gaussian_score_fn, _reward

FULL_WINDOW = (0.0, 1.0)


def _kw(**over: Any) -> dict[str, Any]:
    schedule = over.pop("schedule", LinearSchedule())
    kw: dict[str, Any] = dict(
        flow_map=over.pop("flow_map", None) or GaussianFlowMap(schedule),
        score_fn=_gaussian_score_fn(), schedule=schedule,
        n_steps=16, mc_samples=4, seed=7, guid_window=FULL_WINDOW, ess_threshold=0.5,
    )
    kw.update(over)
    return kw


def _defer(reward: Reward, lam: float, n_particles: int = 8, **over: Any):
    return flowmap_smc_defer_sample(reward, lam, n_particles, **_kw(**over))


def _base(reward: Reward, lam: float, n_particles: int = 8, **over: Any):
    over.pop("resample_window", None)
    return flowmap_smc_sample(reward, lam, n_particles, **_kw(**over))


# ---------------------------------------------------------------------------------------------------
# THE PIN -- this is what licenses the copied loop
# ---------------------------------------------------------------------------------------------------

def test_full_window_is_bit_identical_to_the_base_sampler():
    """``resample_window=(0.0, 1.0)`` must reproduce `flowmap_smc_sample` EXACTLY.

    Run at a real tilt with ``ess_threshold=0.5`` so resampling actually fires -- a comparison on a
    run that never resampled would prove nothing about the branch this module changes.
    """
    reward, lam = _reward(), 6.0
    a = _defer(reward, lam, resample_window=FULL_WINDOW,
               record_r_k=True, project_endpoint=True, keep_steps=True)
    b = _base(reward, lam, record_r_k=True, project_endpoint=True, keep_steps=True)

    assert any(b.resampled_history), "the pin is vacuous unless the baseline resampled"
    assert torch.equal(a.X, b.X)
    assert torch.equal(a.logw, b.logw)
    assert a.ess_history == b.ess_history
    assert a.resampled_history == b.resampled_history
    assert a.uniq_history == b.uniq_history
    assert a.t_history == b.t_history
    assert a.guided_history == b.guided_history
    assert a.eqf_history == b.eqf_history
    assert a.f_mean == b.f_mean and a.f_std == b.f_std
    assert a.l_mean == b.l_mean and a.s_mean == b.s_mean
    for x, y in zip(a.V_history, b.V_history):
        assert torch.equal(x, y)
    for x, y in zip(a.U_pre_history, b.U_pre_history):
        assert torch.equal(x, y)
    for x, y in zip(a.resample_idx_history, b.resample_idx_history):
        assert (x is None) == (y is None)
        if x is not None and y is not None:
            assert torch.equal(x, y)
    for x, y in zip(a.r_k_history, b.r_k_history):
        assert (x is None) == (y is None)
        if x is not None and y is not None:
            assert torch.equal(x, y)


# ---------------------------------------------------------------------------------------------------
# The new behaviour
# ---------------------------------------------------------------------------------------------------

def test_no_resampling_before_the_window_opens():
    """Every step landing below the window must report ``resampled=False`` and no parent map."""
    reward = _reward()
    res = _defer(reward, lam=6.0, resample_window=(0.25, 1.0))
    for t, did, idx in zip(res.t_history, res.resampled_history, res.resample_idx_history):
        if t < 0.25 - 1e-9:
            assert not did, f"resampled at t={t}, below the window"
            assert idx is None
    assert any(res.resampled_history), "the window never opened -- the test proves nothing"


def test_uniq_stays_at_one_while_selection_is_off():
    """No resampling means no lineage can die: ``uniq/M`` must be exactly 1 inside the deferred region.

    This is the mechanism the arm is built on -- `uniq/M` is monotone non-increasing, so every lineage
    lost early is lost for good.
    """
    reward = _reward()
    res = _defer(reward, lam=6.0, resample_window=(0.25, 1.0))
    for t, u in zip(res.t_history, res.uniq_history):
        if t < 0.25 - 1e-9:
            assert u == 1.0


def test_weights_accumulate_instead_of_being_discarded():
    """Deferring must not throw information away: ``U`` keeps growing where selection is off.

    Between resamples ``U_n = V_n - V_{last}`` telescopes into a long-baseline difference, which is
    strictly more information than the single-step increment `_resample` forces selection onto.
    """
    reward = _reward()
    res = _defer(reward, lam=6.0, resample_window=(0.25, 1.0))
    early = [u for t, u in zip(res.t_history, res.U_pre_history) if t < 0.25 - 1e-9]
    assert len(early) >= 2
    assert float(early[-1].std()) > float(early[0].std()), "U should disperse while selection is off"


def test_empty_window_is_pure_importance_sampling():
    """A window that never opens must never resample -- the exact-IS limit."""
    reward = _reward()
    res = _defer(reward, lam=6.0, resample_window=(1.0, 1.0), final_resample=False)
    assert not any(res.resampled_history)
    assert all(u == 1.0 for u in res.uniq_history)
    assert float(res.logw.std()) > 0.0, "IS must leave a spread in logw; it is the only signal left"


def test_lambda_zero_is_unaffected_by_the_window():
    """At ``lam = 0`` weights are uniform, so systematic resampling is the identity either way."""
    reward = _reward()
    a = _defer(reward, lam=0.0, resample_window=(0.25, 1.0))
    b = _base(reward, lam=0.0)
    assert torch.equal(a.X, b.X)


def test_terminal_resample_still_obeys_final_resample_and_the_window():
    """The terminal step needs BOTH permissions -- and, as always, an ESS below the threshold.

    ``ess_threshold=1.0`` is what isolates the window from the ESS trigger: it makes every step want
    to resample, so a terminal step that does NOT is the window's doing and nothing else.
    """
    reward = _reward()
    closed = _defer(reward, lam=6.0, resample_window=(0.0, 0.5),
                    final_resample=True, ess_threshold=1.0)
    assert not closed.resampled_history[-1], "window closed at t=1, so no terminal resample"
    opened = _defer(reward, lam=6.0, resample_window=FULL_WINDOW,
                    final_resample=True, ess_threshold=1.0)
    assert opened.resampled_history[-1]
    off = _defer(reward, lam=6.0, resample_window=FULL_WINDOW,
                 final_resample=False, ess_threshold=1.0)
    assert not off.resampled_history[-1], "final_resample=False must still veto it"


def test_a_non_contiguous_window_is_rejected():
    """Reuses `flowmap_smc._check_window`, so the same guard covers all three windows."""
    reward = _reward()
    with pytest.raises(ValueError, match="resample_window"):
        _defer(reward, lam=1.0, resample_window=(0.7, 0.3))
