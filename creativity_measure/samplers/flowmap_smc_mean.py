"""Algorithm 3 with the **plain mean** lookahead aggregation -- ``V_t = lambda * mean_k r_k``.

Everything except the aggregation of the ``K`` lookahead candidates is `flowmap_smc`: the base
transitions, the renoise, the score correction, the resampling and the terminal update are imported,
not re-implemented. Only `_soft_value` is replaced -- the same mechanism, and the same one-import-surface
discipline, as `flowmap_smc_max`.

**The interpolation.** ``(1/tau) log mean_k exp(tau * lambda * r_k)`` runs from the mean (``tau -> 0``,
this module) through `flowmap_smc`'s soft value (``tau = 1``) to `flowmap_smc_max` (``tau -> inf``).
The max end was measured and is null: ``-0.034 std_p(f)``, CI ``[-0.67, +0.60]``
(`notebooks/flowmap_smc_max/TAKEAWAYS.md`). This is the other end, and it is not the same experiment.

**What changes, and why it is not another rescaling.** Expanding the soft value,
``V/lambda = mean_k r + lambda*Var_k(r)/2 + O(lambda^2)``. The max *rescales* that same ordering --
Spearman 0.93-0.97 against soft, which is why it did nothing. The mean **deletes the second term**.
Three consequences, all measured offline on the recorded ``r_k`` of the existing runs
(`notebooks/flowmap_smc_replay/a1_mean_aggregation.py`):

1. **``K`` starts working again.** The estimator error of the soft value falls like ``k^-p`` with
   ``p = 0.32-0.82`` (and ``p ~ 0`` for the max) -- the signature of a max-dominated statistic, and the
   mechanism behind the measured saturation of ``K``. The mean is a linear statistic of independent
   draws and measures ``p = 0.90-1.21``, i.e. the textbook ``1/k``. **Every conclusion that "raising K
   buys nothing" is a property of the aggregation, not of the lookahead.**
2. **``sd(U)/sd(V)`` improves in every cell** -- mean < soft < max, monotone, 5-11% below soft.
3. **The deleted term points the WRONG WAY.** ``corr(lambda*sigma_k^2/2, f(t=1))`` is negative in 7 of
   11 cells (to ``-0.59``) while ``corr(mean_k r, f(t=1))`` reaches ``+0.94``. The convexity term is
   8-30% of ``sd(V)``, so the soft value spends that fraction of its spread backing particles with
   *uncertain* futures, which finish lower.

**Jensen exposure goes to zero, and that is what matters at strong tilt.** The soft value carries both
the convexity ``lambda^2 sigma_k^2/2`` and a ``K``-sample estimator bias ``~ (e^{a^2}-1)/2K`` with
``a = lambda*sigma_k`` -- **exponential in ``a^2``**. Under any lookahead whose candidate spread
approaches the prior's, ``a(t -> 0) = lambda*std_p(f) = m`` exactly, so at ``m = 3`` the soft value
IS a hard max at ``K = 4`` and stops estimating a value function at all. The mean has no such term and
does not degrade with ``m``. **This module's case is strongest exactly where the project wants to go.**

**Why the target is untouched.** Every intermediate ``V_t`` is a twist: the potentials telescope and
``q_lambda`` is pinned entirely by the mandatory terminal update ``V_N = lambda*f(x_1)``, which runs at
``K = 1`` on a different code path (``flowmap_smc_sample``'s ``guided and is_final`` branch) and never
calls the aggregator. Any aggregation of the K candidates therefore leaves the target exactly
``q_lambda``; what changes is variance and where resampling spends its effort. The mean is the
*first-order* twist -- guide toward a high expected reward rather than a high soft maximum -- which is
a defensible reading of the lookahead in its own right, not merely an approximation of the other one.

**What to expect.** ``spearman(mean, soft) = 0.92-0.98`` on the recorded runs -- the same
"barely reorders" signature that preceded the max's null result. So a large move in terminal
``E_q[f]`` at ``m = 1.25`` would be a surprise. The reason to run it anyway is (1) and the Jensen
argument: it is a strictly better-behaved estimator at equal cost, and it is the only aggregation whose
behaviour does not collapse as the tilt rises. Judge it on ``uniq/M``, on ``sd(U)/sd(V)``, and on
whether a ``K`` sweep through THIS module finally shows the ``1/sqrt(K)`` it should.

**One import surface for notebooks.** A notebook driving this sampler imports from *this module only*::

    from creativity_measure.samplers.flowmap_smc_mean import (
        LinearSchedule, ddpm_step, flow_map_step, flowmap_smc_mean_sample,
    )

Those base-process names are **re-exported, not copied**. The reference files record the base process
by *name* (``BASE_PROCESS["inside_step"] = ddpm_step.__name__``), so a duplicate `ddpm_step` would pass
``_assert_same_p`` while being a different function -- silently invalidating a reference set against
the exact check written to catch that. Re-exporting binds the same objects, so the modules cannot
diverge. `flowmap_smc_sample` is re-exported too, so a PAIRED run can select the aggregation at call
time and still import from one place.

Not exported from ``creativity_measure/__init__.py``.
"""

from typing import Any

from jaxtyping import Float
from torch import Tensor

from creativity_measure.samplers.flowmap_smc import (
    SCHEDULES,
    BaseSchedule,
    FlowMapSMCResult,
    LinearSchedule,
    StepCallback,
    StepSnapshot,
    ddpm_step,
    flow_map_step,
    flowmap_smc_sample,
)
from creativity_measure.samplers.flowmap_smc_max import ValueAgg, _override_value_agg
from creativity_measure.tilt import Reward

__all__ = [
    # this module
    "flowmap_smc_mean_sample",
    "_mean_value",
    # re-exported so a notebook imports from here only -- SAME OBJECTS, not copies.
    "flowmap_smc_sample",
    "ValueAgg",
    "SCHEDULES",
    "BaseSchedule",
    "FlowMapSMCResult",
    "LinearSchedule",
    "StepCallback",
    "StepSnapshot",
    "ddpm_step",
    "flow_map_step",
]


def _mean_value(v_hat: Float[Tensor, "M K"]) -> Float[Tensor, "M"]:
    """``V^m = mean_k v_k`` -- the first-order counterpart of `flowmap_smc._soft_value`.

    ``<= _soft_value(v_hat)`` elementwise by Jensen, with equality exactly at ``K = 1``. The gap is
    the convexity ``lambda*Var_k(r)/2`` this module exists to remove; it is asserted in the tests, as
    is the ordering ``mean <= soft <= max``.

    Note this receives ``v_hat = lambda * r_k`` already scaled, so the returned value is
    ``lambda * mean_k r_k`` and no separate ``lambda`` handling is needed -- the same contract
    `_soft_value` has.
    """
    return v_hat.mean(dim=1)


def flowmap_smc_mean_sample(
    reward: Reward,
    lam: float,
    n_particles: int,
    **kwargs: Any,
) -> FlowMapSMCResult:
    """`flowmap_smc.flowmap_smc_sample` with the lookahead aggregated by ``mean`` instead of ``logmeanexp``.

    Args:
        reward:      the frozen tilt ``f``, as for `flowmap_smc_sample`.
        lam:         inverse temperature of the target.
        n_particles: number of particles ``M``.
        **kwargs:    forwarded verbatim to `flowmap_smc_sample`. The keyword surface is deliberately a
                     passthrough rather than a restated signature: mirroring thirty parameters here
                     would let the two drift, and only the aggregation differs. ``flow_map`` and
                     ``score_fn`` remain required.

    Returns:
        :class:`flowmap_smc.FlowMapSMCResult` -- same fields, same meanings.

    Note:
        Reduces to `flowmap_smc_sample` **exactly** in two cases, both tested: ``mc_samples = 1``
        (the mean of one element is that element, which is also what logmeanexp returns) and
        ``lam = 0`` (every ``v_k`` is 0, so both aggregations give 0 and the run is the base process).
        ``use_full_normalized_v=True`` is accepted and applies the mean to the Z-scored combination,
        exactly as the soft value would -- but that branch is stale and untested here.
    """
    with _override_value_agg(_mean_value):
        return flowmap_smc_sample(reward, lam, n_particles, **kwargs)
