"""Algorithm 3 with a **hard-max** lookahead aggregation -- ``V_t = max_k v_k``.

Everything except the aggregation of the ``K`` lookahead candidates is `flowmap_smc`: the base
transitions, the renoise, the score correction, the resampling and the terminal update are imported,
not re-implemented. Only `_soft_value` is replaced.

**What changes.** The soft value ``V_t = log((1/K) sum_k exp(v_k))`` becomes ``V_t = max_k v_k``. The
motivation is that the mean-of-K plug-in estimates what a particle will *typically* become, whereas
the interesting question for a creativity tilt may be what it could *at best* become -- its
"maximum potential" -- so that selection backs the particle with the better upside rather than the
better average.

**Why this does not break the target.** Every *intermediate* ``V_t`` is a twist (`flowmap_smc`, module
docstring): the potentials telescope, and ``q_lambda`` is pinned entirely by the mandatory terminal
update ``V_N = lambda·f(x_1)``, which runs at ``K = 1`` on a different code path
(``flowmap_smc_sample``'s ``guided and is_final`` branch) and never calls the aggregator at all.
Any monotone aggregation of the K candidates therefore leaves the target exactly ``q_lambda``. The
max is a biased estimator of the value function ``log E[exp(r)]``, but that bias is spent on twist
*quality* -- variance, and where resampling puts its effort -- not on correctness. It is paid in
``ess_history`` and ``uniq_history``, which the sampler already records.

**This is the zero-temperature limit of what is already there.** ``(1/tau)·log mean_k exp(tau·lambda·r_k)``
interpolates: ``tau = 1`` is `flowmap_smc`, ``tau -> inf`` is this module, up to the ``- log K`` that
`_soft_value` carries and the max does not. That residual is **inert**: it is constant across
particles, so it cannot move a softmax weight, and consecutive potentials both carry it, so it
cancels in the telescoping ``U += V_next - V_prev`` (including across a resample, which zeroes ``U``
but keeps ``V_prev``). It survives only in the first guided increment and in the terminal one, where
it is again a shared constant. Writing
``r_k = r_bar + sigma_k·u_k`` and ``a = lambda·sigma_k``, the boost over the plain mean is
``lambda·sigma_k^2/2`` for the soft value against ``sigma_k·E[max_k u]`` here, a ratio of ``2·c_K/a``
with ``c_K ~ 1.03, 1.42, 1.77, 2.07`` at ``K = 4, 8, 16, 32``. **Check ``a`` on a run's
``r_k_history`` before spending a GPU on this**: at ``a >~ 5`` the logsumexp has already saturated
into a max and this module reproduces `flowmap_smc`. See
``notebooks/flowmap_smc_k_sweep/analyze_r_k_spread.py``.

**Two things to expect.**

* ``K`` stops being a nuisance parameter. The max boost grows like ``sigma_k·sqrt(2 ln K)`` while the
  soft boost is K-independent, so under this aggregation **raising K strengthens the twist**. A K
  sweep run through this module measures tilt strength and estimator noise together, which is exactly
  what `notebooks/flowmap_smc_k_sweep/` was written to separate. Do not read the two sweeps the same way.
* The max selects on upside the particle cannot necessarily cash in. The K candidates differ only in
  lookahead noise, so ``max_k r_k`` promotes a particle with a fat-tailed lookahead posterior as much
  as one with a high mean -- and at ``t = 1`` the exact per-particle ``lambda·f`` corrects that
  promotion back down. Whether the net is a higher final ``E_q[f]`` is not derivable a priori; that is
  what makes this an experiment.

**Why a runtime override and not a parameter.** `flowmap_smc_sample` resolves ``_soft_value`` as a
module global at call time, so swapping the module attribute for the duration of one call is enough.
The alternative -- a ``value_agg`` argument on `flowmap_smc_sample` -- is cleaner and is where this
should end up, but it edits a file that concurrently queued Slurm jobs re-read from the live working
tree at their pytest gate, and a K sweep is only valid if every cell runs the same library. Collapse
this into a proper parameter once no sweep is in flight; `flowmap_smc_max_sample` then becomes a
two-liner and these tests keep passing unchanged.

**Diagnostics.** Unchanged, and they already cover the max: ``V_history[n] / lam`` is exactly the
per-particle ``max_k r_k`` on a guided step, and ``record_r_k=True`` keeps the raw ``(M, K)`` rewards,
so the soft value of the same run is reconstructible offline for a paired comparison.

**One import surface for notebooks.** A notebook driving this sampler imports from *this module only*,
never from both -- so nothing about which aggregation ran depends on remembering which of two
imports a cell used::

    from creativity_measure.flowmap_smc_max import (
        LinearSchedule, ddpm_step, flow_map_step, flowmap_smc_max_sample,
    )

Those base-process names are **re-exported, not copied**. A copy would be a second definition of the
process ``p``, and the reference files record it by *name*
(``BASE_PROCESS["inside_step"] = ddpm_step.__name__``), so a duplicate `ddpm_step` would pass
``_assert_same_p`` while being a different function -- silently invalidating a reference set against
the exact check written to catch that. Re-exporting binds the same objects, so the two modules cannot
diverge at all. The only name that differs from a `flowmap_smc` notebook is the sampler itself:
``flowmap_smc_sample`` -> `flowmap_smc_max_sample`, same signature, same result type. Deliberately
*not* aliased to the old name -- a reader must be able to tell from the call which aggregation ran.

Not exported from ``creativity_measure/__init__.py``.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from jaxtyping import Float
from torch import Tensor

from creativity_measure import flowmap_smc
from creativity_measure.flowmap_smc import (
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
from creativity_measure.tilt import Reward

__all__ = [
    # this module
    "flowmap_smc_max_sample",
    "_max_value",
    "_override_value_agg",
    "ValueAgg",
    # re-exported from `flowmap_smc` so a notebook imports from here only -- SAME OBJECTS, not copies.
    # `flowmap_smc_sample` is here so a PAIRED experiment can select the aggregation at run time and
    # still import from one module: the soft arm and the max arm must differ in exactly one call, and
    # having both names come from the same place is what makes that auditable.
    "flowmap_smc_sample",
    "SCHEDULES",
    "BaseSchedule",
    "FlowMapSMCResult",
    "LinearSchedule",
    "StepCallback",
    "StepSnapshot",
    "ddpm_step",
    "flow_map_step",
]

# The aggregation of one particle's K lookahead scores into its potential.
ValueAgg = Callable[[Float[Tensor, "M K"]], Float[Tensor, "M"]]


def _max_value(v_hat: Float[Tensor, "M K"]) -> Float[Tensor, "M"]:
    """``V^m = max_k v_k`` -- the hard-max counterpart of `flowmap_smc._soft_value`.

    ``>= _soft_value(v_hat)`` elementwise for every input, with equality exactly at ``K = 1``, since
    ``log((1/K) sum_k exp(v_k))`` is a mean of exponentials and ``max`` is their supremum. That
    inequality is what makes this a strictly sharper twist and is asserted in the tests.
    """
    return v_hat.max(dim=1).values


@contextmanager
def _override_value_agg(agg: ValueAgg) -> Iterator[None]:
    """Swap `flowmap_smc._soft_value` for ``agg`` for the duration of the block, then restore it.

    Restored in a ``finally``, so an exception raised out of the sampler -- a notebook's deadline
    callback, most importantly -- cannot leave the base sampler patched. **Not thread-safe and not
    reentrant-safe against a concurrent `flowmap_smc_sample`**: it mutates module state. That is
    acceptable for a notebook driving one sampler at a time, and it is temporary; see the module
    docstring on collapsing this into a ``value_agg`` parameter.
    """
    original = flowmap_smc._soft_value
    setattr(flowmap_smc, "_soft_value", agg)
    try:
        yield
    finally:
        setattr(flowmap_smc, "_soft_value", original)


def flowmap_smc_max_sample(
    reward: Reward,
    lam: float,
    n_particles: int,
    **kwargs: Any,
) -> FlowMapSMCResult:
    """`flowmap_smc.flowmap_smc_sample` with the lookahead aggregated by ``max`` instead of ``logmeanexp``.

    Args:
        reward:      the frozen tilt ``f``, as for `flowmap_smc_sample`.
        lam:         inverse temperature of the target.
        n_particles: number of particles ``M``.
        **kwargs:    forwarded verbatim to `flowmap_smc_sample`. The keyword surface is *deliberately*
                     a passthrough rather than a restated signature: mirroring thirty parameters here
                     would let the two drift, and the whole point of this module is that only the
                     aggregation differs. ``flow_map`` and ``score_fn`` remain required.

    Returns:
        :class:`flowmap_smc.FlowMapSMCResult` -- same fields, same meanings.

    Note:
        Reduces to `flowmap_smc_sample` **exactly** in two cases, both tested: ``mc_samples = 1``
        (max over one element is that element, which is also what logmeanexp returns) and
        ``lam = 0`` (every ``v_k`` is 0, so both aggregations give 0 and the run is the base process).
    """
    with _override_value_agg(_max_value):
        return flowmap_smc_sample(reward, lam, n_particles, **kwargs)
