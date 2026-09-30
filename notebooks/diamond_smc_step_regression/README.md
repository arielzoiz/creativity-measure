# Diamond Maps base-sampler step-count regression

Follow-up to `notebooks/diamond_smc_imagenet/`. That directory's `sweep_sampler_quality.py` found
that at `cfg_scale=4.0`, going from `N=6` to `N=12` outer (DDPM-transition) steps made decoded images
*worse* (pixel std 0.048 -> 0.034), not better, while inner-step count (8 vs 16) barely mattered. An
even earlier script in that directory, `ab_base_sampler.py`, had already flagged the same shape of
anomaly at `cfg_scale=1.0`: latent std falls from ~0.53 (N=4) to ~0.39 (N=6), "the deficit GROWS with
the number of transitions." Both are backwards for a converging discretization — more transitions
should reduce integration error, not compound it.

This is a candidate explanation for why the real SMC run's images
(`notebooks/diamond_smc_imagenet/grid_label207_cfg1.0.png`) are so washed out, independent of the
`cfg_scale=1.0` vs `4.0` question that motivated this directory.

## Hypothesis

- **H_inherent**: latent std plotted against *absolute model time* `t` traces one curve regardless of
  how many outer steps `N` it took to reach that `t`. The schedule is legitimately contractive in `t`;
  `N` isn't the driver, and the `N=12` result is just what `t=1` looks like.
- **H_compounding**: std at a given `t` depends on *how many transitions* were taken to reach it — the
  `N=12` curve sits below the `N=6` curve at every matched `t`. Candidate mechanism: each Algorithm 2
  line 6 transition (`calc_xbar_s0` draws fresh noise at the inner-time origin `s=0`, `calc_x_t_prime`
  rescales a sufficient statistic of `(x_t, x_s)` back through `out_interp().alpha`) injects slightly
  too little variance to compensate for what conditioning on `x_t` removes — an ancestral-sampling
  variance leak, invisible at few steps and compounding at many.

These make different, checkable predictions: H_inherent's curves overlap when plotted against `t`;
H_compounding's fan out, worse for larger `N`, even though every `N` sweeps the same `[0, 1]`.

## Files

| file | what it is |
|---|---|
| `step_trace.py` | the measurement: for `N in {4,6,8,12,16,24}` (cfg=4.0, inner=8, batch=16, seed=0), record latent std/mean after every outer transition, plus the final decoded image batch. Pure measurement, no SMC, no reward — `base_step` only. |
| `step_trace.slurm` | submits `step_trace.py`. |
| `paths.py` | checkpoint/environment resolution, copied verbatim from `notebooks/diamond_smc_imagenet/paths.py` (each experiment directory is self-contained). |

Prerequisite: the `diamond-creative` conda env and cached checkpoints from
`notebooks/diamond_smc_imagenet/README.md` — nothing new to install.

## Running

```sh
sbatch step_trace.slurm
```

Budget ~90–120 min: six backends are built in sequence (each unpickles the 8.1 GB + 2.6 GB
checkpoints and JIT-compiles its own step functions), and the grid sums to 70 outer transitions
versus the ~36 in the prior sweep.

## Reading the output

The job prints a `std` trace per `N`, then a table of std at matched `t in {0.25, 0.5, 0.75, 1.0}`
across every `N` — that table *is* the H_inherent/H_compounding test: rows that agree down each
column support H_inherent, columns that decrease left-to-right (increasing `N`) support H_compounding.
Raw traces and final decoded image batches are saved to `step_trace_results.pt` for offline plotting.

## Takeaways

**H_compounding confirmed; H_inherent ruled out.** Job 905506, `cfg_scale=4.0`, `inner=8`, `label=207`,
batch=16, seed=0. Latent std at matched absolute `t`, across `N in {4,6,8,12,16,24}`:

| t | N=4 | N=6 | N=8 | N=12 | N=16 | N=24 |
|---|---|---|---|---|---|---|
| 0.25 | 0.783 | 0.849 | 0.712 | 0.646 | 0.593 | 0.527 |
| 0.50 | 0.618 | 0.575 | 0.527 | 0.472 | 0.442 | 0.389 |
| 0.75 | 0.600 | 0.529 | 0.448 | 0.375 | 0.334 | 0.274 |
| 1.00 | 0.720 | 0.619 | 0.496 | 0.380 | 0.312 | **0.205** |

Every row is monotonically decreasing in `N` (one exception: `t=0.25`, `N=4` vs `N=6`, plausibly
batch=16 sampling noise — every other cell across the whole grid is strictly monotone). At `t=1` std
falls **3.5x** from `N=4` to `N=24`, covering the *same* `[0, 1]` interval both times. The ratio
between successive step-doublings at `t=1` is 0.69 (4->8), 0.63 (8->16), 0.54 (12->24) — **shrinking**,
not approaching 1, which is the signature of a scheme that does not converge to a fixed limiting
distribution as steps -> infinity. An ordinarily-discretized SDE would show the opposite: gaps
between successive `N` shrinking as `N` grows. Decoded images corroborate this directly — `N=4` gives
sharp, recognizable golden-retriever faces; by `N=16`/`N=24` almost everything but a stray eye/nose
blob has washed to a flat gray. `sweep_sampler_quality.py`'s `N=6` vs `N=12` regression, and
`ab_base_sampler.py`'s original `N=4` vs `N=6` anomaly, are both instances of this same mechanism, now
characterized across a 6x range of step counts with no sign of an asymptote.

**This is a property of the base sampler (Algorithm 2 line 6 / GLASS transitions), not of this
project's bridge or config.** `ab_base_sampler.py` already showed the per-step driving matches
upstream's own batched sampler bit-for-bit, and this experiment only varies `N` and `t` — same
`cfg_scale`, same `inner`, same seed throughout. Leading candidate mechanism (not yet verified at the
source level): each transition's `calc_xbar_s0` draws fresh noise at the inner-time origin `s=0`
conditioned on `x_t`, and `calc_x_t_prime` rescales a sufficient statistic of `(x_t, x_s)` back through
`out_interp().alpha`; if that combination returns slightly less variance than conditioning on `x_t`
removed, every additional transition compounds the shortfall. Tracing this through
`common/interpolant.py`'s `alpha`/`sigma`/`gamma` (used by `common/stoch.py`, see
`repos/diamond_maps/posterior_diamond_maps/py/common/stoch.py`) would confirm the exact mechanism but
was not done here — this experiment stops at establishing *that* it compounds and *how fast*, not
*why* at the formula level.

**Practical implication: do not raise `N` to fix image quality — it actively makes things worse.**
The real SMC run's `N=6` is closer to the good end of this range than the `N=12`/`inner=16` configs
tried in the prior sweep. If fidelity still matters after fixing the `cfg_scale` question (separately
under debate — see `notebooks/diamond_smc_imagenet/`), the lever is FEWER outer steps, not more,
traded against Algorithm 2's SMC granularity (fewer steps = fewer resampling/reweighting
opportunities for the tilt).
