# Algorithm 3 — hard max vs plug-in mean in the lookahead

**Question.** The lookahead aggregates `K` clean candidates into one potential. Replacing the plug-in
estimator `V_t = log((1/K) Σ_k exp(λ r_k))` with the hard max `V_t = max_k λ r_k` — a particle's
"maximum potential" rather than its typical one — does it raise `E_q[f]`?

**Answer: no.** Terminal `max − soft = −0.034 std_p(f)` over 3 paired seeds, 95% CI `[−0.67, +0.60]`.
The aggregation is not the bottleneck; the noise in the *increment* `U_n = V̂_n − V̂_{n−1}` is, and the
max does not reduce it (it raises it, +0.22 on average). Run 2026-09-08/09, jobs 870119/870120/870496–870499.

## Design

Frozen and identical across every arm: `M = 8`, `N = 16`, `m = 1.25` → `λ = 175.116`
(`λ_s = 140.093`), `ess_threshold = 0.5`, `guid_window = (0.0, 1.0)`, `η = 1.5`,
`stoch_window = (0.1, 1.0)`, `LinearSchedule`, `ts = None`, `use_full_normalized_v = False`,
`antithetic = True`, `R = 64` refs (`refs_R64_seed7.pt`, md5-identical to the k-sweep's),
`N_GAMMA = 30`, `NUM_EPS = 3`, `dist_seed = 123`, FLUX.1-dev + flowmap LoRA, prompt "A dog",
guidance 1.0, 512², L40S only. **The aggregation is the only thing that differs within a pair.**

`std_p(f) = 1/λ_s = 0.00714` is the unit throughout.

## Headline: terminal `E_q[f]`, K = 8, three paired seeds

| seed | max | soft | `(max−soft)/std_p(f)` | uniq/M max/soft | resamples | collapse |
|---|---|---|---|---|---|---|
| 101 | 1.0014 | 1.0049 | **−0.487** | 0.375 / 0.375 | 3 / 2 | none / none |
| 102 | 1.0088 | 1.0069 | **+0.257** | 0.125 / 0.375 | 4 / 3 | step 11 / none |
| 103 | 1.0079 | 1.0070 | **+0.127** | 0.250 / 0.125 | 3 / 5 | none / step 9 |

**mean −0.034 `std_p(f)`, sd 0.397, se 0.229, 95% CI [−0.671, +0.602]** (t₀.₉₇₅,₂ = 2.776).
In raw units the mean difference is −0.00025 against `std_p(f) = 0.00714`.

Only **seed 101** compares at matched final diversity (0.375 both), and there the max is **behind**.
Seed 102's `+0.257` was bought with one lineage against three — on the repo's selection rule
(`uniq/N ≳ 0.5`) that run is rejected outright, so it is not a win.

## Why: four independent lines, all agreeing

**1. The two potentials rank particles almost identically.** Spearman ρ between `max_k λr_k` and
`logmeanexp_k λr_k` on the *same* particles, per run: 0.943 / 0.940 (K=4), 0.957 / 0.975 (K=8 s101),
0.941 / 0.935 (s102), 0.973 / 0.954 (s103), 0.932 (K=16), 0.925 (K=32). Same argmax at 7 of 8 steps.
It does **not** degrade with K. The max does not find a different "promising" particle — it finds the
same one.

**2. The aggregation acts *only* through resampling.** The base transition never reads `U`, so until a
resample fires both arms carry *identical particles* and differ only in weights — verified directly:
seed 103 `f_cand` was bit-identical (0.9403) at step 3, diverged (0.9609 vs 0.9608) at step 4 only
after the step-3 resample split the ancestries. With **2–5 resamples in 16 steps**, that is the entire
channel through which the choice of potential can act.

**3. It does not fix the noise-dominated increment.** `sd(U_n)/sd(V̂_n)`, the plan's mechanism metric
(target `< 0.5`, baseline ≈ 1.0):

| K | seed | max | soft | max − soft |
|---|---|---|---|---|
| 8 | 101 | 0.524 | 0.586 | −0.062 |
| 8 | 102 | 1.128 | 0.758 | +0.370 |
| 8 | 103 | 1.052 | 0.688 | +0.364 |
| 4 | 101 | 1.113 | 0.840 | +0.273 |
| 16 | 101 | — | 1.653 | — |
| 32 | 101 | — | 0.948 | — |

Mean +0.224 at K=8: the max makes the quantity resampling consumes **noisier**, not cleaner. Nothing
reaches the `< 0.5` target except K=8 seed 101 max (0.524, the run with fewest resamples). Raising K
does not help either — K=16 is the worst at 1.653.

**4. Collapse is a resampling lottery, not a property of the aggregation.** Seed 101 neither arm
collapses; seed 102 the *max* collapses (step 11); seed 103 the *soft* collapses (step 9). The
diversity gaps flip sign across seeds, so the max's occasional lower `uniq/M` is chance, not cost.

## Gate that authorised the run (`analyze_r_k_spread.py`)

`a = λ·sd_k` on the baseline's checkpoints: **1.291** (K=8), **1.186** (K=16). Well below the `a ≳ 5`
no-op threshold, so the two aggregations were genuinely distinguishable —
`b_max/b_soft` = 3.0 (K=8) and 4.3 (K=16), and the extra across-particle dispersion was 0.67–0.70 of
what `U_pre` already carried. **The mechanism was in force and still produced nothing**, which is a
stronger null than "the potentials happened to coincide".
Measured in-run: the max potential exceeded the soft one by **0.61–0.68 log-weight units** on average.
Most of that is a shift common to all particles, which cancels in the softmax and in the telescoping `U`.

The closed form `b_max/b_soft ≈ 2c_K/a` underestimated by 35–45% at `a ≈ 1.2` (predicted 2.20/2.98 vs
measured 3.0/4.3) — it is the small-`a` asymptotic. Use the measured column.

## The untilted control (`ctrl2`, λ=0, K=1, M=8, seed 101, 39 min)

```
E_p[f]:  0.9160 0.9239 0.9455 0.9629 0.9784 0.9916 1.0017 1.0130 1.0182 1.0207
         1.0198 1.0151 1.0106 1.0042 0.9943 0.9943
```

**`E_p[f]` rises to a step-10 peak on its own and then falls.** Most of the within-run `E_q[f]` climb in
every tilted run is the base process, not the tilt — `E_q[f]` curves must be read against this, never
in absolute terms. Tilt at the terminal is only **+1.00 `std_p(f)`** (max K=8) and **+1.48** (soft K=8).
At λ=0 and at K=1 the two aggregations are provably bit-identical, so one control serves both arms.

## All completed runs

| K | seed | agg | terminal | peak | @ | give-back | uniq/M | resamples | collapse | wall (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 101 | — | 0.9943 | 1.0207 | 10 | 0.0264 | 1.000 | 0 | none | 39 |
| 4 | 101 | soft | 1.0000 | 1.0530 | 9 | 0.0530 | 0.125 | 5 | step 3 | 186 |
| 4 | 101 | max | *(13 steps, SIGSEGV)* | 1.0552 | 8 | — | 0.125 | 4 | step 3 | — |
| 8 | 101 | max | 1.0014 | 1.0444 | 10 | 0.0430 | 0.375 | 3 | none | 327 |
| 8 | 101 | soft | 1.0049 | 1.0451 | 10 | 0.0402 | 0.375 | 2 | none | 329 |
| 8 | 102 | max | 1.0088 | 1.0465 | 10 | 0.0378 | 0.125 | 4 | step 11 | 334 |
| 8 | 102 | soft | 1.0069 | 1.0383 | 10 | 0.0314 | 0.375 | 3 | none | 329 |
| 8 | 103 | max | 1.0079 | 1.0322 | 10 | 0.0243 | 0.250 | 3 | none | 332 |
| 8 | 103 | soft | 1.0070 | 1.0348 | 10 | 0.0278 | 0.125 | 5 | step 9 | 327 |
| 16 | 101 | soft | 0.9985 | 1.0254 | 10 | 0.0269 | 0.250 | 4 | none | 622 |

**K does not buy terminal novelty**: 1.0000 (K=4), 1.0049 (K=8), 0.9985 (K=16) — non-monotone and
spread over less than one `std_p(f)`. **Every run peaks at step 9–10 and gives back 60–90%** of it by
`t = 1`; only the terminal defines `q_λ`, so mid-trajectory comparisons are largely transient.

## At K=4 the aggregation changes nothing at all

Both arms collapse at **step 3**, and their `E_q[f]` agree to ~1e−3 over 8 steps (`0.9169/0.9172, 0.9306/0.9305, 0.9882/0.9880, 1.0080/1.0080, …`).
K=4 at this λ has two informative steps per run however it is seeded. **Do not spend seeds on K=4.**

## Determinism (verified, not assumed)

* `k08b` reproduced the SIGSEGV'd `k08` **bit-identically** over 7 steps (`max|ΔE_q[f]| = 0.000e+00`).
* Cross-node: `s103max` on n-803 and `s103soft` on n-805 gave identical `f_cand = 0.9179` at step 1.
* Every run's phase-1 health draw gave `latent_std = 1.0345` (base FLUX 1.0150), across 3 nodes.

Same seed + same L40S ⇒ bit-identical, across jobs and nodes. Reruns splice with dead runs' partials.

## Infrastructure (each cost real hours)

* **`/tmp` fills during the 32 GB stage.** n-804 killed job 869620 after 2h12m; n-801 killed 870099
  after 22 min. Same error, different nodes — not node-specific bad luck.
* **n-802 SIGSEGV'd two jobs in one night** — 870119 at step 14/16 and the k-sweep's 869621 at 5h46.
* **`--nodelist` onto an already-warm node beats `--exclude`.** A warm `/tmp` gives `cache HIT` and
  skips up to 257 min of staging; all four seed-sweep jobs started sampling within ~1 min of launch.
* **Per-step checkpointing paid out**: 870119's SIGSEGV cost only the terminal artifacts; 13 steps survived.
* `RUN_TAG` must differ per arm — `progress_${RUN_TAG}_run.log` and `phase1_health_${RUN_TAG}.png` are
  keyed on the tag alone, so a shared tag interleaves two jobs' logs.

## Caveats

* **M = 8.** Every terminal is a 1–3 lineage ensemble. This CI excludes a *useful* effect, not a small one.
* **One λ** (m = 1.25) and **one K** for the seed sweep.
* Controls exist only at seed 101, so absolute tilt is unavailable at 102/103. The paired difference is
  unaffected — the control cancels exactly.

## What this points at

The plan (`~/claude-config/plans/encapsulated-stargazing-rabin.md`, 2026-08-30) already established that resampling consumes `U_n = V̂_n − V̂_{n−1}`,
whose correlation with `f_final` is ~0 to −0.24 while the *level* `V̂_n` correlates 0.33–0.62.
Any aggregation rescales both terms alike, so the increment stays noise — which is why this experiment had to come out null.
Its Fix 1 (replay the base process for an exact posterior draw) and Fix 2 (common random numbers so consecutive `V̂` share tail noise) attack the increment directly.
Fix 2's own note that CRN *"cancels most of the residual max-of-K inflation"* is this result from the other side.

**The estimator of a particle's reward is not the bottleneck. The variance of its increment is.**

## Files

* `flowmap_smc_max.ipynb` / `.slurm` — both arms, selected by `AGG` ∈ {`max`,`soft`} and `SEED`
* `creativity_measure/flowmap_smc_max.py` — `flowmap_smc_max_sample`; re-exports the base process from
  `flowmap_smc` (same objects, never copies — the refs key `p` by `ddpm_step.__name__`)
* `tests/test_flowmap_smc_max.py` — 14 tests incl. the two exact reductions (`K=1`, `λ=0`)
* `analyze_r_k_spread.py` — the submission gate (`a = λ·sd_k`)
* `result_*.pt` / `partial_*.pt` — per-step `r_k`, `U_pre`, `V`, `f_proj`, ancestries
