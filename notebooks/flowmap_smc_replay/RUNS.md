# Algorithm 3 — the arms launched 2026-09-10

> ## FINAL: the mean aggregation is NULL (n = 3 paired seeds)
>
> | seed | mean | soft | (mean − soft)/std_p | uniq mean/soft | collapse |
> |---|---|---|---|---|---|
> | 101 | 1.0111 | 1.0000 | **$+1.55$** | 0.250 / 0.125 | never / step 3 |
> | 102 | 1.0042 | 1.0054 | **$-0.18$** | 0.375 / 0.500 | never / never |
> | 103 | 0.9997 | 1.0099 | **$-1.42$** | 0.250 / 0.250 | never / never |
>
> **n = 3: mean $-0.014$, sd 1.492, se 0.862, 95% CI $[-3.72, +3.69]$ $\operatorname{std}_p(f)$.**
>
> The point estimate lands almost exactly on the hard-max experiment's $-0.034$. **Both ends of the
> $\tau$ interpolation are now null against the soft value** — $\tau \to 0$ (mean) and
> $\tau \to \infty$ (max) — which together say more than either alone: **the aggregation axis does not
> move terminal novelty.** Do not revisit it without a new reason.
>
> The three seeds span $+1.55$ to $-1.42$ — a 3 sd swing from seed noise alone.
>
> One weak signal survives, not established: the mean **never collapsed at any seed**, while soft
> collapsed at 1 of 3. But at seed 102 soft held *more* lineages, so this is at best marginal.
>
> **Two seed-101 claims are retracted:**
>
> 1. **The novelty gain did not reproduce** — $+1.55 \to -0.18$ sd. Same seed-to-seed sign flip the
>    hard-max experiment found, which is why that one required three paired seeds to conclude null.
> 2. **Collapse-prevention was seed-specific.** At seed 102 the *soft* arm did not collapse either and
>    held MORE lineages (0.500 vs 0.375). "The mean prevents collapse" was a property of seed 101's
>    resampling draws, not of the aggregation.
>
> **What survives:** the offline mechanism findings, which were measured across all 11 existing cells
> rather than one run — $\sigma_\varepsilon^2 \sim k^{-1.0}$ for the mean vs $k^{-0.6}$ for the soft
> value, and the convexity term's negative correlation with $f(t{=}1)$. Those are still true; they have
> simply not translated into terminal novelty.
>
> ### The 3-seed design is underpowered — and n = 3 cannot settle this whatever seed 103 gives
>
> Observed per-seed sd of the paired difference: **1.22 $\operatorname{std}_p(f)$**. With
> $t_{0.975,2} = 4.303$, the n = 3 CI includes zero for **every** possible seed-103 value:
>
> | seed 103 = | −1.5 | 0.0 | +1.5 | +3.0 | +5.0 |
> |---|---|---|---|---|---|
> | n=3 CI | [−3.84, +3.76] | [−1.91, +2.82] | [−1.49, +3.40] | [−2.50, +5.41] | [−4.43, +8.67] |
>
> **"Three paired seeds" was inherited from the hard-max experiment without checking that it
> transfers. It does not.** That experiment's per-seed sd was **0.397**, because max and soft are
> near-identical arms (Spearman 0.93–0.97, same argmax at 7 of 8 steps) so their paired difference
> barely varies. Ours is 3× larger — which is itself evidence the mean **does** behave differently; it
> is simply not consistently better.
>
> Cost to actually settle it at $M = 8$, at 80% power: **12 paired seeds (73 GPU-h)** for a 1.0 sd
> effect, 47 seeds (291 GPU-h) for 0.5 sd. **Not worth it.** If the question returns, cut the per-seed
> variance (larger $M$ — its SE measurably halved) rather than adding seeds at $M = 8$.
>
> **Always compute the detectable effect size BEFORE choosing the number of seeds.**
>
> **The general lesson, and the one worth carrying:** a single-seed Algorithm 3 result at $M = 8$ is
> not evidence. The per-run Monte-Carlo SE alone is $\approx 0.5$ sd, seeds flip sign, and every
> "mechanism" metric measured so far (uniq/M, sd(U)/sd(V)) fails to predict the outcome. **Do not
> report an $M = 8$ single-seed effect as a result again.**
>
> ## Seed 101 in isolation (SUPERSEDED by the above — kept for the record)
>
> Job 872813, everything identical to the k-sweep's $K{=}4$ cell except the aggregation:
>
> | | k04 (soft) | **mean** |
> |---|---|---|
> | terminal $E_q[f]$ | 1.0000 | **1.0111** |
> | vs control | $+0.80$ sd | **$+2.35$ sd** |
> | **arm-vs-arm** (control cancels exactly) | — | **$+1.55 \pm 0.52$ sd** |
> | uniq/M | 0.125 | **0.250** |
> | collapse | step 3 | **never** |
> | sd(U)/sd(V) | 0.534 | **0.364** |
>
> ```
> uniq/M   mean:  1.00 1.00 1.00 0.50 0.38 0.38 0.38 0.38 0.38 0.25 0.25 ...
>          soft:  0.75 0.25 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 ...
> ```
>
> Beats the previous best at $M = 8$ ($+1.49$ sd, uniq 0.375, at $K{=}8$) at **56% of the cost**.
>
> **The pre-run prediction was wrong.** Spearman(mean, soft) $= 0.92$–$0.98$ was read as "expect
> little", by analogy with the max's null. That reasoning is invalid: the rank correlation is measured
> at a **fixed cloud**, but the aggregation changes *which particles survive*, so a small per-step
> ordering difference compounds through selection. The max didn't show this because it made
> `sd(U)/sd(V)` *worse* — the compounding ran against it. **A high rank correlation at fixed cloud does
> not bound the effect under resampling.**
>
> **Caveat: one seed.** Replication at seeds 102/103, paired mean-vs-soft, is jobs 873199–873202. The
> paired difference needs no new control (it cancels exactly), so a seed costs two jobs, not three.
>
> ## AND: `defer25` isolates *why* — it is not diversity
>
> | arm | uniq/M | collapse | vs ctrl | sd(U)/sd(V) | sd(logw) |
> |---|---|---|---|---|---|
> | k04 (soft) | 0.125 | step 3 | $+0.80 \pm 0.48$ | 0.534 | 0.735 |
> | **mean** | 0.250 | never | $\mathbf{+2.35 \pm 0.59}$ | 0.364 | 0.499 |
> | **defer25** | 0.250 | never | $+0.67 \pm 0.78$ | 0.368 | 0.181 |
>
> **`mean` and `defer25` are matched on every mechanism metric and differ by 1.7 sd on the outcome.**
> Identical `uniq/M`, both avoided collapse, near-identical `sd(U)/sd(V)`. One gained $+1.55$ sd over
> the baseline; the other gained nothing.
>
> 1. **The mean's gain is not diversity preservation.** Deferring bought the same lineage survival and
>    converted it into zero novelty. The mean changed *which* particles selection backs — consistent
>    with the deleted convexity term being negatively correlated with $f(t{=}1)$ — not how many survive.
> 2. **$\operatorname{sd}(U)/\operatorname{sd}(V)$ does not predict the outcome.** Both arms cut it by
>    the same 32%, with opposite results. It was the plan's central acceptance criterion ("the number
>    that says CRN did something $K$ could not"). A0 showed CRN could barely move it; this shows moving
>    it implies nothing about novelty. **Do not gate a future experiment on this metric alone.**
>
> Caveat: one seed per arm, and defer25's $\pm 0.78$ is the widest interval in the table.
>
> ## Complete seed-101 table (all four arms finished)
>
> | arm | uniq/M | collapse | vs own control | SE | cost |
> |---|---|---|---|---|---|
> | **mean** ($M{=}8$) | 0.250 | never | **$+2.35$** | $\pm 0.59$ | 3.1 h |
> | m16 (soft, $M{=}16$) | 0.188 | never | $+1.26$ | $\pm 0.31$ | 6.5 h |
> | k04 (soft, $M{=}8$) | 0.125 | step 3 | $+0.80$ | $\pm 0.48$ | 3.1 h |
> | defer25 ($M{=}8$) | 0.250 | never | $+0.67$ | $\pm 0.78$ | 3.1 h |
>
> $M = 16$ works — no collapse, and the tightest interval in the table, since the SE halves as
> $1/\sqrt{M}$ exactly as predicted. But it delivers **half the mean arm's effect at twice the cost**.
>
> **THE LOAD-BEARING OBSERVATION.** Three independent interventions — a different aggregation, a
> deferred resampling schedule, and double the particles — **all prevented collapse and all cut
> $\operatorname{sd}(U)/\operatorname{sd}(V)$ to $\approx 0.36$ from 0.534**, yet their novelty spans
> $+0.67$ to $+2.35$ sd. So neither degeneracy nor the increment-noise metric explains the outcome, and
> **neither should gate a future experiment**. Both were the plan's guiding metrics. What separates the
> arms is *which particles the potential ranks highest*, which is the one thing only the aggregation
> changed.

All at $\lambda = 175.116$ ($m = 1.25$, $\lambda_s = 140.093$), $K = 4$, $N = 16$, seed 101,
`ess_threshold=0.5`, `guid_window=(0,1)`, `stoch_window=(0.1,1.0)`, $R = 64$ refs
(`refs_R64_seed7.pt`, md5 `711880aade334889a6b730c00f5ed48f`), FLUX.1-dev + flowmap LoRA, prompt
"A dog", **L40S only**, `--mem=24000`. Everything except the named variable is frozen.

| job | arm | varies | baseline it is paired against | cost |
|---|---|---|---|---|
| 872811 | `m16` | **$M = 16$** (from 8) | none — new $M$, needs its own control | ~6.1 h |
| 872812 | `ctrl16` | $\lambda = 0$ at $M = 16$ | — it *is* the control for 872811 | ~1.4 h |
| 872813 | `mean` | **mean aggregation** | `flowmap_smc_k_sweep/result_N8_K4_m1.25_seed101_k04.pt` | ~3.1 h |
| 872832 | `defer25` | **no resampling below $t = 0.25$** | same k04 cell | ~3.1 h |

`mean` and `defer25` are at $M = 8$ **on purpose**: that makes the k-sweep's $K{=}4$ cell an exact
paired baseline (terminal 1.0000, $+0.81\operatorname{std}_p(f)$, `uniq/M` 0.125, collapse at step 3,
$\operatorname{sd}(U)/\operatorname{sd}(V)$ 0.534) and lets them reuse `result_N8_K1_m0_seed101_ctrl2.pt`
as the $\lambda = 0$ control — no extra GPU, and no cross-$M$ comparison to defend.

## What each arm is testing

**`mean` (872813).** The aggregation, not the lookahead, is what broke $K$: the mean's estimator error
falls like $k^{-1.0}$ where the soft value manages $k^{-0.6}$, and the term it deletes
($\lambda\sigma_k^2/2$) is *negatively* correlated with $f(t{=}1)$ in 7 of 11 cells while carrying
8–30% of $\operatorname{sd}(V)$. It also removes the Jensen exposure entirely, which is what matters at
strong tilt. **Expect a small terminal effect** — Spearman(mean, soft) = 0.92–0.98, the same signature
that preceded the max's null. Judge it on `uniq/M`, on $\operatorname{sd}(U)/\operatorname{sd}(V)$, and
on whether a later $K$ sweep through it finally shows $1/\sqrt K$.

**`defer25` (872832).** Selection is off for every step landing below $t = 0.25$. Two measurements
point there: the potential's ordering agrees with the true value function at Spearman 0.70–0.82 for
$t \le 0.25$ (and 0.99+ after), and **every collapsed cell collapsed at step 3, $t = 0.1875$** — inside
that window. Nothing is discarded by waiting: $U$ keeps accumulating, and between resamples
$U_n = V_n - V_{\text{last}}$ is a long-baseline difference rather than the noise-dominated single-step
increment. **Read `sd(logw)`** — deferring leaves weight where selection would have spent it, so the
returned ensemble is *not* equally weighted.

**`m16` + `ctrl16`.** The plan's fallback: every failure to date was degeneracy at $M = 8$. A
mitigation rather than a mechanism (`uniq/M` is monotone non-increasing at any $M$), but it attacks
the thing that actually broke.

## Reading the results

1. **`uniq/M` at the final step** first, against 0.125 (k04) and the best-ever 0.375.
2. **Terminal $E_q[f]$, control-subtracted** — never raw; the $\lambda = 0$ control alone gives back
   $3.7\operatorname{std}_p(f)$ and only $t = 1$ is artifact-free.

   **The bar is ±0.48 $\operatorname{std}_p(f)$, not 0.17.** A control-subtracted terminal is a
   *difference of two Monte-Carlo estimates* and carries both errors: the $M{=}8$ control's own SE is
   $\operatorname{std}(f_{\text{proj}})/\sqrt{8} = 0.43\operatorname{std}_p(f)$, giving a paired SE of
   $\approx 0.48$. `compare_arms.py` prints it per arm.

   This is ~3× the 0.17 seed sd the plan's acceptance criteria were written against, and the two are
   not in conflict — 0.17 was estimated from **3 seeds**, and an sd from $n = 3$ carries ~50% of its
   own value, so it is compatible with a true 0.3–0.5. The per-run SE is computed from the particles'
   actual spread and is far better determined; **use it**.

   Consequence for what is already on record: of the K-sweep's $+0.81 / +1.49 / +0.59 / +1.23$, **only
   $K{=}8$'s $+1.49$ clears 2 SE.** The "reference to beat" is a $\approx 3\sigma$ result and the rest
   of that row is noise. Tonight's arms need to clear $\approx +1.0$ to mean anything at $M = 8$ — which
   is also, independently, an argument for $M$: the SE falls like $1/\sqrt{M}$.

   Live confirmation of the same effect: at step 8 the $M{=}16$ control read 1.0224 against the
   $M{=}8$ control's 1.0130 — $1.3\operatorname{std}_p(f)$ apart, from particle sampling alone.
   **Never compare an arm to a control at a different $M$.**
3. **$\operatorname{sd}(U)/\operatorname{sd}(V)$** against k04's 0.534.
4. `mean` and `defer25` carry **identical particles to k04** until the first step where their
   resampling decisions differ, so `f_cand` must match k04 exactly before that step. If it does not,
   something other than the intended variable changed and the comparison is void.

## Operational, 2026-09-10 (each cost real hours)

* **n-802 killed two jobs with hardware faults in one session** — `mean103` SIGBUS (exit 135) at step
  5, `soft103` SIGSEGV (exit 139) at step **13 of 16**. Probed at the time of the first fault: `/tmp`
  had **111 GB free** and a **complete 27/27 file stage**, so it was NOT disk exhaustion or a truncated
  mmap. `flowmap_smc_max/TAKEAWAYS.md` already recorded "n-802 SIGSEGV'd two jobs in one night". **Treat
  n-802 as suspect and `--exclude` it**; two independent sessions is not bad luck.
* **All four replication jobs were preempted simultaneously** at 09:20 and requeued from step 1,
  costing ~10 GPU-h. `killable` preemption is called "tolerable" in the k-sweep header because a
  partial run still delivers the pre-collapse window — **that reasoning does not hold for a paired
  seed comparison**, which needs terminals. A partial contributes nothing to the CI.
* **Determinism survived all of it, verified three ways.** `mean103` step 1 was bit-identical across
  the original run, the preemption restart, and the post-SIGBUS resubmission *on a different node*
  (`E_q[f]=0.9215`, `f_cand=0.9176` every time). Same for `soft102`. So a fault costs wall clock and
  nothing else, and a dead run's partial is a verifiable prefix of its replacement.
* **`/tmp` free space and stage file count are worth probing before blaming the node** — it took one
  5-minute CPU job to rule out the two documented disk failure modes and correctly identify this as
  hardware.

## Code

* `creativity_measure/flowmap_smc_mean.py` — `_soft_value` swap, the proven `flowmap_smc_max` pattern.
* `creativity_measure/flowmap_smc_defer.py` — carries a **copy** of the step loop (the resampling
  decision is an inline predicate, and faking `_ess_from_logw` would corrupt `ess_history`). Pinned by
  `tests/test_flowmap_smc_defer.py::test_full_window_is_bit_identical_to_the_base_sampler`.
* `creativity_measure/flowmap_smc.py` is **not modified**. 79 tests pass across the four sampler test
  files; `pyright` clean.
