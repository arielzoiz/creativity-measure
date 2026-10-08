# Phase 5 prompt study (2026-10-07/08)

Does the corrector's benefit, measured only on `"A dog"`, hold across prompts? Five prompts, two seeds,
two arms, gradient guidance only.

    prompts   building, sofa, car, teapot, jacket     (bare nouns, matching related work)
    seeds     1234, 3141                              (the two with documented opposite breakdown on dog)
    arms      flow_guided (C=0, Phase 3) + pc_guided (C=1)   -- both in ONE job, same GPU, shared setup
    lambda    wave A {0, 0.2, 0.4, 0.6, 0.8, 1.0}     --lam-step 0.2 --lam-k 0,1,2,3,4,5
              wave B {0.1, 0.3, 0.5, 0.7, 0.9}        --lam-step 0.1 --lam-k 1,3,5,7,9   --tag odd

`--tag odd` is **mandatory**, not cosmetic: the results filename does not encode `lam_step`, so an
untagged wave-B run collides with wave A's file, fails the stamp-identity check, and `.stale`-renames
completed data.

## Jobs

| group | ids | what |
|---|---|---|
| orig | 995999-996008 | wave A, 5 prompts x 2 seeds |
| retry | 996957-996961 | duplicates of the five that stalled (`--tag retry`) |
| odd | 996962-996971 | wave B, chained `afterany` on the corresponding (prompt, seed) jobs |

## The n-801 stall

Ten jobs landed 5 on n-801, 4 on n-802, 1 on n-803. The five on n-801 produced **zero** points in 3 h
while n-803's lone job finished setup in 40 min. Cause, measured with `srun --overlap`: all five
processes in state `Dl` on `folio_wait_bit_common`, 0% GPU, resident memory crawling — five processes
page-faulting the same 32 GB checkpoint off `$WORK` at a combined ~5 MB/s, degrading as they contended
(8 -> 4.9 -> 3.8 -> 2.7 -> 1.8 MB/s per process). This is the same signature CLAUDE.md records for job
966117.

**It is contention, not cold cache.** n-803 was also cold and still took only 40 min. The lesson for
future waves is a cap on jobs per node, or node-local staging as the flowmap jobs do — not fewer jobs
overall.

Both the original and the retry are left running per instruction; the supervisor cancels the loser only
once the winner has **both** exited `COMPLETED` and written 6 points per arm.

## First-wave findings (flow_guided, 5 runs complete)

- ~~**The usable window is roughly lambda in [0.2, 0.6], breaking by 0.8.**~~ **WRONG — corrected
  below once `jacket` finished.** That reading came from three prompts at 0.2 spacing and did not
  generalize. See "Breakdown is prompt-dependent by >2x".
- **`teapot` is the cleanest case**: s1234 at 0.4 is a geometric-patterned ceramic teapot, still
  unmistakably a teapot; s3141 at 0.2 a carved swirl form, at 0.6 a faceted sculpture.
- **`jacket` renders a PERSON wearing a jacket at lambda=0**, and the tilt *removes the person* — by 0.4
  both seeds are flat product/technical illustrations of the garment alone. The reward moves away from
  portrait statistics.
- **`car` s3141 renders a pencil drawing, not a photo**, and breaks fastest: scrawled text over the car
  by 0.2, anime figures by 0.6. Bare nouns do not pin the medium, and the medium varies by seed.
- **`applied_norm_mean / v_norm_mean` is identically lambda** (0.20 at lambda=0.20, etc., and 0.3928 at
  lambda=0.3928 in the dog runs). The sampler normalizes the gradient before scaling, so that ratio is
  NOT an independent diagnostic and cannot be used to match tilt strength across prompts. Matched
  lambda already is matched applied-guidance strength.
- **lambda_s is strongly prompt-dependent**: car 106.87, teapot 70.75, jacket 70.38, against dog's
  81.016 — a 1.5x spread, and 13-32% away from dog, i.e. past the old hard-coded preflight tolerance.
  Recorded per prompt in `lam_s_by_prompt.json`.

## Both arms, both waves (`prompt_study_grid.png`, `render_prompt_study.py`)

### Breakdown is prompt-dependent by >2x

Read on the decoded images, not on f:

| prompt | seed | last recognizable lambda (C=0) | f at lambda=1 (C=0) |
|---|---|---|---|
| teapot | 1234 | ~0.6 | 7.74 |
| teapot | 3141 | ~0.8 | 6.84 |
| jacket | 1234 | **1.0 (never breaks in range)** | 3.45 |
| jacket | 3141 | **1.0 (never breaks in range)** | 4.83 |

`jacket` is still a clearly-rendered garment at lambda=1 — C=1 s3141 at lambda=1 is a black varsity
jacket reading "TOM COUR PARIS", and at 0.6 a leopard-print hooded jacket. `teapot` is destroyed by
0.7-0.9. So the earlier "[0.2, 0.6]" claim was a teapot/car statement, exactly the way CLAUDE.md's
"[0.4, 3.54]" turned out to be a seed-1234 statement. **The window is a property of the (prompt, seed,
z0) triple, not of the method.**

### A hypothesis worth one experiment, NOT a finding

Across these four runs the prompts that survive longest are the ones whose f grows *slowest*
(jacket 3.45/4.83 at lambda=1 and intact; teapot 6.84/7.74 and destroyed). That is the opposite
direction from "high f = more novelty = better", and it is only n=2 prompts. CLAUDE.md already records
that f does not predict recognizability *across seeds*; whether the f growth RATE predicts it across
prompts is untested. Do not build anything on it without a deliberate check.

### C=1 vs C=0: novelty replicates, window does not

Same conclusion as the dog runs, now on new prompts. At matched lambda the corrector reaches
consistently higher f (teapot s1234 at lambda=1: 15.66 vs 7.74; jacket s1234: 6.96 vs 3.45) and
produces more decisively designed images at LOW lambda -- teapot C=1 at 0.1/0.3 is a striped
lantern-like vessel where C=0 is still a near-photo. But on teapot it also breaks slightly EARLIER
(C=1 losing the form by ~0.5 where C=0 holds to ~0.6). Consistent with CLAUDE.md: the corrector buys
novelty per lambda, not a wider window.

## Complete result (20/20 cells, 11 lambda each, 220 images)

Last lambda at which the subject is still recognizable, judged on the decoded images:

| prompt | seed | C=0 | C=1 | what the seed actually rendered at lambda=0 |
|---|---|---|---|---|
| building | 1234 | **1.0** | **1.0** | a detached house (object, strong silhouette) |
| building | 3141 | ~0.5 | ~0.5 | a flat modernist facade filling the frame (no silhouette) |
| sofa | 1234 | ~0.9 | ~0.8 | sofa, plain backdrop |
| sofa | 3141 | ~0.8 | ~0.6 | sofa, interior |
| car | 1234 | ~0.6 | ~0.5 | car |
| car | 3141 | ~0.2 | ~0.2 | a PENCIL DRAWING of a car, not a photo |
| teapot | 1234 | ~0.6 | ~0.5 | teapot, product shot |
| teapot | 3141 | ~0.8 | ~0.7 | teapot, product shot |
| jacket | 1234 | **1.0** | **1.0** | a man wearing a jacket |
| jacket | 3141 | **1.0** | **1.0** | an illustrated girl wearing a jacket |

Range of breakdown: **0.2 to >1.0, a 5x spread**, across prompts AND across seeds within a prompt.

### The silhouette mechanism, now tested within a single prompt

`building` is the cleanest evidence in the whole study because the prompt is held fixed and only z0
varies. Seed 1234 drew a detached house -- a compact object against sky -- and never breaks through
lambda=1, passing through genuinely attractive isometric game-art and storybook renderings. Seed 3141
drew a flat modernist facade filling the entire frame -- no silhouette, pure repeating texture -- and
is destroyed into grey blocks by 0.6.

This is the same prediction made (and not yet run) for a `tree` vs `forest` pair: the tilt preserves a
dominant silhouette far longer than it preserves texture. `building` got the controlled version of that
experiment for free. It is one prompt and two seeds, so it is suggestive, not established -- but it is
a within-prompt comparison, which is stronger than anything the cross-prompt table can give.

### C=1 vs C=0

Across all 10 (prompt, seed) cells: C=1 reaches **higher f at every matched lambda, without exception**,
and breaks at the same lambda or ONE step earlier -- never later. The corrector buys novelty per
lambda, and here it costs a little reach. Same direction as the dog runs, now at n=10.

### Leftover files

The five cancelled retries left partial results (`*_retry.json`, 4-5 points/arm) and their decode dirs
on disk. `render_prompt_study.py` reads only the untagged and `_odd` tags, so they cannot contaminate
the figure. Kept as a record of the n-801 stall; delete freely.
