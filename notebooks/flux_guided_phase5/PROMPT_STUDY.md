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

- **The usable window is roughly lambda in [0.2, 0.6], breaking by 0.8** — far narrower than dog's
  (recognizable to 2.36 on seed 1234). Breakdown is strongly prompt-dependent, so `lambda <= 1` was the
  right ceiling and wave B's 0.1 spacing is where the signal is.
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
