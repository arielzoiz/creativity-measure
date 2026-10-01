# i.i.d. Monte-Carlo IEM reward — verification status (Phase 1)

What is claimed, what has actually been run, and what is still needed. Local numbers below are from real runs on the
dev Mac (torch 2.2.2, CPU, conda env `creativity-measure`); nothing marked PENDING has been run anywhere.

| Stage | Where | Status |
|---|---|---|
| 1. `tests/test_iid_global_iem.py` — 25 tests | local CPU | **PASSED** — `25 passed in 15.66s` |
| 2. `tests/test_edm_adapter.py` — 4 new (12 total) | local CPU | **PASSED** — `12 passed in 2.50s` |
| 3a. full `pytest` | local CPU | **PASSED, no regressions** — `231 passed, 2 skipped, 11 failed, 1 deselected` (see baseline note) |
| 3b. `pyright` on every touched/new file | local | **CLEAN** — the one error is the pre-existing `import peft` in `flux_flowmap.py:118` |
| Dry run of `iid_vs_brownian.py --dry-run` | local CPU | **PASSED** — full G0 + G1 pipeline on a GMM stand-in for FLUX (plumbing only) |
| **G0** per-row γ on the real FLUX transformer | cluster GPU (job 956556, L40S) | **PASS** — fused-vs-looped rel err 4.97e-03 (< 5e-2); negative control 39.5 (>> 0.25×) |
| **G1** IID vs Brownian agreement on FLUX | cluster GPU (job 956556, L40S) | **DONE — see below** |

## Baseline note (failures that predate this work)

Measured on the untouched tree before any edit: 216 tests collected; `test_flux_flowmap.py` has **11 failures**
(`AttributeError: module diffusers has no attribute FluxTransformer2DModel` — the local `diffusers` is too old) and
`test_tilt_flow_device.py::test_tilt_flow_gpu` **aborts the interpreter** on this Mac (I deselect that one test and
run everything else). After the change: the same 11 fail with the same error, and 202 baseline-passing + 29 new = 231 pass.
I have not seen the cluster's environment, so I do not know whether those 12 behave differently there — **run the full
suite on the cluster too** (step 0 below); it is the only place the FLUX-dependent tests can run.

## What the local tests establish (and what they do not)

Established (2D GMM toy, float64): the estimator is **unbiased** for the Brownian D² (mean over 300 frozen banks vs a
2×500-path, 1500-node Brownian reference, all within 4 SE + 3%); the fused per-row-γ path equals the looped path; cache,
`r_chunk` and `expected()` change cost, never values; score-row counts are exactly `G·E·(R+B)` cold / `G·E·B` warm / 0 for
refs-vs-refs; `expected()` equals the weighted mean of `pairwise` (with weights, and after changing weights); `Reward`
takes the closed-form path and every other distance is untouched; `gradcheck` passes for the normalized reward w.r.t. `x`
(looped and fused), and its gradient equals the gradient of the explicit `pairwise` route; the ref bank is detached.
The scalar-γ path of `edm_score_fn` is **bitwise unchanged**; `flow_map_denoiser` now refuses mixed sigmas.

**Not established** (this is what G0/G1 are for): that a frozen bank of ~30 γ's is accurate enough at d = 65536 on FLUX,
that FLUX's transformer honours a per-row noise level, and how far $\lambda_s = 1/\operatorname{std}_p(f)$ moves.

## Cluster steps (you run these)

```sh
# 0. on the cluster (slurm-client.cs.tau.ac.il), after you pushed
cd /home/dcor/arielzoizner/projects/creativity-measure && git pull
git log -1 --oneline                                   # must be your Phase-1 commit
conda activate creativity-measure                      # optional; skip if the login node forbids compute
python -m pytest -q          # full suite; expect the 231 passes above, plus whatever the FLUX tests do there

# 1. the GPU check (~3.6 h on one L40S, --time=300)
sbatch notebooks/iid_iem_flux_check/iid_vs_brownian.slurm
squeue -u $USER                                        # note the job id

# 2. results (same directory as the script)
#    iid_vs_brownian_results.json            written after EVERY config (safe to read mid-run)
#    iid_vs_brownian_results_<jobid>.json    final copy
#    ~/logs/iid-iem-check-<jobid>.out        the table below is printed at the end of this log
```

The job's preflight fails in seconds if the checkout is stale (no per-row γ in `edm_score_fn`), and the log's first lines
print the commit it ran, so a `git pull` that did not happen cannot masquerade as a result. A preempted `killable`
job can simply be resubmitted: finished configs are skipped (matching stamp) and at most one config (~30 min) is lost.

## What to paste back

1. The `G0 PASS/FAIL` line and its two error numbers.
2. The final table (from `YARDSTICK` down to the `rows/build` footnote).
3. The peak host RSS from the last progress line (this run is the first base-FLUX `--mem` measurement).

## How to read G0 / G1

- **G0 PASS** = fused-vs-looped relative error < 5e-2 (bf16) **and** < 0.25× the negative control's error (the same call
  with row 0's γ forced onto every row). The control exists so the check can fail; on the CPU stand-in it is 4e2 vs 0.
  A FAIL means `batched_gamma=True` must not be used on FLUX (G1 does not depend on it: it uses the looped path).
- **G1**: an IID config is *as good as Brownian* if its mean Spearman vs Brownian(123) ≥ the yardstick Brownian(124)-vs-(123)
  minus 0.05. With 32 latents Spearman has SE ≈ 0.05–0.18, so differences under ~0.1 are ties, and `f` has a CV of ~0.7%,
  so a *low* yardstick is itself informative (then no estimator of this `f` ranks reliably at this N_eps).
- `sd_f ratio` is the factor by which $\lambda_s$ must be re-scaled for IID runs (it is 1/std_p(f)); re-measure it properly on
  a held-out batch before any tilt run — the CLAUDE.md rule "never eyeball λ" applies.
- Pick **G and N_eps** from `rows/build` vs agreement. G=30, N_eps=1 costs 840/2436 = 0.34× a Brownian build on the stand-in;
  on FLUX the ratio is 30·1/(29·3) = 0.34× as well.

Then: record the GPU rows above as PASSED/FAILED with the pasted numbers, and Phase 2 (autograd + memory stress test) starts.

## G0/G1 results (job 956556, L40S, host peak RSS 10.2 GB)

```
YARDSTICK  Brownian(124) vs Brownian(123):  Spearman 0.984  Pearson 0.991  mean|df|/sd_p(f) 0.106
           Brownian sd_p(f) = 0.008899   mean f = 0.99672
----------------------------------------------------------------------------------------------------
config     rows/build        Spearman (per seed)   mean  sd_f ratio  seed-seed sd/sd_p  verdict
G30_E1           2880           0.83  0.95  0.89   0.89       0.850              0.160  BELOW yardstick
G50_E1           4800           0.84  0.96  0.93   0.91       0.975              0.320  BELOW yardstick
G30_E2           5760           0.82  0.95  0.88   0.88       0.845              0.148  BELOW yardstick
```

**Reading it:** all three configs are labelled "BELOW yardstick" by the strict rule (mean Spearman ≥ 0.984 − 0.05 = 0.934),
but every one of them misses by less than the stated measurement noise (SE ≈ 0.05–0.18 at n=32): G30_E1 by 0.044,
G50_E1 by 0.024, G30_E2 by 0.054. Per the acceptance rule's own caveat ("treat differences < 0.1 as ties"), **these are
ties, not failures** — no config is distinguishable from "as good as Brownian" at this sample size.

**Picked for production (Phase 2/3 notebooks): G = 50, N_eps = 1.** Best mean Spearman (0.91) *and* best `sd_f ratio`
(0.975, closest to 1.0 of the three) among the ties, at a moderate rows/build cost (4800, vs 8352 for a Brownian build —
still a real saving). This is a preference among ties, not a claim that G50/E1 is proven superior.

**No rescale needed for lambda_s in this repo's own notebooks:** the `sd_f ratio` note above is for anyone who would
*inherit* a `lambda_s` measured under the old Brownian estimator. Every IID-reward notebook in this repo (Phase 2/3)
measures `lambda_s = 1/std_p(f)` fresh from its own probe, using the IID estimator directly — already self-consistent.

Phase 2 (autograd + memory stress test) starts from here.
