# Task: fill out the corrector_steps x lambda grid for Phase 5, across the 5 existing seeds

Paste this whole file as the prompt. Repo: `/home/dcor/arielzoizner/projects/creativity-measure`,
conda env `creativity-measure`, cluster partition `killable` / account `gpu-research`.
Everything needed already exists and is committed — this is a run-and-analyse task, not a build task.

---

## What Phase 5 established (don't re-derive)

`flow_guided_pc_sample` interleaves ULA corrector steps with Phase 3's guided ODE. Over 5 seeds, ~25 GPU-h:

- **Novelty gain replicates, 9/9 paired comparisons**: at matched λ and matched `n_steps=10`, PC gives
  +14% to +58% higher `f` than Phase 3 (mean +31% at λ=0.79, +38% at λ=1.18), with recognisability
  preserved in 4/5 images and visibly diverse restyles.
- **The λ window does NOT widen**: 1 seed supports it, 1 contradicts it (seed 4242 at λ=1.18 renders the
  digits "97" while the control still shows a creature face), 2 neutral.
- `predictor_guided=False` does nothing at any λ; `eta_reference="score"` is strictly worse where images
  survive. Both axes are closed — **do not re-test them.**

Full write-up: `notebooks/iid_iem_flux_check/ROADMAP.md`, Phase 5 section.

## The gap this task closes

**All of that is at `corrector_steps=1`.** `corrector_steps=2` was run only at λ = 3.93 / 5.5 / 7.07 —
every one past the breakdown point, i.e. on already-destroyed images. There is **no C>1 data in the λ
range where images survive**, which is exactly where the corrector's entire measured benefit lives.

Open question: does more corrector iteration compound the tilt, or saturate because the
`λ/(1+λ)²` cap shrinks each step's contribution? The `disp` numbers on file are all from the annealed
regime and don't extrapolate. Both outcomes are publishable; the point is that nobody knows.

## Target lattice: all 6 points, λ = 0.393 / 0.786 / 1.179 / 1.571 / 1.964 / 2.357

(`lam_k` = 1,2,3,4,5,6; `LAM_STEP = 5.5/14`.) This is wider than the first draft of this task (which only
went to `lam_k` 1-4, i.e. λ≤1.571) — six points gives more coverage of the per-seed survival range
(breakdown varies ~3x across seeds, up to λ≈2.4 for seed 1234) without re-deriving a new grid.

## What already exists at this lattice — checked against the actual files, don't re-trust the filenames

**Phase 3 control (`flow_guided`, the C=0 comparator): fully covered, no new runs needed.**
`../flux_guided_phase3/fine_lambda_sweep_results_seed{2024,3141,4242,5555}.json` and
`fine_lambda_sweep_results_max5.5.json` (seed 1234) already contain all 6 target λ values, same config
(`n_steps=10`, `NVIDIA L40S`, confirmed from each file's `stamp`). Ignore
`pc_sweep_results_flow_guided_n10_c1_etatotal_snr0.16_s1234.json` in this dir — it's a different,
stray run at λ={6.29, 7.07, 7.86}, not a substitute for the Phase 3 control.

**C=1 (`pc_guided`, corrector_steps=1): split coverage, incomplete for every seed.** Checked each
`pc_sweep_results_pc_guided_n10_c1_etatotal_snr0.16_s<SEED>.json`'s `stamp.lam_k`:

| seed | has (lam_k) | has (λ) | missing (lam_k) | missing (λ) |
|------|-------------|---------|------------------|--------------|
| 1234 | 0,2,4,6,8,...,20 | 0.786, 1.571, 2.357 | 1,3,5 | 0.393, 1.179, 1.964 |
| 2024 | 1,2,3 | 0.393, 0.786, 1.179 | 4,5,6 | 1.571, 1.964, 2.357 |
| 3141 | 1,2,3 | 0.393, 0.786, 1.179 | 4,5,6 | 1.571, 1.964, 2.357 |
| 4242 | 1,2,3 | 0.393, 0.786, 1.179 | 4,5,6 | 1.571, 1.964, 2.357 |
| 5555 | 1,2,3 | 0.393, 0.786, 1.179 | 4,5,6 | 1.571, 1.964, 2.357 |

**C=2: only seed 1234 exists, and only at λ={3.929, 5.5, 7.071}** (`pc_sweep_results_pc_guided_n10_c2_etatotal_snr0.16_s1234.json`,
`lam_k=[10,14,18]`) — this is exactly the past-breakdown gap CLAUDE.md flags. Needs the full 6-point
lattice, all 5 seeds.

**C=4: no existing data at all.** Needs the full 6-point lattice, all 5 seeds.

## What to run

Three groups of jobs, **all requesting the full 6-point lattice** (`--lam-k 1,2,3,4,5,6`) — do NOT try to
request only the missing points for Group 1. `pc_sweep.py`'s `load_results` (lines 157-167) only resumes
into an existing file when the ENTIRE stamp dict matches, and `lam_k`/`lam_grid` are part of that stamp;
a different `--lam-k` triggers a stamp mismatch, renames the existing file to `.stale`, and starts a fresh
file containing *only* the newly requested points — it does not append/merge. (The stamp also embeds
live `preflight` measurements such as `nondet_floor_lam1`, which are themselves nondeterministic run to
run per CLAUDE.md, so even re-running identical args is not guaranteed to hit the resume path.) Requesting
the full lattice every time sidesteps this entirely and costs about the same as the partial-completion
approach once setup overhead is counted.

```bash
cd /home/dcor/arielzoizner/projects/creativity-measure/notebooks/flux_guided_phase5
export HF_TOKEN=$(cat $HOME/.hf_token)          # required; never echo or write it to $WORK

# Group 1: C=1 completion -- full 6-point lattice, all 5 seeds (overwrites each seed's partial C=1 file;
# the old partial file survives as .stale, nothing is lost)
for S in 1234 2024 3141 4242 5555; do
  sbatch --time=280 --job-name="p5-c1-s${S}" \
    pc_sweep.slurm pc_guided --corrector-steps 1 --lam-k 1,2,3,4,5,6 --sweep-seed $S
  sleep 60
done

# Group 2: C=2, full 6-point lattice, all 5 seeds
for S in 1234 2024 3141 4242 5555; do
  sbatch --time=380 --job-name="p5-c2-s${S}" \
    pc_sweep.slurm pc_guided --corrector-steps 2 --lam-k 1,2,3,4,5,6 --sweep-seed $S
  sleep 60
done

# Group 3: C=4, full 6-point lattice, all 5 seeds
for S in 1234 2024 3141 4242 5555; do
  sbatch --time=560 --job-name="p5-c4-s${S}" \
    pc_sweep.slurm pc_guided --corrector-steps 4 --lam-k 1,2,3,4,5,6 --sweep-seed $S
  sleep 60
done
```

`--time` (minutes) comes from `setup(~90min) + N_lambda * (10 + 9*C) * 57s`, with a ~1.5x buffer on top
(the same formula the original draft of this task used, re-derived per group since N and C both changed,
N=6 for all three groups now):
- Group 1 (N=6, C=1): 6*19*57s ≈ 108min + 90min setup ≈ 198min needed → `--time=280`
- Group 2 (N=6, C=2): 6*28*57s ≈ 160min + 90min setup ≈ 250min needed → `--time=380`
- Group 3 (N=6, C=4): 6*46*57s ≈ 262min + 90min setup ≈ 352min needed → `--time=560`

**Estimated cost** (setup + compute, no buffer, converted to GPU-h, x5 seeds per group):
- Group 1: (1.5h + 1.08h) * 5 ≈ 12.9 GPU-h
- Group 2: (1.5h + 2.66h) * 5 ≈ 20.8 GPU-h
- Group 3: (1.5h + 4.37h) * 5 ≈ 29.4 GPU-h
- **Total ≈ 63 GPU-h.** (Larger than the original draft's 23 GPU-h estimate — that draft covered only 4
  λ points and didn't account for C=1's split/incomplete coverage. If you want a cheaper first cut, run
  Group 1 + Group 2 only and gate Group 3 on whether C=2 shows a gain over C=1 once both are filled in.)

Use the `slurm-jobs` skill and arm its watchdog.

## How to analyse

```bash
python render_headline.py          # PC vs Phase 3, paired per seed  -> pc_headline_grid.png
python render_pc_comparison.py     # all arms browser                -> pc_comparison_grid.png
```

`render_pc_comparison.py` discovers rows by globbing results files, so the new C=2/C=4 runs appear
automatically. `render_headline.py` currently filters to `corrector_steps == 1` (see `_pc_rows`) — widen
that filter, or add a C column, to put C=1/2/4 side by side at matched seed and λ.

Compare against, at the same seeds and λ:
- **C=1**: `pc_sweep_results_pc_guided_n10_c1_etatotal_snr0.16_s<SEED>.json` (post Group 1, each seed's
  file was fully replaced with a fresh 6-point run — see "What to run" for why the old partial file isn't
  reused; it survives alongside as `.stale` but the active filename has all 6 points after Group 1).
- **Phase 3 control**: `../flux_guided_phase3/fine_lambda_sweep_results_seed<SEED>.json` and
  `fine_lambda_sweep_results_max5.5.json` (seed 1234), with images in `fine_decoded_seed*/`.
  These are `n_steps=10` on L40S over the same λ lattice, already covering all 6 target points —
  **do not re-run them.**

## Gotchas that cost real GPU-hours last time

1. **LOOK AT THE DECODED PNGs.** No scalar predicts recognisability — not `f`, not `‖x‖/√d`, not `hf`.
   `f` = 14.2 on a clear dog and 14.8 on pure noise; `‖x‖` is *anti*-correlated with quality; `hf` scores
   0.0089 on a dog and 0.0093 on a destroyed mosaic. Images are written per point, free to inspect.
   Four hours went into scalar analyses pointing the wrong way before anyone opened an image.
2. **Never raise `n_steps` to build a "compute-matched" control.** More ODE steps *destroys* guidance
   (`n_steps=19` is abstract blocks at λ=1.571 where `n_steps=10` is a recognisable dog). Phase 3's window
   depends on discretisation error as implicit regularisation. Keep `n_steps=10` everywhere.
3. **Breakdown λ varies ~3× across seeds** (seed 3141 dies by λ≈1.0; seed 1234 survives to ≈2.4). The
   6-point lattice above is deliberately low for this reason — some seeds will already be past breakdown
   by λ=1.571-2.357; report images, don't assume survival.
4. **Right-size `--time` per job.** A blanket `--time=720` left Slurm's backfill unable to schedule and
   pushed starts to the next day. `killable` allows 24 h but don't use it.
5. **FLUX's backward is not bit-reproducible on GPU.** `f` carries a ~2.5% nondeterminism floor, and the
   within-seed floor is 3.7% at λ=3.93. Differences below ~4% are noise. `preflight` records this per job;
   a λ=0 bitwise check still runs and must pass.
6. **Verify results against the job's log** before drawing conclusions. One monitor notification last
   session was fabricated — it reported a point absent from the log, and would have reversed the finding.
7. Setup is NFS-bound (32 GB of mmap'd safetensors at 5–12 MB/s), so 20–90 min of silence before the first
   λ point is normal, not a hang. Check `read_bytes` in `/proc/<pid>/io` via `srun --overlap` if unsure.
8. **The resume path in `load_results` requires an exact stamp match (including `lam_k` and the live
   `preflight` measurements) or it silently renames the existing file to `.stale` and starts fresh** —
   this is why every group above requests the full 6-point lattice instead of trying to patch in just the
   missing points. If you're tempted to save compute by requesting a partial `--lam-k` against an existing
   file, don't — verify the stamp-equality logic in `pc_sweep.py` first, it will not merge.
9. **The "existing data" table above is a snapshot read on 2026-10-03** — if results files have changed
   since, re-check `stamp.lam_k` in each before trusting which points are actually missing.

## What a result looks like

Report, per seed, at each λ: `f` for C=1 vs C=2 vs C=4, **and the images side by side**. The claim to
test is whether higher C raises `f` further *without* costing recognisability. If `f` rises but images
degrade, that is the same trade `eta_reference="score"` already lost — say so plainly rather than
reporting the `f` gain alone.
