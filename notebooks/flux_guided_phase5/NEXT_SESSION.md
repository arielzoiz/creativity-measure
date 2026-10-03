# Task: sweep `corrector_steps` ∈ {2, 4} for Phase 5, across the 5 existing seeds

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

## What to run

10 jobs: C ∈ {2, 4} × seeds {1234, 2024, 3141, 4242, 5555}, at λ-lattice indices 1,2,3,4
(λ = 0.393 / 0.786 / 1.179 / 1.571 / 1.964 / 2.357).

```bash
cd /home/dcor/arielzoizner/projects/creativity-measure/notebooks/flux_guided_phase5
export HF_TOKEN=$(cat $HOME/.hf_token)          # required; never echo or write it to $WORK

for C in 2 4; do
  for S in 1234 2024 3141 4242 5555; do
    # --time: ~90 min worst-case setup + 4 lambda x (10 + 9C) guided units x 57 s
    T=$([ "$C" = 2 ] && echo 300 || echo 420)
    sbatch --time=$T --job-name="p5-c${C}-s${S}" \
      pc_sweep.slurm pc_guided --corrector-steps $C --lam-k 1,2,3,4 --sweep-seed $S
    sleep 60                                     # stagger: spreads jobs across nodes
  done
done
```

Use the `slurm-jobs` skill and arm its watchdog. ~23 GPU-h total; C=2 alone is ~9 GPU-h, so if you want
a cheaper first cut, run C=2 across all 5 seeds and only launch C=4 if C=2 shows a gain over C=1.

## How to analyse

```bash
python render_headline.py          # PC vs Phase 3, paired per seed  -> pc_headline_grid.png
python render_pc_comparison.py     # all arms browser                -> pc_comparison_grid.png
```

`render_pc_comparison.py` discovers rows by globbing results files, so the new C=2/C=4 runs appear
automatically. `render_headline.py` currently filters to `corrector_steps == 1` (see `_pc_rows`) — widen
that filter, or add a C column, to put C=1/2/4 side by side at matched seed and λ.

Compare against, at the same seeds and λ:
- **C=1**: `pc_sweep_results_pc_guided_n10_c1_etatotal_snr0.16_s<SEED>.json` (+ `..._s1234.json`)
- **Phase 3 control**: `../flux_guided_phase3/fine_lambda_sweep_results_seed<SEED>.json` and
  `fine_lambda_sweep_results_max5.5.json` (seed 1234), with images in `fine_decoded_seed*/`.
  These are `n_steps=10` on L40S over the same λ lattice — already matched, do not re-run them.

## Gotchas that cost real GPU-hours last time

1. **LOOK AT THE DECODED PNGs.** No scalar predicts recognisability — not `f`, not `‖x‖/√d`, not `hf`.
   `f` = 14.2 on a clear dog and 14.8 on pure noise; `‖x‖` is *anti*-correlated with quality; `hf` scores
   0.0089 on a dog and 0.0093 on a destroyed mosaic. Images are written per point, free to inspect.
   Four hours went into scalar analyses pointing the wrong way before anyone opened an image.
2. **Never raise `n_steps` to build a "compute-matched" control.** More ODE steps *destroys* guidance
   (`n_steps=19` is abstract blocks at λ=1.571 where `n_steps=10` is a recognisable dog). Phase 3's window
   depends on discretisation error as implicit regularisation. Keep `n_steps=10` everywhere.
3. **Breakdown λ varies ~3× across seeds** (seed 3141 dies by λ≈1.0; seed 1234 survives to ≈2.4). A grid
   chosen from one seed measures nothing on another — that is why the λ range here is deliberately low.
4. **Right-size `--time` per job.** A blanket `--time=720` left Slurm's backfill unable to schedule and
   pushed starts to the next day. `killable` allows 24 h but don't use it.
5. **FLUX's backward is not bit-reproducible on GPU.** `f` carries a ~2.5% nondeterminism floor, and the
   within-seed floor is 3.7% at λ=3.93. Differences below ~4% are noise. `preflight` records this per job;
   a λ=0 bitwise check still runs and must pass.
6. **Verify results against the job's log** before drawing conclusions. One monitor notification last
   session was fabricated — it reported a point absent from the log, and would have reversed the finding.
7. Setup is NFS-bound (32 GB of mmap'd safetensors at 5–12 MB/s), so 20–90 min of silence before the first
   λ point is normal, not a hang. Check `read_bytes` in `/proc/<pid>/io` via `srun --overlap` if unsure.

## What a result looks like

Report, per seed, at each λ: `f` for C=1 vs C=2 vs C=4, **and the images side by side**. The claim to
test is whether higher C raises `f` further *without* costing recognisability. If `f` rises but images
degrade, that is the same trade `eta_reference="score"` already lost — say so plainly rather than
reporting the `f` gain alone.
