#!/bin/sh
# Phase 5, wave 2 -- does wave 1's positive result hold across z0?
#
#   export HF_TOKEN=$(cat $HOME/.hf_token) && sh submit_wave2.sh
#
# WHAT WAVE 1 ESTABLISHED, and therefore what this has to test. At matched lambda and matched n_steps=10,
# differing only by the corrector:
#     lam=1.571  Phase 3 recognizable dog f=14.2  |  PC recognizable dog   f=23.5
#     lam=2.357  Phase 3 DEGRADED to a pictograph |  PC STILL CLEARLY A DOG f=57.2
#     lam=3.143  both broken
# That single lam=2.357 comparison is carrying the entire positive claim of the phase, on ONE z0. Phase 3's
# own seed-to-seed CV of f is 14-27%, so one draw is not evidence. This wave replicates it.
#
# ONLY THE PC ARM IS RUN. The control already exists: Phase 3's fine_lambda_sweep_results_seed{2024,3141,
# 4242,5555}.json were produced at n_steps=10 on L40S over the SAME lambda lattice (k * 5.5/14), and they
# cover k=4,5,6,7 exactly, with decoded images in fine_decoded_seed*/. Re-running it would burn ~5 GPU-h
# to reproduce numbers already on disk. (They are from a different job, so cross-run nondeterminism of
# ~2.5% in f applies -- immaterial against a 14-27% seed CV, and the comparison here is primarily visual.)
#
# WHY NOT THE ORIGINAL WAVE-2 DESIGN. It was 4 seeds x {pc_guided, pc_unguided} x 6 lambda against a
# COMPUTE-MATCHED flow_guided @19 control, ~13 GPU-h. Wave 1 killed all three premises:
#   - flow_guided @19 is not a control: raising n_steps DESTROYS guidance (abstract blocks at lam=1.571
#     where n_steps=10 gives a dog), so it measures a different broken algorithm.
#   - pc_unguided has no effect at any lambda -- f flat to 1.3% across a 10x lambda range, and a pristine
#     untilted dog at lam=7.86. Replicating a null across seeds is worthless.
#   - lambda > 3 is past where either arm survives, so those points measure nothing.
#
# LAMBDA GRID: k = 4,5,6,7 -> 1.571, 1.964, 2.357, 2.750. Brackets the transition: wave 1 has both arms
# intact at 1.571, PC-only intact at 2.357, and Phase 3 already degraded at 2.750. Four points, so a
# per-seed answer to "where does each arm stop being a dog" rather than a single contested comparison.
#
# READ THE IMAGES, NOT f. Wave 1's f was maximised (608) by the run that produced a formless blob. The
# scalars to trust are hf_frac (lower = more intact; tracked recognizability every time) and NOT
# x_norm_final, which is anti-correlated with quality. See pc_sweep.py's header.
#
# Cost: 4 seeds x 4 lambda x ~1065 s = 4.7 GPU-h, plus 35-90 min NFS-bound setup per job.

set -e
cd "$(dirname "$0")"

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: export HF_TOKEN=\$(cat \$HOME/.hf_token) first" >&2
    exit 1
fi

: "${STAGGER:=90}"
K="4,5,6,7"

for SEED in 2024 3141 4242 5555; do
    sbatch --time=240 --job-name="p5-w2-guided-s$SEED" \
        pc_sweep.slurm pc_guided --corrector-steps 1 --lam-k "$K" --sweep-seed "$SEED"
    sleep "$STAGGER"
done

echo
squeue --me -o "%.9i %.24j %.8T %.6l %.8M %R"
