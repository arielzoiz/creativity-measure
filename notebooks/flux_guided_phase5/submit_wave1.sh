#!/bin/sh
# Phase 5, wave 1 -- one seed (1234), nine jobs. Records the experiment rather than leaving it in shell
# history.
#
#   export HF_TOKEN=$(cat $HOME/.hf_token) && sh submit_wave1.sh
#   STAGGER=0 sh submit_wave1.sh            # submit all at once (not recommended, see below)
#
# Lambda grid is every other point of Phase 3's zoom lattice, k = 0,2,...,20 -> lam = 0 .. 7.857.
# Points at k <= 14 pair one-to-one with an existing fine_decoded_max5.5 image; k = 16,18,20 extend PAST
# anything Phase 3 ran, which is the point -- the max5.5 grid was chosen to bracket PHASE 3's collapse,
# so if the corrector pushes the ceiling out, 5.5 is too short to find it.
#
# WHY corrector_steps=1 is the default here: it is Song et al.'s reference PC setting at snr=0.16, and it
# makes pc_unguided cost 524 s/lambda against Phase 3's measured 572 s/lambda -- so the EXISTING max5.5
# run is already a compute-matched control for that arm, for free.
#
# WHY J3 EXISTS, and why it is the job to protect if anything gets cut: pc_guided at n_steps=10,
# corrector_steps=1 spends 10 + 9 = 19 guided units (the corrector is skipped at the schedule's final t=0
# node, so 9 corrector nodes, not 10). flow_guided at --n-steps 19 costs exactly the same 19 units. It is
# therefore the ONLY run that separates "the corrector works" from "more reward-gradient evaluations
# work" -- and if it also extends the window, Phase 3's ceiling was ODE discretization error rather than
# an off-manifold attractor, and the premise of this whole phase needs rewriting.
#
# WHY J5 EXISTS: under eta_reference="total" the corrector's measured displacement falls 12x from
# lam=0.79 to lam=5.5, so at high lam -- exactly where the claim lives -- it has nearly annealed itself
# off, and "PC ~ Phase 3 up there" would be an artifact of the step-size rule rather than a result.
# eta_reference="score" makes eta lam-independent so lam only ROTATES the drift; a validation run
# measured 7.3x more corrector displacement at lam=7.86 (0.097 vs 0.013).
#
# J7a/b/c share one z0 (Phase 3's, seed 1234) and vary ONLY the Langevin noise, which measures PC's
# within-seed variance -- the component Phase 3 structurally does not have, and what sizes wave 2.
#
# WHY SUBMISSIONS ARE STAGGERED: wave 1's first attempt put 5 jobs on n-801, and those 5 were still
# loading when the two that landed alone on other nodes had already finished setup. Co-location HURT --
# they contend for the node's NFS client bandwidth rather than usefully sharing page cache. The model is
# 32 GB of mmap'd safetensors on $WORK and job 966117 was measured page-faulting it in at ~5 MB/s, so
# read bandwidth, not GPU, is the setup bottleneck. Spacing submissions lets Slurm place them as nodes
# free up instead of packing one.
#
# Results and decoded dirs are keyed on the full config inside pc_sweep.py, so these never collide and
# each is independently resumable per lambda.

set -e
cd "$(dirname "$0")"

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: export HF_TOKEN=\$(cat \$HOME/.hf_token) first" >&2
    exit 1
fi

K11=0,2,4,6,8,10,12,14,16,18,20
: "${STAGGER:=90}"

# RIGHT-SIZE --time PER JOB. The slurm header's --time=720 is only a ceiling for the longest arms; a
# blanket 12 h is actively harmful because Slurm's backfill will not slot long jobs when the queue is
# busy. Measured the hard way: with --time=720 on all nine and 106 pending vs 88 running on killable, the
# estimated starts were 02:10 / 02:52 / 03:17 for the first three but 14:06, 14:52, 15:17, 16:53 and
# NEXT-DAY 12:07 for the rest. Re-submitting the same jobs with --time sized to their actual work fixed
# it. Budget = ~90 min worst-case setup (NFS-bound, see pc_sweep.slurm) + the sweep estimate, rounded up.
submit() {                      # submit <minutes> <job-name> <arm> [flags...]
    mins="$1"; name="$2"; shift 2
    sbatch --time="$mins" --job-name="$name" pc_sweep.slurm "$@"
    sleep "$STAGGER"
}

#     min  job name                arm          flags                                             sweep est
submit 360 p5-j1-unguided-c1  pc_unguided --corrector-steps 1 --lam-k $K11                       # 1.6 h
submit 420 p5-j2-guided-c1    pc_guided   --corrector-steps 1 --lam-k $K11                       # 3.3 h
submit 420 p5-j3-ctrl-n19     flow_guided --n-steps 19        --lam-k $K11                       # 3.3 h
submit 180 p5-j4-ctrl-n10-ext flow_guided --n-steps 10        --lam-k 16,18,20                   # 0.5 h
submit 420 p5-j5-guided-score pc_guided   --corrector-steps 1 --eta-reference score --lam-k $K11 # 3.3 h
submit 240 p5-j6-guided-c2    pc_guided   --corrector-steps 2 --lam-k 10,14,18                   # 1.3 h
submit 180 p5-j7a-z0fix-s101  pc_guided   --corrector-steps 1 --z0-seed 1234 --sweep-seed 101 --lam-k 10,14
submit 180 p5-j7b-z0fix-s202  pc_guided   --corrector-steps 1 --z0-seed 1234 --sweep-seed 202 --lam-k 10,14
submit 180 p5-j7c-z0fix-s303  pc_guided   --corrector-steps 1 --z0-seed 1234 --sweep-seed 303 --lam-k 10,14

echo
squeue --me -o "%.9i %.24j %.8T %.8M %R"
