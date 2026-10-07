#!/bin/sh
#SBATCH --job-name=flux-phase5-pc-sweep
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gpus=1
#
# Phase 5 (predictor-corrector Langevin, creativity_measure/samplers/flow_guided_pc.py). ONE CONFIG PER
# JOB: first argument is the arm, everything after it is forwarded verbatim to pc_sweep.py, so a wave is
# just a list of sbatch lines. Results and decoded dirs are keyed on the full config inside pc_sweep.py,
# so concurrent jobs never collide.
#
#   export HF_TOKEN=$(cat $HOME/.hf_token)
#   sbatch --job-name=p5-j1 pc_sweep.slurm pc_unguided --corrector-steps 1
#   sbatch --job-name=p5-j2 pc_sweep.slurm pc_guided   --corrector-steps 1
#   sbatch --job-name=p5-j3 pc_sweep.slurm flow_guided --n-steps 19          # compute-matched control
#   sbatch --job-name=p5-j5 pc_sweep.slurm pc_guided --corrector-steps 1 --eta-reference score
#
# Override --job-name per job (as above) so the logs are identifiable; the header's name is only a default.
#
# WHY THE THIRD ARM IS RE-RUN HERE instead of read out of Phase 3's stored
# fine_lambda_sweep_results_max5.5.json: CLAUDE.md records that a GPU model change alone shifts f by 16%
# of std_p(f) (job 697271) -- the same order as the effect being measured. All three arms must therefore
# share one GPU model, one reward and one reference bank. Each job also checks on the actual hardware,
# before any sweep point, that corrector_steps=0 reduces to flow_guided_sample -- bitwise at lam=0, where
# no backward is taken, and against the backend's own self-reproducibility floor at lam!=0, because FLUX's
# backward is NOT bit-reproducible on GPU (that cost jobs 965868/965869 before the check was corrected;
# the backend-independent claim is asserted in CI instead, by test_flow_guided_pc.py's call-sequence
# test). A failure raises and no number from the job should be interpreted.
#
# --constraint=l40s, pinned to ONE model (never a6000|l40s) for the reason above. Phase 3's
# fine_lambda_sweep_results_max5.5.json stamp records 'NVIDIA L40S', so this also keeps the arms
# comparable to that stored run's images.
#
# --mem=24000, NOT the --mem=64000 Phase 3's scripts carried unmeasured: Phase 3 job 957386 measured host
# peak RSS at 10.2 GB on this exact path (base FLUX.1-dev, no LoRA, same i.i.d. squared-IEM reward), and
# memory -- not GPUs -- is what blocks these jobs in the queue (CLAUDE.md: 2026-08-26, ~2 h lost to a node
# with 2 free A6000s but 11.8 GB free RAM). note() stamps ru_maxrss on every progress line, so a real
# overshoot will be visible in this job's own log; never set --mem equal to an observed peak.
#
# --time=720, NOT Phase 3's 360. `killable` allows 1-00:00:00 (24 h); 360 was a self-imposed choice
# there, and it is the wrong one here because SETUP has been measured at 35-90 min on this path -- the
# 32 GB of safetensors are mmap'd off $WORK and job 966117 was observed page-faulting them in at ~5 MB/s
# (process in state Dl on folio_wait_bit_common, read_bytes climbing 50 MB per 10 s). With a setup that
# expensive, resumption is expensive: every requeue re-pays it. Buying wall clock is strictly cheaper
# than splitting the lambda grid across more jobs, which would re-pay setup per split.
#
# Per-arm sweep estimates, scaled from Phase 3's MEASURED 571 s per guided lambda
# (exact_jacobian=True, N_PARTICLES=1, N_STEPS_ODE=10, L40S) by the per-ODE-step cost ratios the --dry-run
# measured (1.0 / 1.8 / 2.8 for flow_guided / pc_unguided / pc_guided at M=2):
#   pc_guided    ~1600 s/lambda x 8 = 3.6 h   (+ ~10 min ref bank + model load + 2 min reduction check)
#   pc_unguided  ~1030 s/lambda x 8 = 2.3 h
#   flow_guided    571 s/lambda x 8 = 1.3 h
# Every arm is resumable per lambda (atomic results JSON + PNG written the moment each point completes),
# so a timeout or preemption costs only the in-flight point -- just resubmit the same command.
#
# Same TRITON_CACHE_DIR / HF_TOKEN lessons as every other Phase 2/3 slurm script.

set -e

ARM="$1"
case "$ARM" in
    -*|"")
        echo "FATAL: first argument must be the arm (or a comma-separated list of arms); the rest are" >&2
        echo "forwarded to pc_sweep.py." >&2
        echo "  sbatch pc_sweep.slurm {pc_guided|pc_unguided|flow_guided}[,...] [pc_sweep.py flags...]" >&2
        echo "  (got: '${ARM}')" >&2
        exit 1
        ;;
    *) shift ;;
esac
# Not validated here any more: with comma-separated lists the shell case would have to enumerate every
# subset. pc_sweep.py's own --arm parser rejects an unknown name before any model load, which is the
# same failure one second later.
EXTRA="$@"
ARMTAG=$(echo "$ARM" | tr ',' '+')

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
SWEEP_DIR=$REPO/notebooks/flux_guided_phase5

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$WORK/.cache/triton"

mkdir -p "$WORK/logs" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
cd "$SWEEP_DIR"

echo "=== $(date) on $(hostname) -- arm=$ARM extra='$EXTRA' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" log -1 --format=%s))"
git -C "$REPO" status --short | head -5

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set in the environment. Submit with:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch pc_sweep.slurm $ARM" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

echo "=== running pc_sweep.py --arm $ARM $EXTRA ==="
"$PY" -u pc_sweep.py --arm "$ARM" "$@"

echo "=== done $(date) ==="
ls -lt "$SWEEP_DIR"/pc_sweep_results_*.json 2>/dev/null | head -5
