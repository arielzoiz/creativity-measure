#!/bin/sh
#SBATCH --job-name=flux-phase5-pc-sweep
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gpus=1
#
# Phase 5 (predictor-corrector Langevin, creativity_measure/samplers/flow_guided_pc.py). ONE ARM PER JOB:
#
#   export HF_TOKEN=$(cat $HOME/.hf_token)
#   sbatch pc_sweep.slurm pc_guided        # predictor_guided=True,  M=2  -- superset of Phase 3
#   sbatch pc_sweep.slurm pc_unguided      # predictor_guided=False, M=2  -- all tilt from the corrector
#   sbatch pc_sweep.slurm flow_guided      # corrector_steps=0            -- Phase 3, bitwise
#
# WHY THE THIRD ARM IS RE-RUN HERE instead of read out of Phase 3's stored
# fine_lambda_sweep_results_max5.5.json: CLAUDE.md records that a GPU model change alone shifts f by 16%
# of std_p(f) (job 697271) -- the same order as the effect being measured. All three arms must therefore
# share one GPU model, one reward and one reference bank. Each job also asserts on the actual hardware,
# before any sweep point, that corrector_steps=0 is BITWISE equal to flow_guided_sample; if that fails the
# job raises and no number from it should be interpreted.
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
# --time=360: per-arm estimates, scaled from Phase 3's MEASURED 571 s per guided lambda
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
    pc_guided|pc_unguided|flow_guided) ;;
    *)
        echo "FATAL: pass exactly one arm as the first argument." >&2
        echo "  sbatch pc_sweep.slurm {pc_guided|pc_unguided|flow_guided}" >&2
        echo "  (got: '${ARM}')" >&2
        exit 1
        ;;
esac

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

echo "=== $(date) on $(hostname) -- arm=$ARM ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" log -1 --format=%s))"
git -C "$REPO" status --short | head -5

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set in the environment. Submit with:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch pc_sweep.slurm $ARM" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

echo "=== running pc_sweep.py --arm $ARM ==="
"$PY" -u pc_sweep.py --arm "$ARM"

echo "=== done $(date) ==="
ls -l "$SWEEP_DIR"/pc_sweep_results_"$ARM".json "$SWEEP_DIR"/pc_decoded_"$ARM"/*.png 2>/dev/null | tail -20
