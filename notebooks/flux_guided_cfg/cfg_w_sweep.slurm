#!/bin/sh
#SBATCH --job-name=flux-cfg-w-sweep
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gpus=1
#
# Manual CFG x lambda (notebooks/flux_guided_cfg/cfg_w_sweep.py). Every argument is forwarded verbatim
# to the script, so a wave is just a list of sbatch lines:
#
#   export HF_TOKEN=$(cat $HOME/.hf_token)
#   sbatch --job-name=cfg-w1.5-s1234 cfg_w_sweep.slurm --w 1.5 --prompt car --sweep-seed 1234 \
#          --lam-step 0.2 --lam-k 0,1,2,3,4,5
#   sbatch --job-name=cfg-all-s1234  cfg_w_sweep.slurm --w 1.5,2.0,3.0 --prompt car --sweep-seed 1234 ...
#
# Results and decoded dirs are keyed on the full config INCLUDING w inside cfg_w_sweep.py, so concurrent
# jobs never collide. Every point is resumable: the JSON and the PNG hit disk the moment the point
# completes, so a timeout or preemption costs only the in-flight lambda -- resubmit the same command.
#
# NO w=1 JOB IS NEEDED. w=1 is bitwise the Phase 3 baseline (cfg_velocity_fn returns the conditional
# function itself and guided_euler_step detects the identity), and the stored flow_guided runs at the
# same prompt/seed/n_steps/lambda lattice ALREADY ARE IT:
#   "A dog"                                -> ../flux_guided_phase3/fine_lambda_sweep_results*.json
#   building/sofa/car/teapot/jacket        -> ../flux_guided_phase5/pc_sweep_results_*flow_guided*.json
# render_cfg_grid.py reads those as the grid's first column. Match their lambda lattice (--lam-step) or
# there is no baseline to pair against.
#
# --constraint=l40s, pinned to ONE model and never 'a6000|l40s'. A GPU change alone shifts f by 16% of
# std_p(f) (job 697271) -- the same order as the effect being measured -- and every stored w=1 run's
# stamp records 'NVIDIA L40S'. cfg_w_sweep.py's preflight item (1) RAISES on anything else before the
# reference-bank build, so a mis-scheduled job fails in ~2 min instead of producing uncomparable numbers.
#
# --mem=24000, NOT the --mem=64000 Phase 3's own scripts carried unmeasured: Phase 3 job 957386 measured
# host peak RSS at 10.2 GB on this exact path (base FLUX.1-dev, no LoRA, same i.i.d. squared-IEM reward),
# and memory -- not GPUs -- is what blocks these jobs in the queue (CLAUDE.md: 2026-08-26, ~2 h lost to a
# node with 2 free A6000s but 11.8 GB free RAM). Encoding a SECOND prompt does not change this: both
# prompts are encoded before the ~9.5 GB of text encoders are freed, so the peak is the same one already
# measured. note() stamps ru_maxrss on every progress line, so a real overshoot shows in this job's log.
#
# VRAM is NOT the binding constraint and the sequential double-forward is not what keeps us inside it.
# An L40S has 48 GB (44.53 GiB measured with bf16 FLUX resident, generators/flux.py's _freeze docstring);
# --mem above is HOST ram. What sequential evaluation buys is not fitting under --mem but avoiding a
# second concurrent set of transformer activations on the GPU, which a batch-concatenated CFG forward
# would create for zero throughput gain at batch 1.
#
# --time=300: setup (35-90 min, NFS-bound -- 32 GB of mmap'd safetensors off $WORK at 5-12 MB/s; 20-90 min
# of silence before the first lambda point is NORMAL, not a hang) + preflight (~5 min) + N_lambda * per-
# point cost, x1.5. Per-point cost is Phase 3's MEASURED 571 s/lambda (exact_jacobian=True,
# N_PARTICLES=1, n_steps=10, L40S) plus one extra batch-1 no_grad forward per step. That overhead should
# be small -- the reward dominates each step with ~50 denoiser rows at batch <=24, against one batch-1
# transformer call -- but it is NOT yet measured on this path, so the budget below assumes a pessimistic
# 1.25x (~714 s/lambda) and the FIRST job's log should be read for the real number:
#   6 lambda, 1 w : 90 + 5 + 71  min =  166 min -> --time=300 (the header default)
#   6 lambda, 3 w : 90 + 5 + 214 min =  309 min -> sbatch --time=480
# RIGHT-SIZE IT per submission. `killable` allows 24 h, but a blanket --time=720 left Slurm's backfill
# unable to schedule and pushed starts to the next day (Phase 5 NEXT_SESSION.md, gotcha 4).
#
# ONE JOB PER NODE WHERE POSSIBLE. Five concurrent jobs on n-801 produced ZERO points in 3 h while a lone
# job on n-803 -- also cold -- finished setup in 40 min: all five were page-faulting the same 32 GB
# checkpoint off $WORK, degrading to 1.8 MB/s each as they contended (PROMPT_STUDY.md, "The n-801 stall").
# It is contention, not cold cache. Stagger submissions and prefer several w in ONE job (--w 1.5,2.0,3.0),
# which shares the setup rather than re-paying it.
#
# Same TRITON_CACHE_DIR / HF_TOKEN lessons as every other Phase 2/3/5 slurm script.

set -e

case "$1" in
    ""|-h|--help)
        echo "FATAL: pass cfg_w_sweep.py's flags; --w is required." >&2
        echo "  sbatch cfg_w_sweep.slurm --w 1.5[,2.0,3.0] [--prompt P] [--sweep-seed S] \\" >&2
        echo "         [--lam-step X --lam-k K,...] [--cfg-t-window t_hi,t_lo]" >&2
        exit 1
        ;;
esac

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
SWEEP_DIR=$REPO/notebooks/flux_guided_cfg

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$WORK/.cache/triton"

mkdir -p "$WORK/logs" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
cd "$SWEEP_DIR"

echo "=== $(date) on $(hostname) -- args: $* ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" log -1 --format=%s))"
git -C "$REPO" status --short | head -5

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set in the environment. Submit with:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch cfg_w_sweep.slurm $*" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

echo "=== running cfg_w_sweep.py $* ==="
"$PY" -u cfg_w_sweep.py "$@"

echo "=== done $(date) ==="
ls -lt "$SWEEP_DIR"/cfg_sweep_results_*.json 2>/dev/null | head -5
