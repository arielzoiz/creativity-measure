#!/bin/sh
#SBATCH --job-name=flux-guided-notebook
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=330
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Executes flux_guided_lambda_sweep.ipynb in place (nbconvert): the Phase 3 vs Algorithm 1 comparison
# deliverable.
#
# Job 957442 (N_PARTICLES=4, N_STEPS_ODE=16, M_GRID 4-wide, --time=180) TIMED OUT after 3h without
# finishing: job 957386's small-scale measurement (~42.5s/step) did not extrapolate -- at full scale
# every ODE step hit the gamma-chunk OOM fallback, costing ~150s/step (confirmed via repeated
# srun --overlap checks: GPU pinned ~100% util, allocator OOM-warn/retry cycling every ~2.5min, no
# traceback -- genuinely slow, not stuck). Trimmed in the notebook (cell 4): N_PARTICLES 4->2,
# N_STEPS_ODE 16->10, M_GRID to 3 values -- 6 configs x 10 steps x batch=2. --time raised to 330 for
# margin even if per-step cost does not fully recover from trimming the batch axis.
#
# Same TRITON_CACHE_DIR / HF_TOKEN lessons as the other two Phase 2/3 slurm scripts tonight.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
NB_DIR=$REPO/notebooks/flux_guided_phase3
NB=flux_guided_lambda_sweep.ipynb

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$WORK/.cache/triton"
export PYTHONUNBUFFERED=1
export JUPYTER_RUNTIME_DIR=/tmp/$USER/jupyter-runtime-${SLURM_JOB_ID:-manual}

mkdir -p "$WORK/logs" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$JUPYTER_RUNTIME_DIR"
cd "$NB_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" log -1 --format=%s))"
git -C "$REPO" status --short | head -8

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set in the environment. Submit with:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch run_notebook.slurm" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

echo "=== executing $NB ==="
"$PY" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=9000 \
    "$NB"

echo "=== done $(date) ==="
ls -l "$NB_DIR"/*.png "$NB_DIR"/decoded/*.png 2>/dev/null | tail -20
