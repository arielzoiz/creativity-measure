#!/bin/sh
#SBATCH --job-name=flux-fine-lambda-sweep
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Follow-up to flux_guided_lambda_sweep.ipynb (job 957741, COMPLETED): that run found Phase 3's
# gradient guidance collapses to off-manifold noise by m=1 (lam=81, exact_jacobian=True) -- this job
# scans the GAP, 15 lambdas uniform in [0, lambda_s] (m in [0,1]), exact_jacobian=True only,
# N_PARTICLES=1, to see WHERE along the way the image actually breaks down.
#
# Unlike the notebook (nbconvert --execute --inplace, which buffers all output until the end and lost
# everything when job 957442 timed out), fine_lambda_sweep.py is a standalone RESUMABLE script: each
# lambda's image + numbers hit disk (fine_lambda_sweep_results.json, fine_decoded/*.png) the moment
# they're computed. If this run times out or gets preempted, just resubmit -- it picks up where it
# left off (same stamp-matching convention as guided_sweep.py / iid_vs_brownian.py).
#
# --time=360: 14 non-control lambdas x 10 ODE steps. The completed notebook measured exact_jacobian
# per-step cost of ~104s at N_PARTICLES=2; N_PARTICLES=1 here should be at or below that (batch axis
# drives cost, per CLAUDE.md), so this budgets for the worst case (14*10*104s =~ 4.0h) plus the ~10min
# one-time ref-bank build (557s measured in the notebook) and model load, with margin. If it still
# times out, the resumability means a resubmit loses nothing already computed.
#
# Same TRITON_CACHE_DIR / HF_TOKEN lessons as every other Phase 2/3 slurm script.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
SWEEP_DIR=$REPO/notebooks/flux_guided_phase3

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$WORK/.cache/triton"

mkdir -p "$WORK/logs" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
cd "$SWEEP_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" log -1 --format=%s))"
git -C "$REPO" status --short | head -5

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set in the environment. Submit with:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch fine_lambda_sweep.slurm" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

echo "=== running fine_lambda_sweep.py ==="
"$PY" -u fine_lambda_sweep.py

echo "=== done $(date) ==="
ls -l "$SWEEP_DIR"/fine_lambda_sweep_results.json "$SWEEP_DIR"/fine_decoded/*.png 2>/dev/null | tail -20
