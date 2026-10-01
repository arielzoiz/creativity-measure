#!/bin/sh
#SBATCH --job-name=flux-fine-lambda-zoom
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Zoomed-in follow-up to fine_lambda_sweep.py's first run (job 958150, COMPLETED): that sweep, uniform
# in [0, lambda_s=81.02], found the image already almost entirely collapsed to off-manifold noise by
# lam=5.79 (m=0.071) -- the actual threshold lies somewhere in [0, 5.79], finer than that grid resolved.
# This run narrows to --lam-max 5.5 (15 points uniform in [0, 5.5]) to locate it.
#
# Same resumable-script design as the first run (fine_lambda_sweep_results_max5.5.json,
# fine_decoded_max5.5/*.png, written to disk as each lambda completes) -- a timeout/preemption only
# costs the in-flight config, resubmit continues from the JSON.
#
# --time=300: this run pays the SAME one-time setup cost as job 958150 (model load + ref-bank build +
# probe, measured there at ~57min -- longer than the notebook's partial 557s timing, see that job's
# header) plus 14 non-control configs x ~568s/config (measured in 958150 at N_PARTICLES=1,
# exact_jacobian=True) =~ 7950s =~ 2.2h. Total ~3.2h, same order as job 958150's actual 3h12m -- 300min
# budgets with margin.
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
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch fine_lambda_sweep_zoom.slurm" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

echo "=== running fine_lambda_sweep.py --lam-max 5.5 ==="
"$PY" -u fine_lambda_sweep.py --lam-max 5.5

echo "=== done $(date) ==="
ls -l "$SWEEP_DIR"/fine_lambda_sweep_results_max5.5.json "$SWEEP_DIR"/fine_decoded_max5.5/*.png 2>/dev/null | tail -20
