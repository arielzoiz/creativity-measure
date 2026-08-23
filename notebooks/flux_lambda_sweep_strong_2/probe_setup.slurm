#!/bin/sh
#SBATCH --job-name=flux-probe-setup
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=a6000
#SBATCH --time=75
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Find out where run 2.1's setup stalls. Job 754633 held an A6000 for 6 h and wrote NOTHING -- no
# output notebook (nbconvert writes only at the end), no refs, no checkpoint, and a .out log that
# stops at "=== executing ... ===". Resubmitting the 6 h job unchanged would most likely buy another
# 6 h of silence, so this localizes the stall first.
#
# 75 min: 695271 did model load + refs + reference score bank in ~35 min on this same hardware, so a
# healthy run finishes with better than 2x headroom. If it does NOT finish, the faulthandler stack
# dumps (every 5 min, in the .err) name the frame it is stuck in -- which is the actual deliverable.
#
# Not throwaway on success: the probe writes refs_R64_seed7.pt through the notebook's own cell 4, so
# every later leg loads the references instead of regenerating them through G.
#
# PYTHONUNBUFFERED is the other half of the fix: without it Python block-buffers stdout when it is a
# file rather than a tty, so even flush=True prints can sit unseen. Belt and braces with the explicit
# flush=True calls in the probe.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
SWEEP_DIR=$REPO/notebooks/flux_lambda_sweep_strong_2

export HF_HOME="$WORK/.cache/huggingface"
export HF_TOKEN_PATH="/a/home/cc/students/math/arielzoizner/.hf_token"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$SWEEP_DIR"          # the notebook cells derive REPO from cwd ("../..")

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | head -1
df -h "$HF_HOME" | tail -1

"$PY" probe_setup.py

echo "=== done $(date) ==="
ls -l "$SWEEP_DIR"/refs_*.pt 2>/dev/null || echo "no refs written"
