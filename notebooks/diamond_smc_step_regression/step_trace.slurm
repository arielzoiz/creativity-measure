#!/bin/sh
#SBATCH --job-name=dmsmc-steptrace
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000
#SBATCH --time=120
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Per-transition latent std trace of the base sampler (Algorithm 2 line 6, GLASS), for
# N in {4,6,8,12,16,24} at cfg_scale=4.0, inner=8 fixed -- see step_trace.py's docstring for the
# H_inherent vs H_compounding hypothesis this separates. No SMC, no reward: base_step only.
#
# 64 GB: same rationale as notebooks/diamond_smc_imagenet/*.slurm -- the two pickles (8.1 + 2.6 GB)
# are unpickled to host memory before going to device, so peak host usage is well above file size,
# and this script builds SIX backends in sequence (one per N in the grid).
#
# No `source ~/.bashrc` -- that file is empty here; conda and the HF_* redirects live in ~/.cshrc
# (tcsh), so this script sets everything it needs explicitly and works from a fresh shell.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/diamond-creative/bin/python
RUN_DIR=$REPO/notebooks/diamond_smc_step_regression

export DIAMOND_MAPS_ROOT="$WORK/repos/diamond_maps"
export HF_HOME="$WORK/.cache/huggingface"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota

# jax and torch share one GPU: without this jax preallocates ~75% of VRAM at first use and torch
# then OOMs on the reward. TF is CPU-only here but would also grab memory if it saw the device.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1                            # else progress is invisible until the job ends
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$RUN_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

"$PY" step_trace.py

echo "=== done $(date) ==="
