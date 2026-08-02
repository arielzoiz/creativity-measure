#!/bin/sh
#SBATCH --job-name=dmsmc-ab
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000
#SBATCH --time=45
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Plumbing validation for the Diamond Maps bridge. Runs the validation ladder at throwaway settings
# (M=4, K=2, N=4, R=8) so a wiring bug surfaces in minutes instead of inside a real experiment.
# Run this BEFORE the notebook, and again after touching anything in creativity_measure/backends/.
#
# 64 GB: the two pickles are 8.1 GB + 2.6 GB and are unpickled to host memory before being placed on
# the device, so peak host usage is well above the file sizes. Under-requesting --mem is what drains
# nodes, so this is padded deliberately.
#
# No `source ~/.bashrc` -- that file is empty here; conda and the HF_* redirects live in ~/.cshrc
# (tcsh), so this script sets everything it needs explicitly and works from a fresh shell.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/diamond-creative/bin/python
RUN_DIR=$REPO/notebooks/diamond_smc_imagenet

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

"$PY" ab_base_sampler.py

echo "=== done $(date) ==="
