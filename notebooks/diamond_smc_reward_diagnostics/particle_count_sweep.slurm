#!/bin/sh
#SBATCH --job-name=dmsmc-countsweep
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
# Particle-count sweep at a fixed, already-degenerate lambda -- see particle_count_sweep.py's docstring. One
# backend build, then 4 SMC runs reusing it (M=16,32,64,128 at K=4), each batch up to M*K=512
# lookahead candidates through SiT-XL/2 -- comfortably under the >=40GB preflight floor but the
# largest single batch run in this project directory so far, hence the same generous --mem as the
# other diamond-maps jobs.
#
# No `source ~/.bashrc` -- that file is empty here; conda and the HF_* redirects live in ~/.cshrc
# (tcsh), so this script sets everything it needs explicitly and works from a fresh shell.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/diamond-creative/bin/python
RUN_DIR=$REPO/notebooks/diamond_smc_reward_diagnostics

export DIAMOND_MAPS_ROOT="$WORK/repos/diamond_maps"
export HF_HOME="$WORK/.cache/huggingface"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$RUN_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

"$PY" particle_count_sweep.py

echo "=== done $(date) ==="
