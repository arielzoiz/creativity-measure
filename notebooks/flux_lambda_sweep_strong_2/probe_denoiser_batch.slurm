#!/bin/sh
#SBATCH --job-name=flux-probe
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000
#SBATCH --time=45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Measure the real denoiser batch-width ceiling before committing 8 h to run 2.
#
# Run 2 needs NUM_EPS*R = 3*64 = 192 rows for the reference-bank build. Job 694278 (which worked)
# never exceeded 24; job 694256 is recorded as OOMing at 192. This settles which is right by
# measurement instead of inference. Phase B reproduces the 1.36 GB resident bank the sweep holds
# while issuing those forwards, so the answer applies to the real run and not to an empty card.
#
# ONE model load, two phases (empty card, then holding the 1.36 GB bank). Job 695141 reloaded the
# model per pass and its 480 s alarm fired during .to(DEVICE) -- which takes ~7 min here -- so it
# never probed anything. --time=45 covers a slow load; the in-process watchdog now guards only the
# probe loops.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
DIR=$REPO/notebooks/flux_lambda_sweep_strong_2

export HF_HOME="$WORK/.cache/huggingface"
export HF_TOKEN_PATH="/a/home/cc/students/math/arielzoizner/.hf_token"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

echo
"$PY" probe_denoiser_batch.py --timeout 900 --out probe_denoiser_batch.json

echo
echo "=== done $(date) ==="
