#!/bin/sh
#SBATCH --job-name=dmsmc
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000
#SBATCH --time=180
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Diamond Maps SMC (Algorithm 2) with the normalized squared global IEM tilt.
#
# COST. Per lambda this is ~10.8k SiT-XL/2 single-image passes (~1.3 PFLOP) => tens of seconds of
# arithmetic. Wall time is dominated by startup: unpickling 10.7 GB of checkpoints and XLA-compiling
# the GLASS step and the posterior step. Each extra lambda in the same process reuses both, so the
# sweep is nearly free after the first. 180 min is generous; it is not a compute-bound job.
#
# Knobs (LABEL, CFG_SCALE, LAM_MULTS, M, K, N, R, ...) live at the top of the notebook, NOT here --
# `p` is defined at the call site. Override per submission without editing the notebook:
#     sbatch --export=ALL,DMSMC_LABEL=151,DMSMC_LAM_MULTS=0,5,10 diamond_smc.slurm
#
# `killable` is preemptible with a 1-day cap. This job is short enough that preemption is cheap;
# if a run must not be interrupted, --partition=gpu-b200 is not preemptible (premium shared
# hardware -- do not default to it).
#
# No `source ~/.bashrc` -- that file is empty here; conda and the HF_* redirects live in ~/.cshrc
# (tcsh), so this script sets everything it needs explicitly and works from a fresh shell.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/diamond-creative/bin/python
RUN_DIR=$REPO/notebooks/diamond_smc_imagenet
NB=diamond_smc_imagenet.ipynb

export DIAMOND_MAPS_ROOT="$WORK/repos/diamond_maps"
export HF_HOME="$WORK/.cache/huggingface"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota

# jax and torch share one GPU: without this jax preallocates ~75% of VRAM at first use and torch
# then OOMs on the reward.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export PYTHONUNBUFFERED=1                            # else progress is invisible until the job ends
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$RUN_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "knobs: LABEL=${DMSMC_LABEL:-<notebook default>} CFG_SCALE=${DMSMC_CFG_SCALE:-<nb>} \
LAM_MULTS=${DMSMC_LAM_MULTS:-<nb>} M=${DMSMC_M:-<nb>} K=${DMSMC_K:-<nb>} N=${DMSMC_N:-<nb>}"

# Preflight: fail in seconds, not after minutes of unpickling. Catches the failure modes that have
# actually cost runs here -- no CUDA, a card too small or without native bf16, missing checkpoints.
"$PY" - <<'PREFLIGHT'
import sys, os
sys.path.insert(0, os.getcwd())
import paths
for k, v in paths.preflight().items():
    print(f"  {k}: {v}")
PREFLIGHT

# Separate output notebook, never --inplace: nbconvert writes once at the very end, so a kill during
# that write could truncate the source. --allow-errors so a mid-run failure still preserves the
# diagnostics and images already produced, with the traceback kept in the failing cell.
OUT="run_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT ==="
"$PY" -m jupyter nbconvert \
    --to notebook \
    --execute \
    --allow-errors \
    --output "$OUT" \
    --ExecutePreprocessor.timeout=-1 \
    "$NB"

echo "=== done $(date) ==="
echo "notebook with outputs: $RUN_DIR/$OUT"
ls -l "$RUN_DIR"/*.pt "$RUN_DIR"/*.png 2>/dev/null || true
