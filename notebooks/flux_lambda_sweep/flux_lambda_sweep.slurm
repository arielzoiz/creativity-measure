#!/bin/sh
#SBATCH --job-name=flux-lambda-sweep
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000
#SBATCH --time=420
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Executes notebooks/flux_dev_lambda_snapshot_sweep.ipynb in place on one GPU.
#
# Partition choice (verified with `sacctmgr -P -i show user -s arielzoizner`):
#   killable + l40s|a6000  -> 48G card, 1 day cap, PREEMPTIBLE. Plenty of nodes (6 l40s, 2 a6000),
#                             so it usually starts fast. The notebook torch.saves the levels before
#                             decoding, so a preemption costs a rerun, not the expensive result.
#   gpu-b200               -> 5 day cap, NOT preemptible. Use this if a preemption has already cost
#                             a run, or the job must not be interrupted. Premium shared hardware --
#                             don't default to it for an N=8 job (cluster etiquette).
#   gpu-h100-killable      -> faster than l40s, still preemptible, fewer nodes (2) so longer queue.
# To switch: change --partition and drop/adjust --constraint (b200/h100 partitions are already
# hardware-specific, so --constraint must be removed for them).
#
# NOTE: deliberately no `source ~/.bashrc` -- that file is empty here, and conda + HF_* live in
# ~/.cshrc (tcsh). sbatch does inherit the submitting environment, but this script sets everything
# it needs explicitly so it also works from cron, a fresh shell, or a requeue after preemption.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
# Notebook, this script, and the saved latents all live in the same directory.
SWEEP_DIR=$REPO/notebooks/flux_lambda_sweep
NB=flux_dev_lambda_snapshot_sweep.ipynb

# --- caches on $WORK, credential in the private $HOME -------------------------------------------
# $WORK is group-readable, so the token must NOT live there. HF_HOME (weights, ~32G) may.
export HF_HOME="$WORK/.cache/huggingface"
export HF_TOKEN_PATH="/a/home/cc/students/math/arielzoizner/.hf_token"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"   # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
# cd into the notebook's own directory: the notebook derives REPO from cwd ("../.."), and the
# checkpoint it writes lands here too.
cd "$SWEEP_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- preflight: fail in seconds, not after minutes of model loading -----------------------------
# A requeued job that lost its GPU or its gated access should die immediately.
"$PY" - <<'PREFLIGHT'
import torch
from huggingface_hub import auth_check
assert torch.cuda.is_available(), "no CUDA device visible"
free, total = torch.cuda.mem_get_info()
gb = total / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)}  {gb:.0f} GB total, {free / 2**30:.0f} GB free")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
auth_check("black-forest-labs/FLUX.1-dev")
print("FLUX.1-dev access: OK")
PREFLIGHT

# --- execute the notebook -----------------------------------------------------------------------
# nbconvert runs with cwd = the notebook's directory, which is what the notebook's REPO/SWEEP_DIR
# derivation assumes.
#
# Deliberately NOT --inplace: that writes the SOURCE notebook, and nbconvert writes once at the very
# end, so a kill during that write could truncate the only copy. A per-job output file also lets the
# source stay open in an editor while a job runs.
#
# --allow-errors matters just as much: without it, an exception anywhere means nbconvert writes
# NOTHING -- so a mid-run failure would discard the go/no-go images and diagnostics already produced.
# The saved notebook keeps the traceback in the failing cell, which is what you want to read.
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
echo "notebook with outputs: $SWEEP_DIR/$OUT"
ls -l "$SWEEP_DIR"/*.pt 2>/dev/null || true
