#!/bin/sh
#SBATCH --job-name=flux-auto-r
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000
#SBATCH --time=600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Auto-R diagnostic: how many IEM references does f need before its RANKING of samples stabilises?
# Runs the repo's weighted-tau rule for RandomRefs and WeightedFPSRefs at the sweeps' N_GAMMA/NUM_EPS.
#
# Projected ~8.5 h from the measured 174 ms/pass (RandomRefs grid<=32 ~6.2 h, WeightedFPS grid<=8 ~2.4 h); --time=600
# leaves margin. Every auto-R control is scaled far below library defaults, which would cost 120 h and
# 558 h respectively. The notebook saves auto_r_results.json after EACH selector, so a timeout still
# leaves the finished one on disk.
#
# --inplace is deliberate: this notebook IS the deliverable, it is a cheap diagnostic, and it can be
# rerun at will. (The long sampling jobs use a separate output file instead, where a truncated write
# would cost real compute.)

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
DIAG_DIR=$REPO/notebooks/refset_auto_r
NB=refset_auto_r_diagnostic.ipynb

# $WORK is group-readable, so the HF *credential* must not live there; the 32 G of weights may.
export HF_HOME="$WORK/.cache/huggingface"
export HF_TOKEN_PATH="/a/home/cc/students/math/arielzoizner/.hf_token"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$DIAG_DIR"          # the notebook derives REPO from cwd ("../..") and writes results here

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Preflight: fail in seconds, not after minutes of model loading.
"$PY" - <<'PREFLIGHT'
import torch
from huggingface_hub import auth_check
from scipy.stats import weightedtau          # auto-R needs this; fail now, not 40 min in
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)}  {gb:.0f} GB total")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
auth_check("black-forest-labs/FLUX.1-dev")
print("FLUX.1-dev access: OK   scipy.weightedtau: OK")
PREFLIGHT

echo "=== executing $NB (in place) ==="
"$PY" -m jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --allow-errors \
    --ExecutePreprocessor.timeout=-1 \
    "$NB"

echo "=== done $(date) ==="
ls -l "$DIAG_DIR/auto_r_results.json" "$DIAG_DIR/auto_r_tau.png" 2>/dev/null || true
