#!/bin/sh
#SBATCH --job-name=flux-render-results
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000|a5000|geforce_rtx_3090
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1
#
# Executes flux_lambda_sweep_results.ipynb: loads the saved levels, prints every diagnostic, and
# decodes each level's latents back to images through FLUX.1-dev's VAE.
#
# Only the ~168 MB vae/ component is loaded -- not the 24 GB transformer -- so this is minutes, not
# hours, and a 24 GB card is ample. The constraint list is Ampere-or-newer only: the decode runs in
# bf16 on CUDA, which Pascal cards (titan_xp) do not support natively.
#
# --inplace is deliberate here (unlike the sampling job): this notebook IS the deliverable, it is
# pure analysis, and it can be regenerated at will since it re-derives nothing.

set -e

WORK=/home/dcor/arielzoizner
SWEEP=$WORK/projects/creativity-measure/notebooks/flux_lambda_sweep
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
NB=flux_lambda_sweep_results.ipynb

export HF_HOME="$WORK/.cache/huggingface"
export HF_TOKEN_PATH="/a/home/cc/students/math/arielzoizner/.hf_token"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$SWEEP"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

"$PY" - <<'PREFLIGHT'
import torch
from huggingface_hub import auth_check
assert torch.cuda.is_available(), "no CUDA device visible"
print(f"GPU: {torch.cuda.get_device_name(0)}  "
      f"{torch.cuda.mem_get_info()[1] / 2**30:.0f} GB")
assert torch.cuda.is_bf16_supported(), "card lacks native bf16; widen --constraint or decode in fp32"
auth_check("black-forest-labs/FLUX.1-dev")
print("FLUX.1-dev access: OK")
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
ls -l "$SWEEP/$NB" "$SWEEP/lambda_sweep_grid.png" 2>/dev/null || true
echo "PNGs: $(ls "$SWEEP/decoded" 2>/dev/null | wc -l)"
