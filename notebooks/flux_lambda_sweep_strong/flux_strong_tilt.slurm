#!/bin/sh
#SBATCH --job-name=flux-guid-tilt
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000
#SBATCH --time=360
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Wider-p + stronger-tilt follow-up: GUIDANCE 1.5 (vs the baseline's 3.5), m = 5 (vs 3),
# sweep cap 24 (vs 16), 6 h.
#
# GUIDANCE is the headline change. CLAUDE.md records that SMC cannot escape the generator manifold
# (particles are x = G(z)), so at guidance 3.5 with prompt "A dog" that manifold IS "ordinary dog
# photographs" and no lambda can leave it. 1.5 widens p, at zero extra compute -- FLUX.1-dev is
# guidance-distilled, so the scale is an embedding rather than a second forward pass.
#
# TRUNCATION IS EXPECTED AND ACCEPTED. From the baseline's MEASURED 1.91 min/sweep, 24 sweeps is
# ~46 min/level, so 6 h buys ~6.4 levels; reaching beta=1 at m=5 needs ~8.3 (levels ~ m/0.6, with
# c=0.6 calibrated on the baseline's 5 beta-steps at m=3). So expect to stop near beta ~ 0.77, i.e.
# lambda_eff ~ 3.9 sigma. That is safe: the kernel checkpoints every sweep and stops cleanly at a
# deadline derived from SLURM_JOB_END_TIME, and the notebook then renders the levels that completed.
# --time=480 would reach beta=1 if the extra two hours are ever worth spending.
#
# `killable` is preemptible and caps at 1 day. If preemption costs a run, --partition=gpu-b200 has a
# 5-day limit and is not preemptible (but is premium shared hardware -- do not default to it).
#
# No `source ~/.bashrc`: that file is empty here and conda/HF_* live in ~/.cshrc (tcsh), so this
# script sets everything explicitly and works from a fresh shell or a requeue.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
SWEEP_DIR=$REPO/notebooks/flux_lambda_sweep_strong
NB=flux_dev_strong_tilt_sweep.ipynb

# $WORK is group-readable, so the HF *credential* must not live there; the 32 G of weights may.
export HF_HOME="$WORK/.cache/huggingface"
export HF_TOKEN_PATH="/a/home/cc/students/math/arielzoizner/.hf_token"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$SWEEP_DIR"          # the notebook derives REPO from cwd ("../..") and writes results here

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Preflight: fail in seconds, not after minutes of model loading. The baseline run died here in 29 s
# on a torch+cu130 build against a 12.x driver -- exactly the class of failure this catches.
"$PY" - <<'PREFLIGHT'
import torch
from huggingface_hub import auth_check
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)}  {gb:.0f} GB total")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
auth_check("black-forest-labs/FLUX.1-dev")
print("FLUX.1-dev access: OK")
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
echo "notebook with outputs: $SWEEP_DIR/$OUT"
ls -l "$SWEEP_DIR"/*.pt "$SWEEP_DIR"/strong_tilt_grid.png 2>/dev/null || true
