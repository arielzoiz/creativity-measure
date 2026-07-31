#!/bin/sh
#SBATCH --job-name=flux-tilt2
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000
#SBATCH --time=480
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Run 2 -- calibrated IEM knobs on the cached/batched distance. 8 h.
#
# WHAT CHANGED vs job 694278 (which ran the PRE-migration distance):
#   R        4 -> 64    free at sampling now: commit e1bfc99 caches the reference score bank, so the
#                       per-call cost is (N_GAMMA-1)*NUM_EPS*B score rows, INDEPENDENT of R, and the
#                       R x R normalizer costs zero extra rows (batch_scores reuses the bank).
#   N_GAMMA  6 -> 30    the real fix. demo_properties.ipynb Concl. 2 calibrates N_gamma >= 25 for f to
#                       converge, and argues it transfers to high d (a 1-D integral over gamma). At 6
#                       the reward was not the IEM, so nothing 694278 said about WHAT f measures holds.
#   N        8 -> 16    ESS on 8 particles is unusable; 16 also gives honest degeneracy signals.
#   M_TILT   5 -> 3     694278 reached m_eff = 0.67 of a targeted 5. Chasing m=5 guarantees truncation
#                       at low lambda_eff; m=3 is the largest value with a chance of completing.
#   --time 360 -> 480
#
# COST HAS MOVED. At N_GAMMA=30 the reward is ~85% of a sweep (87 score rows/particle vs 15 for G), so
# the generator is no longer worth economising -- N_STEPS stays 8.
#
# TRUNCATION IS STILL EXPECTED and is handled: the kernel checkpoints the cloud every sweep and stops
# cleanly at a deadline derived from SLURM_JOB_END_TIME, then the notebook renders whatever completed.
# 694278 measured d_beta ~ 0.019/level at m=5; even at m=3 reaching beta=1 needs far more levels than
# 8 h buys. The knobs cell prints the projection from a MEASURED per-row cost rather than a constant --
# both previous overruns came from trusting a stale S_PASS.
#
# OOM: MEASURED AND FIXED. `_scores_at_gamma` issues one forward pass of NUM_EPS*P rows, so the
# reference-bank build asks for NUM_EPS*R = 192 rows. Job 695266 probed this card directly:
#   rows    24    48    96   128   192          (with the 1.36 GB ref bank resident)
#   status  ok    ok    ok    ok   OOM
#   free  15.8   8.2   2.3   0.6     -  GB
#   s/row 0.259 0.251 0.255 0.263     -  <- flat, so a narrow cap costs no throughput
# 192 OOMs, reproducing what killed job 694256. `chunked_denoiser` (creativity_measure/distances/
# edm_adapter.py) now caps every denoiser batch at 24 rows, wrapped once in cell 3 and passed to both
# edm_generator and edm_score_fn, which covers the generator path and every score path including the
# bank build. 24 over the 128 ceiling because s/row is flat and 128 leaves only 0.6 GB for an 8 h run.
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
SWEEP_DIR=$REPO/notebooks/flux_lambda_sweep_strong_2
NB=flux_dev_strong_tilt_sweep_2.ipynb

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
