#!/bin/sh
#SBATCH --job-name=flux-tilt2-1
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
# Run 2.1: continue run 2 (job 695271) from beta=0.5422 toward beta=1. 6 h.
#
# Job 695271 reached beta=0.5422 (m_eff 1.63 of 3) in 54 sweeps before its 8 h deadline. This picks up
# that particle cloud instead of restarting from p, saving the 5 levels already paid for.
#
# HOW. pi_beta ~ p*exp(beta*lam*f); with beta = beta0 + (1-beta0)u this is pi_beta0 * exp(lam'*u*f)
# for lam' = (1-beta0)*lam = 126.59. So the residual leg is an ordinary SMC whose STARTING
# DISTRIBUTION is the saved cloud -- no library change, only PCNKernel.init is overridden. The SMC's
# reported beta/lam_eff are therefore relative to the residual leg and are remapped in the results
# cells (lam_eff_true = 149.97 + lam_eff_run); without that every row reads low by 1.63 sigma.
#
# REWARD MUST REPRODUCE BIT-FOR-BIT. R=64, N_GAMMA=30, NUM_EPS=3, ref seed 7, distance seed 123 and
# MAX_DENOISER_ROWS=24 are all carried over unchanged (the last matters because the references come
# from G). The notebook recomputes f on the saved cloud and ABORTS if it disagrees with the stored
# fX by more than 0.01% of std_p(f) -- otherwise the leg would silently target a different
# distribution than the levels it is stitched onto.
#
# OUTPUTS ARE TAGGED (RUN_TAG=2_1). The untagged names derive from N/M_TILT/SEED, none of
# which change on a resume, so an untagged run would overwrite partial_N16_m3_seed101.pt -- its own
# resume source -- on the first sweep. An assertion in the notebook enforces src != dst.
#
# MAX_SWEEPS 24 -> 40: precautionary. The cap was NEVER reached in 695271 (12/11/20/11 sweeps), so it
# was not binding; but acceptance fell 0.75 -> 0.12 and is likely to make it binding as lambda rises.
#
# COST. Remaining beta 0.4578. 6 h buys ~4.5 h of sweeping after setup (~0.5 h) and reserve (1 h),
# i.e. ~40 sweeps. At 695271's d_beta=0.136 and 13.5 sweeps/level that is ~2.9 levels -> beta ~0.94;
# with acceptance falling (d_beta -30%, 20 sweeps/level) ~2.0 levels -> beta ~0.73. BETA=1 IS UNLIKELY
# IN 6 H -- this leg advances the ladder rather than completing it, and a further resume from its own
# checkpoint is the intended continuation (the `s` field is now persisted, so that one keeps the
# adapted pCN step size). The kernel checkpoints every sweep and stops cleanly at a deadline from
# SLURM_JOB_END_TIME, as before.
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
NB=flux_dev_strong_tilt_sweep_2_1.ipynb

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
ls -l "$SWEEP_DIR"/*2_1*.pt "$SWEEP_DIR"/strong_tilt_grid_2_1.png 2>/dev/null || true
