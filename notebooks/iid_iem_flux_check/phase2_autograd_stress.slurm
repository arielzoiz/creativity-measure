#!/bin/sh
#SBATCH --job-name=phase2-autograd-stress
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Phase 2 GPU check (notebooks/iid_iem_flux_check/ROADMAP.md): the three autograd/memory safety
# mechanisms (weight freezing, Flash-Attention double-backward trap + SDPA-MATH fallback, OOM iterative
# accumulation), run against the real FLUX.1-dev differentiable denoiser (build_flux_denoiser). The CPU
# --dry-run already verified all three mechanisms end-to-end against a tiny FluxTransformer2DModel
# (including the double-backward trap actually firing and recovering) -- this job is the same code path
# on the real 12B transformer, for wiring/memory-scale validation only.
#
# --mem=64000 is INHERITED from iid_vs_brownian.slurm / auto_r_common's base-FLUX.1-dev runs (no LoRA):
# CLAUDE.md's 10.2 GB / --mem=24000 measurement is for the flow-map LoRA jobs only and explicitly says to
# measure the base-FLUX runs (this one included) before lowering. note() stamps host peak RSS on every
# progress line, so this run produces that measurement.
# --constraint is pinned to ONE gpu model per CLAUDE.md's rule (a GPU change shifts f by ~16% of
# std_p(f) on the *value* path; not load-bearing for a gradient-mechanics check like this one, but kept
# for consistency/reduced variables with every other job in this repo).
#
# Credential: HF_TOKEN in the environment, NOT HF_TOKEN_PATH -- see
# notebooks/flux_lambda_sweep_strong_2/flux_strong_tilt_3_1.slurm's failure (6): HF_TOKEN_PATH does not
# reliably authenticate on compute nodes even when the file is readable (root cause never pinned down);
# the fix that has always worked is exporting HF_TOKEN in the submitting shell so sbatch propagates it.
# Submit with:
#   export HF_TOKEN=$(cat $HOME/.hf_token) && sbatch phase2_autograd_stress.slurm

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
CHECK_DIR=$REPO/notebooks/iid_iem_flux_check

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$CHECK_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" log -1 --format=%s))"
git -C "$REPO" status --short | head -5

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set in the environment. Submit with:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch phase2_autograd_stress.slurm" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

"$PY" - <<'PREFLIGHT'
import sys
import torch
# A stale checkout fails HERE (seconds) instead of after the queue wait and the model load.
from creativity_measure.generators.flux import build_flux_denoiser, flux_edm_denoiser
import inspect
assert "nullcontext" in inspect.getsource(flux_edm_denoiser), \
    "flux_edm_denoiser lacks the differentiable flag: git pull the repo"
sys.path.insert(0, "../refset_auto_r")
from auto_r_common import check_model_access
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)}  {gb:.0f} GB total")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
check_model_access()
print("preflight: OK")
PREFLIGHT

echo "=== running phase2_autograd_stress.py ==="
"$PY" -u phase2_autograd_stress.py

if [ -n "${SLURM_JOB_ID:-}" ]; then
    cp phase2_autograd_stress_results.json "phase2_autograd_stress_results_${SLURM_JOB_ID}.json"
fi
echo "=== done $(date) ==="
ls -l "$CHECK_DIR"/phase2_autograd_stress_results*.json
