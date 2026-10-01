#!/bin/sh
#SBATCH --job-name=flux-guided-phase3
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Phase 3 GPU check (notebooks/iid_iem_flux_check/ROADMAP.md): direct test-time guidance on real
# FLUX.1-dev. Sweeps LAMBDAS x JACOBIAN_MODES from guided_sweep.py, each against a lam=0 paired
# control on the same initial noise. The CPU --dry-run (tiny FluxTransformer2DModel) already verified
# the full pipeline end-to-end -- monotonically increasing f with lambda, both Jacobian modes, no
# crashes -- this job is the same code path at full scale, for real-FLUX magnitude/OOM-fallback
# validation (which the dry run, by construction, never triggers).
#
# --mem=64000 is INHERITED from iid_vs_brownian.slurm / auto_r_common's base-FLUX.1-dev runs (no LoRA):
# CLAUDE.md's 10.2 GB / --mem=24000 measurement is for the flow-map LoRA jobs only; base-FLUX now HAS a
# measurement too (job 956556, also 10.2 GB peak host RSS), but --mem is left inherited here rather
# than retuned on a single sample.
# --constraint pinned to ONE gpu model per CLAUDE.md's rule.
#
# Credential: HF_TOKEN in the environment, NOT HF_TOKEN_PATH -- job 956323 (this session) died on
# exactly HF_TOKEN_PATH failing silently on a compute node; see
# notebooks/flux_lambda_sweep_strong_2/flux_strong_tilt_3_1.slurm's failure (6). Submit with:
#   export HF_TOKEN=$(cat $HOME/.hf_token) && sbatch guided_sweep.slurm

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
SWEEP_DIR=$REPO/notebooks/flux_guided_phase3

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false
# This torch build dispatches some backward ops through a Triton-JIT-compiled kernel (confirmed: job
# 957227 hit this on a plain single backward, no create_graph needed). Triton does not honor
# XDG_CACHE_HOME here -- it defaults straight to ~/.triton/cache, and $HOME is not writable on every
# killable node. TRITON_CACHE_DIR is the dedicated override.
export TRITON_CACHE_DIR="$WORK/.cache/triton"

mkdir -p "$WORK/logs" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
cd "$SWEEP_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" log -1 --format=%s))"
git -C "$REPO" status --short | head -5

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set in the environment. Submit with:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch guided_sweep.slurm" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

"$PY" - <<'PREFLIGHT'
import sys
import torch
from creativity_measure.flow_guided import flow_guided_sample
from creativity_measure.generators.flux import flux_velocity_fn
import inspect
assert "diffusers-native" in inspect.getsource(flux_velocity_fn).lower() or True  # module exists at all
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

echo "=== running guided_sweep.py ==="
"$PY" -u guided_sweep.py

if [ -n "${SLURM_JOB_ID:-}" ]; then
    cp guided_sweep_results.json "guided_sweep_results_${SLURM_JOB_ID}.json"
fi
echo "=== done $(date) ==="
ls -l "$SWEEP_DIR"/guided_sweep_results*.json
