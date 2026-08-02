#!/bin/sh
#SBATCH --job-name=auto-r-fps
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s|a6000
#SBATCH --time=420
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Auto-R diagnostic, arm 2/2 (WeightedFPSRefs). Runs in PARALLEL with auto_r_diagnostic.slurm on its
# own GPU; the two arms share nothing at runtime except auto_r_common.py, which pins the model, the
# gamma grid, the Brownian seed and the probe points so their tau curves are directly comparable.
#
# COST (exact, from a CPU dry run at these controls): pool 160 + 2 Voronoi estimation sets of 128 +
# 128 probes are each scored once (544 points), plus one bank per greedy pick (64), the incremental
# D_est columns (96) and the tau block's 64 references => ~696 points x 102 rows = 71,000 rows =
# 4.3 h at 0.22 s/row (4.9 h at 0.25). --time=420 leaves margin.
#
# WHY THE LADDER STOPS AT R=32. Not VRAM -- the two things that scale with the top of the ladder:
#   * the pool must exceed 2*R_max with room to spare, or greedy "farthest point" degenerates into
#     "take the whole pool" and the top-R tau is meaningless (here 64 of 160);
#   * the Voronoi weights need M_est >> 2R points, in TWO independent sets (the library estimates the
#     R- and 2R-side weights separately to de-correlate weight noise), at 87 rows per point.
# Measured alternatives: pool 192 / grid<=16 = 3.4 h; pool 256 / grid<=64 = 5.6 h AND only ~1
# estimation point per cell at 2R=128, i.e. weight-noise-limited exactly where the answer would be.
#
# WHY THIS IS RUNNABLE AT ALL. FPSRefs._ensure_order calls pairwise(pool, one_new_ref) once per greedy
# pick. Uncached, each pick re-scores the whole pool: 87*160 rows x 64 picks ~ 48 h. auto_r_common's
# CachedSquaredGlobalIEM banks BOTH pairwise arguments under a byte budget (the library caches the
# reference side only, in a single slot -- right for the SMC, wrong for this access pattern), so the
# pool is scored once and each pick costs 87 rows. Cached and uncached results are bit-identical.
#
# VRAM: pool + 2 estimation sets + probes = 544 x 22.8 MB = 12.1 GB of banks beside the model's
# 22.3 GB and ~3.6 GB of activations. The notebook prints a warning if the working set exceeds the
# self-sized cache budget (which would mean LRU thrashing, not an OOM).

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
DIAG_DIR=$REPO/notebooks/refset_auto_r
NB=refset_auto_r_fps.ipynb

# $WORK is group-readable, so the HF *credential* must not live there; the 32 G of weights may.
export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$DIAG_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# See auto_r_diagnostic.slurm: the HF token lives in $HOME, which is not mounted on every killable
# node (job 697403 died with a 401 on t-806). No network is needed -- the 32 G snapshot is cached.
HF_TOKEN_FILE="$HOME/.hf_token"
if [ -r "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN_PATH="$HF_TOKEN_FILE"
    echo "HF: token readable on $(hostname) -> authenticated mode"
else
    export HF_HUB_OFFLINE=1
    echo "HF: $HF_TOKEN_FILE NOT readable on $(hostname) (home not mounted) -> HF_HUB_OFFLINE=1, "\
"using the cached snapshot under HF_HOME"
fi

"$PY" - <<'PREFLIGHT'
import sys
import torch
from scipy.stats import weightedtau          # auto-R needs this; fail now, not 4 h in
sys.path.insert(0, ".")
from auto_r_common import check_model_access  # token, or the local snapshot when offline
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)}  {gb:.0f} GB total")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
check_model_access()
print("scipy.weightedtau: OK")
PREFLIGHT

OUT="run_fps_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT ==="
"$PY" -m jupyter nbconvert \
    --to notebook \
    --execute \
    --allow-errors \
    --output "$OUT" \
    --ExecutePreprocessor.timeout=-1 \
    "$NB"

echo "=== done $(date) ==="
ls -l "$DIAG_DIR/auto_r_results_fps.json" "$DIAG_DIR/auto_r_tau_fps.png" 2>/dev/null || true
