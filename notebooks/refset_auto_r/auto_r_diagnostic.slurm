#!/bin/sh
#SBATCH --job-name=auto-r-random
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
# Auto-R diagnostic, arm 1/2 (RandomRefs). Submit ALONGSIDE auto_r_fps.slurm -- the two arms are
# independent jobs on separate GPUs and neither waits for the other; whichever finishes second draws
# the joint figure (else run compare_arms.py, CPU, on the login node).
#
# WHAT KILLED THE PREVIOUS ATTEMPT (job 694256): pairwise() fed num_eps*rows samples to the
# transformer in one batch, so 64 probes = 192 rows = OOM in BOTH arms, and --allow-errors let the
# notebook "complete" with nothing. Fixed in the library since, by chunked_denoiser (batch ceiling)
# and reference score-bank caching. Both are used here via notebooks/refset_auto_r/auto_r_common.py,
# which both arms import so their tau curves are computed against a bit-identical f.
#
# COST (exact, from a CPU dry run at these grid sizes -- the row count depends only on the point
# counts, not on d): 128 probes + 2 draws x 256 references = 640 points, each 15 generation rows +
# 87 score rows, at a measured 0.22 s/row (job 695271's R=64 bank: 1404 s / 6528 rows) = 65,520 rows
# = 4.0 h, or 4.6 h at the pessimistic 0.25 s/row from job 695266. --time=420 leaves margin for the
# ~4 min model load and the free post-processing.
#
# WHY 2 DRAWS. It is the dominant cost (3.2 h of the 4.0) and it buys the only unbiased measurement
# available: tau(f_R^A, f_R^B) over DISJOINT reference sets. The library's nested R-vs-2R rule shares
# R references and is optimistic by construction -- both selector docstrings say so without
# quantifying it. The two draws also concatenate into one i.i.d. 512-reference set, giving tau at
# R=256 for free; a real 512-reference block would need 11.4 GB resident (22.8 at the score_bank
# stack peak) beside a 22.3 GB transformer, i.e. OOM on a 48 GB card.
#
# VRAM: 640 points x 22.8 MB of bank = 14.2 GB at the peak, next to the model's 22.3 GB and ~3.6 GB
# of activations at MAX_DENOISER_ROWS=24. The cache in auto_r_common sizes itself from the card's
# free memory and evicts LRU, so a smaller card degrades into re-scoring rather than an OOM.
#
# The notebook writes auto_r_results_random.json after the auto-R call AND after each analysis cell,
# so a timeout still leaves the finished part on disk.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
DIAG_DIR=$REPO/notebooks/refset_auto_r
NB=refset_auto_r_diagnostic.ipynb

# $WORK is group-readable, so the HF *credential* must not live there; the 32 G of weights may.
export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$DIAG_DIR"          # the notebook derives REPO from cwd ("../..") and imports auto_r_common there

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# FLUX.1-dev is gated, but this job needs NO network: the full 32 G snapshot is already in HF_HOME.
# The token has to live in $HOME (/a/home/cc/...) because $WORK is group-readable -- and $HOME is not
# mounted on every node in `killable`. Job 697403 died in 33 s with a 401 on t-806 minutes after the
# identical setup worked on n-805. Depending on the home mount is the bug; falling back to the cached
# snapshot is the fix, and the check below records which path this node took.
HF_TOKEN_FILE="$HOME/.hf_token"
if [ -r "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN_PATH="$HF_TOKEN_FILE"
    echo "HF: token readable on $(hostname) -> authenticated mode"
else
    export HF_HUB_OFFLINE=1
    echo "HF: $HF_TOKEN_FILE NOT readable on $(hostname) (home not mounted) -> HF_HUB_OFFLINE=1, "\
"using the cached snapshot under HF_HOME"
fi

# Preflight: fail in seconds, not after minutes of model loading.
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

# Separate output notebook, never --inplace: nbconvert writes once at the very end, so a kill during
# that write could truncate the source. --allow-errors keeps the diagnostics already produced, with
# the traceback in the failing cell -- the JSON is the real deliverable either way.
OUT="run_random_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT ==="
"$PY" -m jupyter nbconvert \
    --to notebook \
    --execute \
    --allow-errors \
    --output "$OUT" \
    --ExecutePreprocessor.timeout=-1 \
    "$NB"

echo "=== done $(date) ==="
ls -l "$DIAG_DIR/auto_r_results_random.json" "$DIAG_DIR/auto_r_tau_random.png" 2>/dev/null || true
