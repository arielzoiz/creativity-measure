#!/bin/sh
#SBATCH --job-name=iid-iem-check
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Phase 1 GPU check of the i.i.d. Monte-Carlo squared-IEM reward: G0 (per-row gamma on the real FLUX
# transformer, PASS/FAIL) and G1 (f_IID vs f_Brownian on 32 held-out latents, R=64 refs). See the docstring
# of iid_vs_brownian.py for what each stage claims and the acceptance rule.
#
# COST (rows x 0.22 s/row, auto_r_common's measurement):
#   generation           (64 refs + 32 held-out) x 15                          =  1,440 rows
#   Brownian x2 banks    2 x (64+32) x (N_GAMMA-1)*NUM_EPS = 2 x 96 x 87       = 16,704 rows
#   IID (G,E) x 3 seeds  (30*1 + 50*1 + 30*2) x 96 x 3 = 140 x 96 x 3          = 40,320 rows
#   total 58,464 rows = 3.6 h  (+ ~10 min model load).  --time=300 leaves ~1.3 h of margin.
# Progress is written to iid_vs_brownian_results.json after every config, and finished configs are skipped on
# a rerun with the same stamp, so a preempted `killable` job loses at most one config (<= ~30 min).
#
# --constraint is pinned to ONE gpu model on purpose (CLAUDE.md): a GPU change shifts f by ~16% of std_p(f), the
# same order as the differences this job is measuring.
# --mem=64000 is INHERITED, not measured: it is what the refset_auto_r jobs (same auto_r_common.build(), base
# FLUX.1-dev) ran with. CLAUDE.md's 10.2 GB / --mem=24000 is for the flow-map LoRA jobs only and says to measure
# the base-FLUX runs before lowering. iid_vs_brownian.py stamps host peak RSS (ru_maxrss) on every progress line,
# so this run produces that measurement; lower --mem from it next time (never to exactly the peak).

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

# The HF token lives in $HOME, which is not mounted on every killable node (job 697403 died with a 401):
# fall back to the cached snapshot offline, exactly as the refset_auto_r jobs do.
HF_TOKEN_FILE="$HOME/.hf_token"
if [ -r "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN_PATH="$HF_TOKEN_FILE"
    echo "HF: token readable on $(hostname) -> authenticated mode"
else
    export HF_HUB_OFFLINE=1
    echo "HF: $HF_TOKEN_FILE NOT readable on $(hostname) -> HF_HUB_OFFLINE=1, using the cached snapshot"
fi

"$PY" - <<'PREFLIGHT'
import sys
import torch
import scipy.stats                                                    # summary uses rankdata; fail now, not 3 h in
# A stale checkout fails HERE (seconds) instead of after the queue wait and the model load.
from creativity_measure import SquaredIIDGlobalIEMDistance, log_uniform_gammas, edm_score_fn
import inspect
assert "gamma.reshape(-1" in inspect.getsource(edm_score_fn), "edm_score_fn lacks per-row gamma: `git pull` the repo"
sys.path.insert(0, "../refset_auto_r")
from auto_r_common import check_model_access
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)}  {gb:.0f} GB total")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
check_model_access()
print("preflight: OK (new API present)")
PREFLIGHT

echo "=== running iid_vs_brownian.py ==="
"$PY" -u iid_vs_brownian.py

if [ -n "${SLURM_JOB_ID:-}" ]; then
    cp iid_vs_brownian_results.json "iid_vs_brownian_results_${SLURM_JOB_ID}.json"
fi
echo "=== done $(date) ==="
ls -l "$CHECK_DIR"/iid_vs_brownian_results*.json
