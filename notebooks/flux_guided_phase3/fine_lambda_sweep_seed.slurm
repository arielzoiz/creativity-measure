#!/bin/sh
#SBATCH --job-name=flux-fine-lambda-seed
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=260
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Seed-replication of fine_lambda_sweep_zoom.slurm (job 958630, the max5.5 run): SAME model, prompt,
# reward config (R_REFS=64, N_GAMMA=50, NUM_EPS=1, REF_SEED=7, PROBE_SEED=9990) and SAME 11 lambda
# points (uniform in [0, 3.928571428571429], i.e. idx00-idx10 of the max5.5 sweep) -- only SWEEP_SEED
# (the initial noise z0) differs, to see how much of the found creative-but-recognizable window is
# seed-specific vs reproducible. One job per seed (SEED env var, set via `sbatch --export=SEED=<n>`),
# run in parallel across 4 jobs.
#
# Cost: same ~1h one-time setup (model load + ref-bank build + probe) as job 958630, plus 1 untilted
# config (~10s) + 10 guided configs x ~570s (measured in 958630 at N_PARTICLES=1, exact_jacobian=True)
# =~ 5710s =~ 95min. Total ~155min; --time=260 budgets with margin.
#
# Same TRITON_CACHE_DIR / HF_TOKEN lessons as every other Phase 2/3 slurm script.

set -e

if [ -z "$SEED" ]; then
    echo "FATAL: SEED env var not set. Submit with:" >&2
    echo "  sbatch --export=ALL,SEED=2024 fine_lambda_sweep_seed.slurm" >&2
    exit 1
fi

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
SWEEP_DIR=$REPO/notebooks/flux_guided_phase3

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$WORK/.cache/triton"

mkdir -p "$WORK/logs" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
cd "$SWEEP_DIR"

echo "=== $(date) on $(hostname), SEED=$SEED ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" log -1 --format=%s))"
git -C "$REPO" status --short | head -5

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set in the environment. Submit with:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch --export=ALL,SEED=2024 fine_lambda_sweep_seed.slurm" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

echo "=== running fine_lambda_sweep.py --lam-max 3.928571428571429 --n-lambdas 11 --sweep-seed $SEED --tag seed$SEED ==="
"$PY" -u fine_lambda_sweep.py --lam-max 3.928571428571429 --n-lambdas 11 --sweep-seed "$SEED" --tag "seed$SEED"

echo "=== done $(date) ==="
ls -l "$SWEEP_DIR"/fine_lambda_sweep_results_seed"$SEED".json "$SWEEP_DIR"/fine_decoded_seed"$SEED"/*.png 2>/dev/null | tail -20
