#!/bin/sh
#SBATCH --job-name=k-decode
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gpus=1
#
# Decode the k-sweep result latents into images. VAE only (~168 MB), no transformer, no staging --
# so this is minutes, not hours, and --mem can be far below the 24000 the sampling jobs need.
#
#   sbatch --nodelist=n-802,n-803,n-805 decode.slurm
#
# Decodes the lambda = 0 control alongside the tilted runs on purpose: "did the tilt change the
# picture" is only answerable against the untilted cloud from the same seed and base trajectory.

set -e
WORK=/home/dcor/arielzoizner
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
RUN_DIR=$WORK/projects/creativity-measure/notebooks/flowmap_smc_k_sweep

export HF_HOME="$WORK/.cache/huggingface"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR"
cd "$RUN_DIR"

# The VAE lives in the staged snapshot; reuse it if this node has one, else fall back to $HF_HOME.
SNAP=$(ls -d /tmp/$USER/flux-snap-*/ 2>/dev/null | head -1)
if [ -n "$SNAP" ] && [ -d "${SNAP}vae" ]; then
    export FLUX_LOCAL_PATH="${SNAP%/}"
    export HF_HUB_OFFLINE=1
    echo "using node-local stage: $FLUX_LOCAL_PATH"
else
    echo "no node-local stage here; resolving the VAE from $HF_HOME"
fi

echo "=== $(date) on $(hostname) ==="
"$PY" decode_run.py \
    result_N8_K1_m0_seed101_ctrl2.pt \
    result_N8_K4_m1.25_seed101_k04.pt \
    result_N8_K8_m1.25_seed101_k08b.pt \
    result_N8_K16_m1.25_seed101_k16.pt \
    result_N8_K32_m1.25_seed101_k32.pt
echo "=== done $(date) ==="
ls -l "$RUN_DIR/decoded" | head -20
