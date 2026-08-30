#!/bin/sh
#SBATCH --job-name=flowmap-strength
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=a6000
#SBATCH --nodelist=n-602
#SBATCH --time=240
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# PHASE 6 -- does Algorithm 3's base transition regenerate SEMANTIC diversity?  (lambda = 0)
#
#   sbatch flowmap_smc_strength.slurm
#   sbatch --export=ALL,STRENGTH_LIST=0,1.0 flowmap_smc_strength.slurm     # see the comma warning below
#
# THE QUESTION. a3_1 / a3_2 / nv_1 all end with `uniq/N` predicting the number of distinct images
# exactly (a3_2 m=1 ends at 0.375 = 3 ancestors, and the grid shows 3 scenes then 6 copies), while
# those copies sit 188-191 apart in latent space against ~362 for iid draws. So `ddpm_step` moves
# clones without changing what they depict. `posterior_z_ddpm_step` restores the variance the plug-in
# z_hat discards -- c_z^2 * Cov(x_1|x_t), the denoising posterior covariance -- at ZERO extra network
# calls. This job sweeps its `strength` at lambda = 0 and measures whether that changes the pictures.
#
# WHY IT IS CHEAP. guid_window=(1.0,1.0), so only the mandatory terminal update fires: ONE reward call
# on M rows per configuration instead of the usual 2784 rows per guided step. ~15 min of compute. The
# clone probe needs no reward at all. Budget is dominated by the reference bank (~19 min), not the run.
#
# strength=0 IS `ddpm_step` bit-for-bit and is included deliberately: it is the in-job baseline, so the
# comparison never depends on matching a previous run's GPU. That matters here because a3_* ran on an
# L40S and this runs on an A6000 (see decision 2) -- E_p[f] is comparable WITHIN this job only.
#
# ---------------------------------------------------------------------------------------------------
# FOUR DECISIONS, all of them fixes for failures the older flowmap_smc.slurm predates. That script was
# written 2026-08-02 and job 781449 died on the first of them within 60 s of starting.
# ---------------------------------------------------------------------------------------------------
#
# (1) THE HUB IS NEVER CONTACTED. Job 781449 died in the preflight on
#     `401 GatedRepoError ... FLUX.1-dev`: $HOME/.hf_token is stale, and the only working credential
#     was ever an HF_TOKEN exported in an interactive shell, which a submitted script does not inherit.
#     Nothing here needs the hub -- the 27-file snapshot is in $HF_HOME/hub and the flow-map LoRA
#     (gabeguofanclub/flux-1-dev-flowmap-lsd) is cached too. So the auth probe is advisory, staging is
#     the real guard, and HF_HUB_OFFLINE=1 is exported before the notebook runs. To restore the online
#     path, write a valid token to $HOME/.hf_token at 0600 (never under $WORK, which is group-readable).
#     THE PREFLIGHT IS NOT THE ONLY CALLER. Job 781469 got past this preflight and then died in the
#     notebook's own cell 1, which had its own bare `auth_check(...)` -- offline mode turned the 401
#     into `OfflineModeIsEnabled`, i.e. the dodge for failure 1 CAUSED failure 2. That call is now
#     advisory too and records HUB_ACCESS for the log. Audited: the only other hub-touching call is
#     `from_pretrained(MODEL_PATH)` on a staged directory, plus `hf_hub_download` for the flow-map LoRA,
#     which is cached and was verified to resolve under HF_HUB_OFFLINE=1 before resubmitting.
#
# (2) --nodelist=n-602 AND --constraint=a6000. The FLUX snapshot is 32 G on an NFSv3 netapp mount and
#     safetensors are memory-mapped, so a cold node reads it as random 32 KB NFS reads: job 755539
#     MEASURED 257 min at ~2 MB/s. n-602's SHA-keyed /tmp tree is the only warm one and is deliberately
#     never cleaned up. 781449 landed on n-803 (cold, L40S) -- with --time=150 it would have died
#     mid-load even if the token had worked, so the auth failure cost nothing.
#     A6000 also matches the card the persisted reference latents were generated on; bf16 kernels
#     differ across architectures and that shifted `f` by ~11-16% of std_p(f) in job 697271.
#     If a MISS is reported below, the run still fits in --time=240, but cancel and rethink instead:
#     that means the staged tree is gone and every future job pays 257 min too.
#
# (3) NO JUPYTER KERNEL. Jobs 756120/756173/756235 died before cell 1 on a client<->kernel handshake
#     that never completed (60 s and 600 s timeouts, and moving JUPYTER_RUNTIME_DIR did not help).
#     run_notebook.py execs the cells in a plain Python process and streams to the job log. It drops
#     inline images, so PHASE 6 savefig's every grid -- nothing scientific lives in the notebook.
#
# (4) `sbatch --export` SPLITS ON COMMAS, so STRENGTH_LIST=0,0.25,0.5,1.0 CANNOT be passed that way --
#     it parses as four separate assignments. The default lives in the notebook; to override, either
#     edit it there or `export STRENGTH_LIST=...` before submitting without --export.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
RUN_DIR=$REPO/notebooks/flowmap_smc_flux
NB=flowmap_smc_flux.ipynb

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1                          # stdout is a file here, so Python would block-buffer
export JUPYTER_RUNTIME_DIR=/tmp/$USER/jupyter-runtime-${SLURM_JOB_ID:-manual}
mkdir -p "$JUPYTER_RUNTIME_DIR" "$WORK/logs" "$MPLCONFIGDIR"

export STOP_AFTER=strength
export RUN_TAG="${RUN_TAG:-str_1}"

cd "$RUN_DIR"          # the notebook derives REPO from cwd ("../..") and writes results here

echo "=== $(date) on $(hostname)  STOP_AFTER=$STOP_AFTER  RUN_TAG=$RUN_TAG ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Preflight before staging, so a broken environment burns 60 s rather than hours. The auth probe is
# advisory (decision 1) and never exits nonzero.
"$PY" - <<'PREFLIGHT'
import torch, peft
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
name = torch.cuda.get_device_name(0)
print(f"GPU: {name}  {gb:.0f} GB total   peft {peft.__version__}")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
if "A6000" not in name:
    print(f"WARNING: expected an A6000 (the card the persisted reference latents were made on); "
          f"got {name!r}. This job is self-consistent -- strength=0 is its own baseline -- but "
          f"E_p[f] is then not comparable to a3_*.")
try:
    from huggingface_hub import auth_check
    auth_check("black-forest-labs/FLUX.1-dev")
    print("FLUX.1-dev hub access: OK (unused -- this run loads offline from the staged snapshot)")
except Exception as exc:
    print(f"hub access: UNAVAILABLE ({type(exc).__name__}). Not fatal: this run loads offline from "
          f"the staged local snapshot and never resolves a repo id.")
PREFLIGHT

# --- stage the weights to node-local disk (decision 2) ---------------------------------------------
# cp -L dereferences: the HF cache keeps blobs behind symlinks into ../../blobs, and the staged tree
# must stand alone. Parallel over files because a single NFS stream is latency-bound -- hence 6 cpus.
SNAP=$(ls -d "$HF_HOME"/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
    echo "FATAL: no FLUX snapshot under $HF_HOME/hub" >&2
    exit 1
fi

# THE CACHE IS KEYED BY SNAPSHOT SHA AND PERSISTS ACROSS JOBS -- do not "clean it up". There is
# deliberately no EXIT trap removing $STAGE: that is what would throw the 257 min away.
SHA=$(basename "${SNAP%/}")
STAGE=/tmp/$USER/flux-snap-$(echo "$SHA" | cut -c1-8)
N_SRC=$(cd "$SNAP" && find -L . -type f | wc -l)

find /tmp/$USER -maxdepth 1 -name 'flux-snap-*' ! -name "$(basename "$STAGE")" -mtime +7 \
    -exec rm -rf {} + 2>/dev/null || true
find /tmp/$USER -maxdepth 1 -name 'jupyter-runtime-*' -mtime +1 -exec rm -rf {} + 2>/dev/null || true

if [ -f "$STAGE/model_index.json" ] && [ "$(find -L "$STAGE" -type f | wc -l)" -eq "$N_SRC" ]; then
    echo "=== node-local cache HIT: $STAGE ($(du -sh "$STAGE" | cut -f1), $N_SRC files) -- no staging ==="
else
    echo "=== node-local cache MISS on $(hostname): staging $(du -shL "$SNAP" | cut -f1) from NFS." \
         "MEASURED cost 257 min (job 755539). This job's real work is only ~40 min, so a MISS means" \
         "the warm tree is gone -- consider cancelling and investigating rather than paying it here. ==="
    t_stage=$(date +%s)
    rm -rf "$STAGE.partial"
    mkdir -p "$STAGE.partial"
    ( cd "$SNAP" && find -L . -type d -exec mkdir -p "$STAGE.partial/{}" \; )
    ( cd "$SNAP" && find -L . -type f -print0 | xargs -0 -P 6 -I{} cp -L {} "$STAGE.partial/{}" )
    mv "$STAGE.partial" "$STAGE"
    echo "=== staged in $(( ($(date +%s) - t_stage) / 60 )) min: $(du -sh "$STAGE" | cut -f1) ==="
fi
chmod -R u+rwX,go-rwx /tmp/$USER 2>/dev/null || true

if [ ! -f "$STAGE/model_index.json" ]; then
    echo "FATAL: staged tree has no model_index.json -- copy incomplete" >&2
    exit 1
fi
export FLUX_LOCAL_PATH="$STAGE"
export HF_HUB_OFFLINE=1        # weights are local and complete; a stale token cannot 401 mid-load

# The CPU-only suites must be green before any GPU time is spent. Seconds to run.
echo "=== regression (CPU) ==="
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest \
    "$REPO/tests/test_flowmap_smc.py" "$REPO/tests/test_flux_flowmap.py" -q

# --- execute, kernel-free (decision 3) -------------------------------------------------------------
OUT="run_strength_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT (run_notebook.py, kernel-free, streaming) ==="
set +e
"$PY" run_notebook.py "$NB" "$OUT"
rc=$?
set -e

echo "=== done $(date)  (run_notebook exit $rc) ==="
ls -l "$RUN_DIR"/strength_*"$RUN_TAG"* 2>/dev/null || true
exit $rc
