#!/bin/sh
#SBATCH --job-name=flux-tilt2-1
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=a6000
#SBATCH --time=420
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Run 2.1: continue run 2 (job 695271) from beta=0.5422 toward beta=1.
#
# Job 695271 reached beta=0.5422 (m_eff 1.63 of 3) in 54 sweeps before its 8 h deadline. This picks up
# that particle cloud instead of restarting from p, saving the 5 levels already paid for.
#
# HOW. pi_beta ~ p*exp(beta*lam*f); with beta = beta0 + (1-beta0)u this is pi_beta0 * exp(lam'*u*f)
# for lam' = (1-beta0)*lam = 126.61. So the residual leg is an ordinary SMC whose STARTING
# DISTRIBUTION is the saved cloud -- no library change, only PCNKernel.init is overridden. The SMC's
# reported beta/lam_eff are therefore relative to the residual leg and are remapped in the results
# cells (lam_eff_true = 149.97 + lam_eff_run); without that every row reads low by 1.63 sigma.
#
# ================= TWO FAILED ATTEMPTS ARE ENCODED BELOW. DO NOT UNDO EITHER. =================
#
# (1) 697271 -- WRONG GPU. Slurm put 695271 on an A6000 and 697271 on an L40S under the old
# `--constraint=l40s|a6000`. FLUX runs bf16 (8-bit mantissa, ~0.2% per-op resolution) and Ampere vs
# Ada pick different tensor-core kernels and reduction orders, so identical denoiser inputs give
# different outputs at round-off. Through G on the references, the reference score bank and the
# normalizer, f drifted 1.76e-3 -- only 0.17% of f, but 11% of std_p(f) (which is itself just 1.5% of
# the mean), worth up to 0.22 nats of log-weight at lam'=126.6. The resume guard correctly refused.
# Fixes: --constraint=a6000 (below), an early assert in the notebook, and persisted reference latents
# (refs_R64_seed7.pt) that take G out of the reproducibility chain for good.
#
# (2) 754633 -- NFS. Held an A6000 for the full 6 h and wrote NOTHING: no output notebook, no refs, no
# checkpoint, a .out log ending at "=== executing ... ===". A faulthandler probe (755512) localized it
# to `pipe.to(DEVICE)` -> torch Module._apply -> convert. HF_HOME lives on an NFSv3 netapp mount with
# rsize=32768, and safetensors are MEMORY-MAPPED: `.to(cuda)` is where the 24 GB of weights actually
# get read, as random 32 KB NFS reads. It worked on Jul 31 only because the filer was fast that day.
#   Fix: STAGE the snapshot to node-local disk (/tmp on n-602 is ext4 with 799 G free) with a
#   parallel dereferencing copy, then point the notebook at the staged path via FLUX_LOCAL_PATH.
#   After staging, .to(cuda) reads from local disk at GB/s instead of page-faulting over NFS.
#
# HOW BAD THE FILER ACTUALLY IS (measured by 755539, not estimated). The login node reported 28.9 MB/s
# sequential, from which I projected an ~18 min staging pass. The real copy from the COMPUTE node took
# 257 MINUTES for 32 G -- about 2 MB/s, an order of magnitude worse than the login-side measurement.
# Duplicating that same tree local->local afterwards took under a minute (>500 MB/s), which proves the
# local disk is fine and every bit of the cost is NFS read latency. Do not trust a login-node
# throughput number as a proxy for the compute node's mount.
#   Consequence: the staged tree is now a PERSISTENT, SHA-KEYED, PER-NODE CACHE that later jobs reuse,
#   and there is deliberately no cleanup trap on it. Staging is paid once per node, not once per job.
#
# --time 360 -> 420. The extra hour was meant to absorb staging; with a cache HIT it instead goes to
# sweeping. On a cache MISS the run will be badly truncated -- that is unavoidable and visible in the
# log rather than silent. The notebook derives its deadline from SLURM_JOB_END_TIME so it tracks
# --time automatically; WALL_BUDGET_H in cell 1 is only the non-Slurm fallback and was updated to match.
#
# OBSERVABILITY (the other lesson from 754633). nbconvert buffers every cell's output into the
# notebook and writes it once, at the end -- so a stall is undiagnosable by construction and 6 h
# produced zero evidence. This now runs under PAPERMILL with --log-output (cell stdout streams to
# this job's .out as it happens) and --request-save-on-cell-execute (the output notebook is written
# after every cell, so a kill leaves a partial notebook rather than nothing). Cell 1 also arms a
# faulthandler watchdog that dumps the Python stack every 10 min, so any future hang names its frame.
#
# OUTPUTS ARE TAGGED (RUN_TAG=2_1). The untagged names derive from N/M_TILT/SEED, none of which change
# on a resume, so an untagged run would overwrite partial_N16_m3_seed101.pt -- its own resume source --
# on the first sweep. An assertion in the notebook enforces src != dst.
#
# MAX_SWEEPS 24 -> 40: precautionary. The cap was NEVER reached in 695271 (12/11/20/11 sweeps), so it
# was not binding; but acceptance fell 0.75 -> 0.12 and is likely to make it binding as lambda rises.
#
# COST. Remaining beta 0.4578. After ~20 min staging, ~25 min model load + reference bank, and the 1 h
# reserve, 7 h leaves ~4.2 h of sweeping ~ 37 sweeps. At 695271's d_beta=0.136 and 13.5 sweeps/level
# that is ~2.7 levels -> beta ~0.91; with acceptance falling ~1.8 levels -> beta ~0.79. BETA=1 IS
# UNLIKELY -- this leg advances the ladder, and a further resume from its own checkpoint is the
# intended continuation (`s` is persisted, so that one keeps the adapted pCN step size). The kernel
# checkpoints every sweep and stops cleanly at a deadline from SLURM_JOB_END_TIME.
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
export PYTHONUNBUFFERED=1                          # stdout is a file here, so Python would block-buffer

# Jupyter's runtime dir holds the KERNEL CONNECTION FILE, and it defaults to $HOME/.local/share/
# jupyter/runtime -- on this cluster that is /specific/a/home/cc/students/math/$USER, an automounted
# NFS home, NOT $WORK. Jobs 756120 and 756173 both died with "Kernel didn't respond" (at 60 s and then
# at 600 s, so it is not a slow start): the kernel process came up -- its IPKernelApp banner is in the
# log -- wrote its connection file to NFS, and the client never saw a usable one. 755539 got lucky.
# Putting the runtime dir on node-local disk takes NFS out of the kernel handshake entirely. Only
# RUNTIME moves; JUPYTER_DATA_DIR is left alone so the env's python3 kernelspec is still discovered.
export JUPYTER_RUNTIME_DIR=/tmp/$USER/jupyter-runtime-${SLURM_JOB_ID:-manual}
mkdir -p "$JUPYTER_RUNTIME_DIR"

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$SWEEP_DIR"          # the notebook derives REPO from cwd ("../..") and writes results here

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Preflight: fail in seconds, not after minutes of model loading. The baseline run died here in 29 s
# on a torch+cu130 build against a 12.x driver -- exactly the class of failure this catches. Runs
# BEFORE staging, and while still online, so auth_check can reach the hub.
"$PY" - <<'PREFLIGHT'
import torch
from huggingface_hub import auth_check
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
name = torch.cuda.get_device_name(0)
print(f"GPU: {name}  {gb:.0f} GB total")
# Belt and braces behind --constraint=a6000. A resume is only valid on the source run's architecture
# (see the header): bf16 kernels differ across Ampere/Ada and perturb f at round-off. 695271 ran on
# "NVIDIA RTX A6000"; failing here costs seconds, failing in the notebook costs ~25 min of setup.
assert "A6000" in name, f"resume requires the source run's card (A6000), got {name!r}"
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
auth_check("black-forest-labs/FLUX.1-dev")
print("FLUX.1-dev access: OK")
PREFLIGHT

# ---------------------------------------------------------------------------------------------
# Stage the weights to node-local disk. THIS IS THE FIX FOR 754633 -- see the header.
# ---------------------------------------------------------------------------------------------
# cp -L dereferences: the HF cache keeps blobs behind symlinks into ../../blobs, and the staged tree
# must stand alone. Parallel over files because a single NFS stream is latency-bound (28.9 MB/s
# measured); several concurrent streams overlap that latency, which is why --cpus-per-task went 4 -> 6.
SNAP=$(ls -d "$HF_HOME"/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
    echo "FATAL: no FLUX snapshot under $HF_HOME/hub" >&2
    exit 1
fi

# THE CACHE IS KEYED BY SNAPSHOT SHA AND PERSISTS ACROSS JOBS -- do not "clean it up".
# Job 755539 measured the real cost of this copy: 257 MINUTES for 32 G, i.e. ~2 MB/s effective from
# the compute node (the login node had measured 28.9 MB/s sequential -- the compute-side mount is an
# order of magnitude worse). Duplicating the same tree local->local afterwards took under a minute
# (>500 MB/s), so the local disk is not the problem and re-staging per job would burn 4.3 h every
# time. /tmp on n-602 is the node's real ext4 root (877 G, 5% used), not a per-job tmpfs namespace,
# so it survives between jobs and the second run pays nothing.
SHA=$(basename "${SNAP%/}")
STAGE=/tmp/$USER/flux-snap-$(echo "$SHA" | cut -c1-8)
N_SRC=$(cd "$SNAP" && find -L . -type f | wc -l)

# Age-based sweep only, and only of OTHER snapshots: never delete the cache this job is about to use.
# There is deliberately no EXIT trap removing $STAGE -- that is what would throw the 4.3 h away.
find /tmp/$USER -maxdepth 1 -name 'flux-snap-*' ! -name "$(basename "$STAGE")" -mtime +7 \
    -exec rm -rf {} + 2>/dev/null || true
find /tmp/$USER -maxdepth 1 -name 'flux-[0-9]*' -mtime +1 -exec rm -rf {} + 2>/dev/null || true
find /tmp/$USER -maxdepth 1 -name 'jupyter-runtime-*' -mtime +1 -exec rm -rf {} + 2>/dev/null || true

if [ -f "$STAGE/model_index.json" ] && [ "$(find -L "$STAGE" -type f | wc -l)" -eq "$N_SRC" ]; then
    echo "=== node-local cache HIT: $STAGE ($(du -sh "$STAGE" | cut -f1), $N_SRC files) -- no staging ==="
else
    # Stage into .partial and rename only on success: a preempted job must never leave a half-copied
    # tree that the next job's existence check would trust.
    echo "=== node-local cache MISS: staging $(du -shL "$SNAP" | cut -f1) from NFS (expect HOURS at" \
         "~2 MB/s; this is paid once per node) ==="
    t_stage=$(date +%s)
    rm -rf "$STAGE.partial"
    mkdir -p "$STAGE.partial"
    ( cd "$SNAP" && find -L . -type d -exec mkdir -p "$STAGE.partial/{}" \; )
    ( cd "$SNAP" && find -L . -type f -print0 | xargs -0 -P 6 -I{} cp -L {} "$STAGE.partial/{}" )
    mv "$STAGE.partial" "$STAGE"
    echo "=== staged in $(( ($(date +%s) - t_stage) / 60 )) min: $(du -sh "$STAGE" | cut -f1) ==="
fi
chmod -R u+rwX,go-rwx /tmp/$USER 2>/dev/null || true

# from_pretrained accepts a local directory containing model_index.json, so no HF cache layout and no
# HF_HUB_OFFLINE juggling is needed -- the notebook keeps MODEL_ID canonical for the record and only
# LOADS from this path.
if [ ! -f "$STAGE/model_index.json" ]; then
    echo "FATAL: staged tree has no model_index.json -- copy incomplete" >&2
    exit 1
fi
export FLUX_LOCAL_PATH="$STAGE"

# ---------------------------------------------------------------------------------------------
# Execute. NO JUPYTER KERNEL -- see below.
# ---------------------------------------------------------------------------------------------
OUT="run_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT (run_notebook.py, kernel-free, streaming) ==="
# Three jobs died here BEFORE cell 1 with "Kernel didn't respond": 756120 at 60 s, 756173 at 600 s,
# and 756235 three times at 600 s under a retry loop. The kernel process always came up -- its
# IPKernelApp banner is in every log -- but the client<->kernel handshake never completed. Raising
# the timeout did not fix it, and neither did moving JUPYTER_RUNTIME_DIR (where the connection file
# lives) onto node-local disk. Both were hypotheses about a handshake not observable from outside.
#
# run_notebook.py execs the cells in a plain Python process instead: no kernel, no ZMQ, no connection
# file, so the whole failure class is gone rather than worked around. Not a guess -- probe_setup.py
# used this same technique on this node and ran the notebook's setup cells end to end (model load,
# references, score bank) on a day papermill could not start a kernel at all. It keeps what papermill
# was adopted for: live streaming to this log, and the notebook written after EVERY cell so a kill
# leaves a partial record instead of nothing. What it drops is inline images -- and every artifact
# that matters (checkpoints .pt, the grid .png) is written to disk by the cells themselves.
set +e
"$PY" run_notebook.py "$NB" "$OUT"
rc=$?
set -e

echo "=== done $(date)  (run_notebook exit $rc) ==="
ls -l "$SWEEP_DIR"/*2_1*.pt "$SWEEP_DIR"/refs_*.pt "$SWEEP_DIR"/strong_tilt_grid_2_1.png 2>/dev/null || true
exit $rc
