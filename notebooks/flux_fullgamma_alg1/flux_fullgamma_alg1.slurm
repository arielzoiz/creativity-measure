#!/bin/sh
#SBATCH --job-name=flux-fg1
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=a6000
#SBATCH --nodelist=n-602
#SBATCH --time=480
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Run fg1: Algorithm 1 (adaptive-tempering SMC + pCN) on FLUX.1-dev, FROM SCRATCH, with the
# integration window widened to the FULL gamma range.
#
# WHAT CHANGES vs run 2 / 2_1 / 2_2, and nothing else changes:
#   gamma_lo:  max(1/S^2, 1/sigma_max^2) = 0.9713   ->   2^-10 = 0.0009766
#   gamma_hi:  min(2^10, 1/sigma_min^2)  = 1024     ->   unchanged
# The old lower cut came from demo_properties.ipynb Conclusion 1d: below the crossover at the data
# scale the channel is noise-dominated, the marginal score is linear, and the integrand degenerates to
# the plain squared Euclidean distance (Concl. 1a). Run 2's own cell 4 named that cut as the NEXT
# variable to vary and deliberately held it fixed so N_GAMMA moved alone. This run varies it.
#
# 2^-10 is a choice, not a limit -- the denoiser stays valid down to 1/sigma_max^2 = 1.56e-4 = 2^-12.7.
# 2^-10 keeps the grid symmetric about gamma=1 and sits ~2.7 octaves inside the validity edge, so no
# score is ever evaluated at an extrapolated sigma. The notebook asserts this.
#
# OTHER PARAMS ARE 2_2's: R=64, N_GAMMA=30, NUM_EPS=3, ref seed 7, Brownian seed 123,
# MAX_DENOISER_ROWS=24, m=3, ess_target=0.80, SEED=101, MAX_SWEEPS=40, prompt "A dog", guidance 1.5,
# 512 px, 8 Heun steps, sigma in [0.002, 80]. N is 8 (requested) rather than 16.
#
# FROM SCRATCH, NOT A RESUME. There is no SOURCE_CKPT, no ResumePCNKernel, no reward-reproducibility
# guard and no lam_base: the widened window makes f a different function, so no prior checkpoint is
# resumable onto it even in principle. `adapt_rate` therefore goes back to the library default of 0.1
# (2_2's 0.25 was a workaround for a source checkpoint that had not persisted `s`; this run persists it).
#
# lambda_s IS RE-MEASURED, and must be. Concl. 1b: c_window = sqrt(int dgamma/g^2) rescales the whole
# distance, so run 2's cached lam_s = 92.195 does not describe this f. The notebook re-measures it on
# the SAME 32 held-out draws run 2 used -- it re-seeds the generator to 7 and discards the first R rows
# to land exactly where run 2's stream was after the reference draw -- so E_p[f] and std_p(f) differ
# from run 2 for exactly one reason. Without the discard the probe would BE the references and lam_s
# would come out enormous; the notebook asserts min dist(probe, refs) > 0.
#
# ================================ THREE DECISIONS, WITH REASONS ================================
#
# (1) --nodelist=n-602. The FLUX snapshot is 32 G on an NFSv3 netapp mount and safetensors are
# MEMORY-MAPPED, so `.to(cuda)` reads it as random 32 KB NFS reads -- that is what hung job 754633 for
# 6 h. The fix is a node-local staged copy, and job 755539 MEASURED the cost of making one: 257 minutes
# for 32 G (~2 MB/s from the compute node; the login node's 28.9 MB/s is not a valid proxy). Every
# large job in this project's history ran on n-602 and the staged tree there is SHA-keyed and
# deliberately never cleaned up; n-601 was probed on 2026-08-23 and its /tmp was cold. In an 8 h job a
# staging miss would eat over half the budget, so pinning the warm node is worth queueing for.
# n-602 has 8 A6000s. If this ever queues for hours, drop the nodelist and accept a truncated run
# rather than switching cards -- see (2).
#
# (2) --constraint=a6000 even though this is not a resume. FLUX runs bf16 (8-bit mantissa) and Ampere
# vs Ada select different tensor-core kernels and reduction orders; job 697271 measured the resulting
# drift in f at 11% of std_p(f). This run is self-consistent on any card, but its whole point is a
# controlled comparison against run 2, which ran on an A6000 -- and the persisted reference latents
# were generated there. Changing cards would confound the window with round-off. The notebook warns
# (does not abort) if the card differs, since the run itself would still be valid.
#
# (3) THE HUB IS NEVER CONTACTED. 2_2 called huggingface_hub.auth_check() in the preflight and in
# cell 1. Job 776480 died there on a 401 GatedRepoError: the token in $HOME/.hf_token is stale, and
# 776753 only got through by inheriting a valid HF_TOKEN from the submitting shell -- exactly the
# ambient-credential anti-pattern failure 5 of the 2_2 header warns against, and one that makes a job
# unreproducible from a script or an agent. Nothing here needs the hub: the complete 27-file snapshot
# is already in $HF_HOME/hub, this script stages it to local disk, and `from_pretrained` on a directory
# containing model_index.json never resolves a repo id. So the auth check below is ADVISORY (it reports
# and continues) and HF_HUB_OFFLINE=1 is exported before the notebook runs. The real guard is the
# staged-tree check, which fails in seconds with a message that names the problem.
#   To restore the online check: put a VALID token in $HOME/.hf_token at 0600 (never under $WORK,
#   which is group-readable) and drop the HF_HUB_OFFLINE export.
#
# COST MODEL (0.251 s/row on an A6000, validated to 0.5-2.5% by jobs 695266/695271/700517):
#   model load (warm /tmp)      ~25 min
#   reference score bank         ~23 min   (N_GAMMA-1)*NUM_EPS*R = 5568 rows
#   lambda_s probe               ~14 min   32*(15+87) = 3264 rows
#   gamma-profile diagnostic      ~3 min   29*3*9 = 783 rows
#   -> ~1 h fixed, 1 h reserved for decode/render, leaving ~6 h
#   sweep of N=8: 8*(15+87)*0.251 = 205 s = 3.4 min  ->  ~105 sweeps  ->  5-8 levels
# beta=1 is plausible but not promised. Every sweep is checkpointed and `s` IS persisted, so a further
# leg can resume from partial_N8_m3_seed101_fg1.pt.
#
# `killable` is preemptible and caps at 1 day. Preemption costs ~3.4 min plus requeue, not the run.
#
# OBSERVABILITY. nbconvert buffers all cell output and writes it once at the end, so a stall is
# undiagnosable by construction -- 6 silent hours is what job 754633 produced. Papermill fixed that but
# needed a Jupyter kernel, and the client<->kernel handshake failed on this cluster in jobs 756120 /
# 756173 / 756235 (60 s, 600 s, and 600 s three times under a retry loop) for reasons not observable
# from outside. run_notebook.py execs the cells in a plain Python process: no kernel, no ZMQ, no
# connection file, so that failure class is gone rather than worked around. It keeps live streaming to
# this log and writes the output notebook after EVERY cell. What it drops is inline images -- so every
# figure in this notebook is ALSO savefig'd to disk.
#
# No `source ~/.bashrc`: that file is empty here and conda/HF_* live in ~/.cshrc (tcsh), so this script
# sets everything explicitly and works from a fresh shell or a requeue.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
RUN_DIR=$REPO/notebooks/flux_fullgamma_alg1
NB=flux_fullgamma_alg1.ipynb

# $WORK is group-readable, so the HF *credential* must not live there; the 32 G of weights may.
export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1                          # stdout is a file here, so Python would block-buffer

# Jupyter's runtime dir defaults to an automounted NFS home. Nothing here starts a kernel any more,
# but the env var is set anyway so any stray jupyter machinery stays off NFS.
export JUPYTER_RUNTIME_DIR=/tmp/$USER/jupyter-runtime-${SLURM_JOB_ID:-manual}
mkdir -p "$JUPYTER_RUNTIME_DIR"

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$RUN_DIR"            # the notebook derives REPO from cwd ("../..") and writes results here

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Preflight: fail in seconds, not after minutes of model loading. Deliberately ordered BEFORE staging,
# so a broken environment burns 60 s of an A6000 rather than hours. The auth probe is advisory (see
# decision 3 in the header) -- it reports the credential state for the record and never exits nonzero.
"$PY" - <<'PREFLIGHT'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
name = torch.cuda.get_device_name(0)
print(f"GPU: {name}  {gb:.0f} GB total")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
if "A6000" not in name:
    print(f"WARNING: expected an A6000 (run 2's card, and the one the persisted reference latents "
          f"were generated on); got {name!r}. The run is self-consistent but bf16 kernels differ "
          f"across architectures, so the comparison to run 2 carries ~11% of std_p(f) of drift.")
try:
    from huggingface_hub import auth_check
    auth_check("black-forest-labs/FLUX.1-dev")
    print("FLUX.1-dev hub access: OK (unused -- this run loads offline from the staged snapshot)")
except Exception as exc:
    print(f"hub access: UNAVAILABLE ({type(exc).__name__}). Not fatal: this run loads offline from "
          f"the staged local snapshot and never resolves a repo id.")
PREFLIGHT

# ---------------------------------------------------------------------------------------------
# Stage the weights to node-local disk. THIS IS THE FIX FOR 754633 -- see the header.
# ---------------------------------------------------------------------------------------------
# cp -L dereferences: the HF cache keeps blobs behind symlinks into ../../blobs, and the staged tree
# must stand alone. Parallel over files because a single NFS stream is latency-bound; several
# concurrent streams overlap that latency, which is why --cpus-per-task is 6.
SNAP=$(ls -d "$HF_HOME"/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
    echo "FATAL: no FLUX snapshot under $HF_HOME/hub" >&2
    exit 1
fi

# THE CACHE IS KEYED BY SNAPSHOT SHA AND PERSISTS ACROSS JOBS -- do not "clean it up".
SHA=$(basename "${SNAP%/}")
STAGE=/tmp/$USER/flux-snap-$(echo "$SHA" | cut -c1-8)
N_SRC=$(cd "$SNAP" && find -L . -type f | wc -l)

# Age-based sweep only, and only of OTHER snapshots: never delete the cache this job is about to use.
# There is deliberately no EXIT trap removing $STAGE -- that is what would throw the 4.3 h away.
find /tmp/$USER -maxdepth 1 -name 'flux-snap-*' ! -name "$(basename "$STAGE")" -mtime +7 \
    -exec rm -rf {} + 2>/dev/null || true
find /tmp/$USER -maxdepth 1 -name 'jupyter-runtime-*' -mtime +1 -exec rm -rf {} + 2>/dev/null || true

if [ -f "$STAGE/model_index.json" ] && [ "$(find -L "$STAGE" -type f | wc -l)" -eq "$N_SRC" ]; then
    echo "=== node-local cache HIT: $STAGE ($(du -sh "$STAGE" | cut -f1), $N_SRC files) -- no staging ==="
else
    # Stage into .partial and rename only on success: a preempted job must never leave a half-copied
    # tree that the next job's existence check would trust.
    echo "=== node-local cache MISS on $(hostname): staging $(du -shL "$SNAP" | cut -f1) from NFS." \
         "MEASURED cost 257 min at ~2 MB/s (job 755539). In an 8 h job this leaves ~2 h of sweeping," \
         "so expect a badly truncated ladder -- visible here, not silent. ==="
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

# Now that the weights are local and complete, take the hub out of the picture entirely. With this set,
# a stale credential cannot turn into a 401 mid-load (failure 776480), and any file the snapshot is
# missing surfaces as a clear offline error instead of a silent gated download attempt.
export HF_HUB_OFFLINE=1

# ---------------------------------------------------------------------------------------------
# Execute. NO JUPYTER KERNEL -- see OBSERVABILITY in the header.
# ---------------------------------------------------------------------------------------------
OUT="run_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT (run_notebook.py, kernel-free, streaming) ==="
set +e
"$PY" run_notebook.py "$NB" "$OUT"
rc=$?
set -e

echo "=== done $(date)  (run_notebook exit $rc) ==="
ls -l "$RUN_DIR"/*_fg1*.pt "$RUN_DIR"/*_fg1*.png 2>/dev/null || true
exit $rc
