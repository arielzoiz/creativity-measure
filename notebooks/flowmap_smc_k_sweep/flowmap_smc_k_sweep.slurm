#!/bin/sh
#SBATCH --job-name=k-sweep
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=660
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gpus=1
#
# THE K SWEEP AT FIXED LAMBDA -- Algorithm 3.
#
#   sbatch --time=480  --export=ALL,CONTROL=1,K_LOOK=1,M_LIST=0,M_PARTICLES=32,RUN_TAG=ctrl  flowmap_smc_k_sweep.slurm
#   sbatch --time=540  --export=ALL,K_LOOK=4,RUN_TAG=k04   flowmap_smc_k_sweep.slurm
#   sbatch --time=660  --export=ALL,K_LOOK=8,RUN_TAG=k08   flowmap_smc_k_sweep.slurm
#   sbatch --time=960  --export=ALL,K_LOOK=16,RUN_TAG=k16  flowmap_smc_k_sweep.slurm
#   sbatch --time=1440 --export=ALL,K_LOOK=32,RUN_TAG=k32  flowmap_smc_k_sweep.slurm
#
# CONTROL=1 is the lambda = 0 reference: guid_window becomes (1.0, 1.0) so only the mandatory terminal
# update fires and the job costs 16*M projection rows instead of 15*M*K lookahead rows. It is set HERE
# rather than passed as GUID_WINDOW through --export, because sbatch splits --export on commas and
# `GUID_WINDOW=1.0,1.0` would silently become `GUID_WINDOW=1.0` plus a stray variable named `1.0`.
#
# THE QUESTION. At one fixed medium lambda, does raising the lookahead depth K change the within-run
# E_q[f] curve -- specifically BEFORE the cloud degenerates? Every tilted Algorithm 3 run to date used
# K = 4 (the module default); K = 8 has been MEASURED (diag job 782816) but never sampled with.
#
# WHY NOT A LAMBDA SWEEP. At M = 8 a lambda sweep measures degeneracy, not the estimator: a3_2
# collapses to a single root ancestor by step 2 of 16 at m = 2. Degeneracy at strong tilt is accepted
# as inherent (systematic resampling destroys lineages; nothing regenerates them). m = 1 is too weak --
# the mechanism fires ONCE in 16 steps and the deficit is 0.41 std_p(f), below the M = 8 estimator SE.
# m = 1.25 (lambda = 175.1) is the conservative choice in the gap.
#
# THE K = 4 JOB IS A LIVE SCOUT. Base-step noise (`gen_base`) is independent of lookahead noise
# (`gen_look`), so at fixed seed step 0 is identical across all K and the early steps are directly
# comparable across cells. K = 4 reaches step 5 in ~48 min. Watch its progress log:
#   * collapses to 1 lineage before step 5      -> lambda too high, drop to 150, re-run EVERY cell
#   * <=1 resample, or give-back < 1 std_p(f)   -> lambda too low, raise to 210, re-run EVERY cell
#   * otherwise                                 -> let the sweep run
# Cancelling the other cells at that point costs ~48 min x 3 GPUs instead of 15.7 GPU-h.
#
# ---------------------------------------------------------------------------------------------------
# --constraint=l40s IS LOAD-BEARING -- NEVER MAKE IT A DISJUNCTION.
# ---------------------------------------------------------------------------------------------------
# The older flowmap_smc.slurm uses `--constraint=a6000|l40s`. These cells run CONCURRENTLY and land on
# whatever frees up first, so a disjunction would scatter them across hardware. A GPU change once
# shifted f by 16% of std_p(f) (job 697271) -- the SAME ORDER as the effects being measured. The
# notebook records `gpu_name` in every result and the diagnostics cell warns if the set has more than
# one member; that warning invalidates the comparison, it does not soften it.
#
# l40s over a6000 also because K = 32 (19.8 h at 0.206 s/row) fits the partition's 24 h limit there and
# does NOT at the a6000 rate of 0.251 (24.2 h).
#
# ---------------------------------------------------------------------------------------------------
# COST. One reward call on B latents = (N_GAMMA-1)*NUM_EPS*B = 87B denoiser rows at 0.206 s/row on an
# L40S = 17.9 s/latent. Per run: 15*M*K lookahead + M terminal + 15*M projection.
#
#   K=4:  616 latents = 3.1 h     K=16: 2056 latents = 10.2 h
#   K=8: 1096 latents = 5.5 h     K=32: 3976 latents = 19.8 h
#
# --time carries ~4-5 h of headroom above that, because a COLD node pays the staging cost below. The
# notebook's DEADLINE derives from SLURM_JOB_END_TIME, so a slow start degrades GRACEFULLY: it runs
# fewer steps and the partial file still holds the pre-collapse window, which is the deliverable.
# K = 32 is the one cell that cannot absorb a cold stage (19.8 + 4.3 > 24) -- expect it to stop early,
# which is acceptable for the same reason.
#
# --mem=24000 per CLAUDE.md (measured peak RSS 10.2 GB for the flow-map jobs, job 782816). Do NOT
# raise it: memory, not GPUs, is what blocks these jobs. n-602 currently has 4 IDLE A6000s that nobody
# can use because all 1 TB of its host RAM is reserved.
#
# PREEMPTION. `killable` is preemptible and Algorithm 3 checkpoints but cannot resume. That is
# tolerable HERE and only here: `on_step` writes partial_*.pt every step with the full per-step
# history, and the deliverable is the EARLY part of the curve. A job killed at step 12 has already
# delivered what it was for. Only the killed cell is affected; the others are separate jobs.
#
# NOT DONE, ON PURPOSE: the score_bank is still rebuilt every job (~19-27 min, ~2 h across the sweep).
# Persisting it touches the frozen reward path, and doing that immediately before a 40 GPU-h
# submission is the wrong order. See the plan's "Also fixed by default".
#
# ---------------------------------------------------------------------------------------------------
# The four decisions below are inherited verbatim from flowmap_smc_diag.slurm, each a failure a real
# job hit. Do not simplify them away.
#
# (1) THE HUB IS NEVER CONTACTED. Job 781449 died in the preflight on a 401 GatedRepoError
#     ($HOME/.hf_token is stale; the only credential that ever worked was an HF_TOKEN exported in an
#     interactive shell, which a submitted script does not inherit). Job 781469 then died on
#     OfflineModeIsEnabled because the dodge for the first CAUSED the second. Both auth probes are
#     advisory now, staging is the real guard, and HF_HUB_OFFLINE=1 is exported before the notebook.
# (2) THE SNAPSHOT IS STAGED TO NODE-LOCAL DISK. The FLUX snapshot is 32 G on an NFSv3 netapp mount and
#     safetensors are mmap'd, so a cold node reads it as random 32 KB NFS reads: job 755539 MEASURED
#     257 min at ~2 MB/s. The staged tree is keyed on the snapshot SHA and reused across jobs.
# (3) run_notebook.py, NOT nbconvert. nbconvert buffers all cell output into the OUTPUT notebook, which
#     is only written when the job ends -- so a multi-hour run shows nothing until it is over.
#     run_notebook.py streams. The notebook additionally mirrors milestones to progress_*.log.
# (4) THE CPU REGRESSION RUNS FIRST. It costs seconds and catches an import- or API-level break before
#     the job spends an hour loading 32 G of weights.
# ---------------------------------------------------------------------------------------------------

set -e
WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
RUN_DIR=$REPO/notebooks/flowmap_smc_k_sweep
NB=flowmap_smc_k_sweep.ipynb

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1                          # stdout is a file here, so Python would block-buffer
export JUPYTER_RUNTIME_DIR=/tmp/$USER/jupyter-runtime-${SLURM_JOB_ID:-manual}
mkdir -p "$JUPYTER_RUNTIME_DIR" "$WORK/logs" "$MPLCONFIGDIR"

export STOP_AFTER="${STOP_AFTER:-run}"
export RUN_TAG="${RUN_TAG:-k_sweep}"
export K_LOOK="${K_LOOK:-4}"
export M_PARTICLES="${M_PARTICLES:-8}"
export M_LIST="${M_LIST:-1.25}"
# Set here, never through --export: sbatch splits --export on commas (see the header).
if [ "${CONTROL:-0}" = "1" ]; then
    export GUID_WINDOW="1.0,1.0"
else
    export GUID_WINDOW="0.0,1.0"
fi

cd "$RUN_DIR"          # the notebook derives REPO from cwd ("../..") and writes results here
echo "=== $(date) on $(hostname)  K=$K_LOOK  M=$M_PARTICLES  M_LIST=$M_LIST" \
     " GUID_WINDOW=$GUID_WINDOW  TAG=$RUN_TAG ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

"$PY" - <<'PREFLIGHT'
import os
import torch, peft
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
name = torch.cuda.get_device_name(0)
print(f"GPU: {name}  {gb:.0f} GB total   peft {peft.__version__}")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
# THE COMPARISON IS ACROSS CONCURRENT JOBS, so the GPU model must be identical in all of them.
# A GPU change shifted f by 16% of std_p(f) once (job 697271) -- the same order as the effect here.
assert "L40S" in name.upper(), (
    f"this sweep is pinned to L40S and got {name!r}; a mixed-hardware K comparison is not valid")
try:
    from huggingface_hub import auth_check
    auth_check("black-forest-labs/FLUX.1-dev")
    print("FLUX.1-dev hub access: OK (unused -- this run loads offline from the staged snapshot)")
except Exception as exc:
    print(f"hub access: UNAVAILABLE ({type(exc).__name__}). Not fatal: this run loads offline from "
          f"the staged local snapshot and never resolves a repo id.")
PREFLIGHT

echo "=== notebook import check ==="
"$PY" check_notebook_imports.py "$NB"

SNAP=$(ls -d "$HF_HOME"/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
    echo "FATAL: no FLUX snapshot under $HF_HOME/hub" >&2
    exit 1
fi
SHA=$(basename "${SNAP%/}")
STAGE=/tmp/$USER/flux-snap-$(echo "$SHA" | cut -c1-8)
N_SRC=$(cd "$SNAP" && find -L . -type f | wc -l)

# ---------------------------------------------------------------------------------------------------
# STAGING IS SERIALIZED PER NODE. This lock is not optional and it is not defensive decoration.
#
# The whole HIT/MISS block below is shared mutable state: every job on this node uses the SAME $STAGE
# path, and the MISS branch opens with `rm -rf "$STAGE.partial"`. Submitting the sweep CONCURRENTLY --
# which is the point of it -- means two jobs can land on one node, and the second one's `rm -rf` then
# deletes the first one's in-progress 32 G copy while it is still being written. The first job goes on
# to `mv` a gutted tree into $STAGE, and both jobs load a model that is missing shards.
#
# This bit any job that shared a node: 2026-09-08, jobs 869622 (K=16) and 869623 (K=32) landed on
# n-804 and n-805 alongside 869620 and 869619, which were 14 and 17 min into staging. Both newcomers
# were cancelled by hand within ~60 s -- before either reached the `rm -rf` -- and resubmitted against
# this guarded version. The older scripts never hit it only because their jobs were submitted one at
# a time.
#
# flock makes the loser WAIT rather than race. It then re-evaluates the HIT condition and finds a
# complete tree, so it costs one staging on the node instead of two -- strictly better than the
# uncontended case. Holding a lock for the duration of a 4 h copy is intended.
# ---------------------------------------------------------------------------------------------------
exec 9>"/tmp/$USER/flux-stage-$(echo "$SHA" | cut -c1-8).lock"
if ! flock -w 21600 9; then
    echo "FATAL: waited 6 h for the node staging lock and never got it" >&2
    exit 1
fi
echo "=== holding node staging lock on $(hostname) ==="
find /tmp/$USER -maxdepth 1 -name 'flux-snap-*' ! -name "$(basename "$STAGE")" -mtime +7 \
    -exec rm -rf {} + 2>/dev/null || true
find /tmp/$USER -maxdepth 1 -name 'jupyter-runtime-*' -mtime +1 -exec rm -rf {} + 2>/dev/null || true
if [ -f "$STAGE/model_index.json" ] && [ "$(find -L "$STAGE" -type f | wc -l)" -eq "$N_SRC" ]; then
    echo "=== node-local cache HIT: $STAGE ($(du -sh "$STAGE" | cut -f1), $N_SRC files) -- no staging ==="
else
    echo "=== node-local cache MISS on $(hostname): staging $(du -shL "$SNAP" | cut -f1) from NFS." \
         "MEASURED worst case 257 min (job 755539). --time carries headroom for this and the" \
         "notebook's DEADLINE degrades gracefully, but a MISS means fewer steps complete. ==="
    t_stage=$(date +%s)
    rm -rf "$STAGE.partial"
    mkdir -p "$STAGE.partial"
    ( cd "$SNAP" && find -L . -type d -exec mkdir -p "$STAGE.partial/{}" \; )
    ( cd "$SNAP" && find -L . -type f -print0 | xargs -0 -P 6 -I{} cp -L {} "$STAGE.partial/{}" )
    mv "$STAGE.partial" "$STAGE"
    echo "=== staged in $(( ($(date +%s) - t_stage) / 60 )) min: $(du -sh "$STAGE" | cut -f1) ==="
fi
chmod -R u+rwX,go-rwx /tmp/$USER 2>/dev/null || true
# COUNT THE FILES, do not just look for model_index.json. That file is ~1 KB and copies first, so it
# is present in exactly the failure this guard exists for: a node whose /tmp runs out of space part
# way through the 32 G copy. Jobs 869620 (n-804) and 870099 (n-801) both died on
# `cp: No space left on device`; `set -e` caught the xargs failure there, but if it ever does not,
# the next line must not wave a truncated tree through into `from_pretrained`.
# The HIT branch above already counts files; this makes the MISS branch just as strict.
N_STAGED=$(find -L "$STAGE" -type f 2>/dev/null | wc -l)
if [ ! -f "$STAGE/model_index.json" ] || [ "$N_STAGED" -ne "$N_SRC" ]; then
    echo "FATAL: staged tree is incomplete ($N_STAGED of $N_SRC files) -- likely /tmp full on $(hostname)." >&2
    echo "       Removing it so the next job re-stages rather than inheriting a corrupt cache." >&2
    rm -rf "$STAGE" "$STAGE.partial"
    exit 1
fi
exec 9>&-               # release the node staging lock: the tree is complete and read-only from here
echo "=== released node staging lock ==="

export FLUX_LOCAL_PATH="$STAGE"
export HF_HUB_OFFLINE=1        # weights are local and complete; a stale token cannot 401 mid-load

echo "=== regression (CPU) ==="
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest \
    "$REPO/tests/test_flowmap_smc.py" "$REPO/tests/test_flux_flowmap.py" -q

OUT="run_${RUN_TAG}_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT (run_notebook.py, kernel-free, streaming) ==="
set +e
"$PY" run_notebook.py "$NB" "$OUT"
rc=$?
set -e
echo "=== done $(date)  (run_notebook exit $rc) ==="
ls -l "$RUN_DIR"/*"$RUN_TAG"* 2>/dev/null || true
exit $rc
