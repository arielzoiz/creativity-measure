#!/bin/sh
#SBATCH --job-name=a3-defer
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
# THE DEFERRED-RESAMPLING ARM AT FIXED LAMBDA -- Algorithm 3.
#
#   sbatch --time=540 --export=ALL,RESAMPLE_T0=0.25,K_LOOK=4,RUN_TAG=defer25  flowmap_smc_defer.slurm
#   sbatch --time=540 --export=ALL,RESAMPLE_T0=1.01,K_LOOK=4,RUN_TAG=isonly    flowmap_smc_defer.slurm   # the pure-IS limit
#
# CONTROL=1 is the lambda = 0 reference: guid_window becomes (1.0, 1.0) so only the mandatory terminal
# update fires and the job costs 16*M projection rows instead of 15*M*K lookahead rows. It is set HERE
# rather than passed as GUID_WINDOW through --export, because sbatch splits --export on commas and
# `GUID_WINDOW=1.0,1.0` would silently become `GUID_WINDOW=1.0` plus a stray variable named `1.0`.
#
# THE QUESTION. Does aggregating the K lookahead candidates by their plain MEAN, instead of the soft
# value `log mean_k exp(lam r_k)`, produce a better twist? This is the tau -> 0 end of the
# interpolation whose tau -> inf end (`flowmap_smc_max`) was measured and came out null.
#
# WHY IT IS NOT THE MAX EXPERIMENT AGAIN. The max RESCALES the soft ordering (spearman 0.93-0.97) and
# changed nothing. The mean DELETES a term: V/lam = mean_k r + lam*Var_k(r)/2. Measured offline on the
# recorded r_k of all 11 existing cells (notebooks/flowmap_smc_replay/a1_mean_aggregation.py):
#   * the mean's estimator error falls like k^-1.0 (p = 0.90-1.21); the soft value manages k^-0.6
#     (p = 0.32-0.82) and the max k^-0.1. K SATURATING IS A PROPERTY OF THE AGGREGATION, not of the
#     lookahead -- which retires the single firmest conclusion of the K sweep.
#   * sd(U)/sd(V) is lower for the mean in EVERY cell (mean < soft < max, monotone).
#   * the deleted term correlates NEGATIVELY with f(t=1) in 7 of 11 cells (to -0.59) while carrying
#     8-30% of sd(V): the soft value spends that share backing uncertain futures, which finish low.
# It also removes the Jensen exposure entirely, which is what matters at strong tilt: the soft value
# carries a K-sample bias ~ (e^{a^2}-1)/2K with a = lam*sd_k, and a(t->0) = m exactly, so at m = 3 it
# degenerates to a hard max at K = 4. See notebooks/flowmap_smc_replay/A0_RESULTS.md.
#
# TEMPER THE EXPECTATION. spearman(mean, soft) = 0.92-0.98 -- the same "barely reorders" signature the
# max had before it produced nothing. A large move in terminal E_q[f] at m = 1.25 would be a surprise.
# Judge this arm on uniq/M, on sd(U)/sd(V), and on whether a K sweep through the mean finally shows
# the 1/sqrt(K) that the soft value never did.
#
# THE BASELINE IS ALREADY ON DISK -- no paired re-run. This is byte-for-byte the k-sweep's K=4 cell
# except for the aggregation, so `notebooks/flowmap_smc_k_sweep/result_N8_K4_m1.25_seed101_k04.pt`
# (terminal 1.0000, +0.81 std_p(f), uniq/M 0.125, collapse at step 3, sd(U)/sd(V) 0.534) is the soft
# arm, and `result_N8_K1_m0_seed101_ctrl2.pt` is the lambda = 0 control (one control serves both arms:
# at lam = 0 every v_k is 0 and the aggregations are provably bit-identical).
#
# A FORK OF THE K-SWEEP NOTEBOOK, ON PURPOSE. The notebook is read at job START time, so editing the
# k-sweep's copy would change what an already-queued job runs. This directory is a copy; refs
# (md5 711880aade334889a6b730c00f5ed48f) and lam_s are the same files.
#
# WATCH THE PROGRESS LOG. Base-step noise (`gen_base`) is independent of lookahead noise (`gen_look`),
# and the aggregation only enters through resampling, so this arm carries IDENTICAL particles to k04
# until the first step where their resampling decisions differ. `f_cand` must match k04's exactly
# before that step; if it does not, the two runs are not comparable and something else changed.
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
RUN_DIR=$REPO/notebooks/flowmap_smc_replay
NB=flowmap_smc_defer.ipynb

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
export AGG="${AGG:-soft}"          # soft = flowmap_smc_sample, mean = flowmap_smc_mean_sample
export RESAMPLE_T0="${RESAMPLE_T0:-0.25}"   # selection is OFF for every step landing below this t
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
     " GUID_WINDOW=$GUID_WINDOW  AGG=$AGG  TAG=$RUN_TAG ==="
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
    "$REPO/tests/test_flowmap_smc.py" "$REPO/tests/test_flowmap_smc_mean.py" "$REPO/tests/test_flowmap_smc_defer.py" \
    "$REPO/tests/test_flux_flowmap.py" -q

OUT="run_${RUN_TAG}_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT (run_notebook.py, kernel-free, streaming) ==="
set +e
"$PY" run_notebook.py "$NB" "$OUT"
rc=$?
set -e
echo "=== done $(date)  (run_notebook exit $rc) ==="
ls -l "$RUN_DIR"/*"$RUN_TAG"* 2>/dev/null || true
exit $rc
