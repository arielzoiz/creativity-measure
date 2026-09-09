#!/bin/sh
#SBATCH --job-name=maxagg
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
# HARD-MAX LOOKAHEAD vs THE PLUG-IN ESTIMATOR, AT FIXED LAMBDA -- Algorithm 3.
#
#   PAIRED SEED SWEEP at K=8 (the only K worth a seed: K=4 collapses to one lineage at step 3 under
#   BOTH aggregations, so it has two informative steps per run however you seed it):
#
#     for S in 102 103; do
#       sbatch --time=660 --nodelist=n-803,n-805 --export=ALL,K_LOOK=8,SEED=$S,AGG=max,RUN_TAG=s${S}max  flowmap_smc_max.slurm
#       sbatch --time=660 --nodelist=n-803,n-805 --export=ALL,K_LOOK=8,SEED=$S,AGG=soft,RUN_TAG=s${S}soft flowmap_smc_max.slurm
#     done
#
#   BOTH ARMS RUN FROM HERE. The soft arm is no longer taken from ../flowmap_smc_k_sweep/: a paired
#   seed test needs the two arms to differ in exactly one call, same notebook, same cell, same refs.
#   AGG is in the output FILENAME, not just the config -- without it a pair collides on one path.
#   RUN_TAG must ALSO differ per arm: progress_${RUN_TAG}_run.log and phase1_health_${RUN_TAG}.png
#   are keyed on the tag alone, so a shared tag interleaves two jobs' progress logs.
#
#   --nodelist=n-803,n-805 is deliberate. Both hold a WARM /tmp stage (cache HIT, no 4 h staging).
#   n-801 and n-804 filled /tmp and killed jobs 870099 and 869620; n-802 took TWO SIGSEGVs in one
#   night (870119 at step 14, and the other session's 869621). Avoid all three.
#
#   Original single-seed runs (seed 101, done):
#     sbatch --time=540 --export=ALL,K_LOOK=4,RUN_TAG=k04max  flowmap_smc_max.slurm
#     sbatch --time=660 --export=ALL,K_LOOK=8,RUN_TAG=k08max  flowmap_smc_max.slurm
#
# DO NOT SUBMIT BEFORE THE GATE. `analyze_r_k_spread.py` reads the BASELINE's partial checkpoints
# (../flowmap_smc_k_sweep/partial_*.pt, written after every step) and reports a = lambda*sd_k. At
# a >~ 5 the logsumexp has already saturated into a hard max, this run would reproduce its baseline
# exactly, and the GPU-hours would buy nothing:
#
#   python analyze_r_k_spread.py ../flowmap_smc_k_sweep/partial_N8_K4_m1.25_seed101_k04.pt
#
# NO CONTROL JOB. At lambda = 0 every v_k is 0, so max and logmeanexp are BIT-IDENTICAL (asserted in
# tests/test_flowmap_smc_max.py). The k-sweep's `ctrl` job IS this run's untilted baseline; re-running
# it would burn a GPU to reproduce it. CONTROL=1 is therefore not used here, and neither is the
# comma-splitting workaround it needed.
#
# THE QUESTION. The lookahead aggregates K clean candidates into one potential. Until now that has
# been the plug-in estimator log((1/K) sum_k exp(lam*r_k)) -- essentially a mean over K. This run uses
# the hard max, max_k lam*r_k: not what a particle will typically become but what it could AT BEST
# become. Does that raise the within-run E_q[f] curve, and at what cost in diversity?
#
# IT IS LEGAL. Intermediate V_t are TWISTS: they telescope, and q_lambda is pinned by the mandatory
# terminal V_N = lam*f(x_1), which runs at K = 1 and never touches the aggregator. The target is
# identical to the baseline's. The max's bias is spent on twist QUALITY -- variance and where
# resampling spends effort -- which is paid in ess_history / uniq_history, not in correctness.
#
# K IS NOT A NUISANCE PARAMETER HERE. The max boost grows like sigma_k*sqrt(2 ln K) (c_K = 1.03, 1.42
# at K = 4, 8) while the soft boost is K-independent, so raising K STRENGTHENS the twist. The k-sweep
# was written to separate estimator noise from tilt strength; this one deliberately cannot, and the
# K = 4 -> 8 slope is the measurement of that coupling. Do not read the two sweeps the same way.
#
# WHY K = 32 IS NOT HERE. The max/soft gap is largest there (c_32 = 2.07), so it is the natural
# follow-up if both arms come back flat -- but at --time=1440 against 660 it is not the first run.
#
# WHY NOT A LAMBDA SWEEP. Three reasons. (i) The comparison only EXISTS at matched lambda -- the
# baseline is one point, m = 1.25. (ii) The max already raises lambda_eff on its own, so sweeping
# lambda on top moves two knobs that both act on tilt strength and nothing is attributable.
# (iii) At M = 8 a lambda sweep measures degeneracy, not the estimator: a3_2 collapses to a single
# root ancestor by step 2 of 16 at m = 2, and max sharpens selection, so higher arms collapse sooner.
# The SOFT lambda curve already exists (a3_2 at m = 1, 2, 3 plus the k-sweep at m = 1.25), so one max
# point suffices to ask whether max MOVES the E_q[f]-vs-diversity frontier or just slides along it.
#
# EXPECTED FAILURE MODE. A sharper twist should degenerate FASTER. If the cloud collapses to one
# lineage at step 1-2 the run returns nothing interpretable, and the answer is to drop to m = 1.0
# (lambda = 140.1, which has its own a3_2 baseline) -- NOT to weaken the aggregation.
#
# THE COST IS IDENTICAL to the soft run at the same K: same M*K reward rows, same network calls; the
# aggregation is a reduction over a tensor that already exists. The k-sweep's --time values transfer
# unchanged, and any difference in how far a job gets is degeneracy or preemption, never cost.
#
# THE K = 4 JOB IS A LIVE SCOUT. Base-step noise (`gen_base`) is independent of lookahead noise
# (`gen_look`), so at fixed seed step 0 is identical across both K and the early steps are directly
# comparable -- to each other AND to the baseline's k04/k08, which share the same seed. K = 4 reaches
# step 5 in ~48 min. Watch its progress log against ../flowmap_smc_k_sweep/'s at the same step:
#   * collapses to 1 lineage before step 3      -> max is too sharp AT THIS lambda. Cancel k08max
#                                                  (which is sharper still) and re-run at m = 1.0
#                                                  (lambda = 140.1). Do NOT weaken the aggregation.
#   * E_q[f] and uniq/M both track the baseline  -> the max is inert here; check `a` in the gate output,
#                                                  which should have caught it, and stop.
#   * diverges from the baseline, cloud alive    -> let both run.
# Cancelling the other cell at that point costs ~48 min instead of 5.5 GPU-h.
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
# HERE THE PIN IS EVEN MORE LOAD-BEARING THAN IN THE K SWEEP: the soft baseline this run is compared
# against ALL RAN ON L40S. A max run on another card would confound the aggregation with the GPU, and
# the aggregation effect is expected to be small. The diagnostics cell asserts the set is a singleton
# across BOTH directories.
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
# NOT DONE, ON PURPOSE: the score_bank is still rebuilt every job (~19-27 min, ~45 min across these two
# cells). It is an in-process cache on the Distance (`_ref_cache`), never persisted, so NOTHING is
# shared with the baseline jobs and this run waits on none of them -- only on the gate. Persisting it
# touches the frozen reward path, and doing that immediately before an 8.6 GPU-h submission whose whole
# point is that f is bit-identical to the baseline's is the wrong order.
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
RUN_DIR=$REPO/notebooks/flowmap_smc_max
NB=flowmap_smc_max.ipynb

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1                          # stdout is a file here, so Python would block-buffer
export JUPYTER_RUNTIME_DIR=/tmp/$USER/jupyter-runtime-${SLURM_JOB_ID:-manual}
mkdir -p "$JUPYTER_RUNTIME_DIR" "$WORK/logs" "$MPLCONFIGDIR"

export STOP_AFTER="${STOP_AFTER:-run}"
export RUN_TAG="${RUN_TAG:-maxagg}"
export K_LOOK="${K_LOOK:-4}"
export AGG="${AGG:-max}"
export SEED="${SEED:-101}"
export M_PARTICLES="${M_PARTICLES:-8}"
export M_LIST="${M_LIST:-1.25}"
# Set here, never through --export: sbatch splits --export on commas (see the header).
if [ "${CONTROL:-0}" = "1" ]; then
    export GUID_WINDOW="1.0,1.0"
else
    export GUID_WINDOW="0.0,1.0"
fi

cd "$RUN_DIR"          # the notebook derives REPO from cwd ("../..") and writes results here
echo "=== $(date) on $(hostname)  AGG=$AGG  SEED=$SEED  K=$K_LOOK  M=$M_PARTICLES  M_LIST=$M_LIST" \
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
if [ ! -f "$STAGE/model_index.json" ]; then
    echo "FATAL: staged tree has no model_index.json -- copy incomplete" >&2
    exit 1
fi
exec 9>&-               # release the node staging lock: the tree is complete and read-only from here
echo "=== released node staging lock ==="

export FLUX_LOCAL_PATH="$STAGE"
export HF_HUB_OFFLINE=1        # weights are local and complete; a stale token cannot 401 mid-load

echo "=== regression (CPU) ==="
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest \
    "$REPO/tests/test_flowmap_smc.py" "$REPO/tests/test_flowmap_smc_max.py" \
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
