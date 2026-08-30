#!/bin/sh
#SBATCH --job-name=flowmap-diag
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=a6000
#SBATCH --time=600
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# PHASE 8 -- the two cheap questions that gate all further Algorithm 3 work  (lambda = 0 trajectories)
#
#   sbatch flowmap_smc_diag.slurm
#
# WHERE THIS SITS. Phase 6 killed the transition-side fix and found identity commits at t ~ 0.19.
# Phase 7 showed the Algorithm 1 transplant (renoise the PARTICLE back past commitment, replay the
# unmodified base process) mixes and preserves p. Neither touched the actual defect: Alg 3 provably
# does NOT sample q_lambda -- dE_q[f]/dlam is NEGATIVE in every interval of both a3 sweeps, where the
# exponential family requires it to equal Var_q(f) >= 0. Before building any fix, two measurements.
#
# STEP 1 -- IS BASELINE (~30 min). No guidance, no resampling => the trajectory is lambda-INDEPENDENT,
# so ONE batch of M=64 and ONE reward call give E_q[f] and ESS across the WHOLE lambda range by
# reweighting. Self-normalized IS from p: correct by construction. Delivers (a) ground truth the
# equilibrium identity must satisfy, validating that gate on a known-good estimator before we judge
# Alg 3 with it; (b) the lambda where IS runs out of ESS -- below that, SMC buys nothing; (c) a
# possible EARLY EXIT: if IS is adequate at m in {1,2,3}, the SMC apparatus is unnecessary at the
# tilts we care about, and the right answer is to stop building samplers for this regime.
#
# STEP 2 -- IS THE TWIST INFORMATIVE? (~4 h). V_t is a K-sample MC estimate; what matters is
# between-particle signal over within-particle noise. The earlier estimate used `f_std`, taken over all
# M*K candidates, which mixes the two and OVERSTATES noise -- so its SNR ~ 2 is a LOWER bound. This
# records per-particle r_k at t = 0.125 (before commitment) and t = 0.5 (well after) and separates them,
# de-biasing the between-spread by within^2/K. lambda cancels from the SNR entirely -- it scales signal
# and noise alike -- so the deliverable is the REQUIRED K, in closed form, not another sweep.
#
# PREDICTION ON RECORD: Phase 7 showed a renoise from t=0.5 back to 0.375 leaves images identical, and
# eta=1.5 at t=0.5 only reaches t'=0.4495 -- smaller still. So past commitment the K candidates should
# be near-identical, within-spread small, SNR much better than 2. If so, V_t = lam*f(map(x,t,1)) is
# essentially exact at K=1 there: no Monte Carlo, 4x cheaper, noise-free -- and the guidance should be
# restructured to spend its K before t ~ 0.19 instead of after.
#
# COST. Reward is (N_GAMMA-1)*NUM_EPS*B = 87B denoiser rows. Step 1 is 87*64 = 5,568 rows; Step 2 is
# 2 * 87 * (32*8) = 44,544. Total ~50k rows: 4.6 h at the 0.33 s/row measured on n-601 (job 782139),
# 6.9 h at the 0.49 measured on n-602 (781479) -- hence --time=600 rather than 420. The reference bank
# (~27 min) rebuilds every job on top of that; it is not cached.
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
# (2) --constraint=a6000 AND NO --nodelist. BOTH n-601 and n-602 are warm as of 2026-08-26: job 782111
#     logged a cache HIT on n-601, so the older "n-602 is the only warm node" rule is retired. That
#     matters because n-602 is often full -- it had 8 running jobs that day, earliest GPU 6 h out, Slurm
#     projecting a 13 h wait (job 781831 sat PD on "(Resources)"), while n-601 had 6 of 8 idle. Letting
#     Slurm pick either warm node is strictly better than pinning one. Do NOT write
#     `--nodelist=n-601,n-602`: with --nodes=1 a multi-node nodelist is a MINIMUM request, not a choice.
#     --time=420 is kept as headroom in case a third node appears cold; check the HIT/MISS line.
#     Superseded note follows.
#     tree, but on 2026-08-26 it was fully allocated -- 8 running jobs, earliest GPU free in 6 h, Slurm
#     projecting a 13 h wait (job 781831 sat PD on "(Resources)"). n-601 had 6 of 8 A6000s idle, so this
#     deliberately pays the MEASURED 257 min cold stage to start now: ~5.5 h to results instead of ~14,
#     and it leaves n-601 warm so the project has a SECOND viable node from here on. --time=420 exists
#     to cover that stage; do not shorten it while this node is cold. Original note follows.
#
# (2b) --nodelist=n-602 AND --constraint=a6000. The FLUX snapshot is 32 G on an NFSv3 netapp mount and
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

export STOP_AFTER=diag
export RUN_TAG="${RUN_TAG:-diag_1}"

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

# Static import check, BEFORE staging and before anything expensive. Job 782111 died on
# `ImportError: cannot import name 'posterior_z_ddpm_step'` after 28.8 min -- 26.4 of it rebuilding the
# reference bank, which is NOT cached and is re-paid every single job. The import sat at module level
# in a cell whose body was behind `if DO_STRENGTH:`, so reverting the library broke a cell that no
# longer ran. Seconds here, half an hour of A6000 there.
echo "=== notebook import check ==="
"$PY" check_notebook_imports.py "$NB"

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
OUT="run_diag_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT (run_notebook.py, kernel-free, streaming) ==="
set +e
"$PY" run_notebook.py "$NB" "$OUT"
rc=$?
set -e

echo "=== done $(date)  (run_notebook exit $rc) ==="
ls -l "$RUN_DIR"/diag_*"$RUN_TAG"* 2>/dev/null || true
exit $rc
