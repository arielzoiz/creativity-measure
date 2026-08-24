#!/bin/sh
#SBATCH --job-name=flux-tilt2-2
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=a6000
#SBATCH --time=720
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Run 2.2: redo run 2.1. Continue run 2 (job 695271) from beta=0.5422 toward beta=1, with the
# rejuvenation tilt fixed and 695271's unfinished last level completed first.
#
# Job 695271 reached beta=0.5422 (m_eff 1.63 of 3) in 54 sweeps before its 8 h deadline. This picks up
# that particle cloud instead of restarting from p, saving the 5 levels already paid for.
#
# HOW. pi_beta ~ p*exp(beta*lam*f); with beta = beta0 + (1-beta0)u this is pi_beta0 * exp(lam'*u*f)
# for lam' = (1-beta0)*lam = 126.61. So the residual leg is an ordinary SMC whose STARTING
# DISTRIBUTION is the saved cloud. The SMC's reported beta/lam_eff are relative to the residual leg
# and are remapped in the results cells (lam_eff_true = 149.97 + lam_eff_run); without that every row
# reads low by 1.63 sigma. THAT REPARAMETRIZATION IS EXACT FOR THE WEIGHTS AND ONLY FOR THE WEIGHTS --
# see failure (3).
#
# --time 420 -> 720, AND NO --nodelist. These two decisions are linked; do not change one alone.
#
# The staged tree is PER-NODE and only n-602 had one. But n-602's 8 GPUs were fully allocated with
# ~10.5 h of work queued ahead, while n-601 had 5 free and a verified-cold /tmp (785 G free). Waiting
# for the warm node would have delayed the start by longer than the staging pass itself costs.
#
# So this pays the ~4.3 h staging ONCE on n-601, which leaves that node warm for every future leg --
# and the job is stretched to 12 h precisely because staging is a FIXED cost: at 9 h the overhead
# (4.3 staging + 0.5 model/ref bank + 1.0 reserve = 5.8 h) is 64% of the job; at 12 h it is 48%,
# leaving ~6.2 h ~ 50 sweeps ~ the level-4 completion plus 2-3 real levels.
#
# Preemption is the usual argument against a long killable job and it barely applies: the kernel
# checkpoints after EVERY sweep, so being preempted costs ~7.4 min plus requeue, not the run.
# `killable` caps at 1 day, so 1440 is the ceiling if this ever needs to go further.
#
# BETA=1 IS NOT EXPECTED IN ONE JOB -- the intended continuation is a further resume from
# partial_N16_m3_seed101_2_2.pt, which unlike this leg keeps the adapted step size (`s` IS persisted
# here). That next leg should run on n-601 to hit the cache this job builds.
# DO NOT trade the A6000 for a longer non-preemptible --time on gpu-b200: failure (1) is exactly that.
#
# ================ FIVE FAILED ATTEMPTS ARE ENCODED BELOW. DO NOT UNDO ANY. ================
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
# (3) 756301 (RUN 2.1) -- THE MH TILT WAS MISSING beta0*LAM_FULL. The run completed, produced a clean
# grid, and every number in it was wrong. pCN's proposal is reversible w.r.t. the LATENT GAUSSIAN
# PRIOR, so PCNKernel.step's ratio exp(beta*lam*(f'-f)) makes the sweep invariant to p*exp(beta*lam*f):
# its base is ALWAYS p = G#N(0,I), and it cannot represent pi_beta0 as a base no matter how the leg is
# reparametrized. 2.1 passed lam=lam'=126.61 straight to the library, so every rejuvenation sweep
# targeted p*exp(u*126.61*f) -- lam_eff 30.5/50.5/70.3 -- while the rows were reported at
# 180.5/200.4/220.3. The 149.97 already paid for by 695271 was absent from the acceptance ratio: 83%
# of the tilt missing at the first level. The sweeps RELAXED the warm cloud instead of tilting it, and
# E_q[f] fell 1.0335 -> 1.0088 while the reported tilt rose.
#   Two tells, both in 2.1's own artifacts: acceptance jumped 0.12 -> 0.88, and Robbins-Monro drove
#   s 0.4 -> 0.77 -> 0.98 -> 0.999 INTO THE CLAMP -- the adapter demanding ever more global proposals
#   and still unable to reach the 0.23 target. That is what a 5x-too-weak tilt looks like.
#   NOT affected, and deliberately unchanged: the SMC's reweighting. dbeta*lam' == dbeta_true*LAM_FULL
#   exactly, and the dbeta bisection maps exactly too (ceiling 1.0 <-> 1-beta0). So the FIRST level of
#   2.2 must land on beta_true = 0.6525, the same value 2.1 got -- a free check that only `step` moved.
#   Fix: CheckpointingPCNKernel.lam_base in the notebook, added to the MH product while the RECORDING
#   key stays the SMC's own beta. Cannot be done by wrapping the call from outside.
#
# (4) 756301 ALSO ADVANCED FROM A HALF-REJUVENATED LEVEL. 695271's stop_reason_history is
# ['decorr','decorr','decorr','timeout']: its last level did not meet its own stopping rule, the
# deadline fired inside kernel.step 11 sweeps in. 2.1 took that cloud (uniq/N 0.94, acc 0.125) and
# moved beta immediately, baking a short composite kernel into the middle of the stitched ladder.
# 2.2's ResumePCNKernel.init now sweeps at the SOURCE tilt (lam_base = 149.97) until `decorrelation`
# fires, before beta is allowed to move.
#   THE BASELINE IS GONE AND THIS OVERSHOOTS ON PURPOSE: `_record` is keyed by beta and overwrote
#   sweep 0 of that level with sweep 11, so the post-resample origin the decorr metric wants is not on
#   disk. The baseline used is the checkpointed cloud, i.e. a full fresh decorrelation is demanded on
#   top of the 11 sweeps already done. Over-rejuvenating is the right side to err on here.
#   Budgeted by FINISH_MAX_SWEEPS / FINISH_BUDGET_H. If it runs out the run does NOT abort -- it
#   advances with finish_reason='budget' and a loud warning, because a better-but-unfinished level 4
#   still beats discarding the job. READ finish_reason BEFORE trusting the stitched ladder.
#
# (5) 776411 -- NO HF CREDENTIAL. Died 60 s in, in the preflight, on a bare
# `GatedRepoError: 401 ... Access to model black-forest-labs/FLUX.1-dev is restricted`. The cause was
# not a revoked token: HF_TOKEN_PATH pointed at /a/home/cc/STUDENTS/math/$USER/.hf_token, and this
# account's $HOME is /a/home/cc/math/$USER -- no `students/` component. No such file has ever existed.
#   WHY IT WORKED BEFORE AND NOT NOW: sbatch propagates the SUBMITTING shell's environment into the
#   job, so every earlier run inherited an HF_TOKEN that happened to be exported in an interactive
#   session. Submit from a shell without it -- a script, cron, an agent -- and the job is logged out.
#   Never let a credential reach a job by ambient inheritance; it makes the job unreproducible.
#   WHY THE FALLBACK DID NOT SAVE IT: huggingface_hub resolves HF_TOKEN, then HF_TOKEN_PATH. Setting
#   HF_TOKEN_PATH to a nonexistent path REPLACES the $HF_HOME/token default rather than falling
#   through to it, so a typo in the path is indistinguishable from being logged out.
#   Fix: HF_TOKEN_PATH="$HOME/.hf_token" plus an explicit file check before the preflight, so the
#   failure names itself instead of arriving as a 401 traceback. The token must stay in $HOME at 0600
#   and never under $WORK, which is group-readable:
#     umask 077; printf '%s' '<hf_token>' > $HOME/.hf_token
#
# GOOD NEWS FROM 776411: it cost nothing. The preflight is deliberately ordered before staging and
# before the model load, so a missing credential burns 60 s of an A6000 rather than 25 min or 4.3 h.
#
# OBSERVABILITY (the other lesson from 754633). nbconvert buffers every cell's output into the
# notebook and writes it once, at the end -- so a stall is undiagnosable by construction and 6 h
# produced zero evidence. This now runs under PAPERMILL with --log-output (cell stdout streams to
# this job's .out as it happens) and --request-save-on-cell-execute (the output notebook is written
# after every cell, so a kill leaves a partial notebook rather than nothing). Cell 1 also arms a
# faulthandler watchdog that dumps the Python stack every 10 min, so any future hang names its frame.
#
# OUTPUTS ARE TAGGED (RUN_TAG=2_2). The untagged names derive from N/M_TILT/SEED, none of which change
# on a resume, so an untagged run would overwrite partial_N16_m3_seed101.pt -- its own resume source --
# on the first sweep. An assertion in the notebook enforces src != dst.
#   2_1 -> 2_2 IS NOT COSMETIC. Reusing the tag would overwrite partial/levels/grid in place, and
#   decode_levels.py writes ONE PNG PER (level, particle) into decoded_<tag>/: this leg is expected to
#   produce FEWER levels than 2.1's three, so 2.1's lvl05..lvl07 files would survive next to new lvl05
#   files at a different lambda, leaving a directory that silently mixes two runs. 2.1 is also the
#   evidence for failures (3) and (4) and must be kept intact.
#
# MAX_SWEEPS 24 -> 40: precautionary. The cap was NEVER reached in 695271 (12/11/20/11 sweeps), so it
# was not binding; but acceptance fell 0.75 -> 0.12 and is likely to make it binding as lambda rises.
# With the tilt fixed (failure 3) this is now EXPECTED to bind -- levels stopping on `cap` rather than
# `decorr` is the normal reading of this leg, not an anomaly.
#
# COST. Two costs stack. Completing level 4 (failure 4) runs at lam_eff=149.97 where acceptance sits
# near 0.12-0.23; pCN refresh per sweep is roughly acc*(1-sqrt(1-s^2)), so decorr needs order 15-20
# sweeps, ~2-2.5 h at the measured 7.4 min/sweep, BEFORE beta may move (capped by FINISH_BUDGET_H=3 h).
# Then the corrected kernel makes each subsequent level far harder than 2.1's mislabelled ones. Of 9 h:
# ~25 min model load + reference bank (staging is a cache hit on a warm node), 1 h reserve, ~2.5 h
# finishing, leaving ~5 h ~ 40 sweeps ~ 2-3 levels. beta=1 IS NOT EXPECTED HERE -- a further resume from
# this run's own checkpoint is the intended continuation, and `s` IS persisted here so that one keeps
# the adapted step size instead of restarting from s0. The kernel checkpoints every sweep and stops
# cleanly at a deadline from SLURM_JOB_END_TIME.
#
# `killable` is preemptible and caps at 1 day, which is why --time is 1440 and not more. gpu-b200 has a
# 5-day non-preemptible limit and is the obvious temptation -- DO NOT TAKE IT. A resume is only valid
# on the source run's architecture (failure 1), the preflight below asserts A6000, and the assert is
# correct: buying wall time with a different card costs the whole resume.
#
# No `source ~/.bashrc`: that file is empty here and conda/HF_* live in ~/.cshrc (tcsh), so this
# script sets everything explicitly and works from a fresh shell or a requeue.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
SWEEP_DIR=$REPO/notebooks/flux_lambda_sweep_strong_2
NB=flux_dev_strong_tilt_sweep_2_2.ipynb

# $WORK is group-readable, so the HF *credential* must not live there; the 32 G of weights may.
export HF_HOME="$WORK/.cache/huggingface"
# The token must NOT live under $WORK -- that tree is group-readable. It goes in a 0600 file in the
# private $HOME, and HF_TOKEN_PATH points there. NOTE the path: $HOME here is /a/home/cc/math/$USER,
# NOT /a/home/cc/students/math/$USER -- the old hardcoded "students/" path does not exist for this
# account, which is how job 776411 died (see failure 5).
export HF_TOKEN_PATH="$HOME/.hf_token"
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

# Check the token FILE before the preflight calls auth_check, so a missing token reports itself
# instead of surfacing as a bare 401 GatedRepoError traceback (failure 5). huggingface_hub resolves
# HF_TOKEN, then HF_TOKEN_PATH -- and setting HF_TOKEN_PATH to a path that does not exist DISABLES
# the $HF_HOME/token fallback rather than falling through to it, so a wrong path reads as "logged out".
if [ -z "$HF_TOKEN" ] && [ ! -s "$HF_TOKEN_PATH" ]; then
    echo "FATAL: no HF credential. HF_TOKEN is unset and HF_TOKEN_PATH=$HF_TOKEN_PATH is missing/empty." >&2
    echo "  Create it (0600, in \$HOME -- never under \$WORK, which is group-readable):" >&2
    echo "    umask 077; printf '%s' '<hf_token>' > \$HOME/.hf_token" >&2
    exit 1
fi
[ -n "$HF_TOKEN" ] && echo "HF credential: HF_TOKEN from the environment" \
                   || echo "HF credential: $HF_TOKEN_PATH ($(wc -c < "$HF_TOKEN_PATH" | xargs) bytes)"

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
ls -l "$SWEEP_DIR"/*2_2*.pt "$SWEEP_DIR"/refs_*.pt "$SWEEP_DIR"/strong_tilt_grid_2_2.png 2>/dev/null || true
exit $rc
