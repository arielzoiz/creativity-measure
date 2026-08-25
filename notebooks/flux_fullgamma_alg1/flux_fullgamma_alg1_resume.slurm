#!/bin/sh
#SBATCH --job-name=flux-fg2
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
# Run fg2: leg 2 of the full-gamma-window Algorithm 1 ladder. Resumes fg1 (job 776776) from
# beta = 0.6657 toward beta = 1, finishing fg1's unfinished last level first.
#
# WHY RESUME AT ALL. fg1 did not break -- it ran out of wall clock with the kernel healthy:
# level-mean acceptance 0.133 at the last level and 0.221 at the one before (target 0.23),
# uniq/N = 1.00 at every level, pair/iid ~ 1, s = 0.7139 adapting normally and clear of both clamps,
# and E_q[f] still climbing monotonically (+3.2 sigma_p(f) to m_eff = 2.00 of 3).
#
# READ THIS BEFORE TRUSTING ANY `acc` NUMBER. The persisted per-level `acc` is the level's LAST SWEEP,
# not its mean -- `_record` is keyed by beta and overwritten every sweep. At N=8 it is quantized to
# eighths. fg1's level 6 recorded acc = 0.000, which occurs 32% OF THE TIME at that level's true mean
# of 0.133, and it was initially misread as a collapsed kernel. The true means are recoverable by
# inverting the Robbins-Monro step-size recursion; this leg persists `acc_mean` directly so nobody has
# to.
#
# HOW. pi_beta ~ p*exp(beta*lam_max*f); with beta = beta0 + (1-beta0)u this is
# pi_beta0 * exp(lam'*u*f) for lam' = (1-beta0)*lam_max = 78.911. So the residual leg is an ordinary
# SMC whose STARTING DISTRIBUTION is the saved cloud. The SMC's reported beta/lam_eff are relative to
# the residual leg and are remapped in the results cell (beta_true = 0.6657 + u*0.3343,
# lam_eff_true = 157.123 + u*78.911); without that every row reads low by m_eff = 2.00.
#
# ============================ THE FOUR THINGS THAT MAKE A RESUME VALID ============================
#
# (1) lam_base = beta0*lam_max = 157.123 MUST BE IN THE MH RATIO. pCN's proposal is reversible w.r.t.
# the LATENT GAUSSIAN PRIOR, so PCNKernel.step's ratio exp(beta*lam*(f'-f)) makes the sweep invariant
# to p*exp(beta*lam*f): its base is ALWAYS p = G#N(0,I), and it cannot represent pi_beta0 as a base no
# matter how the leg is reparametrized. Run 2_1 (job 756301) passed lam=lam' straight to the library;
# every rejuvenation sweep then targeted a tilt with 83% of the strength missing at the first level,
# the sweeps RELAXED the warm cloud instead of tilting it, and E_q[f] FELL while the reported tilt
# rose. The run completed, produced a clean grid, and every number in it was wrong.
#   THE REPARAMETRIZATION IS EXACT FOR THE WEIGHTS AND ONLY FOR THE WEIGHTS. The SMC's reweighting is
#   deliberately untouched: dbeta*lam' == dbeta_true*LAM_FULL exactly, and the dbeta bisection maps
#   exactly too. So the first level of this leg is a FREE CHECK that only `step` moved.
#
# (2) FINISH fg1's LAST LEVEL BEFORE beta MOVES. fg1's stop_reason_history ends in 'timeout': the
# deadline fired inside kernel.step 16 sweeps into beta = 0.6657, so that level never met its own
# `decorr` rule. Advancing from it bakes a short composite kernel into the middle of the stitched
# ladder (failure 4). ResumePCNKernel.init sweeps at the SOURCE tilt (157.123) until decorrelation
# fires, before beta is allowed to move.
#   THE BASELINE IS GONE AND THIS OVERSHOOTS ON PURPOSE: `_record` is keyed by beta and overwrote
#   sweep 0 of that level with sweep 16, so the post-resample origin the decorr metric wants is not on
#   disk. The baseline used is the checkpointed cloud, i.e. a full fresh decorrelation on top of the 16
#   sweeps already done. Over-rejuvenating is the right side to err on.
#   Budgeted by FINISH_MAX_SWEEPS=30 / FINISH_BUDGET_H=2.0. If it runs out the run does NOT abort -- it
#   advances with finish_reason='budget' (or 'cap') and a loud warning. READ finish_reason BEFORE
#   trusting the stitched ladder.
#
# (3) THE SOURCE IS partial_N8_m3_seed101_fg1.pt, NOT levels_N8_m3_seed101_fg1_TIMEOUT.pt.
# The levels_ file carries ['X','ancestors','beta','ess','fX','lam_eff','logw'] -- no aux/Z (the pCN
# chain state) and no s (the adapted step size). It is NOT resumable. The notebook asserts both.
#
# (4) SAME CARD, SAME REWARD, SAME WINDOW. FLUX runs bf16 (8-bit mantissa) and Ampere vs Ada select
# different tensor-core kernels and reduction orders, so identical denoiser inputs differ at round-off;
# through the score bank and normalizer that shifted f by 11% of std_p(f) and correctly aborted job
# 697271's resume. --constraint=a6000 plus persisted reference latents pin both halves. The notebook
# also asserts the GAMMA WINDOW against the source config: the window is part of what f IS, so a
# resume onto a different one is meaningless rather than merely inaccurate.
#
# lambda_s IS CACHED (78.678) AND MUST NOT BE RE-PROBED. It is a property of the reward and p, both
# unchanged. Re-probing would spend 14 min to re-derive a number we already have -- and, worse, a
# slightly different one, which would move LAM_FULL and make this leg target a different ladder than
# the levels it is stitched onto. This is why fixed costs here are ~40 min against fg1's ~52.
#
# `s` IS RESTORED (0.7139) and adapt_rate stays at the library default 0.1. Run 2_2 raised it to 0.25
# only because ITS source predated `s` persistence and had to re-adapt from s0 = 0.4. fg1 persists it,
# so that workaround is not wanted here.
#
# COST. ~10-25 min model load (warm /tmp) + ~23 min reference bank + ~3 min reward guard ~= 40 min of
# the 8 h, 1 h reserved for decode/render, leaving ~6.3 h ~ 105 sweeps at 3.4 min/sweep. Finishing
# level 6 may take 15-30 of those. At fg1's recent 30 sweeps/level and d_beta ~ 0.10-0.12, expect 2-3
# further levels and beta ~ 0.85-0.95. BETA = 1 IS NOT PROMISED -- a third leg resumes from
# partial_..._fg2.pt with `s` intact.
#
# THE NODE AND THE HUB: identical reasoning to fg1's script, and both were confirmed by that run's own
# log. --nodelist=n-602 because its /tmp holds the SHA-keyed staged FLUX tree (fg1 logged
# "node-local cache HIT"; a cold node costs a MEASURED 257 min, job 755539). HF_HUB_OFFLINE=1 because
# $HOME/.hf_token is stale -- fg1 logged "hub access: UNAVAILABLE (GatedRepoError)" and ran perfectly,
# while job 776480 died on that same 401 under the old hard-exit preflight. Nothing here needs the hub.
#
# `killable` is preemptible. Preemption costs ~3.4 min plus requeue, not the run.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
RUN_DIR=$REPO/notebooks/flux_fullgamma_alg1
NB=flux_fullgamma_alg1_resume.ipynb

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1                          # stdout is a file here, so Python would block-buffer
export JUPYTER_RUNTIME_DIR=/tmp/$USER/jupyter-runtime-${SLURM_JOB_ID:-manual}
mkdir -p "$JUPYTER_RUNTIME_DIR" "$WORK/logs" "$MPLCONFIGDIR"
cd "$RUN_DIR"            # the notebook derives REPO from cwd ("../..") and writes results here

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# The resume source must exist BEFORE anything expensive runs. Named explicitly because the other
# checkpoint in this directory looks equally plausible and is not resumable (decision 3 above).
SRC="$RUN_DIR/partial_N8_m3_seed101_fg1.pt"
if [ ! -s "$SRC" ]; then
    echo "FATAL: resume source $SRC missing or empty." >&2
    echo "  levels_N8_m3_seed101_fg1_TIMEOUT.pt is NOT a substitute: no aux/Z, no s." >&2
    exit 1
fi
echo "resume source: $(basename "$SRC") ($(du -h "$SRC" | cut -f1))"

# Preflight: fail in seconds, not after minutes of model loading. Ordered BEFORE staging. The auth
# probe is advisory -- this run loads offline (see the header).
"$PY" - <<'PREFLIGHT'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
name = torch.cuda.get_device_name(0)
print(f"GPU: {name}  {gb:.0f} GB total")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
# Hard, unlike fg1's warning: fg1 was self-consistent on any card, but this leg is STITCHED onto fg1's
# ladder and its reward guard compares f against fg1's stored values. A different architecture shifts f
# at round-off and the guard would abort ~25 min in. Fail here instead, in seconds.
assert "A6000" in name, f"resume requires fg1's card (A6000), got {name!r}"
try:
    from huggingface_hub import auth_check
    auth_check("black-forest-labs/FLUX.1-dev")
    print("FLUX.1-dev hub access: OK (unused -- this run loads offline from the staged snapshot)")
except Exception as exc:
    print(f"hub access: UNAVAILABLE ({type(exc).__name__}). Not fatal: this run loads offline from "
          f"the staged local snapshot and never resolves a repo id.")
PREFLIGHT

# --- stage the weights to node-local disk (the fix for 754633) -------------------------------------
SNAP=$(ls -d "$HF_HOME"/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
    echo "FATAL: no FLUX snapshot under $HF_HOME/hub" >&2
    exit 1
fi
SHA=$(basename "${SNAP%/}")
STAGE=/tmp/$USER/flux-snap-$(echo "$SHA" | cut -c1-8)
N_SRC=$(cd "$SNAP" && find -L . -type f | wc -l)

# Age-based sweep of OTHER snapshots only; never the one this job needs. No EXIT trap on $STAGE -- that
# is what would throw away a 4.3 h staging pass.
find /tmp/$USER -maxdepth 1 -name 'flux-snap-*' ! -name "$(basename "$STAGE")" -mtime +7 \
    -exec rm -rf {} + 2>/dev/null || true
find /tmp/$USER -maxdepth 1 -name 'jupyter-runtime-*' -mtime +1 -exec rm -rf {} + 2>/dev/null || true

if [ -f "$STAGE/model_index.json" ] && [ "$(find -L "$STAGE" -type f | wc -l)" -eq "$N_SRC" ]; then
    echo "=== node-local cache HIT: $STAGE ($(du -sh "$STAGE" | cut -f1), $N_SRC files) -- no staging ==="
else
    echo "=== node-local cache MISS on $(hostname): staging from NFS. MEASURED cost 257 min at" \
         "~2 MB/s (job 755539). In an 8 h job that leaves ~2 h of sweeping -- expect a badly" \
         "truncated leg. Visible here, not silent. ==="
    # THE PARTIAL PATH IS JOB-UNIQUE, and that is not cosmetic. The older scripts stage into a shared
    # "$STAGE.partial" and open with `rm -rf` on it -- so a second job starting a staging pass on a
    # node where one is already running DELETES the first job's half-copied tree and both then fight
    # over the same NFS reads. Observed live on 2026-08-24: job 778007 was 13 min into a 4.3 h staging
    # pass on n-601 when this leg was being submitted. Keyed by job id, the two cannot collide.
    t_stage=$(date +%s)
    PART="$STAGE.partial.${SLURM_JOB_ID:-manual}"
    rm -rf "$PART"
    mkdir -p "$PART"
    ( cd "$SNAP" && find -L . -type d -exec mkdir -p "$PART/{}" \; )
    ( cd "$SNAP" && find -L . -type f -print0 | xargs -0 -P 6 -I{} cp -L {} "$PART/{}" )
    # Another job may have finished staging while this copy ran. `mv` onto an existing directory would
    # nest it INSIDE rather than replace it, quietly producing flux-snap-XXXX/flux-snap-XXXX -- so
    # check first and discard our duplicate instead.
    if [ -f "$STAGE/model_index.json" ]; then
        echo "=== another job staged $STAGE while we copied; discarding our duplicate ==="
        rm -rf "$PART"
    else
        mv "$PART" "$STAGE"
        echo "=== staged in $(( ($(date +%s) - t_stage) / 60 )) min: $(du -sh "$STAGE" | cut -f1) ==="
    fi
fi
chmod -R u+rwX,go-rwx /tmp/$USER 2>/dev/null || true

if [ ! -f "$STAGE/model_index.json" ]; then
    echo "FATAL: staged tree has no model_index.json -- copy incomplete" >&2
    exit 1
fi
export FLUX_LOCAL_PATH="$STAGE"
export HF_HUB_OFFLINE=1        # weights are local and complete; a stale token cannot 401 mid-load

# --- execute: NO JUPYTER KERNEL (three jobs died on the client<->kernel handshake) -----------------
OUT="run_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT (run_notebook.py, kernel-free, streaming) ==="
set +e
"$PY" run_notebook.py "$NB" "$OUT"
rc=$?
set -e

echo "=== done $(date)  (run_notebook exit $rc) ==="
ls -l "$RUN_DIR"/*_fg2*.pt "$RUN_DIR"/*_fg2*.png 2>/dev/null || true
exit $rc
