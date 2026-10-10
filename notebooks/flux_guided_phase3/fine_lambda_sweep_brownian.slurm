#!/bin/sh
#SBATCH --job-name=flux-fine-lambda-brownian
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --time=300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gpus=1
#
# FIRST FLUX run of the SHARED-BROWNIAN squared-IEM reward as a gradient-guidance reward
# (--reward brownian -> ExpectedSquaredGlobalIEMDistance). Until now every guided run (Phase 3, Phase 5,
# the CFG sweep) used the i.i.d. estimator while Algorithms 1-3 used the Brownian one, so every
# guided-vs-SMC comparison was confounded by the REWARD and not just the sampler. This is the run that
# removes that confound.
#
# WHAT IT TESTS, in order of what would kill the job first:
#   1. That grad_x r fits at all. The Brownian class gained `expected()` (closed-form mean over refs ->
#      the backward graph stops being linear in R: measured 348 MB -> 3.1 MB at R=64, d=16384) and
#      `expected_gamma_chunk()` (partial sum over the gamma INTERVALS, so _reward_grad accumulates and
#      frees chunk by chunk; peak scales exactly g_chunk/(N_gamma-1)). Both verified on CPU; FLUX's own
#      backward is what leaves almost no margin (ROADMAP Phase 2: OOM at 44.51 of 44.53 GiB at batch 2).
#      `oom_fallback_any` in the results JSON says whether the chunked path fired -- on the i.i.d. runs
#      it fired on EVERY guided step, so expect the same here, not an error.
#   2. Preflights (seconds, both free-ish): f(x_refs) == (R-1)/R to 1e-3, and expected() ==
#      pairwise().mean() on the real backend. The second is the one that matters here -- the identity is
#      exact in float64 on CPU, but this is bfloat16 at d=65536 where ||s_x - s_bar||^2 + spread is a
#      sum of same-order terms. A disagreement means every f in the job is on the wrong scale.
#   3. lambda_s for this estimator, recorded under the SEPARATE registry key "A dog [brownian]" (see
#      pc_sweep._lam_s_key). It WILL differ from the i.i.d. 81.016 -- f's spread differs by
#      construction -- so keying on the bare prompt would have tripped LAM_S_TOL and aborted.
#
# CONFIG. N_GAMMA_BROWN=11, NUM_EPS_BROWN=5 -> K = 10*5 = 50, deliberately MATCHED to the i.i.d. path's
# 50*1 = 50. So score rows, reference-bank cost (10*5*64 = 3200 rows, same as 50*1*64) and the
# gamma-chunk count per guided step are all unchanged, and the ONLY difference is the estimator. The
# split favours eps over grid refinement because the Brownian gamma axis is a deterministic quadrature
# grid contributing no variance (refining it does not reduce spread -- measured flat at 18-20% CV from
# 30 to 200 points), so all averaging must come from N_eps. At matched K this config measured roughly
# HALF the CV of the i.i.d. one across d = 32..16384 on an analytic-score toy; whether that survives a
# learned score at d=65536 is open, and this job does not answer it.
#
# LAMBDA GRID: --lam-max 5.5 --n-lambdas 15 and --sweep-seed 1234, i.e. BIT-IDENTICAL to the stored
# i.i.d. zoom run (fine_lambda_sweep_results_max5.5.json: same 15 lambdas, same z0, n_steps=10,
# shift=3.0, exact_jacobian=True). So the decoded images pair ONE-TO-ONE against that run at matched
# lambda and the comparison needs no rescaling. Note lambda is NOT comparable across estimators in
# principle (CLAUDE.md: "m is not comparable across gamma grids -- compare lambda_eff"); pairing at
# matched absolute lambda is still the right first read because it is what the stored images are at.
#
# --constraint=l40s pinned to ONE model, never a6000|l40s: a GPU change alone shifts f by 16% of
# std_p(f) (job 697271), the same order as anything being measured, and every run this pairs against
# stamps 'NVIDIA L40S'. NOT --nodelist: this script loads FLUX from $WORK/.cache/huggingface (32 GB,
# already present) and does NO /tmp staging, so the node-local staging lesson in CLAUDE.md does not
# apply here -- but ALLOCMEM does, and n-801/n-805 were committed to within ~20 GB of their 515 GB at
# submit time, i.e. could not have fit --mem=24000. Letting Slurm place it among the l40s nodes avoids
# pending on one busy node.
#
# --mem=24000 from Phase 3 job 957386's measured 10.2 GB host peak RSS on base FLUX.1-dev. Never set
# --mem equal to the observed peak: ru_maxrss under-reports page cache and the safetensors are mmap'd.
#
# --time=300: 14 guided lambdas at the stored run's measured 534 s each = 2.1 h, plus the ~10 min
# reference-bank build and the model load. Resumable per lambda (results JSON + PNG hit disk the moment
# each point completes), so a timeout or preemption only costs the in-flight point -- just resubmit.
#
# READING THE RESULT: look at fine_decoded_brownian_max5.5/*.png. Do NOT judge from f, ||x|| or hf --
# none of them predicts recognizability, and ||x|| is ANTI-correlated (CLAUDE.md). The question is
# whether the window is where the i.i.d. reward's was, somewhere else, or absent.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
SWEEP_DIR=$REPO/notebooks/flux_guided_phase3

export HF_HOME="$WORK/.cache/huggingface"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$WORK/.cache/triton"

mkdir -p "$WORK/logs" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
cd "$SWEEP_DIR"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" log -1 --format=%s))"
git -C "$REPO" status --short | head -5

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set in the environment. Submit with:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sbatch fine_lambda_sweep_brownian.slurm" >&2
    exit 1
fi
echo "HF credential: HF_TOKEN present in the environment"

echo "=== running fine_lambda_sweep.py --reward brownian ==="
"$PY" -u fine_lambda_sweep.py --reward brownian --lam-max 5.5 --n-lambdas 15 --sweep-seed 1234

echo "=== done $(date) ==="
ls -l "$SWEEP_DIR"/fine_lambda_sweep_results_brownian_max5.5.json \
      "$SWEEP_DIR"/fine_decoded_brownian_max5.5/*.png 2>/dev/null | tail -20
