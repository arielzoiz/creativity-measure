#!/bin/sh
# Manual CFG x lambda -- submit one wave: `sh submit.sh <seed> <batch>`
#
#   export HF_TOKEN=$(cat $HOME/.hf_token)
#   sh submit.sh 1234 1          # batch 1: a-dog{w=1,3} + car + jacket + sofa   (4 jobs)
#   sh submit.sh 1234 2          # batch 2: a-dog{w=1.5,2} + teapot + building   (3 jobs)
#   sh submit.sh 3141 1          # ... same for the second seed, once seed 1234's images are read
#
# WHY TWO BATCHES OF <=4 RATHER THAN 7 AT ONCE. The binding constraint here is NFS read bandwidth during
# setup, not GPU-h: five concurrent jobs on n-801 produced ZERO points in 3 h (all five page-faulting the
# same 32 GB checkpoint off $WORK, degrading to 1.8 MB/s each as they contended) while a lone, equally
# cold job on n-803 finished setup in 40 min -- PROMPT_STUDY.md, "The n-801 stall". It is contention, not
# cold cache, so MORE concurrent jobs is actively harmful. Submit batch 2 once batch 1 is past setup and
# producing points.
#
# WHY ALL w SHARE A JOB. Setup is 35-90 min and NFS-bound, i.e. comparable to the sampling itself.
# Splitting w across jobs re-pays it per w and buys nothing: cfg_w_sweep.py keys results and decoded dirs
# on w, so one job writes one file per w and every point is independently resumable. Sharing the job also
# guarantees every w in it lands on the SAME GPU, which the cross-w comparison wants anyway (a GPU model
# change alone shifts f by 16% of std_p(f), job 697271 -- hence --constraint=l40s on top).
#
# WHY ONLY "A dog" IS SPLIT, AND WHY THE SPLIT IS {1.0, 3.0} / {1.5, 2.0}.
#   - It is the only prompt carrying FOUR w, because it is the only one needing an in-job w=1 (below), so
#     it is the longest job. Splitting it costs one extra setup (+1.58 GPU-h).
#   - The pairing is not arbitrary: w=1.0 and w=3.0 are the two rows that answer "did CFG change anything
#     at all", so they go in ONE job -- same GPU, same reference bank, same setup -- and arrive in batch 1.
#     w=1.5/2.0 are the refinement and can land later on a different card of the same model, which is
#     within the backend's own ~2.5% nondeterminism floor.
#
# WHY "A dog" GETS AN IN-JOB w=1.0 AND THE OTHER FIVE DO NOT.
#   - The five prompt-study prompts already have a stored w=1 column on EXACTLY this lambda lattice
#     (0, 0.1, ..., 1.0), same n_steps=10, same L40S, from the Phase 5 prompt study. Re-running it would
#     cost ~17 GPU-h to reproduce images already on disk.
#   - "A dog" does not. Its stored Phase 3 runs sit on the 5.5/14 lattice -- 0.392857 and 0.785714, not
#     0.4 and 0.8. Reading them as 0.4/0.8 implies a ~5% f shift (df/dlam ~ 6.4 there), LARGER than the
#     ~2.5% nondeterminism floor, so two extra guided points (~0.6 GPU-h) buy an exact-lambda, same-job
#     baseline instead of a relabelled one. The stored Phase 3 images still appear in the grid on their
#     own row, captioned with their TRUE lambda in red (render_cfg_grid.py's _snap).
#
# PROMPT STRINGS ARE EXACT. "A dog" has a capital A -- it is Phase 3's module default and the key in
# lam_s_by_prompt.json ("A dog": 81.016). "a dog" would be a DIFFERENT prompt: different embeddings ->
# different reference latents -> different lambda_s -> the preflight would record a new registry entry
# and nothing would be comparable to the stored baseline. The other five are bare lowercase nouns,
# matching the prompt study exactly.
#
# COST, from Phase 3's MEASURED 571 s/lambda (exact_jacobian=True, N_PARTICLES=1, n_steps=10, L40S) with
# a pessimistic 1.25x for the CFG arms' extra batch-1 no_grad forward (~714 s/lambda), 10 guided points
# per w (lambda=0 is unguided and ~free), plus ~95 min setup+preflight:
#   five prompts, 3 w        : 5.95 h + 1.58 h = 7.53 h/job  -> --time=610
#   "A dog" A, w=1.0 + 3.0   : 3.57 h + 1.58 h = 5.15 h/job  -> --time=420
#   "A dog" B, w=1.5 + 2.0   : 3.97 h + 1.58 h = 5.55 h/job  -> --time=450
# Per seed: 5*7.53 + 5.15 + 5.55 = 48.3 GPU-h over 7 jobs. Both seeds: 96.7 GPU-h over 14 jobs.
# Every point is resumable (JSON + PNG hit disk the moment it completes), so a preemption on `killable`
# costs the in-flight lambda plus one re-paid setup -- resubmit the identical command.

set -e
cd "$(dirname "$0")"

SEED="$1"
BATCH="$2"
if [ -z "$SEED" ] || [ -z "$BATCH" ]; then
    echo "usage: sh submit.sh <seed> <batch:1|2>" >&2
    exit 1
fi
if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: export HF_TOKEN=\$(cat \$HOME/.hf_token) first" >&2
    exit 1
fi

: "${STAGGER:=120}"
: "${LAM_STEP:=0.1}"
: "${LAM_K:=0,1,2,3,4,5,6,7,8,9,10}"

# NODES: optional space-separated node list, consumed ONE PER JOB in order, to guarantee this batch's
# jobs land on distinct nodes. This is placement, not an extra resource request -- it asks for nothing
# beyond the one GPU each job already needs, and it is the direct fix for the n-801 stall above, where
# five of MY OWN jobs on one node page-faulted the same 32 GB checkpoint at a combined ~5 MB/s and
# produced nothing in 3 h. Pick nodes with free GPUs first:
#   sinfo -N -O "nodelist:12,gres:22,gresused:24,statelong:10" | grep l40s
# A pinned job PENDS if its node fills before it starts, rather than going elsewhere -- so if anything
# sits in PENDING for more than ~15 min, resubmit that one without NODES.
: "${NODES:=}"
_node_i=0

sub() {   # sub <time> <name> <w-list> <prompt>
    _nodeopt=""
    if [ -n "$NODES" ]; then
        _node_i=$((_node_i + 1))
        _n=$(echo "$NODES" | cut -d' ' -f"$_node_i")
        [ -n "$_n" ] && _nodeopt="--nodelist=$_n"
    fi
    # shellcheck disable=SC2086  # _nodeopt is intentionally word-split (empty = no flag)
    sbatch --time="$1" --job-name="$2" $_nodeopt cfg_w_sweep.slurm \
        --w "$3" --prompt "$4" --sweep-seed "$SEED" --lam-step "$LAM_STEP" --lam-k "$LAM_K"
    sleep "$STAGGER"
}

case "$BATCH" in
    1)
        # a-dog A first: it carries the w=1 baseline column, so it is the one job whose absence would
        # leave every other row in its group unreadable.
        sub 420 "cfg-adog-A-s$SEED" "1.0,3.0" "A dog"
        sub 610 "cfg-car-s$SEED"    "1.5,2.0,3.0" car
        sub 610 "cfg-jacket-s$SEED" "1.5,2.0,3.0" jacket
        sub 610 "cfg-sofa-s$SEED"   "1.5,2.0,3.0" sofa
        ;;
    2)
        sub 450 "cfg-adog-B-s$SEED"   "1.5,2.0"     "A dog"
        sub 610 "cfg-teapot-s$SEED"   "1.5,2.0,3.0" teapot
        sub 610 "cfg-building-s$SEED" "1.5,2.0,3.0" building
        ;;
    *)
        echo "FATAL: batch must be 1 or 2 (got '$BATCH')" >&2
        exit 1
        ;;
esac

echo
squeue --me -o "%.9i %.26j %.8T %.6l %.8M %R"
