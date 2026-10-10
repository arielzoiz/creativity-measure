#!/bin/sh
# Brownian-reward prompt/seed/alg study -- 16 jobs, the whole campaign.
#
#   export HF_TOKEN=$(cat $HOME/.hf_token) && sh submit_all.sh
#
# WHAT THIS IS. Every guided run to date (Phase 3, Phase 5, the CFG sweep) used the i.i.d. estimator of
# D_IEM^2 while Algorithms 1-3 used the shared-Brownian one, so guided-vs-SMC comparisons were
# confounded by the REWARD and not just the sampler. `--reward brownian` selects
# ExpectedSquaredGlobalIEMDistance (N_gamma=11 x N_eps=5, K=50, matched to the i.i.d. path's 50x1), so
# score rows, reference-bank cost and gamma-chunk count are unchanged and only the estimator differs.
#
# MATRIX. 8 (prompt, seed) pairs x 3 algs x 11 lambda in {0, 0.1, ..., 1.0} = 264 cells.
# The algs are flow_guided, flow_guided_pc (corrector_steps=1), and CFG w=2 (flow_guided with a
# cfg_velocity_fn driving the Euler transport ONLY -- x_hat_0 and the reward gradient stay conditional).
#
# WHY 16 JOBS AND NOT 24. pc_sweep.py runs a comma-separated --arm list on ONE setup, and setup is
# ~36 min (model load + 64 reference latents + the 3200-row score bank + probe + preflights) paid per
# PROCESS because the bank is not persisted. With only ~6 free L40S GPUs the makespan is throughput-
# bound, not critical-path-bound, so fewer-and-longer wins: splitting pc_guided into its own job drops
# the longest job 4.9 h -> 3.6 h but raises the total 54.6 -> 59.7 GPU-h and the makespan 9.1 -> 10.0 h.
# CFG needs its own job only because it lives in a different driver (cfg_w_sweep.py builds the
# null-prompt branch, which pc_sweep.py's setup does not).
#
# MEASURED COSTS (today's job 1004109 for flow_guided; the 352 stored Phase 5 points for the pc ratio):
#     flow_guided   465 s/lambda      (iid was 580 s -- brownian is 19% faster: 10 chunked backward
#                                      passes of 5 eps rows each, vs iid's 50 passes of 1 row)
#     pc_guided    1093 s/lambda      (measured pc/flow_guided = 2.35x, NOT the 1.9x in CLAUDE.md)
#     cfg w=2       480 s/lambda      (one extra batch-1 uncond forward per ODE step)
#     setup        ~36 min/job
#   -> Job A 4.93 h, Job B 1.93 h, 54.6 GPU-h total.
#
# LONGEST-FIRST. All eight 4.93 h jobs are submitted before the eight 1.93 h ones so the long tails
# start earliest; with equal priority Slurm roughly honours submit order.
#
# "A dog"/1234 RERUNS k=4 AND k=8 even though job 1004109 already produced lam=0.3929 and 0.7857.
# --lam-k applies to both arms in a job and pc_guided needs all 11, so skipping them for flow_guided
# would need a second job: 36 min of extra setup to avoid 15.5 min of compute. Rerunning is cheaper AND
# removes the only cross-run splice from the grid.
#
# PROMPT STRINGS ARE EXACT. "A dog" is capital-A (matches LAM_S_SEED_VALUES and every Phase 3/5 run);
# the rest are bare lowercase nouns (matches lam_s_by_prompt.json). A prompt IS a different reward --
# the reference latents are G(z) under the prompt-conditioned velocity -- so case changes the reward.
# lambda_s is recorded per (prompt, reward_kind) under the key "<prompt> [brownian]", which is why these
# do not trip the 5% LAM_S_TOL guard against the stored i.i.d. values.
#
# Every point is resumable (atomic results JSON + PNG the moment it completes), so a killable
# preemption costs only the in-flight lambda -- resubmit the identical line.

set -e

PHASE5=/home/dcor/arielzoizner/projects/creativity-measure/notebooks/flux_guided_phase5
CFGDIR=/home/dcor/arielzoizner/projects/creativity-measure/notebooks/flux_guided_cfg
LAMK=0,1,2,3,4,5,6,7,8,9,10
LAMSTEP=0.1

if [ -z "$HF_TOKEN" ]; then
    echo "FATAL: HF_TOKEN is not set. Use:" >&2
    echo "  export HF_TOKEN=\$(cat \$HOME/.hf_token) && sh submit_all.sh" >&2
    exit 1
fi

# prompt<TAB>seed, in the order the final grid's rows are sorted (prompt outer, seed inner)
PAIRS="car:1234
jacket:1234
jacket:3141
A dog:1234
A dog:4242
A dog:3141
sofa:1234
teapot:1234"

echo "=== Job A (flow_guided + pc_guided), 8 jobs, ~4.93 h each ==="
cd "$PHASE5"
echo "$PAIRS" | while IFS=: read -r PROMPT SEED; do
    [ -n "$PROMPT" ] || continue
    sbatch pc_sweep.slurm flow_guided,pc_guided --reward brownian \
        --prompt "$PROMPT" --sweep-seed "$SEED" --corrector-steps 1 \
        --lam-step "$LAMSTEP" --lam-k "$LAMK"
done

echo "=== Job B (CFG w=2), 8 jobs, ~1.93 h each ==="
cd "$CFGDIR"
echo "$PAIRS" | while IFS=: read -r PROMPT SEED; do
    [ -n "$PROMPT" ] || continue
    sbatch cfg_w_sweep.slurm --w 2 --reward brownian \
        --prompt "$PROMPT" --sweep-seed "$SEED" \
        --lam-step "$LAMSTEP" --lam-k "$LAMK"
done

echo "=== submitted; squeue --me ==="
squeue --me -o "%.9i %.3t %.22j %.11M %.11l %R"
