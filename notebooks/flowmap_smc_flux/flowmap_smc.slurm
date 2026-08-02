#!/bin/sh
#SBATCH --job-name=flowmap-smc
#SBATCH --output=/home/dcor/arielzoizner/logs/%x-%j.out
#SBATCH --error=/home/dcor/arielzoizner/logs/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=a6000|l40s
#SBATCH --time=360
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --gpus=1
#
# Algorithm 3 -- SMC weighted flow maps on the FLUX.1-dev flow map.
#
# PHASES. One notebook serves the whole execution order; STOP_AFTER picks how far it goes, and each
# phase caches to disk so a later job loads what an earlier one built. Submit in this order:
#
#   sbatch --export=ALL,STOP_AFTER=health flowmap_smc.slurm     # ~20 min. THE GATE.
#   sbatch --export=ALL,STOP_AFTER=refs   flowmap_smc.slurm     # ~45 min. Builds + persists refs.
#   sbatch --export=ALL,STOP_AFTER=run    flowmap_smc.slurm     # the tilted runs.
#
# DO NOT SKIP THE GATE. Phase 1 checks that p itself is healthy (latent std against base FLUX's
# known-good 1.0150, and that denoise(x,t) differs from map(x,t,1) -- if they coincide the two-timestep
# conditioning is inert and the LoRA load is suspect). If p looks degraded, stop before building 3 h on
# top of it: every reward and lambda calibration downstream would be measured against the wrong p.
#
# COST, re-measured on this hardware. One reward call on B latents costs (N_GAMMA-1)*NUM_EPS*B = 87B
# denoiser rows at **0.206 s/row** -- job 700483's reference-bank build did 5568 rows in 19.1 min on an
# L40S, which is the exact reward code path the run spends ~97% of its time in. (The A6000 figure the
# plan budgeted with was 0.251; the narrower 8-row flow-map path measures 0.231, since per-call
# overhead is amortised over less work.) At B = M*K = 32 a guided step is 2784 reward rows ~ 570 s
# against M(2K+2) = 80 flow-map rows ~ 18 s.
#
# With guid_window=(0.1, 0.9) that is 2.17-2.44 h per m, so all of {3, 2, 1} fits in a 9 h job even at
# the pessimistic rate. (0.1, 1.0) would be 2.34-2.62 h and drop the third m -- which is why the run
# uses the narrower of the plan's two sanctioned windows. Submit this phase with --time=540.
#
# SETUP is not free: loading ~34 GB of FLUX weights off the shared NFS volume took 22 min on a cold
# node (measured: 4% CPU, 51 s of CPU time in 20 min -- I/O-bound) and 0.5 min on a warm one. The
# reference bank rebuilds every job (~19 min); lambda_s is cached to JSON and does not.
#
# CHECKPOINTING. The SMC is one long library call, so the notebook passes an `on_step` callback that
# writes the cloud after every step (temp file + os.replace, so a kill mid-write cannot corrupt the
# last good file) and raises at a deadline derived from SLURM_JOB_END_TIME. The existing FLUX jobs have
# repeatedly hit their deadline; this is not optional.
#
# REFERENCES ARE PINNED TO p, NOT TO guid_window. The saved reference file records schedule, n_steps,
# ts, stoch_window, inside_step, outside_step, prompt, guidance and the denoiser row cap, and the run
# asserts they match. `guid_window` is deliberately NOT in that list, so a window sweep reuses one
# reference set and one p -- which is what makes the comparison fair. Changing anything else needs a
# reference rebuild (delete refs_R64_seed7.pt).
#
# --constraint=a6000|l40s pins the score half of the reproducibility chain; persisting the reference
# latents takes the model out of the other half. Job 697271 aborted because references regenerated on a
# different GPU shifted f by 16% of std_p(f) -- enough to change weight ratios by 1.25x.
#
# `killable` is preemptible and caps at 1 day. If preemption costs a run, --partition=gpu-b200 has a
# 5-day limit and is not preemptible (but is premium shared hardware -- do not default to it).
#
# No `source ~/.bashrc`: that file is empty here and conda/HF_* live in ~/.cshrc (tcsh), so this script
# sets everything explicitly and works from a fresh shell or a requeue.

set -e

WORK=/home/dcor/arielzoizner
REPO=$WORK/projects/creativity-measure
PY=$WORK/miniconda3/envs/creativity-measure/bin/python
RUN_DIR=$REPO/notebooks/flowmap_smc_flux
NB=flowmap_smc_flux.ipynb

# $WORK is group-readable, so the HF *credential* must not live there; the 32 G of weights may.
export HF_HOME="$WORK/.cache/huggingface"
export HF_TOKEN_PATH="/a/home/cc/students/math/arielzoizner/.hf_token"
export TORCH_HOME="$WORK/.cache/torch"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/.config/matplotlib"     # else matplotlib writes into the 2G $HOME quota
export TOKENIZERS_PARALLELISM=false

export STOP_AFTER="${STOP_AFTER:-run}"
export RUN_TAG="${RUN_TAG:-a3_1}"
# Most-important-first: every m costs the same and the notebook stops as soon as one does not fit the
# remaining budget, so whatever gets dropped should be the least informative value. m=3 is the headline
# comparison against the pCN ladder; m=1 is the mild baseline closest to p.
# NOTE: `sbatch --export` splits on commas, so a list CANNOT be passed that way -- `M_LIST=3,2,1` there
# parses as three separate assignments. Override with `export M_LIST=... ; sbatch ...` instead.
export M_LIST="${M_LIST:-3,2,1}"

mkdir -p "$WORK/logs" "$MPLCONFIGDIR"
cd "$RUN_DIR"          # the notebook derives REPO from cwd ("../..") and writes results here

echo "=== $(date) on $(hostname)  STOP_AFTER=$STOP_AFTER  M_LIST=$M_LIST  RUN_TAG=$RUN_TAG ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Preflight: fail in seconds, not after minutes of model loading. A baseline run once died here in 29 s
# on a torch+cu130 build against a 12.x driver -- exactly the class of failure this catches. `peft` is
# checked because diffusers' LoRA loader needs it and the flow map is delivered as a LoRA.
"$PY" - <<'PREFLIGHT'
import torch, peft
from huggingface_hub import auth_check
assert torch.cuda.is_available(), "no CUDA device visible"
gb = torch.cuda.mem_get_info()[1] / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)}  {gb:.0f} GB total   peft {peft.__version__}")
assert gb >= 40, f"need a >=40GB card for the bf16 transformer, got {gb:.0f} GB"
assert torch.cuda.is_bf16_supported(), "card lacks native bf16"
auth_check("black-forest-labs/FLUX.1-dev")
print("FLUX.1-dev access: OK")
PREFLIGHT

# The CPU-only suites must be green before any GPU time is spent. Seconds to run. Both matter:
# test_flowmap_smc covers the sampler, test_flux_flowmap covers the wiring this job is about to use
# (the DualTimeEmbedder attribute names, the 1-t convention flip, the pack/unpack round trip).
echo "=== phase 2 regression (CPU) ==="
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest \
    "$REPO/tests/test_flowmap_smc.py" "$REPO/tests/test_flux_flowmap.py" -q

# Separate output notebook, never --inplace: nbconvert writes once at the very end, so a kill during
# that write could truncate the source. --allow-errors so a mid-run failure still preserves the
# diagnostics and images already produced, with the traceback kept in the failing cell.
OUT="run_${STOP_AFTER}_${SLURM_JOB_ID:-manual}.ipynb"
echo "=== executing $NB -> $OUT ==="
"$PY" -m jupyter nbconvert \
    --to notebook \
    --execute \
    --allow-errors \
    --output "$OUT" \
    --ExecutePreprocessor.timeout=-1 \
    "$NB"

echo "=== done $(date) ==="
echo "notebook with outputs: $RUN_DIR/$OUT"
ls -l "$RUN_DIR"/*.pt "$RUN_DIR"/*"$RUN_TAG"*.png 2>/dev/null || true
