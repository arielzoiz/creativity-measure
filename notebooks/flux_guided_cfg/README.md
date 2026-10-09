# Manual CFG × λ — does a stiffer base velocity field resist the reward gradient?

**Status (2026-10-08): RUNNING, no findings yet.** Seed 1234 batch 1 submitted; see the Run log.
The Findings section is deliberately empty — do not fill it from scalars alone, see "How to read a result".

## The question

FLUX.1-dev is guidance-distilled, and this repo's whole FLUX path runs at an embedded `GUIDANCE = 1.5`
baked into every reference bank, reward and base velocity. On top of that baseline we add traditional
two-forward classifier-free guidance,

$$v_{\text{CFG}}(x_{t}, t) = v(x_{t}, t \mid \text{null}) + w \cdot (v(x_{t}, t \mid c) - v(x_{t}, t \mid \text{null}))$$

and ask whether the sharper field it produces is a *stiffer* structure that holds its shape further up
the λ ladder before Phase 3's "deep-frying" sets in. $w = 1$ collapses the null terms and **is** the
Phase 3 baseline exactly.

## What is and is not varied

CFG drives the Euler **transport** only. $\hat{x}_{0} = x_{t} - t v_{\text{cond}}$ stays the model's true
conditional denoised estimate. Three consequences, all deliberate:

- the reward is never evaluated on a $w$-extrapolated point no reference bank covers, so **$f$ stays
  comparable across $w$**;
- `grad_scaling="velocity"` scales $\lambda g$ against $\lVert v_{\text{cond}} \rVert$, **not**
  $\lVert v_{\text{CFG}} \rVert$. CFG inflates the velocity norm, so scaling to the transport field would
  raise the applied gradient *with* $w$ and confound "a stiffer base field resists the gradient" with
  "the gradient got bigger". Here $w$ moves the base field and nothing else;
- only **one** autograd graph is ever built — the transport field is evaluated under `no_grad` and never
  differentiated — so `exact_jacobian=True` costs what it costs in Phase 3. CFG adds one batch-1 forward
  per step against the reward's ~50 denoiser rows, i.e. a few percent, not 2×.

The reward is **identical across $w$**: setup imports Phase 3's `_build_reward_and_lam_s` rather than
copying it, so the reference latents, gammas, eps draws and $\lambda_{s}$ are bit-identical to the Phase 3
and Phase 5 runs for the same prompt. That is the whole basis of the comparison.

## Code

| Where | What |
|---|---|
| `creativity_measure/generators/cfg.py` | `cfg_velocity_fn(v_cond, v_uncond, w, *, t_window)`. Backend-agnostic; returns `v_cond` **itself** at `w == 1.0`, since `v_u + 1.0*(v_c - v_u)` is not bitwise `v_c`. |
| `samplers/flow_guided_common.py` | `guided_euler_step(..., transport_velocity_fn=None)`. `None` is today's arithmetic bitwise; an identity-passed field is detected and skipped. Rationale lives on that parameter's docstring. |
| `samplers/flow_guided.py` | threads it through; adds `transport_v_norm_history`. |
| `tests/test_cfg_velocity.py` | 20 tests. The load-bearing ones: `w=1` returns the same object; `transport=None` is bitwise; a split transport moves the step but leaves `x̂₀`/reward/applied-norm untouched. |
| `cfg_w_sweep.py` | the sweep driver, structurally from Phase 5's `pc_sweep.py`. |
| `render_cfg_grid.py` | the figure; splices stored `flow_guided` runs in as the w=1 column. |
| `submit.sh` | `sh submit.sh <seed> <batch:1\|2>`. Header carries the full cost model and the batching rationale. |

**This is not a new sampler.** A CFG-extrapolated field is already a valid `VelocityFn`, so a
`flow_guided_cfg_sample` would have been a byte-for-byte copy of `flow_guided_sample`'s time loop. Per
CLAUDE.md's framing this is a velocity-contract contribution, not a sampling-algorithm one — and routing
it through `transport_velocity_fn` means `flow_guided_pc_sample` could take CFG for free later (not
wired: the corrector's score would need a decision about which field it reads).

## Preflight — five tier-1 checks, all raising

Four are inherited from Phase 5's `pc_sweep.py` (GPU model is L40S; $f(x_{\text{refs}}) = (R-1)/R$ to
`1e-3`; $\lambda_{s}$ within 5% of the per-prompt registry, **shared with Phase 5**; the `w=1` reduction
bitwise at λ=0 and recorded against the nondeterminism floor at λ≠0).

The fifth is new here and is the one that matters most: **the conditional and unconditional fields are
actually two fields.** If `encode_prompt("")` returned the conditional embeddings, or the two closures
got wired to the same conditioning, then $v_{\text{CFG}} = v_{\text{cond}}$ identically and **every $w$
would reproduce $w=1$**. The job would complete, cost its full GPU-hours, report plausible $f$, and
measure nothing — and nothing else in the job would notice. It records
$\lVert v_{c} - v_{u} \rVert / \lVert v_{c} \rVert$ per $t$, which is also the scale to pick a
`--cfg-t-window` from, since
$\lVert v_{\text{CFG}} - v_{\text{cond}} \rVert = (w-1) \lVert v_{\text{cond}} - v_{\text{uncond}} \rVert$.

## The w=1 baseline column

Not re-run for the five prompt-study prompts: they already have a stored `flow_guided` column on
**exactly** this λ lattice (0, 0.1, …, 1.0), same `n_steps=10`, same L40S, from the Phase 5 prompt study.
Re-running would cost ~17 GPU-h to reproduce images already on disk.

`"A dog"` is the exception and gets an **in-job** `--w 1.0`. Its stored Phase 3 runs sit on the 5.5/14
lattice — λ = **0.392857** and **0.785714**, not 0.4 and 0.8. Reading them as 0.4/0.8 implies a ~5% $f$
shift ($df/d\lambda \approx 6.4$ there, from $f$ = 1.8369 → 4.3377 between 0.7857 and 1.1786 at seed
1234), which is **larger than the backend's own ~2.5% nondeterminism floor**. Two extra guided points
(~0.6 GPU-h) buy an exact-λ, same-job, same-GPU baseline instead of a relabelled one.

The stored Phase 3 images still appear, on their own row: `render_cfg_grid.py`'s `_snap` places them in
the visually adjacent column and captions them with their **true** λ in red. Columns are defined by what
was actually swept, so a stored point that does not snap is dropped and reported rather than inventing a
column — CLAUDE.md records a cross-run image result that *reversed* once a mislabelled baseline was
corrected.

**`"A dog"` has a capital A.** It is Phase 3's module default and the key in
`../flux_guided_phase5/lam_s_by_prompt.json` (`"A dog": 81.016`). `"a dog"` would be a *different prompt*:
different embeddings → different reference latents → different $\lambda_{s}$ → nothing comparable to the
stored baseline, and the preflight would silently record a new registry entry rather than assert.

## How to read a result

**LOOK AT THE DECODED PNGs FIRST.** They are written per point the moment it completes. No scalar here
predicts recognizability (CLAUDE.md): $f$ is 14.2 on a clear dog and 14.8 on pure noise,
$\lVert x \rVert$ is *anti*-correlated with quality, and `hf_frac` has no absolute threshold across seeds
(0.0089 on a dog, 0.0093 on a destroyed mosaic). Phase 5 spent four hours on scalar comparisons pointing
the wrong way; five image reads reversed two conclusions.

`cfg_norm_ratio` = $\lVert v_{\text{CFG}} \rVert / \lVert v_{\text{cond}} \rVert$ over guided steps is the
one number saying how strong $w$ actually was at this prompt, in the field's own units. It is **nan at
λ=0 by construction** — with no gradient there is no $\hat{x}_{0}$, so the conditional field is never
evaluated and the ratio would be a tautological 1.0.

The claim to test is whether higher $w$ preserves recognizability at a λ where $w=1$ has broken down. If
$f$ rises but the images degrade, that is the same trade `eta_reference="score"` already lost in Phase 5
— say so plainly rather than reporting the $f$ gain alone. And remember the window is not a fixed λ
range: breakdown varies ~3× across $z_{0}$, so a single seed shows that $w$ moved *this trajectory*, not
that it moved the window.

## Design

    prompts   "A dog", sofa, teapot, jacket, building, car
    seeds     1234, 3141          (the two with documented opposite breakdown on dog)
    w         1.5, 2.0, 3.0       (+ 1.0 in-job for "A dog" only)
    lambda    0, 0.1, ..., 1.0    --lam-step 0.1 --lam-k 0,1,...,10
    fixed     n_steps=10, shift=3.0, exact_jacobian=True, N_PARTICLES=1, L40S

`n_steps=10` is **not** a tuning knob: more ODE steps *destroys* gradient guidance (CLAUDE.md —
`n_steps=19` gives abstract blocks where 10 gives a recognizable dog), because Phase 3's window depends
on discretization error as implicit regularization. It also cannot be used to build a compute-matched
control.

Launched **seed 1234 first, gated on the images**, then seed 3141. Full spec is 93.5 GPU-h over 12 jobs;
seed 1234 alone is ~48.3 GPU-h over 7.

## Run log

| date | jobs | what |
|---|---|---|
| 2026-10-08 16:47 | 999852 (`a-dog` A, w=1.0+3.0, t-806), 999853 (`car`, n-803), 999855 (`sofa`, n-801), 1000244 (`jacket`, t-806) | seed 1234 batch 1. `jacket` took two tries to place — see "Pinning vs. memory". |
| 2026-10-08 17:46 | 999852, 1000244 preempted on t-806, auto-requeued | ~1.8 GPU-h of setup lost, no results (both still in setup). 999853/999855 on separate nodes unaffected. See "Preemption is structural". |
| 2026-10-08 19:12 | 1001017, 1001019 (`a-dog`/`car` seed 3141) stuck in NFS RPC wait on n-801, 20 min / 0 bytes read | relocated via `--exclude=n-801` (1001059, 1001064). `D` state + `rpc_wait_bit_killable` + zero `read_bytes` distinguishes a genuine stall from `sofa`'s slow-but-advancing read on the same node — see "Pinning vs. memory". |
| 2026-10-08 21:04 | 1001012 (`a-dog` B, w=1.5+2.0) and 999853 (`car`, w=1.5/2.0/3.0) COMPLETED clean, seed 1234 | full data through λ=1.0 on both; see Findings. |
| 2026-10-08 23:01 | gate: launched 1001949–1001957, the 5 remaining seed-3141 jobs (`a-dog` B, `sofa`, `teapot`, `building`, `jacket`) | see "Gate decision" in Findings. |
| 2026-10-08 23:18 | 999855 (`sofa`, seed 1234) preempted mid-w=3.0, CUDA OOM warnings in the 90s before the kill (another job landing on the same GPU, not a leak), auto-requeued unpinned | **no results lost** — w=1.5 and w=2.0 arms both fully complete (11/11 each) on disk before the kill; only the in-flight w=3.0 point is repeated. |
| 2026-10-09 00:40 | 999855's requeue sat 42 min reading 343 MB (~0.14 MB/s, state `D`) sharing n-801 with two seed-3141 jobs | relocated via `scancel` + resubmit `--w 3.0` only (1002445, `--exclude=n-801`, landed n-803 immediately). Scoped to w=3.0 so the already-complete w=1.5/w=2.0 result files are untouched. The two seed-3141 jobs left in place on n-801 (`a-dog` B, `sofa`) — confirmed progressing at ~2.3 MB/s (13/32 GB), slow but not stalled; relocating would discard that read for no gain. |

### Pinning vs. memory — pin to spread, but memory decides

Jobs are pinned one per node via `NODES=`. This is placement, not an extra resource request, and it is
the direct fix for the **n-801 stall**: five of *our own* jobs on one node page-faulted the same 32 GB
checkpoint at a combined ~5 MB/s and produced nothing in 3 h, while a lone equally-cold job on n-803
finished setup in 40 min (`../flux_guided_phase5/PROMPT_STUDY.md`). It is contention, not cold cache —
hence batches of ≤4, with batch 2 submitted only once batch 1 is past setup.

**But a pinned job pends forever if its node cannot take it, and GPUs are not what decide that.**
`jacket` was first pinned to n-804 (999854), which filled between the `sinfo` check and the submission;
repinned to n-805 (999856), which pended on `(Resources)` *despite a free GPU and 90 idle CPUs*. The
reason, and the reusable lesson:

    sinfo -N -O "nodelist:10,cpusstate:18,memory:10,allocmem:10,gres:18,gresused:22" | grep l40s

n-805 showed `MEMORY=515600  ALLOCMEM=508752` — **6.8 GB of Slurm-allocatable memory left**, against this
job's `--mem=24000`. Only t-806 qualified cluster-wide (6 free GPUs, ~1.2 TB unallocated; it is a 1.5 TB
node where the others are 512 GB). This is CLAUDE.md's "memory, not GPUs, is what blocks these jobs",
with one refinement worth keeping: **`FREE_MEM` is the OS's figure and is misleading** — n-805 reported
`FREE_MEM=318400` while having 6.8 GB schedulable. Compare `ALLOCMEM` against `MEMORY`, never `FREE_MEM`.

Resubmitted **unpinned** as 1000244, which is strictly better once memory is the binding constraint:
Slurm places it on the one node that fits and will pick a better one if it frees first. It landed on
t-806 next to `a-dog` A — two jobs sharing a node during setup, which is mild against the five that
caused the measured stall. **Rule: pin to spread while nodes have headroom; drop the pin the moment a job
pends, and check `ALLOCMEM` before concluding anything about why.**

### Preemption is structural on L40S, so SETUP COST is the thing to attack

At 17:46:21 **both** t-806 jobs were killed in the same instant — `slurmstepd: error: *** JOB <id> ON
t-806 CANCELLED ... DUE TO PREEMPTION ***` — and auto-requeued (`Requeue=1 Restarts=1`, same job IDs, so
a watchdog survives). t-806 itself was healthy (`allocated`, reason `none`); a higher-priority job simply
took the GPUs. Cost: ~59 + ~49 min of setup. **Nothing else was lost** — neither had reached a λ point, so
no results file existed and the resume path correctly started fresh.

Two things this establishes.

**1. Co-location means CORRELATED preemption.** The pinning rationale above was about NFS bandwidth;
preemption is a second, independent reason to spread. The two jobs on separate nodes (n-803, n-801) were
untouched and ran straight through. Spread for both reasons.

**2. We cannot escape it, because L40S implies `killable`.**

    sinfo -N -n n-801,t-806 -o "%.10N %.22P %.12T"     # -> killable*, for every L40S node

The L40S nodes exist *only* in `killable` (`PreemptMode=REQUEUE`). The non-killable partitions this
account can reach — `gpu-b200`, `gpu-h200`, both 5-day — are **different GPU models**, and a GPU change
alone shifts $f$ by 16% of $\operatorname{std}_{p}(f)$ (job 697271), which is the same order as the effect
being measured. Every stored w=1 baseline is `NVIDIA L40S`. So moving partitions would mean re-running
every baseline, and preemption is simply the price of comparability.

**Therefore the leverage is in setup, not in scheduling.** Each requeue currently re-pays ~50 min of NFS
checkpoint read plus ~19 min of cold reference-bank build. Two optimizations already named in CLAUDE.md
would cut that, and this incident is the concrete argument for prioritizing them:
  - **persist the `score_bank`** (1.46 GB at R=64, keyed on the base process plus
    `gammas`/`num_eps`/`dist_seed`) — CLAUDE.md's "standing optimization"; note its warning that taking
    the model out of the chain for the scores is what broke job 697271;
  - **node-local staging of the checkpoint to `/tmp`** under `flock`, as the Algorithm 3 flowmap jobs do
    — with the documented caveats: verify the staged file *count*, not just that `model_index.json`
    exists, and beware a second job `rm -rf`-ing the first's in-progress 32 GB copy.
Neither was attempted mid-flight.

20–90 min of silence before the first λ point is **normal** — setup is NFS-bound (32 GB of mmap'd
safetensors off `$WORK`; the checkpoint is fully cached there, so nothing downloads). Confirm with
`read_bytes` in `/proc/<pid>/io` via `srun --overlap` if unsure, never by assuming a hang.

## Measured during the runs (model properties, not outcomes)

These are preflight/diagnostic measurements. They are **not** findings about whether CFG helps — that
question is answered only by the decoded images, below.

**The reward is exactly the stored one.** `λ_s` reproduced its registry value to **rel 0.00%** on every
prompt checked (`car` 106.874, `jacket` 70.376), and `f(x_refs) = 0.984375 = 63/64` exactly. Importing
Phase 3's `_build_reward_and_lam_s` rather than copying it did what it was supposed to: λ means the same
thing in the new CFG arms as in the stored w=1 column.

**The w=1 reduction holds on hardware.** Bitwise `True` at λ=0 on every job (no backward is taken there,
so it is genuinely assertable). At λ=1 with `exact_jacobian=True`, `car` measured cross-path 1.1708e+01
against a self-deviation floor of 1.2030e+01 — **ratio 0.97**, i.e. the two paths differ *less* than the
same function run twice at the same seed. Per CLAUDE.md that is the signature of backend nondeterminism,
not of a defect.

**CFG costs ~2%, not ~25%.** Measured 58.41 s/guided-step against Phase 3's 57.1 s/step, i.e. ~9.7 min
per λ point. The extra unconditional forward is ~0.25 s (8 batch-1 forwards in ~2 s during the
field-separation check) against a step dominated by the reward's ~50 denoiser rows at batch ≤24 plus the
backward. **This is the payoff of keeping $\hat{x}_{0}$ on $v_{\text{cond}}$**: one autograd graph, so the
second network evaluation is ~0.4% of a step rather than doubling it.

**The conditional and null fields differ almost only at t → 1, and the profile is prompt-independent.**
$\lVert v_{c} - v_{u} \rVert / \lVert v_{c} \rVert$:

| t | 0.95 | 0.75 | 0.50 | 0.25 | 0.05 |
|---|---|---|---|---|---|
| `car` | 27.9% | 3.8% | 5.0% | 5.3% | 4.9% |
| `jacket` | 18.5% | 4.3% | 4.7% | 5.6% | 5.1% |

The spike magnitude is prompt-dependent; the tail is a flat ~5% for both, at every t ≤ 0.75. Since
$\lVert v_{\text{CFG}} - v_{\text{cond}} \rVert = (w-1)\lVert v_{c} - v_{u} \rVert$ exactly, CFG's
displacement is concentrated in the **first two schedule nodes** (t = 1.0 and 0.964 under
`n_steps=10, shift=3.0`) and is a small constant elsewhere. Consistent with FLUX.1-dev being
guidance-distilled: its empty-prompt branch was never trained as a null, and the prompt only strongly
distinguishes the two fields where the latent is still essentially noise.

Two consequences, both hypotheses to check against images rather than conclusions:
  - `--cfg-t-window 1.0,0.9` should capture nearly all of the displacement while skipping the flat
    region — a cheap follow-up if the full-trajectory runs show an effect.
  - `cfg_norm_ratio` measured **1.0084 at w=1.5**, i.e. the transported field is 0.8% larger in norm.
    Small, because the displacement is largely orthogonal to $v_{\text{cond}}$. This does **not** make
    w=1.5 a no-op — a direction change compounds over 10 Euler steps — but it is a reason to expect the
    informative arms to be at the high-$w$ end.

## Findings

*(w=3.0 still in flight on `car`/`jacket`; `jacket` w=2.0 not yet reached. Everything below was read off
the decoded PNGs. CORRECTION 21:41: an earlier version of this section called the result "w=1.5 narrows
the window, falsified" based on `car`+`jacket` at w=1.5 only. `car`'s own w=2.0 arm then came in and
SURVIVED past where both w=1 and w=1.5 failed — the opposite direction. The result is NOT YET SETTLED;
read the per-weight table below, not a one-line verdict.)*

### Breakdown-λ per arm, seed 1234, `n_steps=10`, λ lattice 0…1.0

| prompt | w=1 | w=1.5 | w=2.0 | w=3.0 |
|---|---|---|---|---|
| `car` | breaks 0.9→1.0 | breaks 0.8→0.9 (**worse**) | **survives past 1.0** (still a car at λ=1.0, albeit glitchy) | in progress (λ=0.4 as of 22:09) |
| `jacket` | never breaks, ≤1.0 | breaks 0.7→0.8 (**worse**) | **survives past 1.0** (legible garment silhouette at λ=1.0, heavily stylized) | in progress |
| `A dog` | never breaks, ≤1.0 | — | — | **confirmed never breaks, ≤1.0** — COMPLETED (job 1000783, clean exit). Uninformative on breakdown: w=1 already survives the whole lattice here, so neither arm has anywhere to fail. $w=3$'s $\text{cfg\_norm\_ratio}$ is 1.04–1.05, ~5x w=1.5/2.0's ~1.01, confirming displacement scales with $(w-1)$ as the preflight field-separation measurement predicted. |

**22:09 update — w=2.0 beats w=1.5 on BOTH prompts tested, consistently.** `jacket` w=2.0 holds a legible
garment silhouette (collar, sleeves, a stylized back graphic) through λ=1.0, where w=1.5 had already
broken by λ=0.8. That is a real, replicated non-monotonicity in $w$ — two prompts, same direction —
independent of whether w=2.0 beats the untilted baseline. Caveat: `jacket`'s own w=1 ALSO never breaks in
0–1.0 (matches the repo's existing prompt-study finding for this prompt), so `jacket` cannot test
"w=2.0 beats w=1" the way `car` can — there is no baseline failure point to compare against. Only `car`
currently supports the plan's actual hypothesis (CFG beats no-CFG); `jacket` only supports
"w=1.5 is worse than w=2.0". `car`/`jacket` w=3.0 (in flight) are what distinguish a genuine dose-response
(w=1.5 bad, w≥2 good) from a one-off at w=1.5.

### 23:04 — `car`/seed 1234 COMPLETE through λ=1.0, all four weights. w=2.0 is the standout, not w=3.0.

| λ | w=1 | w=1.5 | w=2.0 | w=3.0 |
|---|---|---|---|---|
| 0.8 | intact | intact | intact | intact |
| 0.9 | intact (posterized) | **destroyed** | intact (stylized) | intact (clean) |
| 1.0 | **destroyed** | — | **intact** (glitchy but a car) | **destroyed** (abstract shapes, no car) |

Correcting a mid-run read: at λ=0.9 w=3.0 looked cleanest of all three CFG weights, and the text here said
so. λ=1.0 then came in broken — w=3.0's edge matches the untilted baseline's (0.9→1.0), not an
improvement at the top. **w=2.0 is the only weight that survives the full lattice** on `car`; w=3.0 is
better than w=1/w=1.5 at 0.9 but not better than w=1 overall; w=1.5 is worse than everything at every
λ ≥ 0.9. Not a "higher w is better" story — w=2.0 specifically wins here.

### 23:22 — `sofa` CONTRADICTS the "w=1.5 is a bad weight" read from `car`/`jacket`

`sofa`/seed 1234, w=1.5 and w=2.0 both complete to λ=1.0 (job 999855 was preempted/requeued starting
w=3.0, no points lost — see Run log). At λ=1.0, where the untilted baseline has already collapsed into
pure abstraction (bold red/navy bars, no furniture), **BOTH w=1.5 and w=2.0 are clearly a sofa** —
cushions, armrests, legs, throw pillows, legible at both weights.

So `w=1.5` is the worst weight on `car`/`jacket` and TIED-BEST on `sofa`. There is no single weight that
wins across prompts so far, and no clean monotonic trend. What IS consistent across all three prompts:
**some CFG weight beats the untilted baseline** (`car`→w=2.0; `jacket`→w=2.0; `sofa`→w=1.5 and w=2.0).
What is NOT yet supportable: a claim about *which* weight, or a dose-response in $w$. The "w=2.0 is the
standout, w=1.5 is anomalously bad" framing two sections up was written off two prompts and does not
survive a third — treat it as superseded, not as the finding.

### 23:26 — `jacket` COMPLETE through λ=1.0 (job 1000244, clean exit). w=2.0 confirmed standout, w=3.0 degrades badly here.

| λ | w=1 | w=1.5 | w=2.0 | w=3.0 |
|---|---|---|---|---|
| 0.8 | intact | destroyed | intact | intact |
| 1.0 | never breaks (≤1.0) | — | **intact**, legible silhouette + back graphic | **blurred**, barely a silhouette, zipper/pockets gone |

Unlike `car` (w=3.0 fine through 0.9, breaks only at 1.0), `jacket`'s w=3.0 is visibly degrading well
before 1.0 (f jumps 3.0→6.6 between λ=0.8 and 0.9). **w=2.0 is now 2-for-2 as best-or-tied-best** across
the three prompts with substantial data (`car`, `jacket`, `sofa`) — the first consistent cross-prompt
signal tonight, after w=1.5 flipped from worst (`car`/`jacket`) to tied-best (`sofa`). `sofa`'s own w=3.0
arm was interrupted by the 23:18 preemption and is requeued; it is the single most informative remaining
point for telling "sofa's tie was real" from "sofa's w=3.0 would also have lost, like car/jacket's did at
the very top".

### 23:28 — first seed-3141 data: `A dog`/3141, w=3.0 beats w=1 at λ=0.8, both break by 1.0

Seed 3141 is the DOCUMENTED fast-breaking seed for this prompt (repo finding: dies by λ≈1.0 vs seed
1234's ≈2.4), so unlike seed 1234's `A dog` run (never broke, uninformative) this one has a real edge to
test. Job 1001059, clean exit, w=1.0 and w=3.0 both complete to λ=1.0:

| λ | w=1 | w=3.0 |
|---|---|---|
| 0.8 | intact but noticeably off-prompt (drifted into a rabbit/human-hybrid face, not really "a dog") | **intact, clearly a photorealistic dog, no distortion** |
| 1.0 | destroyed, pure abstract shapes | destroyed, different abstract pattern |

Second prompt (after `car`) where a higher CFG weight measurably delays visible degradation relative to
the untilted baseline — replicates the direction, though `A dog`/3141 only has w=1 and w=3 (no w=1.5/2.0
run), so it cannot say whether w=2.0 would have done even better here, the way it did on `car`/`jacket`.

### 00:16 — `teapot`/1234 complete (job 1000940, clean exit): NO EDGE in this λ range, at any weight

All four weights (w=1, 1.5, 2.0, 3.0) are still recognizably a teapot at λ=1.0 — spout, handle, lid knob
all legible, heavily stylized but structurally intact throughout. Joins `A dog`/1234 as a prompt where
0–1.0 simply does not reach the breakdown point, at ANY weight tested. **Do not count this as a fifth
"CFG beats baseline" data point** — there is no baseline failure to beat here, the same caveat as `A dog`
above.

### Scorecard so far (cells with an actual edge to test)

| prompt | seed | result |
|---|---|---|
| `car` | 1234 | w=2.0 best (survives to 1.0); w=1.5 worst (breaks 0.8→0.9) |
| `jacket` | 1234 | w=2.0 best (legible at 1.0); w=1.5 worst (breaks 0.7→0.8); w=3.0 also degrades by 1.0 |
| `sofa` | 1234 | w=1.5 ≈ w=2.0 tied-best (both survive to 1.0, baseline breaks by 1.0); w=3.0 re-running after preemption |
| `A dog` | 3141 | w=3.0 beats w=1 at λ=0.8 (only w=1,3 tested on this prompt/seed) |

No edge in range (uninformative on breakdown, not a null result for CFG): `A dog`/1234, `teapot`/1234.
Pending: `building`/1234, all five seed-3141 prompts other than `A dog`.

Of four cells with a genuine edge and full weight coverage, three point to **w=2.0 as best-or-tied-best**
(`car`, `jacket`, `sofa`); `A dog`/3141 only has w=1,3 so cannot confirm or deny w=2.0 there specifically,
but does confirm the direction (higher w delays breakdown vs. the untilted baseline).

### 00:19 — `building`/1234 complete (job 1000944, clean exit): ALSO no edge in range

Checked the strongest weight (w=3.0) at λ=1.0: still a clean, fully intact building facade illustration.
Third prompt with no breakdown anywhere in 0–1.0 at any weight, joining `A dog` and `teapot`. **Half of the
six seed-1234 prompts (`A dog`, `teapot`, `building`) simply never reach breakdown in this λ range at any
weight** — a property of the λ lattice (inherited from Phase 5's prompt study), not a result about CFG.

### SEED 1234 COMPLETE (all 7 jobs). Final scorecard:

| prompt | edge in 0–1.0? | result |
|---|---|---|
| `car` | yes | w=2.0 best (survives to 1.0); w=1.5 worst (breaks 0.8→0.9) |
| `jacket` | yes | w=2.0 best (legible at 1.0); w=1.5 worst (breaks 0.7→0.8); w=3.0 also degrades by 1.0 |
| `sofa` | yes | **all three CFG weights beat baseline** — w=1.5, w=2.0, w=3.0 all intact to 1.0, baseline destroyed by 1.0 |
| `A dog` | no | w=1 and w=3 both intact throughout |
| `teapot` | no | all four weights intact throughout |
| `building` | no | all four weights intact throughout |

**Of the three prompts where the question could be asked at all, all three favor w=2.0** as best or tied
for best. No prompt favors w=1.0 (the no-CFG baseline) or shows CFG making things uniformly worse. w=1.5
is the one inconsistent weight — worst on two prompts, tied-best on the third.

### 01:05 — `car`/3141 complete (job 1001064, clean exit): replicates the direction, via a DIFFERENT failure mode

Seed 3141 breaks much earlier on `car` than seed 1234 did, and via **prompt drift** rather than visual
collapse: at λ=0.7 the untilted baseline renders a cartoon anime girl — not a car at all, coherent image,
wrong subject — the same failure mode seen on `A dog`/3141's λ=0.8 rabbit/human-hybrid face. By λ=0.8 it
has also become pure abstraction.

w=2.0 at λ=0.7 is still clearly car-related (an engine-bay/headlight assembly, line-art style) — on-prompt
where the baseline had already drifted off it. w=2.0 itself drifts off-prompt by λ=1.0 (a stylized human
figure with hat and sunglasses), so it is not immune, only delayed by roughly one λ step.

**This replicates the DIRECTION (CFG at elevated w delays breakdown) via a mechanism distinct from what
`car`/`jacket`/1234 showed** (there, failure was legibility collapsing into visual noise; here it is the
subject itself drifting to something else entirely while staying figuratively coherent). Breakdown is not
one phenomenon, and "CFG delays breakdown" may need to be read as "delays whichever failure mode this
particular trajectory would have hit first," not a single mechanism.

### 02:27 — `sofa`/1234 COMPLETE across all four weights (job 1002445, clean exit): ALL THREE CFG weights beat baseline

| λ | w=1 | w=1.5 | w=2.0 | w=3.0 |
|---|---|---|---|---|
| 1.0 | **destroyed** (abstract red/navy bars) | intact | intact | **intact** (armrests, cushions, legs, textured upholstery) |

Supersedes the earlier "w=1.5 ≈ w=2.0 tied-best" framing, written before w=3.0 finished: it is a
**three-way tie**, not two. Cleanest result of the night — on this prompt, EVERY CFG weight tested beats
the untilted baseline, not just one.

### 03:18 — `jacket`/3141 w=2.0 arm complete: cleaner than baseline, third seed-3141 prompt replicating the direction

At λ=1.0, `jacket`/3141's untilted baseline is intact but degraded (graffiti-like noise overlaying a
legible collar/sleeve/placket). w=2.0 at the same λ is markedly cleaner — a crisp line-art illustration,
no noise overlay, same garment details more clearly resolved. Third prompt on seed 3141 (with `A dog`,
`car`) where elevated $w$ matches-or-beats the baseline; no seed-3141 prompt has yet contradicted the
direction. `jacket`/3141's own baseline never collapses outright at this λ (consistent with `jacket`
being a resilient prompt on seed 1234 too), so this is a "cleaner," not a "survives vs. destroyed," case.

### 04:31 — `teapot`/3141 (job 1001955, clean exit): FIRST CLEAN COUNTEREXAMPLE — both w=2.0 and w=3.0 break EARLIER than baseline

`teapot`/1234 never broke in 0–1.0 at any weight (no-edge prompt). `teapot`/3141 DOES break, confirming
3141 is genuinely the faster-breaking seed here too — and gives this prompt a real edge for the first
time.

At λ=0.8 the untilted baseline is still a clean, photorealistic metal teapot (spout, handle, lid knob all
legible). **Both w=2.0 and w=3.0 are ALREADY BROKEN at the same λ** — unstructured blurry blobs, no
teapot, no coherent alternative subject either (unlike `car`/3141's drift into a different coherent
subject, this is pure collapse). By λ=1.0 all three (w=1, 2, 3) have failed, but w=1 at least drifted to
something face-like/structured while w=2/w=3 are formless.

**This reverses the direction every other prompt showed tonight.** On `car`, `jacket`, `sofa` and `A dog`,
elevated $w$ matched-or-beat the baseline at every matched λ checked. Here, at a matched λ where the
baseline still works, BOTH CFG weights have already failed. Not a case of "everything fails and CFG fails
slightly worse" — a genuine λ gap where no-CFG succeeds and CFG does not.

**This is real counter-evidence, not noise to explain away.** Whatever mechanism makes CFG delay
breakdown on most prompts (the module docstring's "stiffer base field" hypothesis, or possibly something
about the specific composition each (prompt, w, seed) trajectory lands on) does not hold universally, and
`teapot`/3141 is the cleanest demonstration that it can go the other way just as sharply.

### 04:44 — `building`/3141 (job 1001956, clean exit): w=3.0 breaks, w=2.0 holds — a per-prompt CEILING, not a uniform reversal

Like `building`/1234, the untilted baseline never breaks in 0–1.0 here either (clean facade at λ=1.0,
confirmed on both seeds now). But unlike `building`/1234 (all three CFG weights also held), **w=3.0 is
broken at λ=1.0 on seed 3141** — an abstract geometric/architectural pattern, no recognizable building —
while **w=2.0 at the same λ is a clean, intact stylized cityscape** (skyscrapers, windows, clouds).

Read alongside `teapot`/3141 (where even w=2.0 already failed where baseline held), this looks less like
"CFG is unreliable" and more like **each (prompt, seed) composition has its own tolerance ceiling for how
much CFG extrapolation it can absorb before the extrapolation itself becomes destructive** — `teapot`/3141
has a low ceiling (below w=2.0), `building`/3141 a higher one (between w=2.0 and w=3.0), and `car`/`jacket`
/`sofa`/1234 apparently higher still (above w=3.0, at least at the λ values checked). This reframes the
counterexample from "CFG sometimes hurts across the board" to "there is a per-case $w$ beyond which CFG
hurts, and it is not the same $w$ for every prompt or seed" — which is a harder result to act on (no
single safe $w$) but a more precise one.

### 04:47 — `A dog`/3141 complete across all four weights (job 1001949, clean exit): clean 4-way replication, w=2.0 again best

At λ=0.8 (w=1 already off-prompt, w=3 already confirmed clean — see above): **w=1.5** is on-prompt but
textured/noisy around the fur; **w=2.0** is the sharpest and cleanest of all four — a crisp photorealistic
dog face. Every CFG weight beats the untilted baseline here, matching `car`/`jacket`/`sofa` on seed 1234,
and w=2.0 again stands out as best — now 4-for-4 as best-or-tied-best across every prompt with full weight
coverage (`car`, `jacket`, `sofa`, `A dog`/3141).

### Gate decision: LAUNCHED the remaining 5 seed-3141 jobs at 23:01

Per standing instruction ("read images, auto-launch if positive"). Justification at the time: w=1.5
underperforming the baseline on two prompts, while w=2.0 (both prompts) and w=3.0 (`car`, up to λ=0.9)
outperformed it — real, non-monotonic, replicated structure, not noise. `jacket`/`A dog` w=3.0 were still
in flight; `car`'s final λ=1.0 cell (above) landed minutes after the launch decision and refines but does
not reverse it. Jobs: `a-dog` B, `sofa`, `teapot`, `building`, `jacket`, all at seed 3141, w ∈
{1.5, 2.0, 3.0} (`a-dog` B: 1.5, 2.0 only — its 1.0/3.0 arm already ran as part of batch 1's early start).

**Not monotonic in $w$, and not yet a settled direction.** On `car`, w=1.5 hurt and w=2.0 helped, at the
SAME λ (0.9) and the same seed. Candidate explanations, none yet distinguished: (a) CFG's effect on
robustness is genuinely non-monotonic in $w$; (b) w=1.5 is a bad draw for this particular seed/prompt
composition and w=2.0 a good one, i.e. this is composition variance, not a $w$ effect; (c) plain
seed-to-seed-style variance at $n=1$ per cell — there is no measurement yet of run-to-run spread at fixed
$w$. `jacket` w=2.0/3.0 and `a-dog`'s w=3.0 panel (in flight) are what will distinguish these.

Note what no cell comparison can be read as alone: the arms produce different compositions at every λ,
including λ=0 with no reward applied — w=1.5 alone turns `car`'s vintage Beetle into a modern sports car
— because CFG diverts the trajectory from the first node, where the field separation is ~28%. **The
readout is breakdown-λ per arm**, per the table, never a single matched-λ image pair.

### `hf_frac` is ANTI-correlated with recognizability here — 3 for 3

| cell | image | `hf` |
|---|---|---|
| `car` λ=0.9 w=1 | recognizable car | 0.0970 |
| `car` λ=0.9 w=1.5 | **destroyed** | **0.0649** |
| `jacket` λ=0.8 w=1.5 | **destroyed** | **0.0125** (lowest in its row) |

Every time, the *ruined* image scored lower. This session initially cited `hf` at λ=0.8 as corroborating
a favourable reading of w=1.5; opening λ=0.9 reversed it. CLAUDE.md already warns `hf` has no absolute
threshold across seeds — this extends that: it is not reliable as a **within-prompt, within-λ, cross-arm**
ranking either, which was the one use previously thought safe. `f` is no better: it was *higher* on the
destroyed w=1.5 cell (2.53 vs 2.47). **Nothing but the image answers the question.**
