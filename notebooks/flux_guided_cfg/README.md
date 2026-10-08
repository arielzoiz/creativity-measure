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
| 2026-10-08 | 999852 (`a-dog` A, w=1.0+3.0, t-806), 999853 (`car`, n-803), 999855 (`sofa`, n-801), 1000244 (`jacket`, t-806) | seed 1234 batch 1. `jacket` took two tries to place — see "Pinning vs. memory" below. |

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

20–90 min of silence before the first λ point is **normal** — setup is NFS-bound (32 GB of mmap'd
safetensors off `$WORK`; the checkpoint is fully cached there, so nothing downloads). Confirm with
`read_bytes` in `/proc/<pid>/io` via `srun --overlap` if unsure, never by assuming a hang.

## Findings

*(empty — the runs are in flight. Do not write anything here that was not read off the decoded images.)*
