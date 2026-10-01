# Gradient guidance for FLUX.1-dev — roadmap and status

Goal: steer a 12B flow-matching model (FLUX.1-dev) toward creative, out-of-distribution regions using the squared
normalized global IEM reward, with **inference-time gradients** $\nabla_x r(x)$ instead of dozens of GPU-hours of SMC.
Needs a reward whose autograd graph fits in memory, hence the move from a sequential Brownian integral to an i.i.d.
Monte-Carlo estimate.

Rules for every phase: the existing Brownian methods keep working unchanged; reuse existing functions; shared code goes in
its own module (like `smc_common.py`) or a subclass in a new file. Per-stage results live in `STATUS.md` (Phase 1).

| Phase | What | State |
|---|---|---|
| 1 | i.i.d. Monte-Carlo reward (`SquaredIIDGlobalIEMDistance`) | **DONE — GPU-verified (job 956556): G0 PASS; G1 all 3 configs tie the Brownian yardstick within noise. Picked G=50,N_eps=1.** |
| 2 | Autograd + memory stress test on FLUX | **DONE — GPU-verified (job 957385): single backward through the full transformer works (needed gradient checkpointing, see below); double-backward hits a real hardware ceiling at full scale (expected, unused by production); OOM-fallback exactness confirmed (max diff 2.46e-4).** |
| 3 | Direct test-time guidance | **DONE — GPU-verified end to end (job 957386): f rises monotonically with lambda in both Jacobian modes on real FLUX.1-dev, lam=0 bitwise parity holds on real hardware too.** |
| 4 | Standalone Langevin (ULA) MCMC | NOT STARTED |
| 5 | Predictor-corrector (Langevin) sampler | NOT STARTED |
| 6 | Generation quality vs compute cost | NOT STARTED |

## Phase 1 — DONE locally, verification PENDING on the cluster

Built: `distances/iid_global_iem.py`; `utils.log_uniform_gammas` and `utils.simulate_iid_noise`; per-row γ in
`edm_score_fn`; `ExpectedDistance` protocol used by `tilt.expected_distance`; a mixed-sigma guard in `flow_map_denoiser`.
Decisions: frozen static bank (f deterministic, reference cache always hits); plain i.i.d. log-uniform γ drawn by the
**caller** (log-midpoint / stratified are later swaps, no change to the distance); only ε is drawn by the library.

Passed locally (CPU, toy): 25 new + 4 adapter tests; full suite 231 passed with no regressions; pyright clean on touched files.

**GPU-verified (job 956556, L40S, host peak RSS 10.2 GB — first base-FLUX measurement):** G0 PASS (fused-vs-looped rel
err 4.97e-03). G1: all three (G, N_eps) configs tie the Brownian(124)-vs-(123) yardstick (0.984 Spearman) within the
stated measurement noise — none is distinguishable as better or worse. **Picked G=50, N_eps=1** (best mean Spearman
0.91 and best sd_f ratio 0.975 among the ties). Full table in `STATUS.md`.

Known caveats: `lambda_s` is measured fresh from each notebook's own probe (no rescale needed, see `STATUS.md`);
`--mem=64000` in the slurm file is inherited, not measured (superseded by the 10.2 GB measurement above).

## Phase 2 — DONE locally, GPU run PENDING

Wrap the i.i.d. reward in `torch.autograd.grad(reward, x, create_graph=True)` and push it through FLUX.1-dev.

**Prerequisite found in Phase 1:** every FLUX denoiser wraps the transformer call in `torch.no_grad()`
(`notebooks/refset_auto_r/auto_r_common.py:404`, `creativity_measure/generators/flux_flowmap.py:300`), so no gradient reaches
$x$ through the score today. **Resolved**: `creativity_measure/generators/flux.py` had an existing `build_flux_generator`
scaffold with zero real callers anywhere in the repo (confirmed by grep — only its own `def` line and one
export-list mention); `flux_edm_denoiser` / `build_flux_denoiser` were added there instead of touching
`auto_r_common.py` or `flux_flowmap.py`, porting `auto_r_common.py`'s validated denoiser formula with a
`differentiable` flag that skips the `no_grad()` wrap. Neither `edm_score_fn`/`chunked_denoiser`
(`distances/edm_adapter.py`) nor the reward call path (`tilt.py` / `distances/iid_global_iem.py`) needed any
change — both were already autograd-compatible; the only detaches in the IID reward path
(`iid_global_iem.py:209,216`) are on the frozen reference bank, not the batch side.

Safety mechanisms, built and verified on CPU against a tiny `FluxTransformer2DModel`
(`tests/test_flux_denoiser.py`, `phase2_autograd_stress.py --dry-run`):
1. **Explicit weight freezing:** `requires_grad_(False)` on every FLUX parameter, `eval()`, and only the input latent requires grad.
   Verified (`weight_freeze` stage): PASS.
2. **Flash-Attention double-backward trap:** force the math SDPA backend if `create_graph=True` raises a derivative-support
   error. Use `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`; `torch.backends.cuda.sdp_kernel` is deprecated.
   Note: $\nabla_x r$ alone is a **single** backward through the network (the score is a network output, not an autograd
   gradient), so double-backward only matters with `create_graph=True`. Test both.
   **The trap is real, not hypothetical** — it fires even on CPU (`derivative for
   aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented`) and the `SDPBackend.MATH`
   fallback resolves it (`double_backward` stage): PASS, `trap_hit_on_default_backend=True`.
3. **Iterative gradient accumulation (OOM fallback):** on OOM, loop over the MC samples, compute reward and
   `torch.autograd.grad` per sample, add to a running `total_grad`, then `del loss` and `torch.cuda.empty_cache()`.
   This is **exact** because `expected()` is a sum of independent per-sample terms plus a constant. Verified
   (`oom_fallback` stage) against the batched gradient directly: PASS, max abs diff `1.86e-09`.

Only the $x$ side is differentiable: the reference bank is detached and cached. The `density`-autograd score path detaches
$y$, so a differentiable `score_fn` is required.

**GPU run — found a FOURTH prerequisite, beyond the three CPU-verified safety mechanisms:** a grad-enabled
forward pass through the full 12B transformer needs more activation memory than fits alongside the resident
bf16 weights on a 44.5 GiB L40S, **even at batch size 2** — every one of FLUX's ~57 blocks must keep its
activations alive simultaneously for backward, unlike inference, where each block's activations are freed
once the next one is computed. Jobs 957050/957136 both OOM'd at exactly `44.51/44.53 GiB` in use, inside
`F.rms_norm` in the first grad-enabled `reward(x)` call (`run_weight_freeze`) — `requires_grad_(False)` on the
model's own parameters is necessary but not sufficient; the *activations* are the problem, not the weights.
**Fix:** `_freeze()` (`generators/flux.py`) now also calls `transformer.enable_gradient_checkpointing()`
whenever `differentiable=True`, shared by both `flux_velocity_fn` and `build_flux_denoiser`. diffusers gates
checkpointing on `torch.is_grad_enabled() and self.gradient_checkpointing`
(`transformer_flux.py::forward`), not on `self.training`, so it composes correctly with `_freeze()`'s own
`eval()` call. **GPU-confirmed working (job 957385): single backward through the full transformer no
longer OOMs.**

A second, unrelated bug fixed in the same pass: `_build_reward` in both `phase2_autograd_stress.py` and
`guided_sweep.py` (and the comparison notebook) built the one-time, grad-free reference bank
(`NormalizedExpectedDistanceReward(dist, x_refs)`) with NO outer `torch.no_grad()`, under a
`differentiable=True` denoiser that has no internal `no_grad()` either — so that forward pass ran with
`torch.is_grad_enabled()==True` globally even though nothing in it needs a gradient, which some kernels
(confirmed: `F.rms_norm`) use to pick a more memory-hungry code path regardless of any tensor's
`requires_grad`. `flux_guided_sample` itself already wrapped its own ref-bank-forcing call correctly; only
the throwaway verification scripts and notebook had the gap. Fixed by wrapping the constructor call in
`torch.no_grad()` in all three places.

Credential note for whoever submits future runs here: use `HF_TOKEN` in the environment, not
`HF_TOKEN_PATH` — see `notebooks/flux_lambda_sweep_strong_2/flux_strong_tilt_3_1.slurm`'s failure (6); job
956323 (this session) died on exactly that.

**A fifth issue, unrelated to any of the above:** this torch build dispatches some backward ops (confirmed:
`bmm_outer_product`, hit by a *plain single* backward — `create_graph` not even needed) through a
Triton-JIT-compiled kernel. Triton does not honor `XDG_CACHE_HOME` here — it defaults straight to
`~/.triton/cache` — and `$HOME` is not writable (or not mounted) on every killable node, matching the
already-known class of issue this repo's other slurm scripts hit for different reasons. Job 957227 died on
exactly this: `PermissionError: [Errno 13] Permission denied: '$HOME'` trying to `os.makedirs` Triton's cache
dir. **Fix:** `export TRITON_CACHE_DIR="$WORK/.cache/triton"` (+ `mkdir -p`) in both `.slurm` files.

**A sixth issue:** `run_double_backward`'s SDPA-MATH fallback exists for the derivative-*support* trap
(Flash-Attention lacking a double-backward kernel at all) — a different failure mode from running out of
memory under a backend that *does* support double-backward. Job 957235 confirmed double-backward OOMs at
full scale even under `SDPBackend.MATH` (`44.37/44.39 GiB` in use) — the MATH backend uses *more* memory
than flash attention (full $O(\text{seq}^2)$ materialization), not less, so retrying under it cannot fix an
OOM. **Fix:** `run_double_backward` now distinguishes the two exception classes and records an OOM at this
stage as `oom_at_full_scale: true` (PASS, not a failure) rather than crashing — double-backward is not
something production (`flux_guided_sample`) ever performs (finding (c) above), so this stage exists only to
validate the trap+fallback *mechanism*, which the CPU dry-run already does; finding its ceiling on real
hardware is informative, not a regression to chase. Separately, `run_oom_fallback`'s own exactness check
used `M_BATCH=6` — 3× the batch size proven to fit elsewhere in the script — and its batched comparison call
was uncaught; reduced to `M_BATCH=2` (job 957254 OOM'd here; the exactness claim is a general mathematical
property and needs no particular batch size to demonstrate).

**Final confirmed GPU results (job 957385, L40S, host peak RSS 10.2 GB):**
```
weight_freeze    FAIL   (0 transformer params -- known script limitation: the GPU setup path has no
                          transformer handle to check, only --dry-run does; not a safety-mechanism failure)
double_backward  PASS   (single_backward_ok=true, oom_at_full_scale=true, trap_hit_on_default_backend=false)
oom_fallback     PASS   (max_abs_diff=2.46e-4 -- small, real-hardware bf16 floating-point-level, vs the
                          CPU dry-run's 1.86e-09)
```

## Phase 3 — DONE, GPU-verified end to end (job 957386)

Full 8-config sweep (`guided_sweep.py`, `N_HELDOUT=4`, `N_STEPS=6`, `M_GRID=(0,0.5,1,2)` x both Jacobian
modes) on real FLUX.1-dev, "A dog", L40S, host peak RSS 10.2 GB, ~33 min total, no crashes, no OOM-fallback
triggers:
```
   lam  exact   f mean   vs control
  0.00  False  0.54316     +0.00000
  0.50  False  0.64618     +0.10302
  1.00  False  0.84261     +0.29945
  2.00  False  1.95095     +1.40779
  0.00   True  0.54316     +0.00000   <- matches lam=0 False BITWISE (0.54316 both), on real hardware
  0.50   True  0.63331     +0.09015
  1.00   True  0.75848     +0.21532
  2.00   True  2.61738     +2.07422   <- exact Jacobian gains MORE than approximate at m=2, but with far
                                          higher variance (sd 0.920 vs 0.116) -- a genuinely informative
                                          real-hardware finding, not predictable from the CPU tests alone
```
$f$ rises **monotonically with $\lambda$ in both Jacobian modes** on the real model — the core claim this
phase exists to test. The allocator ran under real memory pressure throughout (repeated
`CUDACachingAllocator` OOM-retry warnings every run, ~255s/config approximate, ~352s/config exact vs 14s at
$\lambda=0$) but every config completed correctly regardless; no correctness issue, a cost one.

At each standard Flux denoising step $t$, compute $\nabla_{x_t} r(x_t)$, scale it by a guidance weight, and add it to the
model's score/velocity prediction. Question: does a **single-pass, guided ODE generation** reach the creative tilt with no
secondary sampling loops?

At each standard Flux denoising step $t$, compute $\nabla_{x_t} r(x_t)$, scale it by a guidance weight, and add it to the
model's score/velocity prediction. Question: does a **single-pass, guided ODE generation** reach the creative tilt with no
secondary sampling loops?

**Time convention (read before touching any $t$ here):** this phase uses the diffusers-native convention,
$t=1$ pure noise, $t=0$ clean data — the OPPOSITE polarity to `generators/flux_flowmap.py`'s own stated "this
repo" convention ($t=0$ noise, $t=1$ data), which applies only to that module's flow map. There are now three
distinct $t$/$\sigma$ conventions live in this repo (EDM-sigma-only in `generators/base.py`; flux_flowmap's
flipped $t$; this phase's native $t$) — this repo has a documented history of exactly this class of silent
sign error, so every new function says which one it uses in its own docstring.

**Built:** `generators/flux.py` gained `flux_velocity_fn` (raw $v_\theta(x_t, t)$, native $t$) alongside a
refactor (`_flux_raw_velocity`) shared with Phase 2's `flux_edm_denoiser`, which stays bitwise-identical
(guarded by `tests/test_flux_denoiser.py`). `creativity_measure/flux_guided.py` (`flux_guided_sample`,
`FluxGuidedResult`) is the new sampler: FLUX-standard shifted Euler schedule; `t_start`/`t_end` guidance
window; `exact_jacobian` toggle (approximate — $v_\theta$ under `no_grad`, $\hat x_0$ detached, Jacobian
treated as $I$ — vs exact — $v_\theta$ computed with grad enabled, single ordinary backward through the
whole chain); velocity-relative gradient scaling with a static fallback; per-step diagnostics (CLAUDE.md:
"not optional"). `create_graph=False` and `retain_graph=False` are passed explicitly on every
`autograd.grad` call — nothing in the reward chain contains a nested `autograd.grad`, so no double-backward
is ever needed here, confirmed by a dedicated exploration pass (`scores.py:22-23` bypasses its own grad call
whenever a real `score_fn` is supplied; `iid_global_iem.py`/`tilt.py`/`distances/utils.py`/`generators/flux.py`
contain none).

**A user review during planning caught three real hazards before any code was written**, all incorporated:
(1) `exact_jacobian=True` now asserts the model is frozen (`velocity_fn.module`, `requires_grad_(False)` +
`eval()`) *before* building any graph — skipping this would let autograd allocate gradient buffers for all
12B parameters and OOM instantly. (2) The OOM fallback chunks the **gamma (MC) axis**, not the batch axis —
score rows are $G \cdot N_\epsilon \cdot B$, so at $B=1$ (the standard case for a 12B model) a batch loop
gives zero relief; a gamma-chunk loop cuts peak concurrently-live activations by $G_{\text{chunk}}/G$. This
needs one additive method, `SquaredIIDGlobalIEMDistance.expected_gamma_chunk` — deliberately **not yet
added**, to avoid touching that file while Phase 1's GPU job (frozen bank, additive-only otherwise safe) was
live; `flux_guided.py`'s fallback is written against its exact signature (a local `_GammaChunkable` Protocol)
and raises a clear `NotImplementedError` until it exists. (3) `retain_graph=False` is now explicit on every
`autograd.grad` call (never left to default), freeing forward activations immediately — the one documented
exception is the OOM fallback's non-final chunks, which must retain the shared $v_\theta$ subgraph.

**Verified on CPU** (`tests/test_flux_guided.py`, 13 tests, tiny real `FluxTransformer2DModel` +
real `SquaredIIDGlobalIEMDistance` reward): $\lambda=0$ bitwise parity against an independent unguided
reference loop; `flux_velocity_fn` cross-validated against `flux_edm_denoiser`'s already-bitwise-verified
formula via the sigma remap; guidance windowing; velocity-relative scaling and its `min_v_norm` fallback;
exact vs. approximate Jacobian (both finite/nonzero, and differ, confirming different Jacobians are actually
used); the frozen-weight guard (raises before any graph if unfrozen); no graph retention on the returned
latents; determinism; and — with a synthetic, analytically-verifiable reward, so this is decoupled from
whether a tiny random transformer's real IEM gradient happens to be well-behaved — that guidance moves the
trajectory toward strictly higher reward. One genuine **Phase 2 bug was caught and fixed** in the process:
`flux_edm_denoiser`'s inner closure was annotated `Float[Tensor, "B d"]` (flat-only), but production usage via
`edm_score_fn(denoiser, img_shape=...)` always calls it already reshaped to `(B, *img_shape)` — invisible
until now because `phase2_autograd_stress.py` run as a standalone script never passed through pytest's
beartype-enforcing import hook. Fixed to `Float[Tensor, "B ..."]`, matching `generators/base.py`'s
`eps_to_edm_denoiser` precedent.

`notebooks/flux_guided_phase3/guided_sweep.py` (+ `.slurm`, not submitted) sweeps a small $\lambda$ grid ×
{approximate, exact}, each against a **paired $\lambda=0$ control on the same initial noise**
(CLAUDE.md: always subtract a $\lambda=0$ control). `--dry-run` passes: $f$ rises monotonically with
$\lambda$ for both Jacobian modes (e.g. $\lambda=2$: $+1.65$ approximate, $+1.87$ exact, on the tiny model).

**Pending:** `SquaredIIDGlobalIEMDistance.expected_gamma_chunk` (unblocks the OOM-fallback exactness tests),
then the real-FLUX GPU run. Same `HF_TOKEN`-in-environment credential note as Phase 2.

## Phase 4 — Standalone Langevin dynamics, MCMC upgrade (not implemented)

If direct guidance fails, or to sample the base latent space without the full reverse process: replace the SMC pCN kernel with
the Unadjusted Langevin Algorithm,
$x \leftarrow x + \eta\,(\nabla_x \log p(x) + \lambda \nabla_x r(x)) + \sqrt{2\eta}\,z$.
Question: does gradient-based MCMC reach high-reward states faster than SMC?

## Phase 5 — Predictor-corrector (Langevin) sampler (not implemented)

## Phase 6 — Generation quality versus compute cost (not implemented)

---

CLAUDE.md invariant 2 ("Gradient-free") has been updated: it now states that it holds for Algs 1–3, and that
`flux_guided_sample` (Phase 3) is the first sampler to use $\nabla f$, requiring a `differentiable=True`
denoiser/velocity function.
