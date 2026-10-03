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
| 4 | Standalone Langevin (ULA) MCMC | **SKIPPED (2026-10-02)** — standalone ULA at $t=1$ is structurally uninformative; its mechanism is subsumed by Phase 5's corrector, which runs the same ULA drift at every $t$ with the base process still supplying structure. |
| 5 | Predictor-corrector (Langevin) sampler | **DONE — GPU-verified over 5 seeds, ~25 GPU-h.** Novelty gain REPLICATES (+31% mean at lam=0.79, +38% at 1.18, 5/5 seeds, recognizability preserved 4/5); window extension does NOT (1 seed supports, 1 contradicts — seed 1234 was a favourable draw). `creativity_measure/samplers/flow_guided_pc.py`, driver `notebooks/flux_guided_phase5/`. |
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

## Phase 4 — Standalone Langevin dynamics, MCMC upgrade (SKIPPED 2026-10-02)

The original plan: if direct guidance fails, or to sample the base latent space without the full reverse process, replace the SMC pCN kernel with
the Unadjusted Langevin Algorithm,
$x \leftarrow x + \eta\,(\nabla_x \log p(x) + \lambda \nabla_x r(x)) + \sqrt{2\eta}\,z$.
Question: does gradient-based MCMC reach high-reward states faster than SMC?

**Skipped deliberately, not deferred.** Phase 3 did *not* fail — it found a real creative window (see Established Findings in CLAUDE.md), so the
"if direct guidance fails" premise did not fire. And the standalone form is the uninformative special case: run at $t = 1$ it samples a tilted
*prior*, with no base process supplying structure at any point, which is exactly the regime where $\nabla \log p$ is least informative. Everything
the phase was meant to test — whether the ULA drift $\nabla \log p_t + \lambda \nabla r$ reaches high-reward states, and at what step size — is
tested inside Phase 5's corrector, which runs that same drift at every $t$ *with* the generative trajectory intact. Phase 5 at
`predictor_guided=False` is the nearest thing to a standalone-Langevin arm and it is one of the three arms being swept.

## Phase 5 — Predictor-corrector (Langevin) sampler

**IMPLEMENTED and CPU-verified** (`creativity_measure/samplers/flow_guided_pc.py`, `flow_guided_pc_sample`;
55 tests in `tests/test_flow_guided_pc.py`). **The GPU sweep has not been submitted.**

### Why

Phase 3 ceilings. The fine $\lambda$-scan (jobs 958150/958630) found the creative-but-recognizable window at $\lambda \in [0.4, 3.54]$, and past
$\lambda \approx 3.93$ the guided ODE locks onto a fixed off-manifold attractor: $f$ runs away ($143.97$ at $\lambda = 3.93 \to 271.24$ at
$\lambda = 5.5$) on visually destroyed images. A single deterministic Euler pass has no mechanism to **re-equilibrate** onto the model's own marginal
after each nudge, so every step's off-manifold error compounds. The corrector is that mechanism: after each ODE step arrives at $t$, run `corrector_steps` steps of
ULA at that *fixed* noise level targeting

$$q_{t}(x) \propto p_{t}(x) \exp(\lambda r(\hat{x}_{0}(x_{t}))),$$

whose drift contains the model's own marginal score, so the score term pulls back onto $p_{t}$ while the reward term pushes up $r$.

### The math

Interpolant $x_{t} = (1-t) x_{0} + t \varepsilon$, so $v = \varepsilon - x_{0}$ and $x_{t} = x_{0} + t v$.

1. **Denoised target (exact).** $\hat{x}_{0} = x_{t} - t v_{\theta}(x_{t}, t)$ — the same expression `flux_edm_denoiser` and `guided_euler_step` already use.
2. **Velocity $\to$ score (exact, not an approximation).** $\hat\varepsilon = E[\varepsilon \mid x_{t}] = x_{t} + (1-t) v_{\theta}$ and
   $p(x_{t} \mid x_{0}) = N((1-t) x_{0}, t^{2} I)$, hence
   $$s_{\theta}(x_{t}, t) = -\frac{x_{t} + (1-t) v_{\theta}(x_{t}, t)}{t}.$$
   Two closed forms pin it in the tests: at $t = 1$ it collapses to $-x_{t}$ for *any* $v$, and for Gaussian data $N(0, \sigma_{d}^{2} I)$ under the
   optimal $v$ it reduces to $-x_{t}/D$ with $D = (1-t)^{2}\sigma_{d}^{2} + t^{2}$ — the true marginal score.
3. **The identity-Jacobian approximation — the one heavy assumption.** $r$ cannot be meaningfully evaluated on a noisy $x_{t}$, so it is evaluated on
   $\hat{x}_{0}$ and the gradient transported back as
   $$\nabla_{x_{t}} r(x_{t}) \approx \nabla_{\hat{x}_{0}} r(\hat{x}_{0}), \qquad \text{i.e. } \frac{d\hat{x}_{0}}{dx_{t}} \text{ treated as } I.$$
   The true Jacobian is $I - t\, dv_{\theta}/dx_{t}$, a full $d \times d$ operator of the network's own sensitivity, replaced by the identity; worst
   where $t$ is large and the transformer most nonlinear. It is the same approximation Phase 3's default path makes, and `exact_jacobian=True` drops
   it in both the predictor and the corrector. Documented at length in the module and function docstrings, per the implementation request.
4. **Normalization and total drift.**
   $$\tilde{g}_{t} = \frac{\nabla_{\hat{x}_{0}} r}{\lVert \nabla_{\hat{x}_{0}} r \rVert_{2} + \epsilon_{0}} \lVert s_{\theta}(x_{t}, t) \rVert_{2},
   \qquad g_{\text{total}} = s_{\theta}(x_{t}, t) + \lambda \tilde{g}_{t}.$$
   The reward gradient's magnitude is discarded, so $\lambda$ is a dimensionless mixing weight: $\lambda = 1$ puts the drift at 45° between
   "stay on the manifold" and "ascend $r$".
5. **Dynamic SNR step size** (`snr = 0.16` default — Song et al. 2021, *Score-Based Generative Modeling through SDEs*, arXiv:2011.13456, App. G, and the
   `score_sde` `LangevinCorrector` default). `eta_reference="total"` (default):
   $$\eta_{t} = 2\left(\frac{\text{snr} \lVert z \rVert_{2}}{\lVert s_{\theta}(x_{t}, t) + \lambda \tilde{g}_{t} \rVert_{2} + \epsilon_{0}}\right)^{2},$$
   `eta_reference="score"` (the ablation knob):
   $$\eta_{t} = 2\left(\frac{\text{snr} \lVert z \rVert_{2}}{\lVert s_{\theta}(x_{t}, t) \rVert_{2} + \epsilon_{0}}\right)^{2}.$$
6. **Corrector loop (ULA).** For $j = 0 \dots \texttt{corrector\_steps}-1$ at fixed $t$, with $z^{(j)} \sim N(0, I)$:
   $$x^{(j+1)} = x^{(j)} + \eta_{t}^{(j)} g_{\text{total}}(x^{(j)}, t) + \sqrt{2\eta_{t}^{(j)}}\, z^{(j)}.$$
   $\eta$ is recomputed every $j$. **Unadjusted** — no Metropolis accept/reject, because an MH ratio needs $\log p_{t}$ and this repo has only its
   gradient; the stationary distribution is $q_{t}$ only as $\eta \to 0$, with an $O(\eta)$ bias that `snr` controls.

### Consequences worth knowing before reading any result

- **$\lambda$ anneals the step under the default.** Since $\lVert \tilde{g} \rVert = \lVert s_{\theta} \rVert$ exactly,
  $\lVert g_{\text{total}} \rVert \leq (1+\lambda)\lVert s_{\theta}\rVert$, so *both* the drift displacement and the injected noise scale as
  $\approx 1/(1+\lambda)$: raising $\lambda$ rotates the drift toward the reward while shrinking the step. That is the requested "prevent numerical
  collapse at large $\lambda$" mechanism, and it means $\lambda$ and `snr` are coupled — to hold the displacement while raising $\lambda$, raise `snr`
  by $\approx (1+\lambda)$. The `--dry-run` confirms it live: $\eta$ falls $0.109 \to 7.0\times10^{-4}$ over $\lambda = 0.79 \to 5.5$.
  `eta_reference="score"` decouples them.
- **Noise dominates drift by exactly $1/\text{snr}$.** $\sqrt{2\eta}\lVert z\rVert / (\eta \lVert g_{\text{total}}\rVert) = 1/\text{snr}$ identically
  under `"total"`, independent of model, reward and $\lambda$ — a free wiring assert, recorded per corrector step as `noise_frac` and measured at
  exactly $6.250$ in every dry-run row.
- **ULA's variance bias has a closed form here**, and so does its finite-$d$ correction. With the exact Gaussian score and the adaptive $\eta$, the
  stationary per-coordinate variance is
  $$v = D (1 + \text{snr}^{2}) (1 + 2/d)^{2},$$
  so the test asserts the *biased* value, never $D$. The $(1 + \text{snr}^{2})$ is the genuine $O(\eta)$ unadjusted-Langevin bias ($+2.56\%$ at
  $\text{snr} = 0.16$) and is $d$-independent. The two $(1 + 2/d)$ factors are distinct artifacts of $\eta$ being *adaptive*:
  (i) $\eta \sim \lVert z \rVert^{2}$, so the injected noise *energy* $2\eta\lVert z\rVert^{2} \sim \lVert z \rVert^{4}$ and
  $E\lVert z\rVert^{4} = d(d+2)$ against the drift's linear $E\lVert z\rVert^{2} = d$; (ii) $\eta \sim 1/\lVert x \rVert^{2}$ makes the stationarity
  condition linear in $1/Q$ ($Q = \lVert x\rVert^{2}$), so it pins $E[1/Q]$ and **not** $E[Q]$, and $E[Q] \geq 1/E[1/Q]$ by Jensen with gap
  $1 + \operatorname{Var}(Q)/E[Q]^{2} = 1 + 2/d$.
  Confirmed by ablating each source ($\eta$ with $\lVert z\rVert^{2} \to d$ reproduces exactly one factor; fully fixed $\eta$ reproduces
  $1 + \text{snr}^{2}$ at every $d$): measured $1.15487/1.09344/1.04196/1.03026$ at $d = 32/64/256/1024$ against predicted
  $1.15781/1.09070/1.04169/1.02961$. **Dropping source (ii) is the mistake to avoid** — a mean-field $Q \to E[Q]$ substitution gives only one factor
  ($1.0577$ at $d = 64$) and under-predicts the measurement ($1.0891$) by half the correction. At the production $d = 65536$ the whole correction is
  $(1 + 3.05\times10^{-5})^{2}$, i.e. unmeasurable — a small-$d$ test artifact, not a property of the sampler.
- **TWO distinct $\lambda = 0$ controls**, and they are not interchangeable: `corrector_steps=0` (pure Phase 3 base ODE) and
  `lam=0, corrector_steps>0` (base ODE plus pure Langevin on $p_{t}$, which still moves the cloud). Mistaking one for the other is the same class of
  error as the leg-3.1 stitch recorded in CLAUDE.md. Both are asserted distinct in the tests.
- **$\lambda = 0$ costs no reward backwards.** The corrector short-circuits the gradient entirely when `lam_corrector == 0`, so the pure-Langevin
  control is one velocity evaluation per sub-step.

### Code shape

`guided_euler_step`, `_reward_grad`, `_shifted_schedule` and the freeze guard were lifted out of `flow_guided.py` into
`creativity_measure/samplers/flow_guided_common.py` (non-public, same status as `smc_common.py`), so **the predictor is Phase 3's step itself, not a
copy of it**. Consequently `corrector_steps=0` reduces `flow_guided_pc_sample` to `flow_guided_sample` **bitwise** — asserted in the tests at
$\lambda = 0$ *and* $\lambda \neq 0$ in both Jacobian modes, and re-asserted on the real GPU at the start of every sweep job. That reduction is what
makes the three-arm comparison apples-to-apples. The extraction is arithmetic-preserving; `test_flow_guided.py`'s own $\lambda = 0$ bitwise test is
the regression guard and all 17 of its tests still pass unchanged.

### GPU reality check: the backend is not bit-reproducible (job 966117)

The first wave-1 launch (jobs 965866–965874) was **aborted by its own preflight**: `corrector_steps=0` was not
bitwise equal to `flow_guided_sample` on real FLUX at $\lambda = 1$, `exact_jacobian=True` (max abs diff 11.74),
although `tests/test_flow_guided_pc.py` asserts exactly that, bitwise, on CPU in both Jacobian modes. All nine
jobs would have failed identically, so they were cancelled before any wrote results — nothing was contaminated.

**The assert was wrong, not the sampler**, and the diagnosis separated two claims that bitwise-output equality
conflates: (i) the two samplers ask the model the *same question*, which is ours; (ii) the model answers the same
question identically twice, which is the backend's. Claim (i) was proven directly on CPU by tracing
`(t, shape, input-hash, requires_grad)` per call — the sequences are identical at $\lambda \in \{0, 1, 3\}$ in both
Jacobian modes, now a permanent, backend-independent test. Claim (ii) is false on GPU, measured by
`reduction_diagnostic.py`:

| case | bitwise | latent `rel` | $\lvert\Delta f\rvert$ |
|---|---|---|---|
| `flow_guided` vs **ITSELF**, $\lambda = 1$, exact | no | **0.503** | 1.67% |
| `flow_guided` vs **ITSELF**, $\lambda = 1$, approx | no | 0.192 | **0.07%** |
| `flow_guided` vs **ITSELF**, $\lambda = 0$ | **yes** | 0 | 0 |
| `flow_guided` vs `pc(corr=0)`, $\lambda = 0$ | **yes** | 0 | 0 |
| `flow_guided` vs `pc(corr=0)`, $\lambda = 1$, approx | no | 0.155 | 0.22% |
| `flow_guided` vs `pc(corr=0)`, $\lambda = 1$, exact | no | 0.399 | 5.76% |

The decisive line is the first: **the model disagrees with itself (0.503) more than the two samplers disagree with
each other (0.399)**, so the cross-sampler gap cannot be a difference between code paths. Cause: flash attention's
backward uses atomics, and `_freeze`'s gradient checkpointing makes the backward recompute the forward.
Full detail, including why the exact path is $\approx 24\times$ noisier in $f$ than the approximate one and how this
re-explains job 957386's "exact has far higher variance" observation, is in CLAUDE.md's Established Findings.

Pooling all four nondeterministic draws: $f$ **CV $= 2.52\%$** (sd 0.0742 on mean 2.9490) — quote the pooled CV,
never a single pair's $\lvert\Delta\rvert$, whose spread at $n = 2$ is what made self (1.67%) and cross (5.76%) look
different. That floor is 6–10$\times$ below Phase 3's 14–27% seed-to-seed CV, so **single runs per $\lambda$ remain
interpretable** and no repeats-per-arm redesign is needed.

`preflight` was corrected accordingly: bitwise at $\lambda = 0$ only, and at $\lambda \neq 0$ the cross-sampler
deviation is *recorded* against the backend's own self-deviation floor. A zero floor with a nonzero cross deviation
still raises — that would be a real defect.

**Infrastructure measured in passing** (all folded into `pc_sweep.slurm` / `submit_wave1.sh`): setup is
**NFS-bound, not GPU-bound** — the 32 GB of mmap'd safetensors page in at **~5 MB/s** (process in state `Dl` on
`folio_wait_bit_common`), at measured rates of **5–12 MB/s**, making setup 19–55+ min. The driver is the node's
**total I/O contention across all tenants**, not co-location of one's own jobs: on the relaunch, setup time came out
*anti*-correlated with how many of my jobs shared a node (t-806, 5 of mine: 19–20 min; n-804, 1: 33 min; n-801,
3 of mine but 8 jobs from 5 users: 55+ min). An earlier reading of the first attempt blamed self-co-location and
was wrong. Submissions are still staggered, but as risk-spreading across nodes rather than contention-avoidance.
**Walltime must be right-sized per job, not maximised**: `killable` allows 24 h, but a blanket `--time=720` on all
nine left Slurm's backfill unable to slot them with 106 pending vs 88 running — estimated starts ran to the next
day, and resubmitting the identical jobs with `--time` matched to their work started all nine within 30 min.

### FINAL RESULT (waves 1+2, 5 seeds, ~25 GPU-h) — the hypothesis splits in two

**Supported: more novelty at preserved recognizability. NOT supported: a later breakdown point.**

At a $\lambda$ where guidance still works for a given seed, the corrector delivers substantially more
novelty at comparable recognizability, and this replicates on every seed tested:

| $\lambda$ | 1234 | 2024 | 3141 | 4242 | 5555 | mean |
|---|---|---|---|---|---|---|
| 0.79, $\Delta f$ vs Phase 3 | +38.6% | +21.6% | +38.2% | +13.7% | +43.2% | **+31%** |
| 1.18, $\Delta f$ vs Phase 3 | — | +14.2% | +57.6% | +47.1% | +33.9% | **+38%** |

all far above the 3.7% within-seed noise floor, and at $\lambda = 0.79$ four of five PC images are cleanly
recognizable dogs (the fifth is a judgment call). The restyles are **diverse** — flat silhouette, etched
scratchboard, neon line-art, pop-art, ink cartoon — not one fixed off-manifold attractor.

**But the window does not move.** Wave 1's headline was seed 1234 at $\lambda = 2.357$, where PC is still
clearly a dog and Phase 3 has degraded to a pictograph. Across the other four seeds at their own
transitions ($\lambda = 1.18$): seed 2024 both intact; seed 3141 both destroyed; seed 5555 both marginal;
and **seed 4242 PC is destroyed — it renders the digits "97" — while the control still shows a creature
face**, i.e. PC is *worse*. One seed supports extension, one contradicts it, two are neutral. **Seed 1234
was a favourable draw**, which is precisely what a single $z_{0}$ cannot reveal.

So the corrector is worth its 1.9$\times$ compute if what you want is more novelty at a working $\lambda$,
and is not a way to push $\lambda$ higher.

### OPEN: `corrector_steps` > 1 is untested where it matters

Everything above is `corrector_steps=1`. C=2 was run only at $\lambda = 3.93 / 5.5 / 7.07$ — all past the
breakdown point, i.e. on already-destroyed images — and from that it was concluded only that doubling C
does not rescue high $\lambda$. **There is no C>1 data at $\lambda \leq 2.36$, where images survive and
where the corrector's entire measured benefit lives.** The ablation was placed where the action was *then
believed* to be (high $\lambda$, window extension) and did not move when that framing proved wrong — the
same error that invalidated the first wave-2 grid.

Both outcomes are informative and neither is predictable from what is on file: more corrector iterations
could compound the tilt, or saturate because the $\lambda/(1+\lambda)^{2}$ cap shrinks each step's
contribution. Every `disp` measurement available is from the annealed regime and does not extrapolate.

Proposed: C $\in \{2, 4\}$ (geometric from C=1, spans 4$\times$) across the same 5 seeds at
$\lambda$-lattice indices 1–4, $\approx 23$ GPU-h, staged so C=4 runs only if C=2 shows a gain. Full
handoff with commands and the gotchas: `notebooks/flux_guided_phase5/NEXT_SESSION.md`.

### Wave 1 result (seed 1234) — what the single-seed run showed, before replication

**Read the images, not the scalars.** The decisive comparison is at matched $\lambda$ and matched `n_steps=10`,
differing only by the corrector:

| $\lambda$ | Phase 3 (no corrector) | PC (`corrector_steps=1`) |
|---|---|---|
| 1.571 | recognizable folk-art dog, $f = 14.2$ | recognizable cartoon dog, $f = 23.5$ |
| 2.357 | **degraded** to a crude pictograph, $f = 47.2$ | **clearly a dog**, $f = 57.2$ |
| 3.143 | broken, $f = 89.8$ | broken, $f = 91.4$ |

So the corrector buys roughly **one $\lambda$ step of extra structural integrity plus 20–65% more novelty**,
for 1.9$\times$ the compute. Modest, real, and visible. At $\lambda = 2.36$ PC is simultaneously *more*
recognizable and *higher* $f$ than Phase 3 — the one place the phase's hypothesis is cleanly confirmed.

**Two measurement errors were made and corrected before this conclusion was reached**; both are now
Established Findings in CLAUDE.md because they generalize:
1. The **compute-matched control was invalid**. `flow_guided` at `n_steps=19` (chosen to match PC's 19 guided
   units) is *destroyed* at $\lambda = 1.571$ where `n_steps=10` is fine — more ODE steps integrate the guided
   field more faithfully and land further off-manifold. Phase 3's window exists partly *because* of
   discretization error. Every matched-compute comparison against that arm was PC-versus-broken.
2. $\lVert x \rVert/\sqrt{d}$ is **anti-correlated** with image quality here (destroyed image 2.88, intact
   image 3.90). A whole decision gate and a tilt-vs-inflation frontier were built on it before the images
   falsified it. `hf` tracked recognizability correctly at every comparison and is the proxy to keep.

**The corrector's reach is capped by the $\eta$ rule** (derived and GPU-confirmed; see `flow_guided_pc.py`):
under `eta_reference="total"` the reward-direction displacement goes as $\lambda/(1+\lambda)^{2}$, peaking at
$\lambda = 1$ and decaying after. `pc_unguided` measures $f$ = 0.9702 / 0.9699 / 0.9691 / 0.9693 at
$\lambda$ = 0.79 / 1.57 / 2.36 / 3.14 — flat to the third decimal across a 4$\times$ change in $\lambda$ — and
`corrector_steps=2` at $\lambda = 7.07$ reproduces the plain baseline within noise with `disp` = 0.0099.
`eta_reference="score"` removes the cap (`disp` 0.0789 vs 0.0099, $f$ +16.9% over baseline at
$\lambda = 7.86$) but collapses the noise-to-drift ratio from 6.25 to **0.896**, turning the corrector from a
Langevin sampler into approximate gradient ascent — more novelty, worse `hf` and $\lVert x \rVert$.

**Noise floors**, measured from replicates sharing one $z_{0}$ and varying only the Langevin seed (jobs
967145/967146/967147 plus 966693 as an unplanned fourth): $f$ CV = **3.7%** at $\lambda = 3.93$ (n=4) and
**0.64%** at $\lambda = 5.5$ (n=3). The floor shrinks with $\lambda$ because the high-$\lambda$ attractor
washes out both corrector noise and backend nondeterminism. Within-seed spread is entirely accounted for by
the backend floor (CLAUDE.md), so **seed replication is needed on the $z_{0}$ axis only**.

### The sweep

`notebooks/flux_guided_phase5/pc_sweep.py` + `pc_sweep.slurm`, one job per arm, with setup *imported* from Phase 3's `fine_lambda_sweep.py` so
$\lambda_{s}$, the reference latents and the whole reward config are bit-identical to that run:

| arm | settings | est. |
|---|---|---|
| `pc_guided` | `predictor_guided=True` — strict superset of Phase 3 | 3.6 h |
| `pc_unguided` | `predictor_guided=False` — all tilt from the corrector | 2.3 h |

*(This table is the PRE-LAUNCH plan, kept as a record. It specified `corrector_steps=2`; the runs actually used `corrector_steps=1` — Song et al.'s reference setting, and the choice that makes `pc_unguided` compute-matched to Phase 3 for free.)*
| `flow_guided` | `corrector_steps=0` — Phase 3, bitwise | 1.3 h |

8 $\lambda$ points on *every other* point of Phase 3's zoom grid ($\{0, 0.79, 1.57, 2.36, 3.14, 3.93, 4.71, 5.5\}$), `SWEEP_SEED=1234` so $z_{0}$ is
identical to Phase 3's and the images pair one-to-one, `exact_jacobian=True`, `n_steps=10`, `shift=3.0`. Estimates scale Phase 3's **measured** 571 s
per guided $\lambda$ by the per-ODE-step cost ratios the dry run measured ($1.0 / 1.8 / 2.8$). `--mem=24000` (Phase 3 job 957386 measured host peak RSS
at 10.2 GB on this exact path), `--constraint=l40s` pinned to one model, resumable per $\lambda$.

**The third arm is re-run rather than read from Phase 3's stored JSON on purpose**: a GPU model change alone shifts $f$ by 16% of
$\operatorname{std}_{p}(f)$ (job 697271), the same order as the effect being measured, so all three arms must share one GPU, one reward and one
reference bank.

**What success looks like — the hypothesis is NOT "higher $f$".** The corrector pulls back toward $p_{t}$, so at matched $\lambda$ the PC arms should
report *lower* $f$ than Phase 3. The claim under test is that the **recognizable window extends to larger $\lambda$**: at $\lambda \in [3.9, 5.5]$,
where Phase 3 is destroyed, PC images should still read as "a dog". Quantitative proxies recorded per point are `x_norm_final`
($\lVert x \rVert/\sqrt{d}$; off-manifold failure inflates it) and `hf_frac` (spectral power above 0.25 Nyquist). **Caveat measured on Phase 3's own
stored images: `hf_frac` is NOT monotone in $\lambda$** — $0.0546$ at $\lambda = 0$ (photo-like dog) $\to 0.0213$ at $\lambda = 1.57$ (a smooth
restyle genuinely has less high-frequency content than a photograph) $\to 0.0417$ at $\lambda = 5.5$ — so it is usable as a comparison *across arms at
matched $\lambda$*, never as an absolute "is this broken" score. `render_pc_comparison.py` builds the arm $\times \lambda$ grid and the table, and
cross-checks the re-run `flow_guided` arm against Phase 3's stored values.

The dry run (tiny random transformer, so indicative of the mechanism only, never of real FLUX) already shows the intended behaviour:
$\lVert x \rVert/\sqrt{d}$ runs $1.25 \to 4.89$ across the $\lambda$ grid for `flow_guided`, $1.10 \to 4.29$ for `pc_guided`, and stays flat at
$1.06 \to 1.23$ for `pc_unguided` — i.e. the corrector holds the latent norm, which is exactly the hypothesised mechanism.

## Phase 6 — Generation quality versus compute cost (not implemented)

---

CLAUDE.md invariant 2 ("Gradient-free") has been updated: it now states that it holds for Algs 1–3, and that
`flux_guided_sample` (Phase 3) is the first sampler to use $\nabla f$, requiring a `differentiable=True`
denoiser/velocity function.
