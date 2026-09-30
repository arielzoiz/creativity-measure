# Gradient guidance for FLUX.1-dev — roadmap and status

Goal: steer a 12B flow-matching model (FLUX.1-dev) toward creative, out-of-distribution regions using the squared
normalized global IEM reward, with **inference-time gradients** $\nabla_x r(x)$ instead of dozens of GPU-hours of SMC.
Needs a reward whose autograd graph fits in memory, hence the move from a sequential Brownian integral to an i.i.d.
Monte-Carlo estimate.

Rules for every phase: the existing Brownian methods keep working unchanged; reuse existing functions; shared code goes in
its own module (like `smc_common.py`) or a subclass in a new file. Per-stage results live in `STATUS.md` (Phase 1).

| Phase | What | State |
|---|---|---|
| 1 | i.i.d. Monte-Carlo reward (`SquaredIIDGlobalIEMDistance`) | **Code done, committed (`4a49e4a`). Local tests PASSED. GPU checks G0/G1 PENDING** |
| 2 | Autograd + memory stress test on FLUX | **Code done locally. CPU `--dry-run` PASSED (all 3 safety mechanisms, including the double-backward trap actually firing and recovering). GPU run PENDING** |
| 3 | Direct test-time guidance | NOT STARTED |
| 4 | Standalone Langevin (ULA) MCMC | NOT STARTED |
| 5 | Predictor-corrector (Langevin) sampler | NOT STARTED |
| 6 | Generation quality vs compute cost | NOT STARTED |

## Phase 1 — DONE locally, verification PENDING on the cluster

Built: `distances/iid_global_iem.py`; `utils.log_uniform_gammas` and `utils.simulate_iid_noise`; per-row γ in
`edm_score_fn`; `ExpectedDistance` protocol used by `tilt.expected_distance`; a mixed-sigma guard in `flow_map_denoiser`.
Decisions: frozen static bank (f deterministic, reference cache always hits); plain i.i.d. log-uniform γ drawn by the
**caller** (log-midpoint / stratified are later swaps, no change to the distance); only ε is drawn by the library.

Passed locally (CPU, toy): 25 new + 4 adapter tests; full suite 231 passed with no regressions; pyright clean on touched files.
**Pending:** G0 (per-row γ on the real transformer) and G1 (IID vs Brownian agreement on FLUX). Details, commands and the
acceptance rules are in `STATUS.md`. **Outputs needed from G1:** which (G, N_eps) to use, and the $\lambda_s$ rescale factor.

Known caveats: $\lambda_s$ must be re-measured for IID runs; `--mem=64000` in the slurm file is inherited, not measured.

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

**Pending:** the same three checks on the real FLUX.1-dev transformer (`phase2_autograd_stress.slurm`, written,
not yet submitted — no GPU job for this ran this session; Phase 1's G0/G1 check was the one GPU allocation in
use). Credential note for whoever submits it: use `HF_TOKEN` in the environment, not `HF_TOKEN_PATH` — see
`notebooks/flux_lambda_sweep_strong_2/flux_strong_tilt_3_1.slurm`'s failure (6).

## Phase 3 — Direct test-time guidance (not implemented)

At each standard Flux denoising step $t$, compute $\nabla_{x_t} r(x_t)$, scale it by a guidance weight, and add it to the
model's score/velocity prediction. Question: does a **single-pass, guided ODE generation** reach the creative tilt with no
secondary sampling loops?

## Phase 4 — Standalone Langevin dynamics, MCMC upgrade (not implemented)

If direct guidance fails, or to sample the base latent space without the full reverse process: replace the SMC pCN kernel with
the Unadjusted Langevin Algorithm,
$x \leftarrow x + \eta\,(\nabla_x \log p(x) + \lambda \nabla_x r(x)) + \sqrt{2\eta}\,z$.
Question: does gradient-based MCMC reach high-reward states faster than SMC?

## Phase 5 — Predictor-corrector (Langevin) sampler (not implemented)

## Phase 6 — Generation quality versus compute cost (not implemented)

---

Not in the repo's existing invariants yet: CLAUDE.md invariant 2 ("Gradient-free") stays true for Algs 1–3. Phases 3–5 are the
first samplers to use $\nabla f$, so update that invariant when Phase 3 lands.
