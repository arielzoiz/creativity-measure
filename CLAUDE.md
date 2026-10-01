
# What This Project Is

**Research goal.** Unsupervised *creative* generation: given only a learned data distribution $p$ (no external notion of creativity),
define and sample a posterior that favors samples which are **novel** (far from ordinary generalizations of the data)
yet **structurally consistent** (still on the manifold / compatible with the geometry of $p$).
Creativity is deliberately *not* low-probability sampling — $p(x)$ alone cannot tell a meaningful gap from empty space.

**The target distribution.** Minimizing $D_{KL}(q \| p) - \lambda E_{q}[f]$ gives the tilted posterior

$$q_{\lambda}(x) = \frac{1}{Z_{\lambda}} p(x) \exp(\lambda f(x))$$

where $f$ is a novelty reward built from the **Information-Estimation Metric (IEM)** (Ohayon et al.) — a distance induced by the geometry of $p$ itself,
comparing the score fields $\nabla \log p_{Y_{\gamma}}$ around two points across blur levels $\gamma$.
The updated reward that should be in use:
- NormalizedExpectedDistanceReward: $f(x) = E_{x' \sim p}[D^{2}_{IEM}(x, x')] / E_{x', x'' \sim p}[D^{2}_{IEM}(x', x'')]$ —
  `NormalizedExpectedDistanceReward` + `SquaredGlobalIEMDistance`. Unitless, so $\lambda$ acts as a plain inverse temperature. **This is the current default.**
- older version used the previous reward, and it shouldn't be used for current work: $f_(x) = E_{x' \sim p}[D_{IEM}(x, x')]$ — `Reward` + `GlobalIEMDistance`

**THE CENTRAL PROBLEM: how to sample from $q_{\lambda}$.** $Z_{\lambda}$ is intractable, $f$ is expensive (many score/UNet evals per call) and **gradient-free by design**,
and in high dimensions there is no grid to enumerate. This repo is a search for good samplers for $q_{\lambda}$.
Everything else (distances, refsets, generators) is fixed infrastructure that any sampler consumes.

**Current samplers** (`creativity_measure/`). Notebooks, commit messages and slurm headers refer to these **by number**:
**Alg 1 = `adaptive_tempering_smc`, Alg 2 = `diamond_smc`, Alg 3 = `flowmap_smc`.**

- **Alg 1** — `adaptive_tempering_smc.py` — the mature one. Adaptive-tempering SMC over $\beta \in [0,1]$ with a pluggable rejuvenation `Kernel`:
  `PCNKernel` should be used (local prior-preserving pCN moves in the generator's latent Gaussian space, $x = G(z)$ — the workhorse for strong/off-manifold tilts)
  while `IndependenceKernel` is stale. The only sampler with a checkpoint-resume path.
- **Alg 2** — `diamond_smc.py` — SMC *inside* the generative trajectory of a pretrained stochastic flow-map model (Algorithm 2 of Holderrieth et al., arXiv:2602.05993):
  a DDPM transition per step, then a posterior lookahead through the diamond map to reweight. Entry point `diamond_smc_sample`.
- **Alg 3** — `flowmap_smc.py` — a **single forward pass** noise $\to$ data with one lookahead per step: no tempering ladder, no rejuvenation, no acceptance rate to collapse.
  Entry point `flowmap_smc_sample`. Cost is dominated entirely by reward evaluations.
- **`flux_guided_sample`** (`flux_guided.py`) — not numbered with Algs 1–3: it is the first sampler built on **inference-time gradients** $\nabla_x f$ rather than SMC/MCMC
  (Phase 3 of the gradient-guidance line, `notebooks/iid_iem_flux_check/ROADMAP.md`). At each step of FLUX's own native flow-matching ODE, nudges the velocity by
  $\lambda \nabla_{\hat x_0} f$ (identity-Jacobian approximate mode, default) or the exact chain-ruled gradient through the transformer (`exact_jacobian=True`) —
  no tempering ladder, no particles, no resampling; a single guided forward pass. Needs a `differentiable=True` denoiser/velocity function
  (`generators/flux.py`'s `flux_edm_denoiser`/`flux_velocity_fn`/`build_flux_denoiser`). See Invariant 2's exception below.
- **More samplers are expected.** Adding one is a first-class contribution, not a refactor.

## Invariants Every Sampler Must Respect

1. **The reward is frozen.** `Reward` is a frozen dataclass; references `x_refs`, their `weights`, and the `Distance`'s Brownian seed are fixed before sampling,
   so $f$ is a deterministic function of $x$ — required by SMC/MCMC theory. Never resample references mid-run.
2. **Gradient-free — holds for Algs 1–3.** $f$ is treated as a black-box scalar; no $\nabla f$, no $\log p$ evaluations in the acceptance ratio (pCN cancels the Gaussian prior).
   `log_p_X` / `grid_normalize` / `tilted_log_density` are only used for a **2D-toy-only** version, never part of a high-d SMC/MCMC sampler.
   **Exception: `flux_guided_sample`** (Phase 3, `notebooks/iid_iem_flux_check/ROADMAP.md`) is the first sampler to use $\nabla f$ by design — it steers FLUX's own
   generative ODE with the reward's gradient instead of reweighting/rejuvenating. It requires a `differentiable=True` denoiser/velocity function and explicitly
   freezes every model parameter (`requires_grad_(False)`) before building any autograd graph through it, so this exception never silently reaches the frozen
   reward's weights or an unrelated sampler.
3. **The 2D-toy-only version can be used as a reference for 2D tests, but is not ground-truth.** The 2D-toy-only sampler fails at strongs tilts and drifts off the grid,
   and therefore is not achieving the goal.
3. **Determinism via a threaded generator.** All randomness comes from a single `torch.Generator(seed)` passed through; never touch global torch RNG.
4. **`reward.x_refs` pins device and dtype** for the whole run.
5. **Cost is dominated by $f$ evaluations** (each is $N_{\gamma} \times N_{\epsilon} \times B$ score calls on a batch of $B$ — **independent of $R$**:
   `GlobalIEMDistance` scores each point once per $\gamma$ interval and broadcasts the pairwise differences, and caches the reference
   scores across calls via `cache_refs=True`). Count reward calls per particle per level when judging a new sampler. With the reward
   this cheap, $G(z)$ is now a comparable share of a pCN sweep.
   *"Independent of $R$" holds per call, after the bank exists.* Building it is $O(R)$ — $(N_{\gamma}-1) N_{\epsilon} R$ score rows, 19.1 min at $R = 64$ on an L40S —
   and is re-paid **every job**, because only the reference latents are persisted, not the `score_bank`. Persisting the bank (1.46 GB at $R = 64$, keyed on the
   base process plus `gammas` / `num_eps` / `dist_seed`) is the standing optimization; it also takes the model out of the chain for the scores, which is what
   broke job 697271 when a GPU change shifted $f$ by 16% of $\operatorname{std}_{p}(f)$.
6. Reuse `smc_common.py` (`_ess_from_logw`, `_systematic_resample`) and `generators/base.py` (`karras_sigma_schedule`, `edm_ode_step`, `edm_generator`).

## Adding a New Sampler

Follow the shape of the existing two: a module in `creativity_measure/` exporting `<name>_sample(reward: Reward, lam: float, n_particles: int, *, ..., seed=None) -> <Name>Result`,
where the result dataclass carries `X`, `logw`, and per-level **diagnostics** (ESS history, acceptance, effort). Diagnostics are not optional — in high dimensions they are the only
way to tell whether the run worked. Export from `__init__.py`, add tests under `tests/`, and validate first on the 2D toy where the grid gives ground truth.

## Established Findings (don't re-derive these)

- **Pixel space fails; use latent-space models.** (`mnist_edm_creative_sweep.ipynb`.)
- **Rejuvenation length is adaptive** (`n_mcmc=None`): sweep until the kernel's own `decorrelation` metric drops below threshold, capped by `max_n_mcmc`.
- **Choosing $\lambda$.** Bracket with the scale unit $\lambda_{s} = 1 / \operatorname{std}_{p}(f_{2})$ (or the older data-derived $\lambda_{0} = \operatorname{std}(\log p) / \operatorname{std}(f)$ at
  the refs) and sweep $m \cdot \lambda_{s}$ for $m \in \{1, 2, 3\}$; then **select the largest $\lambda$ whose ensemble diversity (`uniq/N`, min per-level ESS/N) stays above ~0.5**.
  Novelty $E_{q}[f]$ rises forever, so it is *not* the stopping signal — degeneracy is. (`normalized_squared_iem_tilt.ipynb`.)
  The "rises forever" half is measured on the pCN line only — on the Algorithm 3 flow-map path $E_{q}[f]$ stayed at or below $E_{p}[f]$ at every $m$ (a3_1, a3_2).
- **$\lambda_{s}$ is an anchor, not a measurement.** Measure once on a held-out batch, cache it, never chase precision: at $n_{\text{heldout}} = 32$ it carries
  $\pm 13\%$ ($1/\sqrt{2(n-1)}$), which changes no decision since selection is on in-run `uniq/N` / ESS and the $m$ sweep spans $3\times$.
  **But never eyeball $\lambda$** — $f$ has a CV of $0.7\%$, so $\lambda \sim 1$–$10$ is inert and the working scale ($\approx 140$) is invisible without the measurement.
- **References: $R = 64$, `RandomRefs`, uniform weights.** Independent-draw $\tau = 0.960$ at $d = 65536$, on a plateau for $R \ge 32$ ($\tau$-SE $= 0.060$).
  **Never use `RefSelector.select(None)`** — its nested-$\tau$ rule is biased $+0.211$ at $R = 4$ and stops there, i.e. worst exactly where it stops.
  `WeightedFPSRefs` degenerates to $R_{\text{eff}} = 1.22$ of 4 because $D^{2}_{IEM}$ between FLUX latents concentrates within a few percent — coverage selection has no gradient to follow.
  Raising $R$ buys *scale* stability only ($\operatorname{SE}(f)/\operatorname{std}_{p}(f) \approx 0.65/\sqrt{R}$), never rank stability, and is not the binding error. (`notebooks/refset_auto_r/`.)
- **$f(\text{refs}) = (R-1)/R$ exactly** (uniform weights): the numerator averages $R$ terms including the zero self-pair, `reference_pair_mean` divides by $R(R-1)$.
  Free assert (the bank is already cached, so it costs no score rows) — but use tolerance `1e-3`, not `0.10`, which cannot separate $63/64$ from the failure modes that land on exactly $1.0$.
  It tests **normalization wiring only**: `GlobalIEMDistance.pairwise` aliases `batch_scores` to `ref_scores` when `X is x_refs`, so the zero diagonal is bitwise-free and proves nothing about score-field determinism.
- **An ESS history is only interpretable relative to the `ess_threshold` that generated it** — a threshold of 1.0 zeroes $U$ every step, so each reading is a single telescoped
  increment, not accumulated weight. Report `uniq/N` for degeneracy and pairwise latent distance for what the images are; never `n_distinct` alone. (a3_1, a3_2.)
- **Algorithm 3's ESS-triggered resampling degenerates the cloud** at $M = 8$ for every $m \in \{1,2,3\}$ under thresholds 1.0 *and* 0.5, and on both the plain and Z-scored potentials.
  `uniq/N` is monotone non-increasing — systematic resampling only destroys lineages and nothing regenerates them, so the threshold sets the rate of an inevitable collapse.
  This is a mechanism defect, not a tuning problem. (`notebooks/flowmap_smc_flux/`.)
- **SMC cannot escape the generator manifold** — particles are $x = G(z)$, so its target stays proper even where the literal grid $q_{\lambda}$ diverges.
  This is a feature, and it differs from the toy-2D-grid version, that allows the distribution to drift of the grid.
- **Algorithm 1 ceilings at $m_{\text{eff}} \approx 2.3$ on FLUX, and it is NOT degeneracy** — `uniq/N = 1.00` and min ESS/N $= 0.80$ hold to the last level. Two independent mechanisms stop it, both measured over legs 2.2 / 3.1 (jobs 776753, 778007):
  (i) the ESS schedule gives $d\beta = 0.5 / (\lambda \cdot \operatorname{std}_{q}(f))$ at $\alpha_{\text{ess}} = 0.8$ — predicts every observed level to ~10% — and $\operatorname{std}_{q}(f)$ grows like $\exp(11.3\beta)$, so $d\beta$ collapsed 0.5422 → 0.2071 → 0.0230 per leg; $\beta = 1$ would need **~49 more levels $\approx$ 10.6 GPU-days**.
  (ii) rejuvenation dies faster than the ladder advances: the decorrelation rate fell **3.8× for a 1.38× tilt rise** (0.0191 → 0.0051/sweep) as acceptance dropped below `target_acc` and Robbins–Monro shrank `s` (0.47 → 0.19) — a compounding loop. Neither leg's join reached `decorr < 0.2`.
  **The ceiling is the sampler's, not $p$'s**: $\operatorname{Var}_{q}(f) = 6.8\times10^{-3}$ over the remaining 63 $\lambda$ units means $\geq 0.43$ more $E_{q}[f]$ lies past where pCN stops, against the +0.135 gained. This is the motivation for Algorithm 3 — one forward pass, no ladder, so (i) cannot arise, and no acceptance rate for (ii) to collapse. (`notebooks/flux_lambda_sweep_strong_2/`, Takeaways.)
- **Reported $E_{q}[f]$ is a lower bound whenever a level stopped on `timeout` or `cap`.** Sweeping at *fixed* $\lambda$ still moves it a lot — 55% of leg 3.1's total gain came from its finishing pass with $\beta$ unchanged — and the reward marginal equilibrates in ~28 sweeps even while latents stay correlated (`decorr = 0.801`). Converging at the $\lambda$ you have is cheap; advancing $\beta$ is not.
- **Keep the lower $\gamma$ cut at the data scale ($\gamma_{\text{lo}} = 1/S^{2}$); do NOT start the window at 0.** Extending down to $2^{-10}$ adds 9.2% of $D^{2}_{\text{IEM}}$ and buys **no extra reach** — the ladder stalls at the same $\lambda_{\text{eff}} \approx 213$ with slightly *less* novelty — while samples gain **1.5–1.7$\times$ more spectral power above 0.25 Nyquist**, i.e. high-frequency graininess: below $\gamma \approx 1$ the integrand degenerates to $\lVert x - x_{r}\rVert^{2}$ (Concl. 1a) and in $d = 65536$ noise is the cheapest $L_{2}$. The window also rescales $\lambda_{s}$ (92.195 → 78.678), so **$m$ is not comparable across $\gamma$ grids — compare $\lambda_{\text{eff}}$**. (`notebooks/flux_fullgamma_alg1/`.)
- **Rising $\operatorname{std}_{q}(f)$ under stronger tilt is expected, not degeneracy.** $q_{\lambda}$ is an exponential family, so
  $\frac{d}{d\lambda} E_{q}[f] = \operatorname{Var}_{q}(f)$ and $\frac{d}{d\lambda}\operatorname{Var}_{q}(f) = E_{q}[(f - E_{q}f)^{3}]$ — the spread *grows* whenever $f$ is right-skewed under $q$, which is what moving mass off the bulk of $p$ toward a larger-volume shell does. (It must collapse eventually as $q$ concentrates on the maximizer, but only past the peak.)
  **Use the first identity as a free equilibrium check**: per-level $\Delta E_{q}[f] / \Delta\lambda$ against $\operatorname{Var}_{q}(f)$ needs only numbers every level already records. It holds across the whole FLUX ladder including $\lambda_{\text{eff}} = 207 \to 214$ (ratios 0.72, 0.85), where $\operatorname{std}_{q}(f)$ had grown to $6.8\times \operatorname{std}_{p}(f)$ and looked alarming. At $N = 16$ each variance carries $\approx 37\%$, so treat ratios in $\approx 0.4$–$2.5$ as consistent. Never read spread growth alone as collapse — `uniq/N` and ESS/N remain the degeneracy signals.
- **$E_{q}[f]$ measured on $\operatorname{map}(x_{t}, t, 1)$ at intermediate $t$ is inflated — always subtract a $\lambda = 0$ control.**
  The untilted base process alone runs 0.9160 → 1.0207 (step 10) → 0.9943 ($t = 1$): a give-back of $3.7\operatorname{std}_{p}(f)$ with no tilt, no resampling, `uniq/M = 1.00`.
  **Only $t = 1$ is artifact-free.** (`notebooks/flowmap_smc_k_sweep/RESULTS.md`.)
- **Lookahead depth $K$ saturates at $\approx 8$, and shows NO trend in the outcome; use $K = 4$.** Terminal control-subtracted tilt effect over $K = 4/8/16/32$ is $+0.81 / +1.49 / +0.59 / +1.23\operatorname{std}_{p}(f)$ — scatter, not a peak (seed sd $\approx 0.17$). $\operatorname{sd}(U)/\operatorname{sd}(V)$ does not follow $1/\sqrt{K}$ — the measured drop stays $\approx 1.5\times$ while the prediction reaches $5.66\times$, plateauing at $\approx 0.67$ (within-cell, so not confounded by cloud state).
  $K = 4 \to 8$ buys $\approx 10\%$ for $1.77\times$ the cost; $K \ge 16$ is wasted. **The floor is now measured** (`notebooks/flowmap_smc_replay/A0_RESULTS.md`): estimator noise is only **28–48%** of $\operatorname{Var}(U_n)$ at $K = 4$, so genuine $\Delta V$ dominates and **CRN is inapplicable** (closed, do not revisit). The saturation is a property of the **aggregation**, not the lookahead: $\sigma_{\varepsilon}^{2} \sim k^{-p}$ with $p = 0.2$–$0.8$, never $1$, because `logmeanexp` goes max-dominated as $a = \lambda\operatorname{sd}_{k}$ grows. Both ends of the $\tau$ interpolation are null on the outcome — mean $-0.014$ sd and max $-0.034$ sd, each over 3 paired seeds — so **the aggregation axis is settled; leave `_soft_value` alone.**
- **Alg 3's ceiling is its PROPOSAL, not its sampler: it tilts by reweighting draws from the untilted $p$, so $\operatorname{ESS}/M \approx e^{-m^{2}}$.** Verified at the first guided step, $m = 1.25$: $M = 16$ gives $0.215$ against the predicted $0.210$ — while **every $M = 8$ run reports $0.44$–$0.74$, overstating its own ESS by 2–3$\times$** (8 draws rarely sample the tail that dominates the weight sum). Holding today's effective sample size therefore costs $M \sim e^{m^{2}}$: $\approx 92$ particles at $m = 2$, $\approx 333$ ($\approx 5.4$ GPU-days/run) at Alg 1's $m_{\text{eff}} \approx 2.3$, $\approx 1.4\times10^{4}$ at $m = 3$.
  **And Alg 3 is NOT underperforming at $m = 1.25$ — it hits the target:** first order $E_{q}[f] - E_{p}[f] = \lambda\operatorname{Var}_{p}(f) = m \operatorname{std}_{p}(f)$, i.e. $+1.25$, and `m16` measured $+1.26 \pm 0.31$. The target itself is weak there; Alg 1 reached $+19\operatorname{std}_{p}(f)$ only by getting to $m_{\text{eff}} \approx 2.3$, where $\frac{d}{d\lambda}E_{q}[f] = \operatorname{Var}_{q}(f)$ has blown up. **pCN escapes this by *moving* particles (MCMC on the tilted target) instead of selecting among draws from $p$** — hence `uniq/N = 1.00` throughout Alg 1 against Alg 3's collapse to one lineage by step 3. Raising $M$ or improving the twist cannot repeal $e^{m^{2}}$; only a tilted proposal can.
- **Alg 3's diagnostics do NOT predict novelty, and a single-seed $M = 8$ result is not evidence.** Three interventions (mean aggregation, deferred resampling, $M = 16$) each avoided collapse and each cut $\operatorname{sd}(U)/\operatorname{sd}(V)$ from $0.534$ to $\approx 0.36$, yet spanned $+0.67$ to $+2.35\operatorname{std}_{p}(f)$ in terminal effect — so neither `uniq/M` nor $\operatorname{sd}(U)/\operatorname{sd}(V)$ may gate a decision. Noise at $M = 8$: per-run MC SE $\approx 0.43$ (control-subtracted $\approx 0.5$), and **per-seed sd of a paired difference $= 1.22$**, so detecting $1\operatorname{std}_{p}(f)$ needs $\approx 12$ paired seeds ($\approx 73$ GPU-h). The $0.17$ above is the max experiment's and does **not** generalize — it was small only because max and soft rank particles identically. **Compute the detectable effect size before choosing the number of seeds.**
- **Jensen is present in Alg 3 but is not a correctness defect.** $\hat V = \log\frac1K\sum e^{\lambda r_k}$ carries a bias $\approx (e^{a^{2}}-1)/2K$, and $V/\lambda$ exceeds $\bar f_{\text{cand}}$ by $\lambda\operatorname{Var}_{k}(f)/2$ (measured ratio $0.79$–$1.10$). The target stays exactly $q_{\lambda}$ regardless — intermediate $V_t$ are twists and only the terminal $V_N = \lambda f(x_1)$ pins it. **$\eta$ is what keeps $a$ small, so it is a Jensen-exposure control, not just a free knob**: any lookahead whose candidates approach prior draws has $a(t \to 0) = m$ exactly, i.e. at $m = 3$, $K = 4$ the soft value degenerates to a hard max.
- **$K$ does not predict collapse.** $K = 4$ and $K = 32$ collapsed to one lineage at step 3; $K = 8$ and $K = 16$ did not. Degeneracy is an $M$ problem, not a $K$ problem.
- **Collapse inflates $E_{q}[f]$.** Single-lineage cells show the largest mid-run effects and the steepest decay ($K = 4$: $+4.91 \to +0.81$ sd peak-to-terminal). Never read $E_{q}[f]$ without `uniq/M`.

## Resuming From a Checkpoint

Rules are general; the pCN parentheticals are Algorithm 1, the only sampler with a resume path so far.

- **Do** fold the base target the checkpoint already carries into the **kernel**, explicitly (pCN: `beta=1.0, lam=lam_base + beta*lam`, keeping the SMC's own `beta` as the checkpoint key). Resume on the source run's GPU (`--constraint`) with persisted reference latents, and bump `RUN_TAG` per *leg* — output names derive from $N$ / `M_TILT` / `SEED`, so an untagged leg overwrites its own resume source.
- **Don't** treat correct weights as evidence the resume is correct. Ratio-based weights reparametrize exactly (the base cancels); kernels do not (their base is baked in, not an argument), so **only the kernel breaks, and silently**.
- **Don't** resume from a level that stopped on a wall-clock deadline instead of its own convergence rule — finish it at the source setting first. Its convergence baseline probably did not survive the checkpoint, so re-baseline and deliberately over-shoot.
- **Assert** three things: the first resumed level does not regress on the objective; the kernel does not get *easier* across the join; no adaptive step size runs into its clamp. Each means the target is wrong, not that mixing is good.
- **A resumed leg's checkpoint stores $\beta$ relative to *its own* leg, and does not say so.** Leg 2.2 ended at $u = 0.4524$ meaning $\beta = 0.7493$; every consumer must convert via $\beta_{0} + u(1 - \beta_{0})$ from the source's `config["beta0"]`. Both consumers have already got it wrong — the resume (silently mis-tilts the leg) and the *stitch* (leg 3.1's printed ladder labelled $m = 1.63$ as $m = 0.00$ and reported novelty from $\lambda = 150$ instead of from $p$). **Fix it at the writer**: record `beta_true` / `lam_eff_true` in the checkpoint so no reader has to know, exactly as persisting `s` removed that whole class of problem. It also leaks into **filenames**: `decoded_3_1/lvl00_lam0_*.png` is really $\lambda = 207.3$, and using it as an "untilted" baseline produced a cross-run image result that *reversed* once the true baseline was used — **verify any cross-run baseline by comparing latents, not labels.**
- **Don't compare $d\beta$ across legs** as a correctness check. `_next_dbeta` bisects on the *spread* of `fX`, so any finishing pass that moves the cloud legitimately changes it (2.2: `f-sd` $0.0156 \to 0.0220$, $d\beta$ $0.1103 \to 0.1073$). Compare against the source checkpoint's stored `fX` instead — that is the invariant that survives.
- **Budget rejuvenation from the measured rate, not from $s$ you assumed.** At $\lambda_{\text{eff}} \approx 150$, `s` adapts to $\approx 0.47$ (not $\approx 0.8$), giving refresh $\approx \bar{a}(1 - \sqrt{1-s^{2}}) \approx 0.024$/sweep — so `decorr` needs **~60+ sweeps**, not the ~15–20 a larger $s$ would suggest. 2.2 spent 21 sweeps and only reached 0.598.

(Each rule is a failure a real run hit; write-ups in the `flux_strong_tilt_2_*.slurm` headers.)

## Repo Map

- `distances/` — IEM variants (`GlobalIEMDistance`, `SquaredGlobalIEMDistance`, `GeneralizedGlobalIEMDistance` with $f$ = identity/squared, `LocalIEMDistance`, `LpDistance`),
  plus `edm_adapter.py` (`Denoiser` → `score_fn` via Tweedie).
  `iid_global_iem.py` (`SquaredIIDGlobalIEMDistance`) is the loop-free, frozen **i.i.d. Monte-Carlo** counterpart of `SquaredGlobalIEMDistance`
  (Brownian classes untouched): the caller draws $(\gamma_g, w_g)$ with `utils.log_uniform_gammas`, eps is i.i.d. $\sqrt{\gamma}\,\varepsilon$, cost $G\,N_{\epsilon}(R+B)$ rows cold and
  $G\,N_{\epsilon}B$ warm. Its `expected()` is the closed-form mean over refs (no $(B,R,d)$ tensor) and is differentiable in $x$ given a differentiable `score_fn`.
  **$\lambda_{s}$ must be re-measured for it** — $f$'s spread differs from the Brownian grid's; `notebooks/iid_iem_flux_check/` holds the FLUX agreement check (status in its `STATUS.md`).
- `refset/` — reference selectors (`RandomRefs`, `FPSRefs`, `WeightedFPSRefs`) with auto-$R$ by a weighted-$\tau$ rank-stability rule; `.reward(R)` / `.normalized_reward(R)`
  produce the frozen reward. **The auto-$R$ rule is biased at high $d$ and the FPS selectors degenerate there** — use `RandomRefs` with an explicit $R$ (currently $R = 64$);
  see Established Findings and `notebooks/refset_auto_r/`.
- `generators/` — $G: z \mapsto x$ via the EDM probability-flow ODE, so $G(N(0,I)) \approx p$ (`toy_2d`, `edm_pixel`, `tiny_sd`, `flux`).
- `tilt.py` — rewards; `density.py` — $p$ as `log_p_X` / `log_p_Y` / sampler.
- `flowmap_smc.py` also records, all side-effect-free (bit-identical runs, asserted by
  `test_recording_flags_leave_the_run_bit_identical`): `record_r_k` (keeps $(M, K)$ lookahead rewards,
  so any $K' \le K$ is reconstructible offline by subsetting), `project_endpoint` (per-step $E_{q}[f]$
  from $f(\operatorname{map}(x, t, 1))$, weighted **pre**-resample), plus `U_pre_history` and
  `resample_idx_history` — the parent map is *not* recoverable from `ancestors`, which composes.
- `creative_sampling_flow.ipynb` (repo root) — the end-to-end algorithm spec + API sketch.
- `notebooks/` — the experiment record; each ends in a **Takeaways** cell that is the authoritative statement of what was learned. Read the relevant one before changing behavior it calibrated.

# Project Technical Guidelines

## Environment

This project uses **conda**. The environment is named `creativity-measure`.
Always activate it before running anything:

    conda activate creativity-measure

Run all Python commands, scripts, tests, and installs inside this environment.
Do not use the base environment or a different venv.

## Type Checking

This project uses strict static type checking (**Pylance** / **Pyright**).
When you finish implementing a part, run the type checker and fix any errors
before moving on:

    pyright

Run it inside the `creativity-measure` conda environment.
Write fully type-annotated code so it passes cleanly.

## Slurm `--mem`

Peak host RSS is **10.2 GB** for the **Algorithm 3 flow-map jobs** — FLUX.1-dev *plus the
`flux-1-dev-flowmap-lsd` LoRA*, `notebooks/flowmap_smc_flux/`, job 782816 — against the `--mem=64000`
those scripts inherited unmeasured. **Use `--mem=24000` there.** *Not* measured for the Algorithm 1
runs on base FLUX.1-dev without the LoRA; measure those before changing them.

Over-requesting costs queue time: memory, not GPUs, is what blocks these jobs (2026-08-26: n-601 had
2 free A6000s but only 11.8 GB free RAM, ~2 h lost). `sacct` `MaxRSS` is empty on this cluster, so
measure in-process — `note()` stamps `ru_maxrss` on every progress line. Never set `--mem` *equal* to
the observed peak: `ru_maxrss` under-reports page cache and the safetensors are mmap'd, so the cgroup
can charge more, and an OOM-kill drains the node for everyone.

## Slurm: comparing cells across concurrent jobs

- **Pin `--constraint` to ONE GPU model** — never `a6000|l40s`. Concurrent cells land wherever frees
  first, and a GPU change shifts $f$ by 16% of $\operatorname{std}_{p}(f)$ (job 697271), the same order
  as the effects usually being measured. Record `gpu_name` per result and assert it matches.
- **Serialize node-local staging with `flock`, and verify the staged file COUNT**, not just that
  `model_index.json` exists — that file is ~1 KB and copies first, so it survives the very failure the
  check is for. `/tmp` filled mid-copy on n-801 and n-804 and killed two jobs (`cp: No space left on
  device`); separately, a second job on the same node will `rm -rf` the first's in-progress 32 GB copy.
  A lock only helps if *both* jobs take it — while any pre-fix job is live, node-disjointness is the guard.
- **Best: `--nodelist` a node that already holds a complete stage.** Instant cache HIT, no disk needed,
  and it skips the 2.3–3.7 h stage entirely.

## Jupyter Notebooks

When reading or editing `.ipynb` files, use the **notebook MCP server**.
Do NOT parse, read, or write the notebook JSON directly.

- To read a notebook's contents, use the notebook MCP tools.
- To add, edit, or run cells, use the notebook MCP tools.
- Editing the raw JSON risks corrupting cell structure, outputs, and metadata.