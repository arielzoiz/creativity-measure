
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
- **More samplers are expected.** Adding one is a first-class contribution, not a refactor.

## Invariants Every Sampler Must Respect

1. **The reward is frozen.** `Reward` is a frozen dataclass; references `x_refs`, their `weights`, and the `Distance`'s Brownian seed are fixed before sampling,
   so $f$ is a deterministic function of $x$ — required by SMC/MCMC theory. Never resample references mid-run.
2. **Gradient-free.** $f$ is treated as a black-box scalar; no $\nabla f$, no $\log p$ evaluations in the acceptance ratio (pCN cancels the Gaussian prior).
   `log_p_X` / `grid_normalize` / `tilted_log_density` are only used for a **2D-toy-only** version, never part of a high-d sampler.
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
- **Rising $\operatorname{std}_{q}(f)$ under stronger tilt is expected, not degeneracy.** $q_{\lambda}$ is an exponential family, so
  $\frac{d}{d\lambda} E_{q}[f] = \operatorname{Var}_{q}(f)$ and $\frac{d}{d\lambda}\operatorname{Var}_{q}(f) = E_{q}[(f - E_{q}f)^{3}]$ — the spread *grows* whenever $f$ is right-skewed under $q$, which is what moving mass off the bulk of $p$ toward a larger-volume shell does. (It must collapse eventually as $q$ concentrates on the maximizer, but only past the peak.)
  **Use the first identity as a free equilibrium check**: per-level $\Delta E_{q}[f] / \Delta\lambda$ against $\operatorname{Var}_{q}(f)$ needs only numbers every level already records. It holds across the whole FLUX ladder including $\lambda_{\text{eff}} = 207 \to 214$ (ratios 0.72, 0.85), where $\operatorname{std}_{q}(f)$ had grown to $6.8\times \operatorname{std}_{p}(f)$ and looked alarming. At $N = 16$ each variance carries $\approx 37\%$, so treat ratios in $\approx 0.4$–$2.5$ as consistent. Never read spread growth alone as collapse — `uniq/N` and ESS/N remain the degeneracy signals.

## Resuming From a Checkpoint

Rules are general; the pCN parentheticals are Algorithm 1, the only sampler with a resume path so far.

- **Do** fold the base target the checkpoint already carries into the **kernel**, explicitly (pCN: `beta=1.0, lam=lam_base + beta*lam`, keeping the SMC's own `beta` as the checkpoint key). Resume on the source run's GPU (`--constraint`) with persisted reference latents, and bump `RUN_TAG` per *leg* — output names derive from $N$ / `M_TILT` / `SEED`, so an untagged leg overwrites its own resume source.
- **Don't** treat correct weights as evidence the resume is correct. Ratio-based weights reparametrize exactly (the base cancels); kernels do not (their base is baked in, not an argument), so **only the kernel breaks, and silently**.
- **Don't** resume from a level that stopped on a wall-clock deadline instead of its own convergence rule — finish it at the source setting first. Its convergence baseline probably did not survive the checkpoint, so re-baseline and deliberately over-shoot.
- **Assert** three things: the first resumed level does not regress on the objective; the kernel does not get *easier* across the join; no adaptive step size runs into its clamp. Each means the target is wrong, not that mixing is good.
- **A resumed leg's checkpoint stores $\beta$ relative to *its own* leg, and does not say so.** Leg 2.2 ended at $u = 0.4524$ meaning $\beta = 0.7493$; every consumer must convert via $\beta_{0} + u(1 - \beta_{0})$ from the source's `config["beta0"]`. Both consumers have already got it wrong — the resume (silently mis-tilts the leg) and the *stitch* (leg 3.1's printed ladder labelled $m = 1.63$ as $m = 0.00$ and reported novelty from $\lambda = 150$ instead of from $p$). **Fix it at the writer**: record `beta_true` / `lam_eff_true` in the checkpoint so no reader has to know, exactly as persisting `s` removed that whole class of problem.
- **Don't compare $d\beta$ across legs** as a correctness check. `_next_dbeta` bisects on the *spread* of `fX`, so any finishing pass that moves the cloud legitimately changes it (2.2: `f-sd` $0.0156 \to 0.0220$, $d\beta$ $0.1103 \to 0.1073$). Compare against the source checkpoint's stored `fX` instead — that is the invariant that survives.
- **Budget rejuvenation from the measured rate, not from $s$ you assumed.** At $\lambda_{\text{eff}} \approx 150$, `s` adapts to $\approx 0.47$ (not $\approx 0.8$), giving refresh $\approx \bar{a}(1 - \sqrt{1-s^{2}}) \approx 0.024$/sweep — so `decorr` needs **~60+ sweeps**, not the ~15–20 a larger $s$ would suggest. 2.2 spent 21 sweeps and only reached 0.598.

(Each rule is a failure a real run hit; write-ups in the `flux_strong_tilt_2_*.slurm` headers.)

## Repo Map

- `distances/` — IEM variants (`GlobalIEMDistance`, `SquaredGlobalIEMDistance`, `GeneralizedGlobalIEMDistance` with $f$ = identity/squared, `LocalIEMDistance`, `LpDistance`),
  plus `edm_adapter.py` (`Denoiser` → `score_fn` via Tweedie).
- `refset/` — reference selectors (`RandomRefs`, `FPSRefs`, `WeightedFPSRefs`) with auto-$R$ by a weighted-$\tau$ rank-stability rule; `.reward(R)` / `.normalized_reward(R)`
  produce the frozen reward. **The auto-$R$ rule is biased at high $d$ and the FPS selectors degenerate there** — use `RandomRefs` with an explicit $R$ (currently $R = 64$);
  see Established Findings and `notebooks/refset_auto_r/`.
- `generators/` — $G: z \mapsto x$ via the EDM probability-flow ODE, so $G(N(0,I)) \approx p$ (`toy_2d`, `edm_pixel`, `tiny_sd`, `flux`).
- `tilt.py` — rewards; `density.py` — $p$ as `log_p_X` / `log_p_Y` / sampler.
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

## Jupyter Notebooks

When reading or editing `.ipynb` files, use the **notebook MCP server**.
Do NOT parse, read, or write the notebook JSON directly.

- To read a notebook's contents, use the notebook MCP tools.
- To add, edit, or run cells, use the notebook MCP tools.
- Editing the raw JSON risks corrupting cell structure, outputs, and metadata.