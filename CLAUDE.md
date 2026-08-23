
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

**Current samplers** (`creativity_measure/`):

- `adaptive_tempering_smc.py` — the mature one. Adaptive-tempering SMC over $\beta \in [0,1]$ with a pluggable rejuvenation `Kernel`:
  `PCNKernel` should be used (local prior-preserving pCN moves in the generator's latent Gaussian space, $x = G(z)$ — the workhorse for strong/off-manifold tilts)
  while `IndependenceKernel` is stale.
- `diffusion_smc.py` — **WIP draft**. Twisted-diffusion SMC: guide *inside* the EDM denoising trajectory with a lookahead reward twist instead of moving in data space.
  Currently the approximate (biased, under-tilting) weight; `use_score_correction=True` raises `NotImplementedError` — the unbiased Tweedie/quadrature weight is the follow-up.
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