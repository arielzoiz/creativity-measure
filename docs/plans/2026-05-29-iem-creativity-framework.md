# IEM Creativity Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python framework that takes a user-provided tensor of samples from `p`, computes a creativity score `f(x) = E_{x'~p}[D(x, x')]` for `D` = **local IEM** or **global IEM** distance, and lets the user sample from the creativity-tilted distribution `q_λ(x) ∝ p(x)·exp(λ·f(x))` via importance sampling. The global IEM is structured around the SDE elements `(z_γ, dqv)` so the generalized IEM with a function `f` (Definition 2 of the paper / `information_estimation_metric.py`) plugs in as a reduction.

**Architecture:** The single `Density` class wraps a tensor of samples from `p` and derives everything from them — `log_p_Y` via a Monte Carlo logsumexp approximation (differentiable, used for the IEM Hessian and score), and `sample()` by resampling the tensor. For high-d data where the MC approximation degrades, the user can pass a `score_fn` (e.g. a trained denoiser) to override the score computation. The creativity score is always `E_{x'~p}[D(x, x')]`: for global IEM `D` is the proper pairwise distance; for local IEM `D(x,x') = √((x-x')ᵀM(x)(x-x'))` from the metric tensor. The global IEM core returns `(z_γ, dqv)` and applies a **reduction** (`standard` now; `square_f`/`learned_f` are thin add-ons that consume `z_γ`). `q_λ` is never constructed as a function — the user evaluates `f` on samples from `p` and reweights via importance sampling, which is completely agnostic to which `D`/reduction produced the scores. For 2D, a grid-based path also supports contour plots and exact normalization.

**Scope note (basics only):** This plan implements the building blocks and the `standard` global reduction. The `square_f` and `learned_f` reductions are included as small functions (they consume the already-computed `z_γ`), but actually *learning* an `f`, wiring up real denoisers for high-d data, and tuning are deliberately left for follow-up work. The architecture is set up so those are additive, not rewrites.

**Tech Stack:** PyTorch (autodiff, vmap, jacfwd/jacrev), NumPy, Matplotlib, Python 3.10+

---

## Usage

### How to represent `p`

The only required input is a tensor of samples from `p`. How you get them depends on your use case:

**From `torch.distributions`:**
```python
import torch
from torch.distributions import MultivariateNormal
dist = MultivariateNormal(torch.zeros(2), torch.eye(2))
samples = dist.sample((1000,))          # (1000, 2)
p = Density(samples)
```

**From a real dataset:**
```python
samples = torch.tensor(your_numpy_array)   # (N, d)
p = Density(samples)
```

**From any custom density — just sample from it (e.g. uniform on a disk/ring):**
```python
samples = my_custom_sampler(n=5000)        # (N, d) tensor
p = Density(samples)
# log_p_Y has no closed form for a uniform-on-geometry -> MC path is used.
```

**For a 2D Gaussian / GMM — supply the analytic log_p_Y (exact, no MC noise):**
```python
from torch.distributions import MultivariateNormal
mu, Sigma = torch.zeros(2), torch.eye(2)
samples = MultivariateNormal(mu, Sigma).sample((5000,))   # still needed for refs/sampling

def gaussian_log_p_Y(y, gamma):            # Y = gamma*X + sqrt(gamma)*W ~ N(gamma*mu, gamma^2*Sigma + gamma*I)
    cov = gamma**2 * Sigma + gamma * torch.eye(2)
    return MultivariateNormal(gamma * mu, cov).log_prob(y)

p = Density(samples, log_p_Y_fn=gaussian_log_p_Y)
# Recommended for Gaussians: autograd through the analytic form gives an
# EXACT score and Hessian -- local IEM is clean even at small gamma.
```

**For high-d data with a trained denoiser (replaces MC score approximation):**
```python
samples = dataset_tensor                   # (N, d)
def score_fn(y, gamma):
    # y: (B, d), gamma: scalar -> (B, d)
    # e.g. wraps a diffusion model's denoiser output via Tweedie
    return denoiser_score(y, gamma)

p = Density(samples, score_fn=score_fn)    # log_p_Y / local IEM unavailable; global IEM only
```

### What `Density` derives from its inputs

| Method | How derived | Used for |
|---|---|---|
| `sample(n)` | uniform resample from tensor | reference points in IEM, importance sampling proposal |
| `log_p_Y(y, γ)` | analytic `log_p_Y_fn` if given, else MC logsumexp | local IEM Hessian, global IEM score (via autograd) |
| `score_pY(y, γ)` | autograd through `log_p_Y`, OR `score_fn` if given | global IEM score difference |

| Input mode | `log_p_Y` | Score / Hessian | Local IEM | Typical use |
|---|---|---|---|---|
| `Density(samples)` | MC logsumexp | autograd (noisy at small γ) | ✅ | uniform-on-geometry, generic data |
| `Density(samples, log_p_Y_fn=...)` | analytic | autograd (exact) | ✅ | 2D Gaussian, GMM, box-uniform |
| `Density(samples, score_fn=...)` | unavailable | from `score_fn` | ❌ | high-d images (denoiser) |

### Choosing the distance `D` for the creativity score

The creativity score is always `f(x) = E_{x'~p}[D(x, x')]`. Three interchangeable ways to compute it:

```python
# Global IEM, standard (Def. 1)
f_vals = f_global(x_samples, p, GAMMAS_GLOBAL, iem_type="standard")

# Global IEM, generalized with a function f (Def. 2) -- e.g. f(z)=z^2
f_vals = f_global(x_samples, p, GAMMAS_GLOBAL, iem_type="square_f")
f_vals = f_global(x_samples, p, GAMMAS_GLOBAL, iem_type="learned_f", learned_f=my_net)

# Local IEM, pairwise distance via the metric tensor M(x)
f_vals = f_local_expected_distance(x_samples, p, GAMMAS_LOCAL, num_refs=32)
```

All three return `(N,)` scores that drop into the same importance sampler below.

### How `q_λ` works — importance sampling

There is no explicit `q_λ` function. Instead:

```python
# 1. Draw proposal samples from p
x_samples = p.sample(n=5000)

# 2. Compute the creativity score on each sample (any of the options above)
f_vals = f_global(x_samples, p, GAMMAS_GLOBAL, iem_type="standard")

# 3. Reweight: w_i ∝ exp(λ · f(x_i))
samples_q, weights, ess = importance_sample_q(x_samples, f_vals, lam=5.0, n_out=500)

print(f"ESS = {ess:.1f} / {len(x_samples)}")   # monitor quality
```

`samples_q` are samples approximately from `q_λ`. The sampler is agnostic to which `D` or reduction produced `f_vals`. If ESS is too low (< 10% of proposal size), increase `n_proposal` or reduce `lam`.

### For 2D: grid-based plotting

```python
grid_points, XX, YY = make_grid(xlim=(-6,6), ylim=(-6,6), grid_n=50)
log_p_grid = p.log_p_Y_grid(grid_points)        # evaluate log p on grid
score_grid = f_local_log_trace(grid_points, p, gammas)

log_q_grid, q_grid = compute_q_grid(log_p_grid, score_grid, lam=5.0)
compare_densities(log_p_grid, score_grid, log_q_grid, XX, YY)
```

### Local vs Global IEM

| | Local `f_C(x) = log tr M(x)` | Global `f_G(x) = E[D_IEM(x,x')]` |
|---|---|---|
| Cost | O(d²) per point | O(d) per score eval |
| Quality of MC `log_p_Y` | Needs clean Hessian — noisy for small N or high d | Score is robust for moderate d |
| Recommended for | Low-d (d ≤ ~10), large N | Any d; use `score_fn` for high d |
| γ range | `logspace(-4, 4, 200, base=2)` | `logspace(-10, 10, 200, base=2)` |

---

## File Structure

```
creativity-measure/
├── creativity_measure/
│   ├── __init__.py              # public API re-exports
│   ├── density.py               # Density class: samples + optional score_fn
│   ├── iem/
│   │   ├── __init__.py
│   │   ├── utils.py             # log_p_Y_given_X (always analytic: N(γx, γI))
│   │   ├── local.py             # compute_M(x), f_local_log_trace, f_local_trace, f_local_log_det
│   │   └── global_.py           # f_global: E_{x'~p}[D_IEM(x, x')]
│   ├── gibbs.py                 # importance_sample_q: resample from p by exp(λ·f) weights
│   └── plotting.py              # 2D grid plots, sample scatter, compare_densities
└── demo.ipynb                   # Interactive ring example
```

**Key design decisions:**

- `Density` takes **only samples** — no subclassing, no ABC, no pre-implemented distributions. The user obtains samples however they want (torch.distributions, real data, custom sampler) and passes the tensor.
- `log_p_Y` is derived via **MC logsumexp** by default — differentiable, so autograd gives the Hessian (local IEM) and score (global IEM). Quality degrades with d and improves with N.
- `log_p_Y_fn` override — for toy densities with a closed-form noisy marginal (Gaussian: `Y~N(γμ, γ²Σ+γI)`; GMM; box-uniform via erf), the user passes an analytic `log_p_Y_fn`. Autograd then gives **exact** score and Hessian (no MC noise, important for local IEM at small γ). Samples are still used for sampling and reference points.
- `score_fn` override — for high-d data the user passes a denoiser callable; `log_p_Y` is then unavailable and local IEM cannot be used.
- Three input modes: (1) `Density(samples)` — MC for everything (uniform-on-geometry, generic data); (2) `Density(samples, log_p_Y_fn=...)` — analytic `log_p_Y`, exact autograd score/Hessian (Gaussian, GMM, box-uniform); (3) `Density(samples, score_fn=...)` — denoiser score only, no local IEM (high-d images). `log_p_Y_fn` and `score_fn` are mutually exclusive.
- `q_λ` is never a function — it exists only as **weighted samples** from importance sampling. For 2D, a grid-based normalized array is also provided for plotting.
- `Z_λ` is never computed — it cancels in importance weights and is approximated by grid normalization in 2D.

---

## Task 1: Density Class (`density.py`)

**Files:**
- Create: `creativity_measure/density.py`

**What it does:** Wraps a tensor of samples into a `Density` object that provides `log_p_Y` (analytic if `log_p_Y_fn` given, else MC), `score_pY`, and `sample`.

- [ ] **Step 1: Create `creativity_measure/density.py`**

```python
# creativity_measure/density.py
import math
import torch
from torch.distributions import MultivariateNormal


class Density:
    """
    Represents a probability distribution p via a tensor of samples,
    with optional analytic overrides for the noisy marginal / score.

    Three input modes:
      1. Density(samples)                  -- MC logsumexp for log_p_Y and score.
                                              Generic data, uniform-on-geometry.
      2. Density(samples, log_p_Y_fn=...)  -- analytic log_p_Y; autograd gives
                                              EXACT score and Hessian. Gaussian,
                                              GMM, box-uniform (erf). No MC noise.
      3. Density(samples, score_fn=...)    -- denoiser supplies score directly;
                                              log_p_Y unavailable -> no local IEM.
                                              High-d images.

    Args:
        samples:    (N, d) tensor of iid samples from p (always required;
                    used for sampling and as reference points x' ~ p).
        log_p_Y_fn: optional callable (y: (B, d), gamma: scalar) -> (B,)
                    giving the analytic log-density of Y = gamma*X + sqrt(gamma)*W.
                    For Gaussian p_X = N(mu, Sigma): Y ~ N(gamma*mu, gamma^2*Sigma + gamma*I).
        score_fn:   optional callable (y: (B, d), gamma: scalar) -> (B, d)
                    = grad_y log p_Y(y; gamma). For high-d data (e.g. a denoiser).
        device, dtype: defaults inferred from samples.

    log_p_Y_fn and score_fn are mutually exclusive.
    """

    def __init__(self, samples: torch.Tensor, log_p_Y_fn=None, score_fn=None,
                 device=None, dtype=None):
        if log_p_Y_fn is not None and score_fn is not None:
            raise ValueError("Pass at most one of log_p_Y_fn / score_fn.")
        self.device = device or samples.device
        self.dtype = dtype or samples.dtype
        self.samples = samples.to(device=self.device, dtype=self.dtype)
        self._log_p_Y_fn = log_p_Y_fn
        self._score_fn = score_fn
        self.N, self.d = self.samples.shape

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int, seed: int = None) -> torch.Tensor:
        """Draw n samples by uniform resampling. Returns (n, d)."""
        if seed is not None:
            torch.manual_seed(seed)
        idx = torch.randint(self.N, (n,), device=self.device)
        return self.samples[idx]

    # ------------------------------------------------------------------
    # log p_Y : analytic if provided, else MC logsumexp
    # ------------------------------------------------------------------

    def log_p_Y(self, y: torch.Tensor, gamma) -> torch.Tensor:
        """
        log p_Y(y; gamma) = log p_{gamma*X + sqrt(gamma)*W}(y).

        - If log_p_Y_fn was provided: returns it (exact, differentiable).
        - Else: MC approximation logsumexp_i log N(y; gamma*x_i, gamma*I) - log N.

        y: (B, d) or (d,)  gamma: scalar  ->  (B,) or scalar.
        Fully differentiable via autograd in both cases.

        Not available when score_fn is provided (use score_pY instead).
        """
        if self._score_fn is not None:
            raise RuntimeError(
                "log_p_Y is not available when score_fn is provided. "
                "Local IEM requires an MC or analytic log_p_Y -- use "
                "Density(samples) or Density(samples, log_p_Y_fn=...) instead."
            )
        scalar_input = y.dim() == 1
        if scalar_input:
            y = y.unsqueeze(0)                           # (1, d)

        if self._log_p_Y_fn is not None:
            result = self._log_p_Y_fn(y, gamma)          # (B,)
        else:
            # MC logsumexp over samples
            diff = y.unsqueeze(1) - gamma * self.samples.unsqueeze(0)  # (B, N, d)
            log_liks = -0.5 * (
                diff.pow(2).sum(-1) / gamma
                + self.d * math.log(2 * math.pi * float(gamma))
            )                                            # (B, N)
            result = torch.logsumexp(log_liks, dim=1) - math.log(self.N)  # (B,)
        return result.squeeze(0) if scalar_input else result

    def log_p_Y_scalar(self, y: torch.Tensor, gamma) -> torch.Tensor:
        """
        Single-point version of log_p_Y. y: (d,), gamma: scalar -> scalar.
        Required by jacfwd/jacrev for local IEM Hessian computation.
        """
        return self.log_p_Y(y.unsqueeze(0), gamma).squeeze(0)

    # ------------------------------------------------------------------
    # Score of p_Y
    # ------------------------------------------------------------------

    def score_pY(self, y: torch.Tensor, gamma) -> torch.Tensor:
        """
        grad_y log p_Y(y; gamma).  y: (B, d) -> (B, d).

        If score_fn provided: calls it directly (denoiser path).
        Otherwise: autograd through the MC log_p_Y approximation.
        """
        if self._score_fn is not None:
            return self._score_fn(y, gamma)
        y_in = y.detach().requires_grad_(True)
        lp = self.log_p_Y(y_in, gamma)
        return torch.autograd.grad(lp.sum(), y_in)[0].detach()
```

- [ ] **Step 2: Create package skeleton**

```
creativity_measure/__init__.py      (empty for now)
creativity_measure/iem/__init__.py  (empty for now)
```

- [ ] **Step 3: Smoke-test**

```python
import torch, math
from creativity_measure.density import Density
from torch.distributions import MultivariateNormal

dtype = torch.float64
y = torch.zeros(3, 2, dtype=dtype)
gamma = torch.tensor(1.0, dtype=dtype)

# --- Mode 1: MC (uniform-on-geometry / generic) ---
dist = MultivariateNormal(torch.zeros(2, dtype=dtype), torch.eye(2, dtype=dtype))
samples = dist.sample((500,))
p_mc = Density(samples)
print(p_mc.log_p_Y(y, gamma).shape)       # (3,)
print(p_mc.score_pY(y, gamma).shape)      # (3, 2)
print(p_mc.sample(10).shape)              # (10, 2)
print(p_mc.log_p_Y_scalar(y[0], gamma))   # scalar

# --- Mode 2: analytic log_p_Y for a 2D Gaussian (exact, no MC noise) ---
mu  = torch.zeros(2, dtype=dtype)
Sig = torch.eye(2, dtype=dtype)
def gaussian_log_p_Y(y_, g):
    cov = g**2 * Sig + g * torch.eye(2, dtype=dtype)   # gamma^2 Sigma + gamma I
    return MultivariateNormal(g * mu, cov).log_prob(y_)
p_exact = Density(samples, log_p_Y_fn=gaussian_log_p_Y)
print(p_exact.log_p_Y(y, gamma).shape)    # (3,)  -- exact, matches the analytic Gaussian
print(p_exact.score_pY(y, gamma).shape)   # (3, 2)  -- exact via autograd
```

---

## Task 2: IEM Utilities (`iem/utils.py`)

**Files:**
- Create: `creativity_measure/iem/utils.py`

**What it does:** Provides `log_p_Y_given_X` — always analytic — used by both local and global IEM.

- [ ] **Step 1: Create `creativity_measure/iem/utils.py`**

```python
# creativity_measure/iem/utils.py
# Copied from iem_creativity.ipynb (cell 3, log_p_Y_given_X)
import torch
from torch.distributions import MultivariateNormal


def log_p_Y_given_X(y, x, gamma, device=None, dtype=None):
    """
    log p(y | x, gamma) = log N(y; gamma*x, gamma*I).
    y, x: (..., d)  gamma: scalar
    """
    device = device or y.device
    dtype = dtype or y.dtype
    d = x.shape[-1]
    cov = float(gamma) * torch.eye(d, device=device, dtype=dtype)
    return MultivariateNormal(float(gamma) * x, cov).log_prob(y)
```

---

## Task 3: Local IEM (`iem/local.py`)

**Files:**
- Create: `creativity_measure/iem/local.py`

**What it does:** Computes M(x) (the local Riemannian metric tensor, Theorem 2 Eq. 5 of Fiquet et al. ICLR 2026) and scalar creativity scores from it. Takes a `Density` object directly — no manual wrapping needed.

Adapted from `iem_creativity.ipynb` section 2 and `information-estimation-metric/paper_plots/gmm_local.py`.

Note: uses `vmap(jacfwd(jacrev(...)))`. If the underlying `log_p_Y` uses operations without vmap support (e.g. `torch.special.log_ndtr`), use the batched-autograd fallback shown in `iem_creativity.ipynb` section 6.C instead.

- [ ] **Step 1: Create `creativity_measure/iem/local.py`**

```python
# creativity_measure/iem/local.py
#
# Implements the local IEM metric tensor M(x) from:
#   Fiquet et al., ICLR 2026, Theorem 2, Eq. (5):
#   M(x) = integral_0^inf gamma^2 * E[(grad^2_y log p_Y(y;gamma))^2] dgamma
#
# Adapted from:
#   iem_creativity.ipynb (section 2, cells 4-5)
#   information-estimation-metric/paper_plots/gmm_local.py (compute_H_per_x)

import torch
from torch.func import vmap, jacrev, jacfwd
from creativity_measure.density import Density


def _make_hessian_fn(log_p_Y_scalar_fn):
    """y:(d,), gamma:scalar -> H:(d,d) via forward-over-reverse autodiff."""
    return jacfwd(jacrev(log_p_Y_scalar_fn, argnums=0), argnums=0)


def compute_M(X: torch.Tensor, density: Density,
              gammas: torch.Tensor, num_noises: int = 50, seed: int = 123):
    """
    Local IEM metric tensor M(x) for a batch of points.

    Args:
        X:         (B, d) evaluation points
        density:   Density object (must have log_p_Y available, i.e. no score_fn)
        gammas:    (N_gamma,) integration grid.
                   Paper default for local IEM: logspace(-4, 4, 200, base=2).
                   Check that the integrand gamma^2 * tr(E[H^2]) is near zero
                   at both endpoints before running the full grid computation.
        num_noises: MC samples for E[...] over w_gamma (paper uses 50)
        seed:      random seed

    Returns:
        M: (B, d, d)  positive semi-definite metric tensor at each point

    Note: O(d^2) cost per point. Not recommended for d > ~20.
    """
    torch.manual_seed(seed)
    hess_fn = _make_hessian_fn(density.log_p_Y_scalar)

    dgam = gammas[1:] - gammas[:-1]
    B, d = X.shape[0], X.shape[1]
    num_gamma = gammas.shape[0]

    eps = torch.randn((num_noises, B, num_gamma, d), device=X.device, dtype=X.dtype)
    ggrid = gammas.view(1, 1, num_gamma, 1)
    y = ggrid * X.view(1, B, 1, d) + ggrid.sqrt() * eps  # (num_noises, B, N_gamma, d)

    y_flat = y.reshape(-1, d)
    g_flat = ggrid.expand(num_noises, B, num_gamma, 1).reshape(-1)

    H = vmap(hess_fn)(y_flat, g_flat)                            # (num_noises*B*N_gamma, d, d)
    H2 = torch.bmm(H, H).view(num_noises, B, num_gamma, d, d).mean(0)  # (B, N_gamma, d, d)

    weighted = H2 * (gammas ** 2).view(1, num_gamma, 1, 1)
    M = (weighted[:, :-1] * dgam.view(1, num_gamma - 1, 1, 1)).sum(1)  # (B, d, d)
    return M


def f_local_log_trace(X: torch.Tensor, density: Density,
                      gammas: torch.Tensor, num_noises: int = 50,
                      seed: int = 123, eps: float = 1e-30) -> torch.Tensor:
    """
    Local IEM creativity score: f_C(x) = log tr M(x).
    Main score used throughout iem_creativity.ipynb.
    Returns (B,).
    """
    M = compute_M(X, density, gammas, num_noises=num_noises, seed=seed)
    tr = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).clamp_min(eps)
    return tr.log()


def f_local_trace(X: torch.Tensor, density: Density,
                  gammas: torch.Tensor, num_noises: int = 50,
                  seed: int = 123) -> torch.Tensor:
    """Raw trace tr M(x). Returns (B,)."""
    M = compute_M(X, density, gammas, num_noises=num_noises, seed=seed)
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


def f_local_log_det(X: torch.Tensor, density: Density,
                    gammas: torch.Tensor, num_noises: int = 50,
                    seed: int = 123, jitter: float = 1e-12) -> torch.Tensor:
    """Log-determinant of M(x). Returns (B,)."""
    M = compute_M(X, density, gammas, num_noises=num_noises, seed=seed)
    d = M.shape[-1]
    M = M + jitter * torch.eye(d, device=M.device, dtype=M.dtype).unsqueeze(0)
    sign, logabsdet = torch.linalg.slogdet(M)
    return torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, float('nan')))


def local_iem_distance(X: torch.Tensor, X_refs: torch.Tensor,
                       M_X: torch.Tensor) -> torch.Tensor:
    """
    Local IEM distance D_local(x, x') = sqrt((x-x')^T M(x) (x-x')),
    the second-order Taylor approximation of the IEM (paper Thm. 2).

    Args:
        X:      (B, d) candidate points
        X_refs: (R, d) reference points
        M_X:    (B, d, d) metric tensor at each candidate (from compute_M)

    Returns:
        (B, R) distances D_local(X[b], X_refs[r])

    Note: only exact for x' near x; for far references this is the
    "locally adaptive Mahalanobis" extrapolation, used here as a heuristic.
    """
    diff = X.unsqueeze(1) - X_refs.unsqueeze(0)          # (B, R, d)
    # quadratic form (B,R,d) x (B,d,d) x (B,R,d) -> (B,R)
    Mdiff = torch.einsum('bij,brj->bri', M_X, diff)      # (B, R, d)
    qf = (diff * Mdiff).sum(-1).clamp_min(0.0)           # (B, R)
    return qf.sqrt()


def f_local_expected_distance(X: torch.Tensor, density: Density,
                              gammas: torch.Tensor, num_refs: int = 32,
                              num_noises: int = 50, seed: int = 123) -> torch.Tensor:
    """
    Local-IEM creativity score: f(x) = E_{x'~p}[ D_local(x, x') ].
    This is the local-distance counterpart to f_global, matching the
    uniform formulation q_lambda ∝ p * exp(lambda * E_{x'~p}[D(x,x')]).

    Args:
        X:         (B, d) evaluation points
        density:   Density (must have log_p_Y available; no score_fn)
        gammas:    local-IEM integration grid (e.g. logspace(-4,4,200,base=2))
        num_refs:  number of reference samples x' ~ p
        num_noises, seed: forwarded to compute_M

    Returns: (B,) creativity score per point
    """
    M_X = compute_M(X, density, gammas, num_noises=num_noises, seed=seed)  # (B,d,d)
    refs = density.sample(num_refs, seed=seed)                            # (R,d)
    D = local_iem_distance(X, refs, M_X)                                  # (B,R)
    return D.mean(dim=1)
```

- [ ] **Step 2: Smoke-test**

```python
import torch
from torch.distributions import MultivariateNormal
from creativity_measure.density import Density
from creativity_measure.iem.local import compute_M, f_local_log_trace

# Two-mode distribution: modes at [4,0] and [-4,0]
torch.manual_seed(0)
dist = MultivariateNormal(
    torch.tensor([[4., 0.], [-4., 0.]], dtype=torch.float64),
    0.09 * torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(2,-1,-1)
)
# Sample from mixture manually
idx = torch.randint(2, (2000,))
samples = dist.sample((2000,))[torch.arange(2000), idx] if False else \
    torch.cat([dist[0].sample((1000,)), dist[1].sample((1000,))], dim=0) \
    if False else \
    torch.cat([
        MultivariateNormal(torch.tensor([4.,0.],dtype=torch.float64),
                           0.09*torch.eye(2,dtype=torch.float64)).sample((1000,)),
        MultivariateNormal(torch.tensor([-4.,0.],dtype=torch.float64),
                           0.09*torch.eye(2,dtype=torch.float64)).sample((1000,))
    ])

p = Density(samples)
gammas = torch.logspace(-4, 4, 30, base=2, dtype=torch.float64)  # coarse
X = torch.tensor([[4., 0.], [0., 0.]], dtype=torch.float64)
score = f_local_log_trace(X, p, gammas, num_noises=5, seed=0)
print("score at mode:", score[0].item())      # lower
print("score at midpoint:", score[1].item())  # higher
```

---

## Task 4: Global IEM (`iem/global_.py`)

**Files:**
- Create: `creativity_measure/iem/global_.py`

**What it does:** Computes `f_G(x) = E_{x'~p}[D_IEMf(x, x')]` using `density.score_pY` and `density.sample`. Works with both MC and denoiser-based `score_pY`. The core returns the SDE elements `(z_γ, dqv)`; a **reduction** turns them into the (generalized) IEM. This is the seam that makes the function-`f` extension additive.

`z_γ` math (derived for the notebook's γ-parameterization `y_i = γx_i + W`, matching `information_estimation_metric.py` when `eps_i = −score_diff(y_i, x_i, γ)`):
- `diffusion_coef = eps1 − eps2`,  `dqv = ‖eps1−eps2‖²·dγ`
- `drift = 0.5(‖eps1‖²−‖eps2‖²)·dγ`,  `stochastic = ⟨eps1−eps2, dwγ⟩`
- `z_γ = cumsum(drift + stochastic)` over γ

Reductions adapted from `information-estimation-metric/information_estimation_metric.py` (lines 28-39); SDE/path machinery adapted from `iem_creativity.ipynb` section 3.

- [ ] **Step 1: Create `creativity_measure/iem/global_.py`**

```python
# creativity_measure/iem/global_.py
#
# Implements the (generalized) global IEM from:
#   Fiquet et al., ICLR 2026, Definitions 1 & 2.
#   D_IEMf^2(x1,x2) = integral E[ f'(z_g)^2 * || eps1 - eps2 ||^2 ] dg
#   with f = identity recovering the standard IEM (Def. 1).
#
# Reductions copied/adapted from:
#   information-estimation-metric/information_estimation_metric.py (lines 28-39)
# SDE elements / Brownian path adapted from:
#   iem_creativity.ipynb (section 3, cell 7)

import torch
from creativity_measure.density import Density
from creativity_measure.iem.utils import log_p_Y_given_X


# ----------------------------------------------------------------------
# Reductions: (z_gamma, dqv) -> IEM^2.  alpha = 1/d (paper convention).
# Only `standard` is exercised by the basics; the other two are the
# documented extension point for the generalized IEM with function f.
# ----------------------------------------------------------------------

def standard_iem(z_gamma, dqv, alpha=None, learned_f=None):
    """f = identity (Def. 1). IEM^2 = sum_gamma dqv. dqv: (N_gamma, ...)."""
    return dqv.sum(0)


def square_f_iem(z_gamma, dqv, alpha, learned_f=None):
    """f(z) = z^2  =>  f'(z) = 2z. Copied from information_estimation_metric.py."""
    return ((2 * alpha * z_gamma).pow(2) * dqv).sum(0)


def learned_f_iem(z_gamma, dqv, alpha, learned_f):
    """Learned f. learned_f maps alpha*z_gamma -> f'(.). Copied from repo."""
    integrand = learned_f(alpha * z_gamma).pow(2) * dqv
    return integrand.sum(0)


_REDUCTIONS = {
    "standard": standard_iem,
    "square_f": square_f_iem,
    "learned_f": learned_f_iem,
}


def _eps(y, x, gamma, density: Density):
    """
    eps(y, x, gamma) = -score_diff = grad_y log p_Y(y;g) - grad_y log p(y|x,g).
    Equivalently the denoising-error vector. y,x: (B,d), gamma scalar -> (B,d).
    """
    device, dtype = y.device, y.dtype
    y_in = y.detach().clone().requires_grad_(True)
    g1 = torch.autograd.grad(
        log_p_Y_given_X(y_in, x, gamma, device=device, dtype=dtype).sum(), y_in
    )[0]                                   # grad_y log p(y|x,gamma)
    g2 = density.score_pY(y, gamma)        # grad_y log p_Y(y;gamma)
    return (g2 - g1).detach()              # = -score_diff = eps


def _sde_elements_one_to_many(x_ref, X, W, dW, gammas, density: Density):
    """
    Compute (z_gamma, dqv) for D(x_ref, X[g]) for all g, shared path W.

    x_ref: (1, d), X: (G, d)
    W:  (N_gamma, N_eps, 1, d)   Brownian path values
    dW: (N_gamma-1, N_eps, 1, d) Brownian increments
    Returns:
        z_gamma: (N_gamma-1, N_eps, G)
        dqv:     (N_gamma-1, N_eps, G)
    """
    num_gamma, num_eps = W.shape[0], W.shape[1]
    G = X.shape[0]
    dgam = gammas[1:] - gammas[:-1]
    device, dtype = X.device, X.dtype

    y1_path = gammas.view(-1, 1, 1, 1) * x_ref.view(1, 1, 1, -1) + W   # (N_gamma,N_eps,1,d)
    y2_path = gammas.view(-1, 1, 1, 1) * X.view(1, 1, G, -1) + W       # (N_gamma,N_eps,G,d)

    x_ref_b = x_ref.view(1, -1).expand(num_eps, -1).contiguous()
    X_b     = X.view(1, G, -1).expand(num_eps, G, -1).reshape(num_eps * G, -1).contiguous()

    dqv_list, dz_list = [], []
    for i in range(num_gamma - 1):
        gamma = gammas[i]
        y1 = y1_path[i].reshape(num_eps, -1)            # (N_eps, d)
        y2 = y2_path[i].reshape(num_eps * G, -1)        # (N_eps*G, d)
        e1 = _eps(y1, x_ref_b, gamma, density).view(num_eps, 1, -1)  # (N_eps,1,d)
        e2 = _eps(y2, X_b, gamma, density).view(num_eps, G, -1)      # (N_eps,G,d)
        diff = e1 - e2                                  # (N_eps, G, d)

        dqv_i = diff.pow(2).sum(-1) * dgam[i]           # (N_eps, G)
        # drift: 0.5*(||e1||^2 - ||e2||^2)*dgamma
        drift_i = 0.5 * (e1.pow(2).sum(-1) - e2.pow(2).sum(-1)) * dgam[i]  # (N_eps,G)
        # stochastic: <diff, dW_i>
        dw_i = dW[i].reshape(num_eps, 1, -1)            # (N_eps,1,d)
        stoch_i = (diff * dw_i).sum(-1)                 # (N_eps, G)
        dqv_list.append(dqv_i)
        dz_list.append(drift_i + stoch_i)

    dqv = torch.stack(dqv_list, dim=0)                  # (N_gamma-1, N_eps, G)
    dz = torch.stack(dz_list, dim=0)
    z_gamma = torch.cumsum(dz, dim=0)                   # (N_gamma-1, N_eps, G)
    # prepend z_0 = 0 and drop last to align with left Riemann sum
    z_gamma = torch.cat([torch.zeros_like(z_gamma[:1]), z_gamma[:-1]], dim=0)
    return z_gamma, dqv


def f_global(X: torch.Tensor, density: Density, gammas: torch.Tensor,
             num_eps: int = 50, num_refs: int = 32,
             iem_type: str = "standard", learned_f=None,
             seed: int = 123, verbose: bool = True) -> torch.Tensor:
    """
    Global IEM creativity score: f_G(x) = E_{x'~p}[ D_IEMf(x, x') ].

    Works with any Density -- both MC score_pY and denoiser score_fn.

    Args:
        X:        (G, d) evaluation points (e.g. samples from p)
        density:  Density object
        gammas:   integration grid. Paper default: logspace(-10, 10, 200, base=2)
        num_eps:  Brownian path samples for variance reduction (paper uses 50)
        num_refs: reference samples x' ~ p (paper uses 1; notebook uses 32)
        iem_type: "standard" | "square_f" | "learned_f"
        learned_f: required iff iem_type == "learned_f"
        seed:     random seed

    Returns: (G,) score per evaluation point
    """
    device, dtype = X.device, X.dtype
    torch.manual_seed(seed)
    if iem_type not in _REDUCTIONS:
        raise ValueError(f"unknown iem_type {iem_type!r}")
    reduce_fn = _REDUCTIONS[iem_type]
    alpha = 1.0 / X.shape[-1]

    refs = density.sample(num_refs, seed=seed).to(device=device, dtype=dtype)
    num_gamma = gammas.shape[0]
    dgam = gammas[1:] - gammas[:-1]

    dW = torch.randn(num_gamma - 1, num_eps, 1, X.shape[-1],
                     device=device, dtype=dtype) * dgam.sqrt().view(-1, 1, 1, 1)
    W = torch.zeros(num_gamma, num_eps, 1, X.shape[-1], device=device, dtype=dtype)
    W[1:] = torch.cumsum(dW, dim=0)

    G = X.shape[0]
    accum = torch.zeros(G, device=device, dtype=dtype)
    for m in range(num_refs):
        z_gamma, dqv = _sde_elements_one_to_many(refs[m:m+1], X, W, dW, gammas, density)
        iem_sq = reduce_fn(z_gamma, dqv, alpha=alpha, learned_f=learned_f)  # (N_eps, G)
        accum += iem_sq.mean(0).clamp_min(0).sqrt()
        if verbose and (m + 1) % max(1, num_refs // 4) == 0:
            print(f'  ref {m+1}/{num_refs}')
    return accum / num_refs
```

- [ ] **Step 2: Sanity-check that `iem_type="standard"` matches a direct quadratic-variation sum**

```python
# The standard reduction should equal the plain sum of (eps1-eps2)^2 * dgamma.
# Verify f_global(..., iem_type="standard") gives finite, non-negative scores
# that are larger in the hole region than at a mode (same qualitative check
# as the local IEM smoke test).
```

---

## Task 5: Importance Sampling (`gibbs.py`)

**Files:**
- Create: `creativity_measure/gibbs.py`

**What it does:** Given samples from `p` and their precomputed IEM scores, produces weighted samples from `q_λ` via importance sampling. Reports ESS to diagnose weight degeneracy.

- [ ] **Step 1: Create `creativity_measure/gibbs.py`**

```python
# creativity_measure/gibbs.py
import torch
from typing import Tuple


def importance_sample_q(
    x_proposal: torch.Tensor,
    f_vals: torch.Tensor,
    lam: float,
    n_out: int = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Sample from q_lambda(x) ∝ p(x) * exp(lambda * f(x)) via importance sampling.

    Since x_proposal ~ p, the importance weight is simply:
        w_i ∝ q(x_i) / p(x_i) = exp(lambda * f(x_i))

    Args:
        x_proposal: (N, d) samples drawn from p
        f_vals:     (N,) IEM score f(x_i) for each proposal sample
        lam:        lambda (Gibbs temperature)
        n_out:      number of output samples (default: N, with replacement)
        seed:       random seed for resampling

    Returns:
        samples_q:  (n_out, d) samples approximately from q_lambda
        weights:    (N,) normalized importance weights (sum to 1)
        ess:        effective sample size = (sum w_i)^2 / sum w_i^2
                    Monitor: ESS/N < 0.1 means weights have collapsed --
                    increase N or reduce lam.
    """
    torch.manual_seed(seed)
    n_out = n_out or x_proposal.shape[0]

    log_w = lam * f_vals
    log_w = log_w - log_w.max()              # numerical stability
    w = log_w.exp()
    weights = w / w.sum()                    # normalized

    ess = (weights.sum() ** 2 / weights.pow(2).sum()).item()

    idx = torch.multinomial(weights, num_samples=n_out, replacement=True)
    return x_proposal[idx], weights, ess


def compute_q_grid(
    log_p_grid: torch.Tensor,
    score_grid: torch.Tensor,
    lam: float,
    cell_area: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    2D grid-based q_lambda. Normalized over the grid.

    Args:
        log_p_grid:  (G,) or (grid_n, grid_n) log p values on grid
        score_grid:  (G,) or (grid_n, grid_n) f(x) values on grid
        lam:         lambda
        cell_area:   dx*dy per grid cell (for proper normalization)

    Returns:
        log_q_grid:  (same shape as input) normalized log q on grid
        q_grid:      (same shape) normalized q density values
    """
    shape = log_p_grid.shape
    lp = log_p_grid.reshape(-1)
    sc = score_grid.reshape(-1)

    log_q_un = lp + lam * sc
    log_q_un = log_q_un - log_q_un.nanmax()   # stabilize
    q_un = log_q_un.exp()
    Z = q_un.sum() * cell_area
    q_norm = (q_un / Z).reshape(shape)
    log_q_norm = q_norm.log().reshape(shape)
    return log_q_norm, q_norm
```

---

## Task 6: Plotting (`plotting.py`)

**Files:**
- Create: `creativity_measure/plotting.py`

**What it does:** 2D contour plots and sample scatter. Grid utilities for building evaluation grids.

- [ ] **Step 1: Create `creativity_measure/plotting.py`**

```python
# creativity_measure/plotting.py
import torch
import numpy as np
import matplotlib.pyplot as plt


def make_grid(xlim, ylim, grid_n=50, device="cpu", dtype=torch.float64):
    """
    Build a 2D evaluation grid.
    Returns (grid_points (G,2), XX (grid_n,grid_n), YY (grid_n,grid_n), cell_area float).
    """
    xs = torch.linspace(xlim[0], xlim[1], grid_n, device=device, dtype=dtype)
    ys = torch.linspace(ylim[0], ylim[1], grid_n, device=device, dtype=dtype)
    XX, YY = torch.meshgrid(xs, ys, indexing='xy')
    grid_points = torch.stack([XX, YY], dim=-1).reshape(-1, 2)
    dx = (xlim[1] - xlim[0]) / (grid_n - 1)
    dy = (ylim[1] - ylim[0]) / (grid_n - 1)
    return grid_points, XX, YY, float(dx * dy)


def plot_log_density(log_vals, XX, YY, ax=None, title="", cmap="viridis",
                     levels=20, marked_points=None, marked_label=""):
    """Contour plot of a log-density on a meshgrid."""
    if ax is None:
        _, ax = plt.subplots()
    Z = np.array(log_vals.reshape(XX.shape).cpu() if isinstance(log_vals, torch.Tensor)
                 else log_vals.reshape(XX.shape))
    XXn = XX.cpu().numpy() if isinstance(XX, torch.Tensor) else XX
    YYn = YY.cpu().numpy() if isinstance(YY, torch.Tensor) else YY
    cf = ax.contourf(XXn, YYn, Z, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax)
    ax.set_title(title)
    ax.set_aspect('equal')
    if marked_points is not None:
        pts = marked_points.cpu().numpy() if isinstance(marked_points, torch.Tensor) else marked_points
        ax.scatter(pts[:, 0], pts[:, 1], c='white', marker='x', s=60,
                   linewidths=1.5, label=marked_label, zorder=5)
        if marked_label:
            ax.legend(fontsize=8)
    return ax


def plot_samples(samples, ax=None, title="", alpha=0.3, s=5, color='steelblue'):
    """Scatter plot of 2D samples."""
    if ax is None:
        _, ax = plt.subplots()
    pts = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    ax.scatter(pts[:, 0], pts[:, 1], alpha=alpha, s=s, color=color)
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax


def compare_densities(log_p_grid, score_grid, log_q_grid, XX, YY,
                      marked_points=None, figsize=(15, 4)):
    """Side-by-side: log p | IEM score f(x) | log q_lambda."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_log_density(log_p_grid, XX, YY, ax=axes[0],
                     title=r"$\log\,p(x)$", marked_points=marked_points)
    plot_log_density(score_grid, XX, YY, ax=axes[1],
                     title=r"$f(x)$ (IEM score)", marked_points=marked_points)
    plot_log_density(log_q_grid, XX, YY, ax=axes[2],
                     title=r"$\log\,q_\lambda(x)$", marked_points=marked_points)
    plt.tight_layout()
    return fig, axes
```

---

## Task 7: Public API (`__init__.py`)

**Files:**
- Modify: `creativity_measure/__init__.py`

- [ ] **Step 1: Populate `creativity_measure/__init__.py`**

```python
from creativity_measure.density import Density
from creativity_measure.gibbs import importance_sample_q, compute_q_grid
from creativity_measure.plotting import (
    make_grid, plot_log_density, plot_samples, compare_densities
)
from creativity_measure.iem.local import (
    compute_M, f_local_log_trace, f_local_trace, f_local_log_det,
    local_iem_distance, f_local_expected_distance,
)
from creativity_measure.iem.global_ import (
    f_global, standard_iem, square_f_iem, learned_f_iem,
)
```

---

## Task 8: Demo Notebook (`demo.ipynb`)

**Files:**
- Create: `demo.ipynb` (in `creativity-measure/` root, next to the package)

**What it does:** End-to-end example on a ring distribution with a hole, matching `iem_creativity.ipynb`. Demonstrates both IEM modes, grid-based plotting, and importance sampling from `q_λ`.

**Cell 1 — Setup**
```python
import sys, torch, math, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, '.')
from creativity_measure import *

device, dtype = 'cpu', torch.float64
torch.set_default_dtype(dtype)
```

**Cell 2 — Build ring distribution with hole and draw samples**
```python
# Matches iem_creativity.ipynb section 1
N_TOTAL, HOLE_IDX, RADIUS, SIGMA = 12, 0, 4.0, 0.3
angles = 2 * math.pi * torch.arange(N_TOTAL, dtype=dtype) / N_TOTAL
all_means = torch.stack([RADIUS*torch.cos(angles), RADIUS*torch.sin(angles)], dim=-1)
mask = torch.ones(N_TOTAL, dtype=torch.bool); mask[HOLE_IDX] = False
means = all_means[mask]   # (11, 2)

from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
cov = (SIGMA**2) * torch.eye(2, dtype=dtype)
mix  = Categorical(torch.ones(11)/11)
comp = torch.distributions.Independent(
    MultivariateNormal(means, cov.unsqueeze(0).expand(11,-1,-1)), 0)
ring_dist = MixtureSameFamily(mix, comp)

samples = ring_dist.sample((5000,))   # (5000, 2) -- this is our representation of p
p = Density(samples)
print("p built from", p.N, "samples,  d =", p.d)
```

**Cell 3 — 2D grid plotting of p**
```python
xlim, ylim = (-6., 6.), (-6., 6.)
grid_points, XX, YY, cell_area = make_grid(xlim, ylim, grid_n=50, device=device, dtype=dtype)

# Evaluate log p on grid using MC approximation
with torch.no_grad():
    log_p_grid = p.log_p_Y(grid_points, torch.tensor(1e6, dtype=dtype))
    # large gamma → p_Y ≈ p_X  (alternative: evaluate ring_dist.log_prob directly)
```

**Cell 4 — Local IEM score on grid (low-d, MC-based)**
```python
GAMMAS_LOCAL = torch.logspace(-4, 4, 200, base=2, dtype=dtype)

CHUNK = 128
score_local = torch.empty(grid_points.shape[0], dtype=dtype)
for i in range(0, grid_points.shape[0], CHUNK):
    score_local[i:i+CHUNK] = f_local_log_trace(
        grid_points[i:i+CHUNK], p, GAMMAS_LOCAL, num_noises=50, seed=123)
```

**Cell 5 — Alternative scores: global IEM (standard / generalized) and local distance**
```python
GAMMAS_GLOBAL = torch.logspace(-10, 10, 200, base=2, dtype=dtype)

# Global IEM, standard reduction (Def. 1)
score_global = f_global(grid_points, p, GAMMAS_GLOBAL,
                        num_eps=50, num_refs=32, iem_type="standard", seed=123)

# Global IEM, generalized with f(z)=z^2 (Def. 2) -- same call, different reduction
# score_global_sq = f_global(grid_points, p, GAMMAS_GLOBAL, iem_type="square_f", seed=123)

# Local IEM as a pairwise distance: E_{x'~p}[ D_local(x, x') ]
# score_local_dist = f_local_expected_distance(grid_points, p, GAMMAS_LOCAL, num_refs=32)
```

**Cell 6 — Grid-based q_λ and plot**
```python
LAM = 5.0
dx = (xlim[1]-xlim[0])/(50-1); cell_area = dx**2
log_q_grid, q_grid = compute_q_grid(log_p_grid, score_local, lam=LAM, cell_area=cell_area)
compare_densities(log_p_grid, score_local, log_q_grid, XX, YY, marked_points=all_means)
```

**Cell 7 — Importance sampling from q_λ**
```python
# Evaluate score on fresh proposal samples from p
x_proposal = p.sample(5000, seed=1)
f_proposal  = f_local_log_trace(x_proposal, p, GAMMAS_LOCAL, num_noises=50, seed=123)

samples_q, weights, ess = importance_sample_q(x_proposal, f_proposal, lam=LAM, n_out=500)
print(f"ESS = {ess:.1f} / {len(x_proposal)}  ({100*ess/len(x_proposal):.1f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_samples(x_proposal, ax=ax1, title="Samples from p", color='steelblue')
plot_samples(samples_q,  ax=ax2, title=f"Samples from q_λ (λ={LAM})", color='tomato')
plt.show()
```

---

## Self-Review

**Spec coverage:**
- ✅ Single `Density` class — takes samples tensor only, no subclassing, no pre-implemented distributions
- ✅ Works with `torch.distributions`, real datasets, custom samplers — user just calls `.sample()` first
- ✅ Sample from `q_λ ∝ p·exp(λ·E_{x'~p}[D(x,x')])` for **D = local IEM** (`f_local_expected_distance`) and **D = global IEM** (`f_global`)
- ✅ Global IEM accepts a function `f` (generalized IEM, Def. 2): `iem_type ∈ {standard, square_f, learned_f}`, structured around `(z_γ, dqv)` SDE elements like `information_estimation_metric.py`
- ✅ `score_fn` override for high-d denoiser case (global IEM)
- ✅ `q_λ` via importance sampling — D-agnostic, no explicit q function needed
- ✅ ESS monitoring for weight degeneracy
- ✅ 2D grid-based plotting and normalization

**Basics-only boundary (deferred to follow-up, by design):**
- Actually *training* a `learned_f` network (the reduction function exists and is wired, but no training loop)
- Real denoiser integration for genuinely high-d data (the `score_fn` hook exists; no k_diffusion wrapper)
- Performance tuning / chunking of the global SDE loop for large grids

**Limitations documented in Usage section:**
- Local IEM: O(d²), MC Hessian noisy for small N or high d — use global IEM for d > ~10
- Local IEM distance for far references is the Mahalanobis extrapolation (heuristic), not the exact IEM
- Global IEM with MC score: degrades in high d — pass `score_fn` (denoiser) for high d
- Importance sampling ESS: collapses at large λ — increase proposal N or reduce λ
- `log_p_Y` unavailable when `score_fn` is provided — local IEM requires MC path

**Type consistency:** `f_local_expected_distance(X, density, gammas)`, `f_local_log_trace(X, density, gammas)`, and `f_global(X, density, gammas, iem_type=...)` all take `(G, d)` for X and return `(G,)` — consistent with `importance_sample_q(x_proposal, f_vals, lam)` which expects `(N,)` scores. The global reduction functions share the signature `reduce_fn(z_gamma, dqv, alpha, learned_f) -> (N_eps, G)`. ✅
