# Creativity-Tilt Generalization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the code in `iem_creativity.ipynb` into a small library that computes the creativity tilt

```
q_λ(x) ∝ p(x) · exp(λ · E_{x'~p}[ D(x', x) ])
```

for **given inputs** `x` (datapoint(s)), `D` (a pluggable distance function), and `p` (a density) — with the expectation taken over an externally **supplied sample of points from `p`** (not generated internally).

**Architecture:** `p` is a `Density` object exposing `log_p_X` and `log_p_Y`. `D` is any object implementing `pairwise(X, x_refs) -> (B, R)`. The creativity score is `expected_distance(D, X, x_refs) = mean_r D(X, x_refs[:,r])`, and the tilt is `log p(x) + λ · score`. Two IEM distances are ported verbatim-in-spirit from the notebook: **local IEM** uses the metric tensor `G(x)` (`compute_G`, was `compute_local_M`) as `D_local(x,x') = √((x-x')ᵀ G(x)(x-x'))`; **global IEM** uses the shared-Brownian-path score-difference quadratic variation, with the IEM "activation" `f` exposed as a **pluggable input function** — `f_identity` (the notebook's standard IEM, default) and `f_square` (quadratic) are provided. Both IEM distances take the reference set `x_refs` as an argument, replacing the notebook's internal `sample_from_p_obs`. A Euclidean distance is included as the trivial baseline `D`.

**Tech Stack:** PyTorch (autodiff, vmap, jacfwd/jacrev), NumPy, Matplotlib, Python 3.10+

---

## What changes vs. the notebook

| Notebook (`iem_creativity.ipynb`) | This library |
|---|---|
| Module-level `MEANS/COVS/...` + `log_pX`, `log_p_Y` | `Density` object passed in |
| `sample_from_p_obs(num_refs)` **inside** `f_global` | `x_refs` passed **in** as an argument (a sample from `p`) |
| Local score hard-coded as `f_C = log tr M(x)` | Local IEM expressed as a **pairwise distance** `D_local(x,x')` from `G(x)`, fitting `E_{x'~p}[D]` |
| `M(x)` naming | `G(x)` naming (paper Thm. 2 local Riemannian metric) |
| Global IEM = the only `E_{x'~p}[D]` | Any `D` plugs into the same `expected_distance` + tilt |
| `q` built only from `f_C` on a grid | `tilted_log_density` works for any `D`, any `x` batch, plus 2D grid normalization |

**Note on `D_local`:** the form `D_local(x,x') = √((x-x')ᵀ G(x)(x-x'))` is used **only** for the local-IEM distance (it is the second-order Taylor metric of the IEM around `x`). Global IEM uses its own pairwise `D_IEM`; Euclidean uses `‖x-x'‖`.

---

## Usage (target API)

```python
import torch
from creativity_measure import (
    Density, EuclideanDistance, LocalIEMDistance, GlobalIEMDistance,
    expected_distance, tilted_log_density, grid_normalize, make_grid,
)

# 1) p as a Density (here a GMM via torch.distributions; any callable pair works)
p = Density(log_p_X=my_log_pX, log_p_Y=my_log_pY, sample=my_sampler)

# 2) a sample of reference points from p  (supplied, NOT drawn internally)
x_refs = p.sample(64)                          # (R, d)

# 3) pick a distance D
from creativity_measure import f_identity, f_square
GAMMAS_LOCAL  = torch.logspace(-4, 4, 200, base=2, dtype=torch.float64)
GAMMAS_GLOBAL = torch.logspace(-10, 10, 200, base=2, dtype=torch.float64)
D = GlobalIEMDistance(p, GAMMAS_GLOBAL, num_eps=50, f=f_identity)   # f_square for quadratic
#   or LocalIEMDistance(p, GAMMAS_LOCAL)
#   or EuclideanDistance()

# 4) compute p(x)·exp(λ·E_{x'~p}[D(x',x)])  for given x
X = grid_points                                # (B, d)
log_q_unnorm = tilted_log_density(X, p, D, x_refs, lam=5.0)   # (B,)  = log p(x) + λ·E[D]

# 5) (2D) normalize on a grid to get q and Z_λ
log_q, q, Z = grid_normalize(log_q_unnorm, cell_area)
```

`tilted_log_density` returns the **unnormalized** log of `p(x)·exp(λ·E_{x'~p}[D(x',x)])`. `Z_λ = ∫ p·exp(λ·E[D]) dx` is a single constant per λ; `grid_normalize` approximates it by a Riemann sum (the notebook's approach). For sampling you never need `Z_λ` (it cancels).

---

## File Structure

```
creativity-measure/
├── creativity_measure/
│   ├── __init__.py              # public API re-exports
│   ├── density.py               # Density (log_p_X, log_p_Y, optional sample)
│   ├── distances/
│   │   ├── __init__.py
│   │   ├── base.py              # Distance protocol + EuclideanDistance
│   │   ├── utils.py             # log_p_Y_given_X (analytic N(γx, γI))
│   │   ├── local_iem.py         # compute_G, LocalIEMDistance
│   │   └── global_iem.py        # score_diff_y, D_IEM_sq_one_to_many, GlobalIEMDistance
│   ├── tilt.py                  # expected_distance, tilted_log_density, grid_normalize
│   └── plotting.py              # 2D contour + scatter helpers
└── demo.ipynb                   # ring-GMM-with-hole, local & global, reproduces notebook
```

**Key design decisions:**
- `D` is duck-typed: anything with `pairwise(X, x_refs) -> (B, R)` works (`expected_distance` does the `.mean(1)`).
- `x_refs` is always an argument — `p` only needs `log_p_X`, `log_p_Y`; sampling is the caller's job (`Density.sample` is a convenience).
- Ported notebook functions keep their structure; only the density coupling and the reference source are generalized.
- `Z_λ` handled by grid Riemann sum in 2D (`grid_normalize`); not needed for sampling.

---

## Task 1: Density + analytic conditional (`density.py`, `distances/utils.py`)

**Files:**
- Create: `creativity_measure/density.py`
- Create: `creativity_measure/distances/utils.py`

**What it does:** `Density` is a thin holder of the two evaluable density functions the notebook used (`log_pX`, `log_p_Y`) plus an optional sampler. `log_p_Y_given_X` is the always-analytic `N(γx, γI)` conditional.

- [ ] **Step 1: Create `creativity_measure/density.py`**

```python
# creativity_measure/density.py
import torch


class Density:
    """
    A probability distribution p, represented by evaluable log-densities.

    Generalizes the notebook's module-level log_pX / log_p_Y into an object.

    Args:
        log_p_X: callable (x: (..., d)) -> (...)        log p_X(x)
        log_p_Y: callable (y: (..., d), gamma) -> (...) log p_{Y_gamma}(y),
                 the marginal of Y = gamma*X + sqrt(gamma)*W, W ~ N(0, I).
                 For a GMM this is analytic (see demo). Used by both IEM
                 distances (Hessian for local, score for global).
        sample:  optional callable (n: int) -> (n, d). Convenience for drawing
                 reference points x' ~ p; the library never calls it internally
                 inside the distances -- x_refs is always passed in.
        d:       optional int, the dimensionality (else inferred on first use).
    """

    def __init__(self, log_p_X, log_p_Y, sample=None, d=None):
        self._log_p_X = log_p_X
        self._log_p_Y = log_p_Y
        self._sample = sample
        self.d = d

    def log_p_X(self, x):
        return self._log_p_X(x)

    def log_p_Y(self, y, gamma):
        return self._log_p_Y(y, gamma)

    def log_p_Y_scalar(self, y, gamma):
        """y: (d,), gamma scalar -> scalar. Needed by jacfwd/jacrev (local IEM)."""
        return self._log_p_Y(y.unsqueeze(0), gamma).squeeze(0)

    def sample(self, n, seed=None):
        if self._sample is None:
            raise RuntimeError("This Density has no sampler; pass x_refs explicitly.")
        if seed is not None:
            torch.manual_seed(seed)
        return self._sample(n)
```

- [ ] **Step 2: Create `creativity_measure/distances/utils.py`**

```python
# creativity_measure/distances/utils.py
# log p(y | x, gamma) = log N(y; gamma*x, gamma*I).
# Ported from iem_creativity.ipynb (cell 3, log_p_Y_given_X).
import torch
from torch.distributions import MultivariateNormal


def log_p_Y_given_X(y, x, gamma):
    """y, x: (..., d)   gamma: scalar  ->  (...)."""
    d = x.shape[-1]
    cov = float(gamma) * torch.eye(d, device=y.device, dtype=y.dtype)
    return MultivariateNormal(float(gamma) * x, cov).log_prob(y)
```

- [ ] **Step 3: Create empty `creativity_measure/__init__.py` and `creativity_measure/distances/__init__.py`** (populated in Task 6).

- [ ] **Step 4: Smoke-test**

```python
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from creativity_measure.density import Density

dtype = torch.float64
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)
covs  = 0.09 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
mix   = MixtureSameFamily(Categorical(torch.ones(2)),
                          MultivariateNormal(means, covs))

def log_pX(x): return mix.log_prob(x)
def log_pY(y, g):
    loc = g * means
    cov = g**2 * covs + g * torch.eye(2, dtype=dtype)
    return MixtureSameFamily(Categorical(torch.ones(2)),
                             MultivariateNormal(loc, cov)).log_prob(y)

p = Density(log_pX, log_pY, sample=lambda n: mix.sample((n,)), d=2)
print(p.log_p_X(torch.zeros(3, 2, dtype=dtype)).shape)        # (3,)
print(p.log_p_Y(torch.zeros(3, 2, dtype=dtype), torch.tensor(1.0)).shape)  # (3,)
print(p.sample(5).shape)                                      # (5, 2)
```

---

## Task 2: Distance interface + Euclidean baseline (`distances/base.py`)

**Files:**
- Create: `creativity_measure/distances/base.py`

**What it does:** Defines the `Distance` protocol (`pairwise(X, x_refs) -> (B, R)`) and a trivial `EuclideanDistance` — both a baseline `D` and a test oracle for the tilt machinery.

- [ ] **Step 1: Create `creativity_measure/distances/base.py`**

```python
# creativity_measure/distances/base.py
import torch
from typing import Protocol


class Distance(Protocol):
    """Any D usable in q_lambda. Returns pairwise D(X[b], x_refs[r])."""
    def pairwise(self, X: torch.Tensor, x_refs: torch.Tensor) -> torch.Tensor:
        """X: (B, d), x_refs: (R, d) -> (B, R)."""
        ...


class EuclideanDistance:
    """D(x, x') = ||x - x'||_2. Baseline distance and test oracle."""
    def pairwise(self, X, x_refs):
        return torch.cdist(X, x_refs)            # (B, R)
```

- [ ] **Step 2: Test (Euclidean pairwise shape + value)**

```python
import torch
from creativity_measure.distances.base import EuclideanDistance

X = torch.tensor([[0., 0.], [3., 4.]])
refs = torch.tensor([[0., 0.]])
D = EuclideanDistance()
out = D.pairwise(X, refs)
assert out.shape == (2, 1)
assert torch.allclose(out[:, 0], torch.tensor([0., 5.]))
```

---

## Task 3: Local IEM distance (`distances/local_iem.py`)

**Files:**
- Create: `creativity_measure/distances/local_iem.py`

**What it does:** Ports the notebook's `compute_local_M` (renamed `compute_G`, generalized to `d` dims and to a passed-in `log_p_Y_scalar`) and exposes `LocalIEMDistance` with `D_local(x,x') = √((x-x')ᵀ G(x)(x-x'))`.

Ported from `iem_creativity.ipynb` section 2 (cells 4–5) and `information-estimation-metric/paper_plots/gmm_local.py` (`compute_H_per_x`). Implements the local Riemannian metric `G(x)` of Fiquet et al., ICLR 2026, Theorem 2 Eq. (5):
`G(x) = ∫₀^∞ γ² E[(∇²_y log p_Y(y;γ))²] dγ`.

- [ ] **Step 1: Create `creativity_measure/distances/local_iem.py`**

```python
# creativity_measure/distances/local_iem.py
import torch
from torch.func import vmap, jacrev, jacfwd
from creativity_measure.density import Density


def compute_G(X, log_p_Y_scalar, gammas, num_noises=50, seed=123):
    """
    Local IEM metric tensor G(x) (paper Thm. 2, Eq. 5). Was compute_local_M
    in the notebook; generalized to d dims and a passed-in scalar log p_Y.

    Args:
        X:             (B, d)
        log_p_Y_scalar: callable (y: (d,), gamma scalar) -> scalar
        gammas:        (N_gamma,)  e.g. logspace(-4, 4, 200, base=2)
        num_noises:    MC samples for E[...] over w_gamma
        seed:          RNG seed

    Returns:
        G: (B, d, d) positive semi-definite metric tensor per point.
    """
    torch.manual_seed(seed)
    hess_fn = jacfwd(jacrev(log_p_Y_scalar, argnums=0), argnums=0)

    X = X.to(dtype=gammas.dtype)
    dgam = gammas[1:] - gammas[:-1]
    B, d = X.shape
    num_gamma = gammas.shape[0]

    eps = torch.randn((num_noises, B, num_gamma, d), device=X.device, dtype=X.dtype)
    ggrid = gammas.view(1, 1, num_gamma, 1)
    y = ggrid * X.view(1, B, 1, d) + ggrid.sqrt() * eps      # (num_noises, B, N_gamma, d)
    y_flat = y.reshape(-1, d)
    g_flat = ggrid.expand(num_noises, B, num_gamma, 1).reshape(-1)

    H = vmap(hess_fn)(y_flat, g_flat).view(num_noises * B * num_gamma, d, d)
    H2 = torch.bmm(H, H).view(num_noises, B, num_gamma, d, d).mean(0)   # (B, N_gamma, d, d)
    weighted = H2 * (gammas ** 2).view(1, num_gamma, 1, 1)
    G = (weighted[:, :-1] * dgam.view(1, num_gamma - 1, 1, 1)).sum(1)   # (B, d, d)
    return G


class LocalIEMDistance:
    """
    Local IEM distance:  D_local(x, x') = sqrt( (x-x')^T G(x) (x-x') ).

    G(x) is computed once per evaluation point x; the quadratic form is then
    cheap against every reference x'. Valid as a local (Taylor) metric; for
    far references it is the locally-adaptive-Mahalanobis extrapolation.

    Note: uses vmap(jacfwd(jacrev(log_p_Y))). If log_p_Y uses ops without a
    vmap rule (e.g. torch.special.log_ndtr), swap compute_G for the batched-
    autograd variant in iem_creativity.ipynb section 6.C.
    """

    def __init__(self, density: Density, gammas, num_noises=50, seed=123):
        self.density = density
        self.gammas = gammas
        self.num_noises = num_noises
        self.seed = seed

    def pairwise(self, X, x_refs):
        """X: (B, d), x_refs: (R, d) -> (B, R)."""
        G = compute_G(X, self.density.log_p_Y_scalar, self.gammas,
                      num_noises=self.num_noises, seed=self.seed)     # (B, d, d)
        diff = X.unsqueeze(1) - x_refs.unsqueeze(0)                   # (B, R, d)
        Gdiff = torch.einsum('bde,bre->brd', G, diff)                 # (B, R, d)
        qf = (diff * Gdiff).sum(-1).clamp_min(0.0)                    # (B, R)
        return qf.sqrt()
```

- [ ] **Step 2: Smoke-test (G is PSD; hole midpoint farther than near-mode)**

```python
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from creativity_measure.density import Density
from creativity_measure.distances.local_iem import compute_G, LocalIEMDistance

dtype = torch.float64
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)
covs  = 0.09 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
def log_pY(y, g):
    loc = g * means; cov = g**2 * covs + g * torch.eye(2, dtype=dtype)
    return MixtureSameFamily(Categorical(torch.ones(2)),
                             MultivariateNormal(loc, cov)).log_prob(y)
p = Density(lambda x: x.sum(-1)*0, log_pY, d=2)   # log_p_X unused here

gammas = torch.logspace(-4, 4, 40, base=2, dtype=dtype)   # coarse for speed
X = torch.tensor([[2., 0.], [0., 0.]], dtype=dtype)
G = compute_G(X, p.log_p_Y_scalar, gammas, num_noises=5, seed=0)
assert G.shape == (2, 2, 2)
# PSD check
evals = torch.linalg.eigvalsh(G)
assert (evals >= -1e-8).all()
```

---

## Task 4: Global IEM distance (`distances/global_iem.py`)

**Files:**
- Create: `creativity_measure/distances/global_iem.py`

**What it does:** Ports the notebook's `score_diff_y` and the SDE/path machinery of `D_IEM_sq_one_to_many` / `f_global` (section 3, cell 7), generalized to: (a) a passed-in `Density`, (b) a passed-in `x_refs` (replacing `sample_from_p_obs`), and (c) a **pluggable activation `f`**. The core now also accumulates the log-ratio process `z_γ` (needed by `f_square`); `f_identity` ignores it and recovers the notebook's standard IEM exactly.

`z_γ` decomposition (derived for the notebook's `y = γx + W` parameterization; matches `information_estimation_metric.py` with `eps_i = −score_diff(y_i, x_i, γ)`):
- `diffusion_coef = eps1 − eps2`, `dqv = ‖eps1−eps2‖²·dγ`
- `drift = 0.5(‖eps1‖²−‖eps2‖²)·dγ`, `stochastic = ⟨eps1−eps2, dWγ⟩`, `z_γ = cumsum(drift+stochastic)`

Activation functions adapted from `information-estimation-metric/information_estimation_metric.py` (lines 28-39), renamed `f_identity` / `f_square` per request. `alpha = 1/d`.

- [ ] **Step 1: Create `creativity_measure/distances/global_iem.py`**

```python
# creativity_measure/distances/global_iem.py
#
# Global IEM pairwise distance (Fiquet et al., ICLR 2026, Defs. 1 & 2):
#   D_IEM_f^2(x1,x2) = integral E_W[ f'(z_g)^2 * || s(Y1,x1,g) - s(Y2,x2,g) ||^2 ] dg
#   s(y,x,g) = grad_y log p(y|x,g) - grad_y log p_Y(y;g),  shared path W.
#   f = identity recovers the notebook's standard IEM (Def. 1).
#
# Ported from iem_creativity.ipynb section 3 (cell 7); generalized to a Density,
# an explicit x_refs (replacing sample_from_p_obs), and a pluggable f.
# Activations adapted from information_estimation_metric.py (lines 28-39).

import torch
from creativity_measure.density import Density
from creativity_measure.distances.utils import log_p_Y_given_X


# ----------------------------------------------------------------------
# Pluggable IEM activations. Signature: (z_gamma, dqv, alpha) -> (..., G),
# summing the integrand over the gamma axis (dim 0).
# ----------------------------------------------------------------------

def f_identity(z_gamma, dqv, alpha):
    """Standard IEM, f = identity (f'(z) = 1). Integrand = dqv. (Notebook default.)"""
    return dqv.sum(0)


def f_square(z_gamma, dqv, alpha):
    """Quadratic IEM, f(z) = z^2 (f'(z) = 2z). Integrand = (2*alpha*z_gamma)^2 * dqv."""
    return ((2 * alpha * z_gamma).pow(2) * dqv).sum(0)


def score_diff_y(y, x, gamma, density: Density):
    """s(y, x, gamma). y: (B, d), x: (B, d), gamma scalar -> (B, d)."""
    y = y.detach().clone().requires_grad_(True)
    g1 = torch.autograd.grad(log_p_Y_given_X(y, x, gamma).sum(), y)[0]
    g2 = torch.autograd.grad(density.log_p_Y(y, gamma).sum(), y)[0]
    return (g1 - g2).detach()


def sde_elements_one_to_many(x_ref, X, W, dW, gammas, density: Density):
    """
    Generalizes the notebook's D_IEM_sq_one_to_many: instead of only the
    quadratic-variation sum, returns the per-gamma SDE elements so a pluggable
    f can be applied.

    x_ref: (1, d), X: (G, d), W: (N_gamma, N_eps, 1, d), dW: (N_gamma-1, N_eps, 1, d)
    Returns:
        z_gamma: (N_gamma-1, N_eps, G)   accumulated log-ratio process
        dqv:     (N_gamma-1, N_eps, G)   quadratic-variation increments
    """
    num_gamma, num_eps = W.shape[0], W.shape[1]
    G = X.shape[0]
    d = X.shape[1]
    dgam = gammas[1:] - gammas[:-1]

    y1_path = gammas.view(-1, 1, 1, 1) * x_ref.view(1, 1, 1, d) + W
    y2_path = gammas.view(-1, 1, 1, 1) * X.view(1, 1, G, d) + W
    x_ref_b = x_ref.view(1, d).expand(num_eps, d).contiguous()
    X_b     = X.view(1, G, d).expand(num_eps, G, d).reshape(num_eps * G, d).contiguous()

    dqv_list, dz_list = [], []
    for i in range(num_gamma - 1):
        gamma = gammas[i]
        y1 = y1_path[i].reshape(num_eps, d)
        y2 = y2_path[i].reshape(num_eps * G, d)
        s1 = score_diff_y(y1, x_ref_b, gamma, density).view(num_eps, 1, d)
        s2 = score_diff_y(y2, X_b, gamma, density).view(num_eps, G, d)
        e1, e2 = -s1, -s2                                  # eps_i = -score_diff (repo convention)
        diff = e1 - e2                                     # (N_eps, G, d)
        dqv_i = diff.pow(2).sum(-1) * dgam[i]              # (N_eps, G)
        drift_i = 0.5 * (e1.pow(2).sum(-1) - e2.pow(2).sum(-1)) * dgam[i]  # (N_eps, G)
        dw_i = dW[i].reshape(num_eps, 1, d)
        stoch_i = (diff * dw_i).sum(-1)                    # (N_eps, G)
        dqv_list.append(dqv_i)
        dz_list.append(drift_i + stoch_i)

    dqv = torch.stack(dqv_list, dim=0)                     # (N_gamma-1, N_eps, G)
    dz = torch.stack(dz_list, dim=0)
    z = torch.cumsum(dz, dim=0)
    z = torch.cat([torch.zeros_like(z[:1]), z[:-1]], dim=0)  # z_0 = 0, left-aligned
    return z, dqv


class GlobalIEMDistance:
    """
    Global IEM distance D_IEM_f(x, x') via shared-path score differences.
    Builds one Brownian path bank and evaluates every reference against the
    whole X batch, applying the pluggable activation f.

    Args:
        density:  Density (must expose log_p_Y)
        gammas:   integration grid, e.g. logspace(-10, 10, 200, base=2)
        num_eps:  Brownian path samples (variance reduction)
        f:        activation, f_identity (default, = notebook standard IEM)
                  or f_square. Signature (z_gamma, dqv, alpha) -> (N_eps, G).
        seed:     RNG seed for the Brownian path bank
    """

    def __init__(self, density: Density, gammas, num_eps=50, f=f_identity,
                 seed=123, verbose=False):
        self.density = density
        self.gammas = gammas
        self.num_eps = num_eps
        self.f = f
        self.seed = seed
        self.verbose = verbose

    def _brownian(self, d, device, dtype):
        num_gamma = self.gammas.shape[0]
        dgam = self.gammas[1:] - self.gammas[:-1]
        torch.manual_seed(self.seed)
        dW = torch.randn(num_gamma - 1, self.num_eps, 1, d,
                         device=device, dtype=dtype) * dgam.sqrt().view(-1, 1, 1, 1)
        W = torch.zeros(num_gamma, self.num_eps, 1, d, device=device, dtype=dtype)
        W[1:] = torch.cumsum(dW, dim=0)
        return W, dW

    def pairwise(self, X, x_refs):
        """X: (B, d), x_refs: (R, d) -> (B, R)  with D_IEM_f(X[b], x_refs[r])."""
        device, dtype = X.device, X.dtype
        d = X.shape[1]
        alpha = 1.0 / d
        W, dW = self._brownian(d, device, dtype)
        R = x_refs.shape[0]
        cols = []
        for r in range(R):
            z, dqv = sde_elements_one_to_many(
                x_refs[r:r+1], X, W, dW, self.gammas, self.density)
            iem_sq = self.f(z, dqv, alpha)                  # (N_eps, G)
            cols.append(iem_sq.mean(0).clamp_min(0).sqrt())  # (B,) = D_IEM_f(X, x_refs[r])
            if self.verbose and (r + 1) % max(1, R // 4) == 0:
                print(f'  ref {r+1}/{R}')
        return torch.stack(cols, dim=1)                     # (B, R)
```

- [ ] **Step 2: Smoke-test (shapes, non-negativity, f_identity vs f_square both run)**

```python
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from creativity_measure.density import Density
from creativity_measure.distances.global_iem import GlobalIEMDistance, f_identity, f_square

dtype = torch.float64
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)
covs  = 0.09 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
mix   = MixtureSameFamily(Categorical(torch.ones(2)), MultivariateNormal(means, covs))
def log_pY(y, g):
    loc = g*means; cov = g**2*covs + g*torch.eye(2, dtype=dtype)
    return MixtureSameFamily(Categorical(torch.ones(2)),
                             MultivariateNormal(loc, cov)).log_prob(y)
p = Density(mix.log_prob, log_pY, sample=lambda n: mix.sample((n,)), d=2)

gammas = torch.logspace(-10, 10, 40, base=2, dtype=dtype)
X = torch.tensor([[2., 0.], [0., 0.]], dtype=dtype)
x_refs = p.sample(4, seed=0)

for f in (f_identity, f_square):
    out = GlobalIEMDistance(p, gammas, num_eps=8, f=f).pairwise(X, x_refs)
    assert out.shape == (2, 4)
    assert (out >= 0).all()
```

---

## Task 5: Tilt + expectation + normalization (`tilt.py`)

**Files:**
- Create: `creativity_measure/tilt.py`

**What it does:** The generic glue: `expected_distance` (the `E_{x'~p}` mean over the supplied refs), `tilted_log_density` (`log p + λ·E[D]`), and `grid_normalize` (the notebook's Riemann-sum `Z_λ`).

- [ ] **Step 1: Create `creativity_measure/tilt.py`**

```python
# creativity_measure/tilt.py
import torch
from creativity_measure.density import Density


def expected_distance(distance, X, x_refs):
    """
    E_{x'~p}[ D(x', X) ] approximated by the mean over the supplied refs.
    distance: object with pairwise(X, x_refs) -> (B, R).
    X: (B, d), x_refs: (R, d) -> (B,).
    """
    return distance.pairwise(X, x_refs).mean(dim=1)


def tilted_log_density(X, density: Density, distance, x_refs, lam):
    """
    Unnormalized log of  q_lambda(x) ∝ p(x) * exp(lambda * E_{x'~p}[D(x', x)]).

    Returns log p(X) + lambda * E_{x'~p}[D(x', X)],  shape (B,).
    (The normalizer Z_lambda is omitted; use grid_normalize for 2D, or
    ignore it for sampling where it cancels.)
    """
    score = expected_distance(distance, X, x_refs)        # (B,)
    return density.log_p_X(X) + lam * score


def grid_normalize(log_q_unnorm, cell_area):
    """
    Normalize an unnormalized log-density evaluated on a regular grid.
    Mirrors iem_creativity.ipynb cell 17 (Riemann sum for Z_lambda).

    Args:
        log_q_unnorm: (G,) or (gn, gn) tensor of log q (unnormalized)
        cell_area:    dx*dy of one grid cell

    Returns:
        log_q:  normalized log-density (same shape)
        q:      normalized density      (same shape)
        Z:      scalar Z_lambda ≈ ∫ exp(log_q_unnorm) dx
    """
    shape = log_q_unnorm.shape
    flat = log_q_unnorm.reshape(-1)
    flat = flat - flat.nanmax()                  # stabilize
    q_un = flat.exp()
    Z = q_un.sum() * cell_area
    q = (q_un / Z).reshape(shape)
    return q.clamp_min(1e-300).log(), q, Z
```

- [ ] **Step 2: Test (Euclidean tilt sanity — λ=0 recovers log p; λ>0 raises far-from-refs points)**

```python
import torch
from creativity_measure.density import Density
from creativity_measure.distances.base import EuclideanDistance
from creativity_measure.tilt import expected_distance, tilted_log_density

p = Density(lambda x: torch.zeros(x.shape[0]), lambda y, g: None, d=2)  # flat log p
D = EuclideanDistance()
X = torch.tensor([[0., 0.], [10., 0.]])
x_refs = torch.zeros(5, 2)

assert torch.allclose(tilted_log_density(X, p, D, x_refs, lam=0.0),
                      torch.zeros(2))                       # λ=0 -> log p
ed = expected_distance(D, X, x_refs)
assert ed[1] > ed[0]                                        # far point has larger E[D]
```

---

## Task 6: Plotting + public API (`plotting.py`, `__init__.py`)

**Files:**
- Create: `creativity_measure/plotting.py`
- Modify: `creativity_measure/__init__.py`, `creativity_measure/distances/__init__.py`

**What it does:** 2D grid + contour + scatter helpers, and the public re-exports.

- [ ] **Step 1: Create `creativity_measure/plotting.py`**

```python
# creativity_measure/plotting.py
import torch
import numpy as np
import matplotlib.pyplot as plt


def make_grid(xlim, ylim, grid_n=50, device="cpu", dtype=torch.float64):
    """Returns (grid_points (G,2), XX, YY, cell_area)."""
    xs = torch.linspace(xlim[0], xlim[1], grid_n, device=device, dtype=dtype)
    ys = torch.linspace(ylim[0], ylim[1], grid_n, device=device, dtype=dtype)
    XX, YY = torch.meshgrid(xs, ys, indexing='xy')
    grid_points = torch.stack([XX, YY], dim=-1).reshape(-1, 2)
    dx = (xlim[1] - xlim[0]) / (grid_n - 1)
    dy = (ylim[1] - ylim[0]) / (grid_n - 1)
    return grid_points, XX, YY, float(dx * dy)


def plot_field(vals, XX, YY, ax=None, title="", cmap="viridis", levels=20,
               marked=None, missing=None):
    """Filled contour of a scalar field on a meshgrid."""
    if ax is None:
        _, ax = plt.subplots()
    Z = vals.reshape(XX.shape)
    Z = Z.cpu().numpy() if isinstance(Z, torch.Tensor) else Z
    XXn = XX.cpu().numpy() if isinstance(XX, torch.Tensor) else XX
    YYn = YY.cpu().numpy() if isinstance(YY, torch.Tensor) else YY
    cf = ax.contourf(XXn, YYn, Z, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax, shrink=0.8)
    ax.set_title(title); ax.set_aspect('equal')
    if marked is not None:
        m = marked.cpu().numpy() if isinstance(marked, torch.Tensor) else marked
        ax.scatter(m[:, 0], m[:, 1], marker='x', c='orange', s=25)
    if missing is not None:
        mm = missing.cpu().numpy() if isinstance(missing, torch.Tensor) else missing
        ax.scatter(mm[:, 0], mm[:, 1], marker='*', c='red', s=120,
                   edgecolors='white', linewidths=0.5, zorder=5)
    return ax


def plot_samples(samples, ax=None, title="", alpha=0.3, s=5, color='steelblue'):
    if ax is None:
        _, ax = plt.subplots()
    pts = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    ax.scatter(pts[:, 0], pts[:, 1], alpha=alpha, s=s, color=color)
    ax.set_title(title); ax.set_aspect('equal')
    return ax
```

- [ ] **Step 2: Populate `creativity_measure/distances/__init__.py`**

```python
from creativity_measure.distances.base import Distance, EuclideanDistance
from creativity_measure.distances.local_iem import compute_G, LocalIEMDistance
from creativity_measure.distances.global_iem import (
    GlobalIEMDistance, score_diff_y, sde_elements_one_to_many,
    f_identity, f_square,
)
```

- [ ] **Step 3: Populate `creativity_measure/__init__.py`**

```python
from creativity_measure.density import Density
from creativity_measure.distances import (
    Distance, EuclideanDistance, compute_G, LocalIEMDistance,
    GlobalIEMDistance, score_diff_y, sde_elements_one_to_many,
    f_identity, f_square,
)
from creativity_measure.tilt import (
    expected_distance, tilted_log_density, grid_normalize,
)
from creativity_measure.plotting import make_grid, plot_field, plot_samples
```

---

## Task 7: Demo notebook (`demo.ipynb`)

**Files:**
- Create: `demo.ipynb` (in `creativity-measure/` root)

**What it does:** Reproduces the notebook's ring-GMM-with-hole, but through the general API, building `q_λ` from **both** local and global IEM distances, with `x_refs` supplied externally.

**Cell 1 — Setup**
```python
import sys, math, torch, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, '.')
from creativity_measure import *
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

device, dtype = 'cpu', torch.float64
torch.set_default_dtype(dtype)
```

**Cell 2 — Ring GMM with a hole as a Density (matches notebook section 1)**
```python
N_TOTAL, HOLE_IDX, RADIUS, SIGMA = 12, 0, 4.0, 0.3
angles = 2*math.pi*torch.arange(N_TOTAL, dtype=dtype)/N_TOTAL
all_means = torch.stack([RADIUS*torch.cos(angles), RADIUS*torch.sin(angles)], -1)
mask = torch.ones(N_TOTAL, dtype=torch.bool); mask[HOLE_IDX] = False
MEANS = all_means[mask]
K = MEANS.shape[0]
COVS = (SIGMA**2)*torch.eye(2, dtype=dtype).expand(K, -1, -1).contiguous()
W = torch.ones(K)/K

def make_mix(loc, cov): return MixtureSameFamily(Categorical(W), MultivariateNormal(loc, cov))
def log_pX(x): return make_mix(MEANS, COVS).log_prob(x)
def log_pY(y, g):
    loc = g*MEANS; cov = g**2*COVS + g*torch.eye(2, dtype=dtype)
    return make_mix(loc, cov).log_prob(y)

p = Density(log_pX, log_pY, sample=lambda n: make_mix(MEANS, COVS).sample((n,)), d=2)
```

**Cell 3 — Grid + supplied reference sample from p**
```python
grid_points, XX, YY, cell_area = make_grid((-6, 6), (-6, 6), grid_n=50, dtype=dtype)
with torch.no_grad():
    log_p_grid = p.log_p_X(grid_points)
x_refs = p.sample(32, seed=123)          # supplied, NOT drawn inside the distances
```

**Cell 4 — Choose a distance and compute the tilt (local IEM)**
```python
GAMMAS_LOCAL = torch.logspace(-4, 4, 200, base=2, dtype=dtype)
D_local = LocalIEMDistance(p, GAMMAS_LOCAL, num_noises=50, seed=123)

LAM = 5.0
CHUNK = 128
log_q_un = torch.empty(grid_points.shape[0], dtype=dtype)
for i in range(0, grid_points.shape[0], CHUNK):
    log_q_un[i:i+CHUNK] = tilted_log_density(grid_points[i:i+CHUNK], p, D_local, x_refs, lam=LAM)
log_q, q, Z = grid_normalize(log_q_un, cell_area)
print("Z_lambda =", float(Z))
```

**Cell 5 — Plot log p, E[D] score, log q (local)**
```python
score_local = expected_distance(D_local, grid_points, x_refs)   # for display
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_field(log_p_grid, XX, YY, ax=axes[0], title=r"$\log p(x)$", marked=MEANS, missing=all_means[~mask])
plot_field(score_local, XX, YY, ax=axes[1], title=r"local: $E_{x'}[D(x',x)]$", marked=MEANS, missing=all_means[~mask])
plot_field(log_q, XX, YY, ax=axes[2], title=rf"$\log q_\lambda$ (λ={LAM})", marked=MEANS, missing=all_means[~mask])
plt.tight_layout(); plt.show()
```

**Cell 6 — Same pipeline, global IEM (swap the distance only; f_identity = notebook standard, f_square optional)**
```python
GAMMAS_GLOBAL = torch.logspace(-10, 10, 200, base=2, dtype=dtype)
D_global = GlobalIEMDistance(p, GAMMAS_GLOBAL, num_eps=50, f=f_identity, seed=123, verbose=True)
# quadratic activation instead:  GlobalIEMDistance(p, GAMMAS_GLOBAL, num_eps=50, f=f_square)

score_global = expected_distance(D_global, grid_points, x_refs)
log_q_un_g = p.log_p_X(grid_points) + LAM * score_global
log_q_g, q_g, Z_g = grid_normalize(log_q_un_g, cell_area)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_field(log_p_grid, XX, YY, ax=axes[0], title=r"$\log p(x)$", marked=MEANS, missing=all_means[~mask])
plot_field(score_global, XX, YY, ax=axes[1], title=r"global: $E_{x'}[D_{IEM}(x',x)]$", marked=MEANS, missing=all_means[~mask])
plot_field(log_q_g, XX, YY, ax=axes[2], title=rf"$\log q_\lambda$ global (λ={LAM})", marked=MEANS, missing=all_means[~mask])
plt.tight_layout(); plt.show()
```

**Cell 7 — Euclidean baseline (shows pluggability of D)**
```python
D_euc = EuclideanDistance()
score_euc = expected_distance(D_euc, grid_points, x_refs)
log_q_un_e = p.log_p_X(grid_points) + LAM * score_euc
log_q_e, q_e, _ = grid_normalize(log_q_un_e, cell_area)
plot_field(log_q_e, XX, YY, title=rf"$\log q_\lambda$ Euclidean (λ={LAM})", marked=MEANS, missing=all_means[~mask])
plt.show()
```

---

## Self-Review

**Spec coverage:**
- ✅ Generalizes the notebook: `Density` replaces module-level `log_pX/log_p_Y`; `compute_G` is the notebook's `compute_local_M`; `score_diff_y` and the SDE machinery (`sde_elements_one_to_many`, generalizing `D_IEM_sq_one_to_many`) ported verbatim-in-spirit.
- ✅ Computes `p(x)·exp(λ·E_{x'~p}[D(x',x)])` for given `x`, `D`, `p` via `tilted_log_density`.
- ✅ `D` is a pluggable input (`Distance` protocol): local IEM, global IEM, Euclidean — all interchangeable.
- ✅ Local IEM uses `D_local(x,x') = √((x-x')ᵀ G(x)(x-x'))` (G-notation), used only for local IEM as decided.
- ✅ Global IEM takes a **pluggable activation `f`** as input: `f_identity` (notebook standard IEM, default) and `f_square` (quadratic). The `z_γ` process is accumulated so `f_square` works.
- ✅ Reference points `x_refs` are **supplied** (a sample from `p`); the distances never call `sample_from_p_obs` internally.
- ✅ `Z_λ` via grid Riemann sum (`grid_normalize`); not needed for sampling.
- ✅ Demo reproduces the ring-with-hole for local, global (`f_identity`), and Euclidean `D`.

**Deferred (not in this plan):** sampling from `q_λ` (importance / SMC); denoiser-backed high-d `score`/`log_p_Y`; `learned_f` activation (a trainable `f` — the `f` interface accepts it, but no training). The `Distance` interface and `Density` are structured so these are additive.

**Type consistency:** every distance exposes `pairwise(X:(B,d), x_refs:(R,d)) -> (B,R)`; `expected_distance` returns `(B,)`; `tilted_log_density` returns `(B,)`. Consistent across all tasks and the demo. ✅
