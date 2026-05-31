"""Tests for creativity_measure.distances.local_iem (Task 3)."""
import math
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

from creativity_measure.density import Density
from creativity_measure.distances.local_iem import compute_G, LocalIEMDistance

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
dtype = torch.float64

# 2-mode GMM: means [2,0] and [-2,0], isotropic sigma^2=0.09, equal weights
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)  # (K, d)
sig2  = 0.09          # component variance for p_X


def log_pX(x):
    """log p_X(x): GMM with distribution constructors (not used under vmap)."""
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(Categorical(torch.ones(2)),
                            MultivariateNormal(means, covs))
    return mix.log_prob(x)


def _log_gaussian(y, mu, var_diag):
    """
    log N(y; mu, var_diag*I) in pure tensor ops — vmap-compatible.

    y:  (..., d)
    mu: (d,)
    var_diag: scalar tensor
    -> (...)
    """
    d = y.shape[-1]
    diff = y - mu                                                  # (..., d)
    return -0.5 * (d * (math.log(2 * math.pi) + var_diag.log())
                   + (diff ** 2).sum(-1) / var_diag)


def log_pY(y, gamma):
    """
    Analytic log p_{Y_gamma}(y) for Y = gamma*X + sqrt(gamma)*W, W~N(0,I).

    Written in pure tensor ops so it is vmap-compatible (no distribution
    constructors, no .item() calls).

    y:     (..., d)
    gamma: scalar tensor
    ->     (...)
    """
    # Component k: N(gamma*means[k], (gamma^2*sig2 + gamma)*I)
    var_diag = gamma ** 2 * sig2 + gamma                           # scalar
    log_pi = -math.log(2)                                          # log(1/K), K=2

    # Compute log N(y; gamma*means[k], var_diag*I) for each component k
    loc = gamma * means                                            # (K, d)
    # log_comps shape: (K, ...) — broadcast y over components
    log_comps = torch.stack([_log_gaussian(y, loc[k], var_diag)
                             for k in range(2)], dim=0)            # (K, ...)
    # log mixture = logsumexp_k (log_pi + log_comp_k)
    return torch.logsumexp(log_pi + log_comps, dim=0)             # (...)


p = Density(log_pX, log_pY, d=2)
log_p_Y_scalar = lambda y, g: log_pY(y.unsqueeze(0), g).squeeze(0)

# Coarse gammas for speed
gammas = torch.logspace(-4, 4, 30, base=2, dtype=dtype)

# Candidate points and references
X      = torch.tensor([[2., 0.], [0., 0.]], dtype=dtype)   # (2, 2)
x_refs = torch.tensor([[2., 0.], [-2., 0.], [0., 0.]], dtype=dtype)  # (3, 2)


# ---------------------------------------------------------------------------
# compute_G tests
# ---------------------------------------------------------------------------

def test_compute_G_shape():
    G = compute_G(X, log_p_Y_scalar, gammas, num_noises=5, seed=0)
    assert G.shape == (2, 2, 2), f"Expected (2, 2, 2), got {G.shape}"


def test_compute_G_psd():
    G = compute_G(X, log_p_Y_scalar, gammas, num_noises=5, seed=0)
    eigvals = torch.linalg.eigvalsh(G)   # (B, d)
    assert (eigvals >= -1e-8).all(), (
        f"G has eigenvalue(s) < -1e-8: min={eigvals.min().item()}"
    )


# ---------------------------------------------------------------------------
# LocalIEMDistance tests
# ---------------------------------------------------------------------------

def test_pairwise_shape():
    dist = LocalIEMDistance(p, gammas, num_noises=5, seed=0)
    D = dist.pairwise(X, x_refs)
    assert D.shape == (2, 3), f"Expected (2, 3), got {D.shape}"


def test_pairwise_nonneg():
    dist = LocalIEMDistance(p, gammas, num_noises=5, seed=0)
    D = dist.pairwise(X, x_refs)
    assert (D >= 0).all(), f"Distance matrix has negative values: {D}"


def test_self_distance_approx_zero():
    """D([2,0], [2,0]) should be ~0 because diff = 0."""
    dist = LocalIEMDistance(p, gammas, num_noises=5, seed=0)
    D = dist.pairwise(X, x_refs)
    # X[0] = [2, 0], x_refs[0] = [2, 0]  => D[0, 0] should be 0
    self_dist = D[0, 0].item()
    assert self_dist < 1e-8, (
        f"Self-distance D([2,0], [2,0]) = {self_dist}, expected ~0"
    )


def test_compute_G_mode_vs_midpoint():
    """G at an on-mode point should differ meaningfully from G at the midpoint.

    The GMM has modes at [±2, 0]; the midpoint [0, 0] is between modes.
    The IEM metric tensor should be sensitive to this, so the trace of G
    at the mode [2, 0] must not be approximately equal to the trace at [0, 0].
    This catches gamma-weighting / noise-draw bugs that the PSD test cannot.
    """
    X_test = torch.tensor([[2., 0.], [0., 0.]], dtype=dtype)
    G = compute_G(X_test, log_p_Y_scalar, gammas, num_noises=5, seed=42)
    tr_mode = torch.diagonal(G[0], dim1=-2, dim2=-1).sum(-1)   # scalar
    tr_mid  = torch.diagonal(G[1], dim1=-2, dim2=-1).sum(-1)   # scalar
    assert not torch.allclose(tr_mode, tr_mid, rtol=0.1), (
        f"Traces are too similar: tr(G[mode])={tr_mode.item():.6f}, "
        f"tr(G[mid])={tr_mid.item():.6f}. "
        "Expected the metric to differ between on-mode and between-modes points."
    )
