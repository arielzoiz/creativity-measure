"""Tests for creativity_measure.distances.local_iem."""
import math
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

from creativity_measure.density import Density
from creativity_measure.distances.local_iem import (
    _compute_G_vmap,
    compute_G_autograd,
    compute_G,
    LocalIEMDistance,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
dtype = torch.float64

# 2-mode GMM: means [2,0] and [-2,0], isotropic sigma^2=0.09, equal weights
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)
sig2  = 0.09


def log_pX(x):
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(Categorical(torch.ones(2)),
                            MultivariateNormal(means, covs))
    return mix.log_prob(x)


def _log_gaussian(y, mu, var_diag):
    """log N(y; mu, var_diag*I) in pure tensor ops — vmap-compatible."""
    d = y.shape[-1]
    diff = y - mu
    return -0.5 * (d * (math.log(2 * math.pi) + var_diag.log())
                   + (diff ** 2).sum(-1) / var_diag)


def log_pY(y, gamma):
    """Analytic log p_{Y_gamma}(y); pure tensor ops, vmap-compatible."""
    var_diag = gamma ** 2 * sig2 + gamma
    log_pi = -math.log(2)
    loc = gamma * means
    log_comps = torch.stack([_log_gaussian(y, loc[k], var_diag)
                             for k in range(2)], dim=0)
    return torch.logsumexp(log_pi + log_comps, dim=0)


p = Density(log_pX, log_pY, d=2)
log_p_Y_scalar = lambda y, g: log_pY(y.unsqueeze(0), g).squeeze(0)

gammas = torch.logspace(-4, 4, 30, base=2, dtype=dtype)

X      = torch.tensor([[2., 0.], [0., 0.]], dtype=dtype)
x_refs = torch.tensor([[2., 0.], [-2., 0.], [0., 0.]], dtype=dtype)


# ---------------------------------------------------------------------------
# _compute_G_vmap tests
# ---------------------------------------------------------------------------

def test_compute_G_vmap_shape():
    G = _compute_G_vmap(X, log_p_Y_scalar, gammas, num_noises=5, seed=0)
    assert G.shape == (2, 2, 2), f"Expected (2, 2, 2), got {G.shape}"


def test_compute_G_vmap_psd():
    G = _compute_G_vmap(X, log_p_Y_scalar, gammas, num_noises=5, seed=0)
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals >= -1e-8).all(), f"G has eigenvalue(s) < -1e-8: min={eigvals.min().item()}"


def test_compute_G_vmap_mode_vs_midpoint():
    G = _compute_G_vmap(X, log_p_Y_scalar, gammas, num_noises=5, seed=42)
    tr_mode = torch.diagonal(G[0], dim1=-2, dim2=-1).sum(-1)
    tr_mid  = torch.diagonal(G[1], dim1=-2, dim2=-1).sum(-1)
    assert not torch.allclose(tr_mode, tr_mid, rtol=0.1), (
        f"Traces are too similar: tr(G[mode])={tr_mode.item():.6f}, "
        f"tr(G[mid])={tr_mid.item():.6f}."
    )


# ---------------------------------------------------------------------------
# compute_G_autograd tests
# ---------------------------------------------------------------------------

def test_compute_G_autograd_shape():
    G = compute_G_autograd(X, p, gammas, num_noises=5, seed=0)
    assert G.shape == (2, 2, 2), f"Expected (2, 2, 2), got {G.shape}"


def test_compute_G_autograd_psd():
    G = compute_G_autograd(X, p, gammas, num_noises=5, seed=0)
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals >= -1e-8).all(), f"G has eigenvalue(s) < -1e-8: min={eigvals.min().item()}"


def test_compute_G_autograd_matches_vmap():
    """Both implementations produce the same result given the same seed and noise draws."""
    G_vmap = _compute_G_vmap(X, log_p_Y_scalar, gammas, num_noises=20, seed=0)
    G_auto = compute_G_autograd(X, p, gammas, num_noises=20, seed=0)
    assert torch.allclose(G_vmap, G_auto, rtol=0.05, atol=1e-4), (
        f"vmap and autograd Hessians disagree.\nvmap:\n{G_vmap}\nautograd:\n{G_auto}"
    )


# ---------------------------------------------------------------------------
# compute_G (safe wrapper) tests
# ---------------------------------------------------------------------------

def test_compute_G_shape():
    G = compute_G(X, p, gammas, num_noises=5, seed=0)
    assert G.shape == (2, 2, 2), f"Expected (2, 2, 2), got {G.shape}"


def test_compute_G_psd():
    G = compute_G(X, p, gammas, num_noises=5, seed=0)
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals >= -1e-8).all(), f"G has eigenvalue(s) < -1e-8: min={eigvals.min().item()}"


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
    dist = LocalIEMDistance(p, gammas, num_noises=5, seed=0)
    D = dist.pairwise(X, x_refs)
    self_dist = D[0, 0].item()
    assert self_dist < 1e-8, f"Self-distance D([2,0], [2,0]) = {self_dist}, expected ~0"
