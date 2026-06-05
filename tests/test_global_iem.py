"""Tests for creativity_measure.distances.global_iem (GlobalIEMDistance, direct marginal-score formulation)."""
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
import math

from creativity_measure.density import Density
from creativity_measure.distances.global_iem import GlobalIEMDistance

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
dtype = torch.float64

# 2-mode GMM: means [2,0] and [-2,0], isotropic sigma^2=0.09, equal weights
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)
sig2  = 0.09


def log_pX(x):
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(
        Categorical(torch.ones(2, dtype=dtype)),
        MultivariateNormal(means, covs, validate_args=False),
    )
    return mix.log_prob(x)


def log_pY(y, gamma):
    var_diag = gamma ** 2 * sig2 + gamma
    d = y.shape[-1]
    batch_shape = y.shape[:-1]
    y_flat = y.reshape(-1, d)
    log_comps = []
    for k in range(2):
        mu_k = gamma * means[k]
        cov_k = var_diag * torch.eye(d, device=y.device, dtype=y.dtype)
        dist_k = MultivariateNormal(mu_k, cov_k, validate_args=False)
        log_comps.append(dist_k.log_prob(y_flat))
    log_comps = torch.stack(log_comps, dim=0)
    log_pi = -math.log(2)
    result_flat = torch.logsumexp(log_pi + log_comps, dim=0)
    return result_flat.reshape(batch_shape)


def sampler(n):
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(
        Categorical(torch.ones(2, dtype=dtype)),
        MultivariateNormal(means, covs, validate_args=False),
    )
    return mix.sample(torch.Size((n,)))


p = Density(log_pX, log_pY, sample_fn=sampler, d=2)

gammas = torch.logspace(-10, 10, 40, base=2, dtype=dtype)

X      = torch.tensor([[2., 0.], [0., 0.]], dtype=dtype)   # (2, 2)
x_refs = p.sample(4, seed=0)                               # (4, 2)


# ---------------------------------------------------------------------------
# Test 1: Shape and non-negativity
# ---------------------------------------------------------------------------

def test_shape_and_nonneg():
    dist = GlobalIEMDistance(p, gammas, num_eps=8, seed=123)
    out = dist.pairwise(X, x_refs)
    assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"
    assert (out >= 0).all(), f"Distance matrix has negative values:\n{out}"


# ---------------------------------------------------------------------------
# Test 2: Determinism — calling pairwise twice gives identical results
# ---------------------------------------------------------------------------

def test_determinism():
    dist = GlobalIEMDistance(p, gammas, num_eps=8, seed=123)
    out1 = dist.pairwise(X, x_refs)
    out2 = dist.pairwise(X, x_refs)
    assert torch.allclose(out1, out2), (
        f"Two calls gave different results.\nout1={out1}\nout2={out2}"
    )


# ---------------------------------------------------------------------------
# Test 3: Different seeds produce different results
# ---------------------------------------------------------------------------

def test_different_seeds_differ():
    dist1 = GlobalIEMDistance(p, gammas, num_eps=8, seed=0)
    dist2 = GlobalIEMDistance(p, gammas, num_eps=8, seed=1)
    out1 = dist1.pairwise(X, x_refs)
    out2 = dist2.pairwise(X, x_refs)
    assert not torch.allclose(out1, out2), (
        "Different seeds produced identical distances — seed is likely not consumed."
    )


# ---------------------------------------------------------------------------
# Test 4: pairwise equals manual iem_sq_increments_one_to_many (regression)
# ---------------------------------------------------------------------------

def test_pairwise_matches_manual_increments():
    from creativity_measure.distances.global_iem import iem_sq_increments_one_to_many
    from creativity_measure.distances.utils import simulate_brownian
    dist = GlobalIEMDistance(p, gammas, num_eps=8, seed=123)
    device, dt = X.device, X.dtype
    g = gammas.to(device=device, dtype=dt)
    W = simulate_brownian(g, dist.num_eps, 2, dist.seed, device, dt)
    increments = iem_sq_increments_one_to_many(x_refs[0:1], X, W, g, p)
    expected = increments.sum(0).mean(0).clamp_min(0).sqrt()   # (B,)
    actual = dist.pairwise(X, x_refs[0:1]).squeeze(1)          # (B,)
    assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# Test 5: injected score_fn reproduces the autograd path
# ---------------------------------------------------------------------------

def test_score_fn_matches_autograd():
    from creativity_measure.distances.global_iem import marginal_score
    # a score_fn that simply wraps the autograd marginal score must reproduce the density path
    score_fn = lambda y, g: marginal_score(y, g, p)
    d_auto = GlobalIEMDistance(p, gammas, num_eps=8, seed=123)
    d_inj  = GlobalIEMDistance(None, gammas, num_eps=8, seed=123, score_fn=score_fn)
    assert torch.allclose(d_auto.pairwise(X, x_refs), d_inj.pairwise(X, x_refs))


def test_requires_density_or_score_fn():
    import pytest
    with pytest.raises(ValueError):
        GlobalIEMDistance(None, gammas, num_eps=8, seed=123)
