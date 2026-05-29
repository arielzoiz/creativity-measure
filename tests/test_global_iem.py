"""Tests for creativity_measure.distances.global_iem (Task 4)."""
import math
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

from creativity_measure.density import Density
from creativity_measure.distances.global_iem import (
    GlobalIEMDistance,
    f_identity,
    f_square,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
dtype = torch.float64

# 2-mode GMM: means [2,0] and [-2,0], isotropic sigma^2=0.09, equal weights
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)  # (K, d)
sig2  = 0.09          # component variance for p_X


def log_pX(x):
    """log p_X(x): GMM with distribution constructors."""
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(
        Categorical(torch.ones(2, dtype=dtype)),
        MultivariateNormal(means, covs, validate_args=False),
    )
    return mix.log_prob(x)


def log_pY(y, gamma):
    """
    Analytic log p_{Y_gamma}(y) for Y = gamma*X + sqrt(gamma)*W, W~N(0,I).

    Uses torch.distributions with validate_args=False so autograd works through it.

    y:     (..., d)
    gamma: scalar tensor
    ->     (...)
    """
    # Component k: N(gamma*means[k], (gamma^2*sig2 + gamma)*I)
    var_diag = gamma ** 2 * sig2 + gamma                           # scalar
    d = y.shape[-1]

    # Flatten batch dims for MultivariateNormal
    batch_shape = y.shape[:-1]
    y_flat = y.reshape(-1, d)  # (N, d)

    log_comps = []
    for k in range(2):
        mu_k = gamma * means[k]                                    # (d,)
        cov_k = var_diag * torch.eye(d, device=y.device, dtype=y.dtype)
        dist_k = MultivariateNormal(mu_k, cov_k, validate_args=False)
        log_comps.append(dist_k.log_prob(y_flat))                  # (N,)

    log_comps = torch.stack(log_comps, dim=0)                      # (2, N)
    log_pi = -math.log(2)                                          # log(1/K), K=2
    result_flat = torch.logsumexp(log_pi + log_comps, dim=0)       # (N,)
    return result_flat.reshape(batch_shape)


def sampler(n):
    """Sample from p_X."""
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(
        Categorical(torch.ones(2, dtype=dtype)),
        MultivariateNormal(means, covs, validate_args=False),
    )
    return mix.sample(torch.Size((n,)))


p = Density(log_pX, log_pY, sample=sampler, d=2)

# Coarse grid for speed
gammas = torch.logspace(-10, 10, 40, base=2, dtype=dtype)

# Candidate points
X      = torch.tensor([[2., 0.], [0., 0.]], dtype=dtype)   # (2, 2)
x_refs = p.sample(4, seed=0)                               # (4, 2)


# ---------------------------------------------------------------------------
# Test 1: Shape and non-negativity for both activations
# ---------------------------------------------------------------------------

def test_f_identity_shape_and_nonneg():
    dist = GlobalIEMDistance(p, gammas, num_eps=8, f=f_identity, seed=123)
    out = dist.pairwise(X, x_refs)
    assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"
    assert (out >= 0).all(), f"f_identity distance matrix has negative values:\n{out}"


def test_f_square_shape_and_nonneg():
    dist = GlobalIEMDistance(p, gammas, num_eps=8, f=f_square, seed=123)
    out = dist.pairwise(X, x_refs)
    assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"
    assert (out >= 0).all(), f"f_square distance matrix has negative values:\n{out}"


# ---------------------------------------------------------------------------
# Test 2: Determinism — calling pairwise twice gives identical results
# ---------------------------------------------------------------------------

def test_determinism_f_identity():
    dist = GlobalIEMDistance(p, gammas, num_eps=8, f=f_identity, seed=123)
    out1 = dist.pairwise(X, x_refs)
    out2 = dist.pairwise(X, x_refs)
    assert torch.allclose(out1, out2), (
        f"f_identity: two calls gave different results.\nout1={out1}\nout2={out2}"
    )


def test_determinism_f_square():
    dist = GlobalIEMDistance(p, gammas, num_eps=8, f=f_square, seed=123)
    out1 = dist.pairwise(X, x_refs)
    out2 = dist.pairwise(X, x_refs)
    assert torch.allclose(out1, out2), (
        f"f_square: two calls gave different results.\nout1={out1}\nout2={out2}"
    )


# ---------------------------------------------------------------------------
# Test 3: f_identity and f_square produce DIFFERENT distance values
# (confirms z_gamma actually feeds f_square)
# ---------------------------------------------------------------------------

def test_f_identity_vs_f_square_differ():
    dist_id  = GlobalIEMDistance(p, gammas, num_eps=8, f=f_identity, seed=123)
    dist_sq  = GlobalIEMDistance(p, gammas, num_eps=8, f=f_square,   seed=123)
    out_identity = dist_id.pairwise(X, x_refs)
    out_square   = dist_sq.pairwise(X, x_refs)
    assert not torch.allclose(out_identity, out_square), (
        "f_identity and f_square produced identical distances — "
        "z_gamma is likely not feeding f_square correctly.\n"
        f"out_identity={out_identity}\nout_square={out_square}"
    )


# ---------------------------------------------------------------------------
# Test 4: f_identity reduces to dqv.sum end-to-end (regression)
# ---------------------------------------------------------------------------

def test_f_identity_equals_dqv_sum():
    from creativity_measure.distances.global_iem import sde_elements_one_to_many
    dist = GlobalIEMDistance(p, gammas, num_eps=8, f=f_identity, seed=123)
    device, dtype = X.device, X.dtype
    g = gammas.to(device=device, dtype=dtype)
    W, dW = dist._brownian(2, device, dtype, g)
    z, dqv = sde_elements_one_to_many(x_refs[0:1], X, W, dW, g, p)
    expected = dqv.sum(0).mean(0).clamp_min(0).sqrt()        # (B,)
    actual = dist.pairwise(X, x_refs[0:1]).squeeze(1)        # (B,)
    assert torch.allclose(actual, expected)
