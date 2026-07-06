"""Tests for the deterministic EDM prob-flow generator (creativity_measure/generators/)."""

import math

import torch
import pytest

from creativity_measure import (
    Density,
    density_denoiser,
    density_generator,
    heun_prob_flow,
)

dtype = torch.float64
XLIM = (-6.0, 6.0)


# --- Ring-GMM-with-hole (has log_p_Y; mirrors tests/test_smc.py) -------------------------------

def _ring_density(n_total: int = 12, hole_idx: int = 0, radius: float = 4.0, sigma: float = 0.3):
    angles = 2 * math.pi * torch.arange(n_total, dtype=dtype) / n_total
    all_means = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)
    mask = torch.ones(n_total, dtype=torch.bool)
    mask[hole_idx] = False
    means = all_means[mask]
    K = means.shape[0]
    s2 = sigma ** 2
    logK = math.log(K)

    def log_pX(x):
        diff = x.unsqueeze(-2) - means
        quad = diff.pow(2).sum(-1) / s2
        return torch.logsumexp(-0.5 * quad - 0.5 * 2 * math.log(2 * math.pi * s2), dim=-1) - logK

    def log_pY(y, gamma):
        gamma = torch.as_tensor(gamma, dtype=y.dtype, device=y.device)
        var = gamma ** 2 * s2 + gamma
        diff = y.unsqueeze(-2) - gamma * means
        quad = diff.pow(2).sum(-1) / var
        return torch.logsumexp(-0.5 * quad - 0.5 * 2 * torch.log(2 * math.pi * var), dim=-1) - logK

    def sample(n, generator=None):
        comp = torch.randint(K, (n,), generator=generator)
        return means[comp] + sigma * torch.randn(n, 2, dtype=dtype, generator=generator)

    return Density(log_pX, log_pY, sample_fn=sample, d=2)


def _standard_normal_density():
    def log_pX(x):
        return (-0.5 * x.pow(2).sum(-1) - 0.5 * 2 * math.log(2 * math.pi))

    def log_pY(y, gamma):
        gamma = torch.as_tensor(gamma, dtype=y.dtype, device=y.device)
        var = gamma ** 2 + gamma                      # Var(Y) for X~N(0,I): gamma^2*1 + gamma
        return -0.5 * y.pow(2).sum(-1) / var - 0.5 * 2 * torch.log(2 * math.pi * var)

    return Density(log_pX, log_pY, d=2)


def _hist_pmf(samples, grid_n=24, lo=-6.0, hi=6.0):
    ci = (((samples[:, 0] - lo) / (hi - lo)) * (grid_n - 1)).round().long().clamp(0, grid_n - 1)
    ri = (((samples[:, 1] - lo) / (hi - lo)) * (grid_n - 1)).round().long().clamp(0, grid_n - 1)
    k = ri * grid_n + ci
    counts = torch.zeros(grid_n * grid_n, dtype=dtype)
    counts.scatter_add_(0, k, torch.ones(samples.shape[0], dtype=dtype))
    return counts / counts.sum()


def _tv(a, b) -> float:
    return 0.5 * float((a - b).abs().sum())


# ===========================================================================================

def test_density_denoiser_matches_analytic_normal():
    """For X~N(0,I) the EDM denoiser is D(x,sigma) = x/(1+sigma^2)."""
    p = _standard_normal_density()
    D = density_denoiser(p)
    x = torch.randn(7, 2, dtype=dtype)
    for sig in (0.1, 1.0, 5.0):
        sigma = torch.full((7,), sig, dtype=dtype)
        got = D(x, sigma)
        want = x / (1.0 + sig ** 2)
        assert torch.allclose(got, want, atol=1e-5), f"sigma={sig}: {got[0]} vs {want[0]}"


def test_generator_reproduces_p():
    """X = G(randn(N)) should match p.sample(N): G_#N(0,I) ~ p."""
    p = _ring_density()
    G = density_generator(p, sigma_min=2e-3, sigma_max=80.0, n_steps=64)
    torch.manual_seed(0)
    z = torch.randn(3000, 2, dtype=dtype)
    X = G(z)
    base = p.sample(3000)
    assert X.isfinite().all()
    assert _tv(_hist_pmf(X), _hist_pmf(base)) < 0.15


def test_generator_deterministic():
    """G is a deterministic map: same z -> identical x."""
    p = _ring_density()
    G = density_generator(p, n_steps=32)
    z = torch.randn(64, 2, generator=torch.Generator().manual_seed(0), dtype=dtype)
    assert torch.equal(G(z), G(z))


def test_heun_requires_two_steps():
    p = _standard_normal_density()
    D = density_denoiser(p)
    with pytest.raises(ValueError):
        heun_prob_flow(torch.randn(4, 2, dtype=dtype), D, n_steps=1)
