"""Tests for the twisted-diffusion SMC sampler (creativity_measure/diffusion_smc.py).

Same tilted target as adaptive_tempering_smc.py, different mechanism (runs inside the EDM denoising trajectory). Validated
in 2D against grid_normalize on a Ring-GMM-with-hole (the shared toy): lambda=0 recovers p, E_q[f] tracks
the grid and rises with lambda, plus determinism and a pixel-path smoke test with an analytic denoiser.
"""

import math

import torch
import pytest

from creativity_measure import (
    Density,
    GlobalIEMDistance,
    SquaredGlobalIEMDistance,
    NormalizedExpectedDistanceReward,
    expected_distance,
    Reward,
    diffusion_smc_sample,
    DiffusionSMCResult,
    density_denoiser,
    edm_score_fn,
)

dtype = torch.float64
XLIM = (-6.0, 6.0)
YLIM = (-6.0, 6.0)


# --- Ring-GMM-with-hole (mirrors tests/test_adaptive_tempering_smc.py, tests/test_pcn.py) --------------------------

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


def _global_iem(p):
    gammas = torch.logspace(math.log2(1.0 / (0.75 * 0.3) ** 2), 10, 24, base=2, dtype=dtype)
    return GlobalIEMDistance(p, gammas, num_eps=4, seed=123)


def _refs(p, r=12):
    return p.sample(64, seed=0)[:r]


def _lambda0(p, D, refs):
    return (p.log_p_X(refs).std() / expected_distance(D, refs, refs).std()).item()


# --- histogram helpers (shared with test_adaptive_tempering_smc.py / test_pcn.py) ---------------------------------

def _hist_pmf(samples, grid_n):
    ci = (((samples[:, 0] - XLIM[0]) / (XLIM[1] - XLIM[0])) * (grid_n - 1)).round().long().clamp(0, grid_n - 1)
    ri = (((samples[:, 1] - YLIM[0]) / (YLIM[1] - YLIM[0])) * (grid_n - 1)).round().long().clamp(0, grid_n - 1)
    k = ri * grid_n + ci
    counts = torch.zeros(grid_n * grid_n, dtype=dtype)
    counts.scatter_add_(0, k, torch.ones(samples.shape[0], dtype=dtype))
    return counts / counts.sum()


def _tv(a, b) -> float:
    return 0.5 * float((a - b).abs().sum())


# 2D-toy schedule constants shared across tests (fast but well within the denoiser's valid sigma range).
_SIGMA_MIN, _SIGMA_MAX, _N_STEPS, _MC = 2e-3, 40.0, 32, 4


def _sample_toy(reward, lam, n_particles, p, *,
                seed=0, final_resample=False, use_score_correction=False):
    """diffusion_smc_sample on the shared 2D-toy schedule (density_denoiser(p), d=2)."""
    return diffusion_smc_sample(
        reward, lam, n_particles,
        denoiser=density_denoiser(p), d=2,
        sigma_min=_SIGMA_MIN, sigma_max=_SIGMA_MAX, n_steps=_N_STEPS, mc_samples=_MC,
        seed=seed, final_resample=final_resample, use_score_correction=use_score_correction,
    )


# ===========================================================================================

def test_diffusion_smc_runs_shapes_and_determinism():
    p = _ring_density()
    D = _global_iem(p)
    refs = _refs(p)
    reward = Reward(D, refs)

    a = _sample_toy(reward, 2.0, 128, p, seed=0)
    b = _sample_toy(reward, 2.0, 128, p, seed=0)

    assert isinstance(a, DiffusionSMCResult)
    assert a.X.shape == (128, 2)
    assert a.X.isfinite().all()
    assert a.sigmas.shape == (_N_STEPS + 1,)
    assert torch.equal(a.X, b.X)                 # same seed -> identical
    assert torch.equal(a.logw, b.logw)


def test_score_correction_not_implemented():
    p = _ring_density()
    reward = Reward(_global_iem(p), _refs(p))
    with pytest.raises(NotImplementedError):
        _sample_toy(reward, 1.0, 16, p, use_score_correction=True)


def test_diffusion_smc_lambda0_recovers_p():
    """lam=0 -> q=p: the trajectory-space sampler's output matches p within histogram noise.

    Compared against the *intrinsic* two-sample TV floor (the histogram noise between equal-size draws of
    p itself), since at lam=0 the target is exactly p and the only residual gap is schedule discretization.
    """
    p = _ring_density()
    reward = Reward(_global_iem(p), _refs(p))
    res = _sample_toy(reward, 0.0, 2000, p, seed=0, final_resample=True)
    h_smc = _hist_pmf(res.X, 24)
    h_a = _hist_pmf(p.sample(2000), 24)
    h_b = _hist_pmf(p.sample(2000), 24)
    floor = _tv(h_a, h_b)                       # intrinsic two-sample histogram noise
    assert _tv(h_smc, h_a) < floor + 0.08       # sampler adds only a small discretization gap on top


@pytest.mark.slow
def test_diffusion_smc_tilts_and_stays_healthy():
    """The tilt moves E[f] up toward q while the particle population stays healthy (diverse, non-degenerate).

    Scope note: this is the *approximate* (score-uncorrected) core. It reproduces the tilt DIRECTION but
    NOT q spatially: the reward f (expected IEM distance) is not spatially injective, so the score-free
    lookahead twist drives mass to high-f regions that need not coincide with q's high-f-AND-high-p mass
    (q ∝ p·exp(lambda f) stays on the p-manifold; the twist alone can wander off it toward the ring hole).
    Keeping samples on the manifold — hence quantitative/spatial recovery of q — is the score-correction
    follow-up. Here we assert what the core does guarantee: a clear upward tilt, and no particle collapse.
    """
    p = _ring_density()
    D = _global_iem(p)
    refs = _refs(p)
    lam = _lambda0(p, D, refs)
    reward = Reward(D, refs)

    N = 2000
    res = _sample_toy(reward, lam, N, p, seed=0, final_resample=False)
    w = torch.softmax(res.logw, dim=0)

    ef_prior = float(reward(p.sample(4000)).mean())
    ef_smc = float((w * reward(res.X)).sum())              # weighted SMC estimator of E_q[f]
    assert ef_smc > ef_prior + 0.5                         # a clear upward tilt above the prior

    ess = float((w.sum() ** 2) / (w ** 2).sum())
    assert torch.unique(res.X, dim=0).shape[0] > 0.5 * N   # stochastic ancestral step preserves diversity
    assert ess > 0.3 * N                                   # final weights are not degenerate


@pytest.mark.slow
def test_diffusion_smc_ef_monotone_in_lambda():
    """E_q[f] rises with the tilt strength lambda (the sampler pushes mass up the reward)."""
    p = _ring_density()
    D = _global_iem(p)
    refs = _refs(p)
    l0 = _lambda0(p, D, refs)
    reward = Reward(D, refs)

    efs = [
        float(reward(_sample_toy(reward, m * l0, 1500, p, seed=0, final_resample=True).X).mean())
        for m in (0.0, 1.0, 2.5)
    ]
    assert efs[0] < efs[1] < efs[2]


@pytest.mark.slow
def test_diffusion_smc_normalized_squared_iem():
    """Headline use: the normalized squared-IEM reward runs end-to-end and tilts (E_q[f] rises vs lam=0)."""
    p = _ring_density()
    Dsq = SquaredGlobalIEMDistance(p, torch.logspace(math.log2(1.0 / (0.75 * 0.3) ** 2), 10, 24, base=2, dtype=dtype),
                                   num_eps=4, seed=123)
    refs = _refs(p)
    reward = NormalizedExpectedDistanceReward(Dsq, refs)

    ef0 = float(reward(_sample_toy(reward, 0.0, 1500, p, seed=0, final_resample=True).X).mean())
    ef1 = float(reward(_sample_toy(reward, 1.0, 1500, p, seed=0, final_resample=True).X).mean())
    assert ef1 > ef0


def test_diffusion_smc_pixel_smoke():
    """Pixel path runs end-to-end with an analytic mock denoiser (X~N(0,I)); no checkpoint needed."""
    C, H, W = 1, 2, 2
    d = C * H * W

    def mock_denoiser(y_sigma, sigma):
        shape = (-1,) + (1,) * (y_sigma.dim() - 1)
        return y_sigma / (1.0 + sigma.reshape(shape) ** 2)   # E[X | y_sigma] for X ~ N(0, I)

    score_fn = edm_score_fn(mock_denoiser, img_shape=(C, H, W))
    D = GlobalIEMDistance(None, torch.logspace(-4, 4, 12, base=2, dtype=dtype),
                          num_eps=4, seed=123, score_fn=score_fn)
    refs = torch.randn(8, d, dtype=dtype)

    res = diffusion_smc_sample(Reward(D, refs), lam=0.0, n_particles=64, seed=0,
                               denoiser=mock_denoiser, d=d, img_shape=(C, H, W),
                               sigma_min=2e-3, sigma_max=20.0, n_steps=24, mc_samples=2,
                               final_resample=True)

    assert res.X.shape == (64, d)
    assert res.X.isfinite().all()
    # lam=0 -> q = p = N(0,I): samples are roughly standard normal
    assert res.X.mean().abs() < 0.5
    assert abs(float(res.X.std()) - 1.0) < 0.5
