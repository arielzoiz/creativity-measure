"""Tests for the Phase-2 latent-space pCN kernel (creativity_measure/smc.py::PCNKernel).

Validated in 2D against grid_normalize at mild lambda (the architecture check), plus the high-lambda
payoff (pCN avoids the impoverishment that independence-MH suffers) and a pixel-seam smoke test that runs
the learned-denoiser path end-to-end with an analytic mock denoiser (no checkpoint needed).
"""

import math

import torch
import pytest

from creativity_measure import (
    Density,
    GlobalIEMDistance,
    expected_distance,
    tilted_log_density,
    grid_normalize,
    make_grid,
    smc_sample,
    Reward,
    IndependenceKernel,
    PCNKernel,
    density_generator,
    edm_generator,
    edm_score_fn,
)

dtype = torch.float64
XLIM = (-6.0, 6.0)
YLIM = (-6.0, 6.0)


# --- Ring-GMM-with-hole (mirrors tests/test_smc.py) -------------------------------------------

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


def _grid_q_pmf(p, reward, lam, grid_n):
    gp, _XX, _YY, cell = make_grid(XLIM, YLIM, grid_n=grid_n, dtype=dtype)
    log_q_un = tilted_log_density(gp, p, reward, lam)
    _lq, q, _Z = grid_normalize(log_q_un, cell)
    q_flat = q.reshape(-1)
    return gp, q_flat / q_flat.sum()


def _hist_pmf(samples, grid_n):
    ci = (((samples[:, 0] - XLIM[0]) / (XLIM[1] - XLIM[0])) * (grid_n - 1)).round().long().clamp(0, grid_n - 1)
    ri = (((samples[:, 1] - YLIM[0]) / (YLIM[1] - YLIM[0])) * (grid_n - 1)).round().long().clamp(0, grid_n - 1)
    k = ri * grid_n + ci
    counts = torch.zeros(grid_n * grid_n, dtype=dtype)
    counts.scatter_add_(0, k, torch.ones(samples.shape[0], dtype=dtype))
    return counts / counts.sum()


def _tv(a, b) -> float:
    return 0.5 * float((a - b).abs().sum())


def _pcn(p, **kw):
    return PCNKernel(density_generator(p, n_steps=40), 2, **kw)


# ===========================================================================================

def test_pcn_lambda0_recovers_p():
    """lam=0 -> q=p; pCN samples (X=G(Z)) match p (TV small)."""
    p = _ring_density()
    D = _global_iem(p)
    refs = _refs(p)
    res = smc_sample(Reward(D, refs), lam=0.0, n_particles=3000,
                     kernel=_pcn(p), n_mcmc=2, final_resample=True, seed=0)
    base = p.sample(3000)
    assert _tv(_hist_pmf(res.X, 24), _hist_pmf(base, 24)) < 0.15


@pytest.mark.slow
def test_pcn_recovery_vs_grid_and_independence():
    """pCN-SMC recovers grid_normalize q (GlobalIEMDistance): E_q[f] matches the grid, and the spatial
    distribution is at least as accurate as the validated independence-MH kernel — and far closer to q
    than the untilted prior p (which the tilt deliberately moves mass away from)."""
    p = _ring_density()
    D = _global_iem(p)
    refs = _refs(p)
    lam = _lambda0(p, D, refs)
    reward = Reward(D, refs)
    grid_n = 24
    gp, q_pmf = _grid_q_pmf(p, reward, lam, grid_n=grid_n)

    res_pcn = smc_sample(reward, lam=lam, n_particles=1500,
                         kernel=_pcn(p, s0=0.5), n_mcmc=3, final_resample=True, seed=0)
    res_ind = smc_sample(reward, lam=lam, n_particles=1500,
                         kernel=IndependenceKernel(p), n_mcmc=3, final_resample=True, seed=0)

    # 1) reward match (robust scalar)
    ef_grid = float((q_pmf * reward(gp)).sum())
    ef_pcn = float(reward(res_pcn.X).mean())
    assert ef_pcn == pytest.approx(ef_grid, rel=0.15)

    # 2) spatial match: pCN is at least as close to grid q as the validated independence kernel, and both
    #    are far closer to q than the untilted prior p.
    torch.manual_seed(7)
    tv_pcn = _tv(_hist_pmf(res_pcn.X, grid_n), q_pmf)
    tv_ind = _tv(_hist_pmf(res_ind.X, grid_n), q_pmf)
    tv_prior = _tv(_hist_pmf(p.sample(1500), grid_n), q_pmf)
    assert tv_pcn <= tv_ind + 0.02
    assert tv_pcn < 0.7 * tv_prior


@pytest.mark.slow
def test_pcn_beats_independence_high_lambda():
    """At high lambda the off-manifold target collapses independence-MH; pCN keeps diversity & higher E[f]."""
    p = _ring_density()
    D = _global_iem(p)
    refs = _refs(p)
    lam = 3.5 * _lambda0(p, D, refs)
    reward = Reward(D, refs)
    N = 800

    rp = smc_sample(reward, lam=lam, n_particles=N,
                    kernel=_pcn(p, s0=0.5), n_mcmc=2, final_resample=True, seed=0)
    ri = smc_sample(reward, lam=lam, n_particles=N,
                    kernel=IndependenceKernel(p), n_mcmc=2, final_resample=True, seed=0)

    uniq_p = torch.unique(rp.X, dim=0).shape[0]
    uniq_i = torch.unique(ri.X, dim=0).shape[0]
    ef_p = float(reward(rp.X).mean())
    ef_i = float(reward(ri.X).mean())
    acc_p = sum(rp.acc_history) / len(rp.acc_history)

    assert uniq_p > 2 * uniq_i           # independence collapses to few distinct particles; pCN does not
    assert ef_p >= ef_i - 1e-6           # pCN reaches at least as far up the reward as independence
    assert acc_p > 0.05                  # acceptance has not collapsed to ~0


def test_pcn_determinism():
    p = _ring_density()
    D = GlobalIEMDistance(p, torch.logspace(-2, 6, 12, base=2, dtype=dtype), num_eps=4, seed=123)
    refs = _refs(p, r=8)
    reward = Reward(D, refs)
    a = smc_sample(reward, lam=2.0, n_particles=200, kernel=_pcn(p), n_mcmc=2, seed=0)
    b = smc_sample(reward, lam=2.0, n_particles=200, kernel=_pcn(p), n_mcmc=2, seed=0)
    assert torch.equal(a.X, b.X)
    assert torch.equal(a.logw, b.logw)


def test_pcn_decorrelation_is_dimension_robust():
    """Normalized-ESJD decorrelation: 1.0 at the baseline, and ~0 at independence in ANY latent dim.

    Guards the dimension-robust metric: the old |cos| form floors at ~2/pi in 2D, so the < 0.1
    independence assertion below would fail there — this pins portability from the 2D toy to image dims.
    """
    from creativity_measure.smc import _State
    p = _ring_density()
    ker = PCNKernel(density_generator(p, n_steps=8), 2)     # G is unused by decorrelation()
    for d in (2, 64):
        g = torch.Generator().manual_seed(0)
        z0 = torch.randn(2000, d, generator=g, dtype=dtype)
        st = _State(X=z0.clone(), fX=torch.zeros(2000, dtype=dtype),
                    logw=torch.zeros(2000, dtype=dtype), aux={"Z": z0.clone()})
        base = ker.snapshot_baseline(st)
        assert ker.decorrelation(st, base) == pytest.approx(1.0)          # sits at baseline
        st.aux["Z"] = torch.randn(2000, d, generator=g, dtype=dtype)      # independent redraw
        assert ker.decorrelation(st, base) < 0.1                          # ~0 regardless of d


def test_pcn_pixel_seam_smoke():
    """Pixel path runs end-to-end with an analytic mock denoiser (X~N(0,I)); no checkpoint needed."""
    C, H, W = 1, 2, 2
    d = C * H * W

    def mock_denoiser(y_sigma, sigma):
        # MMSE denoiser for X ~ N(0, I): E[X | y_sigma] = y_sigma / (1 + sigma^2)
        shape = (-1,) + (1,) * (y_sigma.dim() - 1)
        return y_sigma / (1.0 + sigma.reshape(shape) ** 2)

    score_fn = edm_score_fn(mock_denoiser, img_shape=(C, H, W))
    D = GlobalIEMDistance(None, torch.logspace(-4, 4, 12, base=2, dtype=dtype),
                          num_eps=4, seed=123, score_fn=score_fn)
    G = edm_generator(mock_denoiser, img_shape=(C, H, W), sigma_max=20.0, n_steps=24)
    refs = torch.randn(8, d, dtype=dtype)              # flat references in pixel space

    res = smc_sample(Reward(D, refs), lam=0.0, n_particles=64,
                     kernel=PCNKernel(G, d), n_mcmc=2, final_resample=True, seed=0)

    assert res.X.shape == (64, d)
    assert res.X.isfinite().all()
    # lam=0 -> q = p = N(0,I): samples are roughly standard normal
    assert res.X.mean().abs() < 0.5
    assert abs(float(res.X.std()) - 1.0) < 0.5
