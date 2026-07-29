"""Tests for the gradient-free SMC sampler (creativity_measure/adaptive_tempering_smc.py).

Validated in 2D against grid_normalize ground truth on a Ring-GMM-with-hole, plus
unit tests for the SMC helpers, determinism, the lambda=0 sanity case, and a
RandomRefs/WeightedFPSRefs selector swap.
"""

import math

import torch
import pytest

from creativity_measure import (
    Density,
    LpDistance,
    GlobalIEMDistance,
    expected_distance,
    tilted_log_density,
    grid_normalize,
    make_grid,
    adaptive_tempering_smc_sample,
    AdaptiveTemperingSMCResult,
    Reward,
    IndependenceKernel,
)
from creativity_measure.adaptive_tempering_smc import (
    _ess_from_logw,
    _systematic_resample,
    _next_dbeta,
    RejuvenationStop,
    MAX_N_MCMC,
)
from creativity_measure.refset import RandomRefs, WeightedFPSRefs

dtype = torch.float64
XLIM = (-6.0, 6.0)
YLIM = (-6.0, 6.0)


# ---------------------------------------------------------------------------
# Ring-GMM-with-hole density (pure tensor ops, mirrors test_demo_pipeline)
# ---------------------------------------------------------------------------

def _ring_density(n_total: int = 8, hole_idx: int = 0,
                  radius: float = 4.0, sigma: float = 0.3):
    angles = 2 * math.pi * torch.arange(n_total, dtype=dtype) / n_total
    all_means = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1
    )
    mask = torch.ones(n_total, dtype=torch.bool)
    mask[hole_idx] = False
    means = all_means[mask]
    K = means.shape[0]
    s2 = sigma ** 2
    logK = math.log(K)

    def log_pX(x):
        diff = x.unsqueeze(-2) - means
        quad = diff.pow(2).sum(-1) / s2
        logcomp = -0.5 * (quad + 2 * math.log(2 * math.pi * s2))
        return torch.logsumexp(logcomp, dim=-1) - logK

    def log_pY(y, gamma):
        var = gamma ** 2 * s2 + gamma
        diff = y.unsqueeze(-2) - gamma * means
        quad = diff.pow(2).sum(-1) / var
        logcomp = -0.5 * (quad + 2 * torch.log(2 * math.pi * var))
        return torch.logsumexp(logcomp, dim=-1) - logK

    def sample(n, generator=None):
        comp = torch.randint(K, (n,), generator=generator)
        return means[comp] + sigma * torch.randn(n, 2, dtype=dtype, generator=generator)

    return Density(log_pX, log_pY, sample_fn=sample, d=2), means


# ---------------------------------------------------------------------------
# Grid / histogram helpers (PMF over grid nodes; cell_area cancels)
# ---------------------------------------------------------------------------

def _grid_q_pmf(p, reward, lam, grid_n):
    """Ground-truth normalized q on a grid, returned as (grid_points, q_pmf)."""
    gp, _XX, _YY, cell = make_grid(XLIM, YLIM, grid_n=grid_n, dtype=dtype)
    log_q_un = tilted_log_density(gp, p, reward, lam)
    _log_q, q, _Z = grid_normalize(log_q_un, cell)
    q_flat = q.reshape(-1)
    return gp, q_flat / q_flat.sum()


def _hist_pmf(samples, weights, grid_n):
    """Bin (weighted) samples to nearest grid node; return a PMF over nodes.

    Node ordering matches make_grid's grid_points (k = row*grid_n + col,
    col=x-index, row=y-index), so it aligns with _grid_q_pmf's q_flat.
    """
    x, y = samples[:, 0], samples[:, 1]
    ci = (((x - XLIM[0]) / (XLIM[1] - XLIM[0])) * (grid_n - 1)).round().long().clamp(0, grid_n - 1)
    ri = (((y - YLIM[0]) / (YLIM[1] - YLIM[0])) * (grid_n - 1)).round().long().clamp(0, grid_n - 1)
    k = ri * grid_n + ci
    counts = torch.zeros(grid_n * grid_n, dtype=dtype)
    counts.scatter_add_(0, k, weights.to(dtype))
    return counts / counts.sum()


def _tv(pmf_a, pmf_b) -> float:
    return 0.5 * float((pmf_a - pmf_b).abs().sum())


def _smc_weights(res: AdaptiveTemperingSMCResult):
    return torch.softmax(res.logw, dim=0)


# ===========================================================================
# Unit tests: SMC helpers
# ===========================================================================

def test_ess_from_logw_uniform_and_onehot():
    N = 100
    uniform = torch.zeros(N, dtype=dtype)
    assert _ess_from_logw(uniform) == pytest.approx(N, rel=1e-9)

    onehot = torch.full((N,), float("-inf"), dtype=dtype)
    onehot[3] = 0.0
    assert _ess_from_logw(onehot) == pytest.approx(1.0, rel=1e-9)


def test_systematic_resample_uniform_onehot_and_determinism():
    gen = torch.Generator().manual_seed(0)

    # uniform weights over 4 -> systematic resampling yields each index exactly once
    w_uniform = torch.full((4,), 0.25, dtype=dtype)
    idx = _systematic_resample(w_uniform, gen)
    assert sorted(idx.tolist()) == [0, 1, 2, 3]

    # one-hot -> every draw is that index
    w_onehot = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=dtype)
    idx2 = _systematic_resample(w_onehot, torch.Generator().manual_seed(7))
    assert (idx2 == 2).all()

    # determinism under a fixed generator seed; counts approx proportional to weights
    w = torch.softmax(torch.randn(50, generator=torch.Generator().manual_seed(1), dtype=dtype), 0)
    a = _systematic_resample(w, torch.Generator().manual_seed(123))
    b = _systematic_resample(w, torch.Generator().manual_seed(123))
    assert torch.equal(a, b)
    counts = torch.bincount(a, minlength=50).to(dtype)
    # systematic resampling: floor(N w_i) <= count_i <= ceil(N w_i)
    expected = 50 * w
    assert (counts >= expected.floor() - 1e-9).all()
    assert (counts <= expected.ceil() + 1e-9).all()


def test_next_dbeta_hits_ess_target():
    N = 500
    logw = torch.zeros(N, dtype=dtype)
    fX = torch.randn(N, generator=torch.Generator().manual_seed(0), dtype=dtype)
    lam = 2.0
    target = 0.5 * N

    dbeta = _next_dbeta(logw, fX, lam, beta=0.0, ess_target_count=target)
    assert 0.0 < dbeta <= 1.0
    ess = _ess_from_logw(logw + dbeta * lam * fX)
    assert ess == pytest.approx(target, abs=0.01 * N)

    # monotonicity: a larger step drops ESS below target
    ess_more = _ess_from_logw(logw + min(1.0, dbeta * 1.5) * lam * fX)
    assert ess_more <= ess + 1e-6


def test_next_dbeta_clamps_to_final_level_when_lambda_zero():
    N = 200
    logw = torch.zeros(N, dtype=dtype)
    fX = torch.randn(N, generator=torch.Generator().manual_seed(0), dtype=dtype)
    # lam = 0 -> incremental weights vanish, ESS stays at N -> take the whole remaining step
    dbeta = _next_dbeta(logw, fX, lam=0.0, beta=0.3, ess_target_count=0.5 * N)
    assert dbeta == pytest.approx(0.7)


# ===========================================================================
# Determinism
# ===========================================================================

def test_multi_level_schedule_terminates():
    """High lambda + a wide f-spread forces several tempering levels.

    Regression guard: a non-final level sits at the ESS target by construction and MUST
    resample, else the schedule stalls at dβ→0 and never reaches β=1.
    """
    p, _ = _ring_density()
    D = LpDistance(2.0)
    # references clustered on one side -> large spread in f across the ring
    pool = p.sample(400, seed=0)
    refs = pool[pool[:, 0] > 1.5][:40]

    res = adaptive_tempering_smc_sample(Reward(D, refs), lam=6.0, n_particles=2000,
                     kernel=IndependenceKernel(p), n_mcmc=4, final_resample=True, seed=0)
    assert len(res.betas) > 1                 # genuinely multi-level
    assert res.betas[-1] == pytest.approx(1.0)
    assert all(0.0 < b <= 1.0 + 1e-9 for b in res.betas)
    assert res.X.isfinite().all()


def test_adaptive_matches_fixed_when_capped():
    """Refactor guard: adaptive with min=max=k reduces to fixed n_mcmc=k, bit-for-bit.

    Proves the step()/snapshot/decorrelation split did not perturb the RNG threading.
    """
    p, _ = _ring_density()
    D = LpDistance(2.0)
    reward = RandomRefs(p, distance=D, seed=0).reward(6)
    k = 3
    a = adaptive_tempering_smc_sample(reward, lam=3.0, n_particles=500, kernel=IndependenceKernel(p),
                   n_mcmc=None, stop=RejuvenationStop(min_n_mcmc=k, max_n_mcmc=k), seed=0)
    b = adaptive_tempering_smc_sample(reward, lam=3.0, n_particles=500, kernel=IndependenceKernel(p),
                   n_mcmc=k, seed=0)
    assert torch.equal(a.X, b.X)
    assert torch.equal(a.logw, b.logw)
    assert a.n_mcmc_history == b.n_mcmc_history == [k] * len(a.betas)


def test_adaptive_rejuvenation_records_bounded_effort():
    """n_mcmc=None runs adaptive per level: one entry per β, each within [min_n_mcmc, MAX_N_MCMC]."""
    p, _ = _ring_density()
    D = LpDistance(2.0)
    pool = p.sample(400, seed=0)
    refs = pool[pool[:, 0] > 1.5][:40]           # clustered refs -> multi-level schedule
    stop = RejuvenationStop()
    res = adaptive_tempering_smc_sample(Reward(D, refs), lam=6.0, n_particles=1500,
                     kernel=IndependenceKernel(p), n_mcmc=None, stop=stop, seed=0)
    assert len(res.n_mcmc_history) == len(res.betas)
    assert all(stop.min_n_mcmc <= n <= MAX_N_MCMC for n in res.n_mcmc_history)
    assert res.X.isfinite().all()
    # determinism of the adaptive path (schedule + particles)
    res2 = adaptive_tempering_smc_sample(Reward(D, refs), lam=6.0, n_particles=1500,
                      kernel=IndependenceKernel(p), n_mcmc=None, stop=stop, seed=0)
    assert torch.equal(res.X, res2.X)
    assert res.n_mcmc_history == res2.n_mcmc_history


def test_independence_decorrelation_metric():
    """decorrelation() == 1 before any accept (all particles at their parent), drops after a step."""
    p, _ = _ring_density()
    D = LpDistance(2.0)
    reward = RandomRefs(p, distance=D, seed=0).reward(6)
    ker = IndependenceKernel(p)
    gen = torch.Generator().manual_seed(0)
    state = ker.init(500, reward, generator=gen, device=torch.device("cpu"), dtype=dtype)
    base = ker.snapshot_baseline(state)
    assert ker.decorrelation(state, base) == pytest.approx(1.0)
    ker.step(state, beta=1.0, lam=6.0, f=reward, generator=gen)
    assert ker.decorrelation(state, base) < 1.0


def test_determinism_same_seed():
    p, _ = _ring_density()
    D = LpDistance(2.0)
    reward = RandomRefs(p, distance=D, seed=0).reward(6)
    r1 = adaptive_tempering_smc_sample(reward, lam=3.0, n_particles=200, kernel=IndependenceKernel(p), n_mcmc=2, seed=0)
    r2 = adaptive_tempering_smc_sample(reward, lam=3.0, n_particles=200, kernel=IndependenceKernel(p), n_mcmc=2, seed=0)
    assert torch.equal(r1.X, r2.X)
    assert torch.equal(r1.logw, r2.logw)
    assert r1.betas == r2.betas


# ===========================================================================
# Per-level snapshots (keep_levels): one run == a lambda sweep
# ===========================================================================

def _multilevel_run(**kwargs):
    """A genuinely multi-level run (clustered refs -> wide f-spread), plus its reward."""
    p, _ = _ring_density()
    D = LpDistance(2.0)
    pool = p.sample(400, seed=0)
    refs = pool[pool[:, 0] > 1.5][:40]
    reward = Reward(D, refs)
    res = adaptive_tempering_smc_sample(reward, lam=6.0, n_particles=300,
                                        kernel=IndependenceKernel(p), n_mcmc=3, seed=0, **kwargs)
    return res, reward


def test_keep_levels_does_not_perturb_the_run():
    """The load-bearing test: snapshotting only reads state, so it must not touch the RNG stream."""
    off, _ = _multilevel_run()
    on, _ = _multilevel_run(keep_levels=True)
    assert torch.equal(off.X, on.X)
    assert torch.equal(off.logw, on.logw)
    assert off.betas == on.betas
    assert off.ess_history == on.ess_history
    assert off.n_mcmc_history == on.n_mcmc_history
    assert off.levels is None


def test_level_snapshots_bookkeeping():
    """levels[0] is the beta=0 baseline; levels[k] carries betas[k-1] and lam_eff = beta*lam."""
    res, _ = _multilevel_run(keep_levels=True)
    levels = res.levels
    assert levels is not None
    assert len(res.betas) > 1                       # the sweep is only interesting when multi-level
    assert len(levels) == len(res.betas) + 1        # +1 for the free untilted lam_eff=0 row
    assert levels[0].beta == 0.0 and levels[0].lam_eff == 0.0
    for k, beta in enumerate(res.betas, start=1):
        assert levels[k].beta == pytest.approx(beta)
        assert levels[k].lam_eff == pytest.approx(beta * 6.0)
    # lam_eff ascends: that ordering is what makes the snapshots a lambda sweep.
    assert all(a.lam_eff <= b.lam_eff for a, b in zip(levels, levels[1:]))


def test_level_snapshot_fX_matches_its_particles():
    """fX must be the reward *at* the snapshot's X — otherwise the free E_q[f] curve is a lie."""
    res, reward = _multilevel_run(keep_levels=True)
    assert res.levels is not None
    for lvl in res.levels:
        assert torch.allclose(reward(lvl.X), lvl.fX)
        assert lvl.X.shape[0] == lvl.fX.shape[0] == lvl.logw.shape[0] == 300


def test_level_snapshots_are_clones_not_views():
    res, _ = _multilevel_run(keep_levels=True, snapshot_device=None)
    assert res.levels is not None
    before = res.levels[-1].X.clone()
    res.X.add_(1.0)                                  # mutate the live result in place
    assert torch.equal(res.levels[-1].X, before)


def test_level_snapshot_ancestors():
    """ancestors indexes the previous snapshot's X; arange on levels that skipped resampling."""
    res, _ = _multilevel_run(keep_levels=True)
    assert res.levels is not None
    n = res.X.shape[0]
    for lvl in res.levels:
        assert lvl.ancestors.shape == (n,)
        assert lvl.ancestors.dtype == torch.int64
        assert bool((lvl.ancestors >= 0).all() and (lvl.ancestors < n).all())
    assert torch.equal(res.levels[0].ancestors, torch.arange(n))   # nothing precedes the initial cloud


def test_snapshot_device_defaults_to_cpu():
    res, _ = _multilevel_run(keep_levels=True)
    assert res.levels is not None
    assert all(lvl.X.device.type == "cpu" for lvl in res.levels)


# ===========================================================================
# 2D recovery vs grid_normalize ground truth
# ===========================================================================

def test_2d_recovery_lp():
    p, _ = _ring_density()
    D = LpDistance(2.0)
    lam, grid_n, N = 3.0, 24, 4000

    sel = RandomRefs(p, distance=D, seed=0)
    reward = sel.reward(6)
    gp, q_pmf = _grid_q_pmf(p, reward, lam, grid_n)

    res = adaptive_tempering_smc_sample(reward, lam=lam, n_particles=N,
                     kernel=IndependenceKernel(p), n_mcmc=4, seed=0)
    w = _smc_weights(res)
    hist = _hist_pmf(res.X, w, grid_n)

    assert _tv(hist, q_pmf) < 0.15

    # reward match: E_q[f] from SMC ~ grid Sigma q*f
    f_grid = reward(gp)
    Ef_grid = float((q_pmf * f_grid).sum())
    f_smc = reward(res.X)
    Ef_smc = float((w * f_smc).sum())
    assert Ef_smc == pytest.approx(Ef_grid, rel=0.12)


@pytest.mark.slow
def test_2d_recovery_global_iem():
    p, _ = _ring_density()
    gammas = torch.logspace(-6, 6, 12, base=2, dtype=dtype)
    D = GlobalIEMDistance(p, gammas, num_eps=4, seed=123)
    lam, grid_n, N = 2.0, 16, 1500

    sel = RandomRefs(p, distance=D, seed=0)
    reward = sel.reward(4)
    gp, q_pmf = _grid_q_pmf(p, reward, lam, grid_n)

    res = adaptive_tempering_smc_sample(reward, lam=lam, n_particles=N,
                     kernel=IndependenceKernel(p), n_mcmc=2, seed=0)
    w = _smc_weights(res)
    hist = _hist_pmf(res.X, w, grid_n)

    # IEM path on a coarse grid / few particles -> looser tolerance than Lp
    assert _tv(hist, q_pmf) < 0.2

    f_grid = reward(gp)
    Ef_grid = float((q_pmf * f_grid).sum())
    f_smc = reward(res.X)
    Ef_smc = float((w * f_smc).sum())
    assert Ef_smc == pytest.approx(Ef_grid, rel=0.2)


def test_lambda0_recovers_p():
    p, _ = _ring_density()
    D = LpDistance(2.0)
    reward = RandomRefs(p, distance=D, seed=0).reward(6)
    N = 5000

    res = adaptive_tempering_smc_sample(reward, lam=0.0, n_particles=N, kernel=IndependenceKernel(p), n_mcmc=3, seed=0)
    w = _smc_weights(res)
    hist_smc = _hist_pmf(res.X, w, grid_n=20)

    torch.manual_seed(1)
    base = p.sample(N)
    hist_p = _hist_pmf(base, torch.ones(N, dtype=dtype), grid_n=20)

    assert _tv(hist_smc, hist_p) < 0.12


def test_monotonicity_in_lambda():
    """Larger lambda shifts mass to higher-f regions -> larger E_q[f]."""
    p, _ = _ring_density()
    D = LpDistance(2.0)
    reward = RandomRefs(p, distance=D, seed=0).reward(6)

    def Ef(lam):
        res = adaptive_tempering_smc_sample(reward, lam=lam, n_particles=3000, kernel=IndependenceKernel(p), n_mcmc=3, seed=0)
        w = _smc_weights(res)
        return float((w * reward(res.X)).sum())

    assert Ef(0.0) < Ef(2.0) < Ef(5.0)


# ===========================================================================
# Selector swap: RandomRefs (uniform) vs WeightedFPSRefs (Voronoi weights)
# ===========================================================================

def test_selector_swap_random_vs_weighted_fps():
    p, _ = _ring_density()
    D = LpDistance(2.0)
    N = 400

    sel_u = RandomRefs(p, distance=D, seed=0)
    reward_u = sel_u.reward(8)
    assert reward_u.weights is None                         # uniform selector -> no weights
    res_u = adaptive_tempering_smc_sample(reward_u, lam=3.0, n_particles=N,
                       kernel=IndependenceKernel(p), n_mcmc=2, seed=0)
    assert res_u.X.shape == (N, 2)
    assert res_u.X.isfinite().all()

    sel_w = WeightedFPSRefs(
        p, distance=D, seed=0, pool_size=300, est_floor=300, points_per_cell=16
    )
    reward_w = sel_w.reward(8)                              # Voronoi weights ride along, can't be dropped
    refs_w, w = reward_w.x_refs, reward_w.weights
    assert w is not None and w.shape == (8,)
    res_w = adaptive_tempering_smc_sample(reward_w, lam=3.0, n_particles=N,
                       kernel=IndependenceKernel(p), n_mcmc=2, seed=0)
    assert res_w.X.shape == (N, 2)
    assert res_w.X.isfinite().all()

    # the weighted run genuinely USES the Voronoi weights: f with weights differs
    # from the uniform reduction on the same points (modes are unbalanced).
    X = p.sample(64)
    f_weighted = expected_distance(D, X, refs_w, weights=w)
    f_uniform = expected_distance(D, X, refs_w, weights=None)
    assert not torch.allclose(f_weighted, f_uniform)


def test_reward_weighted_pipeline_consistency():
    """A WeightedFPSRefs `reward` drives BOTH the grid ground truth and the SMC sampler with the
    same non-uniform weights, so the recovered E_q[f] matches the grid E_q[f]. This is the property
    the Reward bundle guarantees: switching to a weighted selector is correct on both paths with no
    hand-threaded weights."""
    p, _ = _ring_density()
    D = LpDistance(2.0)
    lam, grid_n, N = 3.0, 24, 4000

    sel = WeightedFPSRefs(
        p, distance=D, seed=0, pool_size=300, est_floor=300, points_per_cell=16
    )
    reward = sel.reward(8)
    assert reward.weights is not None
    # weights are genuinely non-uniform (otherwise the test would not exercise the weighting)
    assert not torch.allclose(reward.weights, torch.full_like(reward.weights, 1.0 / reward.weights.shape[0]))

    gp, q_pmf = _grid_q_pmf(p, reward, lam, grid_n)         # grid truth uses reward's weights
    res = adaptive_tempering_smc_sample(reward, lam=lam, n_particles=N,
                     kernel=IndependenceKernel(p), n_mcmc=4, seed=0)
    w = _smc_weights(res)

    Ef_grid = float((q_pmf * reward(gp)).sum())
    Ef_smc = float((w * reward(res.X)).sum())
    assert Ef_smc == pytest.approx(Ef_grid, rel=0.12)
