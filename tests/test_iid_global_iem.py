"""Tests for creativity_measure.distances.iid_global_iem (frozen i.i.d. Monte-Carlo squared IEM).

All on the 2D GMM toy in float64, on CPU. They prove the math, the caching, the closed-form fast path and gradient
flow -- not that a frozen G ~ 30 bank is accurate enough on FLUX (that is the GPU check in
notebooks/iid_iem_flux_check/).
"""
import math

import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from creativity_measure.density import Density
from creativity_measure.device import set_default_device
from creativity_measure.distances.base import ExpectedDistance
from creativity_measure.distances.global_iem import GlobalIEMDistance, SquaredGlobalIEMDistance
from creativity_measure.distances.iid_global_iem import (
    IIDGlobalIEMDistance,
    SquaredIIDGlobalIEMDistance,
    iid_score_bank,
)
from creativity_measure.distances.lp import LpDistance
from creativity_measure.distances.utils import log_uniform_gammas, simulate_brownian, simulate_iid_noise
from creativity_measure.scores import marginal_score
from creativity_measure.tilt import NormalizedExpectedDistanceReward, Reward, expected_distance

dtype = torch.float64

# ---------------------------------------------------------------------------
# Fixture: the same 2-mode GMM as test_global_iem.py
# ---------------------------------------------------------------------------
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)
sig2 = 0.09


def log_pX(x):
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(Categorical(torch.ones(2, dtype=dtype)),
                            MultivariateNormal(means, covs, validate_args=False))
    return mix.log_prob(x)


def log_pY(y, gamma):
    var = gamma ** 2 * sig2 + gamma
    d = y.shape[-1]
    flat = y.reshape(-1, d)
    comps = [MultivariateNormal(gamma * means[k], var * torch.eye(d, dtype=y.dtype),
                                validate_args=False).log_prob(flat) for k in range(2)]
    return torch.logsumexp(-math.log(2) + torch.stack(comps, dim=0), dim=0).reshape(y.shape[:-1])


def sampler(n, generator=None):
    del generator
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(Categorical(torch.ones(2, dtype=dtype)),
                            MultivariateNormal(means, covs, validate_args=False))
    return mix.sample(torch.Size((n,)))


p = Density(log_pX, log_pY, sample_fn=sampler, d=2)


def gmm_score(y, gamma):
    """Closed-form, differentiable score of p_Yg for the GMM; `gamma` is a scalar OR a (rows,) vector.

    Both components share one variance v = gamma^2 sig2 + gamma, so the normalizers cancel in the responsibilities.
    """
    g = gamma.reshape(-1, 1)                                               # (1, 1) or (rows, 1)
    v = g * g * sig2 + g
    mu = g.unsqueeze(1) * means.unsqueeze(0)                               # (1|rows, 2, d)
    resp = torch.softmax(-((y.unsqueeze(1) - mu) ** 2).sum(-1) / (2 * v), dim=1)   # (rows, 2)
    return (resp.unsqueeze(-1) * (mu - y.unsqueeze(1))).sum(1) / v


LO, HI = 2.0 ** -10, 2.0 ** 10
G0 = 12
gammas, gweights = log_uniform_gammas(LO, HI, G0, seed=3, dtype=dtype)
X = torch.tensor([[2., 0.], [0., 0.], [-1., 0.5]], dtype=dtype)             # (3, 2)
# Pin CPU for this module-level sample: default_device() auto-selects CUDA on a GPU host, which
# would put x_refs on a different device than the CPU literal X above.
set_default_device("cpu")
x_refs = p.sample(4, seed=0)                                                # (4, 2)
set_default_device(None)


def make(**kw) -> SquaredIIDGlobalIEMDistance:
    kw.setdefault("num_eps", 2)
    kw.setdefault("seed", 11)
    kw.setdefault("score_fn", gmm_score)
    return SquaredIIDGlobalIEMDistance(None, gammas, gweights, **kw)


def make_plain(**kw) -> IIDGlobalIEMDistance:
    kw.setdefault("num_eps", 2)
    kw.setdefault("seed", 11)
    kw.setdefault("score_fn", gmm_score)
    return IIDGlobalIEMDistance(None, gammas, gweights, **kw)


# ---------------------------------------------------------------------------
# Helpers: log_uniform_gammas and simulate_iid_noise
# ---------------------------------------------------------------------------

def test_log_uniform_gammas_range_sorted_deterministic():
    g, w = log_uniform_gammas(LO, HI, 50, seed=1, dtype=dtype)
    assert g.shape == w.shape == (50,)
    assert (g >= LO).all() and (g <= HI).all() and (w > 0).all()
    assert (g[1:] >= g[:-1]).all(), "returned sorted"
    g2, w2 = log_uniform_gammas(LO, HI, 50, seed=1, dtype=dtype)
    assert torch.equal(g, g2) and torch.equal(w, w2)
    g3, _ = log_uniform_gammas(LO, HI, 50, seed=2, dtype=dtype)
    assert not torch.equal(g, g3)


def test_log_uniform_gammas_weights_integrate_unbiasedly():
    """sum_g w_g * 1 estimates the integral of 1 over [lo, hi] = hi - lo; the mean over many draws must hit it."""
    lo, hi, G = 1.0, 8.0, 100
    est = torch.stack([log_uniform_gammas(lo, hi, G, seed=s, dtype=dtype)[1].sum() for s in range(400)])
    se = est.std() / math.sqrt(est.numel())
    assert abs(est.mean() - (hi - lo)) < 4 * se, (est.mean(), hi - lo, se)


def test_simulate_iid_noise_shape_marginal_and_determinism():
    g = torch.tensor([0.25, 1.0, 16.0], dtype=dtype)
    W = simulate_iid_noise(g, num_eps=4, d=20000, seed=5, device=torch.device("cpu"), dtype=dtype)
    assert W.shape == (3, 4, 1, 20000)
    # same layout as simulate_brownian's bank
    assert W.shape[1:] == simulate_brownian(g, 4, 20000, 5, torch.device("cpu"), dtype).shape[1:]
    # marginal W_g ~ N(0, g I): per-coordinate variance ~ gamma
    var = W.pow(2).mean(dim=(1, 2, 3))
    assert torch.allclose(var, g, rtol=0.03), (var, g)
    W2 = simulate_iid_noise(g, 4, 20000, 5, torch.device("cpu"), dtype)
    assert torch.equal(W, W2)
    # i.i.d. across (g, eps): no shared randomness between levels (Brownian W would correlate them)
    a, b = W[0, 0, 0] / g[0].sqrt(), W[1, 0, 0] / g[1].sqrt()
    assert abs(float((a * b).mean())) < 0.03


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_fixture_score_matches_autograd_density():
    y = torch.randn(6, 2, dtype=dtype)
    for g in (0.05, 1.0, 30.0):
        gt = torch.tensor(g, dtype=dtype)
        assert torch.allclose(gmm_score(y, gt), marginal_score(y, gt, p), atol=1e-9)
    # per-row gamma == a loop of scalar gammas
    gv = torch.tensor([0.05, 1.0, 30.0, 0.3, 2.0, 9.0], dtype=dtype)
    looped = torch.stack([gmm_score(y[i:i + 1], gv[i]) for i in range(6)]).squeeze(1)
    assert torch.allclose(gmm_score(y, gv), looped, atol=1e-12)


def test_shape_and_nonneg():
    out = make().pairwise(X, x_refs)
    assert out.shape == (3, 4)
    assert (out >= 0).all()


def test_iid_is_sqrt_of_squared_in_the_unsquared_class():
    sq = make().pairwise(X, x_refs)
    plain = make_plain().pairwise(X, x_refs)
    assert torch.allclose(plain, sq.sqrt())


def test_determinism_and_seed_sensitivity():
    d = make()
    assert torch.equal(d.pairwise(X, x_refs), d.pairwise(X, x_refs))
    assert torch.equal(make().pairwise(X, x_refs), make().pairwise(X, x_refs))
    assert not torch.allclose(make(seed=1).pairwise(X, x_refs), make(seed=2).pairwise(X, x_refs))


def test_density_autograd_path_matches_score_fn():
    """score_fn=None (autograd through log_p_Y, looped) reproduces the closed-form score_fn path."""
    a = SquaredIIDGlobalIEMDistance(p, gammas, gweights, num_eps=2, seed=11).pairwise(X, x_refs)
    b = make().pairwise(X, x_refs)
    assert torch.allclose(a, b, atol=1e-8, rtol=1e-8)


def test_input_validation():
    with pytest.raises(ValueError):
        SquaredIIDGlobalIEMDistance(None, gammas, gweights)                      # no density, no score_fn
    with pytest.raises(ValueError):
        SquaredIIDGlobalIEMDistance(p, gammas, gweights[:-1])                    # weights length mismatch
    with pytest.raises(ValueError):
        SquaredIIDGlobalIEMDistance(p, gammas, gweights, batched_gamma=True)     # batched needs a score_fn


# ---------------------------------------------------------------------------
# It estimates the same quantity as the Brownian implementation
# ---------------------------------------------------------------------------

def test_unbiased_against_fine_brownian_reference():
    """Mean over many frozen banks == the Brownian D^2 on a very fine grid with many paths.

    The reference is itself Monte Carlo (its noise is the larger of the two at small num_eps, hence 2 x 500 paths) and
    a left-endpoint rule (~0.5% at 1500 nodes), so the tolerance is 4 SE of the i.i.d. mean plus 3% of the value.
    """
    fine = torch.logspace(-10, 10, 1500, base=2, dtype=dtype)
    ref = sum(SquaredGlobalIEMDistance(p, fine, num_eps=500, seed=s).pairwise(X, x_refs) for s in (5, 6)) / 2
    n = 300
    draws = torch.stack([
        SquaredIIDGlobalIEMDistance(None, *log_uniform_gammas(LO, HI, 30, seed=s, dtype=dtype),
                                    num_eps=1, seed=s, score_fn=gmm_score).pairwise(X, x_refs)
        for s in range(n)])
    mean, se = draws.mean(0), draws.std(0) / math.sqrt(n)
    assert ((mean - ref).abs() < 4 * se + 0.03 * ref).all(), ((mean - ref) / ref, se / ref)


# ---------------------------------------------------------------------------
# batched_gamma, cache, chunking: knobs that change cost, never values
# ---------------------------------------------------------------------------

def test_batched_gamma_matches_looped():
    looped = make().pairwise(X, x_refs)
    fused = make(batched_gamma=True).pairwise(X, x_refs)
    assert torch.allclose(looped, fused, atol=1e-10, rtol=1e-10)
    # and the bank itself
    W = simulate_iid_noise(gammas, 2, 2, 11, torch.device("cpu"), dtype)
    a = iid_score_bank(X, gammas, W, None, gmm_score, batched_gamma=False)
    b = iid_score_bank(X, gammas, W, None, gmm_score, batched_gamma=True)
    assert a.shape == b.shape == (G0, 2, 3, 2)
    assert torch.allclose(a, b, atol=1e-12)


def test_ref_cache_matches_uncached():
    cached, plain = make(cache_refs=True), make(cache_refs=False)
    assert torch.equal(cached.pairwise(X, x_refs), plain.pairwise(X, x_refs))
    first = cached.pairwise(X, x_refs)
    assert torch.equal(cached.pairwise(X, x_refs), first)


def test_r_chunk_is_byte_exact():
    R = x_refs.shape[0]
    ref = make(r_chunk=R).pairwise(X, x_refs)
    for r_chunk in (1, 3, R):
        assert torch.equal(make(r_chunk=r_chunk).pairwise(X, x_refs), ref), r_chunk


@pytest.mark.parametrize("batched", [False, True])
def test_score_row_counts(batched):
    """Exact score rows per call: cold G*E*(R+B), warm G*E*B, refs-vs-refs 0. The regression guard for the cache."""
    rows = {"n": 0}

    def counting(y, g):
        rows["n"] += y.shape[0]
        return gmm_score(y, g)

    B, R, E = X.shape[0], x_refs.shape[0], 2
    d = SquaredIIDGlobalIEMDistance(None, gammas, gweights, num_eps=E, seed=11, score_fn=counting,
                                    batched_gamma=batched)
    d.pairwise(X, x_refs)
    assert rows["n"] == G0 * E * (R + B)
    rows["n"] = 0
    d.pairwise(X, x_refs)
    assert rows["n"] == G0 * E * B
    rows["n"] = 0
    d.pairwise(x_refs, x_refs)
    assert rows["n"] == 0, "refs-vs-refs must be served entirely by the cache"
    rows["n"] = 0
    d.expected(X, x_refs)
    assert rows["n"] == G0 * E * B, "expected() scores the batch only; the ref moments are cached"


# ---------------------------------------------------------------------------
# expected(): the closed-form mean over refs, and the reward fast path
# ---------------------------------------------------------------------------

def test_is_expected_distance_protocol_only_for_squared_iid():
    assert isinstance(make(), ExpectedDistance)
    assert not isinstance(make_plain(), ExpectedDistance)
    assert not isinstance(GlobalIEMDistance(p, gammas), ExpectedDistance)
    assert not isinstance(SquaredGlobalIEMDistance(p, gammas), ExpectedDistance)
    assert not isinstance(LpDistance(2.0), ExpectedDistance)


def test_expected_equals_weighted_mean_of_pairwise():
    d = make()
    pw = d.pairwise(X, x_refs)
    assert torch.allclose(d.expected(X, x_refs), pw.mean(1), rtol=1e-10, atol=1e-10)
    w = torch.tensor([0.1, 2.0, 0.5, 1.4], dtype=dtype)
    want = (pw * w).sum(1) / w.sum()
    assert torch.allclose(d.expected(X, x_refs, w), want, rtol=1e-10, atol=1e-10)
    # a second, different weight vector must not be served from the first one's cached moments
    w2 = torch.tensor([1.0, 1.0, 5.0, 0.2], dtype=dtype)
    assert torch.allclose(d.expected(X, x_refs, w2), (pw * w2).sum(1) / w2.sum(), rtol=1e-10, atol=1e-10)


def test_expected_gamma_chunk_partition_matches_expected():
    """A full partition of [0, G) reproduces expected() exactly (the OOM fallback's exactness claim,
    creativity_measure/flux_guided.py). Checked at B=1 too -- the case a batch-axis chunk could not
    have helped (score rows are G*N_eps*B), which is why this chunks gamma, not the batch."""
    d = make()
    G0 = gammas.shape[0]
    want = d.expected(X, x_refs)
    total = torch.zeros_like(want)
    for g_lo in range(0, G0, 2):
        g_hi = min(g_lo + 2, G0)
        total = total + d.expected_gamma_chunk(X, x_refs, g_lo, g_hi)
    assert torch.allclose(total, want, rtol=1e-10, atol=1e-10)

    want_b1 = d.expected(X[:1], x_refs)
    total_b1 = torch.zeros_like(want_b1)
    for g in range(G0):
        total_b1 = total_b1 + d.expected_gamma_chunk(X[:1], x_refs, g, g + 1)
    assert torch.allclose(total_b1, want_b1, rtol=1e-10, atol=1e-10)


def test_expected_gamma_chunk_gradient_matches_expected():
    """The gradient of the chunk-accumulated total must match the batched gradient -- what the OOM
    fallback actually needs (Phase 2 measured 1.86e-09 for the analogous per-sample split)."""
    d = make()
    G0 = gammas.shape[0]

    Xa = X.clone().requires_grad_(True)
    loss_a = d.expected(Xa, x_refs).sum()
    (ga,) = torch.autograd.grad(loss_a, Xa)

    Xb = X.clone().requires_grad_(True)
    total = None
    for g in range(G0):
        partial = d.expected_gamma_chunk(Xb, x_refs, g, g + 1).sum()
        total = partial if total is None else total + partial
    assert total is not None
    (gb,) = torch.autograd.grad(total, Xb)
    assert torch.allclose(ga, gb, rtol=1e-10, atol=1e-10)


def test_expected_gamma_chunk_rejects_bad_bounds():
    d = make()
    G0 = gammas.shape[0]
    with pytest.raises(ValueError):
        d.expected_gamma_chunk(X, x_refs, 0, G0 + 1)
    with pytest.raises(ValueError):
        d.expected_gamma_chunk(X, x_refs, 3, 3)
    with pytest.raises(ValueError):
        d.expected_gamma_chunk(X, x_refs, -1, 2)


def test_expected_cache_off_matches_on():
    on, off = make(cache_refs=True), make(cache_refs=False)
    assert torch.equal(on.expected(X, x_refs), off.expected(X, x_refs))


def test_expected_distance_takes_the_fast_path():
    calls = {"pairwise": 0}

    class Spy(SquaredIIDGlobalIEMDistance):
        def pairwise(self, X, x_refs):
            calls["pairwise"] += 1
            return super().pairwise(X, x_refs)

    d = Spy(None, gammas, gweights, num_eps=2, seed=11, score_fn=gmm_score)
    got = expected_distance(d, X, x_refs)
    assert calls["pairwise"] == 0, "expected_distance must use .expected, not pairwise"
    assert torch.allclose(got, super(Spy, d).pairwise(X, x_refs).mean(1), rtol=1e-10, atol=1e-10)
    # every other distance is untouched: still goes through pairwise
    assert expected_distance(LpDistance(2.0), X, x_refs).shape == (3,)


def test_reward_and_normalized_reward_use_the_iid_distance():
    d = make()
    r = Reward(d, x_refs)
    assert torch.allclose(r(X), d.pairwise(X, x_refs).mean(1), rtol=1e-10, atol=1e-10)
    nr = NormalizedExpectedDistanceReward(d, x_refs)
    R = x_refs.shape[0]
    # mean_r f(x_r) = (R-1)/R exactly (uniform weights): the numerator averages R terms incl. the zero self-pair, the
    # denominator averages the R(R-1) off-diagonal pairs. Tests the normalization wiring through the closed form.
    assert abs(float(nr(x_refs).mean()) - (R - 1) / R) < 1e-8
    assert (nr(X) > 0).all()


# ---------------------------------------------------------------------------
# Gradients: what Phases 2-4 are for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batched", [False, True])
def test_gradcheck_reward_wrt_x(batched):
    nr = NormalizedExpectedDistanceReward(make(batched_gamma=batched), x_refs)
    Xg = X.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda z: nr(z), (Xg,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_gradient_is_nonzero_and_matches_pairwise_route():
    """grad of the closed-form expected == grad of the explicit pairwise mean (the (B, R, d) route)."""
    d = make()
    Xa = X.clone().requires_grad_(True)
    ga = torch.autograd.grad(d.expected(Xa, x_refs).sum(), Xa)[0]
    Xb = X.clone().requires_grad_(True)
    gb = torch.autograd.grad(d.pairwise(Xb, x_refs).mean(1).sum(), Xb)[0]
    assert ga.abs().max() > 0
    assert torch.allclose(ga, gb, rtol=1e-8, atol=1e-10)


def test_reference_bank_is_detached():
    d = make()
    Xg = X.clone().requires_grad_(True)
    refs = x_refs.clone().requires_grad_(True)
    d.expected(Xg, refs)
    assert d._ref_cache is not None
    bank = d._ref_cache[2]
    assert not bank.requires_grad and bank.grad_fn is None
    assert d._ref_stats is not None and not d._ref_stats[3].requires_grad


def test_refs_vs_refs_alias_does_not_swallow_a_grad_request():
    """If X *is* the reference tensor but wants gradients, its scores must be evaluated with the graph, not aliased."""
    d = make()
    refs = x_refs.clone()
    d.expected(refs, refs)                       # warm the cache; alias path is legitimate here
    refs.requires_grad_(True)
    out = d.expected(refs, refs)
    assert out.requires_grad
