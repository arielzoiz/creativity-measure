"""Tests for creativity_measure.distances.global_iem_expected (ExpectedSquaredGlobalIEMDistance).

The shared-Brownian squared IEM plus `expected` / `expected_gamma_chunk`. All on the same 2D GMM toy
as test_global_iem.py / test_iid_global_iem.py, float64 on CPU unless a test says otherwise.

What these prove: the closed-form reference mean is an algebraic IDENTITY rather than an approximation
(it tightens from ~1e-7 to ~1e-16 when float32 becomes float64); the gamma-chunked partition
reproduces it over the INTERVAL axis, which is one shorter than the grid; and nothing leaked onto
`SquaredGlobalIEMDistance`, whose dispatch every Algorithm 1-3 run depends on. What they do NOT prove
is that an N_gamma = 11 grid is accurate enough on FLUX -- that is the GPU check.
"""
import math

import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from creativity_measure.density import Density
from creativity_measure.device import set_default_device
from creativity_measure.distances.base import ExpectedDistance
from creativity_measure.distances.global_iem import SquaredGlobalIEMDistance
from creativity_measure.distances.global_iem_expected import ExpectedSquaredGlobalIEMDistance
from creativity_measure.distances.iid_global_iem import SquaredIIDGlobalIEMDistance
from creativity_measure.distances.utils import log_uniform_gammas
from creativity_measure.samplers.flow_guided_common import _reward_grad
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
    """Closed-form, differentiable score of p_Yg. `means` is cast to y so the fp32 test stays fp32."""
    g = torch.as_tensor(gamma).to(y).reshape(-1, 1)
    mu_k = means.to(y)
    v = g * g * sig2 + g
    mu = g.unsqueeze(1) * mu_k.unsqueeze(0)
    resp = torch.softmax(-((y.unsqueeze(1) - mu) ** 2).sum(-1) / (2 * v), dim=1)
    return (resp.unsqueeze(-1) * (mu - y.unsqueeze(1))).sum(1) / v


N_GAMMA = 13                       # 13 grid POINTS -> 12 integration INTERVALS; the off-by-one matters
N_INT = N_GAMMA - 1
grid = torch.logspace(-5, 5, N_GAMMA, base=2.0, dtype=dtype)
X = torch.tensor([[2., 0.], [0., 0.], [-1., 0.5]], dtype=dtype)             # (3, 2)
set_default_device("cpu")
x_refs = p.sample(4, seed=0)                                                # (4, 2)
set_default_device(None)
W_NONUNIFORM = torch.tensor([0.4, 1.3, 0.7, 2.1], dtype=dtype)


def make(**kw) -> ExpectedSquaredGlobalIEMDistance:
    kw.setdefault("num_eps", 2)
    kw.setdefault("seed", 11)
    kw.setdefault("score_fn", gmm_score)
    return ExpectedSquaredGlobalIEMDistance(None, grid, **kw)


def make_base(**kw) -> SquaredGlobalIEMDistance:
    kw.setdefault("num_eps", 2)
    kw.setdefault("seed", 11)
    kw.setdefault("score_fn", gmm_score)
    return SquaredGlobalIEMDistance(None, grid, **kw)


def weighted_row_mean(pw, weights):
    if weights is None:
        return pw.mean(dim=1)
    w = weights.to(pw)
    return (pw * w).sum(dim=1) / w.sum()


# ---------------------------------------------------------------------------
# The subclass must not disturb the inherited behaviour
# ---------------------------------------------------------------------------

def test_pairwise_is_unchanged_from_the_base_class():
    """Detaching the reference bank is a graph change, not a numerical one."""
    assert torch.equal(make().pairwise(X, x_refs), make_base().pairwise(X, x_refs))


def test_base_squared_class_is_still_not_an_expected_distance():
    """Regression guard: `expected` must NOT have leaked onto SquaredGlobalIEMDistance.

    `tilt.expected_distance` dispatches on this Protocol, so acquiring an `expected` would silently
    reroute every Algorithm 1-3 caller onto a different float reduction order.
    """
    base = make_base()
    assert not hasattr(base, "expected")
    assert not isinstance(base, ExpectedDistance)
    assert isinstance(make(), ExpectedDistance)


def test_input_validation():
    """A 1-point grid defines ZERO intervals, so it must be rejected at construction.

    (The non-1-D case is already rejected upstream by jaxtyping's runtime check on the `gammas`
    annotation; the `ndim` guard in __init__ is defence for callers that bypass it.)
    """
    with pytest.raises(ValueError, match="at least 2 points"):
        ExpectedSquaredGlobalIEMDistance(None, grid[:1], score_fn=gmm_score)


# ---------------------------------------------------------------------------
# expected(): the closed-form reference mean
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("weights", [None, W_NONUNIFORM], ids=["uniform", "nonuniform"])
def test_expected_equals_weighted_mean_of_pairwise(weights):
    d = make()
    truth = weighted_row_mean(d.pairwise(X, x_refs), weights)
    assert torch.allclose(d.expected(X, x_refs, weights), truth, rtol=1e-10, atol=0.0)


def test_expected_is_an_identity_not_an_approximation():
    """The gap to pairwise().mean() must track machine epsilon, not stay fixed.

    An approximation would be precision-independent; an algebraic identity sits at machine epsilon in
    whatever precision it is evaluated. So the float64 gap must reach ~1e-16, not merely be "small".
    """
    errs = {}
    for dt, tol in ((torch.float32, 1e-5), (torch.float64, 1e-14)):
        g = grid.to(dt)
        d = ExpectedSquaredGlobalIEMDistance(None, g, num_eps=2, seed=11, score_fn=gmm_score)
        Xd, refs = X.to(dt), x_refs.to(dt)
        pw = weighted_row_mean(d.pairwise(Xd, refs), None)
        errs[dt] = ((d.expected(Xd, refs) - pw).abs() / pw.abs()).max().item()
        assert errs[dt] < tol, f"{dt}: {errs[dt]:.3e} >= {tol:.0e}"
    # float64 at ~1e-16 is the identity signature. (float32 can land on exactly 0.0 -- bitwise
    # agreement -- so a ratio test between the two precisions would be flaky, not stronger.)
    assert errs[torch.float64] < 1e-14


def test_expected_distance_takes_the_fast_path():
    """tilt.expected_distance must dispatch to expected(), not build the (B, R) matrix."""
    d = make()
    calls = {"n": 0}
    real_pairwise = d.pairwise

    def counting_pairwise(*a, **kw):
        calls["n"] += 1
        return real_pairwise(*a, **kw)

    d.pairwise = counting_pairwise                      # type: ignore[method-assign]
    out = expected_distance(d, X, x_refs)
    assert calls["n"] == 0
    assert out.shape == (X.shape[0],)


def test_expected_cache_off_matches_on():
    a = make(cache_refs=True).expected(X, x_refs)
    b = make(cache_refs=False).expected(X, x_refs)
    assert torch.allclose(a, b, rtol=1e-12, atol=0.0)


def test_f_at_refs_is_r_minus_one_over_r():
    """Normalization wiring: mean_r f(x_r) = (R-1)/R exactly, for uniform weights.

    The numerator averages R terms including the zero self-pair while `reference_pair_mean` divides by
    R(R-1). Tolerance 1e-3, not 0.10 -- the loose one cannot separate 3/4 from the failure modes that
    land on exactly 1.0.
    """
    R = x_refs.shape[0]
    f = NormalizedExpectedDistanceReward(make(), x_refs)(x_refs)
    assert abs(float(f.mean()) - (R - 1) / R) < 1e-3


# ---------------------------------------------------------------------------
# expected_gamma_chunk(): partition over the INTERVAL axis
# ---------------------------------------------------------------------------

def test_n_gamma_chunks_is_intervals_not_points():
    d = make()
    assert d.n_gamma_chunks == N_INT == d.gammas.shape[0] - 1


@pytest.mark.parametrize("chunk", [1, 2, 5, N_INT])
@pytest.mark.parametrize("weights", [None, W_NONUNIFORM], ids=["uniform", "nonuniform"])
def test_expected_gamma_chunk_partition_matches_expected(chunk, weights):
    d = make()
    full = d.expected(X, x_refs, weights)
    total = torch.zeros_like(full)
    for lo in range(0, N_INT, chunk):
        total = total + d.expected_gamma_chunk(X, x_refs, lo, min(lo + chunk, N_INT), weights)
    assert torch.allclose(total, full, rtol=1e-10, atol=0.0)


def test_expected_gamma_chunk_gradient_matches_expected():
    d = make()
    xg = X.clone().requires_grad_(True)
    (g_full,) = torch.autograd.grad(d.expected(xg, x_refs).sum(), xg)

    xc = X.clone().requires_grad_(True)
    g_chunked = torch.zeros_like(xc)
    for lo in range(N_INT):
        part = d.expected_gamma_chunk(xc, x_refs, lo, lo + 1)
        g_chunked = g_chunked + torch.autograd.grad(part.sum(), xc)[0]
    assert torch.allclose(g_chunked, g_full, rtol=1e-9, atol=1e-12)


def test_expected_gamma_chunk_rejects_bad_bounds():
    d = make()
    # N_GAMMA is a valid POINT index but not a valid interval bound -- the off-by-one this guards.
    for lo, hi in ((0, N_GAMMA), (0, N_INT + 1), (-1, 3), (3, 3), (4, 2)):
        with pytest.raises(ValueError, match="gamma intervals"):
            d.expected_gamma_chunk(X, x_refs, lo, hi)
    assert d.expected_gamma_chunk(X, x_refs, 0, N_INT).shape == (X.shape[0],)


def test_chunking_uses_the_same_frozen_brownian_rows():
    """A chunk must reuse the full bank's rows, never redraw -- invariant 1 (f deterministic in x).

    If the noise were regenerated per chunk the partition would still sum to *something*, just not to
    expected(); comparing a single interval against its slice of the full integral catches that.
    """
    d = make()
    one = d.expected_gamma_chunk(X, x_refs, 3, 4)
    rest = sum(d.expected_gamma_chunk(X, x_refs, i, i + 1) for i in range(N_INT) if i != 3)
    assert torch.allclose(one + rest, d.expected(X, x_refs), rtol=1e-10, atol=0.0)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def test_gradcheck_reward_wrt_x():
    reward = Reward(make(num_eps=1), x_refs)
    x = torch.tensor([[0.3, -0.2]], dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: reward(z).sum(), (x,), eps=1e-6, atol=1e-6)


def test_gradient_is_nonzero_and_matches_pairwise_route():
    d = make()
    xa = X.clone().requires_grad_(True)
    (g_exp,) = torch.autograd.grad(d.expected(xa, x_refs).sum(), xa)
    xb = X.clone().requires_grad_(True)
    (g_pw,) = torch.autograd.grad(d.pairwise(xb, x_refs).mean(dim=1).sum(), xb)
    assert g_exp.abs().max() > 0
    assert torch.allclose(g_exp, g_pw, rtol=1e-9, atol=1e-12)


def test_reference_bank_is_detached():
    d = make()
    refs = x_refs.clone().requires_grad_(True)
    out = d.expected(X, refs)
    assert not out.requires_grad          # X does not require grad; refs are constants regardless


def test_refs_vs_refs_alias_does_not_swallow_a_grad_request():
    """expected(x_refs, x_refs) reuses the detached bank -- unless x_refs itself wants a gradient."""
    d = make()
    with torch.no_grad():
        d.expected(x_refs, x_refs)                     # warm the cache via the alias
    xg = x_refs.clone().requires_grad_(True)
    out = d.expected(xg, x_refs)
    assert out.requires_grad
    (g,) = torch.autograd.grad(out.sum(), xg)
    assert g.abs().max() > 0


# ---------------------------------------------------------------------------
# Wiring into _reward_grad's gamma-chunked OOM fallback
# ---------------------------------------------------------------------------

def test_reward_grad_fallback_matches_unchunked():
    """_reward_grad must partition the INTERVAL axis for this class and reproduce the gradient."""
    reward = Reward(make(), x_refs)
    xa = X.clone().requires_grad_(True)
    g_direct, fell_a = _reward_grad(reward, xa, xa)
    xb = X.clone().requires_grad_(True)
    g_fallback, fell_b = _reward_grad(reward, xb, xb, force_fallback=True, g_chunk=1)
    assert not fell_a and fell_b
    assert torch.allclose(g_fallback, g_direct, rtol=1e-9, atol=1e-12)


def test_reward_grad_chunk_count_is_unchanged_for_the_iid_class():
    """The n_gamma_chunks override must not perturb SquaredIIDGlobalIEMDistance's fallback.

    Its gamma axis IS points, so it must keep falling back to gammas.shape[0]; it therefore must not
    carry the attribute at all.
    """
    g_iid, gw = log_uniform_gammas(2.0 ** -5, 2.0 ** 5, 7, seed=3, dtype=dtype)
    iid = SquaredIIDGlobalIEMDistance(None, g_iid, gw, num_eps=2, seed=11, score_fn=gmm_score)
    assert getattr(iid, "n_gamma_chunks", None) is None

    reward = Reward(iid, x_refs)
    xa = X.clone().requires_grad_(True)
    g_direct, _ = _reward_grad(reward, xa, xa)
    xb = X.clone().requires_grad_(True)
    g_fallback, fell = _reward_grad(reward, xb, xb, force_fallback=True, g_chunk=1)
    assert fell
    assert torch.allclose(g_fallback, g_direct, rtol=1e-9, atol=1e-12)


def test_normalized_reward_gradient_flows_through_the_expected_route():
    reward = NormalizedExpectedDistanceReward(make(), x_refs)
    x = X.clone().requires_grad_(True)
    g, fell = _reward_grad(reward, x, x, force_fallback=True, g_chunk=3)
    assert fell and g.abs().max() > 0
