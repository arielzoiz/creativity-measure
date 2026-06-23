"""Tests for creativity_measure/tilt.py"""
import math

import torch
import pytest

from creativity_measure.density import Density
from creativity_measure.distances.lp import LpDistance
from creativity_measure.tilt import expected_distance, tilted_log_density, grid_normalize
from refset import RandomRefs, WeightedFPSRefs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flat_density():
    """A Density whose log_p_X is identically 0 (uniform / unnormalized).

    No log_p_Y is supplied: these tests score with LpDistance, which needs no marginal score.
    """
    return Density(log_p_X=lambda x: torch.zeros(x.shape[0], dtype=x.dtype))


# ---------------------------------------------------------------------------
# 1. expected_distance: shape and ordering
# ---------------------------------------------------------------------------

def test_expected_distance_shape_and_ordering():
    X = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    x_refs = torch.zeros(5, 2, dtype=torch.float64)

    result = expected_distance(LpDistance(2.0), X, x_refs)

    assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"

    # [0,0] is at zero distance from all refs (which are also zeros)
    assert result[0].item() == pytest.approx(0.0, abs=1e-9), (
        f"Near point expected distance should be 0, got {result[0].item()}"
    )
    # [10,0] is 10 units from all zeros refs
    assert result[1].item() > result[0].item(), (
        "Far point should have larger expected distance than near point"
    )


# ---------------------------------------------------------------------------
# 2. tilted_log_density at lambda=0 recovers log p
# ---------------------------------------------------------------------------

def test_tilted_log_density_lam0_recovers_log_p():
    p = flat_density()
    X = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    x_refs = torch.zeros(5, 2, dtype=torch.float64)

    result = tilted_log_density(X, p, LpDistance(2.0), x_refs, lam=0.0)

    assert result.shape == (2,)
    assert torch.allclose(result, torch.zeros(2, dtype=torch.float64)), (
        f"At lambda=0, tilted log density should equal log p (all zeros), got {result}"
    )


# ---------------------------------------------------------------------------
# 3. tilted_log_density at lambda>0 raises far-from-refs points
# ---------------------------------------------------------------------------

def test_tilted_log_density_lam_positive_raises_far_point():
    p = flat_density()
    X = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    x_refs = torch.zeros(5, 2, dtype=torch.float64)

    result = tilted_log_density(X, p, LpDistance(2.0), x_refs, lam=1.0)

    assert result.shape == (2,)
    assert result[1].item() > result[0].item(), (
        f"Far point should have higher tilted log density at lam>0; "
        f"got near={result[0].item()}, far={result[1].item()}"
    )


# ---------------------------------------------------------------------------
# 4. grid_normalize: shape, non-negativity, integral=1, Z>0
# ---------------------------------------------------------------------------

def test_grid_normalize_basic():
    gn = 10
    # Some arbitrary log-density on a 2D grid
    torch.manual_seed(42)
    log_q_unnorm = torch.randn(gn, gn, dtype=torch.float64)

    dx = 0.5
    cell_area = dx * dx

    log_q, q, Z = grid_normalize(log_q_unnorm, cell_area)

    # (a) shape preserved
    assert log_q.shape == (gn, gn), f"log_q shape mismatch: {log_q.shape}"
    assert q.shape == (gn, gn), f"q shape mismatch: {q.shape}"

    # (b) q >= 0 everywhere
    assert (q >= 0).all(), "q contains negative values"

    # (c) discrete integral ≈ 1
    integral = q.sum() * cell_area
    assert torch.allclose(integral, torch.tensor(1.0, dtype=torch.float64), atol=1e-6), (
        f"Integral of q should be ~1, got {integral.item()}"
    )

    # (d) Z is a positive scalar
    assert Z.ndim == 0, f"Z should be a scalar, got shape {Z.shape}"
    assert Z.item() > 0, f"Z should be positive, got {Z.item()}"


def test_grid_normalize_flat_input():
    """A flat (constant) log-density should normalize to a uniform distribution."""
    gn = 5
    log_q_unnorm = torch.zeros(gn, gn, dtype=torch.float64)
    cell_area = 1.0

    log_q, q, Z = grid_normalize(log_q_unnorm, cell_area)

    # All q values should be equal
    assert torch.allclose(q, q[0, 0].expand_as(q)), "Flat input should give uniform q"
    # Integral should be 1
    integral = q.sum() * cell_area
    assert torch.allclose(integral, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)


def test_grid_normalize_1d_input():
    """grid_normalize should work with 1D flat grids too."""
    G = 20
    log_q_unnorm = torch.linspace(-2.0, 2.0, G, dtype=torch.float64)
    cell_area = 0.2

    log_q, q, Z = grid_normalize(log_q_unnorm, cell_area)

    assert log_q.shape == (G,)
    assert q.shape == (G,)
    integral = q.sum() * cell_area
    assert torch.allclose(integral, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
    assert Z.item() > 0


# ---------------------------------------------------------------------------
# 5. weighted expected_distance / tilted_log_density
# ---------------------------------------------------------------------------

def test_expected_distance_weights_none_equals_mean():
    """weights=None must reproduce the plain mean over refs (backward compat)."""
    X = torch.tensor([[0.0, 0.0], [3.0, 1.0]], dtype=torch.float64)
    x_refs = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]], dtype=torch.float64)
    D = LpDistance(2.0)

    f_none = expected_distance(D, X, x_refs)
    f_mean = D.pairwise(X, x_refs).mean(dim=1)
    assert torch.allclose(f_none, f_mean)


def test_expected_distance_uniform_weights_equal_none():
    """Explicit uniform weights == weights=None."""
    X = torch.tensor([[0.0, 0.0], [3.0, 1.0]], dtype=torch.float64)
    x_refs = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]], dtype=torch.float64)
    D = LpDistance(2.0)
    R = x_refs.shape[0]
    w = torch.ones(R, dtype=torch.float64) / R

    assert torch.allclose(
        expected_distance(D, X, x_refs, weights=w),
        expected_distance(D, X, x_refs),
    )


def test_expected_distance_nonuniform_weights_manual():
    """Non-uniform weights match a hand-computed weighted mean."""
    X = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    # distances from origin: 1, 2, 5
    x_refs = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    D = LpDistance(2.0)
    w = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)

    expected = (0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 5.0) / (0.5 + 0.3 + 0.2)
    got = expected_distance(D, X, x_refs, weights=w)
    assert got.shape == (1,)
    assert got.item() == pytest.approx(expected, abs=1e-9)


def test_tilted_log_density_threads_weights():
    """tilted_log_density forwards weights to expected_distance."""
    p = flat_density()
    X = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    x_refs = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0], [3.0, 3.0]], dtype=torch.float64
    )
    D = LpDistance(2.0)
    w = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

    got = tilted_log_density(X, p, D, x_refs, lam=2.0, weights=w)
    expected = p.log_p_X(X) + 2.0 * expected_distance(D, X, x_refs, weights=w)
    assert torch.allclose(got, expected)


# ---------------------------------------------------------------------------
# 6. module-level reducer agrees with the selector's own reduction
# ---------------------------------------------------------------------------

def _ring_gmm_density() -> Density:
    """Small ring GMM (mirrors tests/test_demo_pipeline.py) for selector tests."""
    dtype = torch.float64
    n_total, radius, sigma = 8, 4.0, 0.4
    angles = 2 * math.pi * torch.arange(n_total, dtype=dtype) / n_total
    means = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)
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

    def sample(n):
        comp = torch.randint(K, (n,))
        return means[comp] + sigma * torch.randn(n, 2, dtype=dtype)

    return Density(log_pX, log_pY, sample_fn=sample, d=2)


def test_expected_distance_matches_randomrefs_selector():
    """Uniform selector: module reducer == selector reducer; weights is None."""
    torch.manual_seed(0)
    p = _ring_gmm_density()
    D = LpDistance(2.0)
    sel = RandomRefs(p, distance=D, seed=0)
    refs = sel.select(8)
    X = p.sample(16)

    f_module = expected_distance(D, X, refs, weights=sel.weights)
    f_sel = sel.expected_distance(X)
    assert sel.weights is None                      # uniform selector -> uniform reduction
    assert torch.allclose(f_module, f_sel)


def test_expected_distance_matches_weightedfps_selector():
    """Weighted selector: passing selector.weights reproduces the selector's weighted f."""
    torch.manual_seed(0)
    p = _ring_gmm_density()
    D = LpDistance(2.0)
    sel = WeightedFPSRefs(
        p, distance=D, seed=0, pool_size=300, est_floor=300, points_per_cell=16
    )
    refs = sel.select(8)
    w = sel.weights
    assert w is not None and w.shape == (8,)        # Voronoi weights populated
    X = p.sample(16)

    f_module = expected_distance(D, X, refs, weights=w)
    f_sel = sel.expected_distance(X)
    assert torch.allclose(f_module, f_sel)
