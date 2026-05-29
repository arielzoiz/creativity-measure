"""Tests for creativity_measure/tilt.py"""
import torch
import pytest

from creativity_measure.density import Density
from creativity_measure.distances.base import EuclideanDistance
from creativity_measure.tilt import expected_distance, tilted_log_density, grid_normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flat_density():
    """A Density whose log_p_X is identically 0 (uniform / unnormalized)."""
    return Density(
        log_p_X=lambda x: torch.zeros(x.shape[0], dtype=x.dtype),
        log_p_Y=lambda y, g: None,
    )


# ---------------------------------------------------------------------------
# 1. expected_distance: shape and ordering
# ---------------------------------------------------------------------------

def test_expected_distance_shape_and_ordering():
    X = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    x_refs = torch.zeros(5, 2, dtype=torch.float64)

    result = expected_distance(EuclideanDistance(), X, x_refs)

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

    result = tilted_log_density(X, p, EuclideanDistance(), x_refs, lam=0.0)

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

    result = tilted_log_density(X, p, EuclideanDistance(), x_refs, lam=1.0)

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
