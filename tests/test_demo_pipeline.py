"""
End-to-end pipeline test for the IEM creativity-tilt demo.

Uses a headless Agg backend and a small/coarse config for speed.
Covers: LocalIEMDistance, GlobalIEMDistance (f_identity & f_square),
EuclideanDistance, tilted_log_density, grid_normalize, plot_field.
"""

import matplotlib
matplotlib.use("Agg")

import math
import torch
import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from creativity_measure import (
    Density,
    EuclideanDistance,
    LocalIEMDistance,
    GlobalIEMDistance,
    f_identity,
    f_square,
    expected_distance,
    tilted_log_density,
    grid_normalize,
    make_grid,
    plot_field,
)

# ---------------------------------------------------------------------------
# Ring GMM with hole — pure tensor ops (vmap-safe, no torch.distributions)
# ---------------------------------------------------------------------------

dtype = torch.float64
N_TOTAL, HOLE_IDX, RADIUS, SIGMA = 12, 0, 4.0, 0.3
_angles = 2 * math.pi * torch.arange(N_TOTAL, dtype=dtype) / N_TOTAL
all_means = torch.stack(
    [RADIUS * torch.cos(_angles), RADIUS * torch.sin(_angles)], dim=-1
)
_mask = torch.ones(N_TOTAL, dtype=torch.bool)
_mask[HOLE_IDX] = False
MEANS = all_means[_mask]   # (K, 2)  observed modes (hole removed)
K = MEANS.shape[0]
S2 = SIGMA ** 2
_logK = math.log(K)


def _log_pX(x):
    """Equal-weight isotropic GMM, pure tensor ops (vmap-safe)."""
    diff = x.unsqueeze(-2) - MEANS          # (..., K, 2)
    quad = diff.pow(2).sum(-1) / S2         # (..., K)
    logcomp = -0.5 * (quad + 2 * math.log(2 * math.pi * S2))
    return torch.logsumexp(logcomp, dim=-1) - _logK


def _log_pY(y, gamma):
    """Y = gamma*X + sqrt(gamma)*W; comp k ~ N(gamma*mu_k, (gamma^2*S2 + gamma)*I)."""
    var = gamma ** 2 * S2 + gamma
    diff = y.unsqueeze(-2) - gamma * MEANS  # (..., K, 2)
    quad = diff.pow(2).sum(-1) / var
    logcomp = -0.5 * (quad + 2 * torch.log(2 * math.pi * var))
    return torch.logsumexp(logcomp, dim=-1) - _logK


def _ring_sample(n):
    comp = torch.randint(K, (n,))
    return MEANS[comp] + SIGMA * torch.randn(n, 2, dtype=dtype)


# Shared Density instance for all tests in this module
_p = Density(_log_pX, _log_pY, sample=_ring_sample, d=2)

# ---------------------------------------------------------------------------
# Small/coarse shared config
# ---------------------------------------------------------------------------
GRID_N = 12                      # 12×12 = 144 grid points
LAM = 5.0
GAMMAS_LOCAL = torch.logspace(-4, 4, 20, base=2, dtype=dtype)
GAMMAS_GLOBAL = torch.logspace(-10, 10, 20, base=2, dtype=dtype)


@pytest.fixture(scope="module")
def grid():
    return make_grid((-6, 6), (-6, 6), grid_n=GRID_N, dtype=dtype)


@pytest.fixture(scope="module")
def x_refs():
    torch.manual_seed(0)
    return _p.sample(6)


# ---------------------------------------------------------------------------
# 1. Local IEM pipeline
# ---------------------------------------------------------------------------

def test_local_iem_pipeline(grid, x_refs):
    grid_points, XX, YY, cell_area = grid

    D = LocalIEMDistance(_p, GAMMAS_LOCAL, num_noises=4)
    log_q_un = tilted_log_density(grid_points, _p, D, x_refs, lam=LAM)

    # shape
    assert log_q_un.shape == (GRID_N * GRID_N,), (
        f"Expected shape ({GRID_N * GRID_N},), got {log_q_un.shape}"
    )
    # all finite
    assert log_q_un.isfinite().all(), "log_q_un has non-finite values (local IEM)"

    # normalize & check integral
    log_q, q, Z = grid_normalize(log_q_un, cell_area)
    integral = q.sum() * cell_area
    assert torch.allclose(
        integral, torch.tensor(1.0, dtype=dtype), atol=1e-6
    ), f"Integral should be ~1, got {integral.item()}"
    assert (q >= 0).all(), "q has negative values (local IEM)"


# ---------------------------------------------------------------------------
# 2. Global IEM pipeline (f_identity)
# ---------------------------------------------------------------------------

def test_global_iem_f_identity_pipeline(grid, x_refs):
    grid_points, XX, YY, cell_area = grid

    D = GlobalIEMDistance(_p, GAMMAS_GLOBAL, num_eps=4, f=f_identity)
    log_q_un = tilted_log_density(grid_points, _p, D, x_refs, lam=LAM)

    assert log_q_un.shape == (GRID_N * GRID_N,)
    assert log_q_un.isfinite().all(), "log_q_un has non-finite values (global IEM f_identity)"

    log_q, q, Z = grid_normalize(log_q_un, cell_area)
    integral = q.sum() * cell_area
    assert torch.allclose(
        integral, torch.tensor(1.0, dtype=dtype), atol=1e-6
    ), f"Integral should be ~1, got {integral.item()}"
    assert (q >= 0).all(), "q has negative values (global IEM f_identity)"


# ---------------------------------------------------------------------------
# 3. Global IEM (f_square) runs and differs from f_identity
# ---------------------------------------------------------------------------

def test_global_iem_f_square_differs_from_f_identity(grid, x_refs):
    grid_points, XX, YY, cell_area = grid

    D_id = GlobalIEMDistance(_p, GAMMAS_GLOBAL, num_eps=4, f=f_identity, seed=42)
    score_id = expected_distance(D_id, grid_points, x_refs)

    D_sq = GlobalIEMDistance(_p, GAMMAS_GLOBAL, num_eps=4, f=f_square, seed=42)
    score_sq = expected_distance(D_sq, grid_points, x_refs)

    assert score_sq.isfinite().all(), "f_square scores contain non-finite values"
    assert not torch.allclose(score_id, score_sq), (
        "f_square expected_distance should differ from f_identity"
    )


# ---------------------------------------------------------------------------
# 4. Euclidean pipeline
# ---------------------------------------------------------------------------

def test_euclidean_pipeline(grid, x_refs):
    grid_points, XX, YY, cell_area = grid

    D = EuclideanDistance()
    log_q_un = tilted_log_density(grid_points, _p, D, x_refs, lam=LAM)

    assert log_q_un.shape == (GRID_N * GRID_N,)
    assert log_q_un.isfinite().all(), "log_q_un has non-finite values (Euclidean)"

    log_q, q, Z = grid_normalize(log_q_un, cell_area)
    integral = q.sum() * cell_area
    assert torch.allclose(
        integral, torch.tensor(1.0, dtype=dtype), atol=1e-6
    ), f"Integral should be ~1, got {integral.item()}"
    assert (q >= 0).all(), "q has negative values (Euclidean)"


# ---------------------------------------------------------------------------
# 5. plot_field smoke test
# ---------------------------------------------------------------------------

def test_plot_field_smoke(grid, x_refs):
    grid_points, XX, YY, cell_area = grid

    # Use a simple log_pX for the field (fast, no distance needed)
    log_q_un = _p.log_p_X(grid_points)
    log_q, q, Z = grid_normalize(log_q_un, cell_area)

    missing = all_means[~_mask]  # the hole mode
    ax = plot_field(log_q, XX, YY, marked=MEANS, missing=missing)

    assert ax is not None
    assert isinstance(ax, Axes)
    plt.close("all")
