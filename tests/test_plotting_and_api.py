import matplotlib
matplotlib.use("Agg")

import torch
import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from creativity_measure.plotting import make_grid, plot_field, plot_samples


# ---------------------------------------------------------------------------
# 1. make_grid
# ---------------------------------------------------------------------------

def test_make_grid_shapes():
    grid_points, XX, YY, cell_area = make_grid((-6, 6), (-6, 6), grid_n=10)
    assert grid_points.shape == (100, 2)
    assert XX.shape == (10, 10)
    assert YY.shape == (10, 10)


def test_make_grid_cell_area():
    _, _, _, cell_area = make_grid((-6, 6), (-6, 6), grid_n=10)
    expected = (12 / 9) ** 2
    assert isinstance(cell_area, float)
    assert abs(cell_area - expected) < 1e-10


# ---------------------------------------------------------------------------
# 2. plot_field smoke
# ---------------------------------------------------------------------------

def test_plot_field_returns_axes():
    _, XX, YY, _ = make_grid((-6, 6), (-6, 6), grid_n=10)
    vals = torch.randn(100, dtype=torch.float64)
    ax = plot_field(vals, XX, YY, title="t",
                    marked=torch.zeros(2, 2, dtype=torch.float64),
                    missing=torch.ones(1, 2, dtype=torch.float64))
    assert ax is not None
    assert isinstance(ax, Axes)
    plt.close('all')


# ---------------------------------------------------------------------------
# 3. plot_samples smoke
# ---------------------------------------------------------------------------

def test_plot_samples_returns_axes():
    ax = plot_samples(torch.randn(20, 2, dtype=torch.float64))
    assert ax is not None
    assert isinstance(ax, Axes)
    plt.close('all')


# ---------------------------------------------------------------------------
# 4. public API imports
# ---------------------------------------------------------------------------

EXPECTED_NAMES = [
    "Density",
    "Distance",
    "LpDistance",
    "LocalIEMDistance",
    "GlobalIEMDistance",
    "GeneralizedGlobalIEMDistance",
    "expected_distance",
    "tilted_log_density",
    "grid_normalize",
    "Reward",
    "make_grid",
    "plot_field",
    "plot_samples",
    "adaptive_tempering_smc_sample",
    "AdaptiveTemperingSMCResult",
    "Kernel",
    "IndependenceKernel",
    "PCNKernel",
    "heun_prob_flow",
    "density_denoiser",
    "density_generator",
    "edm_generator",
    "eps_to_edm_denoiser",
    "build_edm_pixel_generator",
    "build_edm_pixel_generator_from_pkl",
    "build_tiny_sd_generator",
]


def test_public_api():
    import creativity_measure as cm
    for name in EXPECTED_NAMES:
        assert hasattr(cm, name), f"creativity_measure is missing attribute: {name}"
