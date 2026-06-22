import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from creativity_measure.device import default_device


def make_grid(xlim, ylim, grid_n=50, device=None, dtype=torch.float64):
    """Returns (grid_points (G,2), XX, YY, cell_area). device=None => default_device()."""
    if device is None:
        device = default_device()
    xs = torch.linspace(xlim[0], xlim[1], grid_n, device=device, dtype=dtype)
    ys = torch.linspace(ylim[0], ylim[1], grid_n, device=device, dtype=dtype)
    XX, YY = torch.meshgrid(xs, ys, indexing='xy')
    grid_points = torch.stack([XX, YY], dim=-1).reshape(-1, 2)
    dx = (xlim[1] - xlim[0]) / (grid_n - 1)
    dy = (ylim[1] - ylim[0]) / (grid_n - 1)
    return grid_points, XX, YY, float(dx * dy)


def plot_field(vals, XX, YY, ax=None, title="", cmap="viridis", levels=20,
               marked=None, missing=None, refs=None):
    """Filled contour of a scalar field on a meshgrid.

    marked  -- points drawn as orange x   (e.g. observed component means)
    missing -- points drawn as red stars  (e.g. removed/held-out modes)
    refs    -- points drawn as black dots  (e.g. chosen reference points);
               adds a legend entry labelled with the number of refs.
    """
    if ax is None:
        _, ax = plt.subplots()
    Z = vals.reshape(XX.shape)
    Z = Z.cpu().numpy() if isinstance(Z, torch.Tensor) else Z
    XXn = XX.cpu().numpy() if isinstance(XX, torch.Tensor) else XX
    YYn = YY.cpu().numpy() if isinstance(YY, torch.Tensor) else YY
    cf = ax.contourf(XXn, YYn, Z, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_aspect('equal')
    if marked is not None:
        m = marked.cpu().numpy() if isinstance(marked, torch.Tensor) else marked
        ax.scatter(m[:, 0], m[:, 1], marker='x', c='orange', s=25)
    if missing is not None:
        mm = missing.cpu().numpy() if isinstance(missing, torch.Tensor) else missing
        ax.scatter(mm[:, 0], mm[:, 1], marker='*', c='red', s=120,
                   edgecolors='white', linewidths=0.5, zorder=5)
    if refs is not None:
        r = refs.cpu().numpy() if isinstance(refs, torch.Tensor) else refs
        ax.scatter(r[:, 0], r[:, 1], s=12, c='k', edgecolors='white',
                   linewidths=0.3, zorder=6, label=f"refs (R={r.shape[0]})")
        ax.legend(loc='upper right', fontsize=8, framealpha=0.75)
    return ax


def panel_grid(n: int, ncols: int = 2, panel_w: float = 6.0,
               panel_h: float = 4.0) -> tuple[Figure, list[Axes]]:
    """Create a flattened grid for n panels (ncols columns, rows auto).

    Returns (fig, axes) where axes is a list of exactly n visible axes; any
    leftover cells in the final row are created but hidden. The caller is
    responsible for suptitle / tight_layout / show.
    """
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows))
    flat: list[Axes] = list(np.asarray(axs).reshape(-1))
    for ax in flat[n:]:
        ax.axis('off')
    return fig, flat[:n]


def plot_samples(samples, ax=None, title="", alpha=0.3, s=5, color='steelblue'):
    if ax is None:
        _, ax = plt.subplots()
    pts = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    ax.scatter(pts[:, 0], pts[:, 1], alpha=alpha, s=s, color=color)
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax
