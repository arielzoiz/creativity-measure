# creativity_measure/plotting.py
import torch
import numpy as np
import matplotlib.pyplot as plt


def make_grid(xlim, ylim, grid_n=50, device="cpu", dtype=torch.float64):
    """Returns (grid_points (G,2), XX, YY, cell_area)."""
    xs = torch.linspace(xlim[0], xlim[1], grid_n, device=device, dtype=dtype)
    ys = torch.linspace(ylim[0], ylim[1], grid_n, device=device, dtype=dtype)
    XX, YY = torch.meshgrid(xs, ys, indexing='xy')
    grid_points = torch.stack([XX, YY], dim=-1).reshape(-1, 2)
    dx = (xlim[1] - xlim[0]) / (grid_n - 1)
    dy = (ylim[1] - ylim[0]) / (grid_n - 1)
    return grid_points, XX, YY, float(dx * dy)


def plot_field(vals, XX, YY, ax=None, title="", cmap="viridis", levels=20,
               marked=None, missing=None):
    """Filled contour of a scalar field on a meshgrid."""
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
    return ax


def plot_samples(samples, ax=None, title="", alpha=0.3, s=5, color='steelblue'):
    if ax is None:
        _, ax = plt.subplots()
    pts = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    ax.scatter(pts[:, 0], pts[:, 1], alpha=alpha, s=s, color=color)
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax
