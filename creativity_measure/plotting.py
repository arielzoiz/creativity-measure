from collections.abc import Sequence

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from creativity_measure.device import default_device
from creativity_measure.tilt import grid_normalize


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


def lambda_sweep(
    log_p_grid: torch.Tensor,
    fields: "torch.Tensor | Sequence[torch.Tensor]",
    cell_area: float,
    XX: torch.Tensor,
    YY: torch.Tensor,
    *,
    lambdas: "float | Sequence[float] | None" = None,
    lambda0: "float | Sequence[float] | None" = None,
    multipliers: "float | Sequence[float] | None" = None,
    field_labels: "Sequence[str] | None" = None,
    row_labels: "Sequence[str] | None" = None,
    panel_titles: "Sequence[str] | None" = None,
    suptitle: "str | None" = None,
    suptitle_y: "float | None" = None,
    ncols: int = 2,
    normalize: bool = True,
    marked=None,
    missing=None,
    refs=None,
    cmap: str = "viridis",
    levels: int = 20,
    panel_w: float = 5.0,
    panel_h: float = 4.0,
) -> tuple[Figure, list[Axes]]:
    """Grid of tilted log-densities  q_λ ∝ p · exp(coef · f)  on a 2D grid.

    Two axes: the λ axis (the rows) and the tilt fields `fields` (the "variant"
    axis -- e.g. different distances, γ-windows, ref counts). Each panel shows
    grid_normalize(log_p_grid + coef * f)  (or the raw landscape if
    normalize=False).

    The λ axis is given in EXACTLY ONE of two mutually-exclusive ways:
      1. absolute -- `lambdas` (length L):              coef(l, c) = lambdas[l]
      2. relative -- `lambda0` (per-field, length F)
         together with `multipliers` (length L):        coef(l, c) =
                                                         multipliers[l]·lambda0[c]
    Passing both, neither, or only half of (2) raises ValueError.

    Layout:
      * both axes length > 1  -> matrix, rows = λ (ylabel), cols = field (title).
      * one axis length 1     -> flat gallery of ncols columns (via panel_grid).

    fields       -- a single field tensor, or a list of F field tensors.
    lambdas      -- mode 1: absolute λ values (the row axis).
    lambda0      -- mode 2: per-field base values (len F).
    multipliers  -- mode 2: row-axis multipliers applied to every lambda0.
    field_labels -- column labels (len F) for the variant axis.
    row_labels   -- override the λ-axis labels in matrix mode.
    panel_titles -- override per-panel titles in gallery mode.
    suptitle_y   -- y position for the suptitle (e.g. 1.03 to lift it clear of
                    the top row); None uses matplotlib's default placement.
    marked/missing/refs -- overlays forwarded to every panel (see plot_field).

    Returns (fig, axes); the caller is responsible for plt.show().
    """
    def _as_list(v) -> list[float]:
        return [float(v)] if isinstance(v, (int, float)) else [float(x) for x in v]

    field_list = [fields] if isinstance(fields, torch.Tensor) else list(fields)
    F = len(field_list)

    absolute = lambdas is not None
    relative = lambda0 is not None or multipliers is not None
    if absolute and relative:
        raise ValueError("lambda_sweep: pass EITHER `lambdas` (absolute) OR "
                         "`lambda0`+`multipliers` (relative), not both.")
    if not absolute and not relative:
        raise ValueError("lambda_sweep: pass `lambdas` (absolute) OR "
                         "`lambda0`+`multipliers` (relative).")
    if relative and (lambda0 is None or multipliers is None):
        raise ValueError("lambda_sweep: relative mode needs BOTH `lambda0` "
                         "(per-field) and `multipliers`.")

    if absolute:
        lam_list = _as_list(lambdas)        # row axis = absolute λ
        l0 = [1.0] * F
        has_l0 = False
    else:
        lam_list = _as_list(multipliers)    # row axis = multipliers
        l0 = _as_list(lambda0)
        if len(l0) == 1:
            l0 = l0 * F
        if len(l0) != F:
            raise ValueError(f"lambda_sweep: lambda0 has {len(l0)} values "
                             f"but there are {F} fields.")
        has_l0 = True
    L = len(lam_list)

    def cell_field(l_idx: int, c_idx: int) -> torch.Tensor:
        coef = lam_list[l_idx] * l0[c_idx]
        vals = log_p_grid + coef * field_list[c_idx]
        return grid_normalize(vals, cell_area)[0] if normalize else vals

    if L > 1 and F > 1:
        fig, axs = plt.subplots(L, F, squeeze=False,
                                figsize=(panel_w * F, panel_h * L))
        axes: list[Axes] = []
        for l_idx in range(L):
            for c_idx in range(F):
                ax = axs[l_idx][c_idx]
                title = (field_labels[c_idx]
                         if field_labels is not None and l_idx == 0 else "")
                plot_field(cell_field(l_idx, c_idx), XX, YY, ax=ax, title=title,
                           cmap=cmap, levels=levels, marked=marked,
                           missing=missing, refs=refs)
                if c_idx == 0:
                    if row_labels is not None:
                        ax.set_ylabel(row_labels[l_idx], fontsize=12)
                    else:
                        suffix = "·λ₀" if has_l0 else ""
                        ax.set_ylabel(f"λ={lam_list[l_idx]:g}{suffix}", fontsize=12)
                axes.append(ax)
    else:
        fig, axes = panel_grid(L * F, ncols=ncols, panel_w=panel_w, panel_h=panel_h)
        for idx, ax in enumerate(axes):
            l_idx, c_idx = (idx, 0) if F == 1 else (0, idx)
            if panel_titles is not None:
                title = panel_titles[idx]
            elif F == 1:
                title = f"λ={lam_list[l_idx]:g}"
            else:
                lbl = field_labels[c_idx] if field_labels is not None else ""
                title = f"{lbl}  λ={lam_list[0]:g}".strip()
            plot_field(cell_field(l_idx, c_idx), XX, YY, ax=ax, title=title,
                       cmap=cmap, levels=levels, marked=marked,
                       missing=missing, refs=refs)

    if suptitle is not None:
        if suptitle_y is None:
            fig.suptitle(suptitle)
        else:
            fig.suptitle(suptitle, y=suptitle_y)
    fig.tight_layout()
    return fig, axes


def plot_samples(samples, ax=None, title="", alpha=0.3, s=5, color='steelblue'):
    if ax is None:
        _, ax = plt.subplots()
    pts = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    ax.scatter(pts[:, 0], pts[:, 1], alpha=alpha, s=s, color=color)
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax
