"""Labeled contact sheet of m_sweep.py's decoded images, one row per m.

Reads m_sweep_results.pt and writes m_sweep_grid.png next to it. No GPU, no model access -- pure
post-processing, same pattern as notebooks/diamond_smc_step_regression/plot_step_grid.py.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
N_PER_ROW = 6


def main() -> int:
    d = torch.load(os.path.join(HERE, "m_sweep_results.pt"), map_location="cpu")
    rows = d["rows"]

    fig, axes = plt.subplots(
        len(rows), N_PER_ROW, figsize=(1.9 * N_PER_ROW, 1.9 * len(rows)), squeeze=False
    )
    for row_idx, r in enumerate(rows):
        imgs = r["imgs"][:N_PER_ROW]
        f_final = r["f_final"][:N_PER_ROW]
        ok = (r["min_ess_m"] >= 0.5) and (r["uniq_m"] >= 0.5)
        for col in range(N_PER_ROW):
            ax = axes[row_idx][col]
            ax.imshow(imgs[col].clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"f={float(f_final[col]):.3f}", fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
        tag = "OK" if ok else "DEGENERATE"
        axes[row_idx][0].set_ylabel(
            f"m={r['m']:.2f}\nλ={r['lam']:.1f}\nE_q[f]={r['f_mean']:.4f}\n{tag}",
            fontsize=9, rotation=0, ha="right", va="center",
        )

    fig.suptitle(
        f"Diamond Maps SMC · label=207 cfg_scale=1.0 · M={d['M']} K={d['K']} N=6 · "
        f"fine m sweep, images unsorted (particle order == ancestor order)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.09, 0, 1, 0.97))
    out_path = os.path.join(HERE, "m_sweep_grid.png")
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
