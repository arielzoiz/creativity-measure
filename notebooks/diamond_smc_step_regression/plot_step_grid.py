"""Labeled contact sheet of step_trace.py's final decoded images, one row per N.

Reads step_trace_results.pt (produced by step_trace.py / job 905506) and writes
step_regression_grid.png next to it. No GPU, no model access -- pure post-processing.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
N_PER_ROW = 4


def main() -> int:
    d = torch.load(os.path.join(HERE, "step_trace_results.pt"), map_location="cpu")
    n_grid = d["N_GRID"]

    fig, axes = plt.subplots(
        len(n_grid), N_PER_ROW, figsize=(2.2 * N_PER_ROW, 2.2 * len(n_grid)),
        squeeze=False,
    )
    for row, n in enumerate(n_grid):
        imgs = d["final_imgs"][n][:N_PER_ROW]
        for col in range(N_PER_ROW):
            ax = axes[row][col]
            ax.imshow(imgs[col].clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row][0].set_ylabel(f"N={n}", fontsize=13, rotation=0, ha="right", va="center")

    fig.suptitle(
        f"Diamond Maps base sampler · label={d['label']} cfg_scale={d['cfg_scale']} "
        f"inner={d['inner']} · decoded final images, sharpness falls as N rises",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    out_path = os.path.join(HERE, "step_regression_grid.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
