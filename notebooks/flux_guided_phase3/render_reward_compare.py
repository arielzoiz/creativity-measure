"""Builds the Brownian-vs-i.i.d. reward comparison grid for the max5.5 fine lambda sweep.

Two rows, SAME 15 lambda points, SAME z0 (sweep seed 1234), SAME ODE schedule (n_steps=10, shift=3.0,
exact_jacobian=True) -- the ONLY difference is which estimator of D_IEM^2 the reward uses:

    row 0  brownian  ExpectedSquaredGlobalIEMDistance, grid N_gamma=11 x N_eps=5  (K=50)
    row 1  iid       SquaredIIDGlobalIEMDistance,      draws G=50   x N_eps=1     (K=50)

K is matched on purpose, so score rows and reference-bank cost are identical and only the estimator
differs. The lambda=0 column is a free correctness check: no backward is taken there, so the two
images must be byte-identical (verified for this pair -- max abs pixel diff 0).

WHY A WHOLE-ROW VIEW AND NOT FRAME-BY-FRAME. Recognizability is NOT monotonic in lambda on this path
-- seed 1234's brownian run is a clear dog at lam=0.39, a blob at lam=0.79, and a clear cartoon dog
again at lam=1.18. Three consecutive single-frame reads each flipped on the next point while this
sweep was running. Judge the window from the row, never from one cell.

READ THE IMAGES, NOT THE NUMBERS. The per-cell f is printed only for provenance. Neither f nor
||x||/sqrt(d) nor the high-frequency fraction predicts recognizability (CLAUDE.md), and ||x|| is
actively ANTI-correlated -- f here keeps climbing monotonically straight through cells that recover.

No GPU, reads only what is on disk, so it can be re-run any time including mid-sweep (missing cells
are drawn as empty boxes).

    python render_reward_compare.py
"""
from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

N_COLS = 15  # idx00..idx14, lam 0 -> 5.5
ROWS = [
    ("brownian (N_gamma=11, N_eps=5)", "fine_lambda_sweep_results_brownian_max5.5.json",
     "fine_decoded_brownian_max5.5"),
    ("iid (G=50, N_eps=1)", "fine_lambda_sweep_results_max5.5.json", "fine_decoded_max5.5"),
]

THUMB = 184
LABEL_H = 34
PAD = 2
OUT = os.path.join(HERE, "reward_compare_grid.png")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    font = _font(12)
    cell_w = THUMB + 2 * PAD
    cell_h = THUMB + LABEL_H + 2 * PAD
    grid = Image.new("RGB", (cell_w * N_COLS, cell_h * len(ROWS)), "white")
    draw = ImageDraw.Draw(grid)

    for row_idx, (row_label, results_name, decoded_dir) in enumerate(ROWS):
        results_path = os.path.join(HERE, results_name)
        res = None
        if os.path.exists(results_path):
            with open(results_path) as fh:
                res = json.load(fh)
        lam_s = res["stamp"].get("lambda_s") if res else None

        for col in range(N_COLS):
            x0 = col * cell_w
            y0 = row_idx * cell_h
            cell_text = None
            img = None
            if res is not None:
                key = next((k for k in res if k != "stamp" and res[k]["idx"] == col), None)
                if key is not None:
                    r = res[key]
                    png_path = (os.path.join(HERE, decoded_dir, os.path.basename(r["png"]))
                                if r.get("png") else None)
                    if png_path and os.path.exists(png_path):
                        img = Image.open(png_path).convert("RGB").resize(
                            (THUMB, THUMB), Image.Resampling.LANCZOS)
                    m = (r["lam"] / lam_s) if lam_s else float("nan")
                    cell_text = f"lam={r['lam']:.2f} m={m:.4f}\nf={r['f_mean']:.2f} t={r['t_guidance_s']:.0f}s"
            if img is not None:
                grid.paste(img, (x0 + PAD, y0 + PAD))
            else:
                draw.rectangle([x0 + PAD, y0 + PAD, x0 + PAD + THUMB, y0 + PAD + THUMB], outline="gray")
                draw.text((x0 + PAD + 8, y0 + PAD + THUMB // 2 - 6), "pending", fill="gray", font=font)
            label_y = y0 + PAD + THUMB + 3
            if cell_text is not None:
                draw.text((x0 + PAD + 2, label_y), cell_text, fill="black", font=font)
            if col == 0:
                draw.text((x0 + PAD + 2, y0 + 1), row_label, fill="#0066cc", font=font)

    grid.save(OUT)
    print(f"wrote {OUT} ({grid.size[0]}x{grid.size[1]})")
    if ROWS and os.path.exists(os.path.join(HERE, ROWS[0][1])):
        with open(os.path.join(HERE, ROWS[0][1])) as fh:
            done = len([k for k in json.load(fh) if k != "stamp"])
        print(f"brownian row: {done}/{N_COLS} cells populated")


if __name__ == "__main__":
    main()
