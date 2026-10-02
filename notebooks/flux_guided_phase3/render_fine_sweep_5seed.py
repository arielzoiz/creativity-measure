"""Builds the 5-seed replication grid for the max5.5 fine lambda sweep (notebooks/flux_guided_phase3/).

Row 0 reuses the existing max5.5 run (fine_lambda_sweep_results_max5.5.json / fine_decoded_max5.5/),
columns idx00-idx10 (lam 0 -> 3.9286, the creative-but-recognizable window found there). Rows 1-4 repeat
the SAME 11 lambda points, SAME model/prompt/reward config, with a different SWEEP_SEED (the initial
noise z0) each -- produced by fine_lambda_sweep_seed.slurm (one job per seed). No GPU needed here; only
reads what's already on disk, so it can be re-run any time, including before all seed jobs finish (rows
for missing results are left blank with a note).

    python render_fine_sweep_5seed.py
"""
from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

N_COLS = 11  # idx00..idx10, lam 0 -> 3.9286, matching columns 1-11 of fine_sweep_grid_max5.5.png
ROWS = [
    ("max5.5 (orig)", "fine_lambda_sweep_results_max5.5.json", "fine_decoded_max5.5"),
    ("seed2024", "fine_lambda_sweep_results_seed2024.json", "fine_decoded_seed2024"),
    ("seed3141", "fine_lambda_sweep_results_seed3141.json", "fine_decoded_seed3141"),
    ("seed4242", "fine_lambda_sweep_results_seed4242.json", "fine_decoded_seed4242"),
    ("seed5555", "fine_lambda_sweep_results_seed5555.json", "fine_decoded_seed5555"),
]

THUMB = 184
LABEL_H = 34
PAD = 2
OUT = os.path.join(HERE, "fine_sweep_grid_5seed.png")


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

        for col in range(N_COLS):
            x0 = col * cell_w
            y0 = row_idx * cell_h
            cell_text = None
            img = None
            if res is not None:
                key = next((k for k in res if k != "stamp" and res[k]["idx"] == col), None)
                if key is not None:
                    r = res[key]
                    png_path = os.path.join(HERE, decoded_dir, os.path.basename(r["png"])) if r.get("png") else None
                    if png_path and os.path.exists(png_path):
                        img = Image.open(png_path).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
                    cell_text = f"lam={r['lam']:.2f} f={r['f_mean']:.1f}\nt={r['t_guidance_s']:.0f}s"
            if img is not None:
                grid.paste(img, (x0 + PAD, y0 + PAD))
            else:
                draw.rectangle([x0 + PAD, y0 + PAD, x0 + PAD + THUMB, y0 + PAD + THUMB], outline="gray")
                draw.text((x0 + PAD + 8, y0 + PAD + THUMB // 2 - 6), "missing", fill="gray", font=font)
            label_y = y0 + PAD + THUMB + 3
            if cell_text is not None:
                draw.text((x0 + PAD + 2, label_y), cell_text, fill="black", font=font)
            if col == 0:
                draw.text((x0 + PAD + 2, y0 + 1), row_label, fill="#0066cc", font=font)

    grid.save(OUT)
    print(f"wrote {OUT} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
