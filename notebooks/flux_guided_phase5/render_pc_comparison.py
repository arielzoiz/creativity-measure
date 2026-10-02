"""Builds the Phase 5 three-arm comparison: one image grid plus the numeric table.

Rows are the three arms (pc_guided / pc_unguided / flow_guided), columns are the 8 lambda points of
LAM_GRID -- every other point of Phase 3's zoom grid, so a fourth row reads Phase 3's OWN stored
max5.5 images at the same lambdas as an independent cross-check on the re-run flow_guided arm.

No GPU. Only reads what is already on disk, so it can be re-run any time, including before all three
jobs finish (cells with no result are left blank, like render_fine_sweep_5seed.py's).

HOW TO READ IT (the same warning pc_sweep.py's docstring carries, repeated here because this is the
artifact people will actually look at): the hypothesis is NOT that the PC arms show higher f. The
corrector pulls back toward p_t, so at matched lambda they should report LOWER f than flow_guided. The
claim under test is that the RECOGNIZABLE window extends to larger lambda -- so compare the images at
lam >= 3.93, where Phase 3 is known to lock onto a fixed off-manifold attractor, and read `hf` (fraction
of spectral power above 0.25 Nyquist) and `|x|` (latent norm over sqrt(d)) as the quantitative proxies
for that. f is meaningful only within the still-recognizable band.

CAVEAT ON `hf`, measured here on Phase 3's own stored images rather than assumed: it is NOT monotone in
lambda. Recomputed over fine_decoded_max5.5, it runs 0.0546 (lam=0, the untilted photo-like dog) -> 0.0213
(lam=1.57, a smooth restyle) -> 0.0417 (lam=5.5, collapsed) -- it dips before it rises, because a flat
ink-sketch or cartoon reinterpretation genuinely has LESS high-frequency content than a photograph. So
`hf` is usable as a comparison ACROSS ARMS AT MATCHED LAMBDA, and not as an absolute "is this broken"
score. `|x|` (available only for runs this phase produced; Phase 3 never recorded it) is the cleaner
one-directional signal.

    python render_pc_comparison.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pc_sweep import hf_power_fraction                                                     # noqa: E402

PHASE3 = os.path.join(HERE, "..", "flux_guided_phase3")

# (row label, results json, decoded dir, base dir)
ROWS = [
    ("pc_guided (corr=2)", "pc_sweep_results_pc_guided.json", "pc_decoded_pc_guided", HERE),
    ("pc_unguided (corr=2)", "pc_sweep_results_pc_unguided.json", "pc_decoded_pc_unguided", HERE),
    ("flow_guided (corr=0, re-run)", "pc_sweep_results_flow_guided.json", "pc_decoded_flow_guided", HERE),
    ("phase 3 max5.5 (stored)", "fine_lambda_sweep_results_max5.5.json", "fine_decoded_max5.5", PHASE3),
]

LAM_GRID = [k * 5.5 / 14 for k in range(0, 15, 2)]
THUMB = 184
LABEL_H = 44
PAD = 2
OUT = os.path.join(HERE, "pc_comparison_grid.png")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _load(base: str, name: str) -> dict | None:
    path = os.path.join(base, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _find(res: dict, lam: float) -> dict | None:
    """Match by lambda VALUE, not by idx: Phase 3's max5.5 run has 15 points to our 8, so idx differs."""
    for k, r in res.items():
        if k == "stamp":
            continue
        if abs(r["lam"] - lam) < 1e-6:
            return r
    return None


_HF_CACHE: dict[str, float] = {}


def _hf_frac(r: dict, png: str | None) -> float:
    """``hf_frac`` from the results JSON, or recomputed from the PNG if the run never recorded it.

    Phase 3's results predate this metric, so its stored row would otherwise show nan and the one column
    that is supposed to say "is it deep-fried" would be blank exactly where the comparison matters. The
    PNG is the decoded image in [0,1], which is all `hf_power_fraction` needs, so the stored row can be
    brought onto the same footing without re-running anything on a GPU.
    """
    val = r.get("hf_frac")
    if val is not None and val == val:          # present and not nan
        return float(val)
    if png is None or not os.path.exists(png):
        return float("nan")
    if png not in _HF_CACHE:
        rgb = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
        _HF_CACHE[png] = hf_power_fraction(torch.from_numpy(rgb).permute(2, 0, 1))
    return _HF_CACHE[png]


def main() -> None:
    font = _font(12)
    cell_w = THUMB + 2 * PAD
    cell_h = THUMB + LABEL_H + 2 * PAD
    grid = Image.new("RGB", (cell_w * len(LAM_GRID), cell_h * len(ROWS)), "white")
    draw = ImageDraw.Draw(grid)
    loaded = [(label, _load(base, name), os.path.join(base, ddir)) for label, name, ddir, base in ROWS]

    for row_idx, (row_label, res, decoded_dir) in enumerate(loaded):
        for col, lam in enumerate(LAM_GRID):
            x0, y0 = col * cell_w, row_idx * cell_h
            img, cell_text = None, None
            r = _find(res, lam) if res is not None else None
            if r is not None:
                png = os.path.join(decoded_dir, os.path.basename(r["png"])) if r.get("png") else None
                if png and os.path.exists(png):
                    img = Image.open(png).convert("RGB").resize(
                        (THUMB, THUMB), Image.Resampling.LANCZOS)
                hf = _hf_frac(r, png)
                xn = r.get("x_norm_final", float("nan"))
                cell_text = (f"lam={r['lam']:.2f} f={r['f_mean']:.1f}\n"
                             f"hf={hf:.3f} |x|={xn:.2f}\n"
                             f"disp={r.get('corrector_rel_disp_mean', float('nan')):.3g}")
            if img is not None:
                grid.paste(img, (x0 + PAD, y0 + PAD))
            else:
                draw.rectangle([x0 + PAD, y0 + PAD, x0 + PAD + THUMB, y0 + PAD + THUMB], outline="gray")
                draw.text((x0 + PAD + 8, y0 + PAD + THUMB // 2 - 6), "missing", fill="gray", font=font)
            if cell_text is not None:
                draw.text((x0 + PAD + 2, y0 + PAD + THUMB + 3), cell_text, fill="black", font=font)
            if col == 0:
                draw.text((x0 + PAD + 2, y0 + 1), row_label, fill="#0066cc", font=font)

    grid.save(OUT)
    print(f"wrote {OUT} ({grid.size[0]}x{grid.size[1]})")

    # --- the numeric table, and the cross-check the whole comparison rests on ---------------------
    print(f"\n{'lam':>7s} " + " ".join(f"{lab.split()[0]:>26s}" for lab, _, _ in loaded))
    print(f"{'':>7s} " + " ".join(f"{'f / hf / |x|':>26s}" for _ in loaded))
    for lam in LAM_GRID:
        cells = []
        for _, res, ddir in loaded:
            r = _find(res, lam) if res is not None else None
            if r is None:
                cells.append("%26s" % "-- missing --")
                continue
            png = os.path.join(ddir, os.path.basename(r["png"])) if r.get("png") else None
            cells.append("%9.3f %7.4f %7.3f" % (r["f_mean"], _hf_frac(r, png),
                                                r.get("x_norm_final", float("nan"))))
        print(f"{lam:>7.3f} " + " ".join(cells))

    rerun, stored = loaded[2][1], loaded[3][1]
    if rerun is not None and stored is not None:
        print("\ncross-check: re-run flow_guided arm vs Phase 3's stored max5.5 at matched lambda")
        print("(same z0, same lambda_s, same L40S -- a mismatch means the reward config or the "
              "reference bank diverged, and NO cross-arm conclusion is valid yet)")
        for lam in LAM_GRID:
            a, b = _find(rerun, lam), _find(stored, lam)
            if a is None or b is None:
                continue
            rel = abs(a["f_mean"] - b["f_mean"]) / max(abs(b["f_mean"]), 1e-12)
            print(f"  lam={lam:7.3f}  rerun={a['f_mean']:10.4f}  stored={b['f_mean']:10.4f}  "
                  f"rel={rel:8.2%}  {'OK' if rel < 0.02 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
