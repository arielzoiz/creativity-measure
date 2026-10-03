"""The ONE figure for Phase 5's result: PC against Phase 3, paired per seed.

`render_pc_comparison.py` dumps every arm on one canvas (14 rows x 18 lambdas) and is a data browser.
This is the answer instead: for each of the 5 seeds, two adjacent rows -- Phase 3 (no corrector) directly
above PC (corrector_steps=1) -- at matched lambda and matched n_steps=10, so every vertical pair differs
by exactly one thing: the corrector.

WHAT TO LOOK FOR
  Columns lam = 0.39 / 0.79 / 1.18 are the replication range; every seed has both arms there.
  - Novelty (REPLICATES, 5/5): in each pair, PC is the more strongly restyled image, and its f is
    13-58% higher. The restyles differ by seed -- flat silhouette, etched scratchboard, neon line-art,
    pop-art, ink cartoon -- rather than converging on one shape.
  - Window extension (DOES NOT REPLICATE): look at where each pair stops being a dog. Mostly both arms
    fail at the same lambda. Seed 1234 at lam=2.357 is the one case where PC survives and Phase 3 does
    not -- that was wave 1's headline. Seed 4242 at lam=1.179 is the counterexample: PC renders the
    digits "97" while Phase 3 still shows a creature face.

Phase 3 cells come from its own stored runs (fine_decoded_seed*/ and fine_decoded_max5.5/), produced at
n_steps=10 on L40S over the same lambda lattice, so the pairing is like-for-like.

    python render_headline.py
"""
from __future__ import annotations

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PHASE3 = os.path.join(HERE, "..", "flux_guided_phase3")
LAM_STEP = 5.5 / 14

LAM_K = [1, 2, 3, 4, 6]                       # 0.393 0.786 1.179 1.571 2.357
SEEDS = [1234, 2024, 3141, 4242, 5555]
THUMB = 210
LAB_W = 150                                   # left gutter for row labels
LAB_H = 30                                    # per-cell caption strip
HDR_H = 34                                    # column header strip
PAD = 3
OUT = os.path.join(HERE, "pc_headline_grid.png")


def _font(sz: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _pc_rows() -> dict[int, tuple[dict, str]]:
    """seed -> (merged results, decoded dir prefix). PC points for one seed live across several jobs
    (the lam-k 4..7 wave and the lam-k 1..3 supplement), so merge them by lambda."""
    import glob
    out: dict[int, tuple[dict, list[str]]] = {}
    for path in sorted(glob.glob(os.path.join(HERE, "pc_sweep_results_pc_guided_*.json"))):
        if ".dryrun" in path or ".stale" in path:
            continue
        res = json.load(open(path))
        st = res.get("stamp", {})
        if st.get("eta_reference") != "total" or st.get("corrector_steps") != 1:
            continue
        if st.get("z0_seed") is not None or st.get("n_steps_ode") != 10:
            continue
        seed = st.get("sweep_seed")
        key = os.path.basename(path)[len("pc_sweep_results_"):-len(".json")]
        ddir = os.path.join(HERE, f"pc_decoded_{key}")
        merged, dirs = out.get(seed, ({}, []))
        for k, v in res.items():
            if k != "stamp":
                merged[round(v["lam"], 4)] = (v, ddir)
        dirs.append(ddir)
        out[seed] = (merged, dirs)
    return {s: (m, d) for s, (m, d) in out.items()}            # type: ignore[misc]


def _p3_row(seed: int) -> dict[float, tuple[dict, str]]:
    name = ("fine_lambda_sweep_results_max5.5.json" if seed == 1234
            else f"fine_lambda_sweep_results_seed{seed}.json")
    ddir = os.path.join(PHASE3, "fine_decoded_max5.5" if seed == 1234 else f"fine_decoded_seed{seed}")
    path = os.path.join(PHASE3, name)
    if not os.path.exists(path):
        return {}
    res = json.load(open(path))
    return {round(v["lam"], 4): (v, ddir) for k, v in res.items() if k != "stamp"}


def main() -> None:
    f_hdr, f_lab, f_cap = _font(19), _font(16), _font(14)
    pc = _pc_rows()
    lams = [round(k * LAM_STEP, 4) for k in LAM_K]

    rows: list[tuple[str, str, dict]] = []
    for s in SEEDS:
        rows.append((f"seed {s}", "Phase 3", _p3_row(s)))
        rows.append(("", "PC (corr=1)", pc.get(s, ({}, []))[0]))

    cw, ch = THUMB + 2 * PAD, THUMB + LAB_H + 2 * PAD
    W = LAB_W + cw * len(lams)
    H = HDR_H + ch * len(rows)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    for ci, lam in enumerate(lams):
        d.text((LAB_W + ci * cw + 8, 8), f"lambda = {lam:.3f}", fill="black", font=f_hdr)

    for ri, (seedlab, armlab, data) in enumerate(rows):
        y0 = HDR_H + ri * ch
        if seedlab:
            d.text((6, y0 + 6), seedlab, fill="#0066cc", font=f_hdr)
            d.line([(0, y0 - 1), (W, y0 - 1)], fill="#999999", width=2)
        d.text((6, y0 + (30 if seedlab else 6)), armlab,
               fill="#aa0000" if armlab.startswith("PC") else "#333333", font=f_lab)
        for ci, lam in enumerate(lams):
            x0 = LAB_W + ci * cw
            hit = data.get(lam)
            if hit is None:
                d.rectangle([x0 + PAD, y0 + PAD, x0 + PAD + THUMB, y0 + PAD + THUMB], outline="#cccccc")
                d.text((x0 + PAD + 60, y0 + PAD + THUMB // 2), "not run", fill="#999999", font=f_cap)
                continue
            rec, ddir = hit
            png = os.path.join(ddir, os.path.basename(rec["png"])) if rec.get("png") else None
            if png and os.path.exists(png):
                img.paste(Image.open(png).convert("RGB").resize((THUMB, THUMB), Image.Resampling.LANCZOS),
                          (x0 + PAD, y0 + PAD))
            else:
                d.rectangle([x0 + PAD, y0 + PAD, x0 + PAD + THUMB, y0 + PAD + THUMB], outline="#cccccc")
            d.text((x0 + PAD + 2, y0 + PAD + THUMB + 6), f"f = {rec['f_mean']:.2f}",
                   fill="#aa0000" if armlab.startswith("PC") else "#333333", font=f_cap)

    img.save(OUT)
    print(f"wrote {OUT}  ({img.size[0]}x{img.size[1]})")
    print("\nEach vertical pair = same seed, same lambda, same n_steps; differs ONLY by the corrector.")
    print("Novelty gain replicates 5/5. Window extension does not: compare where each pair stops")
    print("being a dog -- seed 1234 @ 2.357 supports it, seed 4242 @ 1.179 contradicts it.")


if __name__ == "__main__":
    main()
