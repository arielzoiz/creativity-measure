"""The prompt study (2026-10-07/08) as one figure: 5 prompts x 2 seeds x {C=0, C=1} over lambda.

Rows are grouped by (prompt, seed), two per group:
    flow_guided (corrector_steps=0, Phase 3's own control)
    pc_guided   (corrector_steps=1)

Columns are the UNION of the lambda points present on disk, sorted ascending. The study was run as two
interleaved waves whose results live in separate files -- wave A {0, 0.2, ..., 1.0} untagged, wave B
{0.1, 0.3, 0.5, 0.7, 0.9} under `--tag odd` -- because the results filename does not encode `lam_step`,
so an untagged wave B would have collided with wave A and .stale-renamed it. This script merges them
back into one lattice for display, which is the only place the split should ever be visible.

Missing points are left blank rather than skipped, so a column means the same lambda in every row and
an incomplete run reads as a gap instead of silently shifting its row.

    python render_prompt_study.py
"""
from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = ["building", "sofa", "car", "teapot", "jacket"]
SEEDS = [1234, 3141]
ARMS = [("flow_guided", "C=0 (Phase 3)"), ("pc_guided", "C=1")]
TAGS = ["", "_odd"]                        # wave A, wave B

THUMB = 150
LAB_W = 150
CAP_H = 16
HDR_H = 22
GROUP_GAP = 8       # between the two seeds of one prompt
PROMPT_GAP = 26     # between prompts
OUT = os.path.join(HERE, "prompt_study_grid.png")


def load(prompt: str, seed: int, arm: str) -> dict[float, dict]:
    """All points for one (prompt, seed, arm), merged across both waves, keyed by lambda."""
    pts: dict[float, dict] = {}
    for tag in TAGS:
        f = os.path.join(
            HERE, f"pc_sweep_results_{prompt}_{arm}_n10_c1_etatotal_snr0.16_s{seed}{tag}.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            res = json.load(fh)
        decoded = os.path.join(
            HERE, f"pc_decoded_{prompt}_{arm}_n10_c1_etatotal_snr0.16_s{seed}{tag}")
        for key, r in res.items():
            if not key.startswith("idx"):
                continue
            png = os.path.join(decoded, f"{key}.png")
            pts[round(float(r["lam"]), 4)] = {**r, "png": png}
    return pts


def main() -> None:
    rows: list[tuple[str, int, str, str, dict[float, dict]]] = []
    lams: set[float] = set()
    for p in PROMPTS:
        for s in SEEDS:
            for arm, label in ARMS:
                pts = load(p, s, arm)
                lams |= set(pts)
                rows.append((p, s, arm, label, pts))
    cols = sorted(lams)
    if not cols:
        print("no results on disk yet")
        return

    w = LAB_W + THUMB * len(cols)
    n_prompts = len(rows) // (len(ARMS) * len(SEEDS))
    h = (HDR_H + len(rows) * (THUMB + CAP_H)
         + n_prompts * PROMPT_GAP + n_prompts * (len(SEEDS) - 1) * GROUP_GAP)
    out = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(out)
    for j, lam in enumerate(cols):
        d.text((LAB_W + j * THUMB + THUMB // 3, 5), f"lam={lam:g}", fill="black")

    # Two nesting levels have to be visually distinguishable, or the figure cannot be read:
    # rows pair into (prompt, seed) -- the C=0 / C=1 comparison -- and pairs pair again into one prompt.
    # A single separator every 2 rows makes the prompt grouping invisible, so the prompt boundary gets a
    # thick rule plus a wider gap, and the seed boundary inside a prompt only a hairline.
    y = HDR_H
    for i, (p, s, arm, label, pts) in enumerate(rows):
        if i % len(ARMS) == 0:
            new_prompt = i % (len(ARMS) * len(SEEDS)) == 0
            y += PROMPT_GAP if new_prompt else GROUP_GAP
            if new_prompt:
                d.line([(0, y - 5), (w, y - 5)], fill="#444444", width=3)
                d.text((4, y + 2), f"=== {p} ===", fill="#444444")
            else:
                d.line([(LAB_W, y - 3), (w, y - 3)], fill="#cccccc")
        d.text((4, y + THUMB // 2 - 14), f"{p}\ns{s}\n{label}",
               fill=("#0000aa" if arm == "flow_guided" else "#aa0000"))
        for j, lam in enumerate(cols):
            r = pts.get(lam)
            x = LAB_W + j * THUMB
            if r is None or not os.path.exists(r["png"]):
                d.rectangle([x + 1, y + 1, x + THUMB - 2, y + THUMB - 2], outline="#eeeeee")
                continue
            out.paste(Image.open(r["png"]).convert("RGB").resize((THUMB, THUMB)), (x, y))
            d.text((x + 2, y + THUMB + 2), f"f={r['f_mean']:.2f}", fill="black")
        y += THUMB + CAP_H

    out.save(OUT)
    print(f"wrote {OUT}  {out.size}  ({len(cols)} lambda x {len(rows)} rows)")
    for p in PROMPTS:
        for s in SEEDS:
            n = [len(load(p, s, a)) for a, _ in ARMS]
            print(f"  {p:9s} s{s}: flow_guided {n[0]:2d} pts, pc_guided {n[1]:2d} pts")


if __name__ == "__main__":
    main()
