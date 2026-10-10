"""Builds the full prompt x seed x alg x lambda grid for the Brownian-reward campaign.

24 rows (prompt -> seed -> alg, outer to inner) x 11 columns (lambda = 0, 0.1, ..., 1.0). Reads only
`pc_sweep_results_*.json` / `cfg_sweep_results_*.json` and their `*_decoded_*/` PNGs -- no GPU, so it
can be re-run any time, including mid-campaign (missing cells drawn as "pending" boxes).

Two IDENTICAL images are written, differing only in the per-cell caption:
    reward_compare_grid.png            lam=.. m=.. r=.. t=..s   (r = reward value, t = seconds)
    reward_compare_grid_plain.png      lam=.. m=..               (no r, no t)

READ THE IMAGES, NOT r. Neither r (f_mean) nor ||x||/sqrt(d) nor the high-frequency fraction predicts
recognizability (CLAUDE.md); r especially climbs roughly geometrically with lambda straight through
cells that visually recover, as job 1004109 showed within this same reward. r is printed for
provenance, not as a quality signal.

    python render_grid.py
"""
from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE5 = os.path.join(HERE, "..", "flux_guided_phase5")
CFGDIR = os.path.join(HERE, "..", "flux_guided_cfg")

N_STEPS = 10            # both drivers' default; must match what submit_all.sh actually ran
CORRECTOR_STEPS = 1
ETA_REFERENCE = "total"
SNR = 0.16
W_CFG = 2.0
LAM_K = list(range(11))      # 0..10
LAM_STEP = 0.1
REWARD = "brownian"

# (prompt, seed), outer-to-inner row order
PAIRS = [
    ("car", 1234),
    ("jacket", 1234),
    ("jacket", 3141),
    ("A dog", 1234),
    ("A dog", 4242),
    ("A dog", 3141),
    ("sofa", 1234),
    ("teapot", 1234),
]
# (label, lookup) -- lookup returns (results_path, decoded_dir) for a given (prompt, seed)
ALGS = [
    ("flow_guided", "pc"),
    ("PC c=1", "pc"),
    (f"CFG w={W_CFG:g}", "cfg"),
]


def _slug(prompt: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() else "-" for ch in prompt).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return s or "empty"


def _pc_key(prompt: str, seed: int, arm: str) -> str:
    return (f"{_slug(prompt)}_{REWARD}_{arm}_n{N_STEPS}_c{CORRECTOR_STEPS}_eta{ETA_REFERENCE}"
            f"_snr{SNR:g}_s{seed}")


def _cfg_key(prompt: str, seed: int) -> str:
    return f"{_slug(prompt)}_{REWARD}_w{W_CFG:g}_n{N_STEPS}_s{seed}"


def _paths(prompt: str, seed: int, kind: str, arm: str | None) -> tuple[str, str]:
    if kind == "pc":
        key = _pc_key(prompt, seed, arm or "")
        return (os.path.join(PHASE5, f"pc_sweep_results_{key}.json"),
                os.path.join(PHASE5, f"pc_decoded_{key}"))
    key = _cfg_key(prompt, seed)
    return (os.path.join(CFGDIR, f"cfg_sweep_results_{key}.json"),
            os.path.join(CFGDIR, f"cfg_decoded_{key}"))


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf") if bold
        else ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf")
    )
    for base in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"):
        for n in names:
            p = os.path.join(base, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


THUMB = 150
LABEL_H = 30            # per-cell caption strip
COL_HEAD_H = 26         # lambda header row
ROWLBL_W = 150           # left label column (prompt/seed/alg)
PROMPT_BAND_W = 10       # thin colour band marking a prompt group, like pc_current_run_grid's blue rule
PAD = 2


def _row_label(prompt: str, seed: int, alg: str, *, is_first_seed_row: bool,
               is_first_prompt_row: bool) -> list[tuple[str, tuple]]:
    """Lines to draw in the row-label column, batched: prompt only on a prompt's first row, seed only
    on a seed's first row (mirrors pc_current_run_grid.png's 'seed N / arm' grouping, one level deeper).
    """
    lines = []
    if is_first_prompt_row:
        lines.append((f'"{prompt}"', "#0066cc"))
    if is_first_seed_row:
        lines.append((f"seed {seed}", "#006600"))
    lines.append((alg, "black"))
    return lines


def build(out_path: str, *, with_stats: bool) -> None:
    n_rows = len(PAIRS) * len(ALGS)
    n_cols = len(LAM_K)
    cell_w = THUMB + 2 * PAD
    cell_h = THUMB + LABEL_H + 2 * PAD
    W = ROWLBL_W + PROMPT_BAND_W + cell_w * n_cols
    H = COL_HEAD_H + cell_h * n_rows
    grid = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(grid)
    f_head = _font(15, bold=True)
    f_lbl = _font(12, bold=True)
    f_cap = _font(11)

    for col, k in enumerate(LAM_K):
        lam = round(k * LAM_STEP, 4)
        x0 = ROWLBL_W + PROMPT_BAND_W + col * cell_w
        draw.text((x0 + PAD + 4, 4), f"λ={lam:g}", fill="black", font=f_head)

    n_populated = 0
    row = 0
    prompt_start_row: dict[str, int] = {}
    n_rows_per_pair = len(ALGS)
    for pi, (prompt, _seed) in enumerate(PAIRS):
        prompt_start_row.setdefault(prompt, pi * n_rows_per_pair)
    for pi, (prompt, seed) in enumerate(PAIRS):
        is_first_seed_row = True
        is_first_prompt_row = (row == prompt_start_row[prompt])
        for alg_label, kind in ALGS:
            arm = {"flow_guided": "flow_guided", "PC c=1": "pc_guided"}.get(alg_label)
            results_path, decoded_dir = _paths(prompt, seed, kind, arm)
            res = None
            if os.path.exists(results_path):
                with open(results_path) as fh:
                    res = json.load(fh)
            lam_s = res["stamp"].get("lambda_s") if res else None

            y0 = COL_HEAD_H + row * cell_h
            # prompt colour band: one flat colour per prompt GROUP (all its seeds), alternating by
            # prompt index -- mirrors pc_current_run_grid.png's blue seed rule, one level up.
            draw.rectangle([ROWLBL_W, y0, ROWLBL_W + PROMPT_BAND_W - 1, y0 + cell_h - 1],
                           fill="#cfe3ff" if pi % 2 == 0 else "#e3f0cf")

            lines = _row_label(prompt, seed, alg_label, is_first_seed_row=is_first_seed_row,
                               is_first_prompt_row=is_first_prompt_row)
            ty = y0 + 4
            for text, color in lines:
                draw.text((4, ty), text, fill=color, font=f_lbl)
                ty += 15
            is_first_seed_row = False
            is_first_prompt_row = False

            for col, k in enumerate(LAM_K):
                x0 = ROWLBL_W + PROMPT_BAND_W + col * cell_w
                cell_text = None
                img = None
                if res is not None:
                    # Match on the recorded LAMBDA VALUE, not the dict key or idx -- the two drivers
                    # format their keys slightly differently (idx-based vs lam-based), but both record
                    # the true lam float, which is the one axis guaranteed comparable across them.
                    target = round(k * LAM_STEP, 4)
                    key = next((kk for kk in res if kk != "stamp"
                               and abs(res[kk].get("lam", -1) - target) < 1e-6), None)
                    if key is not None:
                        r = res[key]
                        png_path = (os.path.join(decoded_dir, os.path.basename(r["png"]))
                                    if r.get("png") else None)
                        if png_path and os.path.exists(png_path):
                            img = Image.open(png_path).convert("RGB").resize(
                                (THUMB, THUMB), Image.Resampling.LANCZOS)
                            n_populated += 1
                        m = (r["lam"] / lam_s) if lam_s else float("nan")
                        t_key = "t_sample_s" if "t_sample_s" in r else "t_guidance_s"
                        if with_stats:
                            cell_text = (f"λ={r['lam']:.2f} m={m:.4f}\n"
                                        f"r={r['f_mean']:.2f} t={r.get(t_key, float('nan')):.0f}s")
                        else:
                            cell_text = f"λ={r['lam']:.2f} m={m:.4f}"
                if img is not None:
                    grid.paste(img, (x0 + PAD, y0 + PAD))
                else:
                    draw.rectangle([x0 + PAD, y0 + PAD, x0 + PAD + THUMB, y0 + PAD + THUMB],
                                   outline="gray")
                    draw.text((x0 + PAD + 8, y0 + PAD + THUMB // 2 - 6), "pending",
                              fill="gray", font=f_cap)
                if cell_text is not None:
                    draw.text((x0 + PAD + 2, y0 + PAD + THUMB + 2), cell_text,
                              fill="black", font=f_cap)
            row += 1

    grid.save(out_path)
    total = n_rows * n_cols
    print(f"wrote {out_path} ({grid.size[0]}x{grid.size[1]})  {n_populated}/{total} cells populated")


def main() -> None:
    build(os.path.join(HERE, "reward_compare_grid.png"), with_stats=True)
    build(os.path.join(HERE, "reward_compare_grid_plain.png"), with_stats=False)


if __name__ == "__main__":
    main()
