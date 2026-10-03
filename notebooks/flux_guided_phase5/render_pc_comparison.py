"""Builds the Phase 5 three-arm comparison: one image grid plus the numeric table.

Rows are DISCOVERED from whatever result files exist (each is keyed on its full config: arm, n_steps,
corrector_steps, eta_reference, snr, seeds), so adding an ablation adds a row with no edit here. Columns
are the union of every lambda present, snapped to Phase 3's zoom lattice. A final row reads Phase 3's OWN
stored max5.5 images at the same lambdas, as an independent cross-check on the re-run flow_guided arm.

Also prints a decision gate and a tilt-vs-inflation frontier keyed on |x|/sqrt(d), plus the within-seed
spread at fixed z0.

!! READ THIS BEFORE TRUSTING EITHER |x|-BASED TABLE. Wave 1's images falsified both of their premises:
   (1) the "compute-matched" flow_guided @19 control is NOT a valid control -- at lam=1.571 it is destroyed
       (abstract blocks, no dog) while the SAME algorithm at n_steps=10 gives a recognizable dog, so more
       ODE steps break guidance rather than merely costing more. Comparisons against it are
       PC-versus-broken.
   (2) |x|/sqrt(d) is ANTI-correlated with image quality: the destroyed image measures 2.88 against the
       intact PC image's 3.90.
   `hf` tracked recognizability correctly at every comparison and is the proxy to keep. The |x| tables are
   retained because the quantity is still worth recording, NOT because low |x| means good. The real
   arbiter is the decoded PNGs, which are written per point -- look at them first.

No GPU. Only reads what is already on disk, so it can be re-run any time, including mid-sweep (cells with
no result are left blank, like render_fine_sweep_5seed.py's).

HOW TO READ IT, rewritten after wave 1 (the original text predicted the opposite of what happened, on a
wrong mechanism -- it is kept nowhere, but the error is worth naming): the corrector does NOT pull back
toward p_t. It targets q_t ~ p_t exp(lam r), which at any meaningful lambda is itself off-manifold, so at
matched lambda the PC arms report HIGHER f, not lower. Measured: +65% at lam=1.57, +21% at lam=2.36.

The claim under test is whether structure survives further up lambda, and only the IMAGES answer it. Wave
1's answer, at matched lambda and matched n_steps=10: at lam=1.571 both Phase 3 and PC are recognizable
dogs (f 14.2 vs 23.5); at lam=2.357 Phase 3 has degraded to a crude pictograph while PC is still clearly a
dog (f 47.2 vs 57.2); by lam=3.143 both are broken. So the corrector buys about one lambda step of extra
structural integrity plus 20-65% more novelty, for 1.9x the compute.

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
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pc_sweep import hf_power_fraction                                                     # noqa: E402

PHASE3 = os.path.join(HERE, "..", "flux_guided_phase3")
LAM_STEP = 5.5 / 14


def _label(stamp: dict) -> str:
    """Short row label from the stamp, naming only the axes a wave actually varies."""
    arm = stamp.get("arm", "?")
    bits = [arm]
    if arm != "flow_guided":
        bits.append(f"c{stamp.get('corrector_steps')}")
        if stamp.get("eta_reference") != "total":
            bits.append(f"eta={stamp.get('eta_reference')}")
    n = stamp.get("n_steps_ode")
    if n != 10:
        bits.append(f"n{n}")
    if stamp.get("z0_seed") is not None:
        bits.append(f"z0={stamp['z0_seed']}/s{stamp.get('sweep_seed')}")
    elif stamp.get("sweep_seed") != 1234:
        bits.append(f"s{stamp.get('sweep_seed')}")
    return " ".join(bits)


def discover_rows() -> list[tuple[str, dict, str]]:
    """(label, results, decoded_dir) for every Phase 5 result on disk, plus Phase 3's stored max5.5.

    Globbed rather than hardcoded: results are keyed on the full config (arm / n_steps /
    corrector_steps / eta_reference / snr / seeds), so a wave that adds an axis adds rows here with no
    edit. Sorted so the three primary arms lead and the ablations follow.
    """
    import glob

    rows: list[tuple[str, dict, str]] = []
    for path in sorted(glob.glob(os.path.join(HERE, "pc_sweep_results_*.json"))):
        if ".dryrun" in path or path.endswith((".stale", ".tmp")):
            continue
        with open(path) as fh:
            res = json.load(fh)
        if "stamp" not in res:
            continue
        key = os.path.basename(path)[len("pc_sweep_results_"):-len(".json")]
        rows.append((_label(res["stamp"]), res, os.path.join(HERE, f"pc_decoded_{key}")))

    order = {"pc_guided": 0, "pc_unguided": 1, "flow_guided": 2}
    rows.sort(key=lambda r: (order.get(r[1]["stamp"].get("arm", ""), 3), r[0]))

    p3 = os.path.join(PHASE3, "fine_lambda_sweep_results_max5.5.json")
    if os.path.exists(p3):
        with open(p3) as fh:
            rows.append(("phase 3 max5.5 (stored)", json.load(fh),
                         os.path.join(PHASE3, "fine_decoded_max5.5")))
    return rows


def lam_grid(rows: list[tuple[str, dict, str]]) -> list[float]:
    """Union of every lambda present, snapped to the Phase 3 lattice and sorted."""
    ks: set[int] = set()
    for _, res, _ in rows:
        for k, v in res.items():
            if k != "stamp":
                ks.add(int(round(v["lam"] / LAM_STEP)))
    return [k * LAM_STEP for k in sorted(ks)]


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
    loaded = discover_rows()
    if not loaded:
        print("no results on disk yet -- nothing to render")
        return
    LAM_GRID = lam_grid(loaded)
    cell_w = THUMB + 2 * PAD
    cell_h = THUMB + LABEL_H + 2 * PAD
    grid = Image.new("RGB", (cell_w * len(LAM_GRID), cell_h * len(loaded)), "white")
    draw = ImageDraw.Draw(grid)

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

    def pick(**want: Any) -> dict | None:
        """The one results dict whose stamp matches every key=value given (rows are config-keyed)."""
        for _, res, _ in loaded:
            st = res.get("stamp", {})
            if all(st.get(k) == v for k, v in want.items()):
                return res
        return None

    # --- cross-check: the re-run Phase 3 arm against Phase 3's own stored numbers -----------------
    rerun = pick(arm="flow_guided", n_steps_ode=10, sweep_seed=1234)
    stored = next((res for lab, res, _ in loaded if lab.startswith("phase 3")), None)
    if rerun and stored:
        print("\ncross-check: re-run flow_guided @10 vs Phase 3's stored max5.5 at matched lambda")
        print("(same z0, same lambda_s, same L40S -- a mismatch means the reward config or the "
              "reference bank diverged, and NO cross-arm conclusion is valid yet)")
        for lam in LAM_GRID:
            a, b = _find(rerun, lam), _find(stored, lam)
            if a is None or b is None:
                continue
            rel = abs(a["f_mean"] - b["f_mean"]) / max(abs(b["f_mean"]), 1e-12)
            print(f"  lam={lam:7.3f}  rerun={a['f_mean']:10.4f}  stored={b['f_mean']:10.4f}  "
                  f"rel={rel:8.2%}  {'OK' if rel < 0.02 else 'MISMATCH'}")

    # --- the decision gate ------------------------------------------------------------------------
    # Primary readout is x_norm, NOT f: off-manifold failure inflates the latent norm, and f rises
    # forever under tilt so it cannot be the signal (CLAUDE.md's standing rule). The gate asks whether
    # a PC arm holds the norm where the COMPUTE-MATCHED control does not -- that is the only comparison
    # that separates "the corrector works" from "more reward-gradient evaluations work".
    pcg = pick(arm="pc_guided", corrector_steps=1, eta_reference="total", z0_seed=None, sweep_seed=1234)
    pcs = pick(arm="pc_guided", corrector_steps=1, eta_reference="score", z0_seed=None, sweep_seed=1234)
    pcu = pick(arm="pc_unguided", corrector_steps=1, z0_seed=None, sweep_seed=1234)
    ctrl19 = pick(arm="flow_guided", n_steps_ode=19)
    if ctrl19:
        print("\n" + "=" * 100)
        print("DECISION GATE -- |x|/sqrt(d) vs the COMPUTE-MATCHED control (flow_guided @19 == 19 guided")
        print("units == pc_guided @10,corr=1). Lower is more on-manifold. 'base' is the lam=0 reference.")
        print("=" * 100)
        cand = [("pc_guided eta=total", pcg), ("pc_guided eta=score", pcs), ("pc_unguided", pcu)]
        print(f"{'lam':>7s} {'ctrl@19':>9s} " + " ".join(f"{n:>21s}" for n, _ in cand))
        for lam in LAM_GRID:
            c = _find(ctrl19, lam)
            if c is None:
                continue
            cx = c.get("x_norm_final", float("nan"))
            cells = []
            for _, res in cand:
                r = _find(res, lam) if res else None
                if r is None:
                    cells.append("%21s" % "--")
                    continue
                x = r.get("x_norm_final", float("nan"))
                cells.append("%10.3f (%+6.1f%%)" % (x, 100.0 * (x / cx - 1.0) if cx else float("nan")))
            print(f"{lam:>7.3f} {cx:>9.3f} " + " ".join(cells))
        print("\nRead it as: a PC arm 'holds' iff its |x| stays near the lam=0 value while ctrl@19 climbs.")
        print("If ctrl@19 ALSO holds, Phase 3's ceiling was ODE discretization error, not an")
        print("off-manifold attractor -- in which case the corrector is not the mechanism and the")
        print("premise of Phase 5 needs rewriting. Confirm against the images before concluding.")

    # --- the TILT-vs-OFF-MANIFOLD FRONTIER, which is the comparison that actually decides -----------
    #
    # Matched-lambda tables ask "which arm tilts harder at this lambda", and pc_guided wins that by
    # construction: its corrector drift is s + lam*g~ with ||g~|| == ||s||, so at lam > 1 the drift is
    # reward-DOMINATED and each corrector step pushes further up r. (The early framing of the corrector as
    # "pulling back toward p_t" was wrong -- it pulls toward q_t ~ p_t exp(lam r), which at any meaningful
    # lambda is itself off-manifold.) Measured at matched compute, lam=1.57: pc_guided reached f=23.48 vs
    # the control's 14.47, but with ||x||/sqrt(d) 3.90 vs 2.88 -- MORE inflation, not less.
    #
    # So the arms sit on a monotone tilt-vs-inflation tradeoff and the useful question is whether any arm
    # shifts the FRONTIER: at matched f, does it reach it with lower ||x|| (and lower hf)? That is what
    # this table answers, by interpolating each arm's ||x|| and hf onto a common f grid. An arm that
    # dominates here is genuinely better; an arm that merely sits further along the same curve is just
    # tilting harder, which costs nothing but lambda.
    print("\n" + "=" * 100)
    print("TILT-vs-OFF-MANIFOLD FRONTIER -- ||x||/sqrt(d) and hf at MATCHED f (linear interp in log f).")
    print("Lower ||x|| at the same f is better. An arm absent from a row does not reach that f.")
    print("=" * 100)

    def interp(xs: list[float], ys: list[float], x: float) -> float | None:
        """y at x by linear interpolation on sorted (xs, ys); None outside the measured range."""
        if len(xs) < 2 or not (min(xs) <= x <= max(xs)):
            return None
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                if xs[i + 1] == xs[i]:
                    return ys[i]
                w = (x - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] * (1 - w) + ys[i + 1] * w
        return None

    def short(lab: str) -> str:
        """Distinguishing label. Naive truncation collides: every pc_guided variant becomes 'pc_guided',
        and 'pc_guided c1 z0=1234/s101' vs '.../s202' both cut to the same 15 chars -- so the seed, which
        is the ONLY thing distinguishing the within-seed replicates, has to survive."""
        import re as _re
        s = (lab.replace("pc_unguided", "unguid").replace("pc_guided", "guid")
                .replace("flow_guided", "ctrl").replace("eta=", "").replace("phase 3 ", "p3")
                .replace(" (stored)", "").replace("max5.5", ""))
        return _re.sub(r"z0=\d+/", "z0/", s)[:15]

    curves: list[tuple[str, list[float], list[float], list[float]]] = []
    for lab, res, _ddir in loaded:
        trips: list[tuple[float, float, float]] = []
        for k, v in res.items():
            if k == "stamp":
                continue
            xv = v.get("x_norm_final")
            if xv is None or xv != xv:          # absent or nan: Phase 3's stored run never recorded it
                continue
            trips.append((float(v["f_mean"]), float(xv), _hf_frac(v, None)))
        trips.sort(key=lambda p: p[0])
        if len(trips) >= 2:
            curves.append((lab, [p[0] for p in trips], [p[1] for p in trips], [p[2] for p in trips]))

    if not curves:
        print("(need >=2 points with x_norm per arm; Phase 3's stored run never recorded it)")
    else:
        print("per-arm f range measured so far:")
        for lab, fs, _, _ in curves:
            print(f"  {short(lab):>16s}  f in [{min(fs):9.3f}, {max(fs):9.3f}]  ({len(fs)} pts)")
        # Define the common range from the FULL-GRID arms only. A deliberately narrow arm (the baseline
        # extension at 3 high lambdas, an ablation at 2 points, a within-seed replicate) would otherwise
        # collapse the overlap to nothing and hide the comparison the full arms can actually support.
        # Narrow arms still appear as columns; they read "--" outside their own measured range.
        widest = max(len(c[1]) for c in curves)
        main = [c for c in curves if len(c[1]) >= max(4, widest // 2)] or curves
        lo = max(min(c[1]) for c in main)
        hi = min(max(c[1]) for c in main)
        if len(main) < len(curves):
            print(f"  (common range set by the {len(main)} full-grid arms; "
                  f"{len(curves) - len(main)} narrow arms shown but not range-limiting)")
        if hi <= lo:
            print(f"\nNO COMMON f RANGE yet (need max-of-mins {lo:.3f} < min-of-maxes {hi:.3f}).")
            print("Arms covering disjoint lambda ranges cannot be compared on the frontier until their")
            print("f ranges overlap -- expected once each arm's grid fills in.")
        else:
            print(f"\n{'target f':>10s} " + " ".join(f"{short(c[0]):>15s}" for c in curves))
            print(f"{'':>10s} " + " ".join(f"{'|x| / hf':>15s}" for _ in curves))
            n_rows = 8
            for i in range(n_rows):
                t = lo * (hi / lo) ** (i / (n_rows - 1))      # geometric, f spans orders of magnitude
                cells = []
                for _, fs, xn, hfs in curves:
                    x, h = interp(fs, xn, t), interp(fs, hfs, t)
                    cells.append("%15s" % "--" if x is None
                                 else "%7.3f /%6.4f" % (x, h if h is not None else float("nan")))
                print(f"{t:>10.3f} " + " ".join(cells))
            print("\n(overlap region only: f in [%.3f, %.3f], where every arm has measurements)"
                  % (lo, hi))

    # --- within-seed (corrector-noise) variance, which sizes the seed replication -----------------
    fixed = [(lab, res) for lab, res, _ in loaded if res.get("stamp", {}).get("z0_seed") is not None]
    if len(fixed) >= 2:
        print("\nwithin-seed spread at FIXED z0 (varies only the Langevin noise -- the variance component")
        print("Phase 3 structurally does not have; this is what sizes the seed replication):")
        lams = sorted({v["lam"] for _, res in fixed for k, v in res.items() if k != "stamp"})
        for lam in lams:
            vals = [(_find(res, lam) or {}).get("f_mean") for _, res in fixed]
            xs = [(_find(res, lam) or {}).get("x_norm_final") for _, res in fixed]
            vals = [v for v in vals if v is not None]
            xs = [v for v in xs if v is not None]
            if len(vals) < 2:
                continue
            import statistics as st
            print(f"  lam={lam:7.3f}  n={len(vals)}  f: mean={st.mean(vals):9.4f} sd={st.stdev(vals):8.4f} "
                  f"cv={st.stdev(vals)/max(abs(st.mean(vals)),1e-12):6.1%}   "
                  f"|x|: mean={st.mean(xs):6.3f} sd={st.stdev(xs):6.3f}")


if __name__ == "__main__":
    main()
