"""The CFG x lambda sweep as one figure: rows are (prompt, seed, w), columns are lambda.

THE w=1 ROW IS NOT RUN BY cfg_w_sweep.py -- it is read from the stored Phase 3 / Phase 5 `flow_guided`
results, because w=1 IS that baseline (``cfg_velocity_fn`` returns the conditional function itself and
``guided_euler_step`` detects the identity, so the path is bitwise the unsplit one). This script knows
where both live and splices them in as each group's first row:

    "A dog"                          ../flux_guided_phase3/fine_lambda_sweep_results{,_seed<S>,_max5.5}.json
                                     + fine_decoded{,_seed<S>,_max5.5}/
    any prompt-study prompt          ../flux_guided_phase5/pc_sweep_results_<p>_flow_guided_n10_c1_eta...json
                                     + pc_decoded_<p>_flow_guided_n10_c1_eta.../

Those were produced in DIFFERENT JOBS from the CFG rows, so the ~2.5% cross-run nondeterminism floor in f
applies to the w=1 column specifically (CLAUDE.md: FLUX's backward is not bit-reproducible on GPU). That
is immaterial against the 14-27% seed-to-seed CV, and the comparison this figure exists for is VISUAL
anyway. The job's GPU model is printed per row so a mismatch (which WOULD matter -- 16% of std_p(f)) is
visible rather than assumed.

Columns are the union of lambdas present on disk. Missing points are left blank rather than skipped, so
a column means the same lambda in every row and an incomplete run reads as a gap instead of silently
shifting its row.

READ THE IMAGES, NOT f. No scalar here predicts recognizability -- f is 14.2 on a clear dog and 14.8 on
pure noise, ||x|| is anti-correlated with quality, and hf_frac has no absolute threshold across seeds
(CLAUDE.md). The captions carry f and hf only as within-row ranking aids.

    python render_cfg_grid.py                          # everything on disk
    python render_cfg_grid.py --prompt car --seed 1234  # one group
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE3 = os.path.join(HERE, "..", "flux_guided_phase3")
PHASE5 = os.path.join(HERE, "..", "flux_guided_phase5")

THUMB = 150
LAB_W = 170
CAP_H = 26          # two caption lines: a snapped baseline prints its TRUE lambda above f
HDR_H = 22
W_GAP = 6            # between the w rows of one (prompt, seed)
GROUP_GAP = 26       # between (prompt, seed) groups
OUT = os.path.join(HERE, "cfg_w_grid.png")

# cfg_sweep_results_<slug>_w<W>_n<N>_s<SEED>[_tw..][_tag].json
_RESULT_RE = re.compile(r"^cfg_sweep_results_(?P<slug>.+?)_w(?P<w>[0-9.eE+-]+)_n(?P<n>\d+)_s(?P<seed>\d+)"
                        r"(?P<rest>.*)\.json$")


def _points(results_path: str, decoded_dir: str) -> tuple[dict[float, dict], dict]:
    """``({lambda: point}, stamp)`` for one results file, with each point's PNG path resolved.

    The stored ``png`` field is an absolute path from the machine that ran the job, so it is re-derived
    from ``decoded_dir`` here -- the same reason render_prompt_study.py re-derives it.
    """
    if not os.path.exists(results_path):
        return {}, {}
    with open(results_path) as fh:
        res = json.load(fh)
    pts = {
        round(float(r["lam"]), 4): {**r, "png": os.path.join(decoded_dir, f"{key}.png")}
        for key, r in res.items() if key.startswith("idx")
    }
    return pts, res.get("stamp", {})


def cfg_rows() -> dict[tuple[str, int], dict[float, tuple[dict[float, dict], dict]]]:
    """``{(slug, seed): {w: (points, stamp)}}`` discovered by globbing this directory."""
    out: dict[tuple[str, int], dict[float, tuple[dict[float, dict], dict]]] = {}
    for path in sorted(glob.glob(os.path.join(HERE, "cfg_sweep_results_*.json"))):
        m = _RESULT_RE.match(os.path.basename(path))
        if m is None or ".dryrun" in path or ".stale" in path:
            continue
        key = os.path.basename(path)[len("cfg_sweep_results_"):-len(".json")]
        pts, stamp = _points(path, os.path.join(HERE, f"cfg_decoded_{key}"))
        if pts:
            out.setdefault((m["slug"], int(m["seed"])), {})[float(m["w"])] = (pts, stamp)
    return out


def baseline_row(slug: str, seed: int) -> tuple[dict[float, dict], dict]:
    """The stored w=1 (``flow_guided``) run for this (prompt, seed), from whichever phase holds it.

    Several Phase 3 filename conventions are tried because that sweep accumulated them over its own legs:
    the seed-replication jobs wrote ``_seed<S>``, the zoom wrote ``_max5.5`` (seed 1234), and the original
    wrote neither. All are the same configuration -- n_steps=10, L40S, exact_jacobian=True -- so merging
    across them is sound; what must NOT be merged is a different n_steps or GPU, and the printed stamp is
    how that gets caught.
    """
    merged: dict[float, dict] = {}
    stamp: dict = {}
    candidates = [
        (os.path.join(PHASE3, f"fine_lambda_sweep_results_seed{seed}.json"),
         os.path.join(PHASE3, f"fine_decoded_seed{seed}")),
        (os.path.join(PHASE3, "fine_lambda_sweep_results_max5.5.json"),
         os.path.join(PHASE3, "fine_decoded_max5.5")),
        (os.path.join(PHASE3, "fine_lambda_sweep_results.json"),
         os.path.join(PHASE3, "fine_decoded")),
    ] if slug == "a-dog" else []
    for tag in ("", "_odd", "_retry"):
        base = f"{slug}_flow_guided_n10_c1_etatotal_snr0.16_s{seed}{tag}"
        candidates.append((os.path.join(PHASE5, f"pc_sweep_results_{base}.json"),
                           os.path.join(PHASE5, f"pc_decoded_{base}")))
    for results_path, decoded in candidates:
        pts, st = _points(results_path, decoded)
        # Phase 3's files are per-seed by FILENAME, not by stamp key, except max5.5/the original, which
        # are seed 1234 -- check the stamp so a wrong-seed file can never be spliced in silently.
        if pts and st.get("sweep_seed", seed) != seed:
            continue
        merged.update({k: v for k, v in pts.items() if k not in merged})
        stamp = stamp or st
    return merged, stamp


def _snap(lam: float, cols: list[float], tol: float) -> float | None:
    """Nearest column to ``lam`` within ``tol``, else None.

    Exists for ONE case, and it is a labelling-integrity case rather than a cosmetic one. The stored
    "A dog" baseline sits on Phase 3's 5.5/14 lattice (0.392857, 0.785714, ...) while the CFG arms are
    swept on a 0.1 lattice, so those points have no column of their own. Snapping lets them share the
    visually-adjacent column -- but the caption then prints the TRUE lambda, never the column's, because
    0.785714 is a ~5% f shift away from 0.8 on this curve (df/dlam ~ 6.4 there), which is larger than the
    backend's own ~2.5% nondeterminism floor. CLAUDE.md records a cross-run image result that REVERSED
    once a mislabelled baseline was corrected; a snapped point must never be allowed to read as exact.
    """
    if not cols:
        return None
    nearest = min(cols, key=lambda c: abs(c - lam))
    return nearest if abs(nearest - lam) <= tol else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default=None, help="prompt SLUG (e.g. 'car', 'a-dog')")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, default=OUT)
    ap.add_argument("--snap-tol", type=float, default=None,
                    help="max |lambda| distance for an off-lattice STORED baseline point to share a "
                         "swept column (default: 0.49 x the smallest column spacing, i.e. never ambiguous "
                         "between two columns). Stored points that do not snap are DROPPED from the "
                         "baseline row and reported, since the columns are defined by what was swept.")
    args = ap.parse_args()

    found = cfg_rows()
    groups = sorted(k for k in found
                    if (args.prompt is None or k[0] == args.prompt)
                    and (args.seed is None or k[1] == args.seed))
    if not groups:
        print("no CFG results on disk yet (looked for cfg_sweep_results_*.json here)")
        return

    # COLUMNS: the swept (CFG) lambdas, PLUS any stored-baseline lambda that lies exactly on the same
    # lattice. A baseline on a DIFFERENT lattice must never invent a column -- that is what would put
    # 0.785714 and 0.8 side by side as if they were two measurements rather than one lambda drawn twice.
    # But a baseline point at 0.9 on a 0.1 lattice is the same lambda the sweep will reach, and
    # suppressing it until the sweep catches up hides exactly the high-lambda baseline cells where
    # breakdown happens -- which is what the figure is for. So: on-lattice extends, off-lattice snaps.
    swept: set[float] = set()
    for slug, seed in groups:
        for w in found[(slug, seed)]:
            swept |= set(found[(slug, seed)][w][0])
    cols = sorted(swept)
    spacing = min((b - a for a, b in zip(cols, cols[1:])), default=0.1)
    tol = args.snap_tol if args.snap_tol is not None else 0.49 * spacing

    def _on_lattice(lam: float) -> bool:
        """True iff lam is an integer multiple of the swept lattice spacing."""
        k = lam / spacing
        return abs(k - round(k)) < 1e-6

    extra: set[float] = set()
    for slug, seed in groups:
        for lam in baseline_row(slug, seed)[0]:
            if lam not in swept and _on_lattice(lam):
                extra.add(lam)
    if extra:
        print(f"  baseline-only columns (on-lattice, not yet swept): {sorted(round(v, 4) for v in extra)}")
    cols = sorted(swept | extra)

    # rows: (slug, seed, w, points, stamp, is_baseline)
    rows: list[tuple[str, int, float, dict[float, dict], dict, bool]] = []
    for slug, seed in groups:
        base_pts, base_stamp = baseline_row(slug, seed)
        snapped: dict[float, dict] = {}
        dropped: list[float] = []
        for lam, r in sorted(base_pts.items()):
            col = _snap(lam, cols, tol)
            if col is None:
                dropped.append(lam)
                continue
            # `lam_true` is what the caption prints; the dict key is only where it is DRAWN.
            snapped.setdefault(col, {**r, "lam_true": lam})
        if snapped:
            rows.append((slug, seed, 1.0, snapped, base_stamp, True))
            off = [f"{v['lam_true']:.4f}->{k:g}" for k, v in sorted(snapped.items())
                   if abs(v["lam_true"] - k) > 1e-9]
            if off:
                print(f"  {slug} s{seed}: stored baseline SNAPPED to swept columns: {', '.join(off)} "
                      "(captions print the true lambda)")
            if dropped:
                print(f"  {slug} s{seed}: {len(dropped)} stored baseline point(s) outside the swept "
                      f"columns, dropped: {[round(v, 4) for v in dropped]}")
        else:
            print(f"  WARNING {slug} s{seed}: no stored w=1 point lands on the swept lambda columns -- "
                  "this group has NO stored baseline row. Check --lam-step/--sweep-seed against an "
                  "existing flow_guided run.")
        for w in sorted(found[(slug, seed)]):
            pts, stamp = found[(slug, seed)][w]
            rows.append((slug, seed, w, pts, stamp, False))
    if not rows:
        print("nothing to draw")
        return
    width = LAB_W + THUMB * len(cols)
    height = (HDR_H + len(rows) * (THUMB + CAP_H)
              + len(groups) * GROUP_GAP + (len(rows) - len(groups)) * W_GAP)
    out = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(out)
    for j, lam in enumerate(cols):
        d.text((LAB_W + j * THUMB + THUMB // 3, 5), f"lam={lam:g}", fill="black")

    y = HDR_H
    prev_group: tuple[str, int] | None = None
    for slug, seed, w, pts, stamp, is_base in rows:
        if (slug, seed) != prev_group:
            y += GROUP_GAP
            d.line([(0, y - 5), (width, y - 5)], fill="#444444", width=3)
            d.text((4, y + 2), f"=== {slug}  seed {seed} ===", fill="#444444")
            prev_group = (slug, seed)
        else:
            y += W_GAP
            d.line([(LAB_W, y - 3), (width, y - 3)], fill="#cccccc")
        # "stored" vs "this run" has to be on the figure: a w=1 row produced in THIS job shares the
        # job's GPU, reward and reference bank with the CFG rows, while the stored one does not and
        # carries the ~2.5% cross-run f floor. Both are legitimate; conflating them is not.
        label = "w=1 STORED\n(Phase 3/5)" if is_base else (
            "w=1 (this run)" if w == 1.0 else f"w={w:g}")
        gpu = str(stamp.get("gpu", "?")).replace("NVIDIA ", "")
        d.text((4, y + THUMB // 2 - 20), f"{label}\n{gpu}", fill=("#0000aa" if is_base else "#aa0000"))
        for j, lam in enumerate(cols):
            r = pts.get(lam)
            x = LAB_W + j * THUMB
            if r is None or not os.path.exists(r["png"]):
                d.rectangle([x + 1, y + 1, x + THUMB - 2, y + THUMB - 2], outline="#eeeeee")
                continue
            out.paste(Image.open(r["png"]).convert("RGB").resize((THUMB, THUMB)), (x, y))
            cap = f"f={r['f_mean']:.2f}"
            lam_true = r.get("lam_true")
            if lam_true is not None and abs(lam_true - lam) > 1e-9:
                # Drawn in this column, but measured somewhere else -- say so, in red, every time.
                d.text((x + 2, y + THUMB + 2), f"lam={lam_true:.4f}!", fill="#cc0000")
                d.text((x + 2, y + THUMB + 2 + 9), cap, fill="black")
                continue
            if r.get("hf_frac") == r.get("hf_frac") and r.get("hf_frac") is not None:
                cap += f" hf={r['hf_frac']:.4f}"
            if not is_base and r.get("cfg_norm_ratio") == r.get("cfg_norm_ratio"):
                cap += f" r={r['cfg_norm_ratio']:.3f}"
            d.text((x + 2, y + THUMB + 2), cap, fill="black")
        y += THUMB + CAP_H

    out.save(args.out)
    print(f"wrote {args.out}  {out.size}  ({len(cols)} lambda x {len(rows)} rows)")
    for slug, seed, w, pts, stamp, is_base in rows:
        tag = " (stored baseline)" if is_base else ""
        print(f"  {slug:10s} s{seed} w={w:<4g} {len(pts):2d} pts  gpu={stamp.get('gpu', '?')}{tag}")
    print("\nLOOK AT THE IMAGES. f/hf/||x|| do not predict recognizability (CLAUDE.md).")


if __name__ == "__main__":
    main()
