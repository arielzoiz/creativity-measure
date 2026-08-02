"""Joint figure + verdict for the two auto-R arms, from their saved JSONs. CPU only, seconds.

The arms run as separate Slurm jobs, so neither notebook can be sure the other has finished. Both call
`plot()` opportunistically at the end; this file is also runnable on the login node once both JSONs
exist:

    conda activate creativity-measure
    python compare_arms.py [dir]
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("random", "fps")
COLORS = {"random": "tab:blue", "fps": "tab:green"}
LABELS = {"random": "RandomRefs (i.i.d., unbiased)", "fps": "WeightedFPSRefs (coverage + Voronoi)"}


def load(out_dir: str) -> dict[str, dict]:
    res: dict[str, dict] = {}
    for arm in ARMS:
        path = os.path.join(out_dir, f"auto_r_results_{arm}.json")
        if os.path.exists(path):
            with open(path) as fh:
                res[arm] = json.load(fh)
    return res


def check_drift(res: dict[str, dict]) -> None:
    """Both arms must have measured the SAME f, or their tau values are not comparable.

    They import the same module, so this can only fire if the jobs ran different revisions of it --
    which is exactly the failure that would otherwise be invisible in the joint plot.
    """
    if len(res) < 2:
        return
    a, b = (res[arm]["config"] for arm in ARMS)
    drift = {k: (a[k], b[k]) for k in a if k in b and a[k] != b[k]}
    if drift:
        print("WARNING: the arms ran different configurations -- their tau curves are NOT comparable:")
        for k, (va, vb) in drift.items():
            print(f"  {k}: random={va!r}  fps={vb!r}")
    else:
        print(f"config check: both arms ran the same f "
              f"(N_GAMMA={a['n_gamma']}, NUM_EPS={a['num_eps']}, guidance={a['guidance']}, "
              f"seed={a['seed']}, probes={a['probe_size']}) -- tau values are directly comparable.")


def plot(out_dir: str | None = None) -> str | None:
    out_dir = out_dir or os.path.dirname(os.path.abspath(__file__))
    res = load(out_dir)
    if not res:
        print("no results files found; nothing to compare.")
        return None
    check_drift(res)

    tau_target = next(iter(res.values()))["config"]["tau_target"]
    probe_se = next(iter(res.values())).get("tau_probe_se", 0.06)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    all_r: set[int] = set()
    for arm, r in res.items():
        nested = r.get("nested_tau") or []
        all_r |= {int(R) for R, _ in nested}
        ax.plot([R for R, _ in nested], [t for _, t in nested], "o-", color=COLORS[arm],
                label=f"{LABELS[arm]} — nested, auto-R = {r['auto_R']}")
        indep = r.get("independent_tau")
        if indep:                       # only the Random arm has independent draws
            all_r |= {int(R) for R, _ in indep}
            ax.plot([R for R, _ in indep], [t for _, t in indep], "o--", color="tab:red",
                    label="RandomRefs — INDEPENDENT draws (unbiased)")
    ax.axhline(tau_target, color="crimson", ls="--", lw=1, label=f"tau_target = {tau_target}")
    ax.axvline(64, color="grey", ls=":", lw=1, label="R = 64 (strong-tilt sweeps)")
    if all_r:
        ax.fill_between([min(all_r), max(all_r)], 1 - probe_se, 1, color="grey", alpha=0.12,
                        label=f"within tau SE ({probe_se:.3f}) of 1")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(all_r))
        ax.set_xticklabels([str(r) for r in sorted(all_r)])
    ax.set_xlabel("R"); ax.set_ylabel(r"weighted-$\tau$")
    ax.set_title("Auto-R: rank stability of $f$ by selector (same $f$, same probe points)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "auto_r_tau_compare.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")

    # --- verdict ------------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"{'selector':>16} {'auto-R':>7} {'ceiling':>8} {'min':>7}   tau by R (nested)")
    for arm, r in res.items():
        hist = " ".join(f"{R}:{t:.3f}" for R, t in (r.get("nested_tau") or []))
        print(f"{arm:>16} {r['auto_R']:7d} {r['ceiling']:8.4f} {r['minutes']:7.1f}   {hist}")
    rand = res.get("random", {})
    if rand.get("independent_tau"):
        print(f"\n{'':>16} {'':>7} {'':>8} {'':>7}   "
              + " ".join(f"{R}:{t:.3f}" for R, t in rand["independent_tau"]) + "   <- independent")
    shared = sorted({int(R) for R, _ in (res.get("fps", {}).get("nested_tau") or [])}
                    & {int(R) for R, _ in (rand.get("nested_tau") or [])})
    if shared:
        f_t = dict(res["fps"]["nested_tau"])
        r_t = dict(rand["nested_tau"])
        print("\nhead-to-head where the ladders overlap (both nested, so both optimistic):")
        for R in shared:
            gap = f_t[R] - r_t[R]
            verdict = ("tie" if abs(gap) < probe_se else
                       "FPS better" if gap > 0 else "Random better")
            print(f"  R={R:4d}  FPS {f_t[R]:.4f}  Random {r_t[R]:.4f}  gap {gap:+.4f}  -> {verdict}")
        if "R_eff" in res["fps"]:
            print(f"\n  NOTE: FPS's weights give R_eff = {res['fps']['R_eff']:.1f} effective "
                  f"references at R = {res['fps']['auto_R']};")
            print("  compare its curve against Random at R_eff, not at R, before declaring a winner.")
    print("=" * 92)
    return path


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else None)
