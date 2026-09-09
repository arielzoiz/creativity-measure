"""Figures for the K sweep at fixed lambda. Run from this directory, no arguments.

Reads every result_*.pt / partial_*.pt on disk, so it works on a half-finished sweep.

The control (lambda = 0) is not decoration: it established that the within-run rise-and-fall of
E_q[f] is a PROJECTION ARTIFACT -- f evaluated on map(x_t, t, 1) scores intermediate-t estimates
higher than the sharp samples they become, and that shape is present with no tilt, no resampling
and an intact cloud. So the raw curves are plotted next to the control-subtracted ones, and only
the latter say anything about the sampler.
"""
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SD = 0.0071                      # std_p(f), the natural unit for every difference here
RUN_DIR = os.path.dirname(os.path.abspath(__file__))
COL = {4: "#d62728", 8: "#2ca02c", 16: "#1f77b4", 32: "#9467bd"}


def load():
    """Every cell on disk; a finished result supersedes its own partial."""
    ctrl, cells, seen = None, [], set()
    for fp in sorted(glob.glob(os.path.join(RUN_DIR, "result_*.pt"))) + \
            sorted(glob.glob(os.path.join(RUN_DIR, "partial_*.pt"))):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        c = d["config"]
        key = (c["K"], c["m_tilt"])
        if key in seen:
            continue
        seen.add(key)
        d["_partial"] = os.path.basename(fp).startswith("partial_")
        if c["m_tilt"] == 0:
            ctrl = d
        else:
            cells.append(d)
    return ctrl, sorted(cells, key=lambda d: d["config"]["K"])


def collapse_step(uniq, m):
    return next((i + 1 for i, u in enumerate(uniq) if u <= 1.0 / m + 1e-9), None)


def v_at(r_k, lam, cols):
    sub = r_k[:, cols]
    return torch.logsumexp(lam * sub, dim=1) - math.log(sub.shape[1])


def sd_ratio(d, k_prime, n_draws=8):
    """Mean over steps of sd(U_n)/sd(V_n), V rebuilt from a K'-column subset of r_k.

    The previous level is mapped through that step's parent map before differencing -- `ancestors`
    composes every resampling and cannot be inverted, which is why `resample_idx` is recorded.
    """
    cfg, rks = d["config"], d.get("r_k_history") or []
    lam, idxs = cfg["lam"], d.get("resample_idx_history") or []
    gen = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n_draws):
        levels = []
        for rk in rks:
            if rk is None:
                levels.append(None)
                continue
            if k_prime > rk.shape[1]:
                return float("nan")
            cols = torch.randperm(rk.shape[1], generator=gen)[:k_prime]
            levels.append(v_at(rk.double(), lam, cols))
        per = []
        for n in range(1, len(levels)):
            cur, prev = levels[n], levels[n - 1]
            if cur is None or prev is None:
                continue
            pidx = idxs[n - 1] if n - 1 < len(idxs) else None
            if pidx is not None:
                prev = prev[pidx]
            if float(cur.std()) > 1e-12:
                per.append(float((cur - prev).std()) / float(cur.std()))
        if per:
            out.append(sum(per) / len(per))
    return sum(out) / len(out) if out else float("nan")


ctrl, cells = load()
if ctrl is None:
    raise SystemExit("no lambda=0 control on disk -- every figure here is control-subtracted")
ce = ctrl["eqf_history"]


def lab(d):
    c = d["config"]
    return f"K={c['K']}" + ("  [partial]" if d["_partial"] else "")


# --- FIGURE 1: raw vs control-subtracted ------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))
ax1.plot(range(1, len(ce) + 1), ce, "--", color="0.35", lw=2, marker="o", ms=4,
         label="control  $\\lambda=0$", zorder=5)
for d in cells:
    c, e, u = d["config"], d["eqf_history"], d["uniq_history"]
    xs = range(1, len(e) + 1)
    ax1.plot(xs, e, "-", color=COL[c["K"]], lw=1.8, marker="o", ms=4, label=lab(d))
    diff = [(a - b) / SD for a, b in zip(e, ce)]
    ax2.plot(xs, diff, "-", color=COL[c["K"]], lw=1.8, marker="o", ms=4, label=lab(d))
    cs = collapse_step(u, c["M"])
    if cs:
        for ax, ys in ((ax1, e), (ax2, diff)):
            ax.plot(cs, ys[cs - 1], "x", ms=13, mew=2.5, color=COL[c["K"]], zorder=6)

ax1.set_title("Raw $E_q[f]$ per step\n(the rise-and-fall is mostly a projection artifact)")
ax1.set_ylabel("$E_q[f]$")
ax2.axhline(0, color="0.35", ls="--", lw=2)
ax2.set_title("Control-subtracted: the actual tilt effect\n"
              "(x = collapse to a single lineage)")
ax2.set_ylabel("$(E_q[f] - E_{\\lambda=0}[f])\\ /\\ \\mathrm{std}_p(f)$")
for ax in (ax1, ax2):
    ax.set_xlabel("step $n$   ($t = n/16$)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "fig1_eqf_overlay.png"), dpi=150)

# --- FIGURE 2: small multiples, one panel per K (as originally requested) ----------------------
n = len(cells)
fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), sharey=True)
axes = [axes] if n == 1 else list(axes)
for ax, d in zip(axes, cells):
    c, e, u = d["config"], d["eqf_history"], d["uniq_history"]
    xs = list(range(1, len(e) + 1))
    ax.plot(range(1, len(ce) + 1), ce, "--", color="0.55", lw=1.6, label="$\\lambda=0$ control")
    ax.plot(xs, e, "-", color=COL[c["K"]], lw=2, marker="o", ms=4.5, label=f"K={c['K']}")
    for i, r in enumerate(d["resampled_history"][:len(e)]):
        if r:
            ax.plot(i + 1, e[i], "v", ms=8, color=COL[c["K"]])
    cs = collapse_step(u, c["M"])
    if cs:
        ax.axvline(cs, color=COL[c["K"]], ls=":", lw=1.6)
        ax.text(cs + 0.15, ax.get_ylim()[0], " collapse", fontsize=8, color=COL[c["K"]])
    ax.set_title(f"K = {c['K']}" + ("  [partial]" if d["_partial"] else "")
                 + f"\nfinal uniq/M = {u[-1]:.3f}, {int(sum(d['resampled_history']))} resamples")
    ax.set_xlabel("step $n$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
axes[0].set_ylabel("$E_q[f]$")
fig.suptitle("$E_q[f]$ within a single run, by lookahead depth $K$   "
             "($\\lambda=175.1$, $M=8$, seed 101)   ▼ = resampled", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "fig2_eqf_per_K.png"), dpi=150, bbox_inches="tight")

# --- FIGURE 3: the mechanism -- sd(U)/sd(V) vs K' -----------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 5.4))
for d in cells:
    c = d["config"]
    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= c["K"]]
    ys = [sd_ratio(d, k) for k in ks]
    ax.plot(ks, ys, "-o", color=COL[c["K"]], lw=1.9, ms=6, label=f"cell K={c['K']}")
    if ys and ys[0] == ys[0]:
        ax.plot(ks, [ys[0] / math.sqrt(k) for k in ks], ":", color=COL[c["K"]], lw=1.2, alpha=0.7)
ax.set_xscale("log", base=2)
ax.set_xlabel("$K'$  (offline subset of the recorded $r_k$)")
ax.set_ylabel("$\\mathrm{sd}(U_n)\\ /\\ \\mathrm{sd}(V_n)$")
ax.set_title("Mechanism: the increment the resampler consumes\n"
             "solid = measured, dotted = $1/\\sqrt{K'}$ if it were pure estimator noise")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "fig3_mechanism.png"), dpi=150)

# --- FIGURE 4: degeneracy ----------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
for d in cells:
    c = d["config"]
    xs = range(1, len(d["uniq_history"]) + 1)
    a1.plot(xs, d["uniq_history"], "-o", color=COL[c["K"]], ms=4, label=lab(d))
    a2.plot(xs, [e / c["M"] for e in d["ess_history"]], "-o", color=COL[c["K"]], ms=4, label=lab(d))
a1.plot(range(1, len(ctrl["uniq_history"]) + 1), ctrl["uniq_history"], "--",
        color="0.35", lw=2, label="control")
a2.axhline(0.5, color="k", ls="--", lw=1.2, label="resample threshold")
a1.set_ylabel("uniq/M  (surviving lineages)"); a2.set_ylabel("ESS/M")
a1.set_title("Ensemble diversity"); a2.set_title("Effective sample size")
for ax in (a1, a2):
    ax.set_xlabel("step $n$"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "fig4_degeneracy.png"), dpi=150)

# --- console summary ---------------------------------------------------------------------------
print(f"control: peak {max(ce):.4f}@{ce.index(max(ce)) + 1}  terminal {ce[-1]:.4f}  "
      f"give-back {max(ce) - ce[-1]:+.4f} = {(max(ce) - ce[-1]) / SD:.2f}sd  (ARTIFACT baseline)")
print(f"\n{'K':>4} {'steps':>6} {'terminal':>9} {'tilt effect':>13} {'uniq/M':>7} "
      f"{'resamp':>7} {'collapse':>9} {'sd(U)/sd(V)':>12}")
for d in cells:
    c, e, u = d["config"], d["eqf_history"], d["uniq_history"]
    te = (e[-1] - ce[len(e) - 1]) / SD
    print(f"{c['K']:>4} {len(e):>6} {e[-1]:>9.4f} {te:>+11.2f}sd {u[-1]:>7.3f} "
          f"{int(sum(d['resampled_history'])):>7} {str(collapse_step(u, c['M'])):>9} "
          f"{sd_ratio(d, c['K']):>12.3f}")
print("\nwrote fig1_eqf_overlay.png  fig2_eqf_per_K.png  fig3_mechanism.png  fig4_degeneracy.png")
