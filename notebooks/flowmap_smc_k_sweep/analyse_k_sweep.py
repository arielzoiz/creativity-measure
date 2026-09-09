"""Cross-cell analysis for the K sweep: does K reduce the increment the resampler consumes?

Run with no arguments from `notebooks/flowmap_smc_k_sweep/`. Reads every result_*.pt and
partial_*.pt in the directory, so it works on a half-finished sweep and on preempted jobs -- the
pre-collapse window is the deliverable and it is early, so partials are first-class here.

THE DECISIVE NUMBER is sd(U_n) / sd(V_n) as a function of K.
  * V_n is the LEVEL of the twist -- it carries real between-particle signal.
  * U_n is the per-step INCREMENT, and that is what `_resample` actually selects on.
If the increment were dominated by Monte-Carlo noise in the K-sample estimate, the ratio would
fall like 1/sqrt(K). If it is dominated by genuine step-to-step movement of the true V, K cannot
help and the fix is common random numbers across steps (CRN), not more lookahead draws.
On the d=2 toy the ratio fell by 1.22x across a 64x range of K, where noise-domination predicts 8x.

OFFLINE K' RECONSTRUCTION. Each cell stores r_k as (M, K) per guided step, so V(K') for any
K' <= K is a subset average of numbers already computed -- one run at K answers for every smaller
K'. This is estimator-noise-at-a-fixed-cloud, NOT a simulation of a real K' run (the trajectory was
driven by the full K), so it is labelled as such and cross-checked against the real K=4 / K=8 cells.

ALIGNMENT ACROSS RESAMPLING is why `resample_idx_history` exists. V_n is recorded in pre-resample
order while the cloud carried into step n+1 is post-resample, so a per-particle increment must map
the previous level through that step's parent map. Skipping this silently mixes lineages -- and
`ancestors` cannot substitute, since it composes every resampling so far and is not invertible.
"""
import glob
import math
import os

import torch

RUN_DIR = os.path.dirname(os.path.abspath(__file__))


def load_cells() -> list[dict]:
    """Every cell on disk, finished results superseding their own partials."""
    cells, seen = [], set()
    for fp in sorted(glob.glob(os.path.join(RUN_DIR, "result_*.pt"))) + \
            sorted(glob.glob(os.path.join(RUN_DIR, "partial_*.pt"))):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        c = d["config"]
        key = (c["K"], c["m_tilt"], c["M"])
        if key in seen:
            continue
        seen.add(key)
        d["_file"] = os.path.basename(fp)
        cells.append(d)
    return cells


def v_at(r_k: torch.Tensor, lam: float, cols: torch.Tensor) -> torch.Tensor:
    """V = log mean_k exp(lam * r_k) over a COLUMN SUBSET -- the K' reconstruction.

    logsumexp, never a running sum of exp: lam*f reaches ~3 at the calibrated lambda.
    """
    sub = r_k[:, cols]
    return torch.logsumexp(lam * sub, dim=1) - math.log(sub.shape[1])


def sd_ratio(d: dict, k_prime: int | None = None, n_draws: int = 8) -> tuple[float, int]:
    """Mean over steps of sd(U_n)/sd(V_n); U_n aligned through the parent map.

    ``k_prime=None`` uses the run's own recorded V (the honest, as-run value). Otherwise V is
    rebuilt from a random K'-column subset of r_k, averaged over ``n_draws`` subsets.
    """
    cfg, rks = d["config"], d.get("r_k_history") or []
    lam, idxs = cfg["lam"], d.get("resample_idx_history") or []
    gen = torch.Generator().manual_seed(0)
    ratios: list[float] = []

    for _ in range(n_draws if k_prime else 1):
        levels: list[torch.Tensor | None] = []
        for rk in rks:
            if rk is None:
                levels.append(None)
                continue
            if k_prime is None:
                cols = torch.arange(rk.shape[1])
            else:
                if k_prime > rk.shape[1]:
                    return float("nan"), 0
                cols = torch.randperm(rk.shape[1], generator=gen)[:k_prime]
            levels.append(v_at(rk.double(), lam, cols))

        per_step: list[float] = []
        for n in range(1, len(levels)):
            cur, prev = levels[n], levels[n - 1]
            if cur is None or prev is None:
                continue
            # The cloud entering step n is step n-1's POST-resample cloud; `prev` is pre-resample.
            pidx = idxs[n - 1] if n - 1 < len(idxs) else None
            if pidx is not None:
                prev = prev[pidx]
            sv = float(cur.std())
            if sv > 1e-12:
                per_step.append(float((cur - prev).std()) / sv)
        if per_step:
            ratios.append(sum(per_step) / len(per_step))

    if not ratios:
        return float("nan"), 0
    return sum(ratios) / len(ratios), len(ratios)


def collapse_step(uniq: list[float], m: int) -> int | None:
    for i, u in enumerate(uniq):
        if u <= 1.0 / m + 1e-9:
            return i + 1
    return None


def main() -> None:
    cells = load_cells()
    if not cells:
        print(f"no result_*.pt or partial_*.pt in {RUN_DIR} yet")
        return

    gpus = {c["config"].get("gpu_name", "?") for c in cells}
    if len(gpus) > 1:
        print(f"*** WARNING: cells span more than one GPU model {gpus}. A GPU change shifted f by "
              f"16% of std_p(f) once (job 697271) -- the same order as the effects here. The K "
              f"comparison across these is NOT valid.\n")

    print("=== per-cell summary " + "=" * 72)
    print(f"{'K':>4} {'M':>4} {'lam':>7} {'steps':>6} {'peak':>8} {'@':>3} {'terminal':>9} "
          f"{'give-back':>10} {'collapse':>9} {'resamp':>7} {'uniq/M':>7} {'sd(U)/sd(V)':>12}  file")
    for d in sorted(cells, key=lambda x: (x["config"]["m_tilt"], x["config"]["K"])):
        c, eqf = d["config"], d["eqf_history"]
        valid = [v for v in eqf if v == v]
        pk = max(valid) if valid else float("nan")
        r, _ = sd_ratio(d)
        print(f"{c['K']:>4} {c['M']:>4} {c['lam']:>7.1f} {len(eqf):>6} {pk:>8.4f} "
              f"{(eqf.index(pk) + 1) if valid else 0:>3} {eqf[-1]:>9.4f} "
              f"{pk - eqf[-1]:>+10.4f} {str(collapse_step(d['uniq_history'], c['M'])):>9} "
              f"{int(sum(d['resampled_history'])):>7} {d['uniq_history'][-1]:>7.3f} "
              f"{r:>12.3f}  {d['_file']}")

    print("\n=== sd(U)/sd(V) vs K' (offline subsets; noise-domination predicts 1/sqrt(K')) " + "=" * 5)
    print("    NOTE: estimator noise at a FIXED cloud, not a simulation of a real K' run.")
    for d in sorted(cells, key=lambda x: x["config"]["K"]):
        c = d["config"]
        if not (d.get("r_k_history") and any(r is not None for r in d["r_k_history"])):
            continue
        row: list[str] = []
        base = last = float("nan")
        for kp in (1, 2, 4, 8, 16, 32):
            if kp > c["K"]:
                break
            last, _ = sd_ratio(d, k_prime=kp)
            base = last if base != base else base       # first finite value wins
            row.append(f"K'={kp}: {last:.3f}")
        obs = (base / last) if last > 0 else float("nan")
        pred = math.sqrt(c["K"] / 1.0)
        print(f"  cell K={c['K']:<3} " + "  ".join(row))
        print(f"           observed drop 1->{c['K']}: {obs:.2f}x    "
              f"1/sqrt(K) predicts: {pred:.2f}x    "
              f"-> {'K IS the lever' if obs > 0.6 * pred else 'K is NOT the lever; CRN next'}")


if __name__ == "__main__":
    main()
