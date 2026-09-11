"""Would the PLAIN MEAN aggregation ``V = lam * mean_k r_k`` fix what soft/max could not?

Offline on the recorded ``r_k``, no GPU. Three aggregations of the same K lookahead candidates:

    mean :  V = lam * mean_k r_k                      -- tau -> 0, NO Jensen at all
    soft :  V = log mean_k exp(lam r_k)               -- tau = 1, today's `flowmap_smc`
    max  :  V = max_k lam r_k                         -- tau -> inf, `flowmap_smc_max` (tested, null)

**Why this is not a repeat of the max experiment.** The max was null because it *rescales the same
ordering* -- Spearman 0.93-0.97 against soft, same argmax at 7 of 8 steps. The mean does something
different: it **removes a term**. Writing the soft value's expansion
``V/lam = mean_k r + lam*Var_k(r)/2``, the second term is the Jensen convexity measured in
`A0_RESULTS.md`, and its ACROSS-PARTICLE spread is ~40% of ``sd(U)`` at early t. Deleting it changes
what resampling sees by that much.

**Three things it could buy, all measurable here.**

1. **``K`` starts working again.** A0 found ``sigma_eps^2 ~ k^-p`` with ``p = 0.17-0.82``, the
   signature of a max-dominated statistic. The mean is a linear statistic of independent draws, so it
   must give ``p = 1`` -- and if it does, every conclusion about ``K`` saturating was a property of the
   aggregation, not of the lookahead.
2. **No Jensen exposure.** Both the convexity (A) and the ``(e^{a^2}-1)/2K`` estimator bias (B) vanish.
   This is the one aggregation whose behaviour does not degrade as the tilt ``m`` rises -- see
   `A0_RESULTS.md` part 3, where ``a(t->0) = m`` makes soft/max collapse to a max at strong tilt.
3. **A cleaner increment**, if the convexity term was adding noise rather than signal. That is exactly
   what question 2 below asks.

**It stays a valid sampler.** Every intermediate ``V`` is a twist: the potentials telescope and the
target is pinned entirely by the mandatory terminal ``V_N = lam*f(x_1)``, which runs at ``K = 1`` on a
separate code path. Any aggregation leaves ``q_lambda`` exactly (`flowmap_smc_max` module docstring).
The mean is the *first-order* twist -- it guides toward a high expected reward rather than a high soft
maximum -- which is a different, defensible reading of what the lookahead should reward.

**QUESTION 2, same data: is resampling selecting on the mean or on the variance?**
``V_soft/lam = mu_k + lam*sigma_k^2/2``. Correlating each part separately with ``f`` at ``t = 1`` says
whether the potential is backing particles with a good expected future or merely an *uncertain* one.
Under the target the variance term belongs there; but at ``K = 4`` it is estimated terribly (that is
what a small ``p`` means), so selecting on it is selecting on noise.

Usage::

    conda activate creativity-measure
    python notebooks/flowmap_smc_replay/a1_mean_aggregation.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a0_increment_decomposition import (  # noqa: E402
    _subsets, decay_exponent, load_cells,
)

AGGS = ("mean", "soft", "max")


def agg_v(r_k: Tensor, lam: float, how: str, cols: Tensor | None = None) -> Tensor:
    """One aggregation of ``(M, K)`` rewards into ``(M,)`` potentials, over an optional column subset."""
    sub = r_k if cols is None else r_k[:, cols]
    if how == "mean":
        return lam * sub.mean(dim=1)
    if how == "soft":
        return torch.logsumexp(lam * sub, dim=1) - math.log(sub.shape[1])
    if how == "max":
        return (lam * sub).max(dim=1).values
    raise ValueError(how)


def _align(levels: list[Tensor | None], idxs: list[Tensor | None], n: int) -> tuple[Tensor, Tensor]:
    """``(V_n, V_{n-1} mapped through step n-1's parent map)`` -- pre/post-resample orders differ."""
    cur, prev = levels[n], levels[n - 1]
    assert cur is not None and prev is not None
    pidx = idxs[n - 1] if n - 1 < len(idxs) else None
    return cur, (prev[pidx] if pidx is not None else prev)


def metrics(cell: dict, how: str) -> dict[str, float]:
    """``sd(U)/sd(V)``, the ``k^-p`` exponent, and both correlations with ``f`` at ``t = 1``."""
    cfg = cell["config"]
    lam, k_full = float(cfg["lam"]), int(cfg["K"])
    rks = cell.get("r_k_history") or []
    idxs = cell.get("resample_idx_history") or []
    f_proj = cell.get("f_proj_history") or []
    f_final = f_proj[-1].double() if f_proj and f_proj[-1] is not None else None
    n_last = len(f_proj) - 1

    levels = [None if rk is None else agg_v(rk.double(), lam, how) for rk in rks]

    ratios, cv, cu = [], [], []
    for n in range(1, len(levels)):
        if levels[n] is None or levels[n - 1] is None:
            continue
        cur, prev = _align(levels, idxs, n)
        if cur.numel() < 2 or float(cur.std()) < 1e-12:
            continue
        ratios.append(float((cur - prev).std(unbiased=True)) / float(cur.std(unbiased=True)))

        if f_final is not None and n < n_last:
            a = torch.arange(f_final.numel())
            for j in range(n_last, n, -1):                 # walk finals back to their step-n ancestor
                if j < len(idxs) and idxs[j] is not None:
                    a = idxs[j][a]
            a_pre = idxs[n][a] if (n < len(idxs) and idxs[n] is not None) else a
            v_n, u_n = cur[a_pre], (cur - prev)[a_pre]
            for dst, x in ((cv, v_n), (cu, u_n)):
                if x.std() > 1e-12 and f_final.std() > 1e-12:
                    dst.append(float(torch.corrcoef(torch.stack([x, f_final]))[0, 1]))

    # sigma_eps^2 vs subset size, pair-respecting, for THIS aggregation -> the k^-p exponent.
    pts: list[tuple[int, float]] = []
    for kt in (1, 2, 4, 8, 16):
        if k_full // kt < 2:
            break
        gen = torch.Generator().manual_seed(0)
        vals = []
        for rk in rks:
            if rk is None:
                continue
            subs = _subsets(k_full, kt, gen)
            vs = torch.stack([agg_v(rk.double(), lam, how, cols) for cols in subs])
            vals.append(float(vs.var(dim=0, unbiased=True).mean()))
        if vals:
            pts.append((kt, sum(vals) / len(vals)))

    def _m(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return {"sd_ratio": _m(ratios), "p": decay_exponent(pts), "corr_v": _m(cv), "corr_u": _m(cu),
            "sig2_at4": next((v for k, v in pts if k == 4), float("nan"))}


def ordering_vs_soft(cell: dict) -> tuple[float, float]:
    """Mean Spearman of ``mean`` and of ``max`` against ``soft``, over guided steps.

    The max scored 0.93-0.97 here and produced nothing. If the mean scores similarly, it is the same
    experiment again and should not be run; if it is materially lower, it is a different twist.
    """
    cfg = cell["config"]
    lam = float(cfg["lam"])
    out: dict[str, list[float]] = {"mean": [], "max": []}
    for rk in (cell.get("r_k_history") or []):
        if rk is None:
            continue
        soft = agg_v(rk.double(), lam, "soft")
        for how in ("mean", "max"):
            other = agg_v(rk.double(), lam, how)
            ra = soft.argsort().argsort().double()
            rb = other.argsort().argsort().double()
            if ra.std() > 1e-12 and rb.std() > 1e-12:
                out[how].append(float(torch.corrcoef(torch.stack([ra, rb]))[0, 1]))
    return (sum(out["mean"]) / len(out["mean"]) if out["mean"] else float("nan"),
            sum(out["max"]) / len(out["max"]) if out["max"] else float("nan"))


def term_split(cell: dict) -> tuple[float, float, float]:
    """``corr(mu_k, f_final)``, ``corr(sigma_k^2, f_final)``, and the variance term's share of sd(V).

    ``V_soft/lam = mu_k + lam*sigma_k^2/2``. If the second correlates with the endpoint about as well
    as the first, the convexity term is signal; if it correlates near zero while carrying a real share
    of the spread, resampling is partly **variance-seeking** -- backing uncertain futures over good ones.
    """
    cfg = cell["config"]
    lam = float(cfg["lam"])
    rks = cell.get("r_k_history") or []
    idxs = cell.get("resample_idx_history") or []
    f_proj = cell.get("f_proj_history") or []
    if not f_proj or f_proj[-1] is None:
        return float("nan"), float("nan"), float("nan")
    f_final = f_proj[-1].double()
    n_last = len(f_proj) - 1

    cm, cs, share = [], [], []
    for n in range(1, min(len(rks), n_last)):
        rk = rks[n]
        if rk is None:
            continue
        rk = rk.double()
        mu, var = rk.mean(dim=1), rk.var(dim=1, unbiased=True)
        conv = lam * var / 2.0
        a = torch.arange(f_final.numel())
        for j in range(n_last, n, -1):
            if j < len(idxs) and idxs[j] is not None:
                a = idxs[j][a]
        a_pre = idxs[n][a] if (n < len(idxs) and idxs[n] is not None) else a
        for dst, x in ((cm, mu[a_pre]), (cs, conv[a_pre])):
            if x.std() > 1e-12:
                dst.append(float(torch.corrcoef(torch.stack([x, f_final]))[0, 1]))
        if float((mu + conv).std()) > 1e-12:
            share.append(float(conv.std()) / float((mu + conv).std()))

    def _m(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return _m(cm), _m(cs), _m(share)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    cells = [c for c in load_cells() if int(c["config"]["K"]) >= 4]
    print(f"{len(cells)} cells with r_k\n")

    print("=" * 108)
    print("Q1  per-aggregation mechanism metrics (all offline from the SAME recorded r_k)")
    print(f"{'cell':<30} {'agg':>5} {'sd(U)/sd(V)':>11} {'p in k^-p':>10} "
          f"{'corr(V,f_fin)':>13} {'corr(U,f_fin)':>13}")
    for cell in cells:
        for how in AGGS:
            m = metrics(cell, how)
            mark = "  <-- today" if how == "soft" else ""
            print(f"{cell['_file'][-30:]:<30} {how:>5} {m['sd_ratio']:>11.3f} {m['p']:>10.2f} "
                  f"{m['corr_v']:>13.3f} {m['corr_u']:>13.3f}{mark}")
        print()

    print("=" * 108)
    print("Q1b does the mean REORDER particles, or merely rescale like the max did?")
    print(f"  {'cell':<30} {'spearman(mean,soft)':>20} {'spearman(max,soft)':>19}")
    for cell in cells:
        sm, sx = ordering_vs_soft(cell)
        print(f"  {cell['_file'][-30:]:<30} {sm:>20.3f} {sx:>19.3f}")

    print("\n" + "=" * 108)
    print("Q2  V_soft/lam = mu_k + lam*sigma_k^2/2 -- which term does selection ride on?")
    print(f"  {'cell':<30} {'corr(mu,f_fin)':>15} {'corr(conv,f_fin)':>17} {'conv share of sd(V)':>20}")
    for cell in cells:
        cm, cs, sh = term_split(cell)
        print(f"  {cell['_file'][-30:]:<30} {cm:>15.3f} {cs:>17.3f} {sh:>20.3f}")


if __name__ == "__main__":
    main()
