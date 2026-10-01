"""Step A0 -- is the increment ``U_n`` estimator noise (CRN can cancel it) or real ``dV`` movement?

The gate for the whole CRN plan (`~/claude-config/plans/encapsulated-stargazing-rabin.md`, Fix 2),
run BEFORE any sampler code. No GPU: it reads the ``r_k`` dumps that `notebooks/flowmap_smc_k_sweep/`
and `notebooks/flowmap_smc_max/` already wrote, and re-runs the same decomposition on the analytic
2-D toy where a large ``K`` gives the answer directly.

THE QUESTION. Resampling consumes ``U_n = V_n - V_{n-1}``, and ``V_n = V_n_true + eps_n`` where
``eps`` is the Monte-Carlo error of the K-candidate estimate. Across particles::

    Var(U_n) = Var(dV_true) + sigma_eps(n)^2 + sigma_eps(n-1)^2

CRN acts ONLY on the two ``sigma_eps`` terms -- it correlates consecutive errors so they cancel in
the difference. It cannot touch ``Var(dV_true)``: nothing cancels real signal. So the decision is a
single number, the **noise share** ``(sigma_n^2 + sigma_{n-1}^2) / Var(U_n)``:

    > 0.5  -- residual dominates, CRN has something to cancel            -> PROCEED
    < 0.5  -- dV_true dominates; CRN is not weak here, it is INAPPLICABLE -> STOP, spend on M

This is the same assumption raising ``K`` made (``K`` also attacks only ``sigma_eps``), and ``K``
measurably stopped helping at ~8 -- which is evidence against the assumption, not for it.

HOW ``sigma_eps`` IS MEASURED, without assuming a form. Each cell stores ``r_k`` as ``(M, K)`` per
guided step. Split the ``K`` columns into ``n_sub`` DISJOINT subsets of size ``k_target`` and compute
one ``V`` per subset: their spread across subsets *is* ``sigma_eps(k_target)``, measured, no jackknife
and no delta-method approximation of a nonlinear ``logsumexp``. The ``K = 32`` cell gives 8 disjoint
subsets of 4 -- a direct read of the error at the ``K = 4`` the plan actually runs at. Monte-Carlo
variance scales as ``1/K``, which is what carries it between column counts, and that scaling is
CHECKED here rather than assumed (`--check`).

ALIGNMENT ACROSS RESAMPLING. ``V_n`` is recorded pre-resample while the cloud entering step ``n+1``
is post-resample, so a per-particle increment must map the previous level through that step's parent
map (`resample_idx_history`). `ancestors` cannot substitute -- it composes every resampling so far.
Same rule as `notebooks/flowmap_smc_k_sweep/analyse_k_sweep.py`.

Usage::

    conda activate creativity-measure
    python notebooks/flowmap_smc_replay/a0_increment_decomposition.py            # FLUX dumps + toy
    python notebooks/flowmap_smc_replay/a0_increment_decomposition.py --no-toy   # dumps only
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from dataclasses import dataclass

import torch
from torch import Tensor

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(RUN_DIR, "..", ".."))
DUMP_DIRS = [
    os.path.join(REPO, "notebooks", "flowmap_smc_k_sweep"),
    os.path.join(REPO, "notebooks", "flowmap_smc_max"),
]

# The plan's configuration: the run this gate is deciding for is K = 4.
K_TARGET = 4
DECISION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------------------------------

@dataclass
class StepRow:
    """One step's decomposition, all variances across the ``M`` particles."""

    n: int
    var_u: float          # Var_M(U_n) as the run actually saw it, at the cell's full K
    var_dv: float         # Var_M(dV_true), the part CRN cannot touch
    sig2_now: float       # sigma_eps^2 at full K, this step
    sig2_prev: float      # sigma_eps^2 at full K, previous step
    sd_v: float           # sd_M(V_n) -- the denominator of the mechanism metric
    noise_share_full: float   # at the cell's own K
    noise_share_target: float  # rescaled to K_TARGET, which is what the plan runs


def _v(r_k: Tensor, lam: float, cols: Tensor) -> Tensor:
    """``V = log mean_k exp(lam r_k)`` over a column subset -- always logsumexp, never sum-of-exp."""
    sub = r_k[:, cols]
    return torch.logsumexp(lam * sub, dim=1) - math.log(sub.shape[1])


def _subsets(k: int, k_target: int, gen: torch.Generator) -> list[Tensor]:
    """Disjoint column subsets of size ``k_target``, **respecting the antithetic pairing**.

    Disjointness is what makes the spread across subsets pure estimator error -- they share no
    candidate, so nothing common inflates their agreement.

    The pairing is not a detail. `_antithetic_noise` lays the columns out as
    ``[e0, -e0, e1, -e1, ...]``, so a subset that splits a pair is an estimator NO REAL RUN USES:
    it loses the antithetic variance reduction, and mixing split and intact subsets inflates the
    between-subset spread on top of that. Permuting *pair blocks* rather than columns makes a
    ``k_target = 4`` subset exactly the four draws a real ``K = 4`` run would take. Odd ``k_target``
    cannot respect pairs and falls back to a plain column permutation (only ``k' = 1`` in practice,
    where there is no pair to keep).
    """
    if k_target % 2 or k % 2:
        perm = torch.randperm(k, generator=gen)
        return [perm[s * k_target:(s + 1) * k_target] for s in range(k // k_target)]
    pairs = torch.arange(k).view(-1, 2)                       # [[0,1],[2,3],...]
    pairs = pairs[torch.randperm(pairs.shape[0], generator=gen)]
    per_sub = k_target // 2
    n_sub = pairs.shape[0] // per_sub
    return [pairs[s * per_sub:(s + 1) * per_sub].reshape(-1) for s in range(n_sub)]


def _sigma2_by_disjoint_subsets(
    r_k: Tensor, lam: float, k_target: int, gen: torch.Generator
) -> float:
    """Measured ``sigma_eps^2`` at ``k_target`` columns: the variance of V across disjoint subsets.

    Averaged over particles; ``nan`` if fewer than two subsets fit, which is why a ``K = 4`` cell
    cannot measure its own error and needs ``K >= 8``.
    """
    subs = _subsets(r_k.shape[1], k_target, gen)
    if len(subs) < 2:
        return float("nan")
    vs = torch.stack([_v(r_k, lam, cols) for cols in subs])
    return float(vs.var(dim=0, unbiased=True).mean())


def sigma2_curve(cell: dict, seed: int = 0) -> list[tuple[int, float]]:
    """``sigma_eps^2`` at every subset size that fits, pair-respecting. Powers the fitted exponent."""
    cfg = cell["config"]
    lam, k_full = float(cfg["lam"]), int(cfg["K"])
    out: list[tuple[int, float]] = []
    for kt in (1, 2, 4, 8, 16):
        if k_full // kt < 2:
            break
        gen = torch.Generator().manual_seed(seed)
        vals = [_sigma2_by_disjoint_subsets(rk.double(), lam, kt, gen)
                for rk in (cell.get("r_k_history") or []) if rk is not None]
        vals = [v for v in vals if v == v]
        if vals:
            out.append((kt, sum(vals) / len(vals)))
    return out


def decay_exponent(pts: list[tuple[int, float]]) -> float:
    """Least-squares ``p`` in ``sigma^2(k) ~ k^-p``, over the pair-respecting points ``k >= 2``.

    **Do not assume ``p = 1``.** Independent draws through a *linear* estimator would give it, but
    ``V`` is a ``logsumexp`` and the draws are antithetically paired, and the measured ``p`` is well
    below 1 on every FLUX cell. That is not a nuisance -- it is the direct explanation of why raising
    ``K`` saturated (`RESULTS.md`: the ``sd(U)/sd(V)`` drop stalls at ~1.5x where ``1/sqrt(K)``
    predicts 5.66x). ``k = 1`` is excluded: it is the one size with no pair to keep, so it sits on a
    different estimator.
    """
    pts = [(k, v) for k, v in pts if k >= 2 and v > 0]
    if len(pts) < 2:
        return 1.0
    xs = [math.log(k) for k, _ in pts]
    ys = [math.log(v) for _, v in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else 1.0


def decompose(
    cell: dict, k_target: int = K_TARGET, seed: int = 0, p_decay: float | None = None
) -> list[StepRow]:
    """Per-step ``Var(U) = Var(dV_true) + sigma_n^2 + sigma_{n-1}^2`` for one run.

    ``sigma_eps^2`` is measured at ``k_target`` and carried to the cell's own ``K`` by the **fitted**
    ``k^-p`` law, not by an assumed ``1/K`` -- see `decay_exponent`. Assuming ``p = 1`` where the
    measurement says ``p ~ 0.65`` under-states ``sigma_eps(K)``, which inflates ``Var(dV_true)`` and
    makes the noise share look SMALLER than it is; the earlier version of this script did exactly
    that. ``Var(dV_true)`` is what is left of the measured ``Var_M(U_n)`` -- clamped at 0, since a
    negative estimate means the two terms are within their own error, which at ``M = 8`` is +-53% on
    every variance here.
    """
    cfg = cell["config"]
    lam, k_full = float(cfg["lam"]), int(cfg["K"])
    if p_decay is None:
        p_decay = decay_exponent(sigma2_curve(cell, seed=seed))
    rks: list[Tensor | None] = cell.get("r_k_history") or []
    idxs: list[Tensor | None] = cell.get("resample_idx_history") or []
    gen = torch.Generator().manual_seed(seed)

    levels: list[Tensor | None] = []
    sig2_target: list[float] = []
    for rk in rks:
        if rk is None:
            levels.append(None)
            sig2_target.append(float("nan"))
            continue
        rk = rk.double()
        levels.append(_v(rk, lam, torch.arange(k_full)))
        sig2_target.append(_sigma2_by_disjoint_subsets(rk, lam, k_target, gen))

    rows: list[StepRow] = []
    for n in range(1, len(levels)):
        cur, prev = levels[n], levels[n - 1]
        if cur is None or prev is None:
            continue
        pidx = idxs[n - 1] if n - 1 < len(idxs) else None
        if pidx is not None:                      # pre- vs post-resample order; see the docstring
            prev = prev[pidx]
        if cur.numel() < 2:
            continue

        # sigma^2 at the cell's own K, and at the K the plan will run.
        scale = (k_target / k_full) ** p_decay     # measured law, NOT an assumed 1/K
        s2_now_full, s2_prev_full = sig2_target[n] * scale, sig2_target[n - 1] * scale
        var_u = float((cur - prev).var(unbiased=True))
        var_dv = max(var_u - s2_now_full - s2_prev_full, 0.0)

        noise_full = ((s2_now_full + s2_prev_full) / var_u) if var_u > 0 else float("nan")
        den_t = var_dv + sig2_target[n] + sig2_target[n - 1]
        noise_t = ((sig2_target[n] + sig2_target[n - 1]) / den_t) if den_t > 0 else float("nan")
        rows.append(StepRow(
            n=n, var_u=var_u, var_dv=var_dv, sig2_now=s2_now_full, sig2_prev=s2_prev_full,
            sd_v=float(cur.std(unbiased=True)), noise_share_full=noise_full,
            noise_share_target=noise_t,
        ))
    return rows


def pooled(rows: list[StepRow], k_full: int, k_target: int, p_decay: float) -> dict[str, float]:
    """Pool across steps by summing variances, not by averaging per-step shares.

    A per-step share is a ratio of two small-sample variances and is badly behaved when the
    denominator is small; summing the numerators and denominators separately weights each step by how
    much variance it actually contributes, which is what the resampler experiences.
    """
    ok = [r for r in rows if r.var_u > 0 and r.sig2_now == r.sig2_now]
    if not ok:
        return {}
    scale = (k_full / k_target) ** p_decay           # sigma^2(k_target) from sigma^2(K), fitted law
    sum_noise_full = sum(r.sig2_now + r.sig2_prev for r in ok)
    sum_u_full = sum(r.var_u for r in ok)
    sum_dv = sum(r.var_dv for r in ok)
    sum_noise_t = sum_noise_full * scale
    sum_sd_v = sum(r.sd_v for r in ok) / len(ok)

    # sd(U)/sd(V) as measured, and the floor perfect CRN could reach (Var(U) -> Var(dV_true)).
    # MEAN OF PER-STEP RATIOS, matching `analyse_k_sweep.py` -- so these are comparable to the 0.534 /
    # 0.346 / 0.671 already on record. Pooling variances first instead would weight the noisiest steps
    # and give a different (larger) number for the same run.
    ratio_now = sum(math.sqrt(r.var_u) / r.sd_v for r in ok) / len(ok)
    ratio_crn = sum(math.sqrt(r.var_dv) / r.sd_v for r in ok) / len(ok)
    return {
        "steps": float(len(ok)),
        "noise_share_full": sum_noise_full / sum_u_full,
        "noise_share_target": sum_noise_t / (sum_dv + sum_noise_t),
        "var_u_full": sum_u_full / len(ok),
        "var_dv": sum_dv / len(ok),
        "sd_ratio_now": ratio_now,
        "sd_ratio_crn_floor": ratio_crn,
    }


def corr_with_final(cell: dict) -> list[tuple[int, float, float]]:
    """Per step: ``corr(V_n, f_final)`` and ``corr(U_n, f_final)`` over the surviving lineages.

    The plan's central table (``V`` correlates 0.33-0.62 with the endpoint, ``U`` does not) was
    measured once on a3_2 over four steps of intact ancestry. It is the load-bearing claim of the
    whole diagnosis, so it is re-measured here on every cell -- and without the intact-ancestry
    restriction, by pushing each FINAL particle back through the parent maps to find which step-``n``
    particle it descends from.

    ``resample_idx[j]`` maps a post-resample slot to its parent's pre-resample slot at step ``j``, so
    composing from the end backwards gives, for each final particle, its ancestor's row in ``V_n``.
    Duplicate ancestors are kept: they are exactly the lineages resampling bet on, and dropping them
    would delete the effect being measured.
    """
    f_proj = cell.get("f_proj_history") or []
    idxs = cell.get("resample_idx_history") or []
    rks = cell.get("r_k_history") or []
    cfg = cell["config"]
    lam, k_full = float(cfg["lam"]), int(cfg["K"])
    if not f_proj or f_proj[-1] is None:
        return []
    f_final = f_proj[-1].double()
    n_last = len(f_proj) - 1

    levels = [None if rk is None else _v(rk.double(), lam, torch.arange(k_full)) for rk in rks]
    out: list[tuple[int, float, float]] = []
    for n in range(1, min(len(levels), n_last)):
        cur, prev = levels[n], levels[n - 1]
        if cur is None or prev is None:
            continue
        # Walk each final particle back to its step-n ancestor.
        a = torch.arange(f_final.numel())
        for j in range(n_last, n, -1):
            if j < len(idxs) and idxs[j] is not None:
                a = idxs[j][a]
        if n < len(idxs) and idxs[n] is not None:
            a_pre = idxs[n][a]                    # into V_n, which is pre-resample
        else:
            a_pre = a
        pidx = idxs[n - 1] if n - 1 < len(idxs) else None
        prev_aligned = prev[pidx] if pidx is not None else prev
        v_n, u_n = cur[a_pre], (cur - prev_aligned)[a_pre]

        def _c(x: Tensor) -> float:
            if x.std() < 1e-12 or f_final.std() < 1e-12:
                return float("nan")
            return float(torch.corrcoef(torch.stack([x, f_final]))[0, 1])

        out.append((n, _c(v_n), _c(u_n)))
    return out


def prediction_check(cell: dict, k_target: int = K_TARGET, seed: int = 3) -> tuple[float, float]:
    """``(measured, predicted)`` ``Var_M(U_n)`` for a ``k_target``-column estimator, WITHIN one cell.

    The additive model ``Var(U) = Var(dV) + 2 sigma^2`` is what the whole verdict rests on, so it is
    worth one direct test. Rebuild ``V`` from a ``k_target``-subset at every step -- an independent
    subset per step, which is what a real ``K = k_target`` run would have -- and compare the resulting
    ``Var(U)`` against ``Var(dV) + sigma^2(k_target)_n + sigma^2(k_target)_{n-1}``.

    Deliberately within-cell: the two real ``K = 4`` cells on disk both collapsed at step 3, so their
    ``Var(U)`` is measured on a one-lineage cloud and is 3-4x any ``K = 8`` cell's. That is a cloud-state
    difference, not an estimator difference, and it makes them useless as a cross-cell check.
    """
    cfg = cell["config"]
    lam, k_full = float(cfg["lam"]), int(cfg["K"])
    rks = cell.get("r_k_history") or []
    idxs = cell.get("resample_idx_history") or []
    gen = torch.Generator().manual_seed(seed)
    rows = decompose(cell, k_target=k_target, seed=seed)
    by_n = {r.n: r for r in rows}

    levels: list[Tensor | None] = []
    for rk in rks:
        if rk is None:
            levels.append(None)
            continue
        subs = _subsets(k_full, k_target, gen)
        levels.append(_v(rk.double(), lam, subs[0]))

    meas, pred = [], []
    scale = (k_full / k_target) ** decay_exponent(sigma2_curve(cell))
    for n in range(1, len(levels)):
        cur, prev, row = levels[n], levels[n - 1], by_n.get(n)
        if cur is None or prev is None or row is None or row.sig2_now != row.sig2_now:
            continue
        pidx = idxs[n - 1] if n - 1 < len(idxs) else None
        if pidx is not None:
            prev = prev[pidx]
        meas.append(float((cur - prev).var(unbiased=True)))
        pred.append(row.var_dv + (row.sig2_now + row.sig2_prev) * scale)
    if not meas:
        return float("nan"), float("nan")
    return sum(meas) / len(meas), sum(pred) / len(pred)


def measured_var_u(cell: dict) -> tuple[float, float]:
    """``(mean_n Var_M(U_n), mean_n sd_M(V_n))`` at the cell's own ``K``, no decomposition.

    The only thing a ``K = 4`` cell can contribute -- it cannot measure its own ``sigma_eps`` (one
    subset of 4 has nothing to be compared against) -- and it is exactly what validates the
    prediction made from a ``K = 8`` cell: ``Var(U) at K' = 4`` should equal
    ``Var(dV) + 2 sigma^2(4)``.
    """
    cfg = cell["config"]
    lam, k = float(cfg["lam"]), int(cfg["K"])
    rks = cell.get("r_k_history") or []
    idxs = cell.get("resample_idx_history") or []
    levels = [None if rk is None else _v(rk.double(), lam, torch.arange(k)) for rk in rks]
    vus, sds = [], []
    for n in range(1, len(levels)):
        cur, prev = levels[n], levels[n - 1]
        if cur is None or prev is None or cur.numel() < 2:
            continue
        pidx = idxs[n - 1] if n - 1 < len(idxs) else None
        if pidx is not None:
            prev = prev[pidx]
        vus.append(float((cur - prev).var(unbiased=True)))
        sds.append(float(cur.std(unbiased=True)))
    if not vus:
        return float("nan"), float("nan")
    return sum(vus) / len(vus), sum(sds) / len(sds)


def check_one_over_k(cell: dict, seed: int = 0) -> list[tuple[int, float]]:
    """Measure ``sigma_eps^2`` at several subset sizes; it must fall like ``1/k``.

    The decomposition carries a measurement at ``k_target`` to the cell's own ``K`` by that scaling,
    so it is worth spending three lines to confirm it holds on these very tensors rather than
    inheriting it from theory that assumes a linear estimator (``V`` is a ``logsumexp``).
    """
    cfg = cell["config"]
    lam, k_full = float(cfg["lam"]), int(cfg["K"])
    out: list[tuple[int, float]] = []
    for kt in (1, 2, 4, 8, 16):
        if k_full // kt < 2:
            break
        gen = torch.Generator().manual_seed(seed)
        vals = [_sigma2_by_disjoint_subsets(rk.double(), lam, kt, gen)
                for rk in (cell.get("r_k_history") or []) if rk is not None]
        vals = [v for v in vals if v == v]
        if vals:
            out.append((kt, sum(vals) / len(vals)))
    return out


# ---------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------

def load_cells() -> list[dict]:
    """Every FLUX cell on disk with ``r_k`` recorded, finished results superseding their partials."""
    cells: list[dict] = []
    seen: set[tuple] = set()
    paths: list[str] = []
    for d in DUMP_DIRS:
        paths += sorted(glob.glob(os.path.join(d, "result_*.pt")))
        paths += sorted(glob.glob(os.path.join(d, "partial_*.pt")))
    for fp in paths:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        cfg = d["config"]
        key = (cfg["K"], cfg["m_tilt"], cfg["M"], cfg["seed"], cfg.get("run_tag", ""))
        if key in seen or not any(r is not None for r in (d.get("r_k_history") or [])):
            continue
        seen.add(key)
        d["_file"] = os.path.relpath(fp, REPO)
        cells.append(d)
    return cells


# ---------------------------------------------------------------------------------------------------
# The toy: the same decomposition where a large K makes dV_true directly observable
# ---------------------------------------------------------------------------------------------------

def toy_cell(
    lam: float, *, n_particles: int = 64, k: int = 256, seed: int = 7, antithetic: bool = True
) -> dict:
    """Run the analytic 2-D toy at a large ``K`` and package it like a FLUX dump.

    The FLUX cells can only measure ``sigma_eps`` at ``k_target`` and rescale; here ``K = 256`` makes
    ``V`` nearly exact, so ``Var(dV_true)`` is observable rather than inferred, and the two routes can
    be compared. The toy's ``V`` moves MORE per step than FLUX's (`RESULTS.md`), so a residual-dominated
    verdict here is the weaker of the two readings -- which is why both are run.
    """
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_flowmap_smc import (  # type: ignore[import-not-found]  # noqa: E402
        GaussianFlowMap, _gaussian_score_fn, _reward,
    )
    from creativity_measure.samplers.flowmap_smc import LinearSchedule, flowmap_smc_sample  # noqa: E402

    schedule = LinearSchedule()
    res = flowmap_smc_sample(
        _reward(), lam, n_particles,
        flow_map=GaussianFlowMap(schedule), score_fn=_gaussian_score_fn(), schedule=schedule,
        n_steps=16, mc_samples=k, ess_threshold=0.5, guid_window=(0.0, 1.0),
        stoch_window=(0.1, 1.0), record_r_k=True, project_endpoint=True, seed=seed,
        antithetic=antithetic,
    )
    tag = f"toy lam={lam:g}{'' if antithetic else ' NO-antithetic'}"
    return {
        "config": {"K": k, "M": n_particles, "lam": lam, "m_tilt": float("nan"), "seed": seed,
                   "run_tag": tag, "gpu_name": "cpu"},
        "r_k_history": res.r_k_history,
        "resample_idx_history": res.resample_idx_history,
        "f_proj_history": res.f_proj_history,
        "uniq_history": res.uniq_history,
        "_file": f"<{tag}, M={n_particles}, K={k}>",
    }


# ---------------------------------------------------------------------------------------------------

def _verdict(share: float) -> str:
    if share != share:
        return "n/a"
    return "PROCEED (residual dominates)" if share > DECISION_THRESHOLD else "STOP (dV dominates)"


def report(cells: list[dict], k_target: int) -> None:
    print(f"\n{'=' * 100}\nnoise share = (sigma_n^2 + sigma_{{n-1}}^2) / Var(U_n), rescaled to "
          f"K' = {k_target} -- the plan's configuration")
    print(f"  > {DECISION_THRESHOLD:.2f}: CRN has something to cancel.   "
          f"< {DECISION_THRESHOLD:.2f}: dV_true dominates, CRN is inapplicable.\n")
    print(f"{'cell':<34} {'K':>3} {'M':>3} {'steps':>5} {'a=lam*sd_k':>10} {'p':>5} {'Var(U)':>9} "
          f"{'Var(dV)':>9} {'share@K':>8} {f'share@{k_target}':>9} {'sd(U)/sd(V)':>11} "
          f"{'CRN floor':>9}  verdict")
    for cell in cells:
        cfg = cell["config"]
        p_dec = decay_exponent(sigma2_curve(cell))
        rows = decompose(cell, k_target=k_target, p_decay=p_dec)
        p = pooled(rows, int(cfg["K"]), k_target, p_dec)
        if not p:
            vu, _ = measured_var_u(cell)
            print(f"{cell['_file'][-34:]:<34} {cfg['K']:>3} {cfg['M']:>3}     -- {'':>10} {'':>5} "
                  f"{vu:>9.4f} {'--':>9} {'--':>8} {'--':>9} {'--':>11} {'--':>9}  "
                  f"cannot self-measure sigma (K < {2 * k_target})")
            continue
        rk = [r for r in (cell.get("r_k_history") or []) if r is not None]
        a = float(cfg["lam"]) * float(torch.stack([r.double().std(dim=1).mean() for r in rk]).mean())
        print(f"{cell['_file'][-34:]:<34} {cfg['K']:>3} {cfg['M']:>3} {int(p['steps']):>5} "
              f"{a:>10.3f} {p_dec:>5.2f} {p['var_u_full']:>9.4f} {p['var_dv']:>9.4f} "
              f"{p['noise_share_full']:>8.3f} {p['noise_share_target']:>9.3f} "
              f"{p['sd_ratio_now']:>11.3f} {p['sd_ratio_crn_floor']:>9.3f}  "
              f"{_verdict(p['noise_share_target'])}")

    print(f"\n{'=' * 100}\ndoes the increment predict the endpoint? mean over steps of "
          f"corr(., f at t=1), lineages traced through the parent maps")
    print(f"  {'cell':<34} {'corr(V,f_fin)':>14} {'corr(U,f_fin)':>14} {'steps':>6}")
    for cell in cells:
        cs = [(v, u) for _, v, u in corr_with_final(cell) if v == v and u == u]
        if not cs:
            continue
        mv = sum(v for v, _ in cs) / len(cs)
        mu = sum(u for _, u in cs) / len(cs)
        print(f"  {cell['_file'][-34:]:<34} {mv:>14.3f} {mu:>14.3f} {len(cs):>6}")

    print(f"\n{'=' * 100}\nadditive-model check, WITHIN cell: Var(U) rebuilt from a real {k_target}"
          f"-column subset vs Var(dV) + 2 sigma^2({k_target})")
    for cell in cells:
        if int(cell["config"]["K"]) < 2 * k_target:
            continue
        meas, pred = prediction_check(cell, k_target)
        if meas != meas:
            continue
        print(f"  {cell['_file'][-34:]:<34} measured {meas:>8.4f}   predicted {pred:>8.4f}   "
              f"ratio {meas / pred:>5.2f}x")

    print(f"\n{'=' * 100}\nsigma_eps^2 vs subset size (pair-respecting). p is the fitted exponent in "
          f"sigma^2 ~ k^-p; independent draws\nthrough a linear estimator would give p = 1, and "
          f"p < 1 is itself why raising K saturated.")
    for cell in cells:
        pts = sigma2_curve(cell)
        if len(pts) < 2:
            continue
        body = "  ".join(f"k={k}: {v:.4g}" for k, v in pts)
        ratios = "  ".join(f"{pts[i][1] / pts[i + 1][1]:.2f}x" for i in range(len(pts) - 1))
        print(f"  {cell['_file'][-34:]:<34} {body}\n{'':<37}ratios (p=1 predicts 2.00x): {ratios}"
              f"   fitted p = {decay_exponent(pts):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-toy", action="store_true", help="skip the toy runs (dumps only)")
    ap.add_argument("--k-target", type=int, default=K_TARGET)
    ap.add_argument("--toy-lam", type=float, nargs="*", default=[8.0, 16.0])
    args = ap.parse_args()

    cells = load_cells()
    print(f"loaded {len(cells)} FLUX cell(s) with r_k recorded")
    gpus = {c["config"].get("gpu_name", "?") for c in cells}
    if len(gpus) > 1:
        print(f"*** NOTE: cells span {gpus}. Fine here -- every number below is WITHIN a cell.")

    if not args.no_toy:
        for lam in args.toy_lam:
            cells.append(toy_cell(lam))
        # One arm with the pairing off: if p jumps to ~1 here, the sub-1 exponent on the FLUX cells
        # is the antithetic structure and not a property of the estimator.
        cells.append(toy_cell(args.toy_lam[-1], antithetic=False))

    report(cells, args.k_target)


if __name__ == "__main__":
    main()
