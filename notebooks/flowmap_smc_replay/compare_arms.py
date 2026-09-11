"""Compare the 2026-09-10 Algorithm 3 arms against their paired baselines. No GPU.

Reads whatever has landed -- `result_*.pt` first, `partial_*.pt` as a fallback -- so it is useful on a
half-finished sweep and on preempted jobs, which is the normal case on `killable`.

WHAT IT REPORTS, in the order `RUNS.md` says to read it:

1. **uniq/M at the final step.** The degeneracy signal, and the thing that actually broke every run to
   date. Baselines: k04 = 0.125, best-ever at M = 8 = 0.375, control = 1.000.
2. **Terminal E_q[f], CONTROL-SUBTRACTED, in units of std_p(f) = 1/lam_s.** Never raw: the lambda = 0
   control alone gives back 3.7 std_p(f) between its peak and t = 1, so a raw within-run curve is
   mostly artifact. Only t = 1 is artifact-free. Seed sd is ~0.17 std_p(f) -- differences below that
   are not differences.
3. **sd(U)/sd(V)**, mean of per-step ratios, aligned through the parent maps -- the mechanism metric,
   against k04's 0.534.
4. **The divergence check.** `mean` and `defer25` carry particles IDENTICAL to k04 until the first step
   whose resampling decision differs, because the base transition never reads U. So `f_cand` must match
   k04 EXACTLY before that step. If it does not, something other than the intended variable changed and
   the comparison is void -- this is the check that catches a silently different base process, a
   different refs file, or a different GPU, none of which announce themselves.

Usage::

    conda activate creativity-measure
    python notebooks/flowmap_smc_replay/compare_arms.py
"""

from __future__ import annotations

import glob
import math
import os

import torch
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
KSWEEP = os.path.join(REPO, "notebooks", "flowmap_smc_k_sweep")

# The paired baselines, both already on disk. One control serves every M = 8 arm: at lambda = 0 no
# lookahead fires, so the aggregation and the resampling window cannot change anything.
BASELINE = os.path.join(KSWEEP, "result_N8_K4_m1.25_seed101_k04.pt")
CONTROL_M8 = os.path.join(KSWEEP, "result_N8_K1_m0_seed101_ctrl2.pt")


def load(tag: str, run_dir: str) -> dict | None:
    """A run by RUN_TAG, preferring the finished result over its own partial.

    Searches BOTH notebook directories. The forked notebooks inherited a hardcoded
    ``RUN_DIR = .../flowmap_smc_k_sweep`` in their setup cell, so the 2026-09-10 arms wrote their
    results next to the k-sweep's despite living in `flowmap_smc_replay`. Searching both is the honest
    fix for the reader: a result is identified by its RUN_TAG, which is unique, not by its directory.
    """
    for pat in (f"result_*_{tag}.pt", f"partial_*_{tag}.pt"):
        hits = sorted(sum((glob.glob(os.path.join(d, pat)) for d in (run_dir, KSWEEP, HERE)), []))
        if hits:
            d = torch.load(hits[0], map_location="cpu", weights_only=False)
            d["_file"] = os.path.basename(hits[0])
            return d
    return None


def sd_ratio(d: dict) -> float:
    """Mean over steps of sd(U_n)/sd(V_n), U aligned through that step's parent map.

    Same estimator as `flowmap_smc_k_sweep/analyse_k_sweep.py`, so the numbers are comparable to the
    0.534 / 0.346 / 0.671 already on record. V_n is recorded pre-resample while the cloud entering
    step n+1 is post-resample, so the previous level MUST be mapped through `resample_idx`.
    """
    lam = float(d["config"]["lam"])
    idxs = d.get("resample_idx_history") or []
    levels = [None if r is None else torch.logsumexp(lam * r.double(), dim=1) - math.log(r.shape[1])
              for r in (d.get("r_k_history") or [])]
    out = []
    for n in range(1, len(levels)):
        cur, prev = levels[n], levels[n - 1]
        if cur is None or prev is None or cur.numel() < 2:
            continue
        pidx = idxs[n - 1] if n - 1 < len(idxs) else None
        if pidx is not None:
            prev = prev[pidx]
        sv = float(cur.std(unbiased=True))
        if sv > 1e-12:
            out.append(float((cur - prev).std(unbiased=True)) / sv)
    return sum(out) / len(out) if out else float("nan")


def terminal_se(d: dict) -> float:
    """SE of a run's terminal E_q[f], in raw f units, from its own particle spread.

    The control is unweighted, so this is ``std(f_proj)/sqrt(M)``. It matters because every "vs ctrl"
    number below is a DIFFERENCE of two Monte-Carlo estimates and inherits both errors -- at M = 8 that
    is ~0.35 std_p(f) on the control alone, which is twice the seed sd the acceptance criteria are
    written against. Measured live: the M=16 control sat 1.3 std_p(f) above the M=8 one at step 8,
    purely from this. **Never compare an arm to a control at a different M.**
    """
    fp = (d.get("f_proj_history") or [None])[-1]
    if not isinstance(fp, Tensor) or fp.numel() < 2:
        return float("nan")
    return float(fp.double().std(unbiased=True)) / math.sqrt(fp.numel())


def collapse_step(d: dict) -> str:
    m = int(d["config"]["M"])
    for i, u in enumerate(d.get("uniq_history") or []):
        if u <= 1.0 / m + 1e-9:
            return f"step {i + 1}"
    return "never"


def divergence_step(arm: dict, base: dict) -> str:
    """First step where `f_cand` differs from the baseline -- must be AFTER a differing resample.

    Until the first step whose resampling decision differs, both runs carry the same particles (the
    base transition never reads U), so this is an exact equality check, not a tolerance one.
    """
    a, b = arm.get("f_mean") or [], base.get("f_mean") or []
    ra = arm.get("resampled_history") or []
    rb = base.get("resampled_history") or []
    first_diff_resample = next((i for i in range(min(len(ra), len(rb))) if ra[i] != rb[i]), None)
    for i in range(min(len(a), len(b))):
        if a[i] != b[i] and not (a[i] != a[i] and b[i] != b[i]):    # nan == nan counts as equal
            ok = (first_diff_resample is not None and i > first_diff_resample)
            return (f"step {i + 1}"
                    + ("" if ok else "  *** BEFORE any differing resample -- COMPARISON VOID"))
    return "never (identical throughout)"


def terminal(d: dict) -> float:
    """E_q[f] at t = 1 -- ``nan`` unless the run actually REACHED t = 1.

    A partial run's last `eqf_history` entry is its last completed step, not its terminal, and the two
    differ by up to 3.7 std_p(f) (the projection artifact: the untilted curve peaks at step 10 and
    falls back). Returning it would silently mis-subtract every control-relative number in the table,
    which is exactly the class of error the control exists to prevent.
    """
    eqf = [v for v in (d.get("eqf_history") or []) if v == v]
    n_steps = int(d["config"].get("n_steps", 16))
    if not eqf or len(d.get("eqf_history") or []) < n_steps:
        return float("nan")
    return eqf[-1]


def paired_by_seed() -> list[tuple[int, dict, dict]]:
    """Every seed for which BOTH a mean and a soft run finished, as ``(seed, mean_run, soft_run)``.

    Pairs are discovered from each file's own ``config`` -- ``seed`` and ``agg`` -- never from
    filenames. The k-sweep's ``k04`` cell predates the ``agg`` key and IS the soft arm at seed 101, so
    a missing ``agg`` is read as ``"soft"``; that is the one inference made here and it is safe because
    every run without the key was produced by `flowmap_smc_sample` itself.
    """
    runs: dict[tuple[int, str], dict] = {}
    for d in (KSWEEP, HERE):
        for fp in sorted(glob.glob(os.path.join(d, "result_*.pt"))):
            r = torch.load(fp, map_location="cpu", weights_only=False)
            c = r["config"]
            if float(c.get("m_tilt", 0)) == 0.0 or int(c.get("M", 0)) != 8 or int(c.get("K", 0)) != 4:
                continue                      # controls, other M, other K are not part of this design
            # A run only belongs to this design if it differs from the baseline in the AGGREGATION
            # and nothing else. `defer25` carries `arm`/`resample_window` instead of `agg`, so the
            # "missing agg means soft" rule below would otherwise adopt it as seed 101's soft arm --
            # and `defer25` sorts before `k04`, so it would silently WIN that slot and the headline
            # would compare mean against the wrong run. Exclude anything with a non-full window.
            rw = c.get("resample_window")
            if c.get("arm") is not None or (rw is not None and tuple(rw) != (0.0, 1.0)):
                continue
            agg = str(c.get("agg", "soft"))
            if agg not in ("mean", "soft"):
                continue
            r["_file"] = os.path.basename(fp)
            runs.setdefault((int(c["seed"]), agg), r)
    seeds = sorted({s for s, _ in runs})
    return [(s, runs[(s, "mean")], runs[(s, "soft")])
            for s in seeds if (s, "mean") in runs and (s, "soft") in runs]


def report_paired(sd_p: float) -> None:
    """The headline of the aggregation experiment: mean - soft, paired on seed.

    **The control cancels exactly in a paired difference** (`flowmap_smc_max/TAKEAWAYS.md`), so no
    lambda = 0 run is needed at a new seed and none is used here. That is also why this is the
    statistic to trust over the control-subtracted column above: it carries neither the control's
    Monte-Carlo error nor any assumption that two seeds share a control.
    """
    pairs = paired_by_seed()
    print(f"\n{'=' * 100}\nPAIRED mean - soft by seed (control cancels exactly; no control used)")
    if not pairs:
        print("  no seed has both arms finished yet")
        return
    print(f"  {'seed':>5} {'mean':>8} {'soft':>8} {'(mean-soft)/std_p':>18} "
          f"{'uniq mean/soft':>16} {'collapse mean/soft':>20}")
    diffs = []
    for s, m, b in pairs:
        tm, tb = terminal(m), terminal(b)
        if tm != tm or tb != tb:
            print(f"  {s:>5} {'--':>8} {'--':>8}   (one arm incomplete)")
            continue
        d = (tm - tb) / sd_p
        diffs.append(d)
        print(f"  {s:>5} {tm:>8.4f} {tb:>8.4f} {d:>+18.2f} "
              f"{m['uniq_history'][-1]:>7.3f}/{b['uniq_history'][-1]:<8.3f} "
              f"{collapse_step(m):>9}/{collapse_step(b):<10}")

    n = len(diffs)
    if n < 2:
        print(f"\n  {n} paired seed(s) -- not enough for an interval. The max experiment needed 3.")
        return
    mu = sum(diffs) / n
    sd = math.sqrt(sum((d - mu) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.0)   # t_{0.975, n-1}
    lo, hi = mu - tcrit * se, mu + tcrit * se
    print(f"\n  n = {n} seeds:  mean {mu:+.3f}  sd {sd:.3f}  se {se:.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}] std_p(f)")
    verdict = ("CI excludes 0 -- the mean aggregation is better" if lo > 0 else
               "CI excludes 0 -- the mean aggregation is WORSE" if hi < 0 else
               "CI includes 0 -- not established at this n")
    print(f"  -> {verdict}")
    print("  (compare: the hard-max experiment came out -0.034 sd, CI [-0.671, +0.602], n = 3)")


def main() -> None:
    base = torch.load(BASELINE, map_location="cpu", weights_only=False)
    base["_file"] = os.path.basename(BASELINE)
    ctrl8 = torch.load(CONTROL_M8, map_location="cpu", weights_only=False)
    lam_s = float(base["config"]["lam_s"])
    sd_p = 1.0 / lam_s                       # std_p(f), the unit for every difference below
    c8 = terminal(ctrl8)

    arms = [
        ("k04 (soft, M=8)", base, c8),
        ("mean (M=8)", load("mean", HERE), c8),
        ("defer25 (M=8)", load("defer25", HERE), c8),
        ("m16 (soft, M=16)", load("m16", KSWEEP), None),      # control filled in below
    ]
    ctrl16 = load("ctrl16", KSWEEP)
    if ctrl16 is not None:
        arms[-1] = (arms[-1][0], arms[-1][1], terminal(ctrl16))

    se8 = terminal_se(ctrl8)
    print(f"std_p(f) = 1/lam_s = {sd_p:.5f}   control(M=8) terminal E_p[f] = {c8:.4f} "
          f"+-{se8 / sd_p:.2f}sd (SE)"
          + (f"   control(M=16) = {terminal(ctrl16):.4f} "
             f"+-{terminal_se(ctrl16) / sd_p:.2f}sd" if ctrl16 is not None else
             "   control(M=16): NOT YET"))
    print(f"\n{'arm':<18} {'steps':>5} {'uniq/M':>7} {'collapse':>9} {'resamp':>7} "
          f"{'terminal':>9} {'vs ctrl(sd)':>12} {'sd(U)/sd(V)':>11} {'sd(logw)':>9}  file")
    for name, d, ctrl in arms:
        if d is None:
            print(f"{name:<18}     -- not started")
            continue
        cfg = d["config"]
        n = len(d.get("eqf_history") or [])
        term = terminal(d)
        rel = ((term - ctrl) / sd_p) if (ctrl is not None and ctrl == ctrl) else float("nan")
        # The difference carries BOTH runs' Monte-Carlo error, not just the arm's.
        se_pair = math.hypot(terminal_se(d), se8 if ctrl is c8 else terminal_se(ctrl16 or {})) / sd_p
        lw = d.get("logw")
        sdlw = float(lw.std()) if isinstance(lw, Tensor) and lw.numel() > 1 else float("nan")
        flag = "" if not d.get("partial") else "  (PARTIAL)"
        print(f"{name:<18} {n:>5} {(d.get('uniq_history') or [float('nan')])[-1]:>7.3f} "
              f"{collapse_step(d):>9} {int(sum(d.get('resampled_history') or [])):>7} "
              f"{term:>9.4f} {rel:>+7.2f}+-{se_pair:<4.2f} {sd_ratio(d):>11.3f} {sdlw:>9.3f}  "
              f"{d['_file']}{flag}")

    print("\nbaselines: k04 uniq/M = 0.125, best-ever at M=8 = 0.375, control = 1.000; "
          "k04 sd(U)/sd(V) = 0.534")
    print("seed sd on terminal E_q[f] is ~0.17 sd; the +- above is the PAIRED Monte-Carlo SE of the\n"
          "difference, which at M = 8 is the larger of the two. Both must be cleared to claim an effect.\n")

    print("=== divergence check: f_cand must match k04 exactly until a resample decision differs ===")
    for name, d, _ in arms[1:3]:                         # the two M=8 arms paired against k04
        if d is not None:
            print(f"  {name:<18} diverges at {divergence_step(d, base)}")

    report_paired(sd_p)

    print("\n=== GPU / refs / base-process agreement (a mismatch invalidates the comparison) ===")
    for name, d, _ in arms:
        if d is None:
            continue
        cfg = d["config"]
        same_p = cfg.get("base_process") == base["config"].get("base_process")
        print(f"  {name:<18} gpu={cfg.get('gpu_name', '?'):<14} refs={cfg.get('refs_file', '?')} "
              f"base_process={'MATCHES' if same_p else '*** DIFFERS ***'}")


if __name__ == "__main__":
    main()
