"""Does a hard-max lookahead differ from the soft value? Read it off a K-sweep checkpoint.

    python analyze_r_k_spread.py partial_N8_K4_m1.25_seed101_k04.pt [more.pt ...]

Runs on CPU against `partial_*.pt` (written after EVERY step by the notebook's `make_checkpointer`)
or `result_*.pt`, so it answers hours before any job finishes and survives a `killable` preemption.

**The number.** With ``r_k = r_bar + sigma_k·u_k`` per particle and ``a = lambda·sigma_k``, the boost
over the plain mean-of-K is ``lambda·sigma_k^2/2`` for the soft value against ``sigma_k·max_k u_k``
for the hard max -- a ratio of about ``2·c_K/a``, ``c_K = E[max_k u] ~ 1.03, 1.42, 1.77, 2.07`` at
``K = 4, 8, 16, 32``. So:

    a <~ 0.1   the max is a ~20x stronger twist -- worth a GPU
    a ~  1     ~2x -- real but modest
    a >~ 5     the logsumexp has already saturated INTO a max; `flowmap_smc_max` would reproduce
               `flowmap_smc` and the run would confirm nothing

**The second column, which is the one that actually decides resampling.** Selection is driven by the
spread of ``U`` ACROSS particles, not by the boost itself, and the max only changes selection insofar
as the boost varies with ``m``. ``d_sd/U_sd`` compares the extra across-particle dispersion the max
introduces at that step against the dispersion already in ``U_pre``. Small means the resampling
decisions do not move and the cloud comes out the same.

**What this does NOT show.** It bounds the systematic difference in the potential, not trajectory
divergence: SMC is chaotic in its resampling decisions, so a tiny potential change can still flip one
resample and send the cloud elsewhere. Read a small ``a`` as "the max adds no systematic tilt", never
as "the images will be identical".

Everything is computed from ``r_k_history`` and ``U_pre_history``, both already recorded whenever the
run set ``record_r_k=True``. Nothing here re-runs the model.
"""

import math
import sys

import torch

C_K: dict[int, float] = {1: 0.000, 2: 0.564, 4: 1.029, 8: 1.423, 16: 1.766, 32: 2.069}


def _c_k(k: int) -> float:
    """``E[max of k standard normals]``; the tabulated values, else the ``sqrt(2 ln k)`` asymptote."""
    return C_K.get(k, math.sqrt(2.0 * math.log(max(k, 2))))


def report(path: str) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    lam, K, M = float(cfg["lam"]), int(cfg["K"]), int(cfg["M"])
    print(f"\n=== {path}")
    print(f"    K={K} M={M} lambda={lam:.2f} (m={cfg['m_tilt']:g})  gpu={cfg.get('gpu_name','?')}  "
          f"partial={ck.get('partial')}  steps recorded={len(ck['r_k_history'])}")
    if not any(r is not None for r in ck["r_k_history"]):
        print("    no r_k recorded (record_r_k was off, or no guided step has completed yet)")
        return

    print(f"    {'step':>4} {'t':>7} {'a=lam*sd_k':>11} {'b_max/b_soft':>13} "
          f"{'d_sd/U_sd':>10} {'dEf_max':>9} {'dEf_soft':>9}")
    for n, r_k in enumerate(ck["r_k_history"]):
        if r_k is None:
            continue
        r_k = r_k.double()                                  # (M, K)
        sd_k = r_k.std(dim=1)                               # within-particle spread, per particle
        a = lam * sd_k

        # The two boosts, in f units: what each aggregation adds on top of the plain mean-of-K.
        b_soft = (torch.logsumexp(lam * r_k, dim=1) - math.log(K)) / lam - r_k.mean(dim=1)
        b_max = r_k.max(dim=1).values - r_k.mean(dim=1)

        # Does the difference move the weights? Compare its ACROSS-PARTICLE spread, in log-weight
        # units, against the spread already in the accumulated potential the step resampled on.
        d = lam * (b_max - b_soft)
        u_pre = ck["U_pre_history"][n]
        u_sd = float(u_pre.double().std()) if u_pre is not None and u_pre.numel() > 1 else float("nan")
        ratio = float(d.std()) / u_sd if u_sd == u_sd and u_sd > 0 else float("nan")

        print(f"    {n + 1:>4} {ck['t_history'][n]:>7.4f} {float(a.mean()):>11.3f} "
              f"{float((b_max / b_soft.clamp_min(1e-30)).mean()):>13.1f} {ratio:>10.3f} "
              f"{float(b_max.mean()):>9.2e} {float(b_soft.mean()):>9.2e}")

    print(f"    predicted from a alone: 2*c_K/a with c_K={_c_k(K):.3f} at K={K}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        report(p)
