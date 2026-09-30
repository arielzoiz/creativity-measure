"""Does raising particle count M rescue ESS/diversity at a fixed, already-degenerate tilt?

The real run (`notebooks/diamond_smc_imagenet/`, job 698463) found m=2.0 (lambda=58.227) DEGENERATE
at M=16: uniq/M=0.38, min ESS/M=0.31, yet E_q[f]=1.0209 -- nominally higher than the "OK" m=1.0 point
(E_q[f]=1.0014), but not trustworthy since so few lineages survived to report it.

Algorithm 3 (`flowmap_smc.py`) has an established result (see CLAUDE.md) that its ESS/M scales like
exp(-m^2), because it only reweights draws from the UNTILTED proposal rather than moving particles
toward the tilted target. Algorithm 2 is structurally closer to Algorithm 3 than to Algorithm 1 in this
respect: it is SMC-with-resampling over a fixed generative trajectory (GLASS + diamond-map lookahead),
not MCMC that actively moves particles (Algorithm 1's pCN kernel). If it shares that mechanism, the fix
for the m=2.0 degeneracy should be M, not lambda -- M ~ exp(m^2) to hold today's ESS/M.

This holds LAMBDA FIXED at the m=2.0 value from job 698463 (58.227 -- not recomputed per M's own probe,
so the target tilt strength is the controlled variable, isolating the effect of M alone) and sweeps M
in {16, 32, 64, 128}, watching whether min ESS/M and uniq/M climb back over the established 0.5 floor,
and whether E_q[f] holds or falls once they do.
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward_common  # noqa: E402
import paths  # noqa: E402

LAMBDA_FIXED = 58.227  # m=2.0 in job 698463 (real run), M=16
K = 4
ESS_THRESHOLD = 1.0
M_LIST = [16, 32, 64, 128]


def main() -> int:
    info = paths.preflight()
    from creativity_measure.diamond_smc import diamond_smc_sample

    backend, reward, lam_s = reward_common.build_backend_and_reward(info, probe_m=16)
    print(
        f"fixed lambda={LAMBDA_FIXED}  (job 698463's m=2.0; this run's own lambda_s={lam_s:.3f} "
        f"is printed for reference only, NOT applied)",
        flush=True,
    )
    print(f"M grid: {M_LIST}", flush=True)

    rows = []
    for M in M_LIST:
        backend.reset_rng(reward_common.SEED)
        t0 = time.time()
        res = diamond_smc_sample(
            reward, LAMBDA_FIXED, M,
            backend=backend, n_steps=reward_common.N, mc_samples=K,
            ess_threshold=ESS_THRESHOLD, seed=reward_common.SEED, verbose=False,
        )
        f_final = reward(res.X.reshape(M, -1))
        min_ess = min(res.ess_history) / M
        uniq = res.uniq_history[-1]
        elapsed = time.time() - t0
        rows.append({
            "M": M, "f_mean": float(f_final.mean()), "f_std": float(f_final.std()),
            "min_ess_m": min_ess, "uniq_m": uniq, "seconds": elapsed,
        })
        print(
            f"  M={M:4d}  E_q[f]={float(f_final.mean()):.4f}  min ESS/M={min_ess:.2f}  "
            f"uniq/M={uniq:.2f}  ({elapsed:.0f}s)",
            flush=True,
        )

    print(f"\n{'M':>5} {'E_q[f]':>9} {'min ESS/M':>10} {'uniq/M':>8}  verdict")
    for r in rows:
        ok = (r["min_ess_m"] >= 0.5) and (r["uniq_m"] >= 0.5)
        print(
            f"{r['M']:5d} {r['f_mean']:9.4f} {r['min_ess_m']:10.2f} {r['uniq_m']:8.2f}  "
            f"{'OK' if ok else 'DEGENERATE'}"
        )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(
        {"rows": rows, "lambda_fixed": LAMBDA_FIXED, "K": K},
        os.path.join(out_dir, "M_sweep_results.pt"),
    )
    print(f"\nsaved {os.path.join(out_dir, 'M_sweep_results.pt')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
