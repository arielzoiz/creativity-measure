"""Fine-grained m sweep at fixed M=16 -- sharpens where uniq/M and min-ESS/M cross the 0.5 floor.

The coarse {0,1,2,3} grid already run in `notebooks/diamond_smc_imagenet/` (job 698463,
`results_label207_cfg1.0_M16K4N6.pt`) showed the answer in outline:

    m     lambda   E_q[f]   min ESS/M  uniq/M   verdict
    0.0    0.000   0.9965      1.00     1.00    OK (control)
    1.0   29.114   1.0014      0.66     0.56    OK  <- selected by the 0.5-floor rule
    2.0   58.227   1.0209      0.31     0.38    DEGENERATE
    3.0   87.341   1.0208      0.12     0.19    DEGENERATE

At the one point the established selection rule actually allows (m=1.0), the tilt buys +0.005 over
the untilted control -- essentially nothing, against E_p[f]=1 by construction. The larger-looking
gains at m=2/3 (+0.02) show up only once uniq/M has already collapsed to 0.38/0.19, i.e. only a few
surviving lineages are left to report a number, which is exactly the failure mode the 0.5 floor exists
to catch -- so those two numbers are not trustworthy evidence that the tilt "worked better" there.

This script re-runs the identical p (label=207, cfg_scale=1.0, N=6, K=4, M=16 -- see reward_common.py) at a
finer m grid to (a) locate the OK/DEGENERATE crossing more precisely than {0,1,2,3} allows, and (b)
see whether E_q[f] rises smoothly as diversity falls or jumps -- which would distinguish "the tilt is
just weak here" from "something breaks abruptly near the floor."
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
import reward_common  # noqa: E402

M = 16
K = 4
ESS_THRESHOLD = 1.0
M_LIST = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def main() -> int:
    info = paths.preflight()
    from creativity_measure.samplers.diamond_smc import diamond_smc_sample

    backend, reward, lam_s = reward_common.build_backend_and_reward(info, probe_m=M)
    print(f"m grid: {M_LIST}", flush=True)

    rows = []
    for m in M_LIST:
        lam = m * lam_s
        backend.reset_rng(reward_common.SEED)
        t0 = time.time()
        res = diamond_smc_sample(
            reward, lam, M,
            backend=backend, n_steps=reward_common.N, mc_samples=K,
            ess_threshold=ESS_THRESHOLD, seed=reward_common.SEED, verbose=False,
        )
        f_final = reward(res.X.reshape(M, -1))
        min_ess = min(res.ess_history) / M
        uniq = res.uniq_history[-1]
        imgs = backend.decode(res.X.reshape(M, -1)).detach().cpu()
        elapsed = time.time() - t0
        rows.append({
            "m": m, "lam": lam, "f_mean": float(f_final.mean()), "f_std": float(f_final.std()),
            "min_ess_m": min_ess, "uniq_m": uniq, "seconds": elapsed,
            "f_final": f_final.detach().cpu(), "imgs": imgs,
        })
        print(
            f"  m={m:5.2f} lam={lam:8.3f}  E_q[f]={float(f_final.mean()):.4f}  "
            f"min ESS/M={min_ess:.2f}  uniq/M={uniq:.2f}  ({elapsed:.0f}s)",
            flush=True,
        )

    f0 = rows[0]["f_mean"]
    print(
        f"\n{'m':>6} {'lambda':>9} {'E_q[f]':>9} {'DeltaE_q[f]':>12} "
        f"{'min ESS/M':>10} {'uniq/M':>8}  verdict"
    )
    for r in rows:
        ok = (r["min_ess_m"] >= 0.5) and (r["uniq_m"] >= 0.5)
        print(
            f"{r['m']:6.2f} {r['lam']:9.3f} {r['f_mean']:9.4f} {r['f_mean'] - f0:+12.4f} "
            f"{r['min_ess_m']:10.2f} {r['uniq_m']:8.2f}  {'OK' if ok else 'DEGENERATE'}"
        )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(
        {"rows": rows, "lambda_s": lam_s, "M": M, "K": K},
        os.path.join(out_dir, "m_sweep_results.pt"),
    )
    print(f"\nsaved {os.path.join(out_dir, 'm_sweep_results.pt')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
