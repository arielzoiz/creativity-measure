"""Does MORE base-sampler outer steps genuinely improve samples, or does variance compound away?

`sweep_sampler_quality.py` (in `notebooks/diamond_smc_imagenet/`) found an anomaly: at cfg_scale=4.0,
going from N=6 to N=12 outer steps (same inner=8) made the decoded images WORSE (pixel std 0.048 ->
0.034), not better -- and inner steps (8 vs 16) barely mattered. That is backwards for a converging
discretization: more transitions should reduce integration error, not compound it. It echoes an
earlier finding in that same directory (`ab_base_sampler.py`): latent std falls from ~0.53 (N=4) to
~0.39 (N=6) at cfg=1.0, "the deficit GROWS with the number of transitions."

Two hypotheses this script separates:

  H_inherent    the base sampler's per-step marginal std, plotted against ABSOLUTE model time t,
                traces out one curve regardless of how many outer steps N it took to get there.
                I.e. std(t) is a property of t alone -- the schedule is legitimately non-monotonic /
                contractive in t, N is not the driver, and the N=12 result is just what t=1 looks like.

  H_compounding std at a given absolute t depends on HOW MANY transitions were taken to reach it: the
                N=12 curve sits below the N=6 curve at every matched t, because each of Algorithm 2
                line 6's DDPM transitions (`calc_xbar_s0` draws fresh noise at s=0, `calc_x_t_prime`
                rescales a sufficient statistic of (x_t, x_s) back through `out_interp().alpha`) injects
                slightly too little variance to compensate for what conditioning on x_t removes -- a
                textbook ancestral-sampling variance leak, invisible until you chain enough of them.

The two are observationally distinct: H_inherent predicts the curves overlap when plotted against t;
H_compounding predicts they fan out, worse for larger N, even though every N sweeps the same [0, 1].

This script only measures. It does not touch `creativity_measure` or the upstream clone.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

LABEL = 207          # golden retriever
CFG_SCALE = 4.0       # where the N=6 -> N=12 regression was observed; "the paper's SMC cfg"
INNER = 8             # ruled out as the driver already (N=6,i8 ~= N=6,i16 in the prior sweep)
BATCH = 16
SEED = 0
N_GRID = [4, 6, 8, 12, 16, 24]


def main() -> int:
    info = paths.preflight()
    print(f"gpu={info['gpu']} ({info['gpu_gb']} GB)  jax={info['jax_devices']}", flush=True)
    from creativity_measure.backends.diamond_maps_jax import DiamondMapsBackend

    traces: dict[int, dict[str, list[float]]] = {}
    final_imgs: dict[int, torch.Tensor] = {}

    for n_steps in N_GRID:
        t0 = time.time()
        backend = DiamondMapsBackend(
            label=LABEL, cfg_scale=CFG_SCALE,
            base_ckpt=info["base_ckpt"], posterior_ckpt=info["posterior_ckpt"],
            repo_root=info["diamond_maps_root"],
            n_steps=n_steps, base_inner_steps=INNER, seed=SEED,
        )
        print(f"\nN={n_steps}: backend built in {time.time() - t0:.0f}s", flush=True)

        backend.reset_rng(SEED)
        x = backend.init_particles(BATCH)
        ts = np.asarray(backend.ts)  # (N+1,), ts[0]=0 ... ts[N]=1

        t_hist = [0.0]
        std_hist = [float(x.std())]
        mean_hist = [float(x.mean())]
        print(f"  t=0.000  std={std_hist[-1]:.4f}  mean={mean_hist[-1]:+.4f}", flush=True)

        for step in range(n_steps):
            x = backend.base_step(x, step)
            t_hist.append(float(ts[step + 1]))
            std_hist.append(float(x.std()))
            mean_hist.append(float(x.mean()))
            print(
                f"  t={t_hist[-1]:.3f}  std={std_hist[-1]:.4f}  mean={mean_hist[-1]:+.4f}",
                flush=True,
            )

        traces[n_steps] = {"t": t_hist, "std": std_hist, "mean": mean_hist}

        img = backend.decode(x.reshape(BATCH, -1)).cpu()
        final_imgs[n_steps] = img
        print(
            f"  decoded: std={float(img.std()):.4f} range=[{float(img.min()):.2f},"
            f"{float(img.max()):.2f}]  ({time.time() - t0:.0f}s total)",
            flush=True,
        )
        del backend

    # ---- the decisive comparison: std at MATCHED absolute t, across different N ---------------------
    print(f"\n{'=' * 78}\nstd at matched t (H_inherent: rows agree; H_compounding: std falls with N)"
          f"\n{'=' * 78}", flush=True)
    probe_ts = [0.25, 0.5, 0.75, 1.0]
    header = "t".rjust(6) + "".join(f"N={n:>4}".rjust(10) for n in N_GRID)
    print(header, flush=True)
    for pt in probe_ts:
        row = f"{pt:6.2f}"
        for n_steps in N_GRID:
            tr = traces[n_steps]
            idx = int(np.argmin(np.abs(np.asarray(tr["t"]) - pt)))
            row += f"{tr['std'][idx]:10.4f}"
        print(row, flush=True)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(
        {"traces": traces, "final_imgs": final_imgs, "N_GRID": N_GRID,
         "label": LABEL, "cfg_scale": CFG_SCALE, "inner": INNER, "batch": BATCH, "seed": SEED},
        os.path.join(out_dir, "step_trace_results.pt"),
    )
    print(f"\nsaved {os.path.join(out_dir, 'step_trace_results.pt')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
