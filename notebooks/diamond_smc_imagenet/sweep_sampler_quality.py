"""Which base-sampler settings actually produce well-dispersed latents?

Run 698463 generated latents with std ~0.39 where scaled SD-VAE latents should be ~1, and the decoded
images were washed out. The A/B (ab_base_sampler.py, job 698517) showed my per-step driving and
upstream's own full-scan sampler agree statistically (0.397 vs 0.405), so this is a property of the
CONFIGURATION, not the bridge.

This is pure measurement, no interpretation: for each setting, generate a batch and report the latent
std and the decoded contrast. Reference points:
  - scaled SD-VAE latents should have std ~1.0
  - upstream's own EVAL config generates samples with the DIAMOND MAP in one step (outer=1, inner=1)
    at cfg 3.0 -- not with a multi-step GLASS run -- so the diamond map is included as a row.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

BATCH = 16
LABEL = 207
SEED = 0

# (cfg_scale, n_steps, base_inner_steps)
GRID = [
    (1.0, 6, 8),      # what run 698463 used
    (4.0, 6, 8),      # the paper's SMC cfg
    (4.0, 6, 16),     # more accurate inner GLASS integration
    (4.0, 12, 8),     # more transitions
    (4.0, 12, 16),    # both
]


def main() -> int:
    info = paths.preflight()
    from creativity_measure.backends.diamond_maps_jax import DiamondMapsBackend

    print(f"\n{'cfg':>5} {'N':>4} {'inner':>6} {'lat std':>9} {'lat mean':>9} "
          f"{'img std':>8} {'img range':>14}   (target lat std ~1.0)", flush=True)

    for cfg_scale, n_steps, inner in GRID:
        backend = DiamondMapsBackend(
            label=LABEL, cfg_scale=cfg_scale,
            base_ckpt=info["base_ckpt"], posterior_ckpt=info["posterior_ckpt"],
            repo_root=info["diamond_maps_root"],
            n_steps=n_steps, base_inner_steps=inner, seed=SEED,
        )
        backend.reset_rng(SEED)
        x = backend.sample_refs(BATCH)
        img = backend.decode(x).cpu()
        print(f"{cfg_scale:5.1f} {n_steps:4d} {inner:6d} {float(x.std()):9.4f} "
              f"{float(x.mean()):+9.4f} {float(img.std()):8.3f} "
              f"[{float(img.min()):.2f},{float(img.max()):.2f}]".rjust(14), flush=True)
        torch.save(img, f"sweepimg_cfg{cfg_scale}_N{n_steps}_i{inner}.pt")
        del backend

    # The diamond map as a one-step sampler from noise -- what upstream's eval config uses.
    backend = DiamondMapsBackend(
        label=LABEL, cfg_scale=4.0,
        base_ckpt=info["base_ckpt"], posterior_ckpt=info["posterior_ckpt"],
        repo_root=info["diamond_maps_root"], n_steps=6, base_inner_steps=8, seed=SEED,
    )
    backend.reset_rng(SEED)
    x0 = backend.init_particles(BATCH)
    z = backend.posterior_sample(x0, -1, 1)          # step_idx -1 -> t_prime = ts[0] = 0.0
    img = backend.decode(z.reshape(BATCH, -1)).cpu()
    print(f"\ndiamond map, 1 step from noise (upstream's eval path): "
          f"lat std={float(z.std()):.4f} img std={float(img.std()):.3f} "
          f"range [{float(img.min()):.2f},{float(img.max()):.2f}]", flush=True)
    torch.save(img, "sweepimg_diamond1step.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
