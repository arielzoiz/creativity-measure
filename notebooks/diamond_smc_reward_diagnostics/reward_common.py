"""Shared backend/reward construction for the reward-diagnostics experiments in this directory.

Mirrors `notebooks/diamond_smc_imagenet/diamond_smc_imagenet.ipynb` cells 1-5 exactly -- same p
(label=207, cfg_scale=1.0), same reward (NormalizedExpectedDistanceReward + SquaredGlobalIEMDistance,
R=64) -- so results here are directly comparable to that notebook's own run (job 698463,
`results_label207_cfg1.0_M16K4N6.pt`). Scripted once so `m_sweep.py` and `M_sweep.py` don't each
duplicate the ~30 lines of reward setup. Within-directory import only, like `paths.py` -- not part of
the library, per the convention documented in `notebooks/diamond_smc_imagenet/README.md`.

Named `reward_common.py`, not `common.py`: the upstream diamond_maps clone has its own top-level
`common` package (`posterior_diamond_maps/py/common/`), imported inside
`creativity_measure.backends.diamond_maps_jax` as `from common import latent_utils, sampling, ...`.
`paths._add_repo_to_path` puts that package's parent dir on `sys.path`, but THIS directory is already
on `sys.path` at position 0 (every script here does `sys.path.insert(0, ...)` for `paths`/this module),
so a same-named `common.py` shadows upstream's package instead of the intended one -- silently, since
both import without error until upstream's `common` is asked for a name only it has. Cost one failed
job (905863/905864) to find.
"""

from __future__ import annotations

import math
import time

import torch

LABEL = 207
CFG_SCALE = 1.0
N = 6
INNER = 8
R = 64
N_GAMMA = 6
NUM_EPS = 3
BROWNIAN_SEED = 123
SEED = 0


def build_backend_and_reward(info: dict, *, probe_m: int):
    """Backend + frozen reward + lambda_s, built exactly as the notebook does it.

    `probe_m` is the batch size for the one-shot lambda_s = 1/std_p(f) probe -- pass the sweep's own
    M so lambda_s is measured the same way the notebook measured it (M=16 there).
    """
    from creativity_measure import NormalizedExpectedDistanceReward, SquaredGlobalIEMDistance
    from creativity_measure.backends.diamond_maps_jax import DiamondMapsBackend

    t0 = time.time()
    backend = DiamondMapsBackend(
        label=LABEL, cfg_scale=CFG_SCALE,
        base_ckpt=info["base_ckpt"], posterior_ckpt=info["posterior_ckpt"],
        repo_root=info["diamond_maps_root"],
        n_steps=N, base_inner_steps=INNER, mc_inner_steps=1, seed=SEED,
    )
    print(f"backend built in {time.time() - t0:.0f}s", flush=True)

    x_refs = backend.sample_refs(R)
    S = float(x_refs.std())
    g_lo_model, g_hi_model = backend.gamma_window()
    GAMMA_LO = max(1.0 / S**2, g_lo_model)
    GAMMA_HI = min(2.0**10, g_hi_model)
    GAMMAS = torch.logspace(
        math.log2(GAMMA_LO), math.log2(GAMMA_HI), N_GAMMA, base=2
    ).to(device=x_refs.device, dtype=x_refs.dtype)

    D2 = SquaredGlobalIEMDistance(
        None, GAMMAS, num_eps=NUM_EPS, seed=BROWNIAN_SEED, score_fn=backend.score_fn
    )
    reward = NormalizedExpectedDistanceReward(distance=D2, x_refs=x_refs)
    f_refs = reward(x_refs)
    print(f"f(refs): mean={float(f_refs.mean()):.4f} (expect {(R - 1) / R:.4f})", flush=True)

    x_probe = backend.sample_refs(probe_m)
    f_probe = reward(x_probe)
    f_std = float(f_probe.std())
    lam_s = (1.0 / f_std) if f_std > 0 else float("inf")
    print(f"lambda_s = 1/std(f) = {lam_s:.3f}  (probe M={probe_m})", flush=True)

    return backend, reward, lam_s
