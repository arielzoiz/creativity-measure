"""GPU smoke test for the Diamond Maps bridge — the validation ladder, minus the long runs.

Runs the rungs that need real models and real checkpoints, at throwaway settings, so that a failure
shows up in minutes rather than inside a real experiment. Everything here is about *plumbing*: the
statistical calibration (lambda sweep, degeneracy) belongs in the notebook.

    python smoke_test.py            # all rungs
    python smoke_test.py --quick    # skip the slowest (5, 6c)

Rungs, and what each one isolates:

1. assets/env      — checkpoints load, both networks build, no `datasets/` needed
2. bridge          — dlpack/numpy round-trip is bit-exact; score_fn returns sane shapes
3. denoiser/gamma  — E[x1|x_t] -> x1 as t->1 and -> 0 as t->0; score ~ -y/gamma at small gamma
4. reward          — f(x_ref) ~ 1 by construction, pairwise symmetric, zero diagonal
5. lambda=0        — the SMC reproduces the untilted base trajectory exactly
6. posterior path  — the one thing lambda=0 cannot test, since the diamond map only acts through f
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

LABEL = 207          # golden retriever; smoke-test only, the real choice lives in the notebook
CFG_SCALE = 1.0
N_STEPS = 4
M = 4
K = 2
R = 8
N_GAMMA = 4
NUM_EPS = 2


def _banner(msg: str) -> None:
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the slowest rungs")
    args = ap.parse_args()

    t_start = time.time()

    # ---- rung 1: assets and environment ---------------------------------------------------------
    _banner("RUNG 1  assets / environment")
    info = paths.preflight()
    for k, v in info.items():
        print(f"  {k}: {v}")

    from creativity_measure import SquaredGlobalIEMDistance, NormalizedExpectedDistanceReward
    from creativity_measure.backends.diamond_maps_jax import DiamondMapsBackend
    from creativity_measure.diamond_smc import diamond_smc_sample

    t0 = time.time()
    backend = DiamondMapsBackend(
        label=LABEL,
        cfg_scale=CFG_SCALE,
        base_ckpt=info["base_ckpt"],
        posterior_ckpt=info["posterior_ckpt"],
        repo_root=info["diamond_maps_root"],
        n_steps=N_STEPS,
        seed=0,
    )
    print(f"  backend built (both checkpoints loaded) in {time.time() - t0:.0f} s")
    print(f"  latent_shape={backend.latent_shape}  label={backend.label}  cfg={backend.cfg_scale}")

    # ---- rung 2: the array bridge ---------------------------------------------------------------
    _banner("RUNG 2  bridge (dlpack / numpy round-trip)")
    x = torch.randn(3, 4, 32, 32, device=backend.device, dtype=backend.dtype)
    back = backend._to_torch(backend._to_jax(x))
    assert torch.equal(x, back), "round-trip is not bit-exact"
    print(f"  round-trip bit-exact; dlpack={'yes' if backend._dlpack_ok else 'no (numpy fallback)'}")

    score = backend.score_fn
    y = torch.randn(5, 4096, device=backend.device, dtype=backend.dtype)
    s = score(y, torch.tensor(1.0, device=backend.device, dtype=backend.dtype))
    assert s.shape == y.shape, f"score_fn shape {s.shape} != {y.shape}"
    assert torch.isfinite(s).all(), "score_fn produced non-finite values"
    print(f"  score_fn OK: shape {tuple(s.shape)}, |score| mean {float(s.abs().mean()):.4f}")

    # ---- rung 3: denoiser limits and the gamma mapping -------------------------------------------
    _banner("RUNG 3  denoiser / gamma mapping")
    x1 = backend.sample_refs(2).reshape(2, 4, 32, 32)
    eps = torch.randn_like(x1)
    rms_x1 = float(x1.pow(2).mean().sqrt())
    errs = {}
    for t in (0.99, 0.5, 0.02):
        sigma = (1.0 - t) / t
        x_t = t * x1 + (1.0 - t) * eps       # the interpolant: x_t = t*x1 + (1-t)*noise
        y_sigma = x_t / t                    # EDM-convention observation, x1 + sigma*noise
        pred = backend.denoiser(y_sigma, torch.full((2,), sigma, device=backend.device))
        errs[t] = (
            float((pred - x1).pow(2).mean().sqrt()),   # distance to the clean latent
            float(pred.pow(2).mean().sqrt()),          # distance to the prior mean (0)
        )
        print(f"  t={t:<5} sigma={sigma:8.3f} gamma={backend.gamma_of_t(t):10.4g}  "
              f"rms(pred-x1)={errs[t][0]:.3f}  rms(pred)={errs[t][1]:.3f}   [rms(x1)={rms_x1:.3f}]")

    # As t -> 1 the observation is nearly clean, so E[x1|x_t] must collapse onto x1; as t -> 0 there
    # is no information left and it must collapse onto the prior mean. Either failing means the
    # sigma <-> t inversion (and so gamma(t)) is wrong -- the error v1 of the plan would have made.
    assert errs[0.99][0] < 0.5 * rms_x1, "denoiser does not approach x1 as t->1"
    assert errs[0.02][1] < 0.5 * rms_x1, "denoiser does not approach the prior mean as t->0"
    print(f"  gamma window over t in [0.01, 0.99]: {backend.gamma_window()}")

    # ---- rung 4: reward sanity -------------------------------------------------------------------
    _banner("RUNG 4  reward")
    g_lo, g_hi = backend.gamma_window()
    import math

    gammas = torch.logspace(
        math.log2(max(g_lo, 1e-3)), math.log2(min(g_hi, 2.0**10)), N_GAMMA, base=2
    ).to(backend.device)
    refs = backend.sample_refs(R).to(backend.dtype)
    d2 = SquaredGlobalIEMDistance(None, gammas, num_eps=NUM_EPS, seed=123, score_fn=backend.score_fn)
    t0 = time.time()
    reward = NormalizedExpectedDistanceReward(distance=d2, x_refs=refs)
    print(f"  normalizer + reference bank built in {time.time() - t0:.0f} s")

    pw = d2.pairwise(refs, refs)
    assert torch.allclose(pw, pw.T, rtol=1e-4, atol=1e-4), "pairwise D^2 is not symmetric"
    assert float(pw.diag().abs().max()) < 1e-6, "D^2(x, x) != 0"
    # Expected value is exactly (R-1)/R, not 1: the numerator averages D^2(x, x') over ALL R
    # references including x itself (contributing a zero), while the denominator is the OFF-diagonal
    # pair mean (self-pairs dropped, see tilt.reference_pair_mean). So evaluating f ON its own
    # reference set is biased low by one term in R. For a point NOT in the set, f is centred on 1.
    f_refs = reward(refs)
    expected = (R - 1) / R
    print(f"  pairwise symmetric, zero diagonal; f(refs) mean={float(f_refs.mean()):.4f} "
          f"(expect (R-1)/R = {expected:.4f})  std={float(f_refs.std()):.4f}")
    assert abs(float(f_refs.mean()) - expected) < 0.15, (
        f"normalizer is off: f on the reference set should centre on (R-1)/R = {expected:.4f}"
    )

    # ---- rung 6: the posterior lookahead path -----------------------------------------------------
    # Before rung 5, because it is fast and it is the path lambda=0 provably cannot reach.
    _banner("RUNG 6  posterior lookahead (the path lambda=0 cannot test)")
    x_t = backend.init_particles(M)
    x_mid = backend.base_step(x_t, 0)
    z = backend.posterior_sample(x_mid, N_STEPS - 2, K)
    assert z.shape == (M * K, 4, 32, 32), f"posterior shape {z.shape}"
    print(f"  shape OK {tuple(z.shape)}; grouping check (view(M,K)) ...")
    zg = z.view(M, K, -1)
    within = float((zg[:, 0] - zg[:, 1]).pow(2).mean()) if K > 1 else float("nan")
    across = float((zg[0, 0] - zg[1, 0]).pow(2).mean())
    print(f"  mean sq. diff within a particle: {within:.4f}   across particles: {across:.4f}")
    assert within < across, (
        "lookahead draws for the SAME particle are not closer to each other than draws for "
        "DIFFERENT particles — repeat_interleave grouping is wrong"
    )
    if not args.quick:
        # (c) at an intermediate t the posterior must agree in distribution with running the base
        # sampler the rest of the way — that agreement is the entire premise of the diamond map.
        x_full = x_mid.clone()
        for step in range(1, N_STEPS):
            x_full = backend.base_step(x_full, step)
        z1 = backend.posterior_sample(x_mid, 0, 4).reshape(-1, 4096)
        print(f"  base-to-1   per-dim mean={float(x_full.reshape(M, -1).mean()):+.4f} "
              f"std={float(x_full.reshape(M, -1).std()):.4f}")
        print(f"  posterior   per-dim mean={float(z1.mean()):+.4f} std={float(z1.std()):.4f}")

    # ---- rung 5: lambda = 0 reproduces the base sampler -------------------------------------------
    if not args.quick:
        _banner("RUNG 5  lambda = 0 reproduces the base sampler")
        # Reuse the one loaded backend (rewinding its RNG) rather than building two more -- each
        # rebuild costs another 10.3 GB of unpickling. This comparison is EXACT because the backend
        # keeps independent key streams for base steps and lookahead draws: at lambda=0 the lookahead
        # still runs (and is multiplied by zero) but cannot perturb the base trajectory.
        backend.reset_rng(42)
        res = diamond_smc_sample(
            reward, 0.0, M, backend=backend, n_steps=N_STEPS, mc_samples=K, seed=0
        )
        backend.reset_rng(42)
        x = backend.init_particles(M)
        for step in range(N_STEPS):
            x = backend.base_step(x, step)
        max_dev = float((res.X - x).abs().max())
        print(f"  logw all zero: {bool(torch.allclose(res.logw, torch.zeros_like(res.logw)))}")
        print(f"  uniq/M at every step: {res.uniq_history}")
        print(f"  max |smc - base| = {max_dev:.3e}")
        assert torch.allclose(res.logw, torch.zeros_like(res.logw), atol=1e-9)
        assert all(u == 1.0 for u in res.uniq_history)
        assert max_dev < 1e-4, (
            "lambda=0 did not reproduce the base sampler: the SMC loop is perturbing the trajectory "
            "it should only reweight, or the base/posterior key streams are not independent"
        )
        backend.reset_rng(0)

    # ---- a real (tiny) tilted run -----------------------------------------------------------------
    _banner("END-TO-END  tiny tilted run")
    t0 = time.time()
    res = diamond_smc_sample(
        reward, 1.0, M, backend=backend, n_steps=N_STEPS, mc_samples=K, seed=0, verbose=True
    )
    print(f"  {N_STEPS} steps on {M} particles in {time.time() - t0:.0f} s")
    print(f"  final f: {[round(v, 4) for v in reward(res.X.reshape(M, -1)).tolist()]}")

    img = backend.decode(res.X.reshape(M, -1))
    assert img.shape[0] == M and img.shape[1] == 3, f"decode shape {img.shape}"
    print(f"  decoded to {tuple(img.shape)} in [{float(img.min()):.2f}, {float(img.max()):.2f}]")

    _banner(f"ALL RUNGS PASSED in {time.time() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
