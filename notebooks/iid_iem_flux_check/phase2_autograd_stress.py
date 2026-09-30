"""Phase 2 of notebooks/iid_iem_flux_check/ROADMAP.md: autograd + memory stress test for the i.i.d.
squared-IEM reward on FLUX.1-dev.

Wraps the reward itself (not just the denoiser -- that plumbing is unit-tested in
tests/test_flux_denoiser.py) in torch.autograd.grad and checks the three safety mechanisms the roadmap
names, each as a real, reproducible PASS/FAIL, each appending to phase2_autograd_stress_results.json:

  weight_freeze     Every transformer parameter has requires_grad=False after
                     build_flux_denoiser(differentiable=True); only the input latent carries a gradient.
  double_backward   grad_x r(x) alone is a single backward and needs no special handling; create_graph=True
                     (double-backward) can raise a Flash-Attention derivative-support error, caught and
                     retried under torch.nn.attention.sdpa_kernel(SDPBackend.MATH).
  oom_fallback      On a (real or forced) OOM, looping torch.autograd.grad per MC sample into a running
                     total is EXACT (expected() is a sum of independent per-sample terms plus a constant) --
                     verified here by comparing to the batched gradient directly, no OOM needed to check it.

    python phase2_autograd_stress.py            # on a GPU node, once build_flux_denoiser is GPU-verified
    python phase2_autograd_stress.py --dry-run  # CPU, tiny FluxTransformer2DModel: exercises all the
                                                 # plumbing and all three safety mechanisms, no GPU needed
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))

from creativity_measure import (                                   # noqa: E402
    NormalizedExpectedDistanceReward, SquaredIIDGlobalIEMDistance, log_uniform_gammas,
)
from creativity_measure.generators.flux import build_flux_denoiser, flux_edm_denoiser  # noqa: E402
from creativity_measure.distances.edm_adapter import edm_score_fn                       # noqa: E402

RESULTS = os.path.join(HERE, "phase2_autograd_stress_results.json")
R_REFS = 8                  # small on purpose: this script stresses the autograd mechanics, not R itself
M_BATCH = 6                  # batch size for the OOM-fallback exactness check


@dataclass
class Setup:
    device: torch.device
    transformer: nn.Module
    reward: Callable[[Tensor], Tensor]      # x: (B, d), differentiable in x -> f(x): (B,)
    d: int
    dtype: torch.dtype
    stamp: dict


# =====================================================================================================
# Setups: the real one (FLUX) and a CPU stand-in that exercises the identical code path
# =====================================================================================================

def _build_reward(denoiser, transformer: nn.Module, img_shape: tuple[int, int, int], d: int,
                   dtype: torch.dtype, device: torch.device) -> Callable[[Tensor], Tensor]:
    """Wire a differentiable Denoiser into the same reward class Phase 1 verifies (SquaredIIDGlobalIEMDistance
    + NormalizedExpectedDistanceReward), with a small frozen reference bank -- the refs' own values don't
    matter here (this script stresses autograd/memory mechanics, not novelty quality)."""
    score_fn = edm_score_fn(denoiser, img_shape)
    gammas, gweights = log_uniform_gammas(2.0 ** -6, 2.0 ** 6, 12, seed=0, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(0)
    x_refs = torch.randn(R_REFS, d, generator=gen, dtype=dtype, device=device)
    dist = SquaredIIDGlobalIEMDistance(None, gammas, gweights, num_eps=1, seed=1, score_fn=score_fn)
    reward = NormalizedExpectedDistanceReward(dist, x_refs)
    return reward


def dry_setup() -> Setup:
    """A tiny real FluxTransformer2DModel (same recipe as tests/test_flux_denoiser.py), CPU, no GPU."""
    import diffusers

    c, h, w = 16, 4, 4
    d = c * h * w
    img_px = h * 8
    dtype = torch.float32
    device = torch.device("cpu")

    transformer: Any = diffusers.FluxTransformer2DModel(
        patch_size=1, in_channels=c * 4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
        pooled_projection_dim=16, guidance_embeds=True, axes_dims_rope=(2, 4, 2),
    ).eval()
    transformer.requires_grad_(False)

    from diffusers import FluxPipeline
    gen = torch.Generator().manual_seed(0)
    prompt_embeds = torch.randn((1, 3, transformer.config.joint_attention_dim), generator=gen)
    pooled_prompt_embeds = torch.randn((1, transformer.config.pooled_projection_dim), generator=gen)
    txt_ids = torch.zeros((3, 3))
    img_ids = FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)

    denoiser = flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=1.0, img_shape=(c, h, w), img_px=img_px, dtype=dtype, differentiable=True,
    )
    reward = _build_reward(denoiser, transformer, (c, h, w), d, dtype, device)
    return Setup(device, transformer, reward, d, dtype, {"dry_run": True, "d": d})


def flux_setup() -> Setup:
    """The real FLUX.1-dev path, via build_flux_denoiser(differentiable=True)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    img = 512
    c, h, w = 16, img // 8, img // 8
    d = c * h * w

    denoiser = build_flux_denoiser(device=device, dtype=dtype, img=img, differentiable=True)
    # build_flux_denoiser freezes its own transformer; there is no handle to it here, so the
    # weight-freeze check below re-derives it from the same module-global diffusers pipeline is not
    # possible through this function alone -- flux_setup exposes no transformer handle by design (the
    # denoiser is the public surface). The weight-freeze check therefore only runs under --dry-run,
    # where the transformer is directly available; on GPU it is implied by build_flux_denoiser's own
    # requires_grad_(False) call, exercised by tests/test_flux_denoiser.py.
    reward = _build_reward(denoiser, nn.Module(), (c, h, w), d, dtype, device)
    return Setup(device, nn.Module(), reward, d, dtype,
                 {"dry_run": False, "d": d, "model": "black-forest-labs/FLUX.1-dev",
                  "gpu": torch.cuda.get_device_name(0)})


# =====================================================================================================
# Results file (atomic, resumable per stage) -- same convention as iid_vs_brownian.py
# =====================================================================================================

def load_results(stamp: dict) -> dict:
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            old = json.load(fh)
        if old.get("stamp") == stamp:
            print(f"[resume] {RESULTS}: {sorted(k for k in old if k != 'stamp')}")
            return old
        os.replace(RESULTS, RESULTS + ".stale")
        print("[resume] stamp mismatch -> old results moved to .stale, starting fresh")
    return {"stamp": stamp}


def save_results(res: dict) -> None:
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=2)
    os.replace(tmp, RESULTS)


def note(msg: str) -> None:
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform != "darwin" else 1024 ** 3)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}  (host peak RSS {rss:.1f} GB)", flush=True)


def _sample_latent(S: Setup, b: int, seed: int) -> Tensor:
    gen = torch.Generator(device=S.device).manual_seed(seed)
    return torch.randn(b, S.d, generator=gen, dtype=S.dtype, device=S.device)


# =====================================================================================================
# Safety mechanism 1 -- explicit weight freezing
# =====================================================================================================

def run_weight_freeze(S: Setup, res: dict) -> None:
    if "weight_freeze" in res:
        note(f"weight_freeze already done: {res['weight_freeze']['verdict']}")
        return
    params = list(S.transformer.parameters())
    frozen = len(params) > 0 and all(not p.requires_grad for p in params)

    x = _sample_latent(S, 2, seed=100).requires_grad_(True)
    out = S.reward(x)
    ok = frozen and bool(out.requires_grad)
    res["weight_freeze"] = {
        "verdict": "PASS" if ok else "FAIL",
        "n_transformer_params": len(params), "transformer_frozen": frozen,
        "reward_output_requires_grad": bool(out.requires_grad),
    }
    save_results(res)
    note(f"weight_freeze {res['weight_freeze']['verdict']}: "
         f"{len(params)} params frozen={frozen}, reward.requires_grad={bool(out.requires_grad)}")


# =====================================================================================================
# Safety mechanism 2 -- Flash-Attention double-backward trap
# =====================================================================================================

def run_double_backward(S: Setup, res: dict) -> None:
    if "double_backward" in res:
        note(f"double_backward already done: {res['double_backward']['verdict']}")
        return
    x = _sample_latent(S, 2, seed=101).requires_grad_(True)

    out = S.reward(x)
    (grad,) = torch.autograd.grad(out.sum(), x)
    single_ok = bool(torch.isfinite(grad).all())

    trap_hit = False
    try:
        x2 = _sample_latent(S, 2, seed=101).requires_grad_(True)
        out2 = S.reward(x2)
        (g1,) = torch.autograd.grad(out2.sum(), x2, create_graph=True)
        (g2,) = torch.autograd.grad(g1.sum(), x2, retain_graph=True)
        fallback_ok = bool(torch.isfinite(g2).all())
    except RuntimeError as e:
        trap_hit = True
        note(f"double_backward: default SDPA backend raised ({e}); retrying under SDPBackend.MATH")
        from torch.nn.attention import SDPBackend, sdpa_kernel
        x2 = _sample_latent(S, 2, seed=101).requires_grad_(True)
        with sdpa_kernel(SDPBackend.MATH):
            out2 = S.reward(x2)
            (g1,) = torch.autograd.grad(out2.sum(), x2, create_graph=True)
            (g2,) = torch.autograd.grad(g1.sum(), x2, retain_graph=True)
        fallback_ok = bool(torch.isfinite(g2).all())

    ok = single_ok and fallback_ok
    res["double_backward"] = {
        "verdict": "PASS" if ok else "FAIL",
        "single_backward_ok": single_ok, "trap_hit_on_default_backend": trap_hit,
        "double_backward_ok_after_fallback": fallback_ok,
    }
    save_results(res)
    note(f"double_backward {res['double_backward']['verdict']} (trap_hit={trap_hit})")


# =====================================================================================================
# Safety mechanism 3 -- iterative gradient accumulation (OOM fallback), checked for exactness
# =====================================================================================================

def run_oom_fallback(S: Setup, res: dict) -> None:
    if "oom_fallback" in res:
        note(f"oom_fallback already done: {res['oom_fallback']['verdict']}")
        return
    x = _sample_latent(S, M_BATCH, seed=102)

    x_batched = x.clone().requires_grad_(True)
    loss_batched = S.reward(x_batched).sum()
    (grad_batched,) = torch.autograd.grad(loss_batched, x_batched)

    total_grad = torch.zeros_like(x)
    for i in range(M_BATCH):
        xi = x[i:i + 1].clone().requires_grad_(True)
        loss = S.reward(xi).sum()
        (gi,) = torch.autograd.grad(loss, xi)
        total_grad[i:i + 1] = gi
        del loss
        if S.device.type == "cuda":
            torch.cuda.empty_cache()

    max_abs_diff = float((grad_batched - total_grad).abs().max())
    ok = max_abs_diff < 1e-3
    res["oom_fallback"] = {"verdict": "PASS" if ok else "FAIL", "max_abs_diff": max_abs_diff,
                            "rule": "per-sample accumulation must match the batched gradient (exact, "
                                    "up to floating-point reduction order): expected() is a sum of "
                                    "independent per-sample terms plus a constant"}
    save_results(res)
    note(f"oom_fallback {res['oom_fallback']['verdict']}: max abs diff {max_abs_diff:.2e}")


# =====================================================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="CPU stand-in for FLUX; writes to a temp results file")
    args = ap.parse_args()
    global RESULTS
    if args.dry_run:
        RESULTS = os.path.join(HERE, "phase2_autograd_stress_results.dryrun.json")
        if os.path.exists(RESULTS):
            os.remove(RESULTS)

    S = dry_setup() if args.dry_run else flux_setup()
    res = load_results(S.stamp)
    note(f"setup ready: d={S.d}, device={S.device}")

    run_weight_freeze(S, res)
    run_double_backward(S, res)
    run_oom_fallback(S, res)

    verdicts = {k: res[k]["verdict"] for k in ("weight_freeze", "double_backward", "oom_fallback")}
    overall = "PASS" if all(v == "PASS" for v in verdicts.values()) else "FAIL"
    print("=" * 60)
    print(f"Phase 2 safety mechanisms: {overall}")
    for k, v in verdicts.items():
        print(f"  {k:16s} {v}")
    print("=" * 60)
    note("done")


if __name__ == "__main__":
    main()
