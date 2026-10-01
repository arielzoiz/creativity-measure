"""Phase 3 of notebooks/iid_iem_flux_check/ROADMAP.md: direct test-time guidance on FLUX.1-dev.

Sweeps a small lambda grid x {approximate, exact} Jacobian, recording terminal reward against a
lambda=0 control on the SAME initial noise (paired, per CLAUDE.md: "always subtract a lambda=0
control" -- the untilted base process alone has been measured to give back several std_p(f) with no
tilt at all, cf. flowmap_smc_k_sweep/RESULTS.md).

CONVENTION NOTE (this repo has a documented history of exactly this class of error, see
creativity_measure/flux_guided.py's module docstring): every t here is diffusers-native
(t=1 noise, t=0 data). The artifact-free endpoint for reading E_q[f] is therefore t_end=0, the
OPPOSITE polarity to CLAUDE.md's "only t=1 is artifact-free" note, which was written for
flowmap_smc's t=0-noise/t=1-data convention. Always read this sweep's f at t_end=0.

    python guided_sweep.py            # on a GPU node (see guided_sweep.slurm)
    python guided_sweep.py --dry-run  # CPU, tiny FluxTransformer2DModel: exercises all the plumbing,
                                       # no GPU needed. Does NOT exercise the OOM fallback's gamma-chunk
                                       # branch (needs SquaredIIDGlobalIEMDistance.expected_gamma_chunk,
                                       # ROADMAP.md Phase 3 step 1b) -- that branch only fires on a real
                                       # or forced OOM, neither of which a normal sweep triggers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "refset_auto_r"))

from creativity_measure import (                                          # noqa: E402
    NormalizedExpectedDistanceReward, SquaredIIDGlobalIEMDistance, log_uniform_gammas,
)
from creativity_measure.distances.edm_adapter import edm_score_fn                          # noqa: E402
from creativity_measure.flux_guided import flux_guided_sample                              # noqa: E402
from creativity_measure.generators.flux import flux_edm_denoiser, flux_velocity_fn          # noqa: E402

RESULTS = os.path.join(HERE, "guided_sweep_results.json")
R_REFS = 8                       # small on purpose: this sweep stresses the guidance mechanics, not R
N_HELDOUT = 4                    # trajectories per config
N_STEPS = 6
LAMBDAS = (0.0, 0.5, 1.0, 2.0)   # 0.0 is the paired control, read at every seed
JACOBIAN_MODES = (False, True)   # exact_jacobian


@dataclass
class Setup:
    device: torch.device
    transformer: Any
    velocity_fn: Callable[[Tensor, float], Tensor]
    reward: NormalizedExpectedDistanceReward
    d: int
    dtype: torch.dtype
    stamp: dict


def _build_reward(denoiser, img_shape: tuple[int, int, int], d: int, dtype: torch.dtype,
                   device: torch.device) -> NormalizedExpectedDistanceReward:
    score_fn = edm_score_fn(denoiser, img_shape)
    gammas, gweights = log_uniform_gammas(2.0 ** -6, 2.0 ** 6, 12, seed=0, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(0)
    x_refs = torch.randn(R_REFS, d, generator=gen, dtype=dtype, device=device)
    dist = SquaredIIDGlobalIEMDistance(None, gammas, gweights, num_eps=1, seed=1, score_fn=score_fn)
    # Ref bank is grad-free (iid_global_iem.py detaches x_refs and the bank), but building it under a
    # differentiable=True denoiser with no outer torch.no_grad() OOM'd job 957050 at 43.9/44.5 GiB on a
    # ref bank this small: several kernels pick a different, more memory-hungry path purely from
    # torch.is_grad_enabled(), independent of any tensor's requires_grad. flux_guided_sample already
    # wraps its own ref-bank-forcing call in no_grad for this reason; this constructor call needs it too.
    with torch.no_grad():
        return NormalizedExpectedDistanceReward(dist, x_refs)


def dry_setup() -> Setup:
    """A tiny real FluxTransformer2DModel (same recipe as tests/test_flux_guided.py), CPU, no GPU."""
    import diffusers
    from diffusers import FluxPipeline

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

    gen = torch.Generator().manual_seed(0)
    prompt_embeds = torch.randn((1, 3, transformer.config.joint_attention_dim), generator=gen)
    pooled_prompt_embeds = torch.randn((1, transformer.config.pooled_projection_dim), generator=gen)
    txt_ids = torch.zeros((3, 3))
    img_ids = FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)

    velocity_fn = flux_velocity_fn(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=1.0, img_shape=(c, h, w), img_px=img_px, dtype=dtype, differentiable=True,
    )
    denoiser = flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=1.0, img_shape=(c, h, w), img_px=img_px, dtype=dtype, differentiable=True,
    )
    reward = _build_reward(denoiser, (c, h, w), d, dtype, device)
    return Setup(device, transformer, velocity_fn, reward, d, dtype, {"dry_run": True, "d": d})


def flux_setup() -> Setup:
    """The real FLUX.1-dev path. Mirrors notebooks/iid_iem_flux_check/iid_vs_brownian.py's flux_setup,
    but builds its own velocity_fn/denoiser directly (build_flux_denoiser's public surface is the
    denoiser only; this needs the raw velocity too, so it duplicates build_flux_denoiser's loading
    steps rather than partially reusing it -- see generators/flux.py for why a shared loader was not
    factored out: prompt encoding + weight loading is a few lines, not worth a third entry point)."""
    import auto_r_common as arc                                # pyright: ignore[reportMissingImports]
    from diffusers import FluxPipeline

    device = torch.device("cuda")
    dtype = torch.bfloat16
    img = 512
    c, h, w = 16, img // 8, img // 8
    d = c * h * w

    pipe = FluxPipeline.from_pretrained(arc.MODEL_ID, torch_dtype=dtype).to(device)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=arc.PROMPT, prompt_2=None, device=device, max_sequence_length=512)
    assert prompt_embeds is not None and pooled_prompt_embeds is not None, "encode_prompt returned None"
    pipe.text_encoder = pipe.text_encoder_2 = pipe.tokenizer = pipe.tokenizer_2 = None
    torch.cuda.empty_cache()
    transformer = pipe.transformer.eval()
    img_ids = FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)
    txt_ids = text_ids if text_ids.ndim == 2 else text_ids[0]

    velocity_fn = flux_velocity_fn(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=arc.GUIDANCE, img_shape=(c, h, w), img_px=img, dtype=dtype, differentiable=True,
    )
    denoiser = flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=arc.GUIDANCE, img_shape=(c, h, w), img_px=img, dtype=dtype, differentiable=True,
    )
    reward = _build_reward(denoiser, (c, h, w), d, dtype, device)
    return Setup(device, transformer, velocity_fn, reward, d, dtype,
                 {"dry_run": False, "d": d, "model": arc.MODEL_ID, "gpu": torch.cuda.get_device_name(0)})


# =====================================================================================================
# Results file (atomic, resumable per config) -- same convention as iid_vs_brownian.py
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


def run_config(S: Setup, lam: float, exact_jacobian: bool, seed: int, res: dict) -> None:
    key = f"lam{lam}_exact{exact_jacobian}_seed{seed}"
    if key in res:
        note(f"{key} already done: f={res[key]['f_mean']:.5f}")
        return
    t0 = time.time()
    result = flux_guided_sample(
        S.reward, lam, N_HELDOUT, velocity_fn=S.velocity_fn, n_steps=N_STEPS,
        exact_jacobian=exact_jacobian, t_start=1.0, t_end=0.0, seed=seed,
    )
    with torch.no_grad():
        f = S.reward(result.X)
    res[key] = {
        "lam": lam, "exact_jacobian": exact_jacobian, "seed": seed,
        "f_mean": float(f.mean()), "f_std": float(f.std()),
        "applied_norm_mean": sum(result.applied_norm_history) / len(result.applied_norm_history),
        "oom_fallback_any": any(result.oom_fallback_history),
        "elapsed_s": time.time() - t0,
    }
    save_results(res)
    note(f"{key}: f mean {res[key]['f_mean']:.5f} sd {res[key]['f_std']:.5f}  "
         f"({res[key]['elapsed_s']:.0f}s)")


def print_table(res: dict) -> None:
    print("=" * 90)
    print(f"{'lam':>6s} {'exact':>6s} {'seed':>5s}  {'f mean':>9s} {'f sd':>9s}  {'vs control':>11s}")
    for jac in JACOBIAN_MODES:
        control = None
        for lam in LAMBDAS:
            rows = [v for k, v in res.items() if k.startswith(f"lam{lam}_exact{jac}_")]
            if not rows:
                continue
            f_mean = sum(r["f_mean"] for r in rows) / len(rows)
            f_sd = sum(r["f_std"] for r in rows) / len(rows)
            if lam == 0.0:
                control = f_mean
            delta = "-" if control is None else f"{f_mean - control:+.5f}"
            print(f"{lam:>6.2f} {str(jac):>6s} {len(rows):>5d}  {f_mean:>9.5f} {f_sd:>9.5f}  {delta:>11s}")
    print("=" * 90)
    print("delta is vs the SAME jacobian-mode's lam=0 control (CLAUDE.md: always subtract a lam=0")
    print("control). Read only at t_end=0 (the clean-image endpoint) -- see module docstring for why")
    print("that is the opposite t-polarity to CLAUDE.md's flowmap-SMC note.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    global RESULTS
    if args.dry_run:
        RESULTS = os.path.join(HERE, "guided_sweep_results.dryrun.json")
        if os.path.exists(RESULTS):
            os.remove(RESULTS)

    S = dry_setup() if args.dry_run else flux_setup()
    res = load_results({**S.stamp, "n_heldout": N_HELDOUT, "n_steps": N_STEPS,
                        "lambdas": list(LAMBDAS), "jacobian_modes": list(JACOBIAN_MODES)})
    note(f"setup ready: d={S.d}, device={S.device}")

    for exact_jacobian in JACOBIAN_MODES:
        for lam in LAMBDAS:
            run_config(S, lam, exact_jacobian, seed=0, res=res)

    print_table(res)
    note("done")


if __name__ == "__main__":
    main()
