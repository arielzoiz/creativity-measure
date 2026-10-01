"""Phase 3, follow-up to flux_guided_lambda_sweep.ipynb: a FINE lambda sweep, exact_jacobian=True only,
to see WHERE between m=0 (untilted) and m=1 (already collapsed to off-manifold noise, per the completed
notebook's results) the image actually breaks down.

Uses the SAME reward config as the notebook (R_REFS=64, N_GAMMA=50, NUM_EPS=1, same seeds) so lambda_s
reproduces the notebook's measured ~81.0 and these lambdas are directly comparable to that run's m=0/m=1
points. N_LAMBDAS values of lambda, uniformly spaced in [0, lambda_s] (i.e. m in [0, 1]), SAME initial
noise z0 across the whole sweep (paired design) so lambda is the only variable.

UNLIKE guided_sweep.py / the notebook: this is a STANDALONE RESUMABLE SCRIPT, not nbconvert-executed --
job 957442 (the notebook's first, untrimmed run) timed out at --time=180 with nbconvert's "all output
buffered until the end" behaviour, which meant a timeout lost every result computed so far. Here, each
lambda's image + numbers are written to disk the moment they're computed (results JSON, atomic + resumable
by key, same convention as guided_sweep.py/iid_vs_brownian.py), so a timeout or preemption only costs the
in-flight config -- resubmitting picks up where it left off. Build the montage/table separately with
render_fine_sweep.py, which only needs what's already on disk (no GPU, runs any time, including mid-sweep).

CONVENTION NOTE: t is diffusers-native (t=1 noise, t=0 data) throughout, same as flow_guided.py and
guided_sweep.py -- read E_q[f] at t_end=0 only (the OPPOSITE polarity to CLAUDE.md's flowmap-SMC note).

    python fine_lambda_sweep.py            # on a GPU node (see fine_lambda_sweep.slurm)
    python fine_lambda_sweep.py --dry-run  # CPU, tiny FluxTransformer2DModel, no image decode (no VAE
                                            # in the dry setup) -- exercises the sweep/resume mechanics.
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

from creativity_measure import (                                                           # noqa: E402
    NormalizedExpectedDistanceReward, SquaredIIDGlobalIEMDistance, log_uniform_gammas,
)
from creativity_measure.distances.edm_adapter import chunked_denoiser, edm_score_fn          # noqa: E402
from creativity_measure.flow_guided import flow_guided_sample                                # noqa: E402
from creativity_measure.generators.base import edm_generator                                 # noqa: E402
from creativity_measure.generators.flux import flux_edm_denoiser, flux_velocity_fn           # noqa: E402

RESULTS = os.path.join(HERE, "fine_lambda_sweep_results.json")
DECODED_DIR = os.path.join(HERE, "fine_decoded")

N_LAMBDAS = 15             # uniform in [0, lambda_s], i.e. m in [0, 1] -- the collapse already seen at m=1
N_PARTICLES = 1            # one image per lambda (this is a "where along the way" scan, not a statistics run)
N_STEPS_ODE = 10            # matches the completed notebook run, for direct comparability
SHIFT = 3.0
SWEEP_SEED = 1234          # SAME z0 for every lambda (paired), same seed as the notebook's sweep

R_REFS = 64                 # CLAUDE.md: R=64, RandomRefs, uniform weights
N_GAMMA, NUM_EPS = 50, 1    # job 956556's G1 best-tie config, same as the notebook
PROBE_SIZE = 32
REF_SEED, PROBE_SEED = 7, 9990
SIGMA_MIN, SIGMA_MAX = 2e-3, 80.0
N_STEPS_GEN = 8
MAX_DENOISER_ROWS = 24
MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT = "A dog"
GUIDANCE = 1.5


@dataclass
class Setup:
    device: torch.device
    dtype: torch.dtype
    velocity_fn: Callable[[Tensor, float], Tensor]
    reward: NormalizedExpectedDistanceReward
    d: int
    lam_s: float
    decode: Callable[[Tensor], Tensor] | None   # None in --dry-run (no VAE built there)
    stamp: dict


def _build_reward_and_lam_s(denoiser, G, img_shape, d, dtype, device) -> tuple[NormalizedExpectedDistanceReward, float, dict]:
    score_fn = edm_score_fn(denoiser, img_shape)
    gen = torch.Generator(device="cpu").manual_seed(REF_SEED)
    with torch.no_grad():
        latent_refs = G(torch.randn(R_REFS, d, generator=gen).to(device))
    S_scale = latent_refs.std().item()
    gamma_lo = max(1.0 / S_scale ** 2, 1.0 / SIGMA_MAX ** 2)
    gamma_hi = min(2.0 ** 10, 1.0 / SIGMA_MIN ** 2)
    gammas, gweights = log_uniform_gammas(gamma_lo, gamma_hi, N_GAMMA, seed=123, dtype=torch.float32)
    gammas, gweights = gammas.to(device), gweights.to(device)
    dist = SquaredIIDGlobalIEMDistance(None, gammas, gweights, num_eps=NUM_EPS, seed=123, score_fn=score_fn)
    # grad-free by construction, but torch.is_grad_enabled() alone can push kernels onto a more
    # memory-hungry path under a differentiable=True denoiser (job 957050) -- wrap regardless.
    with torch.no_grad():
        reward = NormalizedExpectedDistanceReward(dist, latent_refs)

    probe_gen = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    with torch.no_grad():
        z_probe = G(torch.randn(PROBE_SIZE, d, generator=probe_gen).to(device))
        f_probe = reward(z_probe)
    f_std_p = float(f_probe.std())
    lam_s = (1.0 / f_std_p) if f_std_p > 0 else float("inf")
    info = {"S_scale": S_scale, "gamma_lo": gamma_lo, "gamma_hi": gamma_hi,
            "f_probe_mean": float(f_probe.mean()), "f_std_p": f_std_p}
    return reward, lam_s, info


def dry_setup() -> Setup:
    """Tiny real FluxTransformer2DModel (same recipe as tests/test_flow_guided.py), CPU, no GPU, no VAE."""
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
    denoiser_capped = chunked_denoiser(denoiser, MAX_DENOISER_ROWS)
    G = edm_generator(denoiser_capped, img_shape=(c, h, w), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
                      n_steps=N_STEPS_GEN)
    reward, lam_s, info = _build_reward_and_lam_s(denoiser_capped, G, (c, h, w), d, dtype, device)
    stamp = {"dry_run": True, "d": d, **info}
    return Setup(device, dtype, velocity_fn, reward, d, lam_s, None, stamp)


def flux_setup() -> Setup:
    """The real FLUX.1-dev path, mirroring the notebook's cells 2-3 exactly (same reward config, same
    seeds) so lambda_s reproduces that run's measured ~81.0."""
    from diffusers import FluxPipeline

    device = torch.device("cuda")
    dtype = torch.bfloat16
    img = 512
    c, h, w = 16, img // 8, img // 8
    d = c * h * w

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=PROMPT, prompt_2=None, device=device, max_sequence_length=512)
    assert prompt_embeds is not None and pooled_prompt_embeds is not None, "encode_prompt returned None"
    pipe.text_encoder = pipe.text_encoder_2 = pipe.tokenizer = pipe.tokenizer_2 = None
    torch.cuda.empty_cache()
    transformer, vae = pipe.transformer.eval(), pipe.vae.eval()
    vae_sf, vae_shift = vae.config.scaling_factor, vae.config.shift_factor
    img_ids = FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)
    txt_ids = text_ids if text_ids.ndim == 2 else text_ids[0]

    velocity_fn = flux_velocity_fn(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=GUIDANCE, img_shape=(c, h, w), img_px=img, dtype=dtype, differentiable=True,
    )
    denoiser = flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=GUIDANCE, img_shape=(c, h, w), img_px=img, dtype=dtype, differentiable=True,
    )
    denoiser_capped = chunked_denoiser(denoiser, MAX_DENOISER_ROWS)
    G = edm_generator(denoiser_capped, img_shape=(c, h, w), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
                      n_steps=N_STEPS_GEN)
    reward, lam_s, info = _build_reward_and_lam_s(denoiser_capped, G, (c, h, w), d, dtype, device)

    def decode(flat: Tensor, chunk: int = 2) -> Tensor:
        outs = []
        for i in range(0, flat.shape[0], chunk):
            lat = flat[i : i + chunk].reshape(-1, c, h, w).to(device, dtype) / vae_sf + vae_shift
            with torch.no_grad():
                im = vae.decode(lat).sample
            post: Tensor = pipe.image_processor.postprocess(im.float(), output_type="pt")  # type: ignore[assignment]
            outs.append(post.cpu())
        return torch.cat(outs)

    stamp = {"dry_run": False, "d": d, "model": MODEL_ID, "gpu": torch.cuda.get_device_name(0), **info}
    return Setup(device, dtype, velocity_fn, reward, d, lam_s, decode, stamp)


# =====================================================================================================
# Results file (atomic, resumable per lambda) -- same convention as guided_sweep.py / iid_vs_brownian.py
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


def run_one(S: Setup, idx: int, lam: float, res: dict) -> None:
    key = f"idx{idx:02d}_lam{lam:.4f}"
    if key in res:
        note(f"{key} already done: f={res[key]['f_mean']:.5f}, t={res[key]['t_guidance_s']:.1f}s")
        return
    t0 = time.time()
    result = flow_guided_sample(
        S.reward, lam, N_PARTICLES, velocity_fn=S.velocity_fn, n_steps=N_STEPS_ODE, shift=SHIFT,
        t_start=1.0, t_end=0.0, exact_jacobian=True, seed=SWEEP_SEED,
    )
    t_guidance = time.time() - t0
    with torch.no_grad():
        f = S.reward(result.X)
    png_path = None
    if S.decode is not None:
        os.makedirs(DECODED_DIR, exist_ok=True)
        img = S.decode(result.X)[0]
        from PIL import Image
        arr = (img.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        png_path = os.path.join(DECODED_DIR, f"{key}.png")
        Image.fromarray(arr).save(png_path)
    res[key] = {
        "idx": idx, "lam": lam, "m": lam / S.lam_s if S.lam_s else 0.0,
        "f_mean": float(f.mean()), "f_std": float(f.std()),
        "applied_norm_mean": (sum(result.applied_norm_history) / len(result.applied_norm_history)
                               if result.applied_norm_history else 0.0),
        "v_norm_mean": (sum(result.v_norm_history) / len(result.v_norm_history)
                         if result.v_norm_history else 0.0),
        "oom_fallback_any": any(result.oom_fallback_history),
        "t_guidance_s": t_guidance,
        "png": png_path,
    }
    save_results(res)
    note(f"{key}: m={res[key]['m']:.3f} f_mean={res[key]['f_mean']:.4f} f_sd={res[key]['f_std']:.4f} "
         f"t_guidance={t_guidance:.1f}s ({t_guidance / N_STEPS_ODE:.2f}s/step)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lam-max", type=float, default=None,
                     help="upper end of the lambda range (default: lambda_s, i.e. m=1). The lower end "
                          "is always 0.0. Use this to zoom into a sub-range once a coarser sweep has "
                          "bracketed where the collapse happens -- e.g. --lam-max 5.5 after a [0, lambda_s] "
                          "sweep showed collapse already complete by lam=5.79.")
    ap.add_argument("--tag", type=str, default=None,
                     help="suffix for the results JSON / decoded dir, so a zoomed-in sweep doesn't "
                          "overwrite a previous range's results. Defaults to the lam-max value.")
    args = ap.parse_args()
    global RESULTS, DECODED_DIR
    if args.dry_run:
        RESULTS = os.path.join(HERE, "fine_lambda_sweep_results.dryrun.json")
        DECODED_DIR = os.path.join(HERE, "fine_decoded_dryrun")
        if os.path.exists(RESULTS):
            os.remove(RESULTS)
    elif args.lam_max is not None:
        tag = args.tag if args.tag is not None else f"max{args.lam_max:g}"
        RESULTS = os.path.join(HERE, f"fine_lambda_sweep_results_{tag}.json")
        DECODED_DIR = os.path.join(HERE, f"fine_decoded_{tag}")

    S = dry_setup() if args.dry_run else flux_setup()
    note(f"setup ready: d={S.d}, device={S.device}, lambda_s={S.lam_s:.2f}")
    lam_max = args.lam_max if args.lam_max is not None else S.lam_s
    lambdas = torch.linspace(0.0, lam_max, N_LAMBDAS).tolist()

    res = load_results({**S.stamp, "n_particles": N_PARTICLES, "n_steps_ode": N_STEPS_ODE,
                        "n_lambdas": N_LAMBDAS, "lambda_s": S.lam_s, "lam_max": lam_max,
                        "sweep_seed": SWEEP_SEED})
    for idx, lam in enumerate(lambdas):
        run_one(S, idx, lam, res)

    note("done")
    print(f"\n{'idx':>4s} {'m':>7s} {'lam':>8s} {'f_mean':>12s} {'f_sd':>10s} {'t_guid(s)':>10s} {'oom':>4s}")
    for idx in range(len(lambdas)):
        key = f"idx{idx:02d}_lam{lambdas[idx]:.4f}"
        if key not in res:
            print(f"{idx:>4d}  (missing)")
            continue
        r = res[key]
        print(f"{idx:>4d} {r['m']:>7.3f} {r['lam']:>8.2f} {r['f_mean']:>12.4f} {r['f_std']:>10.4f} "
              f"{r['t_guidance_s']:>10.1f} {'Y' if r['oom_fallback_any'] else '-':>4s}")


if __name__ == "__main__":
    main()
