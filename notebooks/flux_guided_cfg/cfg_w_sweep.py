"""Manual CFG x lambda: does a stiffer base velocity field resist the reward gradient's degradation?

THE QUESTION. FLUX.1-dev is guidance-distilled and this repo's whole FLUX path runs at an embedded
``GUIDANCE = 1.5``, baked into every reference bank, reward and base velocity. On top of that baseline we
add traditional two-forward classifier-free guidance,

    v_CFG(x_t, t) = v(x_t, t | null) + w * (v(x_t, t | c) - v(x_t, t | null)),

and ask whether the sharper field it produces is a *stiffer* structure that holds its shape further up
the lambda ladder before Phase 3's "deep-frying" sets in. ``w = 1`` collapses the null terms and IS the
Phase 3 baseline, exactly (``generators/cfg.py`` returns the conditional function itself there).

WHAT IS AND IS NOT VARIED -- read this before interpreting any number here.

CFG is applied to the Euler TRANSPORT only. ``x_hat_0 = x_t - t*v_cond`` stays the model's true
conditional denoised estimate, so:
  - the reward is never evaluated on a ``w``-extrapolated point that no reference bank covers, which is
    what keeps ``f`` comparable across ``w``;
  - ``grad_scaling="velocity"`` scales ``lam * g`` against ``||v_cond||``, NOT ``||v_CFG||``. This is
    load-bearing: CFG extrapolation inflates the velocity norm, so scaling to the transport field would
    silently raise the applied gradient WITH ``w`` and confound "a stiffer field resists the gradient"
    with "the gradient got bigger". Here ``w`` moves the base field and nothing else.
  - only ONE autograd graph is ever built, because the transport field is evaluated under ``no_grad``.
    ``exact_jacobian=True`` therefore costs what it costs in Phase 3; CFG adds one batch-1 forward per
    step against the reward's ~50 denoiser rows, i.e. a few percent, not 2x.
The mechanism and its rationale are documented once, on ``guided_euler_step``'s ``transport_velocity_fn``
parameter (``samplers/flow_guided_common.py``) -- this script only chooses to use it.

THE REWARD IS IDENTICAL ACROSS w. Setup imports Phase 3's ``_build_reward_and_lam_s`` rather than copying
it, so the reference latents, gammas, eps draws and ``lambda_s`` are bit-identical to Phase 3's and
Phase 5's runs for the same prompt. That is the whole basis of the comparison: lambda means the same
thing at every ``w``, and the per-prompt ``lambda_s`` registry (shared with Phase 5) asserts the reward
config did not drift.

NO w=1 ARM IS RUN HERE. It already exists: the Phase 3 / Phase 5 ``flow_guided`` runs at the same prompt,
seed, ``n_steps=10`` and lambda lattice ARE ``w = 1``, and ``render_cfg_grid.py`` pulls them in as the
grid's first column. Keep ``--lam-step``/``--lam-k`` on a lattice those runs already cover or there will
be nothing to pair against (see ``--lam-k``'s help).

CAVEAT, flagged because it may well be the headline result. FLUX.1-dev's empty-prompt branch is not a
trained unconditional model -- distillation removed the need for one -- so large ``w`` on this checkpoint
may simply over-saturate rather than "stiffen". ``--cfg-t-window`` restricts the extrapolation to an
interval of t (the usual workaround when true CFG is applied to a distilled flow model) without needing
a code change. Preflight item (4) measures how far apart the two fields actually are, per t, and raises
if they are identical -- the failure mode where ``encode_prompt("")`` or the wiring silently gives one
field and every ``w`` produces the same image.

CONVENTION NOTE: t is diffusers-native (t=1 noise, t=0 data) throughout, same as flow_guided.py. Read
E_q[f] at the terminal t=0 only (CLAUDE.md: f read at intermediate t is inflated).

    python cfg_w_sweep.py --w 2.0                    # on a GPU node (see cfg_w_sweep.slurm)
    python cfg_w_sweep.py --w 1.5,2.0,3.0            # several w sharing ONE model load + ref bank
    python cfg_w_sweep.py --w 2.0 --dry-run          # CPU, tiny FluxTransformer2DModel, no decode
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
PHASE3 = os.path.join(HERE, "..", "flux_guided_phase3")
PHASE5 = os.path.join(HERE, "..", "flux_guided_phase5")
sys.path.insert(0, PHASE3)
sys.path.insert(0, PHASE5)

from fine_lambda_sweep import (   # noqa: E402  # pyright: ignore[reportMissingImports]  (sys.path above)
    GUIDANCE, MAX_DENOISER_ROWS, MODEL_ID, N_PARTICLES, N_STEPS_GEN, N_STEPS_ODE, PROMPT as PROMPT_DEFAULT,
    SHIFT, SIGMA_MAX, SIGMA_MIN, SWEEP_SEED, _build_reward_and_lam_s, note,
)
# Pure helpers only -- never pc_sweep's load_results/_stamp_identity, which read ITS module globals.
# The lambda_s registry is deliberately the SAME FILE as Phase 5's: it maps prompt -> std_p(f) for this
# exact reward config, which is the quantity both phases depend on and the thing a drift guard must
# compare against. Two registries would let the two phases disagree silently.
from pc_sweep import (            # noqa: E402  # pyright: ignore[reportMissingImports]
    EXPECTED_GPU, LAM_S_TOL, _lam_s_registry, _record_lam_s, hf_power_fraction, prompt_slug,
)

from creativity_measure import NormalizedExpectedDistanceReward                             # noqa: E402
from creativity_measure.distances.edm_adapter import chunked_denoiser                        # noqa: E402
from creativity_measure.generators.base import edm_generator                                 # noqa: E402
from creativity_measure.generators.cfg import cfg_velocity_fn                                # noqa: E402
from creativity_measure.generators.flux import flux_edm_denoiser, flux_velocity_fn           # noqa: E402
from creativity_measure.samplers.flow_guided import flow_guided_sample                       # noqa: E402

# Lambda lattices in use elsewhere in the repo, so a run here can PAIR with stored w=1 images:
#   5.5/14  -- Phase 3's zoom grid ("A dog", seeds 1234/2024/3141/4242/5555), fine_decoded_max5.5/ and
#              fine_decoded_seed*/
#   0.2     -- Phase 5's prompt study (building/sofa/car/teapot/jacket, seeds 1234/3141), wave A k=0..5
#   0.1     -- the same study's wave B, k=1,3,5,7,9 (--tag odd there)
LAM_STEP_PHASE3 = 5.5 / 14
LAM_STEP_PROMPT_STUDY = 0.2
LAM_STEP_DEFAULT = LAM_STEP_PROMPT_STUDY
LAM_K_DEFAULT = (0, 1, 2, 3, 4, 5)

EXACT_JACOBIAN = True          # Phase 3/5 production setting, so lambda is comparable to their runs
NULL_PROMPT = ""               # the unconditional branch. FLUX.1-dev has no trained null; see docstring.
CFG_FIELD_MIN_REL = 1e-3       # below this the two fields are the same field -> preflight raises

LAM_GRID: list[float] = []     # set in main()
LAM_STEP: float = LAM_STEP_DEFAULT
N_STEPS: int = 0
RESULTS = ""
DECODED_DIR = ""


@dataclass
class CFGSetup:
    """Phase 3's ``Setup``, plus the unconditional velocity field manual CFG needs.

    Not a subclass of ``fine_lambda_sweep.Setup``: nothing here is passed to a function expecting that
    type, and spelling the fields out keeps it obvious that ``velocity_fn`` (conditional) is the one the
    reward and the gradient see, while ``velocity_fn_uncond`` only ever feeds the transport.
    """

    device: torch.device
    dtype: torch.dtype
    velocity_fn: Callable[[Tensor, float], Tensor]          # conditional; x_hat_0, reward, gradient
    velocity_fn_uncond: Callable[[Tensor, float], Tensor]   # null prompt; transport only, no_grad only
    reward: NormalizedExpectedDistanceReward
    d: int
    lam_s: float
    decode: Callable[[Tensor], Tensor] | None
    stamp: dict


def dry_setup(prompt: str = PROMPT_DEFAULT) -> CFGSetup:
    """Tiny real FluxTransformer2DModel, CPU, no VAE -- exercises the sweep/resume/preflight mechanics.

    The two "prompts" are two independent random embedding draws (there is no text encoder on this path),
    which is enough to make v_cond != v_uncond and therefore enough to exercise the CFG arithmetic and
    preflight item (4) end to end.
    """
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
    txt_ids = torch.zeros((3, 3))
    img_ids = FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)

    def embeds() -> tuple[Tensor, Tensor]:
        return (torch.randn((1, 3, transformer.config.joint_attention_dim), generator=gen),
                torch.randn((1, transformer.config.pooled_projection_dim), generator=gen))

    pe_c, pp_c = embeds()
    pe_u, pp_u = embeds()
    # dict[str, Any]: without it pyright infers a union of the literal value types and rejects every
    # ** spread below -- same annotation flux_setup's `common` carries, for the same reason.
    common: dict[str, Any] = dict(img_shape=(c, h, w), img_px=img_px, dtype=dtype, guidance=1.0)
    velocity_fn = flux_velocity_fn(transformer, pe_c, pp_c, img_ids, txt_ids, differentiable=True, **common)
    velocity_uncond = flux_velocity_fn(transformer, pe_u, pp_u, img_ids, txt_ids, differentiable=False, **common)
    denoiser = flux_edm_denoiser(transformer, pe_c, pp_c, img_ids, txt_ids, differentiable=True, **common)
    denoiser_capped = chunked_denoiser(denoiser, MAX_DENOISER_ROWS)
    G = edm_generator(denoiser_capped, img_shape=(c, h, w), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
                      n_steps=N_STEPS_GEN)
    reward, lam_s, info = _build_reward_and_lam_s(denoiser_capped, G, (c, h, w), d, dtype, device)
    stamp = {"dry_run": True, "d": d, "prompt": prompt, "null_prompt": NULL_PROMPT, **info}
    return CFGSetup(device, dtype, velocity_fn, velocity_uncond, reward, d, lam_s, None, stamp)


def flux_setup(prompt: str = PROMPT_DEFAULT) -> CFGSetup:
    """Real FLUX.1-dev, mirroring Phase 3's ``flux_setup`` exactly, plus a second (null-prompt) velocity.

    Both prompts are encoded BEFORE the text encoders are freed (they are ~9.5 GB and must not be loaded
    twice), and both velocity closures are built over the SAME transformer at the SAME embedded
    ``guidance=GUIDANCE``. If they differed in the embedded guidance, the CFG difference term would be
    mixing two different fields and ``w`` would not mean what the formula says.

    The unconditional closure is built with ``differentiable=False`` deliberately, not as an oversight: it
    only ever feeds the Euler transport, which is never differentiated, and ``differentiable=False`` wraps
    its forward in ``torch.no_grad()`` so no graph can be built through it even by accident. The
    conditional one stays ``differentiable=True`` (and hence frozen, with ``.module`` attached) because the
    reward gradient flows through it under ``exact_jacobian=True``.
    """
    from diffusers import FluxPipeline

    device = torch.device("cuda")
    dtype = torch.bfloat16
    img = 512
    c, h, w = 16, img // 8, img // 8
    d = c * h * w

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, device=device, max_sequence_length=512)
        null_embeds, null_pooled, null_text_ids = pipe.encode_prompt(
            prompt=NULL_PROMPT, prompt_2=None, device=device, max_sequence_length=512)
    assert prompt_embeds is not None and pooled_prompt_embeds is not None, "encode_prompt returned None"
    assert null_embeds is not None and null_pooled is not None, "encode_prompt(null) returned None"
    pipe.text_encoder = pipe.text_encoder_2 = pipe.tokenizer = pipe.tokenizer_2 = None
    torch.cuda.empty_cache()
    transformer, vae = pipe.transformer.eval(), pipe.vae.eval()
    vae_sf, vae_shift = vae.config.scaling_factor, vae.config.shift_factor
    img_ids = FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)
    txt_ids = text_ids if text_ids.ndim == 2 else text_ids[0]
    null_ids = null_text_ids if null_text_ids.ndim == 2 else null_text_ids[0]

    common: dict[str, Any] = dict(guidance=GUIDANCE, img_shape=(c, h, w), img_px=img, dtype=dtype)
    velocity_fn = flux_velocity_fn(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids, differentiable=True, **common)
    velocity_uncond = flux_velocity_fn(
        transformer, null_embeds, null_pooled, img_ids, null_ids, differentiable=False, **common)
    denoiser = flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids, differentiable=True, **common)
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

    stamp = {"dry_run": False, "d": d, "model": MODEL_ID, "prompt": prompt, "null_prompt": NULL_PROMPT,
             "guidance_embedded": GUIDANCE, "gpu": torch.cuda.get_device_name(0), **info}
    return CFGSetup(device, dtype, velocity_fn, velocity_uncond, reward, d, lam_s, decode, stamp)


# =====================================================================================================
# Results file (atomic, resumable per lambda) -- same convention as pc_sweep.py
# =====================================================================================================

# Excluded from the identity check for the same reasons pc_sweep.py excludes them: `lam_k`/`lam_grid` say
# which lattice points THIS invocation asked for, not what any point means, so a job filling in points an
# earlier job skipped must MERGE rather than .stale-stomp completed data; `preflight` is a fresh
# on-hardware measurement every run and partly nondeterministic by construction (CLAUDE.md: FLUX's
# backward is not bit-reproducible on GPU), so requiring it to match would mean resume never fires.
_VOLATILE_STAMP_KEYS = ("lam_k", "lam_grid", "preflight")


def _stamp_identity(stamp: dict) -> dict:
    return {k: v for k, v in stamp.items() if k not in _VOLATILE_STAMP_KEYS}


def load_results(stamp: dict) -> dict:
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            old = json.load(fh)
        old_stamp = old.get("stamp", {})
        if _stamp_identity(old_stamp) == _stamp_identity(stamp):
            merged_lam_k = sorted(set(old_stamp.get("lam_k", [])) | set(stamp.get("lam_k", [])))
            old["stamp"] = {**stamp, "lam_k": merged_lam_k,
                            "lam_grid": [k * LAM_STEP for k in merged_lam_k]}
            print(f"[resume] {RESULTS}: {sorted(k for k in old if k != 'stamp')}")
            return old
        os.replace(RESULTS, RESULTS + ".stale")
        print("[resume] stamp mismatch on identity fields -> old results moved to .stale, starting fresh")
    return {"stamp": stamp}


def save_results(res: dict) -> None:
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=2)
    os.replace(tmp, RESULTS)


# =====================================================================================================
# Preflight -- tier-1 checks whose failure invalidates the WHOLE job. Raises; never warns.
# =====================================================================================================

def preflight(S: CFGSetup, sweep_seed: int, *, dry_run: bool, prompt: str,
              t_window: tuple[float, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # (1) GPU model. Checked first because it is the only item that needs nothing from the reward, i.e.
    # the only one that can fail BEFORE the ~10-20 min reference-bank build. A GPU change alone shifts f
    # by 16% of std_p(f) (job 697271) -- the same order as the effect being measured, and the stored w=1
    # runs this sweep pairs against are all 'NVIDIA L40S'.
    if not dry_run:
        gpu = torch.cuda.get_device_name(0)
        out["gpu_name"] = gpu
        note(f"preflight gpu: {gpu}")
        if EXPECTED_GPU not in gpu:
            raise AssertionError(
                f"expected a {EXPECTED_GPU} (every stored w=1 run's stamp records 'NVIDIA L40S') but "
                f"landed on {gpu!r}. Nothing from this GPU is comparable to the w=1 column -- fix "
                "--constraint and resubmit rather than interpreting this run."
            )

    # (2) f(x_refs) == (R-1)/R exactly for uniform weights (CLAUDE.md). Free: the bank is already built.
    # Tolerance 1e-3, NOT 0.10 -- 0.10 cannot separate 63/64 from the failure modes landing on exactly 1.0.
    with torch.no_grad():
        f_refs = float(S.reward(S.reward.x_refs).mean())
    r = S.reward.x_refs.shape[0]
    expected = (r - 1) / r
    out["f_refs"] = f_refs
    note(f"preflight f(refs): {f_refs:.6f} vs (R-1)/R = {expected:.6f}  (R={r})")
    if abs(f_refs - expected) > 1e-3:
        raise AssertionError(
            f"f(x_refs) = {f_refs:.6f} but uniform weights demand exactly (R-1)/R = {expected:.6f}. "
            "The reward's normalization is mis-wired; every f in this job would be on the wrong scale."
        )

    # (3) lambda_s reproduces THIS PROMPT's recorded value, from the registry SHARED with Phase 5. Free.
    # This is the check that the reward config did not drift from the w=1 runs being paired against --
    # without it, "CFG changed the images" and "the reward changed under me" are indistinguishable.
    out["lam_s"] = S.lam_s
    if not dry_run:
        known = _lam_s_registry().get(prompt)
        if known is None:
            note(f"preflight lambda_s: {S.lam_s:.3f} -- first run for prompt {prompt!r}, recording it. "
                 "NOTE: a prompt with no recorded value has no stored w=1 column either, so there is "
                 "nothing for render_cfg_grid.py to pair against.")
            _record_lam_s(prompt, S.lam_s)
        else:
            rel = abs(S.lam_s - known) / known
            note(f"preflight lambda_s: {S.lam_s:.3f} vs {prompt!r}'s recorded {known:.3f} (rel {rel:.2%})")
            if rel > LAM_S_TOL:
                raise AssertionError(
                    f"lambda_s = {S.lam_s:.3f} differs from the recorded value for prompt {prompt!r} "
                    f"({known:.3f}) by {rel:.1%} (> {LAM_S_TOL:.0%}). The reference latents or the reward "
                    "config diverged from the stored w=1 runs, so no comparison here is meaningful."
                )

    # (4) THE CFG-SPECIFIC CHECK: the two fields are actually two fields.
    #
    # The silent failure this exists for: if encode_prompt("") returned the conditional embeddings, or the
    # two closures got wired to the same conditioning, then v_CFG == v_cond identically and EVERY w would
    # produce the same image as w=1. The sweep would complete, cost its full GPU-hours, report plausible
    # f values, and measure nothing. Nothing else in this job would notice.
    #
    # The relative separation is also RECORDED per t, not just asserted, because it is the natural scale
    # for reading the sweep: ||v_CFG - v_cond|| = (w-1)*||v_cond - v_uncond||, so this number times (w-1)
    # is exactly how far CFG moves the transported field, and it is what a t-window should be chosen from.
    t0 = time.time()
    gen = torch.Generator(device=S.device).manual_seed(4242)
    seps: dict[str, float] = {}
    for t in (0.95, 0.75, 0.5, 0.25, 0.05):
        x_t = torch.randn(1, S.d, generator=gen, device=S.device, dtype=S.dtype)
        with torch.no_grad():
            v_c = S.velocity_fn(x_t, t)
            v_u = S.velocity_fn_uncond(x_t, t)
        rel = float((v_c - v_u).norm() / v_c.norm().clamp_min(1e-30))
        seps[f"t{t}"] = rel
        in_win = t_window[1] <= t <= t_window[0]
        note(f"preflight cfg field separation t={t:.2f}: ||v_c - v_u||/||v_c|| = {rel:.4e}"
             + ("" if in_win else "   (outside --cfg-t-window: not extrapolated)"))
    out["cfg_field_sep_rel"] = seps
    out["cfg_field_sep_s"] = time.time() - t0
    in_window = {k: v for k, v in seps.items() if t_window[1] <= float(k[1:]) <= t_window[0]}
    worst = max(in_window.values()) if in_window else 0.0
    if worst < CFG_FIELD_MIN_REL:
        raise AssertionError(
            f"the conditional and unconditional velocity fields are identical to within {worst:.3e} "
            f"(< {CFG_FIELD_MIN_REL:.0e}) at every t inside the CFG window {t_window}. v_CFG would equal "
            "v_cond for every w and this sweep would measure NOTHING. Check that encode_prompt was called "
            "twice with different prompts and that the two velocity closures got the different embeddings."
        )

    # (5) The w=1 reduction, asserted bitwise where the hardware can actually deliver it.
    #
    # Same split of claims as pc_sweep.py's item (5), and valid for the same reason. At lam=0 NO BACKWARD
    # is taken, so the path is deterministic and a mismatch is a real defect -- assert it. At lam != 0
    # FLUX's backward is not bit-reproducible on GPU (flash-attention atomics plus _freeze's
    # gradient-checkpoint recompute; CLAUDE.md), so the cross-path deviation is RECORDED against the
    # backend's own self-reproducibility floor instead of gated on. Here the claim is narrower and
    # stronger than Phase 5's: cfg_velocity_fn(.., w=1) returns the CONDITIONAL FUNCTION ITSELF and
    # guided_euler_step detects that identity, so this is the unsplit path exactly -- bitwise at any lam
    # by construction, which is what tests/test_cfg_velocity.py asserts on CPU.
    kw0: dict[str, Any] = dict(
        velocity_fn=S.velocity_fn, n_steps=2, shift=SHIFT, t_start=1.0, t_end=0.0,
        exact_jacobian=EXACT_JACOBIAN, seed=sweep_seed,
    )
    w1 = cfg_velocity_fn(S.velocity_fn, S.velocity_fn_uncond, 1.0)
    t0 = time.time()
    ref0 = flow_guided_sample(S.reward, 0.0, N_PARTICLES, **kw0)
    cfg0 = flow_guided_sample(S.reward, 0.0, N_PARTICLES, transport_velocity_fn=w1, **kw0)
    ok0 = torch.equal(cfg0.X, ref0.X)
    out["w1_reduction_bitwise_lam0"] = ok0
    note(f"preflight w=1 reduction (lam=0, no backward -> must be bitwise): {ok0} ({time.time()-t0:.1f}s)")
    if not ok0:
        raise AssertionError(
            "the w=1 CFG wrapper is NOT bitwise equal to the unsplit flow_guided_sample at lam=0 (max abs "
            f"diff {float((cfg0.X - ref0.X).abs().max()):.3e}). No backward is taken at lam=0, so this is "
            "a REAL defect or a config drift, not backend nondeterminism. Do not interpret this job."
        )

    t0 = time.time()
    a = flow_guided_sample(S.reward, 1.0, N_PARTICLES, **kw0)
    b = flow_guided_sample(S.reward, 1.0, N_PARTICLES, **kw0)
    c = flow_guided_sample(S.reward, 1.0, N_PARTICLES, transport_velocity_fn=w1, **kw0)
    floor = float((a.X.float() - b.X.float()).abs().max())
    cross = float((a.X.float() - c.X.float()).abs().max())
    out["nondet_floor_lam1"] = floor
    out["w1_reduction_cross_lam1"] = cross
    note(f"preflight w=1 reduction (lam=1, exact_jacobian={EXACT_JACOBIAN}): self-deviation floor "
         f"{floor:.4e}, cross-path {cross:.4e}, "
         + (f"ratio {cross / floor:.2f}" if floor > 0 else
            ("BOTH EXACT" if cross == 0 else "floor=0 but cross>0 -- see assert below"))
         + f"  ({time.time() - t0:.1f}s)")
    if floor == 0.0 and cross > 0.0:
        raise AssertionError(
            f"the backend IS bit-reproducible here (self-deviation 0) yet the w=1 path differs by "
            f"{cross:.3e}. That cannot be nondeterminism -- it is a real difference between the paths."
        )
    return out


def check_point(result: Any, lam: float, w: float, n_steps: int) -> None:
    """Tier-2 asserts: free, per-point, and they RAISE rather than warn.

    The sweep is resumable per lambda, so a hard failure costs one point and a resubmit; a silently wrong
    point pollutes a cross-w conclusion someone will act on. Every quantity here is already recorded.
    """
    if not torch.isfinite(result.X).all():
        raise AssertionError(f"w={w} lam={lam}: terminal latents contain non-finite values")
    if len(result.t_history) != n_steps:
        raise AssertionError(
            f"w={w} lam={lam}: {len(result.t_history)} steps recorded, expected {n_steps}")

    guided_steps = [i for i, g in enumerate(result.guided_history) if g]
    if lam == 0.0 and guided_steps:
        raise AssertionError(f"lam=0 must take no guided step, got {len(guided_steps)}")
    if lam != 0.0 and len(guided_steps) != n_steps:
        raise AssertionError(
            f"w={w} lam={lam}: {len(guided_steps)} of {n_steps} steps guided; the window is (1.0, 0.0) "
            "so every step must be")

    # The split transport must have been WIRED, at every step: transport_v_norm is nan iff no separate
    # field was used. At w != 1 a nan here means the sweep silently ran the w=1 path -- the exact failure
    # that would make a whole w column a duplicate of the baseline.
    n_split = sum(1 for v in result.transport_v_norm_history if v == v)
    if w != 1.0 and n_split != n_steps:
        raise AssertionError(
            f"w={w}: only {n_split} of {n_steps} steps recorded a transport-field norm. The CFG transport "
            "was not applied on every step -- this column would be the w=1 baseline in disguise.")


def _lam_s_m(S: CFGSetup, lam: float) -> float:
    return lam / S.lam_s if S.lam_s else 0.0


def run_one(S: CFGSetup, w: float, idx: int, lam: float, res: dict, sweep_seed: int,
            t_window: tuple[float, float]) -> None:
    # Keyed by the LATTICE index (idx == k from --lam-k), not this job's positional index, so the same
    # lambda always lands on the same key no matter which subset of the lattice a job requested and
    # merged entries from different --lam-k invocations never collide.
    key = f"idx{idx:02d}_lam{lam:.4f}"
    if key in res:
        note(f"{key} already done: f={res[key]['f_mean']:.5f}, t={res[key]['t_sample_s']:.1f}s")
        return

    # Rebuilt per point rather than hoisted: it is a closure over two existing callables, so it costs
    # nothing, and building it here keeps w visible at the one call site that uses it.
    transport = cfg_velocity_fn(S.velocity_fn, S.velocity_fn_uncond, w, t_window=t_window)

    t0 = time.time()
    result = flow_guided_sample(
        S.reward, lam, N_PARTICLES, velocity_fn=S.velocity_fn, transport_velocity_fn=transport,
        n_steps=N_STEPS, shift=SHIFT, t_start=1.0, t_end=0.0, exact_jacobian=EXACT_JACOBIAN,
        seed=sweep_seed,
    )
    t_sample = time.time() - t0
    check_point(result, lam, w, N_STEPS)
    with torch.no_grad():
        f = S.reward(result.X)

    mean = lambda vals: (sum(vals) / len(vals)) if vals else float("nan")      # noqa: E731
    finite = lambda vals: [v for v in vals if v == v]                          # noqa: E731
    # v_norm_mean over ALL steps, matching how Phase 3/5's run_one computes it, so the column is directly
    # comparable to their stored JSONs.
    v_norm_mean = mean(finite(result.v_norm_history))
    transport_norm_mean = mean(finite(result.transport_v_norm_history))
    # The ratio, however, is only defined on GUIDED steps. On an unguided step there is no x_hat_0, so the
    # conditional field is never evaluated and `v_norm` records the TRANSPORT field -- making the ratio a
    # tautological 1.0. That is every step at lam=0, where the ratio must read nan, not 1.
    guided = [i for i, g in enumerate(result.guided_history) if g]
    v_guided = mean([result.v_norm_history[i] for i in guided])
    tr_guided = mean([result.transport_v_norm_history[i] for i in guided])
    cfg_norm_ratio = (tr_guided / v_guided) if (guided and v_guided) else float("nan")

    png_path, hf_frac = None, float("nan")
    if S.decode is not None:
        os.makedirs(DECODED_DIR, exist_ok=True)
        img = S.decode(result.X)[0]
        hf_frac = hf_power_fraction(img)
        from PIL import Image
        arr = (img.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        png_path = os.path.join(DECODED_DIR, f"{key}.png")
        Image.fromarray(arr).save(png_path)

    res[key] = {
        "w": w, "idx": idx, "lam": lam, "m": _lam_s_m(S, lam),
        "f_mean": float(f.mean()), "f_std": float(f.std()),
        # predictor diagnostics -- same field names Phase 3/5 log, so the w columns line up with theirs
        "applied_norm_mean": mean(finite(result.applied_norm_history)),
        "v_norm_mean": v_norm_mean,
        # CFG diagnostics. cfg_norm_ratio = ||v_CFG||/||v_cond|| over guided steps: how much the
        # extrapolation inflated the transported field. It is NOT applied to the gradient (which is scaled
        # to ||v_cond||), so this is the one number saying how strong w actually was at this prompt, in the
        # field's own units. nan at lam=0 by construction -- see above.
        "transport_v_norm_mean": transport_norm_mean,
        "cfg_norm_ratio": cfg_norm_ratio,
        # off-manifold proxies. NEITHER predicts recognizability (CLAUDE.md) -- x_norm is ANTI-correlated
        # and hf_frac has no absolute threshold across seeds. Recorded for within-comparison ranking only.
        "x_norm_final": float(result.X.float().norm(dim=1).mean() / (S.d ** 0.5)),
        "hf_frac": hf_frac,
        "oom_fallback_any": any(result.oom_fallback_history),
        "static_fallback_any": any(result.static_fallback_history),
        "t_sample_s": t_sample,
        "png": png_path,
    }
    save_results(res)
    r = res[key]
    note(f"{key}: w={w:g} m={r['m']:.3f} f={r['f_mean']:.4f} cfg_ratio={r['cfg_norm_ratio']:.4f} "
         f"|x|/sqrt(d)={r['x_norm_final']:.4f} hf={r['hf_frac']:.4f} "
         f"t={t_sample:.1f}s ({t_sample / N_STEPS:.2f}s/step)")


def main() -> None:
    global RESULTS, DECODED_DIR, LAM_GRID, LAM_STEP, N_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=str, required=True,
                    help="comma-separated CFG weights. Several w in ONE job share the model load, the "
                         "reference bank and the preflight (35-90 min on this NFS-bound path) and are "
                         "guaranteed to land on the same GPU, which the cross-w comparison requires "
                         "anyway. w=1 is accepted and is exactly the Phase 3 baseline, but it is normally "
                         "NOT run here -- the stored flow_guided runs already are it.")
    ap.add_argument("--prompt", type=str, default=PROMPT_DEFAULT,
                    help="text prompt (default: %(default)r, Phase 3's). A prompt is a SEPARATE REWARD: "
                         "the reference latents are G(z) under the prompt-conditioned velocity, so "
                         "S_scale, gamma_lo and lambda_s all move with it. It enters the results key, so "
                         "prompts never share a file. Use one that already has stored w=1 runs.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", type=str, default=None,
                    help="extra suffix on top of the auto-generated config key. Needed only to separate "
                         "two runs differing in NOTHING the key already captures -- e.g. a second lambda "
                         "lattice at the same prompt/seed/w, since lam_step is not in the filename.")
    ap.add_argument("--n-steps", type=int, default=N_STEPS_ODE,
                    help="ODE steps (default: %(default)s, Phase 3's). DO NOT RAISE IT. More steps "
                         "DESTROYS gradient guidance (CLAUDE.md: n_steps=19 gives abstract blocks where "
                         "n_steps=10 gives a recognizable dog) -- Phase 3's window depends on "
                         "discretization error as implicit regularization, and n_steps cannot be used to "
                         "build a compute-matched control.")
    ap.add_argument("--lam-step", type=float, default=LAM_STEP_DEFAULT,
                    help="lattice spacing; lambda = k * step. Default %(default)g is Phase 5's prompt "
                         f"study. Phase 3's grid is {LAM_STEP_PHASE3:g} (5.5/14) and its wave-B study "
                         f"used {0.1:g}. PICK THE ONE THE STORED w=1 RUNS FOR YOUR (prompt, seed) USED, "
                         "or the grid has no baseline column to pair against. The step enters the stamp, "
                         "so two lattices never merge into one results file -- but it is NOT in the "
                         "filename, so use --tag to keep two lattices from .stale-stomping each other.")
    ap.add_argument("--lam-k", type=str, default=",".join(str(k) for k in LAM_K_DEFAULT),
                    help="comma-separated lattice indices k; lambda = k * --lam-step. (default: "
                         "%(default)s, i.e. the prompt study's wave A). k=0 is the untilted sample, which "
                         "under CFG is still w-dependent and worth having as the 'what does w alone do' "
                         "reference.")
    ap.add_argument("--sweep-seed", type=int, default=SWEEP_SEED,
                    help="seed for z0 (default: %(default)s, Phase 3's, so z0 is IDENTICAL to its sweep). "
                         "Breakdown lambda varies ~3x across z0 (CLAUDE.md), so a single seed cannot "
                         "establish that CFG moved the window -- only that it moved THIS trajectory.")
    ap.add_argument("--cfg-t-window", type=str, default="1.0,0.0",
                    help="'t_hi,t_lo'; CFG is extrapolated only where t_lo <= t <= t_hi, and the plain "
                         "conditional velocity is used outside. Default is the whole trajectory. Narrow it "
                         "if large w over-saturates -- FLUX.1-dev is guidance-distilled and its "
                         "empty-prompt branch is not a trained null (see the module docstring).")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the tier-1 on-hardware checks (~5 min). Only for a resumed job whose "
                         "earlier attempt already logged them all PASS.")
    args = ap.parse_args()

    N_STEPS = args.n_steps
    LAM_STEP = args.lam_step
    lam_k = [int(k) for k in args.lam_k.split(",") if k.strip() != ""]
    LAM_GRID = [k * LAM_STEP for k in lam_k]

    try:
        ws = [float(v) for v in args.w.split(",") if v.strip() != ""]
    except ValueError:
        ap.error(f"--w must be a comma-separated list of floats; got {args.w!r}")
    if not ws:
        ap.error("--w must name at least one weight")

    try:
        t_hi, t_lo = (float(v) for v in args.cfg_t_window.split(","))
    except ValueError:
        ap.error(f"--cfg-t-window must be 't_hi,t_lo'; got {args.cfg_t_window!r}")
    if t_hi < t_lo:
        ap.error(f"--cfg-t-window must be t_hi,t_lo with t_hi >= t_lo (t runs 1 -> 0); got {args.cfg_t_window!r}")
    t_window = (t_hi, t_lo)

    S = dry_setup(args.prompt) if args.dry_run else flux_setup(args.prompt)
    note(f"setup ready: w={ws} prompt={args.prompt!r} d={S.d} device={S.device} "
         f"lambda_s={S.lam_s:.2f} n_steps={N_STEPS} lam_step={LAM_STEP:g} lam_k={lam_k} "
         f"lam_grid={[round(v, 4) for v in LAM_GRID]} cfg_t_window={t_window} seed={args.sweep_seed}")

    # ONE preflight for the whole job, not one per w: every item it checks (GPU, reward normalization,
    # lambda_s, the field separation, the w=1 reduction) is a property of the SETUP, shared by every w.
    pf: dict[str, Any] = {}
    if not args.skip_preflight:
        pf = preflight(S, args.sweep_seed, dry_run=args.dry_run, prompt=args.prompt, t_window=t_window)

    for w in ws:
        # The results key encodes EVERY axis a wave varies, w INCLUDED. Without w in the filename two
        # jobs differing only in --w write to the same file: the stamp check would .stale-rename rather
        # than corrupt, but each job would keep discarding the other's work -- and these jobs run
        # concurrently by design. Same lesson as CLAUDE.md's "bump RUN_TAG per leg".
        key = (f"{prompt_slug(args.prompt)}_w{w:g}_n{N_STEPS}_s{args.sweep_seed}"
               + (f"_tw{t_hi:g}-{t_lo:g}" if t_window != (1.0, 0.0) else "")
               + (f"_{args.tag}" if args.tag else "")
               + (".dryrun" if args.dry_run else ""))
        RESULTS = os.path.join(HERE, f"cfg_sweep_results_{key}.json")
        DECODED_DIR = os.path.join(HERE, f"cfg_decoded_{key}")
        if args.dry_run and os.path.exists(RESULTS):
            os.remove(RESULTS)
        note(f"[w={w:g}] results -> {os.path.basename(RESULTS)}")

        res = load_results({
            **S.stamp, "w": w, "cfg_t_window": list(t_window), "n_particles": N_PARTICLES,
            "n_steps_ode": N_STEPS, "lambda_s": S.lam_s, "lam_step": LAM_STEP, "lam_k": lam_k,
            "lam_grid": LAM_GRID, "sweep_seed": args.sweep_seed, "exact_jacobian": EXACT_JACOBIAN,
            "preflight": {k: v for k, v in pf.items() if k != "cfg_field_sep_s"},
        })
        for k, lam in zip(lam_k, LAM_GRID):
            run_one(S, w, k, lam, res, args.sweep_seed, t_window)

        print(f"\nprompt={args.prompt!r}  w={w:g}  n_steps={N_STEPS}  seed={args.sweep_seed}  "
              f"cfg_t_window={t_window}")
        print(f"{'idx':>4s} {'m':>7s} {'lam':>8s} {'f_mean':>12s} {'cfg_ratio':>10s} "
              f"{'|x|/sqrt d':>11s} {'hf_frac':>8s} {'applied':>9s} {'t(s)':>8s} {'oom':>4s}")
        for k, lam in zip(lam_k, LAM_GRID):
            pkey = f"idx{k:02d}_lam{lam:.4f}"
            if pkey not in res:
                print(f"{k:>4d}  (missing)")
                continue
            r = res[pkey]
            print(f"{k:>4d} {r['m']:>7.3f} {r['lam']:>8.2f} {r['f_mean']:>12.4f} "
                  f"{r['cfg_norm_ratio']:>10.4f} {r['x_norm_final']:>11.4f} {r['hf_frac']:>8.4f} "
                  f"{r['applied_norm_mean']:>9.4f} {r['t_sample_s']:>8.1f} "
                  f"{'Y' if r['oom_fallback_any'] else '-':>4s}")

    note("done")
    print("\nLOOK AT THE DECODED PNGs BEFORE BUILDING ANY ANALYSIS ON THESE SCALARS (CLAUDE.md). None of "
          "f, |x| or hf_frac predicts recognizability; f is 14.2 on a clear dog and 14.8 on pure noise.")


if __name__ == "__main__":
    main()
