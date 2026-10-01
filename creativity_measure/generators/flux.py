"""FLUX.1-schnell/dev denoiser wrapper (GPU production).  SCAFFOLD -- not yet functional.

FLUX is a flow-matching (rectified-flow) transformer operating on *packed* 16-channel latents (patchified
to a sequence, with ``img_ids`` / ``txt_ids`` / ``guidance`` inputs), and its transformer returns a velocity
field -- not an EDM ``E[X | x_sigma]`` on a ``(16, 64, 64)`` grid. Correctly mapping that velocity + latent
packing into the EDM denoiser convention this repo's ODE integrates requires real GPU validation, so it is
deliberately left as a documented ``NotImplementedError`` rather than shipping the (incorrect) placeholder
call from the design notes.

``diffusers`` is imported lazily (optional ``models`` extra); importing this module -- or
``creativity_measure`` -- never requires it.
"""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import GuidableVelocityFn, VelocityFn
from creativity_measure.distances.edm_adapter import Denoiser, chunked_denoiser

# NOTE: once implemented, this module will import and use ``edm_generator`` / ``eps_to_edm_denoiser``
# from ``.base`` to return the standard generator interface (see the scaffold TODO below).

_MISSING_DEPS_MSG = (
    "build_flux_generator needs 'diffusers' and 'transformers'. "
    "Install the optional model deps:  pip install -e '.[models]'"
)


def _flux_raw_velocity(
    transformer: nn.Module,
    x_t: Float[Tensor, "B c h w"],
    t: Float[Tensor, "B"],
    *,
    prompt_embeds: Tensor,
    pooled_prompt_embeds: Tensor,
    img_ids: Tensor,
    txt_ids: Tensor,
    guidance: float,
    img_shape: tuple[int, int, int],
    img_px: int,
    dtype: torch.dtype,
    grad_ctx: Callable[[], contextlib.AbstractContextManager],
) -> Float[Tensor, "B c h w"]:
    """The one FLUX transformer call: pack -> transformer -> unpack, returning the raw velocity.

    ``x_t`` and ``t`` are already in FLUX's own (diffusers-native) parameterization -- this helper does no
    reparameterization of its own, so both callers keep full control of the time convention they present.
    Extracted so ``flux_edm_denoiser`` and ``flux_velocity_fn`` share exactly one transformer call site;
    the arithmetic order is preserved verbatim from the original closure (which was itself ported from
    ``notebooks/refset_auto_r/auto_r_common.py``'s ``build()``), so ``flux_edm_denoiser`` stays bitwise
    identical -- asserted by ``tests/test_flux_denoiser.py``.
    """
    from diffusers import FluxPipeline  # type: ignore[attr-defined]  # older diffusers lack this

    c, h, w = img_shape
    b = x_t.shape[0]
    with grad_ctx():
        v = transformer(
            hidden_states=FluxPipeline._pack_latents(x_t.to(dtype), b, c, h, w),
            timestep=t.to(dtype),
            guidance=torch.full((b,), guidance, device=x_t.device, dtype=torch.float32),
            pooled_projections=pooled_prompt_embeds.expand(b, -1).to(dtype),
            encoder_hidden_states=prompt_embeds.expand(b, -1, -1).to(dtype),
            txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
    return FluxPipeline._unpack_latents(v, img_px, img_px, 8).to(torch.float32)


def _freeze(transformer: nn.Module) -> None:
    """``requires_grad_(False)`` + ``eval()`` + gradient checkpointing: only the caller's input latent
    may carry a gradient, and the backward pass recomputes each block's activations instead of keeping
    all ~57 of them resident at once.

    ``requires_grad_(False)`` alone is not sufficient on real FLUX.1-dev: job 957136 measured a single
    grad-enabled forward through the full 12B transformer at batch=2 using the ENTIRE remaining ~20 GB
    after the bf16 weights (44.51/44.53 GiB total) -- every block's activations must stay alive
    simultaneously for backward, unlike inference where each block's activations can be freed once the
    next one is computed. ``enable_gradient_checkpointing()`` trades this for recompute: diffusers gates
    it on ``torch.is_grad_enabled() and self.gradient_checkpointing`` (`transformer_flux.py`'s
    `forward`), NOT on ``self.training``, so it composes correctly with the ``eval()`` call below.
    """
    transformer.requires_grad_(False)
    transformer.eval()
    enable_checkpointing = getattr(transformer, "enable_gradient_checkpointing", None)
    if getattr(transformer, "_supports_gradient_checkpointing", False) and enable_checkpointing is not None:
        enable_checkpointing()


def flux_edm_denoiser(
    transformer: nn.Module,
    prompt_embeds: Float[Tensor, "1 seq d_txt"],
    pooled_prompt_embeds: Float[Tensor, "1 d_pool"],
    img_ids: Tensor,
    txt_ids: Tensor,
    *,
    guidance: float,
    img_shape: tuple[int, int, int],
    img_px: int,
    dtype: torch.dtype,
    differentiable: bool = False,
) -> Denoiser:
    """Adapt a loaded FLUX transformer + fixed prompt conditioning into the repo's EDM ``Denoiser``
    convention ``D(x_sigma, sigma) = E[X | x_sigma]``.

    Ported from ``notebooks/refset_auto_r/auto_r_common.py``'s ``build()`` (the ``denoiser`` closure,
    lines 393-413): same pack/unpack calls (``FluxPipeline._pack_latents``/``_unpack_latents``), same
    timestep/guidance handling, same ``x_t - t*v`` EDM-from-velocity conversion (flow matching interpolates,
    EDM adds: ``x_t = (1-t)x_0 + t*eps`` vs ``x_sigma = x_0 + sigma*eps``; factoring out ``(1-t)`` makes
    them identical under ``t = sigma/(1+sigma)``, ``x_t = x_sigma/(1+sigma)``; FLUX predicts
    ``v = eps - x_0``, so ``x_t - t*v = x_0`` exactly).

    ``differentiable=False`` (default) wraps the transformer call in ``torch.no_grad()``, matching every
    existing caller bitwise. ``differentiable=True`` skips it (``contextlib.nullcontext()``) so gradients
    flow from the returned prediction back to ``x`` -- the caller is responsible for freezing the
    transformer's own parameters (``requires_grad_(False)``) so only ``x`` carries a gradient; this
    function does not mutate the model.

    Works on flat ``(B, d)`` inputs, matching the ``Denoiser`` protocol used by ``chunked_denoiser`` /
    ``edm_score_fn`` / ``edm_generator``, none of which need any change to support this -- both are
    already autograd-compatible (pure ``torch.cat``/reshape/subtract, no ``.detach()``/``no_grad()``).
    """
    c, h, w = img_shape
    grad_ctx: Callable[[], contextlib.AbstractContextManager] = (
        contextlib.nullcontext if differentiable else torch.no_grad
    )

    def denoiser(x: Float[Tensor, "B ..."], sigma: Float[Tensor, "B"]) -> Float[Tensor, "B ..."]:
        # "B ..." (not "B d"), matching generators.base.eps_to_edm_denoiser's precedent: edm_score_fn /
        # edm_generator call this already reshaped to (B, *img_shape) whenever img_shape is set (the
        # production case -- auto_r_common.py's original always received 4D here), so a flat-only
        # annotation is simply wrong, not just stricter -- it rejects a call shape this function has
        # always had to handle. The reshape below is then a no-op on that path and a real reshape on the
        # (also-supported) flat-input path exercised directly by tests/test_flux_denoiser.py.
        b = x.shape[0]
        x_img = x.reshape(b, c, h, w)
        sig = sigma.reshape(b, 1, 1, 1).to(torch.float32)
        t = (sigma / (1.0 + sigma)).to(torch.float32)
        x_t = x_img.to(torch.float32) / (1.0 + sig)
        v = _flux_raw_velocity(
            transformer, x_t, t, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
            img_ids=img_ids, txt_ids=txt_ids, guidance=guidance, img_shape=img_shape, img_px=img_px,
            dtype=dtype, grad_ctx=grad_ctx,
        )
        x_pred = (x_t - t.reshape(b, 1, 1, 1) * v).to(torch.float32)
        return x_pred.reshape(b, c * h * w)

    return denoiser


def flux_velocity_fn(
    transformer: nn.Module,
    prompt_embeds: Float[Tensor, "1 seq d_txt"],
    pooled_prompt_embeds: Float[Tensor, "1 d_pool"],
    img_ids: Tensor,
    txt_ids: Tensor,
    *,
    guidance: float,
    img_shape: tuple[int, int, int],
    img_px: int,
    dtype: torch.dtype,
    differentiable: bool = False,
) -> GuidableVelocityFn:
    """Raw FLUX velocity ``v_theta(x_t, t)`` in the DIFFUSERS-NATIVE time convention: ``t = 1`` pure
    noise, ``t = 0`` clean data.

    **Convention warning** (this repo has a history of silent sign errors exactly here -- see
    ``generators/flux_flowmap.py``'s docstring and its ``denoise`` vs ``map(x,t,t)`` warning): this ``t``
    is the SAME as ``flux_edm_denoiser``'s internal ``t = sigma/(1+sigma)`` (both native-diffusers,
    noise-increasing), and the OPPOSITE polarity to ``flux_flowmap.py``'s own stated "this repo" convention
    (``t=0`` noise, ``t=1`` data), which applies only to that module's flow map. Do not mix the two.

    Unlike ``flux_edm_denoiser``, this function takes ``t`` directly with no EDM-sigma reparameterization:
    the sigma remap ``sigma = t/(1-t)`` is singular exactly where FLUX generation starts (``t=1``), and
    recovering ``v`` from a returned ``x_hat_0`` needs ``v = (x_t - x_hat_0)/t``, a 0/0 at ``t=0``. This
    function returns ``v`` itself, so callers needing ``x_hat_0 = x_t - t*v`` (e.g. test-time guidance)
    compute it themselves with no division.

    ``differentiable=True`` freezes every transformer parameter (``requires_grad_(False)``) and sets
    ``eval()`` itself (unlike ``flux_edm_denoiser``, which leaves freezing to the caller) -- the returned
    callable also carries a ``.module`` attribute pointing at the transformer, so a caller building an
    autograd graph through it can assert the freeze actually held before doing so, rather than assume it.
    Works on flat ``(B, d)`` inputs, matching ``img_shape``, like every other model wrapper in this module.
    """
    if differentiable:
        _freeze(transformer)
    c, h, w = img_shape
    grad_ctx: Callable[[], contextlib.AbstractContextManager] = (
        contextlib.nullcontext if differentiable else torch.no_grad
    )

    def velocity(x_t: Float[Tensor, "B d"], t: float) -> Float[Tensor, "B d"]:
        b = x_t.shape[0]
        x_img = x_t.reshape(b, c, h, w).to(torch.float32)
        t_row = torch.full((b,), t, device=x_t.device, dtype=torch.float32)
        v = _flux_raw_velocity(
            transformer, x_img, t_row, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
            img_ids=img_ids, txt_ids=txt_ids, guidance=guidance, img_shape=img_shape, img_px=img_px,
            dtype=dtype, grad_ctx=grad_ctx,
        )
        return v.reshape(b, c * h * w)

    velocity.module = transformer  # type: ignore[attr-defined]  # exposed so callers can verify the freeze
    return cast(GuidableVelocityFn, velocity)  # true as of the line above; pyright can't see the dynamic attr


def build_flux_denoiser(
    *,
    model_id: str = "black-forest-labs/FLUX.1-dev",
    prompt: str = "A dog",
    guidance: float = 1.5,
    img: int = 512,
    max_denoiser_rows: int = 24,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    differentiable: bool = False,
) -> Denoiser:
    """Load FLUX.1-dev (or another FLUX checkpoint) and return a chunked EDM ``Denoiser``.

    ``differentiable=True`` explicitly freezes every transformer parameter
    (``requires_grad_(False)``) so that, when the caller also sets ``x.requires_grad_(True)``, only the
    input latent carries a gradient -- safety mechanism 1 from ``notebooks/iid_iem_flux_check/ROADMAP.md``
    Phase 2. The prompt encoders are always run under ``no_grad()`` and discarded after encoding: the
    prompt conditioning itself is never a gradient target, matching ``auto_r_common.py``'s ``build()``.

    Default ``model_id`` is FLUX.1-dev (not FLUX.1-schnell, unlike ``build_flux_generator`` above) to
    match Phase 1 and every FLUX finding in CLAUDE.md, all anchored on FLUX.1-dev.
    """
    try:
        from diffusers import FluxPipeline  # type: ignore[attr-defined]  # older diffusers lack this
    except ImportError as e:                                  # pragma: no cover - exercised without deps
        raise ImportError(_MISSING_DEPS_MSG) from e

    device = torch.device(device)
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, device=device, max_sequence_length=512)
    assert prompt_embeds is not None and pooled_prompt_embeds is not None, "encode_prompt returned None"
    # Free both text encoders (~9.5 GB, T5 dominates): the embeddings above are all we need.
    pipe.text_encoder = pipe.text_encoder_2 = pipe.tokenizer = pipe.tokenizer_2 = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    transformer = pipe.transformer.eval()
    if differentiable:
        _freeze(transformer)

    c, h, w = 16, img // 8, img // 8
    img_ids = FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)
    txt_ids = text_ids if text_ids.ndim == 2 else text_ids[0]

    denoiser = flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=guidance, img_shape=(c, h, w), img_px=img, dtype=dtype, differentiable=differentiable,
    )
    return chunked_denoiser(denoiser, max_denoiser_rows)


def build_flux_generator(
    prompt: str,
    *,
    model_id: str = "black-forest-labs/FLUX.1-schnell",
    n_steps: int = 4,
    device: str = "cuda",
) -> Callable[[Float[Tensor, "B d"]], Float[Tensor, "B d"]]:
    """Build a deterministic latent-space generator ``G(z)`` for a text prompt on FLUX (GPU).

    SCAFFOLD: pre-encodes the prompt (dual encoders) to fix the interface, but the velocity->denoiser +
    latent-packing conversion is not yet implemented (needs GPU validation).
    """
    try:
        from diffusers import FluxPipeline  # type: ignore[attr-defined]  # older diffusers lack this
    except ImportError as e:                                  # pragma: no cover - exercised without deps
        raise ImportError(_MISSING_DEPS_MSG) from e

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

    # Pre-encode the prompt with FLUX's dual (CLIP + T5) encoders to lock the interface.
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, max_sequence_length=256
        )

    # TODO(gpu-validation): wire the FLUX transformer into the EDM denoiser convention.
    #   FLUX predicts a velocity on *packed* latents (B, seq, 16*2*2) with img_ids/txt_ids/guidance, so a
    #   correct denoiser must (a) pack (16,64,64) -> sequence + build the position ids, (b) call
    #   pipe.transformer(hidden_states=..., timestep=sigma-mapped, encoder_hidden_states=prompt_embeds,
    #   pooled_projections=pooled_prompt_embeds, ...), (c) convert the returned velocity to E[X|x_sigma],
    #   and (d) unpack back to (16,64,64).  Validate end-to-end on GPU before enabling.
    raise NotImplementedError(
        "build_flux_generator is a scaffold: the FLUX velocity->EDM-denoiser + latent-packing mapping "
        "is not implemented yet (requires GPU validation). Use build_tiny_sd_generator on CPU or "
        "build_edm_pixel_generator with an EDM checkpoint in the meantime."
    )

    # Once implemented, the module will end with:
    #   return edm_generator(denoiser, img_shape=(16, 64, 64), n_steps=n_steps)


@dataclass
class GuidanceBackend:
    """Everything a ``flow_guided_sample`` caller needs from one pretrained flow-matching model.

    The shape any ``build_<backend>_guidance`` function should return (this module's
    ``build_flux_guidance`` is the reference implementation) -- defined here rather than in
    ``_types.py`` to avoid a circular import (``Denoiser`` comes from ``distances/edm_adapter.py``,
    which itself imports ``ScoreFn`` from ``_types.py``); promote it to a shared location if/when a
    second backend actually needs to import this exact type.
    """

    velocity_fn: GuidableVelocityFn
    denoiser: Denoiser                                         # for building the reward's score_fn
    decode: Callable[[Float[Tensor, "B d"]], Tensor] | None    # flat latents -> images in [0, 1]
    d: int
    device: torch.device
    dtype: torch.dtype


def build_flux_guidance(
    *,
    model_id: str = "black-forest-labs/FLUX.1-dev",
    prompt: str = "A dog",
    guidance: float = 1.5,
    img: int = 512,
    max_denoiser_rows: int = 24,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    differentiable: bool = True,
) -> GuidanceBackend:
    """Load FLUX.1-dev once and return everything a ``flow_guided_sample`` caller needs: a
    differentiable ``velocity_fn`` (for the guided ODE), a chunked EDM ``denoiser`` (for the reward's
    ``score_fn``), and a ``decode`` closure (latents -> images).

    Mirrors the loading steps currently duplicated by hand in
    ``notebooks/flux_guided_phase3/{guided_sweep.py, fine_lambda_sweep.py}`` and the deliverable
    notebook's setup cell -- this is the single-call replacement for that duplication, for future
    callers. Today's three callers are intentionally left as they are; this function is additive.
    """
    try:
        from diffusers import FluxPipeline  # type: ignore[attr-defined]  # older diffusers lack this
    except ImportError as e:                                  # pragma: no cover - exercised without deps
        raise ImportError(_MISSING_DEPS_MSG) from e

    device = torch.device(device)
    c, h, w = 16, img // 8, img // 8
    d = c * h * w

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, device=device, max_sequence_length=512)
    assert prompt_embeds is not None and pooled_prompt_embeds is not None, "encode_prompt returned None"
    # Free both text encoders (~9.5 GB, T5 dominates): the embeddings above are all we need.
    pipe.text_encoder = pipe.text_encoder_2 = pipe.tokenizer = pipe.tokenizer_2 = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    transformer, vae = pipe.transformer.eval(), pipe.vae.eval()
    vae_sf, vae_shift = vae.config.scaling_factor, vae.config.shift_factor
    img_ids = FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)
    txt_ids = text_ids if text_ids.ndim == 2 else text_ids[0]

    velocity_fn = flux_velocity_fn(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=guidance, img_shape=(c, h, w), img_px=img, dtype=dtype, differentiable=differentiable,
    )
    denoiser = flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=guidance, img_shape=(c, h, w), img_px=img, dtype=dtype, differentiable=differentiable,
    )
    denoiser = chunked_denoiser(denoiser, max_denoiser_rows)

    def decode(flat: Float[Tensor, "B d"], chunk: int = 2) -> Tensor:
        outs = []
        for i in range(0, flat.shape[0], chunk):
            lat = flat[i : i + chunk].reshape(-1, c, h, w).to(device, dtype) / vae_sf + vae_shift
            with torch.no_grad():
                im = vae.decode(lat).sample
            post: Tensor = pipe.image_processor.postprocess(im.float(), output_type="pt")  # type: ignore[assignment]
            outs.append(post.cpu())
        return torch.cat(outs)

    return GuidanceBackend(
        velocity_fn=velocity_fn, denoiser=denoiser, decode=decode, d=d, device=device, dtype=dtype,
    )
