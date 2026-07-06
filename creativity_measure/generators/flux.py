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

from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor

# NOTE: once implemented, this module will import and use ``edm_generator`` / ``eps_to_edm_denoiser``
# from ``.base`` to return the standard generator interface (see the scaffold TODO below).

_MISSING_DEPS_MSG = (
    "build_flux_generator needs 'diffusers' and 'transformers'. "
    "Install the optional model deps:  pip install -e '.[models]'"
)


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
