"""Tiny text-conditioned Stable-Diffusion-1.5 denoiser wrapper (CPU sandbox).

SD 1.5's UNet is a VP epsilon-predictor, so it is adapted to the EDM denoiser convention via
``eps_to_edm_denoiser`` (input scaling ``c_in``, ``sigma -> discrete timestep`` lookup, ``eps -> x0``).
``G`` operates in the 4-channel latent space ``(4, 64, 64)`` -- VAE decode to pixels is out of scope for
``G`` (the SMC works in latent space, matching ``test_pcn_pixel_seam_smoke``).

``diffusers`` / ``transformers`` are imported lazily (optional ``models`` extra); importing this module -- or
``creativity_measure`` -- never requires them.
"""

from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from .base import edm_generator, eps_to_edm_denoiser

_MISSING_DEPS_MSG = (
    "build_tiny_sd_generator needs 'diffusers' and 'transformers'. "
    "Install the optional model deps:  pip install -e '.[models]'"
)


def build_tiny_sd_generator(
    prompt: str,
    *,
    model_id: str = "segmind/tiny-sd",
    n_steps: int = 4,
    device: str = "cpu",
) -> Callable[[Float[Tensor, "B d"]], Float[Tensor, "B d"]]:
    """Build a deterministic latent-space generator ``G(z)`` for a text prompt on tiny-SD (CPU).

    Pre-encodes the prompt once, derives the scheduler's discrete sigmas
    (``sqrt((1 - alphas_cumprod)/alphas_cumprod)``), and adapts the UNet via ``eps_to_edm_denoiser``.
    """
    try:
        import safetensors.torch  # noqa: F401  # diffusers 0.27 needs the .torch submodule populated
        from diffusers import StableDiffusionPipeline
    except ImportError as e:                                  # pragma: no cover - exercised without deps
        raise ImportError(_MISSING_DEPS_MSG) from e

    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
    pipe = pipe.to(device)

    # 1. Pre-encode the text prompt once.
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # 2. Discrete noise schedule sigmas of the underlying DDPM scheduler (ascending in noise).
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device=device, dtype=prompt_embeds.dtype)
    model_sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod).sqrt()

    # 3. Epsilon predictor with text conditioning baked in.
    def eps_fn(x_in: Tensor, timestep: Tensor) -> Tensor:
        with torch.no_grad():
            return pipe.unet(x_in, timestep, encoder_hidden_states=prompt_embeds).sample

    denoiser = eps_to_edm_denoiser(eps_fn, model_sigmas)

    # 4. Standard generator interface, in SD latent space (4, 64, 64).
    return edm_generator(
        denoiser,
        img_shape=(4, 64, 64),
        sigma_min=2e-3,
        sigma_max=float(model_sigmas[-1]),
        n_steps=n_steps,
    )
