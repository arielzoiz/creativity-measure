"""Tests for the differentiable base-FLUX EDM denoiser (creativity_measure/generators/flux.py).

Phase 2 of notebooks/iid_iem_flux_check/ROADMAP.md needs `torch.autograd.grad(reward, x, create_graph=True)`
through FLUX; every existing FLUX denoiser (auto_r_common.py's build(), flux_flowmap.py's FluxFlowMap)
hardcodes `torch.no_grad()`. These tests drive a **tiny** `FluxTransformer2DModel` (same recipe as
test_flux_flowmap.py) on the CPU to pin: differentiable=False reproduces the exact reference formula
(notebooks/refset_auto_r/auto_r_common.py's build(), lines 393-413) bitwise; differentiable=True lets a
real gradient flow back to x, with the transformer's own parameters frozen; and chunking
(edm_adapter.chunked_denoiser) doesn't change the value or the gradient.
"""
from typing import Any

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from creativity_measure.distances.edm_adapter import chunked_denoiser
from creativity_measure.generators.flux import flux_edm_denoiser

C, H, W = 16, 4, 4          # a tiny "image": d = C*H*W, H//2 = 2 packed rows (same shape as test_flux_flowmap.py)
D = C * H * W
IMG_PX = H * 8              # _unpack_latents wants pixel-space height/width, i.e. latent_dim * 8


def _tiny_transformer() -> Any:
    """A structurally faithful FLUX transformer, ~50k params instead of 12 B."""
    return diffusers.FluxTransformer2DModel(
        patch_size=1, in_channels=C * 4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
        pooled_projection_dim=16, guidance_embeds=True, axes_dims_rope=(2, 4, 2),
    ).eval()


def _conditioning(transformer: Any) -> tuple[Any, ...]:
    from diffusers import FluxPipeline  # noqa: E402

    cfg = transformer.config
    gen = torch.Generator().manual_seed(0)
    prompt_embeds = torch.randn((1, 3, cfg.joint_attention_dim), generator=gen)
    pooled_prompt_embeds = torch.randn((1, cfg.pooled_projection_dim), generator=gen)
    txt_ids = torch.zeros((3, 3))
    img_ids = FluxPipeline._prepare_latent_image_ids(1, H // 2, W // 2, torch.device("cpu"), torch.float32)
    return prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids


def _build(transformer: Any, *, differentiable: bool):
    prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids = _conditioning(transformer)
    return flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=1.0, img_shape=(C, H, W), img_px=IMG_PX, dtype=torch.float32,
        differentiable=differentiable,
    )


def _batch(b: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((b, D), generator=gen, dtype=torch.float32)
    sigma = torch.full((b,), 1.5, dtype=torch.float32)
    return x, sigma


def _reference_denoise(transformer: Any, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """auto_r_common.py's build() denoiser closure (lines 393-413), inlined verbatim on a tiny model."""
    from diffusers import FluxPipeline  # noqa: E402

    prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids = _conditioning(transformer)
    b = x.shape[0]
    x_img = x.reshape(b, C, H, W)
    sig = sigma.reshape(b, 1, 1, 1).to(torch.float32)
    t = (sigma / (1.0 + sigma)).to(torch.float32)
    x_t = x_img.to(torch.float32) / (1.0 + sig)
    with torch.no_grad():
        v = transformer(
            hidden_states=FluxPipeline._pack_latents(x_t.to(torch.float32), b, C, H, W),
            timestep=t.to(torch.float32),
            guidance=torch.full((b,), 1.0, device=x.device, dtype=torch.float32),
            pooled_projections=pooled_prompt_embeds.expand(b, -1).to(torch.float32),
            encoder_hidden_states=prompt_embeds.expand(b, -1, -1).to(torch.float32),
            txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
    v = FluxPipeline._unpack_latents(v, IMG_PX, IMG_PX, 8).to(torch.float32)
    return (x_t - t.reshape(b, 1, 1, 1) * v).to(torch.float32).reshape(b, D)


def test_differentiable_false_matches_the_reference_formula_bitwise():
    transformer = _tiny_transformer()
    denoiser = _build(transformer, differentiable=False)
    x, sigma = _batch(3)

    out = denoiser(x, sigma)
    ref = _reference_denoise(transformer, x, sigma)

    assert not out.requires_grad
    assert torch.equal(out, ref)


def test_differentiable_true_produces_a_finite_nonzero_gradient_with_frozen_weights():
    transformer = _tiny_transformer()
    transformer.requires_grad_(False)                       # what build_flux_denoiser does
    denoiser = _build(transformer, differentiable=True)
    x, sigma = _batch(3)
    x.requires_grad_(True)

    out = denoiser(x, sigma)
    assert out.requires_grad
    (grad,) = torch.autograd.grad(out.sum(), x)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    assert all(not p.requires_grad for p in transformer.parameters())


def test_differentiable_true_matches_the_reference_value():
    """differentiable=True must not change the *value*, only whether the graph is built."""
    transformer = _tiny_transformer()
    denoiser = _build(transformer, differentiable=True)
    x, sigma = _batch(3)

    out = denoiser(x, sigma)
    ref = _reference_denoise(transformer, x, sigma)
    assert torch.equal(out.detach(), ref)


def test_double_backward_needs_the_math_sdpa_backend():
    """Reproduces ROADMAP.md Phase 2's "Flash-Attention double-backward trap" -- real, not hypothetical:
    the default SDPA backend raises even on CPU, and `create_graph=True` is exactly when it matters
    (a single backward for grad_x alone does not hit this path; see the next test)."""
    transformer = _tiny_transformer()
    transformer.requires_grad_(False)
    denoiser = _build(transformer, differentiable=True)
    x, sigma = _batch(2)
    x.requires_grad_(True)

    out = denoiser(x, sigma)
    (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)
    with pytest.raises(RuntimeError, match="derivative for .*not implemented"):
        torch.autograd.grad(grad.sum(), x, retain_graph=True)


def test_double_backward_succeeds_under_the_math_sdpa_backend():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    transformer = _tiny_transformer()
    transformer.requires_grad_(False)
    denoiser = _build(transformer, differentiable=True)
    x, sigma = _batch(2)
    x.requires_grad_(True)

    with sdpa_kernel(SDPBackend.MATH):
        out = denoiser(x, sigma)
        (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)
        (grad2,) = torch.autograd.grad(grad.sum(), x, retain_graph=True)
    assert torch.isfinite(grad2).all()


def test_single_backward_does_not_need_the_math_backend():
    """ROADMAP.md: grad_x r(x) alone is a single backward (the score is a network output, not an
    autograd gradient), so double-backward only matters with create_graph=True -- confirm the default
    backend is fine for the single-backward case the actual reward gradient uses."""
    transformer = _tiny_transformer()
    transformer.requires_grad_(False)
    denoiser = _build(transformer, differentiable=True)
    x, sigma = _batch(2)
    x.requires_grad_(True)

    out = denoiser(x, sigma)
    (grad,) = torch.autograd.grad(out.sum(), x)
    assert torch.isfinite(grad).all()


def test_chunking_preserves_value_and_gradient():
    """Not bitwise: BLAS/attention kernel selection legitimately varies with batch size (same
    convention as test_edm_adapter.py's test_max_rows_chunking_is_a_pure_vram_ceiling)."""
    transformer = _tiny_transformer()
    transformer.requires_grad_(False)
    denoiser = _build(transformer, differentiable=True)
    chunked = chunked_denoiser(denoiser, max_rows=2)
    x, sigma = _batch(5)

    x_a = x.clone().requires_grad_(True)
    x_b = x.clone().requires_grad_(True)
    out_a = denoiser(x_a, sigma).sum()
    out_b = chunked(x_b, sigma).sum()
    assert torch.allclose(out_a, out_b, atol=1e-4)

    (ga,) = torch.autograd.grad(out_a, x_a)
    (gb,) = torch.autograd.grad(out_b, x_b)
    assert torch.allclose(ga, gb, atol=1e-3)
