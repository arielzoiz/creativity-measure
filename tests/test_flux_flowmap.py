"""Tests for the FLUX flow-map wiring (creativity_measure/generators/flux_flowmap.py).

The real checkpoint is a 12 B bf16 transformer, so these drive a **tiny** `FluxTransformer2DModel`
with the same module structure on the CPU. That is enough to pin the two things that fail silently:

* the `DualTimeEmbedder`'s attribute names, which are dictated by the published weight file's keys --
  if they drift, `load_lora_adapter` finds nothing to attach and the run samples plain FLUX.1-dev;
* the packing / time-convention flip in `FluxFlowMap._velocity`, where a sign error is invisible
  downstream because everything still produces plausible-looking latents.

`flow_map_denoiser` is covered numerically in ``test_flowmap_smc.py`` against the analytic Gaussian
flow map (it is model-agnostic).
"""

import math
from typing import Any

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from creativity_measure.generators.flux_flowmap import (  # noqa: E402
    DEFAULT_FLOW_MAP_WEIGHT_NAME,
    DualTimeEmbedder,
    FluxFlowMap,
    flow_map_denoiser,
)

# The attribute path every LoRA key in the published file is written against; see the repo's
# `01-12-26/runs/.../pytorch_lora_weights.safetensors`, whose 8 embedder keys are exactly
# transformer.time_text_embed.{original,second}_embedder.timestep_embedder.linear_{1,2}.lora_{A,B}.weight
WEIGHT_FILE_EMBEDDER_KEYS = [
    f"time_text_embed.{which}_embedder.timestep_embedder.linear_{i}.lora_{ab}.weight"
    for which in ("original", "second") for i in (1, 2) for ab in ("A", "B")
]

C, H, W = 16, 4, 4          # a 32x32 "image": d = 256, and H//2 = 2 packed rows
D = C * H * W


def _tiny_transformer() -> Any:
    """A structurally faithful FLUX transformer, ~50k params instead of 12 B.

    Returned as ``Any``: `nn.Module.__getattr__` is typed ``Tensor | Module``, so every ``.config`` /
    ``.time_text_embed`` access below would otherwise need its own cast.
    """
    return diffusers.FluxTransformer2DModel(
        patch_size=1, in_channels=C * 4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
        pooled_projection_dim=16, guidance_embeds=True, axes_dims_rope=(2, 4, 2),
    ).eval()


def _tiny_flow_map(transformer: Any = None, *, guidance: float = 1.0) -> FluxFlowMap:
    transformer = transformer if transformer is not None else _tiny_transformer()
    if not isinstance(transformer.time_text_embed, DualTimeEmbedder):
        # `load_flow_map_weights` does this (before loading); here there are no weights to load.
        transformer.time_text_embed = DualTimeEmbedder(transformer.time_text_embed)
    cfg = transformer.config
    gen = torch.Generator().manual_seed(0)
    return FluxFlowMap(
        transformer,
        torch.randn((1, 3, cfg.joint_attention_dim), generator=gen),
        torch.randn((1, cfg.pooled_projection_dim), generator=gen),
        torch.zeros((3, 3)),
        guidance=guidance, img_shape=(C, H, W), dtype=torch.float32,
    )


# ---------------------------------------------------------------------------------------------------
# DualTimeEmbedder
# ---------------------------------------------------------------------------------------------------

def test_dual_time_embedder_attribute_names_match_the_weight_file():
    """The two sub-embedders must be reachable at exactly the paths the LoRA keys name.

    This is the check that a rename would otherwise turn into "the load was a silent no-op".
    """
    tr: Any = _tiny_transformer()
    tr.time_text_embed = DualTimeEmbedder(tr.time_text_embed)
    names = {n for n, _ in tr.named_modules()}
    for key in WEIGHT_FILE_EMBEDDER_KEYS:
        module_path = key.rsplit(".lora_", 1)[0]
        assert module_path in names, f"{module_path} is missing; the LoRA key {key} would be dropped"
    assert "res_512" in DEFAULT_FLOW_MAP_WEIGHT_NAME     # the 512-res file, matching d = 65536


def test_dual_time_embedder_averages_the_pair_and_falls_through_on_a_scalar():
    inner: Any = _tiny_transformer().time_text_embed
    dual = DualTimeEmbedder(inner)
    g = torch.full((2,), 1000.0)
    pooled = torch.randn((2, 16), generator=torch.Generator().manual_seed(0))
    t1, t2 = torch.full((2,), 200.0), torch.full((2,), 800.0)

    # A 1-D timestep behaves exactly like the unwrapped embedder.
    assert torch.allclose(dual(t1, g, pooled), inner(t1, g, pooled))

    # A pair averages the two conditionings. With `second_embedder` still a copy of the original,
    # that is exactly the mean of the two single-time outputs.
    pair = torch.stack([t1, t2], dim=-1)
    assert pair.shape == (2, 2)
    assert torch.allclose(dual(pair, g, pooled), 0.5 * (inner(t1, g, pooled) + inner(t2, g, pooled)))

    # Order matters: swapping the pair changes the answer (t_from and t_to are not interchangeable).
    swapped = torch.stack([t2, t1], dim=-1)
    second: Any = dual.second_embedder
    second.timestep_embedder.linear_2.weight.data.mul_(1.7)                 # break the symmetry
    assert not torch.allclose(dual(pair, g, pooled), dual(swapped, g, pooled))


def test_wrapped_transformer_accepts_a_timestep_pair():
    """End to end: a (B, 2) timestep survives `forward`'s ``* 1000`` and reaches both embedders."""
    tr: Any = _tiny_transformer()
    tr.time_text_embed = DualTimeEmbedder(tr.time_text_embed)
    kw = dict(hidden_states=torch.randn(2, 4, C * 4), encoder_hidden_states=torch.randn(2, 3, 32),
              pooled_projections=torch.randn(2, 16), guidance=torch.full((2,), 1.0),
              img_ids=torch.zeros(4, 3), txt_ids=torch.zeros(3, 3), return_dict=False)
    with torch.no_grad():
        paired = tr(timestep=torch.tensor([[0.3, 0.7], [0.3, 0.7]]), **kw)[0]
        single = tr(timestep=torch.full((2,), 0.3), **kw)[0]
    assert paired.shape == single.shape
    assert not torch.allclose(paired, single)     # the second time argument actually does something


# ---------------------------------------------------------------------------------------------------
# FluxFlowMap
# ---------------------------------------------------------------------------------------------------

def test_map_is_the_identity_at_equal_times_without_calling_the_network():
    fm = _tiny_flow_map()
    calls = []
    fm._velocity = lambda *a, **k: calls.append(a) or torch.zeros(1)   # type: ignore[method-assign]
    x = torch.randn((3, D))
    assert torch.equal(fm.map(x, 0.4, 0.4), x)
    assert not calls, "X_{t,t} is the identity by definition -- no forward pass needed"


def test_map_and_denoise_use_the_documented_velocity_coefficients():
    """``map = x + (t_from - t_to)·u``  and  ``denoise = x - sigma_t·u``, from the SAME network call."""
    fm = _tiny_flow_map()
    seen: list[tuple[float, float]] = []
    v = torch.randn((3, D), generator=torch.Generator().manual_seed(1))

    def fake(x, t_from, t_to):
        seen.append((t_from, t_to))
        return v

    fm._velocity = fake                                              # type: ignore[method-assign]
    x = torch.randn((3, D), generator=torch.Generator().manual_seed(2))

    assert torch.allclose(fm.map(x, 0.25, 1.0), x + (0.25 - 1.0) * v)
    assert seen[-1] == (0.25, 1.0)
    assert torch.allclose(fm.denoise(x, 0.25), x - (1.0 - 0.25) * v)
    assert seen[-1] == (0.25, 0.25), "denoise must ask for the INSTANTANEOUS velocity u(x; t -> t)"

    # map(x, t, 1) = x - sigma_t·u is the same coefficient denoise uses -- the two differ only in the
    # second time argument, which is precisely why they cannot be derived from one another.
    assert torch.allclose(fm.map(x, 0.25, 1.0), fm.denoise(x, 0.25))
    assert seen[-2:] == [(0.25, 1.0), (0.25, 0.25)]


def test_velocity_flips_the_time_convention_and_does_not_prescale_by_1000():
    """``timestep = stack([1 - t_from, 1 - t_to])`` as 0-1 floats -- ``forward`` multiplies by 1000."""
    fm = _tiny_flow_map()
    captured: dict[str, torch.Tensor] = {}
    real = fm.transformer

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    fm.transformer = spy                                             # type: ignore[assignment]
    fm._velocity(torch.randn((2, D)), 0.25, 0.75)
    ts = captured["timestep"]
    assert ts.shape == (2, 2)
    assert torch.allclose(ts[0], torch.tensor([0.75, 0.25]))          # 1 - t, and NOT scaled by 1000
    assert float(ts.max()) <= 1.0
    assert torch.allclose(captured["guidance"], torch.full((2,), 1.0))


def test_velocity_round_trips_the_flat_latent_shape():
    fm = _tiny_flow_map()
    x = torch.randn((3, D))
    with torch.no_grad():
        v = fm._velocity(x, 0.3, 0.9)
    assert v.shape == x.shape and v.dtype == x.dtype
    assert torch.isfinite(v).all()
    assert fm.d == D and fm.img_ids.shape == (H // 2 * W // 2, 3)


def test_construction_rejects_an_unwrapped_transformer():
    """A `FluxFlowMap` over plain FLUX would crash deep inside diffusers; fail at construction instead."""
    with pytest.raises(TypeError, match="load_flow_map_weights"):
        tr = _tiny_transformer()
        FluxFlowMap(tr, torch.randn((1, 3, 32)), torch.randn((1, 16)), torch.zeros((3, 3)),
                    guidance=1.0, img_shape=(C, H, W), dtype=torch.float32)


def test_max_rows_chunking_is_a_pure_vram_ceiling():
    """Chunking changes nothing but peak memory: rows do not interact inside the transformer.

    This is what lets reference generation (R = 64 rows through `map`) and the M*K = 32-row lookahead
    run on a card whose weights already occupy ~24 GB.

    Agreement is to float epsilon, **not bitwise** -- BLAS selects different kernels per batch size, so
    the reduction order differs. That is why `max_rows` is a constructor argument fixed for the whole
    run rather than a per-call knob: a varying cap would make ``f`` non-deterministic in ``x``.
    """
    tr = _tiny_transformer()
    fm_all = _tiny_flow_map(tr)
    fm_all.max_rows = None
    fm_chunked = _tiny_flow_map(tr)          # same transformer object, so same weights
    fm_chunked.max_rows = 3

    x = torch.randn((10, D), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        whole = fm_all._velocity(x, 0.3, 0.9)
        parts = fm_chunked._velocity(x, 0.3, 0.9)
    assert whole.shape == parts.shape == (10, D)
    assert torch.allclose(whole, parts, atol=1e-5, rtol=1e-5), "chunking changed the result"
    rel = float((whole - parts).abs().max() / whole.abs().max())
    assert rel < 1e-5, f"chunking drift {rel:.2e} is larger than kernel-selection noise"

    # A batch at or below the cap takes the unchunked path, and is then bitwise identical.
    small = torch.randn((2, D), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(fm_chunked._velocity(small, 0.3, 0.9),
                           fm_all._velocity(small, 0.3, 0.9))
    with pytest.raises(ValueError, match="max_rows"):
        FluxFlowMap(tr, torch.randn((1, 3, 32)), torch.randn((1, 16)), torch.zeros((3, 3)),
                    guidance=1.0, img_shape=(C, H, W), dtype=torch.float32, max_rows=0)


def test_pack_unpack_is_lossless():
    """The (B, d) <-> patchified-sequence conversion `_velocity` relies on must be a pure reshuffle."""
    fm = _tiny_flow_map()
    x = torch.randn((3, C, H, W))
    packed = fm._pack(x, 3, C, H, W)
    assert packed.shape == (3, (H // 2) * (W // 2), C * 4)
    back = fm._unpack(packed, H * 8, W * 8, 8)
    assert torch.equal(back, x)


def test_end_to_end_notebook_wiring():
    """The exact chain the notebook builds, on a tiny model: flow map -> denoiser -> score_fn -> reward
    -> `flowmap_smc_sample`.

    Nothing here checks values (a randomly-initialised transformer has no meaningful ``p``); it checks
    that the pieces compose -- the shapes, the gamma/sigma conversions, the chunking, and the one
    ``score_fn`` object feeding BOTH the reward's distance and the sampler's ``S_k``.
    """
    from creativity_measure import (
        LinearSchedule, NormalizedExpectedDistanceReward, SquaredGlobalIEMDistance,
        chunked_denoiser, edm_score_fn, flowmap_smc_sample,
    )

    torch.manual_seed(0)
    schedule = LinearSchedule()
    fm = _tiny_flow_map()

    denoiser = chunked_denoiser(flow_map_denoiser(fm, schedule), 4)
    score_fn = edm_score_fn(denoiser, (C, H, W))

    x_refs = torch.randn((3, D))
    gammas = torch.logspace(-2.0, 2.0, 4, base=2.0)
    distance = SquaredGlobalIEMDistance(None, gammas, num_eps=1, seed=123, score_fn=score_fn)
    reward = NormalizedExpectedDistanceReward(distance=distance, x_refs=x_refs)

    res = flowmap_smc_sample(
        reward, 1.0, 2, flow_map=fm, score_fn=score_fn, schedule=schedule,
        n_steps=4, mc_samples=2, eta=1.5, guid_window=(0.3, 1.0), stoch_window=(0.3, 1.0),
        seed=0, keep_steps=True,
    )
    assert res.X.shape == (2, D)
    assert torch.isfinite(res.X).all() and torch.isfinite(res.logw).all()
    # ts = [0, .25, .5, .75, 1]; guided on ts[n+1] in [0.3, 1.0], so the first step is skipped.
    assert res.guided_history == [False, True, True, True]
    assert res.steps is not None and len(res.steps) == 4
    assert all(math.isfinite(v) for v in res.ess_history)


def test_flow_map_denoiser_inverts_the_snr_relation():
    """``t = t_of_snr(sigma^2)`` and ``x_t = alpha_t·y_sigma`` -- the EDM adapter's whole contract."""
    from creativity_measure import LinearSchedule

    schedule = LinearSchedule()
    seen: list[tuple[torch.Tensor, float]] = []

    class Spy:
        def map(self, x, t_from, t_to):
            raise AssertionError("flow_map_denoiser must call denoise, never map")

        def denoise(self, x, t):
            seen.append((x, t))
            return x

    d = flow_map_denoiser(Spy(), schedule)
    for sigma in (0.1, 1.0, 8.0):
        y = torch.randn((2, 5))
        out = d(y, torch.full((2,), sigma))
        t = schedule.t_of_snr(sigma * sigma)
        assert t == pytest.approx(1.0 / (1.0 + sigma))
        assert torch.allclose(seen[-1][0], schedule.alpha(t) * y)
        assert seen[-1][1] == pytest.approx(t)
        assert out.shape == y.shape
    # Image-shaped input keeps its shape.
    out = d(torch.randn((2, C, H, W)), torch.full((2,), 1.0))
    assert out.shape == (2, C, H, W)
    assert math.isclose(schedule.t_of_snr(1.0), 0.5)
