"""FLUX.1-dev two-timestep **flow map** wiring -- model mechanics only, no prompt, no ``p``.

``gabeguofanclub/flux-1-dev-flowmap-lsd`` is a two-timestep flow map distilled from FLUX.1-dev. Given a
state and a *pair* of times it returns a velocity ``u(x; t -> t_to)``, and the map is
``X_{t,t_to}(x) = x + (t - t_to)·u(x; t -> t_to)``. The weights ship as a rank-64 LoRA over the
FLUX.1-dev transformer plus a **second timestep embedder**, and are applied to it at load time; from
that point on there is one model, loaded once, never modified during a run.

This module lives in ``generators/`` next to ``flux.py`` (``backends/`` exists for the JAX bridge,
whereas this is an ordinary torch model factory) and is the only module that imports ``diffusers``.
It provides the pieces the notebook assembles -- following how ``generators/base.py`` exposes
``edm_generator`` while the notebook supplies the denoiser closure. **Everything that defines ``p``
(the prompt, the guidance scale, the encoded embeddings) is the notebook's business.**

Two operations, and they are not interchangeable::

    map(x, t, 1)   = x - sigma_t·u(x; t -> 1)     ODE endpoint, a sharp image  -> REWARD, LIKELIHOOD
    denoise(x, t)  = x - sigma_t·u(x; t -> t)     Tweedie mean E[x_1|x_t]      -> the score model

Both come from the same network, differing only in the second time argument, but **``denoise`` is not
``map(x, t, t)``**: ``X_{t,t}`` is the identity by definition (``x + 0·u``), which is correct flow-map
behaviour and exactly why the map alone cannot express the denoiser -- the velocity is multiplied away.
``denoise`` keeps ``u(x; t -> t)`` and applies ``-sigma_t`` to it. Getting this wrong fails *silently*:
``s_t`` would come out as ``(alpha_t·x - x)/sigma_t^2 = -x/(1-t)``, a plausible-looking score (that of
``N(0, sigma_t^2)``, i.e. data collapsed to the origin), quietly corrupting both the reward and ``S_k``.
Hence they are separate named methods, never derived from one another.

Time conventions: this repo uses ``t = 0`` noise, ``t = 1`` data; FLUX/``diffusers`` uses
``1 = noise, 0 = data``, i.e. ``sigma_flux = 1 - t``. Every value crossing that boundary is flipped
**here**, in ``_velocity``, and nowhere else -- it is the most likely place for a silent sign error.
"""

import copy
from collections.abc import Callable
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn

from creativity_measure._types import FlowMap, Schedule
from creativity_measure.distances.edm_adapter import Denoiser

__all__ = [
    "DualTimeEmbedder",
    "load_flow_map_weights",
    "FluxFlowMap",
    "flow_map_denoiser",
    "DEFAULT_FLOW_MAP_REPO",
    "DEFAULT_FLOW_MAP_WEIGHT_NAME",
]

DEFAULT_FLOW_MAP_REPO = "gabeguofanclub/flux-1-dev-flowmap-lsd"
DEFAULT_FLOW_MAP_WEIGHT_NAME = (
    "01-12-26/runs/res_512_steps_50k_rank_64_lr_1e-4/checkpoint-43000/pytorch_lora_weights.safetensors"
)

_MISSING_PEFT_MSG = (
    "load_flow_map_weights needs 'peft' (diffusers' LoRA loader requires it). Install it with the "
    "cache off the small $HOME quota:  pip install --cache-dir $WORK/.cache/pip peft"
)


class DualTimeEmbedder(nn.Module):
    """Wraps FLUX's ``time_text_embed`` so it accepts a **pair** of timesteps.

    A one-timestep FLUX embedder maps ``(timestep, guidance, pooled_projection) -> conditioning``. The
    flow map needs to condition on ``(t_from, t_to)``, which the checkpoint implements as a second copy
    of the whole embedder whose ``timestep_embedder`` carries its own LoRA; the two conditionings are
    then averaged. Guidance and text projections are identical in both copies (no LoRA lands on them),
    so averaging leaves those parts untouched and averages only the two timestep embeddings.

    ``original_embedder`` / ``second_embedder`` are **pinned by the weight file's keys**
    (``transformer.time_text_embed.{original,second}_embedder.timestep_embedder.linear_{1,2}.lora_{A,B}.weight``)
    -- they must match exactly or nothing loads.

    A one-dimensional ``timestep`` falls through to the original embedder unchanged, so the wrapped
    transformer still behaves like plain FLUX when called the old way.
    """

    # Declared, not just assigned: these two names are a load-bearing part of the file format.
    original_embedder: nn.Module
    second_embedder: nn.Module

    def __init__(self, embedder: nn.Module):
        super().__init__()
        self.original_embedder = embedder
        self.second_embedder = copy.deepcopy(embedder)

    def forward(self, timestep: Tensor, *args: Tensor) -> Tensor:
        if timestep.ndim >= 2 and timestep.shape[-1] == 2:
            first = self.original_embedder(timestep[..., 0], *args)
            second = self.second_embedder(timestep[..., 1], *args)
            return 0.5 * (first + second)
        return self.original_embedder(timestep, *args)


def load_flow_map_weights(
    transformer: nn.Module,
    *,
    repo_id: str = DEFAULT_FLOW_MAP_REPO,
    weight_name: str = DEFAULT_FLOW_MAP_WEIGHT_NAME,
    adapter_name: str = "flowmap",
    validate: bool = True,
) -> nn.Module:
    """Wrap ``time_text_embed`` in a `DualTimeEmbedder`, then load the flow-map LoRA into ``transformer``.

    **Order matters**: wrap *before* loading, or the two-embedder keys have nowhere to land.

    Asserts every tensor in the file is consumed, that there are no unexpected keys, and (when
    ``validate``) that the loaded model differs from the unmodified one on a fixed input. Without that
    last check a silently no-op load means sampling plain FLUX while believing it is the flow map, and
    nothing downstream would catch it.

    Returns the same ``transformer``, modified in place.
    """
    try:
        import peft  # noqa: F401
    except ImportError as e:                                   # pragma: no cover - exercised without deps
        raise ImportError(_MISSING_PEFT_MSG) from e
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(repo_id=repo_id, filename=weight_name)
    state_dict = load_file(path)
    n_tensors = len(state_dict)

    embedder = getattr(transformer, "time_text_embed")
    if not isinstance(embedder, DualTimeEmbedder):
        transformer.time_text_embed = DualTimeEmbedder(embedder)     # type: ignore[assignment]

    expected = {k for k in state_dict if k.startswith("transformer.")}
    if len(expected) != n_tensors:
        raise RuntimeError(
            f"{n_tensors - len(expected)} of {n_tensors} tensors in {weight_name} are not prefixed "
            f"'transformer.' and would be silently dropped by the adapter loader"
        )

    probe = _probe_input(transformer)
    before = _probe_forward(transformer, probe) if validate else None

    transformer.load_lora_adapter(                                    # type: ignore[attr-defined]
        state_dict, prefix="transformer", adapter_name=adapter_name
    )

    # Every LoRA target must have materialized; a typo'd module name is dropped silently otherwise.
    loaded = {n for n, _ in transformer.named_parameters() if f"lora_A.{adapter_name}" in n}
    n_expected_a = sum(1 for k in expected if k.endswith("lora_A.weight"))
    if len(loaded) != n_expected_a:
        raise RuntimeError(
            f"flow-map load consumed {len(loaded)} of {n_expected_a} lora_A tensors -- "
            f"unexpected or unmatched keys in {weight_name}"
        )
    if not any("second_embedder" in n for n in loaded):
        raise RuntimeError(
            "the second timestep embedder received no LoRA weights: DualTimeEmbedder must be "
            "installed BEFORE load_lora_adapter, and its attribute names must be "
            "'original_embedder' / 'second_embedder'"
        )

    if validate:
        after = _probe_forward(transformer, probe)
        assert before is not None
        delta = float((after - before).abs().max())
        if delta == 0.0:
            raise RuntimeError(
                "loading the flow-map weights did not change the transformer's output: the load was a "
                "no-op, and sampling would silently run plain FLUX.1-dev"
            )
        print(f"flow map loaded: {n_tensors} tensors, {len(loaded)} LoRA sites, "
              f"max|delta| on a fixed input = {delta:.3e}")
    return transformer


def _probe_input(transformer: nn.Module) -> dict[str, Tensor]:
    """A tiny fixed forward-pass input (2 image tokens, 1 text token), for the load-changed-it check."""
    cfg: Any = transformer.config                                     # type: ignore[attr-defined]
    device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype
    gen = torch.Generator(device="cpu").manual_seed(0)
    return {
        "hidden_states": torch.randn((1, 2, cfg.in_channels), generator=gen).to(device, dtype),
        "encoder_hidden_states": torch.randn((1, 1, cfg.joint_attention_dim), generator=gen).to(device, dtype),
        "pooled_projections": torch.randn((1, cfg.pooled_projection_dim), generator=gen).to(device, dtype),
        "timestep": torch.full((1,), 0.5, device=device, dtype=dtype),
        "guidance": torch.full((1,), 1.0, device=device, dtype=torch.float32),
        "img_ids": torch.zeros((2, 3), device=device, dtype=dtype),
        "txt_ids": torch.zeros((1, 3), device=device, dtype=dtype),
    }


def _probe_forward(transformer: nn.Module, probe: dict[str, Tensor]) -> Tensor:
    with torch.no_grad():
        out = transformer(**probe, return_dict=False)[0]
    return out.detach().float().cpu()


class FluxFlowMap:
    """The `FlowMap` protocol backed by the FLUX flow-map transformer.

    Takes **already-encoded embeddings**, so the prompt is the notebook's business. Packing flat
    ``(B, d)`` particles to ``(B, C, H, W)`` and on to FLUX's patchified sequence happens inside, so the
    sampler stays in flat ``(M, d)`` throughout -- exactly as ``edm_generator(..., img_shape=...)`` does.

    Args:
        transformer:  a FLUX transformer already carrying the flow-map weights
                      (see :func:`load_flow_map_weights`).
        prompt_embeds:        ``(1, L, D)`` T5 embeddings.
        pooled_prompt_embeds: ``(1, P)`` CLIP pooled embeddings.
        text_ids:             ``(L, 3)`` text position ids (2-D in diffusers 0.39).
        guidance:     the guidance-embedding scalar. Part of what defines ``p``; the notebook picks it.
        img_shape:    latent shape ``(C, H, W)``, e.g. ``(16, 64, 64)`` at 512 res (``d = 65536``).
        dtype:        the transformer's compute dtype. Inputs and outputs stay in the *caller's* dtype
                      (fp32 for the SMC), which is what keeps weight/ESS arithmetic out of bf16.
        max_rows:     VRAM ceiling on a single forward pass, exactly like
                      ``edm_adapter.chunked_denoiser``. Rows are independent -- FLUX attends over the
                      token sequence *within* a sample, never across the batch -- so splitting is
                      **mathematically** identical. It is *not* bitwise identical: BLAS picks different
                      kernels per batch size, so the reduction order changes and results differ at
                      float epsilon (measured 2.5e-7 relative). ``max_rows`` must therefore be **FIXED
                      for a run**, like the denoiser cap, or ``f`` stops being a deterministic function
                      of ``x`` (CLAUDE.md invariant 1) -- which is why it is a constructor argument and
                      not a per-call knob. It matters because the sampler calls ``map`` on ``M*K`` rows
                      for the lookahead (32 at the operating point) and reference generation calls it
                      on ``R`` (64), while a 12 B bf16 transformer already holds ~24 GB of weights.
                      ``None`` disables chunking.
    """

    def __init__(
        self,
        transformer: nn.Module,
        prompt_embeds: Tensor,
        pooled_prompt_embeds: Tensor,
        text_ids: Tensor,
        *,
        guidance: float,
        img_shape: tuple[int, int, int],
        dtype: torch.dtype = torch.bfloat16,
        max_rows: int | None = 24,
    ):
        from diffusers import FluxPipeline                            # type: ignore[attr-defined]

        if not isinstance(getattr(transformer, "time_text_embed", None), DualTimeEmbedder):
            raise TypeError(
                "this transformer's time_text_embed is not a DualTimeEmbedder, so it cannot accept the "
                "(t_from, t_to) pair a flow map needs. Call load_flow_map_weights(transformer) first -- "
                "it wraps the embedder and then loads the flow-map LoRA, in that order."
            )
        self.transformer = transformer
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.text_ids = text_ids if text_ids.ndim == 2 else text_ids[0]
        self.guidance = float(guidance)
        self.img_shape = img_shape
        self.dtype = dtype
        if max_rows is not None and max_rows < 1:
            raise ValueError(f"max_rows must be >= 1 or None, got {max_rows}")
        self.max_rows = max_rows
        self._pack = FluxPipeline._pack_latents
        self._unpack = FluxPipeline._unpack_latents

        c, h, w = img_shape
        self.d = c * h * w
        self.height_px, self.width_px = h * 8, w * 8
        device = next(transformer.parameters()).device
        # Both must be 2-D in diffusers 0.39; _prepare_latent_image_ids takes the *packed* (half) grid.
        self.img_ids = FluxPipeline._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)

    def _velocity(
        self, x: Float[Tensor, "B d"], t_from: float, t_to: float
    ) -> Float[Tensor, "B d"]:
        """``u(x; t_from -> t_to)`` -- the network call both public methods route through.

        Split into blocks of at most ``max_rows`` (see the class docstring): a pure VRAM ceiling, since
        rows do not interact inside the transformer.
        """
        if self.max_rows is not None and x.shape[0] > self.max_rows:
            return torch.cat(
                [self._velocity_block(x[i:i + self.max_rows], t_from, t_to)
                 for i in range(0, x.shape[0], self.max_rows)],
                dim=0,
            )
        return self._velocity_block(x, t_from, t_to)

    def _velocity_block(
        self, x: Float[Tensor, "B d"], t_from: float, t_to: float
    ) -> Float[Tensor, "B d"]:
        """One forward pass, no chunking.

        The ONLY place the ``sigma_flux = 1 - t`` convention flip happens. The pair is handed over as
        ``timestep = stack([1 - t_from, 1 - t_to], dim=-1)`` **as 0-1 floats, NOT scaled by 1000** --
        ``FluxTransformer2DModel.forward`` does ``timestep = timestep.to(dtype) * 1000`` itself
        (verified in the installed diffusers 0.39), and it broadcasts over the trailing pair axis.
        """
        c, h, w = self.img_shape
        b = x.shape[0]
        x_img = x.reshape(b, c, h, w)
        packed = self._pack(x_img.to(self.dtype), b, c, h, w)
        timestep = torch.tensor([1.0 - t_from, 1.0 - t_to], device=x.device, dtype=torch.float32)
        with torch.no_grad():
            v = self.transformer(
                hidden_states=packed,
                timestep=timestep.expand(b, 2).to(self.dtype),
                guidance=torch.full((b,), self.guidance, device=x.device, dtype=torch.float32),
                pooled_projections=self.pooled_prompt_embeds.expand(b, -1).to(self.dtype),
                encoder_hidden_states=self.prompt_embeds.expand(b, -1, -1).to(self.dtype),
                txt_ids=self.text_ids,
                img_ids=self.img_ids,
                return_dict=False,
            )[0]
        v = self._unpack(v, self.height_px, self.width_px, 8)
        return v.reshape(b, self.d).to(x.dtype)

    def map(
        self, x: Float[Tensor, "B d"], t_from: float, t_to: float
    ) -> Float[Tensor, "B d"]:
        """``X_{t_from,t_to}(x) = x + (t_from - t_to)·u(x; t_from -> t_to)``.

        Sign check: ``map(x, t, 1) = x - sigma_t·u``, matching the validated ``x_t - t·v`` of
        ``flux_dev_strong_tilt_sweep_2_1.ipynb`` cell 3 under that notebook's flipped time convention.
        ``X_{t,t}`` is the identity, by construction -- which is why `denoise` exists separately.
        """
        if t_from == t_to:
            return x
        return x + (t_from - t_to) * self._velocity(x, t_from, t_to)

    def denoise(self, x: Float[Tensor, "B d"], t: float) -> Float[Tensor, "B d"]:
        """``E[x_1 | x_t] = x - sigma_t·u(x; t -> t)`` -- the *instantaneous* velocity, kept, not multiplied away."""
        return x - (1.0 - t) * self._velocity(x, t, t)


def flow_map_denoiser(flow_map: FlowMap, schedule: Schedule) -> Denoiser:
    """Adapt a `FlowMap` to the repo's EDM `Denoiser` convention ``D(y_sigma, sigma) = E[X | y_sigma]``.

    EDM observes ``y_sigma = x_1 + sigma_EDM·eps``; the interpolant carries
    ``x_t = alpha_t·x_1 + sigma_t·eps``. They are the same observation up to scale with
    ``sigma_EDM = sigma_t/alpha_t`` and ``x_t = alpha_t·y_sigma``, so::

        t   = schedule.t_of_snr(sigma_EDM^2)      (g(t) = sigma_t^2/alpha_t^2)
        D   = flow_map.denoise(alpha_t·y_sigma, t)

    Note ``denoise``, never ``map(·, t, 1)``: only a conditional mean backs a marginal score, and this
    is the denoiser ``edm_score_fn`` wraps to produce the run's single ``score_fn`` -- the one object
    used by both the IEM reward and the sampler's ``S_k``.

    Works on flat ``(B, d)`` or on ``(B, *img_shape)``; the caller's shape is restored on return, so it
    slots into ``chunked_denoiser`` / ``edm_score_fn(..., img_shape)`` exactly like the EDM denoisers.
    The ODE uses one sigma across the batch, so a batched ``sigma`` is read at index 0.
    """
    def denoiser(x_sigma: Float[Tensor, "B ..."], sigma: Float[Tensor, "B"]) -> Float[Tensor, "B ..."]:
        sig = sigma.reshape(-1)
        s = float(sig[0])
        # One flow-map time per call: a batch with mixed sigmas would silently be denoised at sigma[0].
        if sig.numel() > 1 and not bool((sig == sig[0]).all()):
            raise ValueError("flow_map_denoiser needs one sigma per call; got mixed sigmas in a batch "
                             "(per-row gamma / batched_gamma=True is not supported for the flow map)")
        t = schedule.t_of_snr(s * s)
        shape = x_sigma.shape
        y = x_sigma.reshape(shape[0], -1)
        x_t = schedule.alpha(t) * y
        return flow_map.denoise(x_t, t).reshape(shape)

    return denoiser


def build_flux_flow_map(
    prompt_embeds: Tensor,
    pooled_prompt_embeds: Tensor,
    text_ids: Tensor,
    *,
    transformer: nn.Module,
    guidance: float = 1.0,
    img_shape: tuple[int, int, int] = (16, 64, 64),
    dtype: torch.dtype = torch.bfloat16,
    load_weights: bool = True,
    repo_id: str = DEFAULT_FLOW_MAP_REPO,
    weight_name: str = DEFAULT_FLOW_MAP_WEIGHT_NAME,
) -> FluxFlowMap:
    """Convenience: load the flow-map weights into ``transformer`` (once) and wrap it as a `FluxFlowMap`.

    The notebook still owns the pipeline, the prompt and the encoding -- this only saves repeating the
    wrap-then-load order, which is the part that fails silently when it is done backwards.
    """
    if load_weights:
        load_flow_map_weights(transformer, repo_id=repo_id, weight_name=weight_name)
    return FluxFlowMap(transformer, prompt_embeds, pooled_prompt_embeds, text_ids,
                       guidance=guidance, img_shape=img_shape, dtype=dtype)


# Kept explicit rather than inferred: a `Denoiser` is a bare callable, so this alias documents what
# `flow_map_denoiser` returns at the call site in the notebook.
FlowMapDenoiserFactory = Callable[[FlowMap, Schedule], Denoiser]
