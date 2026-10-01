"""Tests for the Phase-3 guided sampler (creativity_measure/flux_guided.py).

CPU-only, driving a tiny real FluxTransformer2DModel (same recipe as test_flux_denoiser.py) plus the
real i.i.d. squared-IEM reward wired to it -- proves the plumbing (schedule, windowing, scaling,
freezing, memory isolation, determinism) end-to-end without a GPU. Whether guidance actually raises
E_q[f] on real FLUX is deliberately NOT tested here -- that is the GPU question (CLAUDE.md: E_q[f] read
at intermediate t is inflated; only t=1 is artifact-free), and guidance-direction correctness is checked
separately below with a synthetic, analytically-verifiable reward so it does not depend on whether a
tiny random transformer's gradient happens to be well-behaved.
"""
from typing import Any

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from creativity_measure import (
    NormalizedExpectedDistanceReward, Reward, SquaredIIDGlobalIEMDistance, log_uniform_gammas,
)
from creativity_measure.distances.base import Distance
from creativity_measure.distances.edm_adapter import edm_score_fn
from creativity_measure.flux_guided import FluxGuidedResult, _flux_shifted_schedule, _reward_grad, flux_guided_sample
from creativity_measure.generators.flux import flux_edm_denoiser, flux_velocity_fn

C, H, W = 16, 4, 4          # same tiny "image" shape as test_flux_denoiser.py
D = C * H * W
IMG_PX = H * 8


def _tiny_transformer() -> Any:
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


def _build_velocity_fn(transformer: Any, *, differentiable: bool):
    prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids = _conditioning(transformer)
    return flux_velocity_fn(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=1.0, img_shape=(C, H, W), img_px=IMG_PX, dtype=torch.float32, differentiable=differentiable,
    )


def _build_reward(transformer: Any, *, r_refs: int = 4):
    """A real (tiny-model-backed) SquaredIIDGlobalIEMDistance reward, mirroring
    notebooks/iid_iem_flux_check/phase2_autograd_stress.py's `_build_reward`."""
    prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids = _conditioning(transformer)
    denoiser = flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=1.0, img_shape=(C, H, W), img_px=IMG_PX, dtype=torch.float32, differentiable=True,
    )
    score_fn = edm_score_fn(denoiser, (C, H, W))
    gammas, gweights = log_uniform_gammas(2.0 ** -4, 2.0 ** 4, 8, seed=0, dtype=torch.float32)
    gen = torch.Generator().manual_seed(1)
    x_refs = torch.randn(r_refs, D, generator=gen, dtype=torch.float32)
    dist = SquaredIIDGlobalIEMDistance(None, gammas, gweights, num_eps=1, seed=2, score_fn=score_fn)
    # Harmless on this tiny CPU model, but matches the real fix needed on GPU (job 957050 OOM'd here
    # without it, even at R_REFS=8): the ref bank is grad-free by construction, but differentiable=True
    # has no internal no_grad(), so this needs an outer one -- see phase2_autograd_stress.py's comment.
    with torch.no_grad():
        return NormalizedExpectedDistanceReward(dist, x_refs)


def _run(transformer, *, lam: float = 0.0, n_samples: int = 2, seed: int | None = 0, **kw: Any) -> FluxGuidedResult:
    velocity_fn = _build_velocity_fn(transformer, differentiable=kw.pop("differentiable", True))
    reward = _build_reward(transformer)
    # dict[str, Any]: kw carries a mix of int/float/bool/str, which pyright would otherwise narrow from
    # these two literals alone (dict[str, float]) and then reject at the **defaults spread below -- same
    # fix as tests/test_diamond_smc.py's `defaults` earlier this session.
    defaults: dict[str, Any] = dict(n_steps=3, shift=1.0)
    defaults.update(kw)
    return flux_guided_sample(reward, lam, n_samples, velocity_fn=velocity_fn, seed=seed, **defaults)


# ---------------------------------------------------------------------------------------------------
# lambda = 0 bitwise parity -- the load-bearing test. Never checked at runtime inside the sampler
# (computing an unguided step too would double inference cost for no production benefit).
# ---------------------------------------------------------------------------------------------------

def test_lambda_zero_matches_an_unguided_reference_loop_bitwise():
    transformer = _tiny_transformer()
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)

    res = flux_guided_sample(reward, 0.0, 2, velocity_fn=velocity_fn, n_steps=4, shift=1.0, seed=7)

    gen = torch.Generator().manual_seed(7)
    x = torch.randn(2, D, generator=gen, dtype=torch.float32)
    schedule = _flux_shifted_schedule(4, 1.0, device=x.device, dtype=torch.float32)
    for i in range(4):
        t_from, t_to = float(schedule[i]), float(schedule[i + 1])
        with torch.no_grad():
            v = velocity_fn(x, t_from)
        x = x - (t_from - t_to) * v

    assert torch.equal(res.X, x)
    assert not any(res.guided_history)


# ---------------------------------------------------------------------------------------------------
# flux_velocity_fn cross-validated against flux_edm_denoiser via the sigma remap (ties Phase 3's new
# native-t function to Phase 2's already bitwise-validated formula).
# ---------------------------------------------------------------------------------------------------

def test_velocity_fn_matches_edm_denoiser_under_the_sigma_remap():
    transformer = _tiny_transformer()
    velocity_fn = _build_velocity_fn(transformer, differentiable=False)
    prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids = _conditioning(transformer)
    denoiser = flux_edm_denoiser(
        transformer, prompt_embeds, pooled_prompt_embeds, img_ids, txt_ids,
        guidance=1.0, img_shape=(C, H, W), img_px=IMG_PX, dtype=torch.float32, differentiable=False,
    )

    gen = torch.Generator().manual_seed(3)
    x_t = torch.randn(2, D, generator=gen, dtype=torch.float32)
    t = 0.6
    sigma = t / (1.0 - t)
    x_in = x_t / (1.0 - t)

    v = velocity_fn(x_t, t)
    lhs = x_t - t * v
    rhs = denoiser(x_in, torch.full((2,), sigma, dtype=torch.float32))
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------------------------------
# Guidance window
# ---------------------------------------------------------------------------------------------------

def test_full_window_guides_every_step():
    transformer = _tiny_transformer()
    res = _run(transformer, lam=1.0, n_steps=4, t_start=1.0, t_end=0.0)
    assert all(res.guided_history)


def test_narrow_window_guides_only_the_in_window_steps():
    transformer = _tiny_transformer()
    res = _run(transformer, lam=1.0, n_steps=6, shift=1.0, t_start=0.7, t_end=0.3)
    # schedule (shift=1, n_steps=6): t = 1, 5/6, 4/6, 3/6, 2/6, 1/6, 0 -- in [0.3, 0.7]: 5/6? no; check by t.
    expected = [0.3 <= t <= 0.7 for t in res.t_history]
    assert res.guided_history == expected
    assert any(res.guided_history) and not all(res.guided_history)


# ---------------------------------------------------------------------------------------------------
# Velocity-relative scaling and its static fallback
# ---------------------------------------------------------------------------------------------------

def test_velocity_relative_scaling_matches_lam_times_v_norm():
    transformer = _tiny_transformer()
    res = _run(transformer, lam=0.5, n_steps=3, grad_scaling="velocity")
    for applied, v_norm, static in zip(res.applied_norm_history, res.v_norm_history, res.static_fallback_history):
        if not static:
            assert applied == pytest.approx(0.5 * v_norm, rel=1e-4)


def test_min_v_norm_triggers_the_static_fallback():
    transformer = _tiny_transformer()
    res = _run(transformer, lam=0.5, n_steps=3, min_v_norm=1e9, static_scale=2.0)  # force the fallback always
    assert all(res.static_fallback_history)
    for applied in res.applied_norm_history:
        assert applied == pytest.approx(0.5 * 2.0, rel=1e-4)


# ---------------------------------------------------------------------------------------------------
# Exact vs approximate Jacobian
# ---------------------------------------------------------------------------------------------------

def test_exact_and_approximate_both_give_finite_nonzero_gradients_and_differ():
    transformer = _tiny_transformer()
    res_approx = _run(transformer, lam=1.0, n_steps=2, shift=1.0, exact_jacobian=False, seed=5)
    res_exact = _run(transformer, lam=1.0, n_steps=2, shift=1.0, exact_jacobian=True, seed=5)
    for res in (res_approx, res_exact):
        for gn in res.grad_norm_history:
            assert gn == gn and gn > 0  # not nan, not zero
    assert not torch.equal(res_approx.X, res_exact.X)  # different Jacobians -> different trajectories


def test_exact_jacobian_needs_no_create_graph_on_the_default_backend():
    """Confirms finding (c): completes on the default SDPA backend, no double-backward involved."""
    transformer = _tiny_transformer()
    res = _run(transformer, lam=1.0, n_steps=2, shift=1.0, exact_jacobian=True)
    assert all(gn == gn for gn in res.grad_norm_history)


# ---------------------------------------------------------------------------------------------------
# Frozen-weight guard for exact_jacobian=True
# ---------------------------------------------------------------------------------------------------

def test_exact_jacobian_raises_before_building_a_graph_if_unfrozen():
    transformer = _tiny_transformer()  # NOT frozen
    velocity_fn = _build_velocity_fn(transformer, differentiable=False)  # differentiable=False -> not frozen
    reward = _build_reward(transformer)
    with pytest.raises(ValueError, match="frozen"):
        flux_guided_sample(reward, 1.0, 2, velocity_fn=velocity_fn, n_steps=2, exact_jacobian=True, seed=0)


def test_differentiable_velocity_fn_freezes_and_evals_its_module():
    transformer = _tiny_transformer()
    transformer.train()  # perturb both checks on purpose
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    module = getattr(velocity_fn, "module")  # VelocityFn's static type has no .module; see flux.py
    assert all(not p.requires_grad for p in module.parameters())
    assert not module.training


# ---------------------------------------------------------------------------------------------------
# No graph retention across steps (memory isolation)
# ---------------------------------------------------------------------------------------------------

def test_no_graph_retained_on_the_returned_latents():
    transformer = _tiny_transformer()
    for exact in (False, True):
        res = _run(transformer, lam=1.0, n_steps=3, shift=1.0, exact_jacobian=exact)
        assert not res.X.requires_grad
        assert res.X.grad_fn is None


# ---------------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------------

def test_same_seed_gives_identical_x_seed_none_seeds_from_zero():
    transformer = _tiny_transformer()
    res_a = _run(transformer, lam=1.0, n_steps=2, shift=1.0, seed=42)
    res_b = _run(transformer, lam=1.0, n_steps=2, shift=1.0, seed=42)
    assert torch.equal(res_a.X, res_b.X)

    res_none = _run(transformer, lam=1.0, n_steps=2, shift=1.0, seed=None)
    res_zero = _run(transformer, lam=1.0, n_steps=2, shift=1.0, seed=0)
    assert torch.equal(res_none.X, res_zero.X)


# ---------------------------------------------------------------------------------------------------
# Guidance direction -- synthetic, analytically-verifiable reward (decoupled from whether a tiny random
# transformer's real IEM gradient happens to be well-behaved; that question is for the GPU run).
# ---------------------------------------------------------------------------------------------------

class _QuadraticDistance(Distance):
    """D(X, x_refs) = -0.5*||X||^2 (constant over refs), so Reward(this, zeros(R,d))(X) = -0.5*||X||^2 --
    maximized at the origin, with an analytically known, everywhere-finite gradient -X (no singularity
    at X=0, unlike e.g. a bare L2 norm, which matters since step 0 of the derivation below passes
    through x_hat_0 = 0 exactly)."""

    def pairwise(self, X: torch.Tensor, x_refs: torch.Tensor) -> torch.Tensor:
        R = x_refs.shape[0]
        val = -0.5 * (X ** 2).sum(dim=1, keepdim=True)
        return val.expand(-1, R)


def _identity_velocity(x_t: torch.Tensor, t: float) -> torch.Tensor:
    return x_t.clone()


def test_guidance_moves_toward_higher_reward():
    """Derivation (n_steps=2, shift=1.0, v=x identity): x1 = 0.5*x0 (step 0 is t=1, x_hat_0 = 0 there,
    so guidance is a no-op regardless of lam); at step 1 (t=0.5), x_hat_0 = 0.5*x1 = 0.25*x0 (nonzero),
    grad = -x_hat_0, and the velocity-relative-scaled step gives X = 0.25*x0*(1 - lam) for small lam --
    strictly smaller ||X|| (hence strictly higher f = -0.5||X||^2) than the unguided 0.25*x0."""
    reward = Reward(distance=_QuadraticDistance(), x_refs=torch.zeros(1, 8))

    res_unguided = flux_guided_sample(reward, 0.0, 4, velocity_fn=_identity_velocity, n_steps=2, shift=1.0, seed=11)
    res_guided = flux_guided_sample(reward, 0.3, 4, velocity_fn=_identity_velocity, n_steps=2, shift=1.0, seed=11)

    assert reward(res_guided.X).mean() > reward(res_unguided.X).mean()


# ---------------------------------------------------------------------------------------------------
# OOM fallback (gamma-chunked, SquaredIIDGlobalIEMDistance.expected_gamma_chunk) -- now that step 1b
# has landed. force_fallback=True exercises the chunked branch directly without needing a real OOM.
# ---------------------------------------------------------------------------------------------------

def test_oom_fallback_gradient_matches_the_batched_gradient():
    transformer = _tiny_transformer()
    transformer.requires_grad_(False)
    reward = _build_reward(transformer)
    x = torch.randn(3, D, generator=torch.Generator().manual_seed(21), dtype=torch.float32)

    x_a = x.clone().requires_grad_(True)
    grad_batched, fell_back_batched = _reward_grad(reward, x_a, x_a, force_fallback=False)
    assert not fell_back_batched

    x_b = x.clone().requires_grad_(True)
    grad_chunked, fell_back_chunked = _reward_grad(reward, x_b, x_b, force_fallback=True, g_chunk=1)
    assert fell_back_chunked
    assert torch.allclose(grad_batched, grad_chunked, atol=1e-5, rtol=1e-4)


def test_oom_fallback_exact_path_matches_the_batched_gradient():
    """The exact path's grad target (x_req) differs from the reward's input (x_hat_0) -- confirm the
    fallback's retain_graph=True-until-last-chunk handling (flux_guided.py's documented caveat: the
    shared v_theta subgraph must outlive every non-final chunk) still gives the right gradient."""
    transformer = _tiny_transformer()
    transformer.requires_grad_(False)
    reward = _build_reward(transformer)
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    x = torch.randn(2, D, generator=torch.Generator().manual_seed(22), dtype=torch.float32)
    t = 0.5

    x_req_a = x.clone().requires_grad_(True)
    v_a = velocity_fn(x_req_a, t)
    x_hat0_a = x_req_a - t * v_a
    grad_batched, _ = _reward_grad(reward, x_hat0_a, x_req_a, force_fallback=False)

    x_req_b = x.clone().requires_grad_(True)
    v_b = velocity_fn(x_req_b, t)
    x_hat0_b = x_req_b - t * v_b
    grad_chunked, fell_back = _reward_grad(reward, x_hat0_b, x_req_b, force_fallback=True, g_chunk=1)
    assert fell_back
    assert torch.allclose(grad_batched, grad_chunked, atol=1e-5, rtol=1e-4)


def test_guided_sample_records_oom_fallback_when_forced():
    """End-to-end: flux_guided_sample's own oom_fallback_history reflects the fallback firing, using a
    Distance whose expected() raises OOM on exactly the first grad-requiring call (i.e. the first real
    guided step, whichever call-count that happens to be -- not counted precisely on purpose, so this
    doesn't depend on exactly how many no_grad diagnostic/setup calls precede it) so the real fallback
    branch in _reward_grad (not force_fallback) is what fires -- the closest CPU proxy for a real OOM."""
    transformer = _tiny_transformer()
    transformer.requires_grad_(False)
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)

    real_distance = reward.distance
    state = {"raised": False}

    class _OOMOnFirstGradCall(type(real_distance)):
        def expected(self, X, x_refs, weights=None):
            if X.requires_grad and not state["raised"]:
                state["raised"] = True
                raise RuntimeError("CUDA out of memory.")
            return super().expected(X, x_refs, weights)  # type: ignore[reportAttributeAccessIssue]

    reward.distance.__class__ = _OOMOnFirstGradCall   # swap the class in place; fields are unchanged

    res = flux_guided_sample(reward, 1.0, 2, velocity_fn=velocity_fn, n_steps=2, shift=1.0, seed=5,
                              g_chunk=1)
    assert state["raised"]
    assert any(res.oom_fallback_history)
