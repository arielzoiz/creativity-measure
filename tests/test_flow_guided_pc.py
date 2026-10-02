"""Tests for the Phase-5 predictor-corrector Langevin sampler (samplers/flow_guided_pc.py).

CPU-only. Three kinds of coverage, deliberately separated:

1. ANALYTIC. `velocity_to_score` is an exact identity, so it is checked against closed forms (the `t=1`
   collapse to `-x`, and the true marginal score of Gaussian data), and the ULA machinery is checked
   against the stationary variance it is supposed to produce -- including the known O(eta) discretization
   bias, which has a clean closed form here. None of this depends on a model.
2. REDUCTION. `corrector_steps=0` must reproduce `flow_guided_sample` BITWISE, on the real tiny-FLUX
   reward. This is the load-bearing test: the whole Phase-5 sweep is a three-way comparison that is only
   apples-to-apples because of it, and it doubles as the regression guard on the `guided_euler_step`
   extraction into `flow_guided_common.py`.
3. PLUMBING. Windowing, determinism, freeze guard, memory isolation, OOM fallback, effort accounting --
   the same checklist `test_flow_guided.py` runs for Phase 3, against the tiny real FLUX backend.

Whether the corrector actually extends the creative window on real FLUX is the GPU question
(notebooks/flux_guided_phase5/), not this file's.
"""
import math
from typing import Any

import jaxtyping
import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from creativity_measure import (
    NormalizedExpectedDistanceReward, Reward, SquaredIIDGlobalIEMDistance, log_uniform_gammas,
)
from creativity_measure.distances.base import Distance
from creativity_measure.distances.edm_adapter import edm_score_fn
from creativity_measure.generators.flux import flux_edm_denoiser, flux_velocity_fn
from creativity_measure.samplers.flow_guided import flow_guided_sample
from creativity_measure.samplers.flow_guided_common import _shifted_schedule
from creativity_measure.samplers.flow_guided_pc import (
    SNR_SONG_2021, CorrectorSnapshot, FlowGuidedPCResult, _corrector_drift, _ula_step,
    denoised_from_velocity, flow_guided_pc_sample, ula_step_size, velocity_to_score,
)

C, H, W = 16, 4, 4          # same tiny "image" shape as test_flow_guided.py / test_flux_denoiser.py
D = C * H * W
IMG_PX = H * 8


# ---------------------------------------------------------------------------------------------------
# Fixtures -- mirroring tests/test_flow_guided.py's, so the two suites exercise the same backend
# ---------------------------------------------------------------------------------------------------

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
        guidance=1.0, img_shape=(C, H, W), img_px=IMG_PX, dtype=torch.float32,
        differentiable=differentiable,
    )


def _build_reward(transformer: Any, *, r_refs: int = 4):
    """A real (tiny-model-backed) SquaredIIDGlobalIEMDistance reward -- identical recipe and seeds to
    test_flow_guided.py's, so the bitwise reduction tests below compare like with like."""
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
    with torch.no_grad():
        return NormalizedExpectedDistanceReward(dist, x_refs)


def _run(transformer, *, lam: float = 0.0, n_samples: int = 2, seed: int | None = 0,
         **kw: Any) -> FlowGuidedPCResult:
    velocity_fn = _build_velocity_fn(transformer, differentiable=kw.pop("differentiable", True))
    reward = _build_reward(transformer)
    defaults: dict[str, Any] = dict(n_steps=3, shift=1.0, corrector_steps=1)
    defaults.update(kw)
    return flow_guided_pc_sample(reward, lam, n_samples, velocity_fn=velocity_fn, seed=seed, **defaults)


# ---------------------------------------------------------------------------------------------------
# Analytic backend: p = N(0, sd^2 I) under the rectified-flow interpolant x_t = (1-t) x_0 + t eps.
#
#   Var(x_t) = DD(t) = (1-t)^2 sd^2 + t^2
#   E[x_0|x_t] = (1-t) sd^2 / DD * x_t,   E[eps|x_t] = t / DD * x_t
#   v = E[eps - x_0 | x_t] = (t - (1-t) sd^2) / DD * x_t
#   true score = -x_t / DD
#
# so velocity_to_score(x, v, t) must equal -x/DD identically. This is the one place the score formula is
# checked against ground truth rather than against itself.
# ---------------------------------------------------------------------------------------------------

def _gauss_var(t: float, sd: float) -> float:
    return (1.0 - t) ** 2 * sd ** 2 + t ** 2


def _gaussian_velocity(sd: float):
    def velocity(x_t: torch.Tensor, t: float) -> torch.Tensor:
        return ((t - (1.0 - t) * sd ** 2) / _gauss_var(t, sd)) * x_t
    return velocity


def test_velocity_to_score_collapses_to_minus_x_at_t_one():
    """At t=1 the interpolant IS the prior N(0,I), so the score must be -x for ANY velocity -- the
    (1-t) coefficient kills the model's contribution exactly. Catches a flipped (1-t) vs t."""
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(5, 32, generator=gen)
    for v in (torch.randn(5, 32, generator=gen), torch.zeros(5, 32), 1e3 * torch.ones(5, 32)):
        assert torch.allclose(velocity_to_score(x, v, 1.0), -x, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("t", [0.99, 0.75, 0.5, 0.25, 0.05, 0.01])
@pytest.mark.parametrize("sd", [0.5, 1.0, 2.0])
def test_velocity_to_score_matches_the_true_gaussian_marginal_score(t: float, sd: float):
    """In float64, so this tests the FORMULA rather than its conditioning.

    `eps_hat = x_t + (1-t) v` is a difference of same-order terms, and this synthetic p makes the true
    `eps_hat = -t * x / DD` genuinely tiny at small t with sd != 1 (e.g. sd=2, t=0.01: a 1 - 0.99746
    cancellation), so float32 carries only ~4 digits there. That is a property of the test's p, not of
    the reparametrization -- on real latents `eps_hat` stays O(sqrt(d)) at every t. The float32
    production path is covered separately below.
    """
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(7, 16, generator=gen, dtype=torch.float64) * math.sqrt(_gauss_var(t, sd))
    v = _gaussian_velocity(sd)(x, t)
    s = velocity_to_score(x, v, t)
    assert s.dtype == torch.float64, "float64 must be preserved, not forced down to float32"
    assert torch.allclose(s, -x / _gauss_var(t, sd), atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("t", [0.99, 0.5, 0.05])
def test_velocity_to_score_is_accurate_in_float32_on_the_production_path(t: float):
    """sd=1 makes the interpolant variance-preserving (DD = (1-t)^2 + t^2 with v = 0 at t=0.5), which is
    the well-conditioned regime real latents live in; float32 must be good to ~1e-5 there."""
    sd = 1.0
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(7, 16, generator=gen) * math.sqrt(_gauss_var(t, sd))
    v = _gaussian_velocity(sd)(x, t)
    assert torch.allclose(velocity_to_score(x, v, t), -x / _gauss_var(t, sd), atol=1e-5, rtol=1e-5)


def test_denoised_from_velocity_matches_the_gaussian_posterior_mean():
    """x_hat_0 = x_t - t v must equal E[x_0|x_t] = (1-t) sd^2 / DD * x_t under the optimal velocity."""
    sd, t = 1.5, 0.4
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(6, 16, generator=gen)
    v = _gaussian_velocity(sd)(x, t)
    expected = ((1.0 - t) * sd ** 2 / _gauss_var(t, sd)) * x
    assert torch.allclose(denoised_from_velocity(x, v, t), expected, atol=1e-5, rtol=1e-5)


def test_velocity_to_score_returns_float32_from_bfloat16_inputs():
    """The 1/t factor reaches 100x at the default corrector_t_min; bf16's ~8-bit mantissa cannot carry
    it, so the reparametrization must upcast rather than inherit the latents' dtype."""
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    v = torch.randn(2, 16, dtype=torch.bfloat16)
    assert velocity_to_score(x, v, 0.01).dtype == torch.float32


def test_ula_step_size_algebra():
    ref = torch.tensor([[2.0], [4.0]])
    z = torch.tensor([[3.0], [3.0]])
    eta = ula_step_size(ref, z, 0.16)
    assert torch.allclose(eta, 2.0 * (0.16 * z / ref) ** 2, atol=0, rtol=1e-6)


def test_snr_default_is_songs_value():
    assert SNR_SONG_2021 == 0.16


# ---------------------------------------------------------------------------------------------------
# The ULA step itself, against the stationary distribution it is supposed to sample.
#
# With the exact Gaussian score s = -x/DD and the adaptive eta = 2 (snr ||z|| / ||s||)^2, the chain is
# x <- x(1 - eta/DD) + sqrt(2 eta) z. At stationarity with Var = kappa*DD and the concentrated norms
# ||x||^2 ~ kappa*DD*d, ||z||^2 ~ d, we get eta/DD = 2 snr^2 / kappa; substituting into the AR(1) fixed
# point kappa = 2/(2 - eta/DD) gives kappa = 1 + snr^2 exactly. So ULA at snr=0.16 is expected to
# OVERSHOOT the target variance by 2.56% -- that is the O(eta) unadjusted-Langevin bias, with a closed
# form. Asserting the biased value (not DD) is what makes this a test of the step size rather than of a
# tolerance wide enough to hide it.
#
# Finite d adds two more (1 + 2/d) factors because eta is adaptive -- see `ula_step_size`'s docstring for
# why, and ROADMAP.md Phase 5 for the derivation. They vanish as d grows (6e-5 at the production
# d = 65536), so this is a small-d test artifact; d=1024 below keeps it at the 0.4% level.
# ---------------------------------------------------------------------------------------------------

def test_pure_langevin_corrector_reaches_the_predicted_stationary_variance():
    sd, t, snr, d, n = 1.0, 0.5, 0.16, 1024, 512
    dd = _gauss_var(t, sd)
    velocity_fn = _gaussian_velocity(sd)
    reward = Reward(distance=_NegL2Distance(), x_refs=torch.zeros(1, d))  # unused: lam_corrector = 0
    gen = torch.Generator().manual_seed(11)

    x = torch.randn(n, d, generator=gen) * math.sqrt(dd)   # start AT the target, so this measures drift
    for _ in range(400):
        x = _ula_step(
            x, t, reward=reward, lam_corrector=0.0, velocity_fn=velocity_fn, snr=snr,
            eta_reference="total", exact_jacobian=False, grad_clip_percentile=None, g_chunk=None,
            generator=gen, snapshot=CorrectorSnapshot(t=t),
        )

    predicted = (1.0 + snr ** 2) * (1.0 + 2.0 / d) ** 2 * dd     # both finite-d factors, see above
    measured = float(x.var())
    assert measured == pytest.approx(predicted, rel=0.01), (
        f"{measured=} {predicted=} {dd=} asymptote={(1 + snr ** 2) * dd}")


def test_pure_langevin_bias_grows_with_snr():
    """The (1 + snr^2) overshoot is the discretization bias, so a larger snr must land further above the
    target -- the ordering is the qualitative half of the test above, independent of its tolerance."""
    sd, t, d, n = 1.0, 0.5, 256, 512
    dd = _gauss_var(t, sd)
    velocity_fn = _gaussian_velocity(sd)
    reward = Reward(distance=_NegL2Distance(), x_refs=torch.zeros(1, d))

    measured = []
    for snr in (0.16, 0.5):
        gen = torch.Generator().manual_seed(12)
        x = torch.randn(n, d, generator=gen) * math.sqrt(dd)
        for _ in range(300):
            x = _ula_step(
                x, t, reward=reward, lam_corrector=0.0, velocity_fn=velocity_fn, snr=snr,
                eta_reference="total", exact_jacobian=False, grad_clip_percentile=None, g_chunk=None,
                generator=gen, snapshot=CorrectorSnapshot(t=t),
            )
        measured.append(float(x.var()))
    assert dd < measured[0] < measured[1]


def test_noise_to_drift_ratio_is_exactly_one_over_snr():
    """sqrt(2 eta)||z|| / (eta ||g_total||) = 1/snr identically under eta_reference='total' -- a free
    wiring assert on the step size, independent of the model, the reward and lam."""
    snr, d = 0.16, 64
    velocity_fn = _gaussian_velocity(1.0)
    refs = torch.full((1, d), 3.0)
    reward = Reward(distance=_NegL2Distance(), x_refs=refs)
    gen = torch.Generator().manual_seed(13)
    snapshot = CorrectorSnapshot(t=0.5)
    x = torch.randn(8, d, generator=gen)
    for lam_c in (0.0, 2.0):
        _ula_step(
            x, 0.5, reward=reward, lam_corrector=lam_c, velocity_fn=velocity_fn, snr=snr,
            eta_reference="total", exact_jacobian=False, grad_clip_percentile=None, g_chunk=None,
            generator=gen, snapshot=snapshot,
        )
    assert snapshot.noise_frac == pytest.approx([1.0 / snr] * 2, rel=1e-4)


def test_eta_reference_score_is_lam_independent_while_total_anneals():
    """||g_tilde|| = ||s|| by construction, so 'total' divides by ~(1+lam)||s|| and shrinks eta by
    ~1/(1+lam)^2, while 'score' leaves it untouched. This is the documented coupling, pinned."""
    d = 64
    velocity_fn = _gaussian_velocity(1.0)
    reward = Reward(distance=_NegL2Distance(), x_refs=torch.full((1, d), 3.0))
    gen = torch.Generator().manual_seed(14)
    x = torch.randn(4, d, generator=gen)

    etas: dict[str, list[float]] = {}
    for ref in ("total", "score"):
        snapshot = CorrectorSnapshot(t=0.5)
        for lam_c in (0.0, 3.0):
            _ula_step(
                x, 0.5, reward=reward, lam_corrector=lam_c, velocity_fn=velocity_fn, snr=0.16,
                eta_reference=ref, exact_jacobian=False, grad_clip_percentile=None, g_chunk=None,
                generator=torch.Generator().manual_seed(15), snapshot=snapshot,
            )
        etas[ref] = snapshot.eta

    assert etas["score"][1] == pytest.approx(etas["score"][0], rel=1e-5)
    assert etas["total"][1] < 0.5 * etas["total"][0]


# ---------------------------------------------------------------------------------------------------
# Guidance direction: the corrector's reward term must have the right SIGN.
#
# _QuadraticDistance (test_flow_guided.py's toy) is maximized at the ORIGIN, which is exactly where the
# prior score already points -- so it cannot separate the reward term from the score term. This toy puts
# the reward's optimum at the refs instead, so grad r and s point in genuinely different directions.
# ---------------------------------------------------------------------------------------------------

class _NegL2Distance(Distance):
    """D(X, x_refs)[b, r] = -0.5 ||X[b] - x_refs[r]||^2.

    Reward(this, refs)(X) = -0.5 * mean_r ||X - x_r||^2, maximized at the refs' centroid with the
    everywhere-finite gradient -(X - centroid). Unlike test_flow_guided.py's `_QuadraticDistance`
    (optimum at the origin, i.e. collinear with the prior score of any symmetric p), a centroid away
    from the origin makes grad r and s_theta point in different directions -- required to tell the
    corrector's reward term apart from its score term at all.
    """

    def pairwise(self, X: torch.Tensor, x_refs: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((X[:, None, :] - x_refs[None, :, :]) ** 2).sum(dim=-1)


def test_corrector_drift_tilts_toward_the_reward_gradient():
    """Deterministic sign check, no sampling: projecting g_total onto grad r's unit direction must be
    strictly larger at lam > 0 than at lam = 0, and grow with lam."""
    d, t = 64, 0.5
    velocity_fn = _gaussian_velocity(1.0)
    reward = Reward(distance=_NegL2Distance(), x_refs=torch.full((1, d), 3.0))
    gen = torch.Generator().manual_seed(16)
    x = torch.randn(4, d, generator=gen)

    # grad r = -(x_hat_0 - centroid); recover its direction from the lam=1 drift minus the lam=0 drift.
    g0, _, _, _, _ = _corrector_drift(
        x, t, reward=reward, lam_corrector=0.0, velocity_fn=velocity_fn, exact_jacobian=False,
        grad_clip_percentile=None, g_chunk=None)
    g1, _, _, _, _ = _corrector_drift(
        x, t, reward=reward, lam_corrector=1.0, velocity_fn=velocity_fn, exact_jacobian=False,
        grad_clip_percentile=None, g_chunk=None)
    g3, _, _, _, _ = _corrector_drift(
        x, t, reward=reward, lam_corrector=3.0, velocity_fn=velocity_fn, exact_jacobian=False,
        grad_clip_percentile=None, g_chunk=None)

    u = (g1 - g0)                                 # = 1 * g_tilde, i.e. grad r's direction scaled
    u = u / u.norm(dim=1, keepdim=True)
    proj0 = (g0 * u).sum(dim=1)
    proj1 = (g1 * u).sum(dim=1)
    proj3 = (g3 * u).sum(dim=1)
    assert torch.all(proj1 > proj0)
    assert torch.all(proj3 > proj1)


def test_corrector_raises_the_reward_with_an_unguided_predictor():
    """End-to-end, corrector-only (predictor_guided=False) on the analytic backend: the tilt must raise
    f. eta_reference='score' so lam does not simultaneously anneal the step -- this isolates the sign and
    the magnitude of the reward term from the documented 1/(1+lam) coupling."""
    d = 64
    velocity_fn = _gaussian_velocity(1.0)
    reward = Reward(distance=_NegL2Distance(), x_refs=torch.full((1, d), 3.0))

    kw: dict[str, Any] = dict(
        velocity_fn=velocity_fn, n_steps=6, shift=1.0, predictor_guided=False, corrector_steps=4,
        eta_reference="score", snr=0.3, seed=17,
    )
    res_0 = flow_guided_pc_sample(reward, 0.0, 64, **kw)
    res_lam = flow_guided_pc_sample(reward, 3.0, 64, **kw)
    assert float(reward(res_lam.X).mean()) > float(reward(res_0.X).mean())


# ---------------------------------------------------------------------------------------------------
# Reduction to Phase 3 -- the load-bearing tests
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.0, 1.5])
@pytest.mark.parametrize("exact_jacobian", [False, True])
def test_corrector_steps_zero_reproduces_flow_guided_bitwise(lam: float, exact_jacobian: bool):
    """With no corrector, this sampler IS flow_guided_sample: the same shared guided_euler_step, the same
    generator consumption (z0 and nothing else). Exercised at lam != 0 and in both Jacobian modes, so it
    covers the guided branch too -- not just the trivially-unguided one."""
    transformer = _tiny_transformer()
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    kw: dict[str, Any] = dict(
        velocity_fn=velocity_fn, n_steps=3, shift=1.0, exact_jacobian=exact_jacobian, seed=21,
    )

    ref = flow_guided_sample(reward, lam, 2, **kw)
    pc = flow_guided_pc_sample(reward, lam, 2, corrector_steps=0, **kw)

    assert torch.equal(pc.X, ref.X)
    assert pc.guided_history == ref.guided_history
    assert pc.t_history == ref.t_history
    assert pc.v_norm_history == ref.v_norm_history
    assert pc.applied_norm_history == ref.applied_norm_history
    assert all(not s.ran for s in pc.corrector_history)
    assert len(pc.corrector_history) == 3


def test_predictor_unguided_reproduces_the_plain_ode_bitwise():
    """predictor_guided=False with no corrector must be the model's own ODE, independent of lam -- the
    baseline the corrector-only mode is measured against."""
    transformer = _tiny_transformer()
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)

    pc = flow_guided_pc_sample(reward, 5.0, 2, velocity_fn=velocity_fn, n_steps=4, shift=1.0,
                               predictor_guided=False, corrector_steps=0, seed=22)

    gen = torch.Generator().manual_seed(22)
    x = torch.randn(2, D, generator=gen, dtype=torch.float32)
    schedule = _shifted_schedule(4, 1.0, device=x.device, dtype=torch.float32)
    for i in range(4):
        t_from, t_to = float(schedule[i]), float(schedule[i + 1])
        with torch.no_grad():
            v = velocity_fn(x, t_from)
        x = x - (t_from - t_to) * v

    assert torch.equal(pc.X, x)
    assert not any(pc.guided_history)


def test_lam_zero_with_a_corrector_is_not_the_unguided_ode():
    """The OTHER lam=0 control: base ODE plus pure Langevin on p_t still moves the cloud, so the two
    controls are genuinely different objects. Reading the wrong one as 'untilted' is the exact mistake
    CLAUDE.md records for the leg-3.1 stitch."""
    transformer = _tiny_transformer()
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    kw: dict[str, Any] = dict(velocity_fn=velocity_fn, n_steps=3, shift=1.0, seed=23)

    no_corr = flow_guided_pc_sample(reward, 0.0, 2, corrector_steps=0, **kw)
    with_corr = flow_guided_pc_sample(reward, 0.0, 2, corrector_steps=2, **kw)

    assert not torch.equal(no_corr.X, with_corr.X)
    assert all(s.ran for s in with_corr.corrector_history[:-1])
    # lam=0 never computes a reward gradient, so the corrector is score-only and costs no backwards.
    assert with_corr.n_reward_grads == 0
    assert all(math.isnan(v) for s in with_corr.corrector_history for v in s.grad_norm)


# ---------------------------------------------------------------------------------------------------
# Corrector windowing and the 1/t guard
# ---------------------------------------------------------------------------------------------------

def test_corrector_never_runs_at_the_schedules_final_zero_node():
    """s_theta = -(x_t + (1-t)v)/t is undefined at t=0, and the last schedule node is exactly 0, so the
    default floor must exclude it. Without this the run would produce inf/nan silently."""
    res = _run(_tiny_transformer(), lam=1.0, n_steps=4, corrector_steps=2)
    assert res.corrector_history[-1].t == 0.0
    assert not res.corrector_history[-1].ran
    assert torch.isfinite(res.X).all()


def test_corrector_window_gates_exactly_the_in_window_arrival_nodes():
    res = _run(_tiny_transformer(), lam=1.0, n_steps=6, corrector_steps=1,
               corrector_t_min=0.3, corrector_t_max=0.7)
    expected = [0.3 <= s.t <= 0.7 for s in res.corrector_history]
    assert [s.ran for s in res.corrector_history] == expected
    assert any(expected) and not all(expected)


def test_nonpositive_corrector_t_min_raises():
    transformer = _tiny_transformer()
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    with pytest.raises(ValueError, match="corrector_t_min must be > 0"):
        flow_guided_pc_sample(reward, 1.0, 1, velocity_fn=velocity_fn, n_steps=2,
                              corrector_steps=1, corrector_t_min=0.0)


@pytest.mark.parametrize("kw,match", [
    (dict(corrector_steps=-1), "corrector_steps must be >= 0"),
    (dict(snr=0.0), "snr must be > 0"),
])
def test_invalid_arguments_raise(kw: dict[str, Any], match: str):
    transformer = _tiny_transformer()
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    with pytest.raises(ValueError, match=match):
        flow_guided_pc_sample(reward, 1.0, 1, velocity_fn=velocity_fn, n_steps=2, **kw)


def test_unknown_eta_reference_is_rejected():
    """Under pytest, conftest.py's jaxtyping/beartype import hook rejects the bad Literal before the
    function body runs; without the hook (a notebook script run standalone) the explicit ValueError in
    the body is what fires. Both are acceptable, so accept either -- but SOMETHING must reject it."""
    transformer = _tiny_transformer()
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)
    with pytest.raises((ValueError, jaxtyping.TypeCheckError), match="eta_reference"):
        flow_guided_pc_sample(reward, 1.0, 1, velocity_fn=velocity_fn, n_steps=2,
                              eta_reference="prior")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------
# Effort accounting -- the cheapest check that the sampler did what its kwargs said
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("predictor_guided,corrector_steps", [(True, 2), (False, 2), (True, 0)])
def test_velocity_and_reward_call_counts_match_the_predicted_effort(
    predictor_guided: bool, corrector_steps: int
):
    n_steps = 4
    res = _run(_tiny_transformer(), lam=1.0, n_steps=n_steps, corrector_steps=corrector_steps,
               predictor_guided=predictor_guided, corrector_t_min=1e-2)
    n_corr_nodes = sum(s.ran for s in res.corrector_history)

    assert res.n_velocity_evals == n_steps + n_corr_nodes * corrector_steps
    assert res.n_reward_grads == (n_steps if predictor_guided else 0) + n_corr_nodes * corrector_steps
    assert len(res.x_norm_history) == n_steps
    for s in res.corrector_history:
        assert len(s.eta) == len(s.f_hat0) == len(s.rel_displacement) == (corrector_steps if s.ran else 0)


def test_corrector_actually_displaces_the_latents():
    """rel_displacement is the diagnostic that says whether the corrector did anything at all; it must be
    strictly positive and finite wherever the corrector ran."""
    res = _run(_tiny_transformer(), lam=1.0, n_steps=4, corrector_steps=2)
    disps = [v for s in res.corrector_history if s.ran for v in s.rel_displacement]
    assert disps and all(0.0 < v < 10.0 and math.isfinite(v) for v in disps)


# ---------------------------------------------------------------------------------------------------
# Plumbing: determinism, memory isolation, freeze guard, OOM fallback
# ---------------------------------------------------------------------------------------------------

def test_same_seed_gives_identical_x_and_seed_none_seeds_from_zero():
    transformer = _tiny_transformer()
    a = _run(transformer, lam=1.0, seed=31, corrector_steps=2)
    b = _run(transformer, lam=1.0, seed=31, corrector_steps=2)
    assert torch.equal(a.X, b.X)

    c = _run(transformer, lam=1.0, seed=32, corrector_steps=2)
    assert not torch.equal(a.X, c.X)

    none_ = _run(transformer, lam=1.0, seed=None, corrector_steps=2)
    zero = _run(transformer, lam=1.0, seed=0, corrector_steps=2)
    assert torch.equal(none_.X, zero.X)


@pytest.mark.parametrize("exact_jacobian", [False, True])
def test_no_graph_retained_on_the_returned_latents(exact_jacobian: bool):
    """M corrector graphs per ODE step is where a retained graph would compound fastest."""
    res = _run(_tiny_transformer(), lam=1.0, corrector_steps=2, exact_jacobian=exact_jacobian)
    assert not res.X.requires_grad
    assert res.X.grad_fn is None


def test_exact_jacobian_raises_before_building_a_graph_if_unfrozen():
    transformer = _tiny_transformer()
    velocity_fn = _build_velocity_fn(transformer, differentiable=False)  # not frozen
    reward = _build_reward(transformer)
    with pytest.raises(ValueError, match="frozen"):
        flow_guided_pc_sample(reward, 1.0, 2, velocity_fn=velocity_fn, n_steps=2,
                              corrector_steps=1, exact_jacobian=True)


def test_exact_and_approximate_jacobians_both_work_and_differ():
    transformer = _tiny_transformer()
    approx = _run(transformer, lam=1.0, n_steps=3, corrector_steps=2, exact_jacobian=False)
    exact = _run(transformer, lam=1.0, n_steps=3, corrector_steps=2, exact_jacobian=True)
    for res in (approx, exact):
        assert torch.isfinite(res.X).all()
        for s in res.corrector_history:
            assert all(math.isfinite(v) and v > 0 for v in s.grad_norm)
    assert not torch.equal(approx.X, exact.X)


def test_corrector_records_the_oom_fallback_when_forced():
    """Same CPU proxy for a real OOM as test_flow_guided.py's: a Distance whose expected() raises on the
    first grad-requiring call, so _reward_grad's genuine (not force_fallback) branch is what fires. Here
    the fallback must be visible in the CORRECTOR's snapshot, which is a separate call path."""
    transformer = _tiny_transformer()
    transformer.requires_grad_(False)
    velocity_fn = _build_velocity_fn(transformer, differentiable=True)
    reward = _build_reward(transformer)

    real_distance = reward.distance
    state = {"raised": 0}

    class _OOMOnGradCalls(type(real_distance)):
        def expected(self, X, x_refs, weights=None):
            if X.requires_grad and state["raised"] < 2:
                state["raised"] += 1
                raise RuntimeError("CUDA out of memory.")
            return super().expected(X, x_refs, weights)  # type: ignore[reportAttributeAccessIssue]

    reward.distance.__class__ = _OOMOnGradCalls   # swap the class in place; fields are unchanged

    res = flow_guided_pc_sample(reward, 1.0, 2, velocity_fn=velocity_fn, n_steps=2, shift=1.0,
                                corrector_steps=2, g_chunk=1, seed=41)
    assert state["raised"] == 2
    assert any(f for s in res.corrector_history for f in s.oom_fallback)


# ---------------------------------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------------------------------

def test_public_surface_is_wired():
    import creativity_measure

    assert creativity_measure.flow_guided_pc_sample is flow_guided_pc_sample
    assert creativity_measure.FlowGuidedPCResult is FlowGuidedPCResult
    assert creativity_measure.CorrectorSnapshot is CorrectorSnapshot
    assert creativity_measure.velocity_to_score is velocity_to_score
    assert creativity_measure.SNR_SONG_2021 == SNR_SONG_2021
    # flow_guided_common is shared, non-public infra (like smc_common) -- not on the package surface.
    assert not hasattr(creativity_measure, "guided_euler_step")
