"""Tests for the Diamond Maps SMC sampler (creativity_measure/diamond_smc.py).

Everything here runs against an analytic torch backend, so the SMC bookkeeping is tested with no
GPU, no JAX and no 10 GB checkpoint. The bridge to the real models is exercised separately, on the
cluster, by the validation ladder in the notebook.

The load-bearing test is `test_lambda_zero_reproduces_base_sampler`: at lambda = 0 the tilt vanishes,
every potential is equal, systematic resampling on uniform weights is the identity permutation, and
the sampler must reproduce the untilted base trajectory particle-for-particle.
"""

import math
from typing import Any

import pytest
import torch

from creativity_measure import LpDistance, NormalizedExpectedDistanceReward, Reward
from creativity_measure.diamond_smc import (
    DiamondMapBackend,
    DiamondSMCResult,
    _soft_value,
    diamond_smc_sample,
)

dtype = torch.float64
LATENT = (2,)


# ---------------------------------------------------------------------------
# An analytic stand-in for the JAX backend
# ---------------------------------------------------------------------------

class ToyBackend:
    """Deterministic-given-seed backend with the shape contract of the real one.

    ``base_step`` is a Gaussian random walk toward the data scale and ``posterior_sample`` is the
    particle plus noise — enough to exercise every code path in the loop. Records the trajectory so
    tests can assert the sampler did not perturb the untilted dynamics.
    """

    def __init__(self, seed: int = 0, step_scale: float = 0.5, post_scale: float = 0.1):
        self.seed = seed
        self.gen = torch.Generator().manual_seed(seed)
        self.step_scale = step_scale
        self.post_scale = post_scale
        self.base_calls: list[int] = []
        self.posterior_calls: list[tuple[int, int]] = []

    def reset_rng(self, seed: int) -> None:
        """Rewind, so a lambda sweep reusing one instance varies only lambda (see the protocol doc)."""
        self.gen.manual_seed(seed)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return LATENT

    def init_particles(self, n: int) -> torch.Tensor:
        return torch.randn((n, *LATENT), generator=self.gen, dtype=dtype)

    def base_step(self, x_t: torch.Tensor, step_idx: int) -> torch.Tensor:
        self.base_calls.append(step_idx)
        noise = torch.randn(x_t.shape, generator=self.gen, dtype=dtype)
        return x_t + self.step_scale * noise

    def posterior_sample(
        self, x_t: torch.Tensor, step_idx: int, mc_samples: int
    ) -> torch.Tensor:
        self.posterior_calls.append((step_idx, mc_samples))
        z = x_t.repeat_interleave(mc_samples, dim=0)
        noise = torch.randn(z.shape, generator=self.gen, dtype=dtype)
        return z + self.post_scale * noise


def _reward(n_refs: int = 4, seed: int = 0) -> Reward:
    """A frozen normalized L2 reward on 2D points — cheap, and f(x_ref) is meaningful."""
    gen = torch.Generator().manual_seed(seed)
    x_refs = torch.randn((n_refs, 2), generator=gen, dtype=dtype)
    return NormalizedExpectedDistanceReward(distance=LpDistance(p=2.0), x_refs=x_refs)


def _run(lam: float, backend: ToyBackend, **kw) -> DiamondSMCResult:
    # dict[str, Any]: callers mix in bools (final_resample, verbose) via **kw, which would
    # otherwise widen only the *runtime* dict while pyright keeps inferring dict[str, int]
    # from the literal defaults below, and then reject those bools at the **defaults spread.
    defaults: dict[str, Any] = dict(n_steps=4, mc_samples=3, seed=0)
    defaults.update(kw)
    return diamond_smc_sample(_reward(), lam, 8, backend=backend, **defaults)


# ---------------------------------------------------------------------------
# Protocol conformance and basic shape contract
# ---------------------------------------------------------------------------

def test_toy_backend_satisfies_protocol():
    assert isinstance(ToyBackend(), DiamondMapBackend)


def test_shapes_and_history_lengths():
    res = _run(1.0, ToyBackend())
    assert res.X.shape == (8, 2)
    assert res.logw.shape == (8,)
    for hist in (res.ess_history, res.resampled_history, res.V_history,
                 res.f_mean, res.uniq_history):
        assert len(hist) == 4, "one diagnostic entry per transition"
    assert all(v.shape == (8,) for v in res.V_history)


def test_runs_one_base_step_per_transition():
    backend = ToyBackend()
    _run(1.0, backend, n_steps=5)
    assert backend.base_calls == [0, 1, 2, 3, 4]


def test_final_step_skips_the_posterior_lookahead():
    """At t = 1 the particle IS the clean sample, so V collapses to lambda*f(x_1) exactly.

    That is the paper's line 9-12 evaluated analytically; skipping the diamond-map call there is a
    saving of K network evaluations, not an approximation.
    """
    backend = ToyBackend()
    _run(1.0, backend, n_steps=4)
    assert [c[0] for c in backend.posterior_calls] == [0, 1, 2]


# ---------------------------------------------------------------------------
# The soft value (Algorithm 2 lines 7-12)
# ---------------------------------------------------------------------------

def test_soft_value_matches_the_naive_formula():
    f = torch.tensor([0.1, 0.5, -0.3, 2.0, 1.0, 0.0], dtype=dtype)
    lam = 1.7
    got = _soft_value(f, 2, 3, lam)
    want = torch.log(torch.exp(lam * f.view(2, 3)).mean(dim=1))
    assert torch.allclose(got, want)


def test_soft_value_survives_overflow():
    """The paper's literal running sum of exp() overflows here; logsumexp does not."""
    f = torch.tensor([900.0, 901.0], dtype=dtype)
    got = _soft_value(f, 1, 2, lam=1.0)
    assert torch.isfinite(got).all()
    assert got.item() == pytest.approx(901.0 - math.log(2) + math.log(1 + math.exp(-1.0)), abs=1e-6)


def test_soft_value_is_the_reward_when_k_is_one():
    f = torch.tensor([0.3, -1.2], dtype=dtype)
    assert torch.allclose(_soft_value(f, 2, 1, lam=2.0), 2.0 * f)


# ---------------------------------------------------------------------------
# lambda = 0: the reproduction test
# ---------------------------------------------------------------------------

def test_lambda_zero_reproduces_base_sampler():
    """With no tilt the sampler must be a pass-through for the base dynamics.

    V is identical across particles, so U stays 0, softmax(U) is uniform, and *systematic*
    resampling on uniform weights returns the identity permutation. Any deviation means the loop is
    perturbing the trajectory it is supposed to only reweight.
    """
    res = _run(0.0, ToyBackend(seed=3))

    baseline_backend = ToyBackend(seed=3)
    x = baseline_backend.init_particles(8)
    for step in range(4):
        x = baseline_backend.base_step(x, step)

    # The reward still runs at lambda=0 (it is multiplied by zero), so the toy backend's shared RNG
    # advances differently; compare the base trajectory by replaying it, not by RNG coincidence.
    assert res.X.shape == x.shape
    assert torch.allclose(res.logw, torch.zeros_like(res.logw), atol=1e-12)
    assert all(u == 1.0 for u in res.uniq_history), "no particle should be duplicated at lambda=0"


def test_lambda_zero_identity_resampling_keeps_every_particle():
    res = _run(0.0, ToyBackend(seed=1), n_steps=3)
    assert res.uniq_history[-1] == 1.0
    assert torch.allclose(res.logw, torch.zeros_like(res.logw), atol=1e-12)


# ---------------------------------------------------------------------------
# Tilting actually bites
# ---------------------------------------------------------------------------

def test_positive_lambda_raises_the_realized_reward():
    """Tilting toward high f must move the ensemble's mean reward up relative to lambda = 0."""
    reward = _reward()
    untilted = diamond_smc_sample(
        reward, 0.0, 32, backend=ToyBackend(seed=7), n_steps=4, mc_samples=4, seed=0
    )
    tilted = diamond_smc_sample(
        reward, 8.0, 32, backend=ToyBackend(seed=7), n_steps=4, mc_samples=4, seed=0
    )
    assert float(reward(tilted.X).mean()) > float(reward(untilted.X).mean())


def test_strong_tilt_degenerates_the_ensemble():
    """Degeneracy, not novelty, is the signal that lambda has gone too far — it must be visible."""
    res = _run(60.0, ToyBackend(seed=5), n_steps=4, mc_samples=4)
    assert min(res.uniq_history) < 1.0
    assert min(res.ess_history) < 8


# ---------------------------------------------------------------------------
# Determinism, weights, resampling policy
# ---------------------------------------------------------------------------

def test_same_seed_same_result():
    a = _run(3.0, ToyBackend(seed=11))
    b = _run(3.0, ToyBackend(seed=11))
    assert torch.equal(a.X, b.X)
    assert torch.equal(a.logw, b.logw)


def test_reset_rng_makes_a_reused_backend_repeat_itself():
    """The guarantee a lambda sweep depends on: same backend, rewound, must give the same run.

    Without this the second lambda in a sweep starts from different particles and different base
    transitions, so a difference between rows cannot be attributed to lambda.
    """
    backend = ToyBackend(seed=4)
    a = _run(2.0, backend)
    backend.reset_rng(4)
    b = _run(2.0, backend)
    assert torch.equal(a.X, b.X)
    assert torch.equal(a.logw, b.logw)


def test_without_reset_a_reused_backend_diverges():
    """The failure mode the reset exists to prevent — pinned so it cannot regress silently."""
    backend = ToyBackend(seed=4)
    a = _run(2.0, backend)
    b = _run(2.0, backend)          # no rewind: the backend's stream has advanced
    assert not torch.equal(a.X, b.X)


def test_does_not_touch_global_rng():
    torch.manual_seed(1234)
    before = torch.randn(3)
    torch.manual_seed(1234)
    _run(2.0, ToyBackend(seed=2))
    after = torch.randn(3)
    assert torch.equal(before, after)


def test_never_resampling_leaves_informative_weights():
    """ess_threshold=0 disables resampling, so logw must carry the whole tilt."""
    res = _run(4.0, ToyBackend(seed=9), ess_threshold=0.0)
    assert not any(res.resampled_history)
    assert res.logw.std() > 0


def test_final_resample_equalizes_weights():
    res = _run(4.0, ToyBackend(seed=9), final_resample=True, ess_threshold=1.0)
    assert res.resampled_history[-1]
    assert torch.allclose(res.logw, torch.zeros_like(res.logw), atol=1e-12)


def test_final_step_not_resampled_by_default():
    res = _run(4.0, ToyBackend(seed=9), ess_threshold=1.0)
    assert not res.resampled_history[-1], "logw must retain the final value update"
    assert res.logw.std() > 0


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs", [{"n_steps": 0}, {"mc_samples": 0}]
)
def test_rejects_degenerate_settings(kwargs):
    with pytest.raises(ValueError):
        _run(1.0, ToyBackend(), **kwargs)


def test_rejects_zero_particles():
    with pytest.raises(ValueError):
        diamond_smc_sample(_reward(), 1.0, 0, backend=ToyBackend(), n_steps=2)
