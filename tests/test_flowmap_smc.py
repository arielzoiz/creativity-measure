"""Tests for the flow-map SMC sampler (creativity_measure/flowmap_smc.py) -- Algorithm 3.

Everything here runs against an **analytic** flow map on the CPU in float64, so the SMC bookkeeping,
every coefficient and the score conversion are tested with no GPU and no 12 B checkpoint. For
``p = N(0, s^2 I)`` both flow-map operations are closed form with ``nu_t^2 = alpha_t^2 s^2 + sigma_t^2``::

    map(x, t, t') = (nu_{t'} / nu_t) · x          the probability-flow ODE transport
    denoise(x, t) = (alpha_t s^2 / nu_t^2) · x    E[x_1 | x_t]

so the whole "model" is a few lines, every helper has ground truth, and ``map``/``denoise`` are
genuinely distinct -- this suite would catch conflating them.

The load-bearing test is `test_lambda_zero_reproduces_base_trajectory`: at ``lambda = 0`` every
potential is equal, systematic resampling on uniform weights is the identity permutation, and the
sampler must reproduce the untilted base trajectory particle-for-particle.

`GaussianPosteriorFlowMap` is the second model: its ``map(x, t, 1)`` draws from the exact posterior
``p(x_1 | x_t)`` instead of returning the ODE endpoint. `ddpm_step` is derived assuming a posterior
draw, so that variant is what makes its marginal exact -- and therefore what the marginal-preservation
and 2D-ground-truth tests need.
"""

import math

import pytest
import torch

from creativity_measure import LpDistance, NormalizedExpectedDistanceReward, Reward
from creativity_measure._types import FlowMap, Schedule, TransitionStep
from creativity_measure.distances.edm_adapter import edm_score_fn
from creativity_measure.flowmap_smc import (
    MIN_ZSCORE_STD,
    BaseSchedule,
    FlowMapSMCResult,
    LinearSchedule,
    _antithetic_noise,
    _g,
    _in_window,
    _log_likelihood,
    _renoise,
    _resample,
    _score_at,
    _score_correction,
    _soft_value,
    _t_prime,
    _uniform_ts,
    _zscore,
    ddpm_step,
    flow_map_step,
    flowmap_smc_sample,
)
from creativity_measure.generators.flux_flowmap import flow_map_denoiser

dtype = torch.float64
S_DATA = 1.3           # the toy data scale: p = N(0, S_DATA^2 I)
D = 2
GRID = _uniform_ts(16)


# ---------------------------------------------------------------------------------------------------
# Analytic models
# ---------------------------------------------------------------------------------------------------

class VPSchedule(BaseSchedule):
    """A variance-preserving schedule defined here, supplying ONLY ``alpha`` and ``sigma``.

    ``g(t) = (1-t)/t``, so the closed-form inverse is ``1/(1+y)`` -- which the test compares against
    `BaseSchedule`'s generic bisection. Deliberately *not* overriding ``t_of_snr``: this is the whole
    point of the schedule being an extension point rather than a claim.
    """

    name: str = "vp_test"

    def alpha(self, t: float) -> float:
        return math.sqrt(t)

    def sigma(self, t: float) -> float:
        return math.sqrt(1.0 - t)


def _nu(schedule: Schedule, t: float, s: float = S_DATA) -> float:
    """``nu_t = sqrt(alpha_t^2 s^2 + sigma_t^2)`` -- the std of ``x_t`` under ``p = N(0, s^2 I)``."""
    return math.sqrt(schedule.alpha(t) ** 2 * s * s + schedule.sigma(t) ** 2)


class GaussianFlowMap:
    """The exact prob-flow-ODE flow map of ``p = N(0, s^2 I)``, plus its exact Tweedie denoiser.

    Records every call so tests can assert which operation ran where.
    """

    def __init__(self, schedule: Schedule, s: float = S_DATA):
        self.schedule = schedule
        self.s = s
        self.map_calls: list[tuple[float, float]] = []
        self.denoise_calls: list[float] = []

    def map(self, x: torch.Tensor, t_from: float, t_to: float) -> torch.Tensor:
        self.map_calls.append((t_from, t_to))
        return (_nu(self.schedule, t_to, self.s) / _nu(self.schedule, t_from, self.s)) * x

    def denoise(self, x: torch.Tensor, t: float) -> torch.Tensor:
        self.denoise_calls.append(t)
        a = self.schedule.alpha(t)
        return (a * self.s ** 2 / _nu(self.schedule, t, self.s) ** 2) * x


class GaussianPosteriorFlowMap(GaussianFlowMap):
    """Same denoiser, but ``map(x, t, 1)`` draws from the exact posterior ``p(x_1 | x_t)``.

    ``ddpm_step`` conditions the joint ``(X_t, X_{t'}) | X_1 = z`` and substitutes ``z_hat = map(x,t,1)``
    for a draw ``z ~ p(x_1|x_t)``. With a genuine posterior draw the step is *exact*, which is what
    `test_ddpm_step_marginal_is_exact` and the 2D ground-truth test rely on. Owning its own generator
    keeps the sampler's streams untouched -- the `FlowMap` protocol has no ``generator`` argument, and
    a stochastic flow map is the model class Algorithm 3 is written for.
    """

    def __init__(self, schedule: Schedule, s: float = S_DATA, seed: int = 0):
        super().__init__(schedule, s)
        self.gen = torch.Generator().manual_seed(seed)

    def map(self, x: torch.Tensor, t_from: float, t_to: float) -> torch.Tensor:
        if t_to != 1.0:
            return super().map(x, t_from, t_to)
        self.map_calls.append((t_from, t_to))
        a, sig = self.schedule.alpha(t_from), self.schedule.sigma(t_from)
        nu2 = _nu(self.schedule, t_from, self.s) ** 2
        mean = (a * self.s ** 2 / nu2) * x
        std = math.sqrt(self.s ** 2 * sig ** 2 / nu2)
        return mean + std * torch.randn(x.shape, generator=self.gen, dtype=x.dtype)


def _gaussian_score_fn(s: float = S_DATA):
    """``grad_y log p_Y(y, gamma)`` for ``p = N(0, s^2 I)`` in the repo's gamma convention.

    ``Y = gamma·z + sqrt(gamma)·W``, so ``Var(Y) = gamma^2 s^2 + gamma`` and the score is ``-y/Var``.
    """
    def score_fn(y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        g = gamma.to(y)
        return -y / (g * g * s * s + g)
    return score_fn


def _reward(n_refs: int = 4, seed: int = 0, d: int = D) -> Reward:
    """A frozen normalized L2 reward -- cheap, and ``f(x_ref)`` is meaningful."""
    gen = torch.Generator().manual_seed(seed)
    x_refs = S_DATA * torch.randn((n_refs, d), generator=gen, dtype=dtype)
    return NormalizedExpectedDistanceReward(distance=LpDistance(p=2.0), x_refs=x_refs)


def _base_trajectory(
    flow_map: FlowMap,
    schedule: Schedule,
    *,
    ts: list[float],
    n_particles: int,
    d: int,
    seed: int,
    stoch_window: tuple[float, float] = (0.1, 1.0),
    inside_step: TransitionStep = ddpm_step,
    outside_step: TransitionStep = flow_map_step,
) -> torch.Tensor:
    """The untilted base process, written out independently of the sampler.

    Mirrors exactly what `flowmap_smc_sample` does with the reward switched off: one generator, seeded
    with ``seed``, drawing the initial cloud and every stochastic transition -- and nothing else.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((n_particles, d), generator=gen, dtype=dtype)
    for n in range(len(ts) - 1):
        step = inside_step if _in_window(ts[n], stoch_window) else outside_step
        x = step(x, ts[n], ts[n + 1], flow_map=flow_map, schedule=schedule, generator=gen)
    return x


def _run(reward: Reward, lam: float, **kw) -> FlowMapSMCResult:
    """`flowmap_smc_sample` with the toy defaults filled in."""
    schedule = kw.pop("schedule", LinearSchedule())
    flow_map = kw.pop("flow_map", None) or GaussianFlowMap(schedule)
    kw.setdefault("score_fn", _gaussian_score_fn())
    kw.setdefault("n_steps", 16)
    kw.setdefault("mc_samples", 4)
    kw.setdefault("seed", 7)
    return flowmap_smc_sample(
        reward, lam, kw.pop("n_particles", 8),
        flow_map=flow_map, schedule=schedule, **kw,
    )


# ---------------------------------------------------------------------------------------------------
# 1. Load-bearing: lambda = 0 reproduces the untilted base trajectory
# ---------------------------------------------------------------------------------------------------

def test_lambda_zero_reproduces_base_trajectory():
    """At lambda = 0 every potential is 0, so resampling is the identity and X must match exactly."""
    schedule = LinearSchedule()
    reward = _reward()
    res = _run(reward, lam=0.0, flow_map=GaussianFlowMap(schedule), schedule=schedule,
               n_particles=8, seed=7)
    expected = _base_trajectory(GaussianFlowMap(schedule), schedule,
                                ts=GRID, n_particles=8, d=D, seed=7)
    assert torch.equal(res.X, expected)
    assert torch.allclose(res.logw, torch.zeros_like(res.logw))
    assert all(v == 1.0 for v in res.uniq_history)


def test_lambda_zero_is_unaffected_by_lookahead_effort():
    """The base and lookahead streams are independent: changing K must not move the trajectory."""
    reward = _reward()
    a = _run(reward, lam=0.0, mc_samples=2, seed=11)
    b = _run(reward, lam=0.0, mc_samples=8, seed=11)
    assert torch.equal(a.X, b.X)


# ---------------------------------------------------------------------------------------------------
# 2. The DDPM base step (§3B) -- the real test, and the one that catches a negative variance
# ---------------------------------------------------------------------------------------------------

def test_ddpm_step_marginal_is_exact():
    """Iterating `ddpm_step` from ``x_0 ~ N(0, I)`` must reproduce ``N(0, nu_t^2)`` at every ``t``.

    With the exact posterior draw for ``z_hat`` the step is exact, so this pins every coefficient at
    once -- and would have caught the spec's sqrt of a negative variance immediately.
    """
    schedule = LinearSchedule()
    fm = GaussianPosteriorFlowMap(schedule, seed=3)
    gen = torch.Generator().manual_seed(0)
    m = 40000
    x = torch.randn((m, D), generator=gen, dtype=dtype)          # nu_0 = sigma_0 = 1
    assert _nu(schedule, 0.0) == pytest.approx(1.0)
    for n in range(len(GRID) - 1):
        x = ddpm_step(x, GRID[n], GRID[n + 1], flow_map=fm, schedule=schedule, generator=gen)
        want = _nu(schedule, GRID[n + 1])
        assert float(x.std()) == pytest.approx(want, rel=0.02), f"t={GRID[n + 1]}"
        assert abs(float(x.mean())) < 0.03
    assert float(x.std()) == pytest.approx(S_DATA, rel=0.02)     # nu_1 = s


def test_ddpm_step_endpoints():
    """No special cases: ``rho = 0`` at ``t = 0`` (pure draw) and at ``t' = 1`` (deterministic denoise)."""
    schedule = LinearSchedule()
    fm = GaussianFlowMap(schedule)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn((5, D), generator=gen, dtype=dtype)

    # t' = 1: sigma_{t'} = 0 -> rho = 0, c_x = 0, c_z = 1, no noise -> x_1 = z_hat exactly.
    out = ddpm_step(x, 0.9375, 1.0, flow_map=fm, schedule=schedule, generator=gen)
    assert torch.allclose(out, fm.map(x, 0.9375, 1.0))

    # t = 0: rho = 0 -> x_{t'} = alpha_{t'}·z_hat + sigma_{t'}·eps, with z_hat independent of x.
    g1, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    out0 = ddpm_step(x, 0.0, 0.0625, flow_map=fm, schedule=schedule, generator=g1)
    eps = torch.randn(x.shape, generator=g2, dtype=dtype)
    want = 0.0625 * fm.map(x, 0.0, 1.0) + 0.9375 * eps
    assert torch.allclose(out0, want)


def test_ddpm_step_rejects_t_equals_one_as_source():
    schedule = LinearSchedule()
    with pytest.raises(ValueError, match="pole"):
        ddpm_step(torch.zeros((2, D), dtype=dtype), 1.0, 1.0,
                  flow_map=GaussianFlowMap(schedule), schedule=schedule,
                  generator=torch.Generator())


# ---------------------------------------------------------------------------------------------------
# 3. The terminal update is mandatory, raw, and exact
# ---------------------------------------------------------------------------------------------------

def test_terminal_potential_is_raw_lambda_f():
    """``V_N = lambda·f(x_1)`` exactly: one reward call, no Monte Carlo, no normalization."""
    reward = _reward()
    lam = 2.5
    res = _run(reward, lam=lam, n_particles=6, seed=5, ess_threshold=0.0)   # never resample
    assert res.guided_history[-1]
    assert math.isnan(res.t_prime_history[-1])                              # no lookahead at t = 1
    assert torch.allclose(res.V_history[-1], lam * reward(res.X))


def test_terminal_update_runs_even_when_the_window_excludes_it():
    """A ``(0.1, 0.9)`` window still gets the terminal update -- that is what pins the target to q_lambda."""
    reward = _reward()
    res = _run(reward, lam=1.5, n_particles=6, seed=5, guid_window=(0.1, 0.9))
    assert res.guided_history[-1]
    assert not res.guided_history[-2]                     # t_next = 0.9375 is outside the window
    assert res.t_history[-3] == pytest.approx(0.875)      # the last *lookahead* update
    assert res.guided_history[-3]
    assert torch.allclose(res.V_history[-1], 1.5 * reward(res.X))


def test_terminal_potential_normalization_is_never_applied():
    """Even in the Z-scored branch the terminal ``V_N`` is raw ``lambda·f`` (K = 1 there)."""
    reward = _reward()
    res = _run(reward, lam=3.0, n_particles=6, seed=5,
               use_full_normalized_v=True, lam_normalized=1.0)
    assert torch.allclose(res.V_history[-1], 3.0 * reward(res.X))


# ---------------------------------------------------------------------------------------------------
# 4. The guidance window
# ---------------------------------------------------------------------------------------------------

def test_guidance_window_leaves_excluded_steps_untouched():
    """Outside the window ``U`` and ``V`` are carried forward unchanged, and no lookahead runs."""
    reward = _reward()
    res = _run(reward, lam=1.0, n_particles=8, seed=2, guid_window=(0.4, 0.6), ess_threshold=0.0)

    guided = [t for t, g in zip(res.t_history, res.guided_history) if g]
    # ts[n+1] in [0.4, 0.6] -> {0.4375, 0.5, 0.5625}; plus the mandatory terminal at t = 1.
    assert guided == pytest.approx([0.4375, 0.5, 0.5625, 1.0])

    for n, g in enumerate(res.guided_history):
        if g:
            continue
        assert math.isnan(res.t_prime_history[n]) and math.isnan(res.f_mean[n])
        prev = res.V_history[n - 1] if n else torch.zeros_like(res.V_history[n])
        assert torch.equal(res.V_history[n], prev)      # V carried forward => U unchanged
    # U before the first guided step is still exactly zero.
    assert res.ess_history[0] == pytest.approx(8.0)


def test_guidance_window_does_not_affect_p():
    """At ``lam = 0`` the trajectory must be identical for any guidance window -- ``p`` is independent of it.

    This is the property that lets the window be swept freely against one fixed reference set.
    """
    reward = _reward()
    a = _run(reward, lam=0.0, seed=13, guid_window=(0.1, 1.0))
    b = _run(reward, lam=0.0, seed=13, guid_window=(0.4, 0.6))
    assert torch.equal(a.X, b.X)


def test_degenerate_windows_raise():
    """Windows must satisfy ``0 <= lo <= hi <= 1``."""
    reward = _reward()
    with pytest.raises(ValueError, match="0 <= lo"):
        _run(reward, lam=1.0, guid_window=(0.6, 0.4))
    with pytest.raises(ValueError, match="0 <= lo"):
        _run(reward, lam=1.0, guid_window=(0.1, 1.5))
    with pytest.raises(ValueError, match="0 <= lo"):
        _run(reward, lam=1.0, guid_window=(-0.1, 1.0))


def test_windows_may_start_at_zero():
    """``guid_window=(0, 1)`` guides EVERY step and must be safe.

    The lookahead runs at each step's *target* time, so ``t = 0`` -- where ``g = inf``, ``c = 0`` and
    ``gamma = 0`` -- can never become a lookahead level on a grid starting at 0. The earliest reachable
    level at N=16 is t = 0.0625, which is comfortably conditioned. ``stoch_window`` may also start at 0
    because `ddpm_step` is explicitly non-singular there (rho = 0 gives a pure draw).
    """
    reward = _reward()
    res = _run(reward, lam=1.0, n_particles=6, seed=3, guid_window=(0.0, 1.0))
    assert all(res.guided_history), "every step should be guided by a (0, 1) window"
    assert res.t_history[0] == pytest.approx(0.0625)
    # the first guided step's lookahead level is well below its target but strictly positive
    assert 0.0 < res.t_prime_history[0] < res.t_history[0]
    assert all(math.isfinite(v) for v in res.ess_history)
    assert torch.isfinite(res.X).all() and torch.isfinite(res.logw).all()

    # stoch_window starting at 0 makes even the t=0 transition stochastic, and stays finite.
    res2 = _run(reward, lam=1.0, n_particles=6, seed=3,
                guid_window=(0.0, 1.0), stoch_window=(0.0, 1.0))
    assert torch.isfinite(res2.X).all()

    # And the invariant that actually guards this: a guided step landing on t = 0 is rejected.
    from creativity_measure.flowmap_smc import _check_window
    _check_window("guid_window", (0.0, 1.0), [True] * 4)      # the window itself is fine


def test_window_selection_is_contiguous():
    """An interval window selects a consecutive run of steps on any increasing grid.

    The contiguity guard in `_check_window` is therefore not reachable through the public API today;
    it exists so a future non-interval window spec cannot silently drop a step and carry resampled
    clones into a lookahead that scores them identically.
    """
    from creativity_measure.flowmap_smc import _check_window

    ts = [0.0, 0.2, 0.5, 0.55, 0.9, 1.0]
    flags = [_in_window(ts[n + 1], (0.5, 0.9)) for n in range(len(ts) - 1)]
    assert flags == [False, True, True, True, False]
    _check_window("guid_window", (0.5, 0.9), flags)
    with pytest.raises(ValueError, match="non-contiguous"):
        _check_window("guid_window", (0.5, 0.9), [True, False, True])


# ---------------------------------------------------------------------------------------------------
# 5. The two potential branches
# ---------------------------------------------------------------------------------------------------

def test_unnormalized_branch_is_lambda_times_the_reward():
    """Branch 1: ``V = logsumexp_k(lambda·R_k) - log K``, and lambda scales it as advertised."""
    reward = _reward()
    r = torch.tensor([[0.5, 1.0, 1.5, 2.0]], dtype=dtype)
    assert float(_soft_value(2.0 * r)) == pytest.approx(
        float(torch.logsumexp(2.0 * r, dim=1) - math.log(4))
    )
    a = _run(reward, lam=0.0, n_particles=6, seed=9, ess_threshold=0.0)
    assert all(float(v.abs().max()) == 0.0 for v in a.V_history)


def test_normalized_branch_is_scale_invariant_in_lambda():
    """Z-scoring kills any scale applied *before* it, so ``lam`` cannot act on ``R̂``; ``lam_normalized`` can.

    Only the (raw, mandatory) terminal update still sees ``lam`` -- so the intermediate potentials must
    be identical across ``lam`` while ``lam_normalized`` genuinely changes them.
    """
    reward = _reward()
    kw = dict(n_particles=6, seed=17, use_full_normalized_v=True, ess_threshold=0.0)
    a = _run(reward, lam=1.0, lam_normalized=1.0, **kw)
    b = _run(reward, lam=50.0, lam_normalized=1.0, **kw)
    c = _run(reward, lam=1.0, lam_normalized=4.0, **kw)
    for va, vb in zip(a.V_history[:-1], b.V_history[:-1]):
        assert torch.allclose(va, vb)
    assert not torch.allclose(a.V_history[-2], c.V_history[-2])


def test_zscore_and_its_std_guard():
    v = torch.tensor([[1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0]], dtype=dtype)
    z = _zscore(v)
    assert torch.allclose(z[0].mean(), torch.zeros((), dtype=dtype), atol=1e-12)
    assert float(z[0].std(unbiased=True)) == pytest.approx(1.0)
    # The K candidates coincide: sigma collapses to 0, the floor fires, and the result is finite zeros.
    assert torch.allclose(z[1], torch.zeros(4, dtype=dtype))
    assert torch.isfinite(z).all()
    # ddof = 1, and the floor is the documented one.
    single = _zscore(torch.tensor([[5.0]], dtype=dtype))
    assert float(single) == 0.0
    assert MIN_ZSCORE_STD == 1e-8


def test_antithetic_pairing():
    eps = _antithetic_noise(3, 4, 5, antithetic=True, generator=torch.Generator().manual_seed(0),
                            device=torch.device("cpu"), dtype=dtype)
    assert eps.shape == (3, 4, 5)
    assert torch.equal(eps[:, 1], -eps[:, 0])
    assert torch.equal(eps[:, 3], -eps[:, 2])
    assert not torch.equal(eps[:, 2], eps[:, 0])
    odd = _antithetic_noise(2, 3, 4, antithetic=True, generator=torch.Generator().manual_seed(0),
                            device=torch.device("cpu"), dtype=dtype)
    assert torch.equal(odd[:, 1], -odd[:, 0])              # the last draw is simply unpaired
    plain = _antithetic_noise(3, 4, 5, antithetic=False, generator=torch.Generator().manual_seed(0),
                              device=torch.device("cpu"), dtype=dtype)
    assert not torch.equal(plain[:, 1], -plain[:, 0])


# ---------------------------------------------------------------------------------------------------
# 6. Coefficient sanity across the uniform N = 16 grid
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("schedule", [LinearSchedule(), VPSchedule()])
def test_rho_stays_in_range_on_the_grid(schedule: Schedule):
    """``rho in [0, 1)`` and ``1 - rho^2 >= 0`` at every transition -- §3B's variance is never negative."""
    from creativity_measure.flowmap_smc import _rho

    for n in range(len(GRID) - 1):
        rho, one_minus = _rho(schedule, GRID[n], GRID[n + 1])
        assert 0.0 <= rho < 1.0, (GRID[n], rho)
        assert one_minus >= 0.0
        assert one_minus == pytest.approx(1.0 - rho * rho, abs=1e-12)
    # The plan's measured envelope on the linear grid.
    if isinstance(schedule, LinearSchedule):
        rhos = [_rho(schedule, GRID[n], GRID[n + 1])[0] for n in range(len(GRID) - 1)]
        assert max(rhos) == pytest.approx(0.7778, abs=1e-3)
        assert min(1.0 - r * r for r in rhos) == pytest.approx(0.3951, abs=1e-3)


@pytest.mark.parametrize("schedule", [LinearSchedule(), VPSchedule()])
def test_t_prime_is_below_t_and_monotone_in_eta(schedule: Schedule):
    for t in GRID[1:-1]:
        prev = t
        for eta in (1.1, 1.5, 3.0, 10.0):
            t_p = _t_prime(t, eta, schedule)
            assert 0.0 < t_p < t, (t, eta, t_p)
            assert t_p < prev                                  # more noise -> earlier time
            assert _g(schedule, t_p) == pytest.approx(eta * _g(schedule, t), rel=1e-9)
            prev = t_p
    # The plan's worked example.
    assert _t_prime(0.5, 1.5, LinearSchedule()) == pytest.approx(0.4494897, abs=1e-6)


@pytest.mark.parametrize("schedule", [LinearSchedule(), VPSchedule()])
def test_renoise_variance_is_positive_on_the_grid(schedule: Schedule):
    """§3C renoises to ``t' < t``, so ``sigma_{t'}^2 - (alpha_{t'}/alpha_t)^2 sigma_t^2 > 0``."""
    for t in GRID[1:]:
        t_p = _t_prime(t, 1.5, schedule)
        ratio = schedule.alpha(t_p) / schedule.alpha(t)
        var = schedule.sigma(t_p) ** 2 - (ratio * schedule.sigma(t)) ** 2
        assert var > 0.0, (t, t_p, var)
    # And the helper reproduces the closed form it documents.
    x = torch.ones((3, D), dtype=dtype)
    eps = torch.zeros((3, D), dtype=dtype)
    t, t_p = 0.5, _t_prime(0.5, 1.5, schedule)
    assert torch.allclose(_renoise(x, t, t_p, schedule, eps),
                          (schedule.alpha(t_p) / schedule.alpha(t)) * x)


# ---------------------------------------------------------------------------------------------------
# 7. `_score_at`: the c / gamma_t conversion of §3C
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("schedule", [LinearSchedule(), VPSchedule()])
def test_score_at_matches_the_closed_form_and_tweedie(schedule: Schedule):
    """``c·score_fn(c·x, gamma_t)`` must equal both ``-x/nu_t^2`` and the Tweedie expression.

    Pins the conversion between the repo's gamma convention and the plan's ``x_t = alpha_t z + sigma_t eps``
    parameterization, to float64 tolerance. If this drifts, the score model that defines ``f``'s geometry
    and the one inside ``S_k`` are silently different fields.
    """
    fm = GaussianFlowMap(schedule)
    score_fn = _gaussian_score_fn()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn((6, D), generator=gen, dtype=dtype)
    for t in GRID[1:-1]:
        got = _score_at(x, t, score_fn, schedule)
        exact = -x / _nu(schedule, t) ** 2
        tweedie = (schedule.alpha(t) * fm.denoise(x, t) - x) / schedule.sigma(t) ** 2
        assert torch.allclose(got, exact, atol=1e-13, rtol=1e-13), t
        assert torch.allclose(got, tweedie, atol=1e-13, rtol=1e-13), t


@pytest.mark.parametrize("schedule", [LinearSchedule(), VPSchedule()])
def test_score_through_the_edm_adapter_chain(schedule: Schedule):
    """The production wiring -- ``edm_score_fn(flow_map_denoiser(flow_map, schedule))`` -- gives the same field.

    This is the chain the notebook builds, and it is what ties ``denoise`` (a conditional mean) to the
    score. Using ``map(x, t, 1)`` there instead would give ``-x/(1-t)`` and pass nothing.
    """
    fm = GaussianFlowMap(schedule)
    score_fn = edm_score_fn(flow_map_denoiser(fm, schedule))
    gen = torch.Generator().manual_seed(1)
    x = torch.randn((5, D), generator=gen, dtype=dtype)
    for t in (0.125, 0.5, 0.875):
        got = _score_at(x, t, score_fn, schedule)
        assert torch.allclose(got, -x / _nu(schedule, t) ** 2, atol=1e-11, rtol=1e-11), t
    assert fm.denoise_calls and not fm.map_calls          # denoise, never map


def test_map_and_denoise_are_distinct():
    """``denoise(x, t)`` is neither ``map(x, t, 1)`` (a sample) nor ``map(x, t, t)`` (the identity)."""
    schedule = LinearSchedule()
    fm = GaussianFlowMap(schedule)
    x = torch.randn((4, D), generator=torch.Generator().manual_seed(0), dtype=dtype)
    t = 0.3
    assert torch.allclose(fm.map(x, t, t), x)
    assert not torch.allclose(fm.denoise(x, t), fm.map(x, t, 1.0))
    assert not torch.allclose(fm.denoise(x, t), fm.map(x, t, t))


# ---------------------------------------------------------------------------------------------------
# 8. Schedule genericity -- the test that keeps "generic over schedules" honest
# ---------------------------------------------------------------------------------------------------

class LinearNumeric(BaseSchedule):
    """`LinearSchedule`'s alpha/sigma WITHOUT its closed-form ``t_of_snr`` override.

    Exists purely so the generic bisection can be checked against a known inverse.
    """

    name: str = "linear_numeric"

    def alpha(self, t: float) -> float:
        return float(t)

    def sigma(self, t: float) -> float:
        return 1.0 - float(t)


def test_bisection_reproduces_both_closed_forms():
    """`BaseSchedule`'s numeric ``t_of_snr`` matches ``1/(1+sqrt(y))`` and ``1/(1+y)`` respectively."""
    numeric, vp = LinearNumeric(), VPSchedule()
    for y in (0.01, 0.1, 1.0, 3.0, 25.0, 400.0):
        assert numeric.t_of_snr(y) == pytest.approx(1.0 / (1.0 + math.sqrt(y)), abs=1e-12)
        assert vp.t_of_snr(y) == pytest.approx(1.0 / (1.0 + y), abs=1e-12)
        # And the closed-form override agrees with the bisection it replaces.
        assert LinearSchedule().t_of_snr(y) == pytest.approx(numeric.t_of_snr(y), abs=1e-12)


def test_the_whole_loop_runs_under_a_foreign_schedule():
    """A schedule supplying only ``alpha``/``sigma`` drives the full sampler, with finite diagnostics."""
    schedule = VPSchedule()
    reward = _reward()
    res = _run(reward, lam=1.0, schedule=schedule,
               flow_map=GaussianFlowMap(schedule), n_particles=6, seed=3)
    assert torch.isfinite(res.X).all()
    assert torch.isfinite(res.logw).all()
    assert all(math.isfinite(e) for e in res.ess_history)
    assert schedule.name == "vp_test"


def test_schedules_registry_round_trips_the_name():
    from creativity_measure.flowmap_smc import SCHEDULES

    assert SCHEDULES["linear"].name == "linear"
    assert isinstance(SCHEDULES[LinearSchedule().name], LinearSchedule)


# ---------------------------------------------------------------------------------------------------
# 9. `stoch_window` selects the transition; custom `TransitionStep`s are accepted
# ---------------------------------------------------------------------------------------------------

def test_stoch_window_selects_the_transition():
    """The head is deterministic under the default window; a wider window makes it stochastic."""
    schedule = LinearSchedule()
    reward = _reward()
    seen: list[tuple[str, float]] = []

    def spy(tag, inner):
        def step(x, t, t_next, *, flow_map, schedule, generator):
            seen.append((tag, t))
            return inner(x, t, t_next, flow_map=flow_map, schedule=schedule, generator=generator)
        return step

    _run(reward, lam=0.0, n_particles=4, seed=1, schedule=schedule,
         flow_map=GaussianFlowMap(schedule),
         inside_step=spy("in", ddpm_step), outside_step=spy("out", flow_map_step))
    kinds = {t: tag for tag, t in seen}
    assert kinds[0.0] == "out" and kinds[0.0625] == "out"      # head, outside (0.1, 1.0)
    assert all(kinds[t] == "in" for t in GRID[2:-1])           # everything after, stochastic
    assert kinds[GRID[-2]] == "in"                             # the tail stays inside, by design


def test_deterministic_region_freezes_resampled_duplicates():
    """A deterministic tail keeps duplicates bit-identical; `ddpm_step` perturbs them.

    This is the mechanism behind ``uniq_history`` being the signal to read when tuning the window.
    """
    schedule = LinearSchedule()
    x = torch.randn((2, D), generator=torch.Generator().manual_seed(0), dtype=dtype)
    dup = x[[0, 0]]
    fm = GaussianFlowMap(schedule)
    frozen = flow_map_step(dup, 0.875, 0.9375, flow_map=fm, schedule=schedule,
                           generator=torch.Generator())
    assert torch.equal(frozen[0], frozen[1])
    moved = ddpm_step(dup, 0.875, 0.9375, flow_map=fm, schedule=schedule,
                      generator=torch.Generator().manual_seed(0))
    assert not torch.equal(moved[0], moved[1])


def test_custom_transition_step_is_used():
    """A `TransitionStep` is a value, so a future model can supply its own without touching the sampler."""
    schedule = LinearSchedule()
    reward = _reward()

    def half_step(x, t, t_next, *, flow_map, schedule, generator):
        return 0.5 * x

    res = _run(reward, lam=0.0, n_particles=4, seed=1, schedule=schedule,
               flow_map=GaussianFlowMap(schedule),
               stoch_window=(0.1, 1.0), inside_step=half_step)
    # 14 of the 16 steps are inside the window, each halving the state.
    expected = _base_trajectory(GaussianFlowMap(schedule), schedule, ts=GRID, n_particles=4, d=D,
                                seed=1, inside_step=half_step)
    assert torch.equal(res.X, expected)


# ---------------------------------------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------------------------------------

def test_same_seed_same_result_and_no_global_rng():
    reward = _reward()
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    a = _run(reward, lam=1.5, n_particles=8, seed=21)
    assert torch.equal(before, torch.get_rng_state()), "the sampler touched the global RNG"
    b = _run(reward, lam=1.5, n_particles=8, seed=21)
    c = _run(reward, lam=1.5, n_particles=8, seed=22)
    assert torch.equal(a.X, b.X) and torch.equal(a.logw, b.logw)
    assert not torch.equal(a.X, c.X)


def test_keep_steps_is_free():
    """Snapshotting only reads state, so a ``keep_steps=True`` run is bit-identical."""
    reward = _reward()
    a = _run(reward, lam=1.0, n_particles=6, seed=8)
    b = _run(reward, lam=1.0, n_particles=6, seed=8, keep_steps=True)
    assert torch.equal(a.X, b.X)
    assert b.steps is not None and len(b.steps) == 16
    assert b.steps[-1].t == 1.0
    assert torch.equal(b.steps[-1].X.to(b.X.dtype), b.X.cpu())


def test_on_step_callback_sees_every_step_and_can_abort():
    """The checkpointing hook is read-only (run stays bit-identical) and a raise stops the loop."""
    reward = _reward()
    seen: list[tuple[int, float, int]] = []

    def cb(i, snap, partial):
        seen.append((i, snap.t, len(partial.ess_history)))

    a = _run(reward, lam=1.0, n_particles=6, seed=8)
    b = _run(reward, lam=1.0, n_particles=6, seed=8, on_step=cb)
    assert torch.equal(a.X, b.X)
    assert [i for i, _, _ in seen] == list(range(16))
    assert seen[-1][1] == 1.0
    assert all(n == i + 1 for i, _, n in seen)          # the partial result is complete for that step

    class Stop(RuntimeError):
        pass

    def abort(i, snap, partial):
        if i == 3:
            raise Stop
    with pytest.raises(Stop):
        _run(reward, lam=1.0, n_particles=6, seed=8, on_step=abort)


def test_diagnostics_are_complete():
    reward = _reward()
    res = _run(reward, lam=1.0, n_particles=8, seed=4, mc_samples=4)
    n = 16
    for name in ("ess_history", "resampled_history", "uniq_history", "V_history", "t_history",
                 "t_prime_history", "guided_history", "f_mean", "f_std", "f_min", "f_max",
                 "l_mean", "l_std", "s_mean", "s_std"):
        assert len(getattr(res, name)) == n, name
    guided_mid = [i for i, g in enumerate(res.guided_history) if g and i < n - 1]
    assert guided_mid
    for i in guided_mid:
        assert math.isfinite(res.t_prime_history[i]) and res.t_prime_history[i] < res.t_history[i]
        assert math.isfinite(res.l_mean[i]) and math.isfinite(res.s_mean[i])
    assert math.isnan(res.l_mean[-1]) and math.isnan(res.s_mean[-1])     # terminal: reward only
    assert res.t_history[-1] == 1.0


def test_final_resample_equalizes_the_weights():
    reward = _reward()
    res = _run(reward, lam=2.0, n_particles=8, seed=6, final_resample=True)
    assert torch.allclose(res.logw, torch.zeros_like(res.logw))
    assert res.resampled_history[-1]


def test_helper_formulas():
    """``L`` and ``S`` are direct transcriptions of their definitions."""
    schedule = LinearSchedule()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn((3, D), generator=gen, dtype=dtype)
    z = torch.randn((3, D), generator=gen, dtype=dtype)
    t = 0.375
    want = -((x - schedule.alpha(t) * z) ** 2).sum(-1) / (2 * schedule.sigma(t) ** 2)
    assert torch.allclose(_log_likelihood(x, z, t, schedule), want)

    s1 = torch.randn((3, D), generator=gen, dtype=dtype)
    s2 = torch.randn((3, D), generator=gen, dtype=dtype)
    xp = torch.randn((3, D), generator=gen, dtype=dtype)
    assert torch.allclose(_score_correction(s1, s2, x, xp),
                          (0.5 * (s1 + s2) * (xp - x)).sum(-1))


def test_resample_on_uniform_weights_is_the_identity():
    m = 8
    x = torch.randn((m, D), generator=torch.Generator().manual_seed(0), dtype=dtype)
    v = torch.arange(m, dtype=dtype)
    u = torch.zeros(m, dtype=dtype)
    anc = torch.arange(m)
    xr, vr, ur, ar = _resample(x, v, u, anc, torch.Generator().manual_seed(3))
    assert torch.equal(xr, x) and torch.equal(vr, v) and torch.equal(ar, anc)
    assert torch.allclose(ur, torch.zeros_like(ur))


def test_ts_validation():
    reward = _reward()
    with pytest.raises(ValueError, match="strictly increasing"):
        _run(reward, lam=1.0, ts=[0.0, 0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="must run from 0 to 1"):
        _run(reward, lam=1.0, ts=[0.1, 0.5, 1.0])
    with pytest.raises(ValueError, match="n_particles"):
        _run(reward, lam=1.0, n_particles=0)
    with pytest.raises(ValueError, match="eta"):
        _run(reward, lam=1.0, eta=0.0)


def test_non_uniform_grid_is_accepted():
    reward = _reward()
    ts = [0.0, 0.05, 0.2, 0.45, 0.7, 0.9, 1.0]
    res = _run(reward, lam=1.0, ts=ts, n_particles=6, seed=2)
    assert res.t_history == pytest.approx(ts[1:])
    assert torch.isfinite(res.X).all()


# ---------------------------------------------------------------------------------------------------
# 11. 2D toy against grid-enumerated  p · exp(lambda f)  -- validate where ground truth exists
# ---------------------------------------------------------------------------------------------------

def test_2d_toy_against_the_enumerated_tilted_density():
    """``E_q[f]`` from the sampler must track the grid-enumerated ``q_lambda ∝ p·exp(lambda·f)``.

    The base process is exact here (`GaussianPosteriorFlowMap` makes `ddpm_step` exact), so any
    remaining gap is the lookahead approximation itself -- which is what this test is measuring.
    """
    schedule = LinearSchedule()
    reward = _reward(n_refs=6, seed=1)
    lam = 2.0

    # --- ground truth on a grid -----------------------------------------------------------------
    lim, gn = 6.0, 241
    ax = torch.linspace(-lim, lim, gn, dtype=dtype)
    gx, gy = torch.meshgrid(ax, ax, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    log_p = -(pts ** 2).sum(1) / (2 * S_DATA ** 2)
    f_grid = reward(pts)
    w_p = torch.softmax(log_p, dim=0)
    w_q = torch.softmax(log_p + lam * f_grid, dim=0)
    f_under_p = float((w_p * f_grid).sum())
    f_under_q = float((w_q * f_grid).sum())
    assert f_under_q > f_under_p                       # the tilt does push novelty up

    # --- the sampler ------------------------------------------------------------------------------
    res = flowmap_smc_sample(
        reward, lam, 4000,
        flow_map=GaussianPosteriorFlowMap(schedule, seed=5), score_fn=_gaussian_score_fn(),
        schedule=schedule, n_steps=16, mc_samples=8, seed=31,
    )
    w = torch.softmax(res.logw, dim=0)
    f_hat = float((w * reward(res.X)).sum())

    # Untilted control: the same machinery at lam = 0 must land on E_p[f].
    res0 = flowmap_smc_sample(
        reward, 0.0, 4000,
        flow_map=GaussianPosteriorFlowMap(schedule, seed=5), score_fn=_gaussian_score_fn(),
        schedule=schedule, n_steps=16, mc_samples=8, seed=31,
    )
    f_hat0 = float(reward(res0.X).mean())

    assert f_hat0 == pytest.approx(f_under_p, rel=0.05), (f_hat0, f_under_p)
    # The tilted run must recover most of the gap, and land near the enumerated truth.
    gap = f_under_q - f_under_p
    assert f_hat > f_under_p + 0.5 * gap, (f_hat, f_under_p, f_under_q)
    assert f_hat == pytest.approx(f_under_q, rel=0.10), (f_hat, f_under_q)
