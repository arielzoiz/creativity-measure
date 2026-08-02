"""Sequential Monte Carlo with Diamond Maps — Algorithm 2 of Holderrieth et al. (arXiv:2602.05993).

Samples  q_lambda(x) ∝ p(x) · exp(lambda · f(x))  by running SMC *inside* the generative trajectory of a
pretrained stochastic flow-map model, instead of moving particles in data space
(``adaptive_tempering_smc.py``) or twisting an EDM denoising path (``diffusion_smc.py``).

The loop walks ``t: 0 -> 1`` (noise -> data) in ``n_steps`` transitions. At each step:

1. every particle is pushed through one **DDPM transition** ``x_{t+h} ~ p^DDPM_{t+h|t}(.|x_t)``
   (Algorithm 2 line 6; in the released implementation this is the GLASS inner-flow sampler);
2. ``mc_samples`` **posterior lookahead** samples ``z^k ~ p(x_1 | x_{t+h})`` are drawn in *one* network
   call each by the diamond map (line 9) — this is the whole point of the model class: a consistent,
   cheap estimate of the value function without simulating the remaining trajectory;
3. the reward is evaluated on those clean candidates and reduced to the soft value
   ``V = log( (1/K) sum_k exp(lambda·f(z^k)) )`` (lines 7-12);
4. the potential accumulates ``U += V_{t+h} - V_t`` (line 13) and drives ESS-triggered resampling
   (lines 16-20).

Both model calls live behind :class:`DiamondMapBackend`, so this module is pure PyTorch and has no
JAX dependency: the real backend (``creativity_measure.backends.diamond_maps_jax``) bridges to the
authors' JAX implementation, while the tests drive a small analytic backend.

Notes
-----
* **The reward is frozen**, exactly as for the other samplers: ``x_refs``, ``weights`` and the
  ``Distance``'s Brownian seed are fixed before sampling, so ``f`` is a deterministic function of ``x``.
* **lambda enters in exactly one place** — the ``lam * f`` inside the log-sum-exp of step 3. It is the
  inverse temperature of the target; the paper's separate resampling ``temperature`` is kept as a
  distinct knob (default 1.0) and only reshapes the resampling weights, not the target.
* **Determinism.** Torch randomness (resampling) comes from a caller-owned ``torch.Generator``; the
  backend owns its own stream and is seeded from the same ``seed``. No global RNG is touched.
* **Final step.** At ``t+h = 1`` the particle *is* the clean sample, so the lookahead collapses
  analytically to ``V_N = lambda · f(x_1)`` — one reward call, no Monte Carlo. The released code skips
  this update entirely; we keep it, so ``logw`` is a correct final weight (see ``final_resample``).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Float, Int
from torch import Tensor

from creativity_measure.smc_common import _ess_from_logw, _systematic_resample
from creativity_measure.tilt import Reward

RewardFn = Callable[[Tensor], Tensor]   # f(X) -> (B,)


@runtime_checkable
class DiamondMapBackend(Protocol):
    """The two generative operations Algorithm 2 needs, plus the shape they act on.

    Deliberately narrow: everything model-, framework- and checkpoint-specific stays on the far side
    of this boundary, so :func:`diamond_smc_sample` is testable without a GPU, without JAX and without
    a 10 GB checkpoint. Implementations must be **stateless with respect to the sampler** — the loop
    threads no hidden state through them beyond their own RNG.

    A backend that carries its own RNG (all real ones do) should also offer ``reset_rng(seed)``. It
    is not part of this protocol because the loop never calls it, but a **lambda sweep that reuses one
    backend instance must** call it before each run — otherwise every lambda sees different initial
    particles and different base transitions, and the comparison confounds the tilt with the noise.
    """

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-particle latent shape, e.g. ``(4, 32, 32)``."""
        ...

    def init_particles(self, n: int) -> Float[Tensor, "n *latent"]:
        """Draw ``n`` initial particles ``x_0 ~ N(0, I)`` (Algorithm 2 line 2)."""
        ...

    def base_step(
        self, x_t: Float[Tensor, "n *latent"], step_idx: int
    ) -> Float[Tensor, "n *latent"]:
        """One DDPM transition ``x_{t+h} ~ p^DDPM_{t+h|t}(.|x_t)`` for transition ``step_idx``.

        ``step_idx`` indexes the backend's own time grid, which must have the same number of
        transitions the sampler was given.
        """
        ...

    def posterior_sample(
        self, x_t: Float[Tensor, "n *latent"], step_idx: int, mc_samples: int
    ) -> Float[Tensor, "nk *latent"]:
        """``mc_samples`` draws of ``x_1 ~ p(.|x_t)`` per particle, at time ``ts[step_idx + 1]``.

        Returns ``(n * mc_samples, *latent)`` ordered so that ``.view(n, mc_samples, ...)`` groups the
        draws by their originating particle — i.e. ``repeat_interleave`` order, not ``repeat`` order.
        """
        ...


@dataclass
class DiamondSMCResult:
    """Output of :func:`diamond_smc_sample`.

    Attributes:
        X:          final particles at ``t = 1``, ``(M, *latent_shape)``.
        logw:       final unnormalized log-weights ``U``, ``(M,)``, *including* the final-step value
                    update. ``softmax(logw)`` gives the self-normalized q̂_lambda weights; all-equal
                    when the run ended on a resample or with ``final_resample=True``.
        ess_history:      ESS after each step's potential update, before any resampling. Length ``N``.
        resampled_history: whether each step actually resampled. Length ``N``.
        V_history:        per-step soft value ``V`` per particle, each ``(M,)``. Length ``N``.
        f_mean/f_std/f_min/f_max: reward statistics over the lookahead candidates at each step
                    (over the final particles at the last step). Length ``N`` each.
        uniq_history:     distinct ancestors surviving after each step, as a fraction of ``M``. With
                    ``ess_history`` this is the degeneracy signal that decides how far lambda can be
                    pushed — novelty ``E_q[f]`` rises forever and is *not* the stopping signal.
    """

    X: Float[Tensor, "M ..."]
    logw: Float[Tensor, "M"]
    ess_history: list[float] = field(default_factory=list)
    resampled_history: list[bool] = field(default_factory=list)
    V_history: list[Tensor] = field(default_factory=list)
    f_mean: list[float] = field(default_factory=list)
    f_std: list[float] = field(default_factory=list)
    f_min: list[float] = field(default_factory=list)
    f_max: list[float] = field(default_factory=list)
    uniq_history: list[float] = field(default_factory=list)


def _soft_value(
    f: Float[Tensor, "MK"], n_particles: int, mc_samples: int, lam: float
) -> Float[Tensor, "M"]:
    """``V^m = log( (1/K) sum_k exp(lambda · f(z^{k,m})) )`` — Algorithm 2 lines 7-12.

    Computed with ``logsumexp`` rather than the paper's literal running sum of ``exp``, which
    overflows as soon as ``lambda · f`` exceeds ~700. Mathematically identical.
    """
    f_grouped = f.view(n_particles, mc_samples)
    return torch.logsumexp(lam * f_grouped, dim=1) - math.log(mc_samples)


def _resample(
    x: Float[Tensor, "M ..."],
    V_prev: Float[Tensor, "M"],
    U: Float[Tensor, "M"],
    ancestors: Int[Tensor, "M"],
    temp: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Algorithm 2 lines 17-20: draw parents ∝ softmax(U/temp), reindex, zero the potential.

    Systematic (not multinomial) resampling, per ``smc_common``. Besides the lower variance, this
    makes the ``lambda = 0`` case *exact*: on uniform weights systematic resampling returns the
    identity permutation, so an untilted run reproduces the base sampler particle-for-particle.
    """
    weights = torch.softmax(U / temp, dim=0)
    idx = _systematic_resample(weights, generator)
    return x[idx], V_prev[idx], torch.zeros_like(U), ancestors[idx]


def diamond_smc_sample(
    reward: Reward,
    lam: float,
    n_particles: int,
    *,
    backend: DiamondMapBackend,
    n_steps: int = 6,
    mc_samples: int = 4,
    ess_threshold: float = 1.0,
    temp: float = 1.0,
    final_resample: bool = False,
    seed: int | None = None,
    verbose: bool = False,
) -> DiamondSMCResult:
    """Sample ``q_lambda ∝ p · exp(lambda·f)`` by SMC over a diamond map's generative trajectory.

    Args:
        reward:     the frozen tilt ``f``. Called on **flattened** particles, ``(B, d)``, matching the
                    ``Reward`` contract; ``reward.x_refs`` pins device and dtype for the whole run.
        lam:        inverse temperature of the target. ``lam = 0`` reduces the run to the base sampler.
        n_particles: number of particles ``M``.
        backend:    the two model calls (see :class:`DiamondMapBackend`).
        n_steps:    number of DDPM transitions ``N`` over ``t ∈ [0, 1]``.
        mc_samples: lookahead draws ``K`` per particle per step. Cost is linear in ``K``; the reward
                    dominates the step, so this is the main cost knob.
        ess_threshold: resample when ``ESS < ess_threshold · M``. ``1.0`` (the released config's
                    value) means resample at every step.
        temp:       resampling temperature. ``1.0`` matches the paper; the released code folds an
                    unscaled pixel reward into ``temp=0.05`` instead, which ``lam`` now does explicitly.
        final_resample: resample once more after the final value update, yielding an equally-weighted
                    ensemble. Off by default: ``logw`` carries the weight, matching
                    ``AdaptiveTemperingSMCResult``.
        seed:       seeds both the torch generator (resampling) and the backend's own stream.
        verbose:    print per-step ESS / reward statistics.

    Returns:
        :class:`DiamondSMCResult` with the final particles, their log-weights and per-step diagnostics.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if mc_samples < 1:
        raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")

    device, dtype = reward.x_refs.device, reward.x_refs.dtype
    generator = torch.Generator(device=device)
    generator.manual_seed(0 if seed is None else seed)

    M, K = n_particles, mc_samples
    x_t = backend.init_particles(M).to(device=device, dtype=dtype)
    U = torch.zeros(M, device=device, dtype=dtype)
    V_prev = torch.zeros(M, device=device, dtype=dtype)
    ancestors = torch.arange(M, device=device)

    result = DiamondSMCResult(X=x_t, logw=U)

    for n in range(n_steps):
        # --- Algorithm 2 line 6: one DDPM transition -------------------------------------------
        x_next = backend.base_step(x_t, n).to(device=device, dtype=dtype)
        is_final = n == n_steps - 1

        # --- lines 7-12: soft value from the posterior lookahead --------------------------------
        # At the last transition t+h = 1, so x_next is already the clean sample and the lookahead
        # collapses to f(x_1) exactly: no diamond-map call, no Monte Carlo, K times cheaper.
        if is_final:
            f = reward(x_next.reshape(M, -1))
            V_next = lam * f
        else:
            z = backend.posterior_sample(x_next, n, K).to(device=device, dtype=dtype)
            f = reward(z.reshape(M * K, -1))
            V_next = _soft_value(f, M, K, lam)

        # --- line 13: accumulate the potential ---------------------------------------------------
        U = U + V_next - V_prev
        V_prev = V_next

        ess = _ess_from_logw(U / temp)
        result.ess_history.append(ess)
        result.V_history.append(V_next.detach().clone())
        result.f_mean.append(float(f.mean()))
        result.f_std.append(float(f.std()) if f.numel() > 1 else 0.0)
        result.f_min.append(float(f.min()))
        result.f_max.append(float(f.max()))

        # --- lines 16-20: ESS-triggered resampling ----------------------------------------------
        do_resample = (ess < ess_threshold * M) and (not is_final or final_resample)
        if do_resample:
            x_next, V_prev, U, ancestors = _resample(
                x_next, V_prev, U, ancestors, temp, generator
            )
        result.resampled_history.append(do_resample)
        result.uniq_history.append(len(torch.unique(ancestors)) / M)

        if verbose:
            print(
                f"  step {n + 1}/{n_steps}  ESS/M={ess / M:.2f}  "
                f"resampled={do_resample}  uniq/M={result.uniq_history[-1]:.2f}  "
                f"f mean={result.f_mean[-1]:.4f} std={result.f_std[-1]:.4f}"
            )

        x_t = x_next

    result.X = x_t
    result.logw = U
    return result
