# Work in progress: this file is a draft

"""Twisted-diffusion SMC sampler for  q_lambda(x) ∝ p(x) · exp(lambda · f(x)).

A second sampler for the same tilted target as ``creativity_measure.adaptive_tempering_smc.adaptive_tempering_smc_sample``, but built on a
completely different mechanism. Where ``adaptive_tempering_smc.py`` moves particles in **data space** and treats the
generator ``G(z) -> x`` as a black box, this sampler runs **inside the EDM denoising trajectory**:
particles descend the Karras sigma-schedule via the model's own ancestral (predict-``x0``-then-renoise)
step, and at each guided level are reweighted by a *lookahead reward twist* and resampled. High-reward
trajectories survive and duplicate; the stochastic ancestral step keeps duplicated particles diverse.

The tilt reward ``f`` is the frozen `~creativity_measure.tilt.Reward` (identical to the one ``adaptive_tempering_smc.py``
consumes) — for the intended use, the **normalized squared-IEM** reward
(`~creativity_measure.tilt.NormalizedExpectedDistanceReward` + ``SquaredGlobalIEMDistance``, via
``selector.normalized_reward(R)``). ``lambda`` enters at full strength through the twist increments;
there is no separate ``beta`` tempering (the diffusion schedule plays that role).

Convention
----------
Everything is in the repo's EDM / Karras **variance-exploding** convention (``alpha ≡ 1``): the observed
state at noise level ``sigma`` is ``x_sigma = x0 + sigma·eps``, and the injected ``denoiser``
``D(x_sigma, sigma) = E[X | x_sigma]`` is the "1-step lookahead / flow map". The schedule and the
deterministic step are the shared ``generators.base.karras_sigma_schedule`` / ``edm_ode_step``.

Scope (core, approximate weight)
--------------------------------
This is the *approximate* twist: each guided level scores its ``K`` clean-lookahead estimates by the
soft value ``V = logmeanexp_k(lambda·f(z_k))`` and accumulates the telescoping increment ``V - V_prev``.
Like ``adaptive_tempering_smc.py``'s ``IndependenceKernel`` it **under-tilts** — the denoiser-projected lookahead cannot
reach the off-manifold mass of ``q``, so ``E_q[f]`` lands only partway from the prior toward the exact
target. Quantitative recovery of ``q`` (and the recovery-tether ``-||x_sigma - z||^2/(2 sigma^2)`` +
Tweedie score quadrature that cancel this bias) is the unbiased score-correction follow-up;
``use_score_correction=True`` raises ``NotImplementedError`` for now.

Determinism
-----------
All randomness (initial noise, ancestral + branch ``eps``, resampling) is driven from a single
``torch.Generator(seed)``; no global torch RNG state is touched, so same ``seed`` -> identical output
(given the caller fixed the ``Distance`` Brownian seed upstream, as ``adaptive_tempering_smc.py`` also assumes).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure.distances.edm_adapter import Denoiser
from creativity_measure.generators.base import edm_ode_step, karras_sigma_schedule
from creativity_measure.smc_common import _ess_from_logw, _systematic_resample
from creativity_measure.tilt import Reward


@dataclass
class DiffusionSMCResult:
    """Output of :func:`diffusion_smc_sample`.

    Attributes:
        X:              final particles, (M, d) — samples from q̂_lambda.
        logw:           final unnormalized log-weights, (M,). ``softmax(logw)`` gives the self-normalized
                        weights (all-equal if the run ended on a resample or with ``final_resample``).
        sigmas:         the Karras sigma-schedule actually used, (n_steps + 1,) (trailing 0 included).
        ess_history:    ESS after each guided level's reweight (pre-resample); one entry per guided level.
        resample_steps: schedule indices at which a resample fired.
    """

    X: Float[Tensor, "M d"]
    logw: Float[Tensor, "M"]
    sigmas: Float[Tensor, "steps_plus_1"]
    ess_history: list[float]
    resample_steps: list[int]


def diffusion_smc_sample(
    reward: Reward,
    lam: float,
    n_particles: int,
    *,
    denoiser: Denoiser,
    d: int,
    img_shape: tuple[int, ...] | None = None,
    sigma_min: float = 2e-3,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    n_steps: int = 64,
    mc_samples: int = 4,
    guidance_window: tuple[float, float] | None = None,
    ess_threshold: float = 0.5,
    final_resample: bool = False,
    use_score_correction: bool = False,
    seed: int | None = None,
) -> DiffusionSMCResult:
    """Sample from  q̂_lambda(x) ∝ p(x) · exp(lambda · f(x))  via twisted-diffusion SMC.

    Particles descend the Karras sigma-schedule. At each **guided** level (see ``guidance_window``) every
    particle:

    1. forms a base clean estimate ``x0_hat = D(x_sigma, sigma)`` (1-step lookahead);
    2. advances by an ancestral **predict-x0-then-renoise** step ``x' = x0_hat + sigma'·eps`` (this
       stochastic step is what keeps resampled duplicates diverse);
    3. draws ``mc_samples`` branch posterior draws, looks ahead ``z_k = D(x_branch, sigma')``, and scores
       each by the reward ``lambda·f(z_k)``;
    4. gets a soft-value level potential ``V = logmeanexp_k(lambda·f(z_k))`` and accumulates the **increment**
       ``V - V_prev`` into its log-weight (telescoping, so the terminal weight ∝ exp(lambda·f(x_final)) once
       ``sigma -> 0``).

    Outside ``guidance_window`` the step is the plain deterministic ``edm_ode_step`` and no reweight/resample
    happens (the compute-saving "critical window" of the source algorithm).

    Args:
        reward:        frozen reward ``f`` (distance + refs + weights); ``reward.x_refs`` pins device/dtype.
        lam:           tilt strength ``lambda``.
        n_particles:   number of SMC particles ``M``.
        denoiser:      EDM denoiser ``D(x_sigma, sigma) = E[X | x_sigma]`` — the flow map / 1-step lookahead.
                       Obtain it from the exported factories (``generators.density_denoiser`` for the 2D toy,
                       ``generators.eps_to_edm_denoiser`` for SD, or a raw ``EDMPrecond`` net for pixels).
        d:             flat data dimensionality (must match ``reward.x_refs.shape[-1]``).
        img_shape:     per-sample shape ``(C, H, W)`` the denoiser expects; ``None`` keeps the flat ``(B, d)``
                       layout. Reshapes flat<->image around ``denoiser``, exactly as ``edm_generator`` does.
        sigma_min/max, rho, n_steps: the Karras prob-flow schedule (``n_steps`` = number of transitions ``N``).
        mc_samples:    branches per particle ``K`` used to estimate the level potential (``K=1`` is valid).
        guidance_window: ``(sigma_lo, sigma_hi)`` — guide (reweight/resample) only while ``sigma_i`` is in
                       ``[sigma_lo, sigma_hi]``. ``None`` (default) guides at every level.
        ess_threshold: resample when ESS < ``ess_threshold * M``.
        final_resample: if ``True``, resample once at the end so returned particles are equal-weight.
        use_score_correction: reserved for the unbiased Tweedie/quadrature weight; ``True`` raises
                       ``NotImplementedError`` (follow-up).
        seed:          seeds the single internal ``torch.Generator`` driving all randomness.

    Returns: `DiffusionSMCResult`.
    """
    if use_score_correction:
        raise NotImplementedError(
            "use_score_correction=True (unbiased Tweedie + trapezoidal quadrature) is not implemented yet; "
            "use the approximate weight (use_score_correction=False)."
        )
    if mc_samples < 1:
        raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")

    x_refs = reward.x_refs
    device, dtype = x_refs.device, x_refs.dtype
    if x_refs.shape[-1] != d:
        raise ValueError(f"d={d} does not match reward.x_refs feature dim {x_refs.shape[-1]}")

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # Flat (B, d) <-> image reshape around the denoiser, mirroring ``edm_generator``.
    def D(x: Float[Tensor, "B d"], sigma: Float[Tensor, ""]) -> Float[Tensor, "B d"]:
        sig_b = sigma.reshape(1).expand(x.shape[0])
        if img_shape is None:
            return denoiser(x, sig_b)
        return denoiser(x.reshape(x.shape[0], *img_shape), sig_b).reshape(x.shape)

    def in_window(sig: float) -> bool:
        if guidance_window is None:
            return True
        lo, hi = guidance_window
        return lo <= sig <= hi

    sigmas = karras_sigma_schedule(sigma_min, sigma_max, rho, n_steps, device=device, dtype=dtype)

    M, K = n_particles, mc_samples
    x = sigma_max * torch.randn(M, d, generator=gen, device=device, dtype=dtype)   # pure noise at sigma_0
    logw = torch.zeros(M, device=device, dtype=dtype)
    v_prev = torch.zeros(M, device=device, dtype=dtype)      # previous guided level's potential (telescoping)
    ess_target = ess_threshold * M

    ess_history: list[float] = []
    resample_steps: list[int] = []

    for i in range(n_steps):
        s0, s1 = sigmas[i], sigmas[i + 1]

        if not in_window(float(s0)):
            x = edm_ode_step(x, D, s0, s1)                   # deterministic advance, no twist
            continue

        # (1) base clean estimate at the current level (1-step lookahead).
        x0_hat = D(x, s0)                                    # (M, d)

        # VE / DDPM ancestral reverse transition to the next level, sigma_i -> sigma_{i+1}:
        #   x' = x0_hat + (sigma'^2/sigma^2)(x - x0_hat) + sigma'*sqrt(1 - sigma'^2/sigma^2)*eps.
        # The DDIM drift term keeps the current sample's residual direction (dropping it over-randomizes
        # and under-disperses the x0 posterior); the extra noise makes the step stochastic so resampled
        # duplicates diverge. At the final step (sigma' = 0) this collapses to the deterministic x = x0_hat.
        ratio = (s1 / s0) ** 2                               # sigma'^2 / sigma^2  (0 at the final step)
        mu = x0_hat + ratio * (x - x0_hat)                   # (M, d) posterior mean
        post_std = s1 * (1.0 - ratio).clamp_min(0.0).sqrt()  # scalar posterior std

        # (2) soft-value lookahead potential: K posterior draws to the next level, look ahead to a clean
        #     estimate z_k, and score by lambda*f(z_k). The level potential is the soft value
        #     V = logmeanexp_k(lambda*f(z_k)); the telescoping increment V - V_prev twists the ancestral
        #     process toward exp(lambda*f). (The recovery tether -||x - z||^2/(2 sigma^2) is deliberately
        #     NOT applied here: on its own it is only half of a correction whose other half is the Tweedie
        #     score-quadrature, and unbalanced it just injects late-trajectory weight variance / degeneracy.
        #     The tether + quadrature enter together in the unbiased score-correction follow-up.)
        #     At sigma' = 0 the lookahead is x0_hat itself -- the denoiser is undefined at sigma = 0
        #     (gamma = 1/sigma^2 -> inf), so skip the renoise-and-redenoise.
        if float(s1) > 0.0:
            eps = torch.randn(M, K, d, generator=gen, device=device, dtype=dtype)
            x_branch = (mu.unsqueeze(1) + post_std * eps).reshape(M * K, d)
            z = D(x_branch, s1).reshape(M, K, d)             # (M, K, d) clean estimates from the branches
        else:
            z = x0_hat.unsqueeze(1).expand(M, K, d)          # (M, K, d) all K = the clean landing point
        fz = reward(z.reshape(M * K, d)).reshape(M, K)       # (M, K) normalized-IEM reward
        v_curr = torch.logsumexp(lam * fz, dim=1) - math.log(K)   # (M,) soft value V = logmeanexp_k lambda*f(z_k)

        # (3) accumulate the telescoping increment.
        logw = logw + (v_curr - v_prev)
        v_prev = v_curr

        # (4) ancestral advance of the carried particle (one more posterior draw -> stochastic step).
        eps0 = torch.randn(M, d, generator=gen, device=device, dtype=dtype)
        x = mu + post_std * eps0

        # (5) resample on low ESS.
        ess = _ess_from_logw(logw)
        ess_history.append(ess)
        if ess < ess_target:
            idx = _systematic_resample(torch.softmax(logw, dim=0), gen)
            x = x[idx]
            v_prev = v_prev[idx]
            logw = torch.zeros_like(logw)
            resample_steps.append(i)

    if final_resample:
        idx = _systematic_resample(torch.softmax(logw, dim=0), gen)
        x = x[idx]
        logw = torch.zeros_like(logw)

    return DiffusionSMCResult(
        X=x,
        logw=logw,
        sigmas=sigmas,
        ess_history=ess_history,
        resample_steps=resample_steps,
    )


# Callable alias mirroring ``smc.RewardFn`` for callers that build a bare reward function.
RewardFn = Callable[[Tensor], Tensor]
