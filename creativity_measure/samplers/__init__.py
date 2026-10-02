"""Samplers for q_lambda(x) = p(x) * exp(lambda * f(x)) -- see repo CLAUDE.md's "Current samplers".

Alg 1 = adaptive_tempering_smc, Alg 2 = diamond_smc, Alg 3 = flowmap_smc (with flowmap_smc_defer/
flowmap_smc_max/flowmap_smc_mean as ablations of its lookahead aggregation, imported directly by
their own test files and notebooks, not re-exported here). flow_guided is the first sampler built on
inference-time gradients rather than SMC/MCMC, and flow_guided_pc (Phase 5) is the second: the same
guided ODE step, interleaved with Unadjusted Langevin correction at fixed t, so the model's own marginal
score re-equilibrates the particle onto p_t after each nudge. At `corrector_steps=0` it reduces to
flow_guided bitwise. See "Adding a New Flow-Matching Backend" in CLAUDE.md.

smc_common.py holds shared SMC infrastructure (`_ess_from_logw`, `_systematic_resample`) reused by
adaptive_tempering_smc/diamond_smc/flowmap_smc; flow_guided_common.py holds the gradient-guidance
infrastructure (`guided_euler_step`, `_reward_grad`, `_shifted_schedule`) shared by flow_guided and
flow_guided_pc. Neither is part of this package's public surface.
"""

from creativity_measure.samplers.adaptive_tempering_smc import (
    adaptive_tempering_smc_sample, AdaptiveTemperingSMCResult, LevelSnapshot,
    Kernel, IndependenceKernel, PCNKernel,
    RejuvenationStop, MAX_N_MCMC, MIN_N_MCMC,
)
from creativity_measure.samplers.diamond_smc import (
    diamond_smc_sample, DiamondSMCResult, DiamondMapBackend,
)
from creativity_measure.samplers.flowmap_smc import (
    flowmap_smc_sample, FlowMapSMCResult, StepSnapshot, StepCallback,
    BaseSchedule, LinearSchedule, SCHEDULES, ddpm_step, flow_map_step,
)
from creativity_measure.samplers.flow_guided import (
    flow_guided_sample, FlowGuidedResult,
)
from creativity_measure.samplers.flow_guided_pc import (
    flow_guided_pc_sample, FlowGuidedPCResult, CorrectorSnapshot,
    SNR_SONG_2021, velocity_to_score, denoised_from_velocity, ula_step_size,
)

__all__ = [
    "adaptive_tempering_smc_sample", "AdaptiveTemperingSMCResult", "LevelSnapshot",
    "Kernel", "IndependenceKernel", "PCNKernel",
    "RejuvenationStop", "MAX_N_MCMC", "MIN_N_MCMC",
    "diamond_smc_sample", "DiamondSMCResult", "DiamondMapBackend",
    "flowmap_smc_sample", "FlowMapSMCResult", "StepSnapshot", "StepCallback",
    "BaseSchedule", "LinearSchedule", "SCHEDULES", "ddpm_step", "flow_map_step",
    "flow_guided_sample", "FlowGuidedResult",
    "flow_guided_pc_sample", "FlowGuidedPCResult", "CorrectorSnapshot",
    "SNR_SONG_2021", "velocity_to_score", "denoised_from_velocity", "ula_step_size",
]
