"""Samplers for q_lambda(x) = p(x) * exp(lambda * f(x)) -- see repo CLAUDE.md's "Current samplers".

Alg 1 = adaptive_tempering_smc, Alg 2 = diamond_smc, Alg 3 = flowmap_smc (with flowmap_smc_defer/
flowmap_smc_max/flowmap_smc_mean as ablations of its lookahead aggregation, imported directly by
their own test files and notebooks, not re-exported here). flow_guided is the first sampler built on
inference-time gradients rather than SMC/MCMC; see "Adding a New Flow-Matching Backend" in CLAUDE.md.

smc_common.py holds shared SMC infrastructure (`_ess_from_logw`, `_systematic_resample`) reused by
adaptive_tempering_smc/diamond_smc/flowmap_smc; not part of this package's public surface.
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

__all__ = [
    "adaptive_tempering_smc_sample", "AdaptiveTemperingSMCResult", "LevelSnapshot",
    "Kernel", "IndependenceKernel", "PCNKernel",
    "RejuvenationStop", "MAX_N_MCMC", "MIN_N_MCMC",
    "diamond_smc_sample", "DiamondSMCResult", "DiamondMapBackend",
    "flowmap_smc_sample", "FlowMapSMCResult", "StepSnapshot", "StepCallback",
    "BaseSchedule", "LinearSchedule", "SCHEDULES", "ddpm_step", "flow_map_step",
    "flow_guided_sample", "FlowGuidedResult",
]
