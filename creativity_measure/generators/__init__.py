"""Generator library: model-agnostic EDM prob-flow core + per-model factory functions.

Every factory returns the same interface -- a deterministic ``G(z): N(0,I) -> x ~ p`` (flat ``(B, d)``) --
so ``creativity_measure.adaptive_tempering_smc`` / ``tilt`` stay pure math and never reference any generative-model framework.
Model wrappers (``tiny_sd``, ``flux``) import ``diffusers`` lazily, so importing this package never requires
the optional ``models`` extra.
"""

from .base import (
    edm_generator,
    edm_ode_step,
    eps_to_edm_denoiser,
    heun_prob_flow,
    karras_sigma_schedule,
)
from .toy_2d import density_denoiser, density_generator
from .edm_pixel import build_edm_pixel_generator, build_edm_pixel_generator_from_pkl
from .tiny_sd import build_tiny_sd_generator
from .flux import build_flux_generator
from .flux_flowmap import (
    DualTimeEmbedder,
    FluxFlowMap,
    build_flux_flow_map,
    flow_map_denoiser,
    load_flow_map_weights,
)

__all__ = [
    "heun_prob_flow",
    "karras_sigma_schedule",
    "edm_ode_step",
    "edm_generator",
    "eps_to_edm_denoiser",
    "density_denoiser",
    "density_generator",
    "build_edm_pixel_generator",
    "build_edm_pixel_generator_from_pkl",
    "build_tiny_sd_generator",
    "build_flux_generator",
    "DualTimeEmbedder",
    "FluxFlowMap",
    "build_flux_flow_map",
    "flow_map_denoiser",
    "load_flow_map_weights",
]
