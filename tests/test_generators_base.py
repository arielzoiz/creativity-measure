"""Tests for the generators package seams that need no model download.

Covers:
  (a) ``eps_to_edm_denoiser`` -- the VP epsilon -> EDM denoiser adapter (SD path), via a mock eps model.
  (b) ``build_edm_pixel_generator`` -- the pretrained-EDM-checkpoint path, via a mock ``EDMPrecond`` net.
  (c) ``build_tiny_sd_generator`` -- clean ``ImportError`` when the optional ``diffusers`` dep is absent.
"""

import torch
import pytest

from creativity_measure import (
    eps_to_edm_denoiser,
    build_edm_pixel_generator,
    build_tiny_sd_generator,
)

dtype = torch.float64


# --- (a) eps_to_edm_denoiser --------------------------------------------------------------------

def _t_to_sigma(t: torch.Tensor, log_sigmas: torch.Tensor) -> torch.Tensor:
    """Inverse of the helper's sigma->t: interpolate the log-sigma grid at fractional index ``t``."""
    t = t.reshape(-1)
    n = log_sigmas.shape[0]
    low = t.floor().long().clamp(0, n - 1)
    high = (low + 1).clamp(0, n - 1)
    w = (t - low).clamp(0.0, 1.0)
    return ((1.0 - w) * log_sigmas[low] + w * log_sigmas[high]).exp()


def test_eps_to_edm_denoiser_matches_analytic_normal():
    """For X~N(0,I) the EDM denoiser is D(x,sigma)=x/(1+sigma^2). A mock eps model that yields exactly
    that (eps = (c_in*x) * sigma * c_in) must be recovered through the c_in / sigma->t / eps->x0 conversion."""
    model_sigmas = torch.logspace(-3, 2, 2000, dtype=dtype)      # ascending discrete schedule
    log_sigmas = model_sigmas.log()

    def eps_fn(u, t):
        sigma_rec = _t_to_sigma(t, log_sigmas).reshape(-1, 1)     # reconstruct sigma from timestep
        c_in = 1.0 / (sigma_rec ** 2 + 1.0).sqrt()
        return u * (sigma_rec * c_in)                            # -> denoiser output x/(1+sigma^2)

    D = eps_to_edm_denoiser(eps_fn, model_sigmas)
    x = torch.randn(7, 2, dtype=dtype)
    for sig in (0.1, 1.0, 5.0):
        sigma = torch.full((7,), sig, dtype=dtype)
        got = D(x, sigma)
        want = x / (1.0 + sig ** 2)
        assert torch.allclose(got, want, atol=1e-3), f"sigma={sig}: {got[0]} vs {want[0]}"


# --- (b) build_edm_pixel_generator (pretrained EDM checkpoint path) ------------------------------

class _MockEDMNet:
    """Analytic EDMPrecond-like net: E[X | x_sigma] = x_sigma/(1+sigma^2) for X~N(0,I)."""
    sigma_min = 2e-3
    sigma_max = 80.0
    label_dim = 0

    def __call__(self, x, sigma, class_labels=None):
        return x / (1.0 + sigma.reshape(-1, 1, 1, 1) ** 2)


def test_build_edm_pixel_generator_runs_and_deterministic():
    """The no-conversion EDM path: wrap a net, get a deterministic flat (B, d) generator."""
    C, H, W = 1, 2, 2
    d = C * H * W
    G = build_edm_pixel_generator(_MockEDMNet(), img_shape=(C, H, W), n_steps=8)
    z = torch.randn(16, d, generator=torch.Generator().manual_seed(0), dtype=dtype)
    x = G(z)
    assert x.shape == (16, d)
    assert x.isfinite().all()
    assert torch.equal(G(z), G(z))                               # deterministic map


def test_build_edm_pixel_generator_uses_net_sigma_range():
    """sigma_min/max default to the net's supported range (overridable)."""
    net = _MockEDMNet()
    # Should run without error using net.sigma_min / net.sigma_max defaults.
    G = build_edm_pixel_generator(net, img_shape=(1, 2, 2), n_steps=4)
    z = torch.randn(4, 4, dtype=dtype)
    assert G(z).isfinite().all()


# --- (c) optional-dependency guard --------------------------------------------------------------

def test_tiny_sd_raises_clean_import_error_without_diffusers():
    try:
        import diffusers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="diffusers"):
            build_tiny_sd_generator("a prompt")
    else:
        pytest.skip("diffusers is installed; missing-deps guard not exercised")
