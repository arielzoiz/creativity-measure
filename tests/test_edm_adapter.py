"""Tests for the EDM -> ScoreFn adapter: wrap a denoiser into a gamma-convention score."""
import torch

from creativity_measure.density import Density
from creativity_measure.distances.edm_adapter import edm_score_fn
from creativity_measure.distances.generalized_global_iem import (
    GeneralizedGlobalIEMDistance,
    IEMFType,
)

dtype = torch.float64


# Exact MMSE denoiser for X ~ N(0, I): for y_sigma = x + sigma*eps,
# E[X | y_sigma] = y_sigma / (1 + sigma^2)   (linear-Gaussian posterior mean).
def mock_denoiser(y_sigma, sigma):
    return y_sigma / (1.0 + sigma.reshape(-1, 1) ** 2)


def test_adapter_recovers_marginal_score():
    # For X ~ N(0, I): Y ~ N(0, (g^2+g) I) => grad log p_Y(y) = -y/(g^2+g).
    # The adapter must turn the MMSE denoiser into exactly that score.
    score_fn = edm_score_fn(mock_denoiser)
    y = torch.randn(5, 2, dtype=dtype)
    for g in (2.0 ** -4, 1.0, 2.0 ** 4):
        gamma = torch.tensor(g, dtype=dtype)
        got = score_fn(y, gamma)
        want = -y / (gamma ** 2 + gamma)
        assert torch.allclose(got, want), (g, got, want)


# Standard-normal prior, analytic density for the autograd reference.
def log_pX(x):
    z = torch.zeros(x.shape[-1], dtype=x.dtype)
    return torch.distributions.Normal(z, torch.ones_like(z)).log_prob(x).sum(-1)


def log_pY(y, gamma):
    std = (gamma ** 2 + gamma).sqrt()
    z = torch.zeros(y.shape[-1], dtype=y.dtype)
    return torch.distributions.Normal(z, std).log_prob(y).sum(-1)


def test_adapter_end_to_end_matches_density():
    p = Density(log_pX, log_pY, d=2)
    gammas = torch.logspace(-10, 10, 40, base=2, dtype=dtype)
    X = torch.tensor([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5]], dtype=dtype)
    x_refs = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype)
    score_fn = edm_score_fn(mock_denoiser)
    d_auto = GeneralizedGlobalIEMDistance(p, gammas, num_eps=8, seed=123, f_type=IEMFType.SQUARED)
    d_edm = GeneralizedGlobalIEMDistance(None, gammas, num_eps=8, seed=123, f_type=IEMFType.SQUARED,
                                         score_fn=score_fn)
    assert torch.allclose(d_auto.pairwise(X, x_refs), d_edm.pairwise(X, x_refs))
