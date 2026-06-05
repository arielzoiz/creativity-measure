"""Tests for GeneralizedGlobalIEMDistance score_fn wiring."""
import torch
import pytest

from creativity_measure.density import Density
from creativity_measure.distances.generalized_global_iem import (
    GeneralizedGlobalIEMDistance,
    IEMFType,
)

dtype = torch.float64


# Standard-normal prior X ~ N(0, I); Y = gamma*X + sqrt(gamma)*W => Y ~ N(0, (gamma^2 + gamma) I).
def log_pX(x):
    z = torch.zeros(x.shape[-1], dtype=x.dtype)
    return torch.distributions.Normal(z, torch.ones_like(z)).log_prob(x).sum(-1)


def log_pY(y, gamma):
    std = (gamma ** 2 + gamma).sqrt()
    z = torch.zeros(y.shape[-1], dtype=y.dtype)
    return torch.distributions.Normal(z, std).log_prob(y).sum(-1)


p = Density(log_pX, log_pY, d=2)
gammas = torch.logspace(-10, 10, 40, base=2, dtype=dtype)
X = torch.tensor([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5]], dtype=dtype)
x_refs = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype)


# ---------------------------------------------------------------------------
# Injected score_fn reproduces the autograd path (for every f), proving the wiring
# and that score_fn=None is unchanged.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f_type", [IEMFType.IDENTITY, IEMFType.SQUARED])
def test_score_fn_matches_autograd(f_type):
    # Closed-form marginal score for X ~ N(0, I): Y ~ N(0, (g^2+g) I) => ∇log p_Y(y) = -y/(g^2+g).
    # Independent of the autograd path, so this validates both the analytic conditional term
    # (x - y/g) and the autograd marginal score against ground truth.
    score_fn = lambda y, g: -y / (g ** 2 + g)
    d_auto = GeneralizedGlobalIEMDistance(p, gammas, num_eps=8, seed=123, f_type=f_type)
    d_inj = GeneralizedGlobalIEMDistance(None, gammas, num_eps=8, seed=123, f_type=f_type,
                                         score_fn=score_fn)
    assert torch.allclose(d_auto.pairwise(X, x_refs), d_inj.pairwise(X, x_refs))


def test_requires_density_or_score_fn():
    with pytest.raises(ValueError):
        GeneralizedGlobalIEMDistance(None, gammas, num_eps=8, seed=123)
