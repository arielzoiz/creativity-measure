"""Smoke tests for creativity_measure.density (Task 1)."""
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from creativity_measure.density import Density

dtype = torch.float64
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)
covs  = 0.09 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
mix   = MixtureSameFamily(Categorical(torch.ones(2)),
                          MultivariateNormal(means, covs))


def log_pX(x):
    return mix.log_prob(x)


def log_pY(y, g):
    loc = g * means
    cov = g**2 * covs + g * torch.eye(2, dtype=dtype)
    return MixtureSameFamily(Categorical(torch.ones(2)),
                             MultivariateNormal(loc, cov)).log_prob(y)


p = Density(log_pX, log_pY, sample=lambda n: mix.sample((n,)), d=2)


def test_log_p_X_shape():
    out = p.log_p_X(torch.zeros(3, 2, dtype=dtype))
    assert out.shape == (3,), f"Expected (3,), got {out.shape}"


def test_log_p_Y_shape():
    out = p.log_p_Y(torch.zeros(3, 2, dtype=dtype), torch.tensor(1.0))
    assert out.shape == (3,), f"Expected (3,), got {out.shape}"


def test_sample_shape():
    out = p.sample(5)
    assert out.shape == (5, 2), f"Expected (5, 2), got {out.shape}"


def test_log_p_Y_scalar_is_0dim():
    out = p.log_p_Y_scalar(torch.zeros(2, dtype=dtype), torch.tensor(1.0))
    assert out.ndim == 0, f"Expected 0-dim tensor, got ndim={out.ndim}"


def test_no_sampler_raises():
    p2 = Density(log_pX, log_pY)
    try:
        p2.sample(1)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_sample_with_seed_reproducible():
    a = p.sample(10, seed=42)
    b = p.sample(10, seed=42)
    assert torch.allclose(a, b), "Seeded samples should be reproducible"


if __name__ == "__main__":
    test_log_p_X_shape()
    test_log_p_Y_shape()
    test_sample_shape()
    test_log_p_Y_scalar_is_0dim()
    test_no_sampler_raises()
    test_sample_with_seed_reproducible()
    print("All smoke tests passed.")
