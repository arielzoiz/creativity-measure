"""Smoke tests for creativity_measure.distances.utils.log_p_Y_given_X."""
import pytest
import torch
from torch.distributions import MultivariateNormal

from creativity_measure.distances.utils import log_p_Y_given_X, simulate_brownian

dtype = torch.float64
B, d = 4, 2
GAMMA = torch.tensor(2.5, dtype=torch.float64)


@pytest.fixture(scope="module")
def batch_inputs():
    """Returns (y, x) of shape (B, d) and a scalar gamma."""
    torch.manual_seed(0)
    x = torch.randn(B, d, dtype=dtype)
    y = torch.randn(B, d, dtype=dtype)
    return y, x, GAMMA


@pytest.fixture(scope="module")
def single_inputs():
    """Returns (y, x) of shape (d,) and a scalar gamma."""
    torch.manual_seed(1)
    x = torch.randn(d, dtype=dtype)
    y = torch.randn(d, dtype=dtype)
    return y, x, GAMMA


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------

def test_batch_output_shape(batch_inputs):
    y, x, gamma = batch_inputs
    out = log_p_Y_given_X(y, x, gamma)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


def test_single_point_output_is_0dim(single_inputs):
    y, x, gamma = single_inputs
    out = log_p_Y_given_X(y, x, gamma)
    assert out.ndim == 0, f"Expected 0-dim tensor, got ndim={out.ndim}"


# ---------------------------------------------------------------------------
# Correctness against independent reference
# ---------------------------------------------------------------------------

def _reference(y, x, gamma):
    """Compute log N(y; gamma*x, gamma*I) independently."""
    g = float(gamma)
    n = x.shape[-1]
    cov = g * torch.eye(n, device=x.device, dtype=x.dtype)
    return MultivariateNormal(g * x, cov).log_prob(y)


def test_batch_matches_reference(batch_inputs):
    y, x, gamma = batch_inputs
    out = log_p_Y_given_X(y, x, gamma)
    ref = _reference(y, x, gamma)
    assert torch.allclose(out, ref), (
        f"Output does not match reference.\nout={out}\nref={ref}"
    )


def test_single_matches_reference(single_inputs):
    y, x, gamma = single_inputs
    out = log_p_Y_given_X(y, x, gamma)
    ref = _reference(y, x, gamma)
    assert torch.allclose(out, ref), (
        f"Output does not match reference.\nout={out}\nref={ref}"
    )


def test_at_conditional_mode():
    """y = gamma*x is the conditional mode; value should match reference."""
    torch.manual_seed(2)
    x = torch.randn(B, d, dtype=dtype)
    gamma = torch.tensor(1.5, dtype=dtype)
    y = float(gamma) * x  # conditional mode
    out = log_p_Y_given_X(y, x, gamma)
    ref = _reference(y, x, gamma)
    assert torch.allclose(out, ref), (
        f"Mode check failed.\nout={out}\nref={ref}"
    )
    # At the mode the log-prob should be the highest for any y sharing that x.
    # Verify it is at least as large as at a random displaced y.
    y_shifted = y + 1.0
    out_shifted = log_p_Y_given_X(y_shifted, x, gamma)
    assert (out >= out_shifted).all(), (
        "log_p at mode should be >= log_p at shifted point"
    )


# ---------------------------------------------------------------------------
# simulate_brownian: faithful Wiener process => Var(W_gamma) = gamma at every grid point
# ---------------------------------------------------------------------------

def test_simulate_brownian_variance_matches_gamma():
    gammas = torch.logspace(-4, 4, 50, base=2, dtype=dtype)
    W = simulate_brownian(gammas, num_eps=20000, d=3, seed=0,
                          device=torch.device("cpu"), dtype=dtype)   # (N_gamma, num_eps, 1, d)
    var = W.var(dim=1, unbiased=False).mean(dim=(-1, -2))            # (N_gamma,) avg over coords
    # Must hold at i=0 too (Var=gamma0, not 0).
    for i in (0, 10, 25, 49):
        assert torch.allclose(var[i], gammas[i], rtol=0.1), (
            f"Var(W[{i}])={var[i].item():.4g} should be ~ gamma={gammas[i].item():.4g}"
        )
