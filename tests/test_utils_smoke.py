"""Smoke tests for creativity_measure.distances.utils."""
import torch

from creativity_measure.distances.utils import simulate_brownian

dtype = torch.float64


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
