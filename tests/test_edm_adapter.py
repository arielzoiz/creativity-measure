"""Tests for the EDM -> ScoreFn adapter: wrap a denoiser into a gamma-convention score."""
import torch

from creativity_measure.density import Density
from creativity_measure.distances.edm_adapter import chunked_denoiser, edm_score_fn
from creativity_measure.distances.global_iem import GlobalIEMDistance
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


# ---------------------------------------------------------------------------
# chunked_denoiser: bound the denoiser batch width without changing the result
# ---------------------------------------------------------------------------
# Motivation is a measurement, not a guess: on an RTX A6000 (48 GB) with FLUX.1-dev at 512x512,
# job 695266 found 128 rows to be the ceiling and 192 rows to OOM, while an R=64 / NUM_EPS=3 run
# asks for N_eps*R = 192 rows in a single `score_bank` call. s/row was flat (0.246-0.263) across
# widths 24..128, so splitting is free.


def _counting_denoiser(counter):
    def dn(y_sigma, sigma):
        counter.append(y_sigma.shape[0])
        return mock_denoiser(y_sigma, sigma)
    return dn


def test_chunked_denoiser_matches_unchunked():
    """Splitting must be a pure memory device: identical output for every chunk size."""
    y = torch.randn(16, 3, dtype=dtype)
    sigma = torch.rand(16, dtype=dtype) + 0.5
    want = mock_denoiser(y, sigma)
    for k in (1, 3, 7, 16, 64):          # divides evenly, does not divide, equals, exceeds
        got = chunked_denoiser(mock_denoiser, k)(y, sigma)
        assert torch.equal(got, want), f"chunk {k} changed the result"


def test_chunked_denoiser_actually_splits():
    """The wrapper must bound the width, and never issue a block wider than max_rows."""
    calls: list[int] = []
    y, sigma = torch.randn(10, 2, dtype=dtype), torch.rand(10, dtype=dtype) + 0.5
    chunked_denoiser(_counting_denoiser(calls), 4)(y, sigma)
    assert calls == [4, 4, 2], calls

    calls.clear()                         # at or below the cap it must pass straight through
    chunked_denoiser(_counting_denoiser(calls), 32)(y, sigma)
    assert calls == [10], calls


def test_chunked_denoiser_slices_sigma_with_x():
    """Per-row sigma must be sliced alongside x, or later blocks get the wrong noise level."""
    y = torch.randn(9, 2, dtype=dtype)
    sigma = torch.linspace(0.1, 2.0, 9, dtype=dtype)      # distinct per row, so a mis-slice shows
    got = chunked_denoiser(mock_denoiser, 2)(y, sigma)
    assert torch.equal(got, mock_denoiser(y, sigma))


def test_chunked_denoiser_composes_with_score_fn():
    """edm_score_fn over a chunked denoiser still recovers the analytic marginal score."""
    score_fn = edm_score_fn(chunked_denoiser(mock_denoiser, 3))
    y = torch.randn(11, 2, dtype=dtype)
    for g in (2.0 ** -4, 1.0, 2.0 ** 4):
        gamma = torch.tensor(g, dtype=dtype)
        assert torch.allclose(score_fn(y, gamma), -y / (gamma ** 2 + gamma))


def test_chunked_denoiser_leaves_pairwise_unchanged():
    """The property the whole fix rests on: capping the batch cannot move the distance matrix."""
    gammas = torch.logspace(-10, 10, 40, base=2, dtype=dtype)
    X = torch.tensor([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5]], dtype=dtype)
    x_refs = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype)
    plain = GlobalIEMDistance(None, gammas, num_eps=8, seed=123,
                              score_fn=edm_score_fn(mock_denoiser))
    # num_eps=8 with 2 refs / 3 batch rows means widths of 16 and 24: chunk at 5 to force splitting.
    capped = GlobalIEMDistance(None, gammas, num_eps=8, seed=123,
                               score_fn=edm_score_fn(chunked_denoiser(mock_denoiser, 5)))
    assert torch.allclose(plain.pairwise(X, x_refs), capped.pairwise(X, x_refs))


def test_chunked_denoiser_rejects_nonpositive_max_rows():
    for bad in (0, -1):
        try:
            chunked_denoiser(mock_denoiser, bad)
        except ValueError:
            continue
        raise AssertionError(f"max_rows={bad} should have raised ValueError")


# ---------------------------------------------------------------------------
# Per-row gamma: the fused i.i.d. bank scores G*N_eps*P rows in one call, each row at its own level
# ---------------------------------------------------------------------------

def per_row_denoiser(y_sigma, sigma):
    """Like mock_denoiser but asserts sigma really is one value per row (no scalar leaking through)."""
    assert sigma.shape == (y_sigma.shape[0],)
    return y_sigma / (1.0 + sigma.reshape(-1, 1) ** 2)


def test_adapter_accepts_per_row_gamma_and_matches_scalar_loop():
    score_fn = edm_score_fn(per_row_denoiser)
    y = torch.randn(6, 2, dtype=dtype)
    gv = torch.tensor([2.0 ** -4, 0.3, 1.0, 2.0, 9.0, 2.0 ** 4], dtype=dtype)
    fused = score_fn(y, gv)
    looped = torch.cat([score_fn(y[i:i + 1], gv[i]) for i in range(6)], dim=0)
    assert torch.allclose(fused, looped, atol=1e-12)
    assert torch.allclose(fused, -y / (gv.reshape(-1, 1) ** 2 + gv.reshape(-1, 1)))


def test_adapter_scalar_gamma_is_bitwise_unchanged():
    """A scalar gamma must give exactly y/gamma-style arithmetic as before the per-row change."""
    score_fn = edm_score_fn(mock_denoiser)
    y = torch.randn(5, 3, dtype=dtype)
    gamma = torch.tensor(0.7, dtype=dtype)
    y_sigma = y / gamma
    want = mock_denoiser(y_sigma, gamma.rsqrt().reshape(1).expand(5)) - y_sigma
    assert torch.equal(score_fn(y, gamma), want)


def test_adapter_per_row_gamma_with_img_shape_and_chunking():
    score_fn = edm_score_fn(chunked_denoiser(per_row_denoiser_img, 2), img_shape=(1, 2, 2))
    y = torch.randn(5, 4, dtype=dtype)
    gv = torch.tensor([0.1, 0.5, 1.0, 3.0, 8.0], dtype=dtype)
    got = score_fn(y, gv)
    want = torch.cat([edm_score_fn(per_row_denoiser_img, img_shape=(1, 2, 2))(y[i:i + 1], gv[i])
                      for i in range(5)], dim=0)
    assert torch.allclose(got, want, atol=1e-12)


def per_row_denoiser_img(y_sigma, sigma):
    return y_sigma / (1.0 + sigma.reshape(-1, 1, 1, 1) ** 2)


def test_flow_map_denoiser_rejects_mixed_sigma_but_accepts_uniform():
    import pytest

    from creativity_measure import LinearSchedule
    from creativity_measure.generators.flux_flowmap import flow_map_denoiser

    class Stub:
        def map(self, x, t_from, t_to):
            raise AssertionError("must call denoise")

        def denoise(self, x, t):
            return x

    d = flow_map_denoiser(Stub(), LinearSchedule())
    y = torch.randn(3, 4)
    d(y, torch.full((3,), 0.5))                                   # uniform: fine
    with pytest.raises(ValueError, match="one sigma per call"):
        d(y, torch.tensor([0.5, 1.0, 2.0]))                       # mixed: would silently use sigma[0]
