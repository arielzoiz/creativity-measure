"""Tests for transparent CPU/GPU device selection (creativity_measure.device + pipeline placement)."""
import math

import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from creativity_measure.density import Density
from creativity_measure.device import (
    default_device, set_default_device, default_dtype, set_default_dtype,
)
from creativity_measure.distances.global_iem import GlobalIEMDistance
from creativity_measure.refset.fps import FPSRefs
from creativity_measure.refset.random_refs import RandomRefs

# ---------------------------------------------------------------------------
# Shared fixtures: 2-mode GMM (same setup as test_global_iem)
# ---------------------------------------------------------------------------
dtype = torch.float64
means = torch.tensor([[2., 0.], [-2., 0.]], dtype=dtype)
sig2 = 0.09


def log_pX(x):
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(
        Categorical(torch.ones(2, dtype=dtype)),
        MultivariateNormal(means, covs, validate_args=False),
    )
    return mix.log_prob(x)


def log_pY(y, gamma):
    var_diag = gamma ** 2 * sig2 + gamma
    d = y.shape[-1]
    batch_shape = y.shape[:-1]
    y_flat = y.reshape(-1, d)
    log_comps = []
    for k in range(2):
        mu_k = gamma * means[k].to(device=y.device, dtype=y.dtype)
        cov_k = var_diag * torch.eye(d, device=y.device, dtype=y.dtype)
        dist_k = MultivariateNormal(mu_k, cov_k, validate_args=False)
        log_comps.append(dist_k.log_prob(y_flat))
    log_comps = torch.stack(log_comps, dim=0)
    log_pi = -math.log(2)
    return torch.logsumexp(log_pi + log_comps, dim=0).reshape(batch_shape)


def sampler(n, generator=None):
    # MixtureSameFamily.sample() takes no generator; this sampler is only used on the
    # seed/global-RNG path (refset selection), so the threaded generator is unused here.
    del generator
    covs = sig2 * torch.eye(2, dtype=dtype).expand(2, -1, -1).contiguous()
    mix = MixtureSameFamily(
        Categorical(torch.ones(2, dtype=dtype)),
        MultivariateNormal(means, covs, validate_args=False),
    )
    return mix.sample(torch.Size((n,)))


gammas = torch.logspace(-10, 10, 40, base=2, dtype=dtype)


def _run_pipeline() -> torch.Tensor:
    """Tiny end-to-end run: sample refs, FPS-select, score a batch. Returns f(X)."""
    p = Density(log_pX, log_pY, sample_fn=sampler, d=2)
    dist = GlobalIEMDistance(p, gammas, num_eps=8, seed=123)
    sel = FPSRefs(p, distance=dist, pool_size=16, seed=0)
    sel.select(R=4)
    X = p.sample(3, seed=1)
    return sel.expected_distance(X, R=4)


# ---------------------------------------------------------------------------
# default_device / set_default_device
# ---------------------------------------------------------------------------

def test_default_device_auto():
    # On a machine without CUDA this is CPU; on a CUDA host it is cuda.
    dev = default_device()
    expected = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert dev.type == expected.type


def test_set_default_device_override_and_reset():
    try:
        set_default_device("cpu")
        assert default_device() == torch.device("cpu")
        p = Density(log_pX, log_pY, sample_fn=sampler, d=2)
        assert p.sample(2, seed=0).device.type == "cpu"
    finally:
        set_default_device(None)   # re-enable auto so other tests are unaffected


def test_set_default_dtype_override_and_reset():
    assert default_dtype() == torch.float64           # float64 is the library default
    try:
        set_default_dtype(torch.float32)
        assert default_dtype() == torch.float32
        p = Density(log_pX, log_pY, sample_fn=sampler, d=2)
        assert p.sample(2, seed=0).dtype == torch.float32   # Density.sample stamps the default dtype
    finally:
        set_default_dtype(None)    # re-enable the float64 default so other tests are unaffected
    assert default_dtype() == torch.float64


# ---------------------------------------------------------------------------
# Pipeline places tensors on the resolved device
# ---------------------------------------------------------------------------

def test_pipeline_output_on_default_device():
    f = _run_pipeline()
    assert f.shape == (3,)
    assert f.device.type == default_device().type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_pipeline_runs_on_cuda():
    try:
        set_default_device("cuda")
        f = _run_pipeline()                  # exercises FPS index-tensor + scipy .cpu() paths on GPU
        assert f.device.type == "cuda"
        assert torch.isfinite(f).all()
        # auto-R sweep (scipy weightedtau) must also survive GPU tensors
        p = Density(log_pX, log_pY, sample_fn=sampler, d=2)
        dist = GlobalIEMDistance(p, gammas, num_eps=8, seed=123)
        sel = RandomRefs(p, distance=dist, seed=0, auto_r_grid=(1, 2, 4), auto_r_draws=2, fallback="best")
        refs = sel.select()                  # R=None -> auto-R
        assert refs.device.type == "cuda"
    finally:
        set_default_device(None)
