"""
Full tilt-calculation flow run end-to-end on CPU and on GPU.

The flow: Density -> GlobalIEMDistance -> tilted_log_density over a grid -> grid_normalize,
checking the normalized q integrates to 1. The density's constants are built on the target
(device, dtype) so the whole pipeline is device-portable.

GPU policy: CUDA uses float64 (full precision); Apple MPS uses float32 (MPS has no float64) and
is reached via the explicit set_default_device("mps") opt-in. Skips if no GPU is present.
"""
import math

import pytest
import torch

from creativity_measure import (
    Density,
    GlobalIEMDistance,
    tilted_log_density,
    grid_normalize,
    make_grid,
    set_default_device,
    Reward,
)

# Ring GMM with a hole (same shape as the demo pipeline), as device/dtype-agnostic constants.
N_TOTAL, HOLE_IDX, RADIUS, SIGMA = 12, 0, 4.0, 0.3
S2 = SIGMA ** 2
_angles = 2 * math.pi * torch.arange(N_TOTAL, dtype=torch.float64) / N_TOTAL
_all_means = torch.stack([RADIUS * torch.cos(_angles), RADIUS * torch.sin(_angles)], dim=-1)
_mask = torch.ones(N_TOTAL, dtype=torch.bool)
_mask[HOLE_IDX] = False
MEANS_CPU = _all_means[_mask]            # (K, 2) reference means on CPU/float64
K = MEANS_CPU.shape[0]
_logK = math.log(K)

GRID_N, LAM = 12, 5.0


def build_density(device: torch.device, dtype: torch.dtype) -> Density:
    """A ring-GMM Density whose constants live on (device, dtype) so log_p_X/log_p_Y/sample match the grid."""
    means = MEANS_CPU.to(device=device, dtype=dtype)

    def log_pX(x):
        diff = x.unsqueeze(-2) - means
        quad = diff.pow(2).sum(-1) / S2
        logcomp = -0.5 * (quad + 2 * math.log(2 * math.pi * S2))
        return torch.logsumexp(logcomp, dim=-1) - _logK

    def log_pY(y, gamma):
        var = gamma ** 2 * S2 + gamma
        diff = y.unsqueeze(-2) - gamma * means
        quad = diff.pow(2).sum(-1) / var
        logcomp = -0.5 * (quad + 2 * torch.log(2 * math.pi * var))
        return torch.logsumexp(logcomp, dim=-1) - _logK

    def sample(n, generator=None):
        comp = torch.randint(K, (n,), device=device, generator=generator)
        return means[comp] + SIGMA * torch.randn(n, 2, device=device, dtype=dtype, generator=generator)

    return Density(log_pX, log_pY, sample_fn=sample, d=2, device=device)


def run_tilt_flow(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Full flow on (device, dtype). Returns the normalized-density integral (should be ~1)."""
    p = build_density(device, dtype)
    grid_points, _, _, cell_area = make_grid((-6, 6), (-6, 6), grid_n=GRID_N, device=device, dtype=dtype)
    x_refs = p.sample(6, seed=0)                                  # lands on `device` via Density.sample
    # logspace is built on CPU then moved: aten::logspace has no MPS kernel (a known MPS op gap).
    gammas = torch.logspace(-10, 10, 20, base=2, dtype=dtype).to(device)

    D = GlobalIEMDistance(p, gammas, num_eps=4)
    log_q_un = tilted_log_density(grid_points, p, Reward(D, x_refs), lam=LAM)

    assert log_q_un.shape == (GRID_N * GRID_N,)
    assert log_q_un.device.type == device.type, f"expected {device.type}, got {log_q_un.device.type}"
    assert log_q_un.isfinite().all(), "log_q_un has non-finite values"

    _, q, _ = grid_normalize(log_q_un, cell_area)
    assert (q >= 0).all()
    return q.sum() * cell_area


def _pick_gpu() -> tuple[torch.device, torch.dtype] | None:
    """CUDA(float64) preferred; else MPS(float32, opt-in); else None."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float64
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return None


def test_tilt_flow_cpu():
    integral = run_tilt_flow(torch.device("cpu"), torch.float64)
    assert torch.allclose(integral.cpu(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6), \
        f"CPU integral should be ~1, got {integral.item()}"


@pytest.mark.skipif(_pick_gpu() is None, reason="no GPU (CUDA or MPS) available")
def test_tilt_flow_gpu():
    gpu = _pick_gpu()
    assert gpu is not None
    device, dtype = gpu
    atol = 1e-6 if dtype == torch.float64 else 1e-4   # float32 (MPS) accumulates more rounding
    try:
        if device.type != "cuda":
            set_default_device(device)                # opt into MPS for the resolver
        integral = run_tilt_flow(device, dtype)
    finally:
        set_default_device(None)                       # restore auto-selection
    assert torch.allclose(integral.cpu().double(), torch.tensor(1.0, dtype=torch.float64), atol=atol), \
        f"{device.type} integral should be ~1, got {integral.item()}"
