from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor
from torch.func import jacfwd, jacrev, vmap

from creativity_measure.density import Density


def _compute_G_vmap(
    X: Float[Tensor, "B d"],
    log_p_Y_scalar: Callable[[Float[Tensor, "d"], Float[Tensor, ""]], Float[Tensor, ""]],
    gammas: Float[Tensor, "N_gamma"],
    num_noises: int = 50,
    chunk_size: int = 64,
    seed: int = 123,
) -> Float[Tensor, "B d d"]:
    """
    Local IEM metric tensor G(x) (paper Thm. 2, Eq. 5).

    Args:
        log_p_Y_scalar: callable (y: (d,), gamma scalar) -> scalar
        num_noises:     MC samples for E[...] over w_gamma
        chunk_size:     internal batch size over points, to cap peak memory.
                        Result is identical regardless of chunk_size.
    Returns:
        G: positive semi-definite metric tensor per point.
    """
    hess_fn = jacfwd(jacrev(log_p_Y_scalar, argnums=0), argnums=0)

    gammas = gammas.to(device=X.device, dtype=X.dtype)
    dgam = gammas[1:] - gammas[:-1]
    B, d = X.shape
    num_gamma = gammas.shape[0]

    Gs = []
    for start in range(0, B, chunk_size):
        X_chunk = X[start:start + chunk_size]
        B_chunk = X_chunk.shape[0]

        gen = torch.Generator(device=X.device).manual_seed(seed)
        eps = torch.randn((num_noises, B_chunk, num_gamma, d), device=X.device, dtype=X.dtype, generator=gen)
        ggrid = gammas.view(1, 1, num_gamma, 1)
        y = ggrid * X_chunk.view(1, B_chunk, 1, d) + ggrid.sqrt() * eps      # (num_noises, B_chunk, N_gamma, d)
        y_flat = y.reshape(-1, d)
        g_flat = ggrid.expand(num_noises, B_chunk, num_gamma, 1).reshape(-1)

        H = vmap(hess_fn)(y_flat, g_flat).view(num_noises * B_chunk * num_gamma, d, d)
        H2 = torch.bmm(H, H).view(num_noises, B_chunk, num_gamma, d, d).mean(0)   # (B_chunk, N_gamma, d, d)
        weighted = H2 * (gammas ** 2).view(1, num_gamma, 1, 1)
        G_chunk = (weighted[:, :-1] * dgam.view(1, num_gamma - 1, 1, 1)).sum(1)   # (B_chunk, d, d)
        Gs.append(G_chunk)
    G = torch.cat(Gs, dim=0)   # (B, d, d)
    return G


def _compute_G_vmap_autograd(
    X: Float[Tensor, "B d"],
    density: Density,
    gammas: Float[Tensor, "N_gamma"],
    num_noises: int = 50,
    chunk_size: int = 64,
    seed: int = 123,
) -> Float[Tensor, "B d d"]:
    """
    Local IEM metric tensor G(x) via batched autograd — no vmap.
    Use when log_p_Y contains ops without a vmap rule (e.g. torch.special.log_ndtr).
    Slower than _compute_G_vmap but compatible with all differentiable ops.
    """
    gammas = gammas.to(device=X.device, dtype=X.dtype)
    dgam = gammas[1:] - gammas[:-1]
    B, d = X.shape
    num_gamma = gammas.shape[0]

    Gs = []
    for start in range(0, B, chunk_size):
        X_chunk = X[start:start + chunk_size]
        B_chunk = X_chunk.shape[0]

        gen = torch.Generator(device=X.device).manual_seed(seed)
        eps = torch.randn((num_noises, B_chunk, num_gamma, d), device=X.device, dtype=X.dtype, generator=gen)
        ggrid = gammas.view(1, 1, num_gamma, 1)
        y = ggrid * X_chunk.view(1, B_chunk, 1, d) + ggrid.sqrt() * eps   # (num_noises, B_chunk, N_gamma, d)

        G_chunk = torch.zeros(B_chunk, d, d, device=X.device, dtype=X.dtype)
        for gi in range(num_gamma - 1):
            gamma = gammas[gi]
            y_gi = y[:, :, gi, :].reshape(num_noises * B_chunk, d)   # all noise/point pairs for this gamma
            with torch.enable_grad():
                y_in = y_gi.detach().requires_grad_(True)
                lp = density.log_p_Y(y_in, gamma)                                         # (num_noises * B_chunk,)
                score = torch.autograd.grad(lp.sum(), y_in, create_graph=True)[0]         # (num_noises * B_chunk, d)
                rows = [
                    torch.autograd.grad(score[:, d_idx].sum(), y_in,
                                        retain_graph=(d_idx < d - 1))[0]
                    for d_idx in range(d)
                ]
            H = torch.stack(rows, dim=1).detach()                                         # (num_noises * B_chunk, d, d)
            H2 = torch.bmm(H, H).view(num_noises, B_chunk, d, d).mean(0)                 # (B_chunk, d, d)
            G_chunk += H2 * (gamma ** 2) * dgam[gi]

        Gs.append(G_chunk)
    return torch.cat(Gs, dim=0)   # (B, d, d)


def compute_G(
    X: Float[Tensor, "B d"],
    density: Density,
    gammas: Float[Tensor, "N_gamma"],
    num_noises: int = 50,
    chunk_size: int = 64,
    seed: int = 123,
) -> Float[Tensor, "B d d"]:
    """
    Tries vmap-based _compute_G_vmap first; falls back to _compute_G_vmap_autograd if vmap
    raises (e.g. when log_p_Y uses ops without a vmap batching rule).
    """
    log_p_Y_scalar = lambda y, g: density.log_p_Y(y.unsqueeze(0), g).squeeze(0)
    try:
        return _compute_G_vmap(X, log_p_Y_scalar, gammas,
                         num_noises=num_noises, chunk_size=chunk_size, seed=seed)
    except Exception:
        return _compute_G_vmap_autograd(X, density, gammas,
                                  num_noises=num_noises, chunk_size=chunk_size, seed=seed)


class LocalIEMDistance:
    """
    Local IEM distance:  D_local(x, x') = sqrt( (x-x')^T G(x) (x-x') ).

    G(x) is computed once per evaluation point x; the quadratic form is then
    cheap against every reference x'. Valid as a local (Taylor) metric; for
    far references it is the locally-adaptive-Mahalanobis extrapolation.

    Uses compute_G: tries vmap(jacfwd(jacrev(log_p_Y))) and automatically
    falls back to batched autograd if log_p_Y contains ops without a vmap rule
    (e.g. torch.special.log_ndtr).
    """

    def __init__(
        self,
        density: Density,
        gammas: Float[Tensor, "N_gamma"],
        num_noises: int = 50,
        seed: int = 123,
    ):
        self.density = density
        self.gammas = gammas
        self.num_noises = num_noises
        self.seed = seed

    def pairwise(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
    ) -> Float[Tensor, "B R"]:
        G = compute_G(X, self.density, self.gammas,
                           num_noises=self.num_noises, seed=self.seed)   # (B, d, d)
        diff = X.unsqueeze(1) - x_refs.unsqueeze(0)                      # (B, R, d)
        Gdiff = torch.einsum('bde,bre->brd', G, diff)                    # (B, R, d)
        qf = (diff * Gdiff).sum(-1).clamp_min(0.0)                       # (B, R)
        return qf.sqrt()
