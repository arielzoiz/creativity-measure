# creativity_measure/distances/local_iem.py
import torch
from torch.func import vmap, jacrev, jacfwd
from creativity_measure.density import Density


def compute_G(X, log_p_Y_scalar, gammas, num_noises=50, seed=123):
    """
    Local IEM metric tensor G(x) (paper Thm. 2, Eq. 5). Was compute_local_M
    in the notebook; generalized to d dims and a passed-in scalar log p_Y.

    Args:
        X:             (B, d)
        log_p_Y_scalar: callable (y: (d,), gamma scalar) -> scalar
        gammas:        (N_gamma,)  e.g. logspace(-4, 4, 200, base=2)
        num_noises:    MC samples for E[...] over w_gamma
        seed:          RNG seed

    Returns:
        G: (B, d, d) positive semi-definite metric tensor per point.
    """
    torch.manual_seed(seed)
    hess_fn = jacfwd(jacrev(log_p_Y_scalar, argnums=0), argnums=0)

    X = X.to(dtype=gammas.dtype)
    dgam = gammas[1:] - gammas[:-1]
    B, d = X.shape
    num_gamma = gammas.shape[0]

    eps = torch.randn((num_noises, B, num_gamma, d), device=X.device, dtype=X.dtype)
    ggrid = gammas.view(1, 1, num_gamma, 1)
    y = ggrid * X.view(1, B, 1, d) + ggrid.sqrt() * eps      # (num_noises, B, N_gamma, d)
    y_flat = y.reshape(-1, d)
    g_flat = ggrid.expand(num_noises, B, num_gamma, 1).reshape(-1)

    H = vmap(hess_fn)(y_flat, g_flat).view(num_noises * B * num_gamma, d, d)
    H2 = torch.bmm(H, H).view(num_noises, B, num_gamma, d, d).mean(0)   # (B, N_gamma, d, d)
    weighted = H2 * (gammas ** 2).view(1, num_gamma, 1, 1)
    G = (weighted[:, :-1] * dgam.view(1, num_gamma - 1, 1, 1)).sum(1)   # (B, d, d)
    return G


class LocalIEMDistance:
    """
    Local IEM distance:  D_local(x, x') = sqrt( (x-x')^T G(x) (x-x') ).

    G(x) is computed once per evaluation point x; the quadratic form is then
    cheap against every reference x'. Valid as a local (Taylor) metric; for
    far references it is the locally-adaptive-Mahalanobis extrapolation.

    Note: uses vmap(jacfwd(jacrev(log_p_Y))). If log_p_Y uses ops without a
    vmap rule (e.g. torch.special.log_ndtr), swap compute_G for the batched-
    autograd variant in iem_creativity.ipynb section 6.C.
    """

    def __init__(self, density: Density, gammas, num_noises=50, seed=123):
        self.density = density
        self.gammas = gammas
        self.num_noises = num_noises
        self.seed = seed

    def pairwise(self, X, x_refs):
        """X: (B, d), x_refs: (R, d) -> (B, R)."""
        G = compute_G(X, self.density.log_p_Y_scalar, self.gammas,
                      num_noises=self.num_noises, seed=self.seed)     # (B, d, d)
        diff = X.unsqueeze(1) - x_refs.unsqueeze(0)                   # (B, R, d)
        Gdiff = torch.einsum('bde,bre->brd', G, diff)                 # (B, R, d)
        qf = (diff * Gdiff).sum(-1).clamp_min(0.0)                    # (B, R)
        return qf.sqrt()
