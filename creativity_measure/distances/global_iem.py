# creativity_measure/distances/global_iem.py
#
# Global IEM pairwise distance (Fiquet et al., ICLR 2026, Defs. 1 & 2):
#   D_IEM_f^2(x1,x2) = integral E_W[ f'(z_g)^2 * || s(Y1,x1,g) - s(Y2,x2,g) ||^2 ] dg
#   s(y,x,g) = grad_y log p(y|x,g) - grad_y log p_Y(y;g),  shared path W.
#   f = identity recovers the notebook's standard IEM (Def. 1).
#
# Ported from iem_creativity.ipynb section 3 (cell 7); generalized to a Density,
# an explicit x_refs (replacing sample_from_p_obs), and a pluggable f.
# Activations adapted from information_estimation_metric.py (lines 28-39).

import torch
from creativity_measure.density import Density
from creativity_measure.distances.utils import log_p_Y_given_X


# ----------------------------------------------------------------------
# Pluggable IEM activations. Signature: (z_gamma, dqv, alpha) -> (..., G),
# summing the integrand over the gamma axis (dim 0).
# ----------------------------------------------------------------------

def f_identity(z_gamma, dqv, alpha):
    """Standard IEM, f = identity (f'(z) = 1). Integrand = dqv. (Notebook default.)"""
    return dqv.sum(0)


def f_square(z_gamma, dqv, alpha):
    """Quadratic IEM, f(z) = z^2 (f'(z) = 2z). Integrand = (2*alpha*z_gamma)^2 * dqv."""
    return ((2 * alpha * z_gamma).pow(2) * dqv).sum(0)


def score_diff_y(y, x, gamma, density: Density):
    """s(y, x, gamma). y: (B, d), x: (B, d), gamma scalar -> (B, d)."""
    y = y.detach().clone().requires_grad_(True)
    g1 = torch.autograd.grad(log_p_Y_given_X(y, x, gamma).sum(), y)[0]
    g2 = torch.autograd.grad(density.log_p_Y(y, gamma).sum(), y)[0]
    return (g1 - g2).detach()


def sde_elements_one_to_many(x_ref, X, W, dW, gammas, density: Density):
    """
    Generalizes the notebook's D_IEM_sq_one_to_many: instead of only the
    quadratic-variation sum, returns the per-gamma SDE elements so a pluggable
    f can be applied.

    x_ref: (1, d), X: (G, d), W: (N_gamma, N_eps, 1, d), dW: (N_gamma-1, N_eps, 1, d)
    Returns:
        z_gamma: (N_gamma-1, N_eps, G)   accumulated log-ratio process
        dqv:     (N_gamma-1, N_eps, G)   quadratic-variation increments
    """
    num_gamma, num_eps = W.shape[0], W.shape[1]
    G = X.shape[0]
    d = X.shape[1]
    dgam = gammas[1:] - gammas[:-1]

    y1_path = gammas.view(-1, 1, 1, 1) * x_ref.view(1, 1, 1, d) + W
    y2_path = gammas.view(-1, 1, 1, 1) * X.view(1, 1, G, d) + W
    x_ref_b = x_ref.view(1, d).expand(num_eps, d).contiguous()
    X_b     = X.view(1, G, d).expand(num_eps, G, d).reshape(num_eps * G, d).contiguous()

    dqv_list, dz_list = [], []
    for i in range(num_gamma - 1):
        gamma = gammas[i]
        y1 = y1_path[i].reshape(num_eps, d)
        y2 = y2_path[i].reshape(num_eps * G, d)
        s1 = score_diff_y(y1, x_ref_b, gamma, density).view(num_eps, 1, d)
        s2 = score_diff_y(y2, X_b, gamma, density).view(num_eps, G, d)
        e1, e2 = -s1, -s2                                  # eps_i = -score_diff (repo convention)
        diff = e1 - e2                                     # (N_eps, G, d)
        dqv_i = diff.pow(2).sum(-1) * dgam[i]              # (N_eps, G)
        drift_i = 0.5 * (e1.pow(2).sum(-1) - e2.pow(2).sum(-1)) * dgam[i]  # (N_eps, G)
        dw_i = dW[i].reshape(num_eps, 1, d)
        stoch_i = (diff * dw_i).sum(-1)                    # (N_eps, G)
        dqv_list.append(dqv_i)
        dz_list.append(drift_i + stoch_i)

    dqv = torch.stack(dqv_list, dim=0)                     # (N_gamma-1, N_eps, G)
    dz = torch.stack(dz_list, dim=0)
    z = torch.cumsum(dz, dim=0)
    z = torch.cat([torch.zeros_like(z[:1]), z[:-1]], dim=0)  # z_0 = 0, left-aligned
    return z, dqv


class GlobalIEMDistance:
    """
    Global IEM distance D_IEM_f(x, x') via shared-path score differences.
    Builds one Brownian path bank and evaluates every reference against the
    whole X batch, applying the pluggable activation f.

    Args:
        density:  Density (must expose log_p_Y)
        gammas:   integration grid, e.g. logspace(-10, 10, 200, base=2)
        num_eps:  Brownian path samples (variance reduction)
        f:        activation, f_identity (default, = notebook standard IEM)
                  or f_square. Signature (z_gamma, dqv, alpha) -> (N_eps, G).
        seed:     RNG seed for the Brownian path bank
    """

    def __init__(self, density: Density, gammas, num_eps=50, f=f_identity,
                 seed=123, verbose=False):
        self.density = density
        self.gammas = gammas
        self.num_eps = num_eps
        self.f = f
        self.seed = seed
        self.verbose = verbose

    def _brownian(self, d, device, dtype):
        num_gamma = self.gammas.shape[0]
        dgam = self.gammas[1:] - self.gammas[:-1]
        gen = torch.Generator(device=device).manual_seed(self.seed)
        dW = torch.randn(num_gamma - 1, self.num_eps, 1, d,
                         device=device, dtype=dtype, generator=gen) * dgam.sqrt().view(-1, 1, 1, 1)
        W = torch.zeros(num_gamma, self.num_eps, 1, d, device=device, dtype=dtype)
        W[1:] = torch.cumsum(dW, dim=0)
        return W, dW

    def pairwise(self, X, x_refs):
        """X: (B, d), x_refs: (R, d) -> (B, R)  with D_IEM_f(X[b], x_refs[r])."""
        device, dtype = X.device, X.dtype
        d = X.shape[1]
        alpha = 1.0 / d
        W, dW = self._brownian(d, device, dtype)
        R = x_refs.shape[0]
        cols = []
        for r in range(R):
            z, dqv = sde_elements_one_to_many(
                x_refs[r:r+1], X, W, dW, self.gammas, self.density)
            iem_sq = self.f(z, dqv, alpha)                  # (N_eps, G)
            cols.append(iem_sq.mean(0).clamp_min(0).sqrt())  # (B,) = D_IEM_f(X, x_refs[r])
            if self.verbose and (r + 1) % max(1, R // 4) == 0:
                print(f'  ref {r+1}/{R}')
        return torch.stack(cols, dim=1)                     # (B, R)
