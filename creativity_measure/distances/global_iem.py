#
# Global IEM pairwise distance (Ohayon et al., ICLR 2026, Def. 1, f = identity):
#   D_IEM^2(x1,x2) = ∫_0^∞ E_W[ || ∇log p_Yg(g x1 + W) - ∇log p_Yg(g x2 + W) ||^2 ] dg
#   then D_IEM = sqrt(D_IEM^2).
# Direct transcription: differentiates the marginal log-density log p_Yg w.r.t. y at each
# of the two noisy points (shared Brownian path W), with no conditional-score term.

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import ScoreFn
from creativity_measure.density import Density


def marginal_score(
    y: Float[Tensor, "B d"],
    gamma: Float[Tensor, ""],
    density: Density,
) -> Float[Tensor, "B d"]:
    """∇_y log p_Yg(y) — the score of the (blurred) marginal density at y."""
    y = y.detach().clone().requires_grad_(True)   # isolate y so we can take d/dy at this point
    # .sum() lets one grad call return per-row gradients (rows are independent)
    g = torch.autograd.grad(density.log_p_Y(y, gamma).sum(), y)[0]
    return g.detach()


def iem_sq_increments_one_to_many(
    x_ref: Float[Tensor, "1 d"],
    X: Float[Tensor, "G d"],
    W: Float[Tensor, "N_gamma N_eps 1 d"],
    gammas: Float[Tensor, "N_gamma"],
    density: Density,
) -> Float[Tensor, "N_gamma_minus_1 N_eps G"]:
    """
    IEM^2 increments (Def. 1) for one reference vs the whole batch X, shared Brownian path W.
    x_ref: (1, d), X: (G, d), W: (N_gamma, N_eps, 1, d)
    Returns:
        score_diff_sq_increments: (N_gamma-1, N_eps, G)  summed over gamma -> IEM^2
    """
    num_gamma, num_eps = W.shape[0], W.shape[1]
    G = X.shape[0]
    d = X.shape[1]
    dgamma = gammas[1:] - gammas[:-1]   # gamma step sizes (integration widths)

    # Noisy observation paths y_g = g*x + W, shared W across both points
    y1_path = gammas.view(-1, 1, 1, 1) * x_ref.view(1, 1, 1, d) + W     # (N_gamma, N_eps, 1, d)
    y2_path = gammas.view(-1, 1, 1, 1) * X.view(1, 1, G, d) + W         # (N_gamma, N_eps, G, d)

    increment_list = []
    for i in range(num_gamma - 1):  # one step per integration interval
        gamma = gammas[i]
        y1 = y1_path[i].reshape(num_eps, d)
        y2 = y2_path[i].reshape(num_eps * G, d)
        s1 = marginal_score(y1, gamma, density).view(num_eps, 1, d)   # ∇log p_Yg(g x1 + W)
        s2 = marginal_score(y2, gamma, density).view(num_eps, G, d)   # ∇log p_Yg(g x2 + W)
        score_diff = s1 - s2                                          # (N_eps, G, d)
        increment = score_diff.pow(2).sum(-1) * dgamma[i]             # ||score diff||^2 dg  (N_eps, G)
        increment_list.append(increment)

    return torch.stack(increment_list, dim=0)   # (N_gamma-1, N_eps, G)


class GlobalIEMDistance:
    """
    Global IEM distance D_IEM(x, x') (Def. 1, f = identity), direct marginal-score formulation.
    Builds one Brownian path bank and evaluates every reference against the whole X batch.

    Args:
        density:  Density (must expose log_p_Y)
        gammas:   integration grid, e.g. logspace(-10, 10, 200, base=2)
        num_eps:  Brownian path samples (variance reduction)
        seed:     RNG seed for the Brownian path bank
        score_fn: Reserved for future use; when supplied, will replace autograd
                  through `density.log_p_Y` with a learned score. Not yet consumed.
    """

    def __init__(
        self,
        density: Density,
        gammas: Float[Tensor, "N_gamma"],
        num_eps: int = 50,
        seed: int = 123,
        verbose: bool = False,
        score_fn: ScoreFn | None = None,
    ):
        self.density = density
        self.gammas = gammas
        self.num_eps = num_eps
        self.seed = seed
        self.verbose = verbose
        self.score_fn = score_fn   # reserved; not yet consumed

    def _brownian(
        self,
        d: int,
        device: torch.device,
        dtype: torch.dtype,
        gammas: Float[Tensor, "N_gamma"],
    ) -> Float[Tensor, "N_gamma N_eps 1 d"]:
        """Simulate num_eps Wiener paths W on the gamma grid."""
        num_gamma = gammas.shape[0]
        dgamma = gammas[1:] - gammas[:-1]
        generator = torch.Generator(device=device).manual_seed(self.seed)
        # Brownian increments: dW ~ N(0, dgamma * I), so scale standard normals by sqrt(dgamma)
        dW = torch.randn(num_gamma - 1, self.num_eps, 1, d,
                         device=device, dtype=dtype, generator=generator) * dgamma.sqrt().view(-1, 1, 1, 1)
        W = torch.zeros(num_gamma, self.num_eps, 1, d, device=device, dtype=dtype)
        W[1:] = torch.cumsum(dW, dim=0)     # W[0]=0; cumulative sum of increments
        return W

    def pairwise(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
    ) -> Float[Tensor, "B R"]:
        """Returns D_IEM(X[b], x_refs[r]) for all b, r."""
        device, dtype = X.device, X.dtype
        d = X.shape[1]
        gammas = self.gammas.to(device=device, dtype=dtype)
        W = self._brownian(d, device, dtype, gammas)   # built once, reused across all references
        num_refs = x_refs.shape[0]
        cols = []
        for ref in range(num_refs):
            increments = iem_sq_increments_one_to_many(
                x_refs[ref:ref+1], X, W, gammas, self.density)
            iem_sq = increments.sum(0)                      # (N_eps, G): ∫dg  -> IEM^2 per path
            cols.append(iem_sq.mean(0).clamp_min(0).sqrt()) # (B,): E_W -> sqrt -> D_IEM
            if self.verbose and (ref + 1) % max(1, num_refs // 4) == 0:
                print(f'  ref {ref+1}/{num_refs}')
        return torch.stack(cols, dim=1)                     # (B, num_refs)
