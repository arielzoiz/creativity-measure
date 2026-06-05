#
# Generalized / conditional global IEM pairwise distance (Ohayon et al., ICLR 2026).
# Generalized-f IEM (Def. 2):
#   D_IEM_f^2(x1,x2) = ∫_0^∞ E_W[ f'(alpha * Z_g)^2 || ∇log p_Yg(g x1 + W) - ∇log p_Yg(g x2 + W) ||^2 ] dg
#   Z_g(x1,x2) = log( p_Yg(g x1 + W) / p_Yg(g x2 + W) )      (Def. 2, Eq. 9; log-likelihood-ratio process)
#   alpha = 1/d is a normalization of f's argument (used in IEM's repo, not in the paper; keeps it O(1) for d-dim signals).
#
# Implementation: Z_g is reconstructed along the shared path W from its Itô increments
#   Z_g = ∫_0^g <s1 - s2, dW> + 1/2 * ∫_0^g (||s1||^2 - ||s2||^2) dg' = (stochastic part) + (drift part).
# using the per-point conditional score difference
#   s(y,x,g) = grad_y log p(y|x,g) - grad_y log p_Y(y;g)              (= denoising error x - E[X|y], Tweedie).
# Under the shared path W, s1 - s2 = ∇log p_Yg(g x1 + W) - ∇log p_Yg(g x2 + W) (the Def. 1/2 score diff).
# Sign note: like IEM's repo, this Z is -z_g of Eq. 9 (both drift and stochastic terms signs flipped).
# Harmless for identity/squared f (Z unused / squared away). For a learned f (yet to be implemented):
# IEM's repo feeds it |z| (sign-invariant), and an f trained and evaluated on this Z gives the same
# metric; only a foreign signed-z f (trained on +z, reused without retraining) would mismatch.
#
# With f = identity (f' ≡ 1) this recovers Def. 1, the plain IEM:
#   D_IEM^2 = ∫_0^∞ E_W[ || ∇log p_Yg(g x1 + W) - ∇log p_Yg(g x2 + W) ||^2 ] dg
# also available, without forming Z, as the direct marginal-score formulation in global_iem.py.

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import ScoreFn
from creativity_measure.density import Density
from creativity_measure.distances.base import Distance
from creativity_measure.distances.utils import log_p_Y_given_X, simulate_brownian


@dataclass(frozen=True)
class FSpec:
    """
    A choice of f for the generalized global IEM.
    Only f_prime enters the integrand: integrand = f'(alpha * Z)^2 ||s1 - s2||^2 dg);
    f is kept for clarity.
    """
    entry_name: str
    f: Callable[[Tensor], Tensor]        # z -> f(z)
    f_prime: Callable[[Tensor], Tensor]  # z -> f'(z)


class IEMFType(Enum):
    """Available f's. Wrapped in FSpec so Enum stores values."""
    IDENTITY = FSpec("identity", lambda z: z, lambda z: torch.ones_like(z))
    SQUARED = FSpec("squared", lambda z: z.pow(2), lambda z: 2 * z)


def score_diff_y(
    y: Float[Tensor, "B d"],
    x: Float[Tensor, "B d"],
    gamma: Float[Tensor, ""],
    density: Density,
) -> Float[Tensor, "B d"]:
    """s(y, x, g) = ∇_y log p(y|x,g) - ∇_y log p_yg(y)   (= denoising error x - E[X|y], Tweedie)."""
    if density.log_p_Y is None:
        raise RuntimeError("Density must provide log_p_Y; score_fn support is not yet implemented")
    y = y.detach().clone().requires_grad_(True) # isolate y so we can take d/dy at this point
    # .sum() lets one grad call return per-row gradients (rows are independent)
    g1 = torch.autograd.grad(log_p_Y_given_X(y, x, gamma).sum(), y)[0]
    g2 = torch.autograd.grad(density.log_p_Y(y, gamma).sum(), y)[0]
    return (g1 - g2).detach()


def iem_sq_increments_one_to_many(
    x_ref: Float[Tensor, "1 d"],
    X: Float[Tensor, "G d"],
    W: Float[Tensor, "N_gamma N_eps 1 d"],
    gammas: Float[Tensor, "N_gamma"],
    density: Density,
) -> tuple[Float[Tensor, "N_gamma_minus_1 N_eps G"], Float[Tensor, "N_gamma_minus_1 N_eps G"]]:
    """
    Per-gamma increments of the log-ratio process Z, for one reference vs the whole
    batch X, sharing one Brownian path bank W.
    x_ref: (1, d), X: (G, d), W: (N_gamma, N_eps, 1, d)

    Z evolves (Itô, gamma as time): dZ = 0.5(||s1||^2 - ||s2||^2) dg + <s1 - s2, dW>,
    where s_i = x_i - E[X | y_i] is the per-point denoising error (= conditional score diff).

    Returns:
        quad_var_increments: (N_gamma-1, N_eps, G)  ||s1 - s2||^2 dg  (squared score-diff increment)
        z_increments:        (N_gamma-1, N_eps, G)  left-point value of Z at each interval (Z[0]=0)
    The general-f IEM^2 is sum_g f'(alpha * z_increments)^2 * quad_var_increments.
    """
    num_gamma, num_eps = W.shape[0], W.shape[1]
    G = X.shape[0]
    d = X.shape[1]
    dgamma = gammas[1:] - gammas[:-1]   # gamma step sizes (integration widths)

    # Noisy observation paths y_g = g*x + W, shared W (so conditional scores cancel -> Def. 1)
    y1_path = gammas.view(-1, 1, 1, 1) * x_ref.view(1, 1, 1, d) + W     # (N_gamma, N_eps, 1, d)
    y2_path = gammas.view(-1, 1, 1, 1) * X.view(1, 1, G, d) + W         # (N_gamma, N_eps, G, d)
    x_ref_b = x_ref.view(1, d).expand(num_eps, d).contiguous()
    X_b     = X.view(1, G, d).expand(num_eps, G, d).reshape(num_eps * G, d).contiguous()

    quad_var_increment_list = []
    z_increment_list = []
    z = torch.zeros(num_eps, G, device=X.device, dtype=X.dtype)  # left-point value of Z, matches Itô convention
    for i in range(num_gamma - 1):  # one step per integration interval
        gamma = gammas[i]
        y1 = y1_path[i].reshape(num_eps, d)
        y2 = y2_path[i].reshape(num_eps * G, d)
        s1 = score_diff_y(y1, x_ref_b, gamma, density).view(num_eps, 1, d)
        s2 = score_diff_y(y2, X_b, gamma, density).view(num_eps, G, d)
        marginal_score_diff = s1 - s2       # (N_eps, G, d); = marginal score diff (Def. 1)
        quad_var_increment = marginal_score_diff.pow(2).sum(-1) * dgamma[i]  # ||diff||^2 dg  (N_eps, G)
        quad_var_increment_list.append(quad_var_increment)

        # Log-ratio process Z: record its value, then advance by one Itô step.
        z_increment_list.append(z)
        dW = W[i + 1] - W[i]                                  # (N_eps, 1, d); Brownian increment
        drift = 0.5 * (s1.pow(2).sum(-1) - s2.pow(2).sum(-1)) * dgamma[i]  # (N_eps, G)
        stoch = (marginal_score_diff * dW).sum(-1)           # <s1 - s2, dW>  (N_eps, G)
        z = z + drift + stoch

    quad_var_increments = torch.stack(quad_var_increment_list, dim=0)   # (N_gamma-1, N_eps, G)
    z_increments = torch.stack(z_increment_list, dim=0)                 # (N_gamma-1, N_eps, G)
    return quad_var_increments, z_increments


class GlobalConditionalIEMDistance(Distance):
    """
    Global IEM distance D_IEM(x, x') via conditional score differences (slower; see
    GlobalIEMDistance for the faster direct marginal-score formulation).

    Args:
        density:  Density (must expose log_p_Y)
        gammas:   integration grid, e.g. logspace(-10, 10, 200, base=2)
        num_eps:  Brownian path samples (variance reduction)
        seed:     RNG seed for the Brownian path bank
        f_type:   choice of f for the general-f IEM (default IDENTITY)
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
        f_type: IEMFType = IEMFType.IDENTITY,
        score_fn: ScoreFn | None = None,
    ):
        self.density = density
        self.gammas = gammas
        self.num_eps = num_eps
        self.seed = seed
        self.verbose = verbose
        self.f_type = f_type
        # TODO: when score_fn is provided, pass it through pairwise ->
        #       iem_sq_increments_one_to_many -> score_diff_y and use it
        #       instead of autograd, to support pre-learned score models.
        self.score_fn = score_fn

    def pairwise(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
    ) -> Float[Tensor, "B R"]:
        """Returns D_IEM(X[b], x_refs[r]) for all b, r."""
        device, dtype = X.device, X.dtype
        d = X.shape[1]
        gammas = self.gammas.to(device=device, dtype=dtype)
        # alpha is a normalization parameter (1/d) keeping f's argument O(1) for d-dim signals
        alpha = 1.0 / d
        f_prime = self.f_type.value.f_prime
        # builds the Brownian paths W once, reused across all references
        W = simulate_brownian(gammas, self.num_eps, d, self.seed, device, dtype)
        num_refs = x_refs.shape[0]
        cols = []
        for ref in range(num_refs):
            quad_var_increments, z_increments = iem_sq_increments_one_to_many(
                x_refs[ref:ref+1], X, W, gammas, self.density)
            # generalized-f integrand: f'(alpha * Z_g)^2 ||s1 - s2||^2 dg
            integrand = f_prime(alpha * z_increments).pow(2) * quad_var_increments  # (N_gamma-1, N_eps, B)
            iem_squared = integrand.sum(0)      # ∫…dg; integrand summed over gamma; (N_eps, B)
            cols.append(iem_squared.mean(0)     # IEM_f^2 = E_W[...]; average over Brownian paths
                        .clamp_min(0)           # ensure non-negative before sqrt
                        .sqrt())                # sqrt(IEM_f^2) = IEM_f; (B,) = D_IEM(X, x_refs[ref])
            if self.verbose and (ref + 1) % max(1, num_refs // 4) == 0:
                print(f'  ref {ref+1}/{num_refs}')
        pairwise_dist = torch.stack(cols, dim=1)   # (B, num_refs): D_IEM(X[b], x_refs[r])
        return pairwise_dist
