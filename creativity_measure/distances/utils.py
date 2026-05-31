# log p(y | x, gamma) = log N(y; gamma*x, gamma*I).
import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributions import MultivariateNormal


def log_p_Y_given_X(
    y: Float[Tensor, "... d"],
    x: Float[Tensor, "... d"],
    gamma: Float[Tensor, ""],
) -> Float[Tensor, "..."]:
    """log p(y | x, gamma) under Y = gamma*x + sqrt(gamma)*W, W ~ N(0, I)."""
    d = x.shape[-1]
    cov = float(gamma) * torch.eye(d, device=x.device, dtype=x.dtype)
    return MultivariateNormal(float(gamma) * x, cov).log_prob(y)
