# creativity_measure/distances/utils.py
# log p(y | x, gamma) = log N(y; gamma*x, gamma*I).
# Ported from iem_creativity.ipynb (cell 3, log_p_Y_given_X).
import torch
from torch.distributions import MultivariateNormal


def log_p_Y_given_X(y, x, gamma):
    """y, x: (..., d)   gamma: scalar  ->  (...)."""
    d = x.shape[-1]
    cov = float(gamma) * torch.eye(d, device=x.device, dtype=x.dtype)
    return MultivariateNormal(float(gamma) * x, cov).log_prob(y)
