# creativity_measure/density.py
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import LogP, LogPY, Sampler


@dataclass
class Density:
    """
    A probability distribution p, represented by evaluable log-densities.

    Required:
        log_p_X: (x: (..., d)) -> (...)  log p_X(x)

    Optional:
        log_p_Y:   (y: (..., d), gamma) -> (...)  log p_{Y_gamma}(y),
                   the marginal of Y = gamma*X + sqrt(gamma)*W, W ~ N(0, I).
                   Required by both IEM distances (Hessian for local, score
                   for global). May be omitted only if no IEM distance is used.
        sample_fn: (n: int) -> (n, d). Convenience sampler; the library never
                   calls it internally inside the distances -- x_refs is always
                   passed in.
        d:         dimensionality, else inferred on first use.
    """

    log_p_X: LogP
    log_p_Y: LogPY | None = None
    sample_fn: Sampler | None = None
    d: int | None = None

    def sample(self, n: int, seed: int | None = None) -> Float[Tensor, "n d"]:
        if self.sample_fn is None:
            raise RuntimeError("This Density has no sampler; pass x_refs explicitly.")
        if seed is not None:
            torch.manual_seed(seed)
        return self.sample_fn(n)
