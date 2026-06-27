from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure._types import LogP, LogPY, Sampler
from creativity_measure.device import default_device


@dataclass
class Density:
    """
    A probability distribution p, represented by evaluable log-densities.

    Required:
        log_p_X: (x: (..., d)) -> (...)  log p_X(x)

    Optional:
        log_p_Y:   (y: (..., d), gamma) -> (...)  log p_{Y_gamma}(y), the marginal of Y = gamma*X + sqrt(gamma)*W, W ~ N(0, I).
                   Required by both IEM distances (Hessian for local, score for global). May be omitted only if no IEM distance is used.
        sample_fn: (n: int, generator) -> (n, d). Convenience sampler; the library never calls it internally inside the distances - x_refs is always
                   passed in. The ``generator`` argument (a ``torch.Generator`` or ``None``) lets callers thread a local RNG so sampling is reentrant
                   and reproducibility does not couple to global RNG state; when ``None`` the sampler may fall back to the global torch RNG.
        d:         dimensionality, else inferred on first use.
        device:    where sample() places its output. None => default_device() (CUDA if available, else CPU), resolved at call time so the
                   whole pipeline follows the chosen device with no user action.
    """

    log_p_X: LogP
    log_p_Y: LogPY | None = None
    sample_fn: Sampler | None = None
    d: int | None = None
    device: str | torch.device | None = None

    def sample(
        self, n: int, seed: int | None = None, *, generator: torch.Generator | None = None
    ) -> Float[Tensor, "n d"]:
        """Draw ``n`` points from the density.

        Pass ``generator`` to drive sampling from a caller-owned ``torch.Generator`` (reentrant / thread-safe, no global RNG state). 
        ``seed`` is a convenience for the global-RNG path only; if ``generator`` is given it takes precedence and ``seed`` is ignored.
        """
        if self.sample_fn is None:
            raise RuntimeError("This Density has no sampler; pass x_refs explicitly.")
        if generator is None and seed is not None:
            torch.manual_seed(seed)
        dev = self.device if self.device is not None else default_device()
        return self.sample_fn(n, generator).to(dev)
