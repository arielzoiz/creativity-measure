import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure.distances.base import Distance


class LpDistance(Distance):
    """D(x, x') = ||x - x'||_p  (Minkowski). p=1 Manhattan, p=2 Euclidean, p=inf Chebyshev."""
    def __init__(self, p: float = 2.0):
        self.p = p

    def pairwise(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
    ) -> Float[Tensor, "B R"]:
        return torch.cdist(X, x_refs, p=self.p)
