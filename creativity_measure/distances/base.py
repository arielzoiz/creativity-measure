import torch
from jaxtyping import Float
from torch import Tensor
from typing import Protocol


class Distance(Protocol):
    """Any D usable in q_lambda. Returns pairwise D(X[b], x_refs[r])."""
    def pairwise(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
    ) -> Float[Tensor, "B R"]:
        ...


class EuclideanDistance:
    """D(x, x') = ||x - x'||_2. Baseline distance and test oracle."""
    def pairwise(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
    ) -> Float[Tensor, "B R"]:
        return torch.cdist(X, x_refs)
