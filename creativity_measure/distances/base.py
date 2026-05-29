import torch
from typing import Protocol


class Distance(Protocol):
    """Any D usable in q_lambda. Returns pairwise D(X[b], x_refs[r])."""
    def pairwise(self, X: torch.Tensor, x_refs: torch.Tensor) -> torch.Tensor:
        """X: (B, d), x_refs: (R, d) -> (B, R)."""
        ...


class EuclideanDistance:
    """D(x, x') = ||x - x'||_2. Baseline distance and test oracle."""
    def pairwise(self, X, x_refs):
        return torch.cdist(X, x_refs)            # (B, R)
