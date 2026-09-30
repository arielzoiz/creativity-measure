from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


class Distance(ABC):
    """Any D usable in q_lambda. Returns pairwise D(X[b], x_refs[r])."""
    @abstractmethod
    def pairwise(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
    ) -> Float[Tensor, "B R"]:
        ...


@runtime_checkable
class ExpectedDistance(Protocol):
    """Optional fast path: a Distance that can return the (weighted) mean over refs WITHOUT forming the (B, R) matrix.

    `tilt.expected_distance` prefers `expected` when a distance provides it. It must equal
    sum_r w_r D(X[b], x_refs[r]) / sum_r w_r (uniform when weights is None), i.e. the same number `pairwise` yields.
    """

    def expected(
        self,
        X: Float[Tensor, "B d"],
        x_refs: Float[Tensor, "R d"],
        weights: Float[Tensor, "R"] | None = None,
    ) -> Float[Tensor, "B"]:
        ...
