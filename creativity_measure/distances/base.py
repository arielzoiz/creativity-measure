from abc import ABC, abstractmethod
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
