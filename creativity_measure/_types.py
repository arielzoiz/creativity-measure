from collections.abc import Callable
from typing import Protocol

from jaxtyping import Float
from torch import Tensor

LogP = Callable[[Float[Tensor, "... d"]], Float[Tensor, "..."]]
LogPY = Callable[[Float[Tensor, "... d"], Float[Tensor, ""]], Float[Tensor, "..."]]
ScoreFn = Callable[[Float[Tensor, "B d"], Float[Tensor, ""]], Float[Tensor, "B d"]]
Sampler = Callable[[int], Float[Tensor, "n d"]]


class SampleableDensity(Protocol):
    """
    Structural contract for a sampleable normalized density.
    Used in refset/ selectors, for sampling reference points from a distribution
    """

    def sample(self, n: int, seed: int | None = None) -> Float[Tensor, "n d"]: ...
