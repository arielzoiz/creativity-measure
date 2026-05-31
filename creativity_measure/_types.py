from collections.abc import Callable

from jaxtyping import Float
from torch import Tensor

LogP = Callable[[Float[Tensor, "... d"]], Float[Tensor, "..."]]
LogPY = Callable[[Float[Tensor, "... d"], Float[Tensor, ""]], Float[Tensor, "..."]]
ScoreFn = Callable[[Float[Tensor, "B d"], Float[Tensor, ""]], Float[Tensor, "B d"]]
Sampler = Callable[[int], Float[Tensor, "n d"]]
