# Random reference selection: i.i.d. draws from p, uniform weights.
#
# UNBIASED for f(x) = E_{x'~p}[D_IEM(x, x')]: with x'_m ~ p i.i.d. and uniform weights,
# (1/R) sum_m D_IEM(x, x'_m) is an unbiased Monte-Carlo estimate of E_{x'~p}[D_IEM(x, .)],
# variance tr Cov_{x'}[D]/R. No reweighting. Preferred when p is not strongly multimodal.
#
# Auto-R uses the inherited weighted-τ level target, averaging over INDEPENDENT random draws per R
# (auto_r_draws), since reference draw-to-draw variance is exactly what sets R for an i.i.d. estimator.

from jaxtyping import Float
from torch import Tensor

from .base import RefSelector


class RandomRefs(RefSelector):
    """i.i.d. references from p (uniform weights) -> unbiased estimator of E_{x'~p}[D_IEM]."""

    def _refs_for_size(self, n: int, draw: int) -> Float[Tensor, "n d"]:
        # Distinct seed per draw so the auto-R τ sweep averages over independent random reference sets.
        seed = None if self.seed is None else self.seed + draw
        return self.p.sample(n, seed=seed)

    # _auto_r_effective_draws inherited (= auto_r_draws): random IS stochastic, so average over draws.