from collections.abc import Callable
from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Generator, Tensor, nn

LogP = Callable[[Float[Tensor, "... d"]], Float[Tensor, "..."]]
LogPY = Callable[[Float[Tensor, "... d"], Float[Tensor, ""]], Float[Tensor, "..."]]
ScoreFn = Callable[[Float[Tensor, "B d"], Float[Tensor, ""]], Float[Tensor, "B d"]]
Sampler = Callable[[int, Generator | None], Float[Tensor, "n d"]]

# Flow-matching velocity, diffusers-native t (t = 1 pure noise, t = 0 clean data): v_theta(x_t, t) -> v.
# NOTE the polarity: this is the OPPOSITE of FlowMap below, which uses t = 0 noise / t = 1 data (that
# repo-internal convention, see samplers/flowmap_smc.py). Getting this backwards fails silently -- see
# creativity_measure/samplers/flow_guided.py's module docstring for the full history of this exact trap.
VelocityFn = Callable[[Float[Tensor, "B d"], float], Float[Tensor, "B d"]]


@runtime_checkable
class GuidableVelocityFn(Protocol):
    """A ``VelocityFn`` that also exposes ``.module``, the underlying frozen network.

    ``flow_guided_sample``'s ``exact_jacobian=True`` path needs this to verify the model is frozen
    (``requires_grad_(False)``, ``.eval()``) before building any autograd graph through it -- skipping
    that check lets autograd allocate gradient buffers for every parameter and OOM on the first backward.
    A backend's velocity-function builder (e.g. ``generators.flux.flux_velocity_fn``) should return a
    closure conforming to this so ``exact_jacobian=True`` works for it too.
    """

    def __call__(self, x_t: Float[Tensor, "B d"], t: float) -> Float[Tensor, "B d"]: ...

    module: nn.Module


@runtime_checkable
class SampleableDensity(Protocol):
    """
    Structural contract for a sampleable normalized density.
    Used in refset/ selectors, for sampling reference points from a distribution
    """

    def sample(
        self, n: int, seed: int | None = None, *, generator: Generator | None = None
    ) -> Float[Tensor, "n d"]: ...


# ---------------------------------------------------------------------------------------------------
# Flow-map vocabulary (consumed by ``creativity_measure.samplers.flowmap_smc``)
#
# All three are structural, so a notebook can satisfy them with a small closure-holding object and no
# inheritance -- the same freedom ``PCNKernel(generator_fn=...)`` gives today. Concrete implementations
# (``LinearSchedule``, ``ddpm_step``, ``flow_map_step``) live in ``samplers/flowmap_smc.py``.
# ---------------------------------------------------------------------------------------------------

@runtime_checkable
class FlowMap(Protocol):
    """A two-time flow map. Time convention: ``t = 0`` is pure noise, ``t = 1`` is clean data.

    ``map(x, t_from, t_to)``: transport between noise levels. ``t_to > t_from`` denoises,
        ``t_to < t_from`` renoises, and ``X_{t,t}`` is the identity by definition.
    ``denoise(x, t)``: ``E[x_1 | x_t]``. **NOT** ``map(x, t, 1)`` (an ODE endpoint, i.e. a sample) and
        **NOT** ``map(x, t, t)`` (the identity) -- it is built from the *instantaneous* velocity.

    The distinction is load-bearing and fails silently if collapsed. ``map(x, t, 1)`` is a *sample*,
    which is what the reward should score; ``denoise(x, t)`` is a *conditional mean*, and only a
    conditional mean can back a marginal score. For a flow map ``X_{t,t_to}(x) = x + (t - t_to)·u(x)``,
    so ``map(x, t, t)`` multiplies the velocity away: a denoiser derived that way would give the score
    of ``N(0, sigma_t^2)`` -- plausible-looking, and wrong everywhere.
    """

    def map(
        self, x: Float[Tensor, "B d"], t_from: float, t_to: float
    ) -> Float[Tensor, "B d"]: ...

    def denoise(self, x: Float[Tensor, "B d"], t: float) -> Float[Tensor, "B d"]: ...


@runtime_checkable
class Schedule(Protocol):
    """The interpolant ``x_t = alpha(t)*x_1 + sigma(t)*eps``, and the SNR inverse for ``t'``.

    ``t_of_snr`` is a *method* rather than a selectable enum member because ``g(t) = sigma^2/alpha^2``
    is derived from ``alpha``/``sigma``: choosing the pair independently would allow a mismatched
    inverse where ``g(t') = eta·g(t)`` silently fails to hold. ``name`` exists only for serialization
    (a reference file records the base process, and a string round-trips where a live object does not).
    """

    name: str

    def alpha(self, t: float) -> float: ...

    def sigma(self, t: float) -> float: ...

    def t_of_snr(self, y: float) -> float:
        """``g^-1(y)`` where ``g(t) = sigma(t)^2 / alpha(t)^2``."""
        ...


@runtime_checkable
class TransitionStep(Protocol):
    """One base transition ``x_t -> x_{t_next}``, given everything it might need.

    The DDPM transition (``flowmap_smc.ddpm_step``) is the canonical implementation; the deterministic
    flow-map jump (``flowmap_smc.flow_map_step``) satisfies the same signature and ignores ``generator``.
    Being a value rather than a string literal means a future model can supply its own transition
    without touching the sampler.
    """

    def __call__(
        self,
        x: Float[Tensor, "B d"],
        t: float,
        t_next: float,
        *,
        flow_map: FlowMap,
        schedule: Schedule,
        generator: Generator,
    ) -> Float[Tensor, "B d"]: ...
