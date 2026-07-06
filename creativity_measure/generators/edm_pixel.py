"""Pretrained EDM-checkpoint generator (NVIDIA ``edm`` / Karras ``EDMPrecond``, e.g. the MNIST EDM model).

This is the clean, primary pixel path. An ``EDMPrecond`` network already returns the MMSE estimate
``E[X | x_sigma]`` in the exact convention ``edm_generator`` expects -- it applies its own
``c_in/c_out/c_skip/c_noise`` preconditioning internally (``edm/training/networks.py``), and the EDM
sampler computes ``d = (x - net(x, sigma, labels)) / sigma`` identically to ``heun_prob_flow``. So there is
**no** epsilon->x0 conversion and **no** timestep mapping: we only adapt the call signature and hand the
network straight to ``edm_generator``. No third-party dependency beyond ``torch`` and the net the caller loads.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from creativity_measure.distances.edm_adapter import Denoiser

from .base import edm_generator


@runtime_checkable
class EDMNet(Protocol):
    """Structural type for a Karras ``EDMPrecond``-style network: ``net(x, sigma, class_labels) = E[X|x]``.

    ``@runtime_checkable`` so the ``beartype`` import hook (see ``tests/conftest.py``) can check it.
    """

    sigma_min: float
    sigma_max: float

    def __call__(self, x: Tensor, sigma: Tensor, class_labels: Tensor | None = ...) -> Tensor: ...


def build_edm_pixel_generator(
    net: EDMNet,
    *,
    img_shape: tuple[int, ...],
    class_labels: Tensor | None = None,
    n_steps: int = 64,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    rho: float = 7.0,
) -> Callable[[Float[Tensor, "B d"]], Float[Tensor, "B d"]]:
    """Wrap a pretrained ``EDMPrecond`` network into the flat ``(B, d)`` generator interface.

    Args:
        net:          a Karras ``EDMPrecond`` (or duck-typed equivalent) with ``.sigma_min``/``.sigma_max``.
        img_shape:    per-sample shape ``(C, H, W)`` the network expects.
        class_labels: one-hot labels ``(B, label_dim)`` for conditional models; ``None`` = unconditional.
        n_steps:      number of Heun steps in the prob-flow ODE.
        sigma_min:    override the network's ``sigma_min`` (defaults to ``net.sigma_min``).
        sigma_max:    override the network's ``sigma_max`` (defaults to ``net.sigma_max``).

    Returns the deterministic ``G(z): N(0,I) -> x ~ p`` consumed by ``smc.PCNKernel``.
    """
    def denoiser(x_sigma: Float[Tensor, "B C H W"], sigma: Float[Tensor, "B"]) -> Float[Tensor, "B C H W"]:
        return net(x_sigma, sigma, class_labels)

    d: Denoiser = denoiser
    return edm_generator(
        d,
        img_shape=img_shape,
        sigma_min=float(net.sigma_min) if sigma_min is None else sigma_min,
        sigma_max=float(net.sigma_max) if sigma_max is None else sigma_max,
        rho=rho,
        n_steps=n_steps,
    )


def build_edm_pixel_generator_from_pkl(
    pkl_path: str,
    *,
    img_shape: tuple[int, ...],
    class_idx: int | None = None,
    device: str = "cpu",
    n_steps: int = 64,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    rho: float = 7.0,
) -> Callable[[Float[Tensor, "B d"]], Float[Tensor, "B d"]]:
    """Convenience: load an ``edm`` ``.pkl`` checkpoint (its ``ema`` net) and build the generator.

    Loading requires the ``edm`` package (its ``dnnlib`` / ``torch_utils`` modules) importable so the
    pickled network class resolves; that is the caller's responsibility (the ``edm/`` repo on ``sys.path``).
    For a conditional model, ``class_idx`` selects the one-hot label; ``None`` leaves it unconditional.
    """
    import pickle

    with open(pkl_path, "rb") as f:
        net: Any = pickle.load(f)["ema"]
    net = net.to(device)

    class_labels: Tensor | None = None
    label_dim = int(getattr(net, "label_dim", 0))
    if class_idx is not None:
        if label_dim == 0:
            raise ValueError("class_idx given but the network is unconditional (label_dim == 0).")
        class_labels = torch.zeros(1, label_dim, device=device)
        class_labels[0, class_idx] = 1.0

    return build_edm_pixel_generator(
        net,
        img_shape=img_shape,
        class_labels=class_labels,
        n_steps=n_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
    )
