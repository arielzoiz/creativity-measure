# Single source of truth for the compute device.
#
# Auto-selects CUDA when available, else CPU, so the pipeline transparently uses a remote GPU host while staying on CPU locally.
import torch

_OVERRIDE: torch.device | None = None


def default_device() -> torch.device:
    """CUDA if available, else CPU (unless overridden via set_default_device)."""
    if _OVERRIDE is not None:
        return _OVERRIDE
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_default_device(device: str | torch.device | None) -> None:
    """Override the auto-selected device (e.g. force "cpu", or opt into "mps"). None re-enables auto."""
    global _OVERRIDE
    _OVERRIDE = None if device is None else torch.device(device)
