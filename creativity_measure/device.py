# Single source of truth for the compute device and dtype.
#
# Auto-selects CUDA when available, else CPU, so the pipeline transparently uses a remote GPU host
# while staying on CPU locally. dtype defaults to float64 (the analytic / IEM core is double-precision);
# override to float32 for the pixel / EDM path.
#
# These are the single *origination* authority for freshly-created tensors, everything downstream flows from its input tensor's device/dtype,
# so there is one place to set them.
import torch

_OVERRIDE: torch.device | None = None
_DTYPE_OVERRIDE: torch.dtype | None = None


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


def default_dtype() -> torch.dtype:
    """float64 by default (the analytic / IEM core is double-precision), unless overridden via set_default_dtype."""
    if _DTYPE_OVERRIDE is not None:
        return _DTYPE_OVERRIDE
    return torch.float64


def set_default_dtype(dtype: str | torch.dtype | None) -> None:
    """Override the default dtype (e.g. torch.float32 for the pixel/EDM path). None re-enables float64."""
    global _DTYPE_OVERRIDE
    if dtype is None:
        _DTYPE_OVERRIDE = None
        return
    resolved = getattr(torch, dtype, None) if isinstance(dtype, str) else dtype
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Not a torch dtype: {dtype!r}")
    _DTYPE_OVERRIDE = resolved
