"""Device and precision selection."""

from __future__ import annotations

import torch

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64
}


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Turns a device name into a device this machine actually has.

    Args:
        device: A device name or object. "auto" prefers CUDA, then MPS, then CPU.

    Returns:
        The resolved device.

    Raises:
        ValueError: If the requested device is unavailable. An explicit request
            never falls back to another device, since that would change speed
            and numerics without saying so.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not _cuda_device_exists(resolved):
        raise ValueError(f"Device '{resolved}' is not available on this machine.")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError(f"Device '{resolved}' is not available on this machine.")
    return resolved


def resolve_dtype(
    dtype: str | torch.dtype,
    device: torch.device,
    *,
    prefer: torch.dtype = torch.float32
) -> torch.dtype:
    """Picks a floating point precision the device supports.

    Args:
        dtype: A precision name or object. "auto" uses `prefer` where supported.
        device: The device the precision will be used on.
        prefer: The precision "auto" aims for.

    Returns:
        The resolved precision.

    Raises:
        ValueError: If the name is unknown, or float64 is requested on MPS.
    """
    if dtype == "auto":
        if prefer == torch.float64 and device.type == "mps":
            return torch.float32
        return prefer
    if isinstance(dtype, str) and dtype not in _DTYPES:
        raise ValueError(f"Unknown dtype '{dtype}'. Choose from {sorted(_DTYPES)}.")
    resolved = _DTYPES[dtype] if isinstance(dtype, str) else dtype
    if resolved == torch.float64 and device.type == "mps":
        raise ValueError("MPS does not support float64. Use float32 or run on CPU.")
    return resolved


def _cuda_device_exists(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    return device.index is None or device.index < torch.cuda.device_count()
