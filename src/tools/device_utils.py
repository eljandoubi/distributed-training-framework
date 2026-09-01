"""Accelerator-agnostic device module discovery and a default-dtype context manager."""

import contextlib
from typing import Any, Protocol, cast

import torch
from torch._utils import _get_available_device_type, _get_device_module


class DeviceModule(Protocol):
    """Structural type for the subset of `torch.cuda`-like device module APIs this project depends on."""

    def set_device(self, device: torch.device | int) -> None: ...
    def current_device(self) -> int: ...
    def synchronize(self) -> None: ...
    def get_device_name(self, device: torch.device | int) -> str: ...
    def get_device_properties(self, device: torch.device | int) -> Any: ...
    def reset_peak_memory_stats(self) -> None: ...
    def empty_cache(self) -> None: ...
    def memory_stats(self, device: torch.device | int) -> dict[str, Any]: ...


def get_device_info() -> tuple[str, DeviceModule]:
    """Return the available accelerator type (e.g. "cuda", "xpu") and its corresponding device module."""
    device_type = _get_available_device_type() or "cuda"
    device_module = cast(DeviceModule, _get_device_module(device_type))
    return device_type, device_module


device_type, device_module = get_device_info()


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype):
    """Temporarily set torch's default dtype to `dtype`, restoring the previous value on exit."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
