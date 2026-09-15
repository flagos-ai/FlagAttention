# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Accelerator discovery shared by tests and benchmarks.

Set ``FLAG_ATTN_TEST_DEVICE`` to override automatic discovery, for example
``cuda:1`` or ``gcu:0``.  The legacy ``S60_TEST_DEVICE`` variable is retained
as the default GCU index.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

import torch


_OPTIONAL_BACKEND_PACKAGES = {
    "gcu": "torch_gcu",
    "npu": "torch_npu",
    "mlu": "torch_mlu",
    "musa": "torch_musa",
}

_AUTO_PRIORITY = ("cuda", "gcu", "npu", "mlu", "musa", "xpu")
_BACKEND_CACHE: dict[str, object | None] = {}


@dataclass(frozen=True)
class Accelerator:
    device_type: str
    index: int
    name: str
    architecture: str
    capability: tuple[int, int] | None = None

    @property
    def device(self) -> torch.device:
        return torch.device(f"{self.device_type}:{self.index}")

    @property
    def description(self) -> str:
        details = self.name
        if self.architecture and self.architecture != self.name:
            details = f"{details} ({self.architecture})"
        return f"{self.device} {details}".strip()


def _load_backend(device_type: str):
    if device_type in _BACKEND_CACHE:
        return _BACKEND_CACHE[device_type]
    package = _OPTIONAL_BACKEND_PACKAGES.get(device_type)
    if package and not hasattr(torch, device_type):
        try:
            importlib.import_module(package)
        except (ImportError, OSError, RuntimeError):
            _BACKEND_CACHE[device_type] = None
            return None
    backend = getattr(torch, device_type, None)
    _BACKEND_CACHE[device_type] = backend
    return backend


def _is_available(device_type: str) -> bool:
    backend = _load_backend(device_type)
    if backend is None:
        return False
    is_available = getattr(backend, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except (OSError, RuntimeError):
        return False


def _device_name(backend, index: int, device_type: str) -> str:
    get_name = getattr(backend, "get_device_name", None)
    if callable(get_name):
        try:
            return str(get_name(index))
        except (TypeError, OSError, RuntimeError):
            pass

    get_properties = getattr(backend, "get_device_properties", None)
    if callable(get_properties):
        try:
            properties = get_properties(index)
            for attribute in ("name", "device_name"):
                value = getattr(properties, attribute, None)
                if value:
                    return str(value)
        except (TypeError, OSError, RuntimeError):
            pass
    return device_type.upper()


def _device_capability(backend, index: int) -> tuple[int, int] | None:
    get_capability = getattr(backend, "get_device_capability", None)
    if not callable(get_capability):
        return None
    try:
        capability = get_capability(index)
    except (TypeError, OSError, RuntimeError):
        return None
    if isinstance(capability, (tuple, list)) and len(capability) >= 2:
        return int(capability[0]), int(capability[1])
    return None


def _make_accelerator(device_type: str, index: int) -> Accelerator:
    backend = _load_backend(device_type)
    if backend is None or not _is_available(device_type):
        raise RuntimeError(f"requested accelerator {device_type}:{index} is unavailable")

    count = getattr(backend, "device_count", None)
    if callable(count):
        try:
            available_count = int(count())
            if index < 0 or index >= available_count:
                raise RuntimeError(
                    f"requested accelerator {device_type}:{index}, but only "
                    f"{available_count} device(s) are available"
                )
        except (OSError, TypeError, ValueError):
            pass

    capability = _device_capability(backend, index)
    name = _device_name(backend, index, device_type)
    architecture = (
        f"sm_{capability[0]}{capability[1]}"
        if device_type == "cuda" and capability is not None
        else name
    )
    return Accelerator(device_type, index, name, architecture, capability)


def _parse_device(specification: str) -> tuple[str, int]:
    device_type, separator, raw_index = specification.strip().lower().partition(":")
    if not device_type:
        raise ValueError("FLAG_ATTN_TEST_DEVICE must name an accelerator")
    if separator:
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(
                f"invalid FLAG_ATTN_TEST_DEVICE value: {specification!r}"
            ) from exc
    elif device_type == "gcu":
        index = int(os.environ.get("S60_TEST_DEVICE", "0"))
    else:
        index = 0
    return device_type, index


def detect_accelerator(
    supported: tuple[str, ...] = _AUTO_PRIORITY,
) -> Accelerator | None:
    """Return the selected accelerator when its device type is supported."""

    explicit = os.environ.get("FLAG_ATTN_TEST_DEVICE")
    if explicit:
        device_type, index = _parse_device(explicit)
        if device_type not in _AUTO_PRIORITY:
            raise ValueError(
                f"unknown FLAG_ATTN_TEST_DEVICE backend: {device_type!r}"
            )
        if device_type not in supported:
            return None
        return _make_accelerator(device_type, index)

    for device_type in _AUTO_PRIORITY:
        if device_type not in supported or not _is_available(device_type):
            continue
        index = int(os.environ.get("S60_TEST_DEVICE", "0")) if device_type == "gcu" else 0
        return _make_accelerator(device_type, index)
    return None
def set_device(accelerator: Accelerator) -> None:
    backend = _load_backend(accelerator.device_type)
    setter = getattr(backend, "set_device", None)
    if callable(setter):
        setter(accelerator.index)


def synchronize(accelerator: Accelerator | None = None) -> None:
    accelerator = accelerator or detect_accelerator()
    if accelerator is None:
        return
    backend = _load_backend(accelerator.device_type)
    sync = getattr(backend, "synchronize", None)
    if callable(sync):
        sync()


def empty_cache(accelerator: Accelerator | None = None) -> None:
    device_types = (
        (accelerator.device_type,) if accelerator is not None else _AUTO_PRIORITY
    )
    for device_type in device_types:
        if accelerator is None and not _is_available(device_type):
            continue
        backend = _load_backend(device_type)
        clear = getattr(backend, "empty_cache", None)
        if callable(clear):
            clear()


def supports_fp8(accelerator: Accelerator | None) -> bool:
    if accelerator is None or not hasattr(torch, "float8_e4m3fn"):
        return False
    # The CUDA kernels require Ada/Hopper-or-newer FP8 Tensor Cores.  GCU FP8
    # is not enabled here until its capability API can express that guarantee.
    return (
        accelerator.device_type == "cuda"
        and accelerator.capability is not None
        and accelerator.capability >= (8, 9)
    )


__all__ = [
    "Accelerator",
    "detect_accelerator",
    "empty_cache",
    "set_device",
    "supports_fp8",
    "synchronize",
]
