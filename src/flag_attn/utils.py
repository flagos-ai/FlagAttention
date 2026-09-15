# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities shared by all FlagAttention operators."""

from __future__ import annotations

import contextlib
import functools
import re
from collections.abc import Callable
from typing import Any

import torch


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    """Return whether the installed Triton exposes the requested TLE API."""
    try:
        import triton
        import triton.experimental.tle.language  # noqa: F401
    except ImportError:
        return False
    parts = re.findall(r"\d+", str(getattr(triton, "__version__", "0.0.0")))
    values = [int(part) for part in parts[:3]]
    values += [0] * (3 - len(values))
    return tuple(values[:3]) >= (major, minor, patch)


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Cache the eight most recent calls using tensor identity as the key."""
    cache_entries: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, (last_args, last_kwargs, last_result) in enumerate(cache_entries):
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(
                    key in last_kwargs and value is last_kwargs[key]
                    for key, value in kwargs.items()
                )
            ):
                cache_entries = cache_entries[:i] + cache_entries[i + 1:] + [
                    (args, kwargs, last_result)
                ]
                return last_result
        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Make tensor inputs contiguous and execute on the input tensor's device."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        contiguous_args = (
            value if not isinstance(value, torch.Tensor) else value.contiguous()
            for value in args
        )
        contiguous_kwargs = {
            key: value if not isinstance(value, torch.Tensor) else value.contiguous()
            for key, value in kwargs.items()
        }
        tensor = next((value for value in args if isinstance(value, torch.Tensor)), None)
        if tensor is None:
            tensor = next(
                (value for value in kwargs.values() if isinstance(value, torch.Tensor)),
                None,
            )
        ctx = (
            torch.cuda.device(tensor.device)
            if tensor is not None and tensor.is_cuda
            else contextlib.nullcontext()
        )
        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """Check whether the active CUDA device meets the shared-memory heuristic."""
    del arch, tensor_idx
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:
        return False
    for name in (
        "shared_memory_per_multiprocessor",
        "max_shared_mem",
        "max_shared_memory_per_multiprocessor",
        "max_shared_memory",
    ):
        value = getattr(props, name, None)
        if value is not None:
            return value >= 166_000
    return False


def round_up(n: int, d: int) -> int:
    """Round n up to the nearest multiple of d."""
    return (n + d - 1) // d * d


class _CurrentPlatform:
    """Small platform adapter for backend capabilities shared by operators."""

    def is_arch_support_pdl(self) -> bool:
        """Return whether the active CUDA device supports launch PDL."""
        if not torch.cuda.is_available():
            return False
        # HIP/ROCm does not accept CUDA's launch_pdl runtime argument.
        if torch.version.hip is not None:
            return False
        try:
            capability = torch.cuda.get_device_capability()
        except RuntimeError:
            return False
        return capability >= (9, 0)


current_platform = _CurrentPlatform()
