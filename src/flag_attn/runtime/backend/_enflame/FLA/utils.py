# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Shared tensor and device helpers for Enflame FLA operators."""

import contextlib
import functools
from collections.abc import Callable
from typing import Any

import torch


ENFLAME_MAX_GRID_X = 65_535
ENFLAME_MAX_GRID_YZ = 255

def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    cache_entries: list[tuple[tuple, dict, Any]] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for index, (last_args, last_kwargs, last_result) in enumerate(cache_entries):
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(arg is last_arg for arg, last_arg in zip(args, last_args))
                and all(
                    key in last_kwargs and value is last_kwargs[key]
                    for key, value in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:index]
                    + cache_entries[index + 1 :]
                    + [(args, kwargs, last_result)]
                )
                return last_result
        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = tuple(
            value.contiguous() if isinstance(value, torch.Tensor) else value
            for value in args
        )
        contiguous_kwargs = {
            key: value.contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        tensor = next(
            (
                value
                for value in (*args, *kwargs.values())
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        if tensor is not None and tensor.device.type == "gcu" and hasattr(torch, "gcu"):
            context = torch.gcu.device(tensor.device)
        else:
            context = contextlib.nullcontext()
        with context:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    del arch, tensor_idx
    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        return False
    try:
        properties = torch.gcu.get_device_properties(torch.gcu.current_device())
    except (AttributeError, RuntimeError):
        return False
    for name in (
        "shared_memory_per_multiprocessor",
        "max_shared_mem",
        "max_shared_memory_per_multiprocessor",
        "max_shared_memory",
    ):
        max_shared = getattr(properties, name, None)
        if max_shared is not None:
            return max_shared >= 166_000
    return False


def requires_grid_remap(grid_x: int, grid_y: int = 1, grid_z: int = 1) -> bool:
    """Return whether a logical grid exceeds any Enflame physical limit."""
    return (
        grid_x > ENFLAME_MAX_GRID_X
        or grid_y > ENFLAME_MAX_GRID_YZ
        or grid_z > ENFLAME_MAX_GRID_YZ
    )


def _remap_grid(total_programs: int) -> tuple[int, int, int]:
    """Pack a linear program space into the three physical GCU dimensions."""
    max_programs = ENFLAME_MAX_GRID_X * ENFLAME_MAX_GRID_YZ**2
    if total_programs > max_programs:
        raise NotImplementedError(
            "Enflame grid remapping supports at most "
            f"{max_programs} programs, but got {total_programs}."
        )

    grid_z = max(
        1,
        (total_programs + ENFLAME_MAX_GRID_X * ENFLAME_MAX_GRID_YZ - 1)
        // (ENFLAME_MAX_GRID_X * ENFLAME_MAX_GRID_YZ),
    )
    programs_per_z = (total_programs + grid_z - 1) // grid_z
    grid_y = max(
        1,
        (programs_per_z + ENFLAME_MAX_GRID_X - 1) // ENFLAME_MAX_GRID_X,
    )
    grid_x = (total_programs + grid_y * grid_z - 1) // (grid_y * grid_z)
    return grid_x, grid_y, grid_z


def make_grid_3d(
    grid_x: int,
    grid_y: int,
    grid_z: int,
    remap: bool | None = None,
) -> tuple[int, ...]:
    """Keep a legal 3D grid or remap its linear program space within limits."""
    if remap is None:
        remap = requires_grid_remap(grid_x, grid_y, grid_z)
    if remap:
        return _remap_grid(grid_x * grid_y * grid_z)
    return grid_x, grid_y, grid_z


def make_grid_2d(
    grid_x: int,
    grid_y: int,
    remap: bool | None = None,
) -> tuple[int, ...]:
    """Keep a legal 2D grid or remap its linear program space within limits."""
    if remap is None:
        remap = requires_grid_remap(grid_x, grid_y)
    if remap:
        return _remap_grid(grid_x * grid_y)
    return grid_x, grid_y


__all__ = [
    "check_shared_mem",
    "input_guard",
    "make_grid_2d",
    "make_grid_3d",
    "requires_grid_remap",
    "tensor_cache",
]
