# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Shared utilities for the S60 NSA implementation."""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Cache the eight most recent identity-equal inputs."""
    cache_entries: list[tuple[tuple, dict, Any]] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries

        for index, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry

            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(
                    current is previous
                    for current, previous in zip(
                        args,
                        last_args,
                    )
                )
                and all(
                    key in last_kwargs
                    and value is last_kwargs[key]
                    for key, value in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:index]
                    + cache_entries[index + 1:]
                    + [(args, kwargs, last_result)]
                )
                return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]

        cache_entries.append(
            (args, kwargs, result)
        )
        return result

    return wrapper


def _device_context(
    tensor: torch.Tensor,
):
    device_api = getattr(
        torch,
        tensor.device.type,
        None,
    )
    device_context = getattr(
        device_api,
        "device",
        None,
    )

    if not callable(device_context):
        return contextlib.nullcontext()

    try:
        return device_context(tensor.device)
    except (RuntimeError, TypeError, ValueError):
        return contextlib.nullcontext()


def input_guard(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Make tensor inputs contiguous and select their device."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        contiguous_args = tuple(
            value
            if not isinstance(value, torch.Tensor)
            else value.contiguous()
            for value in args
        )

        contiguous_kwargs = {
            key: (
                value
                if not isinstance(value, torch.Tensor)
                else value.contiguous()
            )
            for key, value in kwargs.items()
        }

        tensor = next(
            (
                value
                for value in contiguous_args
                if isinstance(value, torch.Tensor)
            ),
            None,
        )

        if tensor is None:
            tensor = next(
                (
                    value
                    for value in contiguous_kwargs.values()
                    if isinstance(value, torch.Tensor)
                ),
                None,
            )

        context = (
            _device_context(tensor)
            if tensor is not None
            else contextlib.nullcontext()
        )

        with context:
            return fn(
                *contiguous_args,
                **contiguous_kwargs,
            )

    return wrapper


def check_shared_mem(
    arch: str = "none",
    tensor_idx: int = 0,
) -> bool:
    """Return whether the active device has large shared memory."""
    del arch, tensor_idx

    for backend_name in ("gcu", "cuda"):
        device_api = getattr(
            torch,
            backend_name,
            None,
        )

        if device_api is None:
            continue

        get_properties = getattr(
            device_api,
            "get_device_properties",
            None,
        )

        if not callable(get_properties):
            continue

        try:
            current_device = getattr(
                device_api,
                "current_device",
                lambda: 0,
            )()

            try:
                properties = get_properties(
                    current_device
                )
            except TypeError:
                properties = get_properties()

        except (RuntimeError, TypeError, ValueError):
            continue

        for attribute in (
            "shared_memory_per_multiprocessor",
            "shared_memory_per_block",
            "max_shared_memory_per_block",
            "max_shared_mem",
        ):
            value = getattr(
                properties,
                attribute,
                None,
            )

            if value is not None:
                return int(value) >= 166_000

    return False


@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [
        n_outer * 2**i,
        2,
        2 ** (n_dims - i - 1),
    ]

    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]

    left = tl.broadcast_to(
        tl.sum(y * (1 - mask), 1)[:, None, :],
        shape,
    ).to(y.dtype)

    right = tl.broadcast_to(
        tl.sum(y * mask, 1)[:, None, :],
        shape,
    ).to(y.dtype)

    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    y_idx = tl.reshape(ids, shape)

    left_idx = tl.broadcast_to(
        tl.sum(y_idx * (1 - mask), 1)[:, None, :],
        shape,
    )

    right_idx = tl.broadcast_to(
        tl.sum(y_idx * mask, 1)[:, None, :],
        shape,
    )

    left_idx = tl.reshape(
        left_idx,
        x.shape,
    ).to(y_idx.dtype)

    right_idx = tl.reshape(
        right_idx,
        x.shape,
    ).to(y_idx.dtype)

    integer_dtype = tl.core.get_int_dtype(
        bitwidth=x.dtype.primitive_bitwidth,
        signed=True,
    )

    integer_left = left.to(
        integer_dtype,
        bitcast=True,
    )

    integer_right = right.to(
        integer_dtype,
        bitcast=True,
    )

    integer_x = x.to(
        integer_dtype,
        bitcast=True,
    )

    condition = (left > right) != flip

    result = integer_x ^ tl.where(
        condition,
        integer_left ^ integer_right,
        tl.zeros_like(integer_x),
    )

    new_ids = ids ^ tl.where(
        condition,
        left_idx ^ right_idx,
        tl.zeros_like(ids),
    )

    return (
        result.to(x.dtype, bitcast=True),
        new_ids,
    )


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)

    if order == 2:
        shape: tl.constexpr = [
            n_outer * 2 ** (n_dims - 1 - stage),
            2,
            2**stage,
        ]

        flip = tl.reshape(
            tl.broadcast_to(
                tl.arange(0, 2)[None, :, None],
                shape,
            ),
            x.shape,
        )
    else:
        flip = order

    for index in tl.static_range(stage):
        x, ids = _compare_and_swap(
            x,
            ids,
            flip,
            index + (n_dims - stage),
            n_dims,
        )

    return x, ids


__all__ = [
    "_bitonic_merge",
    "check_shared_mem",
    "input_guard",
    "tensor_cache",
]
