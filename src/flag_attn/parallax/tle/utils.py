"""Runtime-independent helpers for Parallax kernels."""

import contextlib
import functools
from collections.abc import Callable
from typing import Any

import torch


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Cache the eight most recent calls by input object identity."""
    entries: list[tuple[tuple, dict, Any]] = []

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal entries
        for index, (old_args, old_kwargs, result) in enumerate(entries):
            if (
                len(args) == len(old_args)
                and len(kwargs) == len(old_kwargs)
                and all(value is old for value, old in zip(args, old_args))
                and all(
                    key in old_kwargs and value is old_kwargs[key]
                    for key, value in kwargs.items()
                )
            ):
                entries = entries[:index] + entries[index + 1 :] + [
                    (args, kwargs, result)
                ]
                return result
        result = fn(*args, **kwargs)
        entries = (entries + [(args, kwargs, result)])[-8:]
        return result

    return wrapper


def input_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Make tensor inputs contiguous and select their CUDA device."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        guarded_args = tuple(
            value.contiguous() if isinstance(value, torch.Tensor) else value
            for value in args
        )
        guarded_kwargs = {
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
        context = (
            torch.cuda.device(tensor.device)
            if tensor is not None and tensor.is_cuda
            else contextlib.nullcontext()
        )
        with context:
            return fn(*guarded_args, **guarded_kwargs)

    return wrapper
