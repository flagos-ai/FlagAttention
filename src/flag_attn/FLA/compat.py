"""Compatibility helpers shared by FLA-derived kernels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import triton

F = TypeVar("F", bound=Callable[..., Any])


def libentry() -> Callable[[F], F]:
    """Keep FLA kernel declarations usable without the FlagGems runtime."""

    def decorator(fn: F) -> F:
        return fn

    return decorator


def libtuner(
    *,
    configs: Sequence[triton.Config],
    key: Sequence[str],
    use_cuda_graph: bool = False,
    **kwargs: Any,
):
    """Map FlagGems autotuning metadata to Triton's native autotuner."""
    del use_cuda_graph
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)
