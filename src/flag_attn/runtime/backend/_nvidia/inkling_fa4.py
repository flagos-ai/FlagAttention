# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA dispatch for portable Triton and optional Hopper Inkling FA4."""

import os
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch

from flag_attn.inkling_fa4.relative_attention import (
    _TLE_ALIASES,
    _TRITON_ALIASES,
    inkling_fa4_rel_attention_triton,
)


@lru_cache(maxsize=None)
def _is_hopper(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


@lru_cache(maxsize=1)
def _tle_dependency() -> tuple[bool, str]:
    try:
        tle = import_module("triton.experimental.tle.language")
    except ImportError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    required = (
        "alloc", "alloc_barriers", "READY", "copy", "barrier_wait",
        "barrier_arrive", "warp_specialize", "wgmma", "wgmma_wait",
    )
    gpu = getattr(tle, "gpu", None)
    missing = [name for name in required if not hasattr(gpu, name)]
    if missing:
        return False, f"TLE GPU API is missing: {', '.join(missing)}"
    return True, ""


def tle_available() -> tuple[bool, str]:
    """Check the current device before probing the optional extension."""
    if not torch.cuda.is_available() or not _is_hopper(torch.device("cuda", torch.cuda.current_device())):
        return False, "Inkling FA4 TLE requires NVIDIA Hopper SM90"
    return _tle_dependency()


def inkling_fa4_rel_attention(
    *args: Any,
    backend: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Select Hopper only for compatible hardware, dependencies and shapes."""
    selected = (backend or os.getenv("INKLING_FA4_BACKEND", "auto")).strip().lower()
    if selected in _TRITON_ALIASES:
        return inkling_fa4_rel_attention_triton(*args, **kwargs)
    if selected not in _TLE_ALIASES | {"auto"}:
        choices = ", ".join(sorted(_TRITON_ALIASES | _TLE_ALIASES | {"auto"}))
        raise ValueError(f"unknown Inkling FA4 backend={selected!r}; choose from {choices}")

    q, k, v = (
        args[index] if len(args) > index else kwargs.get(name)
        for index, name in enumerate(("q", "key_cache", "value_cache"))
    )
    reason = "Inkling FA4 TLE requires NVIDIA Hopper SM90"
    available = isinstance(q, torch.Tensor) and _is_hopper(q.device)
    if available:
        available, reason = _tle_dependency()
    if not available:
        if selected in _TLE_ALIASES:
            raise RuntimeError(reason)
        return inkling_fa4_rel_attention_triton(*args, **kwargs)

    compatible = (
        all(isinstance(t, torch.Tensor) for t in (q, k, v))
        and q.ndim == 3 and k.ndim == v.ndim == 4
        and k.shape[1] == v.shape[1] == 16
        and q.shape[2] == k.shape[3] == v.shape[3] == 128
        and all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (q, k, v))
    )
    if not compatible:
        if selected in _TLE_ALIASES:
            raise NotImplementedError(
                "Hopper Inkling FA4 requires block_size=16, Q/K/V head_dim=128, "
                "and contiguous, 16-byte aligned Q/K/V tensors"
            )
        return inkling_fa4_rel_attention_triton(*args, **kwargs)

    implementation = import_module(".hopper.inkling_fa4.relative_attention", __package__)
    with torch.cuda.device(q.device):
        return implementation.inkling_fa4_rel_attention_tle(*args, **kwargs)
