# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers shared by NVIDIA MSA index and attention."""

from functools import lru_cache

import torch
import triton


def _tensor_metadata(tensor):
    """A launch-plan key, never an input identity or retained data pointer."""
    if tensor is None:
        return None
    return (tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device, tensor.data_ptr() % 16)


def _can_reuse_launch(*kernels):
    """Keep instrumented/debug JIT execution on its ordinary path."""
    try:
        debug = triton.knobs.runtime.debug
        instrumentation = triton.knobs.compilation.instrumentation_mode
    except AttributeError:
        return False
    return not (
        debug
        or instrumentation
        or any(getattr(kernel, "pre_run_hooks", ()) for kernel in kernels)
    )


def _cache_plan(cache, key, plan):
    """Bound metadata-only recipes consistently across attention and index."""
    if len(cache) >= 64:
        cache.clear()
    cache[key] = plan


class _KernelLaunch:
    """Cache scalar launch metadata and compiled code, not tensor arguments.

    Triton's compiled-kernel call resolves the current stream and retains
    launch profiling hooks.
    """

    def __init__(self, kernel, grid, tail, *, suffix=(), **options):
        self.kernel = kernel
        self.grid = tuple(grid) + (1,) * (3 - len(grid))
        self.tail = tuple(tail)
        self.suffix = tuple(suffix)
        self.options = options
        self.compiled = None
        if any(value is not None and type(value) not in (bool, int, float) for value in self.tail + self.suffix):
            raise TypeError("Launch recipes may retain only scalar metadata")

    def __call__(self, *tensors, extra=()):
        args = (*tensors, *self.tail, *extra, *self.suffix)
        if not _can_reuse_launch(self.kernel):
            return self.kernel[self.grid](*args, **self.options)
        if self.compiled is None:
            self.compiled = self.kernel.warmup(*args, grid=self.grid, **self.options)
        return self.compiled[self.grid](*args)


@lru_cache(maxsize=32)
def _sm_count(device: torch.device) -> int:
    """SM count for a concrete tensor device, reused by launch-grid policies."""
    return torch.cuda.get_device_properties(device).multi_processor_count


def _packed_bytes(tensor):
    allowed = (torch.uint8, torch.int8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))
    if tensor.dtype not in allowed:
        raise TypeError("FP4 values require packed E2M1 byte storage")
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)
