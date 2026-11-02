# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Cached low-level launcher for the Ascend AttnRes kernel.

Triton's Python launch path re-parses options, rebuilds the specialization key,
binds arguments, canonicalises the grid and walks metadata/hook plumbing on
every call.  For a single-kernel operator whose device time can be a few
microseconds this dominates the public-API latency, so the first call for a
specialization compiles through ``warmup`` and caches the generated raw
launcher; later calls only resolve the stream and call it.

The generated launcher ABI on this Triton build (3.5.x + triton-ascend) is::

    launch(gridX, gridY, gridZ, stream, function, packed_metadata,
           launch_metadata, launch_enter_hook, launch_exit_hook, *signature_args)

where ``signature_args`` are all kernel parameters in declaration order; the
constexpr ones are passed as PyObjects and ignored by the launcher.
"""

from __future__ import annotations

import os

import torch

_OFF = frozenset({"off", "0", "false", "no"})

try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _raw_stream

    _HAS_RAW_STREAM = True
except Exception:  # pragma: no cover - depends on the torch_npu build
    _raw_stream = None
    _HAS_RAW_STREAM = False

try:
    from torch_npu._C import _npu_getDevice as _get_device
except Exception:  # pragma: no cover - depends on the torch_npu build
    _get_device = None


def _enabled() -> bool:
    return os.environ.get("FLAG_ATTNRES_FASTLAUNCH", "1").lower() not in _OFF


def _device() -> int:
    if _get_device is not None:
        try:
            return int(_get_device())
        except Exception:
            pass
    return int(torch.npu.current_device())


def _stream(dev: int) -> int:
    if _HAS_RAW_STREAM:
        return _raw_stream(dev)
    import triton

    return triton.runtime.driver.active.get_current_stream(dev)


def _hooks_active() -> bool:
    """A profiler/trace hook needs launch metadata, so keep the regular path."""
    try:
        import triton

        runtime = triton.knobs.runtime
        # Newer Triton stores hooks in HookChain objects that are always truthy;
        # the registered callables live in ``.calls``. Older builds use a plain
        # callable or None, so handle both shapes.
        for name in ("launch_enter_hook", "launch_exit_hook"):
            hook = getattr(runtime, name, None)
            if hook is None:
                continue
            calls = getattr(hook, "calls", hook)
            if calls:
                return True
        if getattr(runtime, "debug", False):
            return True
    except Exception:
        return True
    return False


class _Fallback:
    """Regular Triton launch, used whenever the raw launcher cannot be cached."""

    __slots__ = ("fn", "grid", "constexprs")

    def __init__(self, fn, grid, constexprs):
        self.fn = fn
        self.grid = grid
        self.constexprs = constexprs

    def __call__(self, vals):
        self.fn[self.grid](*vals, **self.constexprs)


class _CachedLaunch:
    __slots__ = (
        "launch", "function", "packed", "g0", "g1", "g2", "dev", "tail",
        "fn", "grid", "constexprs", "int_args",
    )

    def __init__(self, fn, grid, launch, function, packed, tail, constexprs, int_args=False):
        self.fn = fn
        self.grid = grid
        self.constexprs = constexprs
        self.launch = launch
        self.function = function
        self.packed = packed
        self.g0 = grid[0]
        self.g1 = grid[1] if len(grid) > 1 else 1
        self.g2 = grid[2] if len(grid) > 2 else 1
        self.dev = _device()
        self.tail = tuple(tail)
        self.int_args = int_args

    def __call__(self, vals):
        if _hooks_active():
            self.fn[self.grid](*vals, **self.constexprs)
            return
        if self.int_args:
            vals = tuple(v.data_ptr() if type(v) is torch.Tensor else v for v in vals)
        self.launch(
            self.g0,
            self.g1,
            self.g2,
            _stream(self.dev),
            self.function,
            self.packed,
            None,
            None,
            None,
            *vals,
            *self.tail,
        )


def build(fn, grid, vals, constexprs, int_args=False):
    """Compile ``fn`` once for this argument specialization and cache the launcher."""
    grid = tuple(grid)
    if not _enabled() or _hooks_active() or not 1 <= len(grid) <= 3 or callable(grid):
        return _Fallback(fn, grid, constexprs)
    for value in vals:
        if isinstance(value, torch.Tensor) and (value.data_ptr() & 15):
            return _Fallback(fn, grid, constexprs)
    if getattr(fn, "used_global_vals", None):
        return _Fallback(fn, grid, constexprs)

    names = fn.arg_names
    tail = [constexprs[name] for name in names[len(vals):]]
    try:
        kernel = fn.warmup(*(list(vals) + tail), grid=grid)
    except Exception:
        return _Fallback(fn, grid, constexprs)
    if hasattr(kernel, "result"):
        kernel = kernel.result()
    if kernel is None:
        return _Fallback(fn, grid, constexprs)
    try:
        launch, function, packed = kernel.run.launch, kernel.function, kernel.packed_metadata
    except AttributeError:
        return _Fallback(fn, grid, constexprs)
    return _CachedLaunch(fn, grid, launch, function, packed, tail, constexprs, int_args)


__all__ = ["build"]
