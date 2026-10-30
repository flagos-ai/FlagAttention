# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit cached launcher for this backend's Triton kernels.

The GDN fast path is host-bound on small shapes: four ``kernel[grid](...)``
calls per forward, and every call re-runs Triton's Python side (option/kwargs
parsing, device/stream lookup, generated binder, cache-key build, global-value
check, grid canonicalization).  That Python work outweighs the actual C launch
on this backend, so caching the compiled launcher removes a large part of the
per-call host cost.

This module exposes one explicit entry point, :func:`fast_launch`, that a call
site opts into: the first call of a site compiles the kernel (Triton's
compile-only ``warmup``) and keeps the compiled launcher; later calls with the
same site invoke that launcher directly.  Nothing is patched globally, so
kernels outside the opt-in call sites keep Triton's regular launch path.

Every call site keeps the regular ``kernel[grid](*args, ...)`` as a fallback.
The launcher uses it when the mechanism is disabled
(``FLAG_ATTN_ASCEND_GDN_FASTLAUNCH=0``), when a profiler/launch hook or
``debug`` is active, when a tensor argument is not 16-byte aligned (the
compile-only ``warmup`` path models pointers as divisible), when the grid is
callable, when the kernel reads global values, or when the installed Triton
build does not expose the internals used here.  A fallback is correct but
slower; it does not change results.

The per-site cache is keyed by ``(function identity, grid, num_warps,
num_stages)``.  Kernels and grids are shape-specialized at the call site, so a
site corresponds to one shape/config.  Arguments may be passed positionally or
by keyword (keyword arguments are folded into the kernel signature order).
"""

from __future__ import annotations

import os
import threading

import torch
import triton

_OFF = {"0", "false", "off", "no"}
_MAX_SITES = 2048
_SITES: dict[tuple, "_Site"] = {}
_LAUNCHES: dict[tuple, "_KeyedLaunch"] = {}
_MISSING = object()

try:  # pragma: no cover - availability depends on the torch_npu build
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _raw_stream
    _HAS_RAW_STREAM = True
except Exception:  # noqa: BLE001
    _raw_stream = None
    _HAS_RAW_STREAM = False

try:  # pragma: no cover
    from torch_npu._C import _npu_getDevice as _get_device
except Exception:  # noqa: BLE001
    _get_device = None


class _Scope(threading.local):
    """Per-thread forward scope: cache the device/stream for one forward call."""

    stream = None
    dev = None
    fast = None


_SCOPE = _Scope()


def _enabled() -> bool:
    return os.environ.get(
        "FLAG_ATTN_ASCEND_GDN_FASTLAUNCH", "1"
    ).lower() not in _OFF


def _device() -> int:
    if _get_device is not None:
        try:
            return _get_device()
        except Exception:  # noqa: BLE001
            pass
    return torch.npu.current_device()


def _stream(dev: int) -> int:
    if _HAS_RAW_STREAM:
        return _raw_stream(dev)
    from triton.runtime.driver import driver
    return driver.active.get_current_stream(dev)


def _hooks_active() -> bool:
    """True when a profiler/trace hook (or debug) needs the regular path.

    The cached low-level launch passes ``launch_metadata=None``; a registered
    profiler hook expects the metadata object, so fall back while one is set.
    """
    try:
        runtime = triton.knobs.runtime
        if getattr(runtime.launch_enter_hook, "calls", None):
            return True
        if getattr(runtime.launch_exit_hook, "calls", None):
            return True
        if getattr(runtime, "debug", False):
            return True
    except Exception:  # noqa: BLE001 - unknown knob layout: be conservative
        return True
    return False


def begin_forward() -> None:
    """Cache the device, stream and enable flag once for a forward call."""
    fast = _enabled()
    _SCOPE.fast = fast
    if fast:
        dev = _device()
        _SCOPE.dev = dev
        _SCOPE.stream = _stream(dev)
    else:
        _SCOPE.dev = None
        _SCOPE.stream = None


def end_forward() -> None:
    _SCOPE.stream = None
    _SCOPE.dev = None
    _SCOPE.fast = None


def _fast_enabled() -> bool:
    """Env switch, read once per forward instead of once per launch."""
    fast = _SCOPE.fast
    return _enabled() if fast is None else fast


def _bind(fn, args, kwargs):
    """Fold keyword kernel arguments into the canonical positional list.

    Returns ``None`` when the call cannot be represented positionally (missing
    trailing arguments, too many arguments, no signature information); the
    caller then uses the regular launch path.
    """
    if not kwargs:
        return list(args)
    names = getattr(fn, "arg_names", None)
    if not names or len(args) > len(names):
        return None
    vals = list(args)
    for j in range(len(vals), len(names)):
        if names[j] not in kwargs:
            return None
        vals.append(kwargs[names[j]])
    return vals


class _Site:
    """One ``(function, grid, options)`` launch site.

    The compiled launcher is cached per *specialization* (argument dtypes,
    pointer alignment, scalar/constexpr values and device), because the same
    function/grid/options pair is used for different dtypes (bf16/fp16),
    varlen flags and chunk sizes.
    """

    __slots__ = ("fn", "grid", "options", "entries", "keyed")

    def __init__(self, fn, grid, options):
        self.fn = fn
        self.grid = tuple(grid)
        self.options = options
        self.entries: dict[tuple, tuple] = {}
        self.keyed: dict[tuple, tuple | None] = {}

    def _regular(self, args, kwargs):
        return self.fn[self.grid](*args, **kwargs, **self.options)

    def _prepare(self, vals):
        """Compile the kernel with the regular path (no launch) and return it."""
        try:
            grid = self.grid
            if not (1 <= len(grid) <= 3) or callable(grid):
                return None
            if getattr(self.fn, "used_global_vals", None):
                return None
            for a in vals:
                if isinstance(a, torch.Tensor) and (a.data_ptr() & 15):
                    # ``warmup`` models pointers as 16-byte aligned; a
                    # misaligned real pointer needs a different kernel.
                    return None
            kern = self.fn.warmup(*vals, grid=grid, **self.options)
        except Exception:  # noqa: BLE001 - any incompatibility: regular path
            return None
        if hasattr(kern, "result"):
            kern = kern.result()
        if kern is None:
            return None
        try:
            return (kern.run.launch, kern.function, kern.packed_metadata)
        except AttributeError:  # pragma: no cover - other Triton builds
            return None

    def __call__(self, *args, **kwargs):
        if not _fast_enabled():
            return self._regular(args, kwargs)
        vals = _bind(self.fn, args, kwargs)
        if vals is None:
            return self._regular(args, kwargs)
        try:
            dev = _SCOPE.dev
            if dev is None:
                dev = _device()
            key = [dev]
            for a in vals:
                if isinstance(a, torch.Tensor):
                    key.append((a.dtype, a.data_ptr() & 15))
                else:
                    hash(a)  # unhashable argument: regular path
                    key.append(a)
            key = tuple(key)
        except TypeError:
            return self._regular(args, kwargs)
        entry = self.entries.get(key)
        if entry is None:
            if _hooks_active():
                return self._regular(args, kwargs)
            entry = self._prepare(vals)
            if entry is None:
                return self._regular(args, kwargs)
            if len(self.entries) >= 128:
                self.entries.pop(next(iter(self.entries)))
            self.entries[key] = entry
        elif _hooks_active():
            return self._regular(args, kwargs)
        stream = _SCOPE.stream
        if stream is None:
            stream = _stream(dev)
        launch, function, packed = entry
        grid = self.grid
        launch(grid[0], grid[1] if len(grid) > 1 else 1,
               grid[2] if len(grid) > 2 else 1, stream, function, packed,
               None, None, None, *vals)
        return None


class _KeyedLaunch:
    """Launch one preparation whose specialization key is supplied by the site.

    Used by call sites that already know their shape/flags (plus dtype), so the
    hot path is a dictionary lookup and the low-level launch: no per-call
    argument inspection at all.
    """

    __slots__ = ("site", "key", "g0", "g1", "g2")

    def __init__(self, site, key):
        self.site = site
        self.key = key
        grid = site.grid
        self.g0 = grid[0]
        self.g1 = grid[1] if len(grid) > 1 else 1
        self.g2 = grid[2] if len(grid) > 2 else 1

    def __call__(self, *args, **kwargs):
        site = self.site
        if not _fast_enabled():
            return site._regular(args, kwargs)
        entry = site.keyed.get(self.key, _MISSING)
        if entry is _MISSING:
            if _hooks_active():
                return site._regular(args, kwargs)
            vals = _bind(site.fn, args, kwargs)
            if vals is None:
                return site._regular(args, kwargs)
            prepared = site._prepare(vals)
            # ``False`` marks this key as permanently unsupported (e.g. a
            # misaligned pointer); do not retry the compile on every call.
            site.keyed[self.key] = prepared if prepared is not None else False
            if prepared is None:
                return site._regular(args, kwargs)
            entry = prepared
        elif entry is False:
            return site._regular(args, kwargs)
        elif _hooks_active():
            return site._regular(args, kwargs)
        vals = args if not kwargs else _bind(site.fn, args, kwargs)
        if vals is None:
            return site._regular(args, kwargs)
        stream = _SCOPE.stream
        if stream is None:
            dev = _SCOPE.dev
            stream = _stream(dev if dev is not None else _device())
        launch, function, packed = entry
        launch(self.g0, self.g1, self.g2, stream, function, packed,
               None, None, None, *vals)
        return None


def fast_launch(fn, grid, num_warps: int, num_stages: int, key=None,
                cache=None):
    """Return the cached opt-in launcher for one kernel site.

    ``fn`` is a ``triton.jit`` function, ``grid`` its concrete launch grid and
    ``num_warps``/``num_stages`` its launch options.  The returned object is
    called exactly like the regular ``fn[grid]`` with the same kernel arguments
    (positional or keyword).
    """
    if cache is not None and key is not None:
        # Hot path: the caller owns a per-shape dictionary next to its scratch
        # buffers, so the lookup key carries no shape values.
        wrapper = cache.get(key)
        if wrapper is None:
            wrapper = fast_launch(fn, grid, num_warps, num_stages, key)
            cache[key] = wrapper
        return wrapper
    options = {"num_warps": num_warps, "num_stages": num_stages}
    site_key = (id(fn), grid if type(grid) is tuple else tuple(grid),
                num_warps, num_stages)
    site = _SITES.get(site_key)
    if site is None or site.fn is not fn:
        if len(_SITES) >= _MAX_SITES:
            _SITES.pop(next(iter(_SITES)))
        site = _Site(fn, grid, options)
        _SITES[site_key] = site
    if key is not None:
        launch_key = (site_key, key)
        wrapper = _LAUNCHES.get(launch_key)
        if wrapper is None:
            if len(_LAUNCHES) >= _MAX_SITES:
                _LAUNCHES.pop(next(iter(_LAUNCHES)))
            wrapper = _KeyedLaunch(site, key)
            _LAUNCHES[launch_key] = wrapper
        return wrapper
    return site


__all__ = ["fast_launch", "begin_forward", "end_forward"]
