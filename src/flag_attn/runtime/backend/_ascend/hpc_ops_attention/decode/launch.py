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

"""Host-side launch fast path for the Ascend decode kernels.

Every Triton launch goes through ``JITFunction.run``, which rebuilds the argument
binding, the specialization key and the launch metadata on every call.  Host time
is not free on this backend: measured against the FP8 decode entry it is about
0.155 ms per call, which is a large share of the small shapes (uniform_512 runs
the whole kernel in about 0.4 ms).

The prepared launch below caches everything that depends only on the kernel and
on the *specialization* of its arguments -- the compiled kernel handle, its
``run``/``function``/``packed_metadata``/``launch_metadata``, the padded grid, the
trailing argument defaults and the driver/knobs handles -- so a repeated launch
only has to rebuild the argument tuple and call ``run`` directly.

This deliberately reaches into Triton's private launch API (``kernel.run``,
``kernel.function``, ``kernel.packed_metadata``, ``kernel.launch_metadata``,
``module.driver``, ``module.knobs.runtime``).  It was validated against FlagTree
0.6.2a1 and must be re-validated after a FlagTree upgrade: a mismatch shows up as
a crash or a wrong launch rather than as a silent numeric difference, and the
unhashable-key fallback below keeps a plain ``jit_fn[grid](...)`` available.

The cache key adds the tensor shape to Triton's own "dtype + 16B alignment"
specialization.  That only ever makes the key finer (more entries, more
fallbacks), never coarser, so it cannot hit a wrong expert.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


_LAUNCH_CACHE: dict = {}


@dataclass(frozen=True)
class PreparedLaunch:
    """One specialized launch: kernel handle plus fixed parameter names/defaults."""

    kernel: object
    run: object
    function: object
    packed_metadata: dict
    launch_metadata: object
    grid: tuple
    tail: tuple  # ((name, default), ...) for parameters after *args, in signature order
    driver: object
    knobs_rt: object


def _spec_tag(value):
    """Triton specialize looks only at a tensor's dtype and 16B alignment; shape is

    Carrying shape only makes the key finer (a few more fallbacks), never coarser, so
    a wrong expert cannot be hit.  Non-tensors (int/bool/constexpr) are distinguished
    by value directly, finer than specialize's ``%16`` rule.
    """
    if hasattr(value, "data_ptr"):
        return (value.dtype, value.data_ptr() & 15, value.shape)
    return value


def _kernel_cache_key(jit_fn, grid, args, kwargs):
    parts = []
    for value in args:
        parts.append(_spec_tag(value))
    for name, value in kwargs.items():
        parts.append(name)
        parts.append(_spec_tag(value))
    return (jit_fn, len(args), grid, tuple(parts))


def prepare_kernel_launch(jit_fn, grid, args, kwargs) -> PreparedLaunch:
    # First time: Triton's own path -- specialize + compile (or cache hit) + launch.
    kernel = jit_fn[grid](*args, **kwargs)
    module = sys.modules[type(jit_fn).__module__]
    parameters = jit_fn.signature.parameters
    tail = tuple(
        (name, parameters[name].default) for name in jit_fn.arg_names[len(args):]
    )
    grid_size = len(grid)
    grid_tuple = (
        grid[0],
        grid[1] if grid_size > 1 else 1,
        grid[2] if grid_size > 2 else 1,
    )
    return PreparedLaunch(
        kernel=kernel,
        run=kernel.run,
        function=kernel.function,
        packed_metadata=kernel.packed_metadata,
        launch_metadata=kernel.launch_metadata,
        grid=grid_tuple,
        tail=tail,
        driver=module.driver,
        knobs_rt=module.knobs.runtime,
    )


def launch_kernel(jit_fn, grid, *args, **kwargs):
    """Launch a Triton kernel, skipping repetitive JITFunction.run host work on hit."""
    try:
        key = _kernel_cache_key(jit_fn, grid, args, kwargs)
    except TypeError:  # unhashable argument: old path
        return jit_fn[grid](*args, **kwargs)
    prep = _LAUNCH_CACHE.get(key)
    if prep is None:
        prep = prepare_kernel_launch(jit_fn, grid, args, kwargs)
        _LAUNCH_CACHE[key] = prep
    driver = prep.driver
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    tail = prep.tail
    values = args + tuple(
        kwargs.get(name, default) for (name, default) in tail
    )
    knobs_rt = prep.knobs_rt
    launch_enter = knobs_rt.launch_enter_hook
    metadata = None
    if launch_enter is not None:
        metadata = prep.launch_metadata(prep.grid, stream, *values)
    grid_0, grid_1, grid_2 = prep.grid
    prep.run(
        grid_0,
        grid_1,
        grid_2,
        stream,
        prep.function,
        prep.packed_metadata,
        metadata,
        launch_enter,
        knobs_rt.launch_exit_hook,
        *values,
    )
    return prep.kernel


__all__ = [
    "prepare_kernel_launch",
    "launch_kernel",
]
