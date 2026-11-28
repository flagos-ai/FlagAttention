# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral compiled launch metadata and opt-in tracing."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

_trace = ContextVar("v76_launch_trace", default=None)


@contextmanager
def capture():
    records = []
    token = _trace.set(records)
    try:
        yield records
    finally:
        _trace.reset(token)


def cdiv(x, y):
    return (x + y - 1) // y


def pow2(x):
    return 1 << (x - 1).bit_length()


class LaunchCache:
    def __init__(self, fn, limit=256):
        from triton import knobs
        from triton.runtime import jit as runtime_jit

        self.knobs = knobs
        self.fn = fn
        self.limit = limit
        self.plans = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.dist_context = getattr(runtime_jit, "DistributedRtContext", None)

    def __call__(self, key, grid, args, kwargs, fast=True):
        fn = self.fn
        knobs = self.knobs
        # Preserve ordinary JIT behavior for debug/instrumented/hooked/distributed use.
        allowed = (
            fast and not fn.pre_run_hooks and not knobs.runtime.debug and not knobs.compilation.instrumentation_mode
        )
        if self.dist_context is not None and self.dist_context().is_lite_mode:
            allowed = False
        plan = self.plans.get(key) if allowed else None
        if plan is None:
            compiled = fn[grid](*args, **kwargs)
            self.misses += 1
            if allowed:
                tail = tuple(kwargs[n] for n in fn.arg_names[len(args) :])
                assert all(isinstance(x, (int, float, bool, str)) for x in tail)
                canonical = tuple(grid) + (1,) * (3 - len(grid))
                # Runner retains code/grid metadata only, not this call's arguments.
                self.plans[key] = (compiled, compiled[canonical], tail)
                if len(self.plans) > self.limit:
                    self.plans.popitem(last=False)
        else:
            compiled, runner, tail = plan
            runner(*args, *tail)
            self.hits += 1
        records = _trace.get()
        if records is not None:
            records.append((fn, compiled, grid, dict(kwargs), plan is not None))
        return compiled


def record_launch(fn, compiled, grid, kwargs):
    records = _trace.get()
    if records is not None:
        records.append((fn, compiled, grid, dict(kwargs), False))
