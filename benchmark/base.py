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

"""Minimal benchmark harness for FlagAttention inference operators on GCU."""

from __future__ import annotations

import gc
import os
from typing import Any, Generator

import pytest
import torch
import triton


class Benchmark:
    """Measure a reference callable and an injected FlagAttention callable."""

    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES: list[tuple[int, ...]] = []
    DEFAULT_SHAPE_DESC = "shape"

    def __init__(self, op_name: str, torch_op, dtypes=None, **kwargs):
        self.op_name = op_name
        self.torch_op = torch_op
        self.gems_op = kwargs.get("gems_op")
        self.dtypes = dtypes if dtypes is not None else self.DEFAULT_DTYPES
        self.metrics = list(self.DEFAULT_METRICS)
        self.shapes = list(self.DEFAULT_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        self.to_bench_dtypes = self.dtypes
        self.to_bench_metrics = self.metrics

    def init_user_config(self) -> None:
        """Keep the shapes, dtypes, and metrics declared by the benchmark."""

        self.to_bench_dtypes = self.dtypes
        self.to_bench_metrics = self.metrics

    def set_gems(self, gems_op) -> None:
        """Set the FlagAttention implementation being benchmarked."""

        self.gems_op = gems_op

    def get_input_iter(self, dtype) -> Generator:
        raise NotImplementedError

    @staticmethod
    def unpack_to_args_kwargs(input_tuple) -> tuple[list[Any], dict[str, Any]]:
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        for item in input_tuple:
            if isinstance(item, dict):
                kwargs.update(item)
            else:
                args.append(item)
        return args, kwargs

    @staticmethod
    def _describe(value):
        if isinstance(value, torch.Tensor):
            return {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        if isinstance(value, dict):
            return {key: Benchmark._describe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(Benchmark._describe(item) for item in value)
        return value

    def record_shapes(self, *args, **kwargs):
        return {
            "args": self._describe(args),
            "kwargs": self._describe(kwargs),
        }

    @staticmethod
    def get_latency(op, *args, **kwargs) -> float:
        warmup_ms = int(os.getenv("FLAG_ATTN_KDA_BENCH_WARMUP_MS", "100"))
        repetition_ms = int(os.getenv("FLAG_ATTN_KDA_BENCH_REP_MS", "500"))
        return triton.testing.do_bench(
            lambda: op(*args, **kwargs),
            warmup=warmup_ms,
            rep=repetition_ms,
            return_mode="median",
        )

    def run(self) -> None:
        if self.gems_op is None:
            raise RuntimeError("FlagAttention benchmark callable has not been set")

        self.init_user_config()
        for dtype in self.to_bench_dtypes:
            for input_tuple in self.get_input_iter(dtype):
                args, kwargs = self.unpack_to_args_kwargs(input_tuple)
                try:
                    latency_base = self.get_latency(self.torch_op, *args, **kwargs)
                    latency = self.get_latency(self.gems_op, *args, **kwargs)
                except Exception as exc:
                    pytest.fail(f"{self.op_name} benchmark failed: {exc}")

                speedup = latency_base / latency
                shape_detail = self.record_shapes(*args, **kwargs)
                print(
                    f"Operator={self.op_name} dtype={dtype} "
                    f"latency_base={latency_base:.6f} ms "
                    f"latency={latency:.6f} ms speedup={speedup:.3f} "
                    f"inputs={shape_detail}",
                    flush=True,
                )
                assert latency > 0

                del args, kwargs, input_tuple
                gc.collect()
                if hasattr(torch, "gcu"):
                    torch.gcu.empty_cache()
