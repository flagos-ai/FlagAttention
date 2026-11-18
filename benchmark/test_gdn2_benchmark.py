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


import argparse
import math

import pytest
import torch

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:
    from recording import BenchmarkRecorder

from flag_attn.testing import backend as test_backend
from flag_attn import chunk_gdn2, runtime
from flag_attn.gdn2.chunk import HAS_TLE_GDN2
from flag_attn.gdn2.native.chunk_fwd import chunk_gdn2_fwd


ALL_SHAPES = [
    (2, 512, 8, 64, 64),
    (4, 1024, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (1, 4096, 16, 64, 64),
    (1, 8192, 96, 128, 128),
    (2, 2048, 16, 256, 512),
    (2, 16384, 16, 128, 128),
    (4, 1024, 8, 256, 512),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]

DEFAULT_SHAPES = [
    (2, 512, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (2, 2048, 16, 256, 512),
]


def _parse_shape(value: str) -> tuple[int, int, int, int, int]:
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must contain integers") from error
    if len(shape) != 5 or any(item <= 0 for item in shape):
        raise argparse.ArgumentTypeError("shape must be B,T,H,K,V with positive values")
    return shape


def _make_inputs(shape: tuple[int, int, int, int, int], dtype: torch.dtype):
    B, T, H, K, V = shape
    q = torch.randn(B, T, H, K, device=test_backend.device, dtype=dtype) / math.sqrt(K)
    k = torch.randn(B, T, H, K, device=test_backend.device, dtype=dtype) / math.sqrt(K)
    v = torch.randn(B, T, H, V, device=test_backend.device, dtype=dtype)
    g = (-torch.rand(B, T, H, K, device=test_backend.device, dtype=torch.float32) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K, device=test_backend.device, dtype=dtype)
    w = torch.rand(B, T, H, V, device=test_backend.device, dtype=dtype)
    initial_state = torch.randn(B, H, K, V, device=test_backend.device, dtype=torch.float32) * 0.01
    return q, k, v, g, b, w, initial_state


@torch.inference_mode()
def _native_forward(inputs):
    q, k, v, g, b, w, initial_state = inputs
    return chunk_gdn2_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w_gate=w,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
        disable_recompute=True,
    )[:2]


@torch.inference_mode()
def _public_forward(inputs):
    q, k, v, g, b, w, initial_state = inputs
    return chunk_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=(
            16
            if HAS_TLE_GDN2 or runtime.device.vendor_name in {"mthreads", "metax"}
            else 64
        ),
    )


def run_benchmark(
    argv: list[str] = (),
    record_property=None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--full", action="store_true", help="run all migrated workload shapes")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args(argv)

    if not test_backend.supports_operator("chunk_gdn2"):
        raise RuntimeError("GDN2 benchmark requires an available accelerator")
    compare_native = HAS_TLE_GDN2
    baseline_name = "FlagAttention native Triton" if compare_native else None

    dtype = getattr(torch, args.dtype)
    shapes = args.shape or (ALL_SHAPES if args.full else DEFAULT_SHAPES)
    print(f"GPU: {test_backend.get_device_name()} | dtype: {args.dtype}")
    print(f"baseline: {baseline_name or 'unavailable'}")
    print("B,T,H,K,V                       base_ms   public_ms   base/public")
    print("-" * 66)
    recorder = BenchmarkRecorder(
        record_property,
        op_name="chunk_gdn2",
        dtype=str(dtype),
        baseline=baseline_name,
        phase="forward",
    )
    for shape in shapes:
        torch.manual_seed(42)
        inputs = _make_inputs(shape, dtype)
        if compare_native:
            _native_forward(inputs)
        _public_forward(inputs)
        test_backend.device_fn.synchronize()
        native_ms = test_backend.do_bench(
            lambda: _native_forward(inputs), warmup=args.warmup, rep=args.rep
        ) if compare_native else None
        public_ms = test_backend.do_bench(
            lambda: _public_forward(inputs), warmup=args.warmup, rep=args.rep
        )
        shape_text = ",".join(str(item) for item in shape)
        baseline_text = "N/A" if native_ms is None else f"{native_ms:.4f}"
        speedup_text = "N/A" if native_ms is None else f"{native_ms / public_ms:.3f}x"
        print(f"{shape_text:<28} {baseline_text:>10} {public_ms:>11.4f} {speedup_text:>11}")
        recorder.add(shape_detail=shape, latency_base=native_ms, latency=public_ms)
        del inputs
        test_backend.device_fn.empty_cache()
    recorder.record()


@pytest.mark.chunk_gdn2
@pytest.mark.skipif(not test_backend.supports_operator("chunk_gdn2"), reason=test_backend.skip_reason("chunk_gdn2"))
def test_chunk_gdn2_benchmark(record_property) -> None:
    run_benchmark([], record_property)
