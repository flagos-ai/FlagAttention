"""Standalone latency benchmark for the GDN2 native and TLE forward paths."""

from __future__ import annotations

import argparse
import math

import torch
import triton

from flag_attn import chunk_gdn2
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
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) / math.sqrt(K)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) / math.sqrt(K)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    g = (-torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K, device="cuda", dtype=dtype)
    w = torch.rand(B, T, H, V, device="cuda", dtype=dtype)
    initial_state = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.01
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
def _tle_forward(inputs):
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
        chunk_size=16,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--full", action="store_true", help="run all migrated workload shapes")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GDN2 benchmark requires CUDA")
    if not HAS_TLE_GDN2:
        raise RuntimeError("TLE GDN2 is unavailable; install a compatible Triton TLE build")

    dtype = getattr(torch, args.dtype)
    shapes = args.shape or (ALL_SHAPES if args.full else DEFAULT_SHAPES)
    print(f"GPU: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    print("B,T,H,K,V                     native_ms      tle_ms     speedup")
    print("-" * 66)
    for shape in shapes:
        torch.manual_seed(42)
        inputs = _make_inputs(shape, dtype)
        _native_forward(inputs)
        _tle_forward(inputs)
        torch.cuda.synchronize()
        native_ms = triton.testing.do_bench(
            lambda: _native_forward(inputs), warmup=args.warmup, rep=args.rep
        )
        tle_ms = triton.testing.do_bench(
            lambda: _tle_forward(inputs), warmup=args.warmup, rep=args.rep
        )
        shape_text = ",".join(str(item) for item in shape)
        print(f"{shape_text:<28} {native_ms:>10.4f} {tle_ms:>11.4f} {native_ms / tle_ms:>10.3f}x")
        del inputs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
