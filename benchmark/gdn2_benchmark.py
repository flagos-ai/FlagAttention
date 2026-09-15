"""Steady-state latency of NVIDIA TLE GDN2 (no native baseline dependency)."""

import argparse

import torch
import triton

from flag_attn import chunk_gdn2

SHAPES = [
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


def benchmark(warmup=20, rep=50):
    if not torch.cuda.is_available():
        raise RuntimeError("GDN2 benchmark requires NVIDIA CUDA with Triton TLE")
    torch.manual_seed(42)
    print("B\tT\tH\tK\tV\tdtype\tlatency_ms")
    for dtype in (torch.float16, torch.bfloat16):
        for B, T, H, K, V in SHAPES:
            q = torch.randn(B, T, H, K, dtype=dtype, device="cuda") / K**0.5
            k = torch.randn_like(q) / K**0.5
            v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
            g = -torch.rand_like(q) * 0.1
            b, w = torch.rand_like(q), torch.rand_like(v)
            h0 = torch.randn(B, H, K, V, device="cuda") * 0.1

            def run():
                return chunk_gdn2(q, k, v, g, b, w, initial_state=h0, output_final_state=True)

            ms = triton.testing.do_bench(run, warmup=warmup, rep=rep, return_mode="median")
            print(f"{B}\t{T}\t{H}\t{K}\t{V}\t{dtype}\t{ms:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20, help="warmup duration in ms")
    parser.add_argument("--rep", type=int, default=50, help="measurement duration in ms")
    args = parser.parse_args()
    benchmark(args.warmup, args.rep)
