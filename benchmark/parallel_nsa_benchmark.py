# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Controlled forward benchmark for Enflame S60 NSA.

The benchmark covers selected attention, compression attention,
and the complete compression-plus-selection pipeline.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch_gcu

from flag_attn.runtime.backend._enflame.nsa import (
    parallel_nsa,
)
from flag_attn.runtime.backend._enflame.nsa.parallel_nsa_compression import (
    parallel_nsa_compression,
)


@dataclass(frozen=True)
class BenchmarkCase:
    batch: int
    tokens: int
    kv_heads: int
    query_heads: int
    head_dim: int


CASES = {
    "SMOKE": BenchmarkCase(
        batch=1,
        tokens=512,
        kv_heads=2,
        query_heads=32,
        head_dim=64,
    ),
    "H4_S2K": BenchmarkCase(
        batch=1,
        tokens=2048,
        kv_heads=4,
        query_heads=64,
        head_dim=64,
    ),
    "H16_S2K": BenchmarkCase(
        batch=1,
        tokens=2048,
        kv_heads=16,
        query_heads=256,
        head_dim=64,
    ),
    "H16_S8K": BenchmarkCase(
        batch=1,
        tokens=8192,
        kv_heads=16,
        query_heads=256,
        head_dim=64,
    ),
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def synchronize() -> None:
    torch.gcu.synchronize()


def build_block_indices(
    case: BenchmarkCase,
    selected_blocks: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    token_ids = torch.arange(
        case.tokens,
        dtype=torch.int32,
        device=device,
    )
    valid_blocks = torch.div(
        token_ids + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).clamp_min(1)

    block_ids = torch.arange(
        selected_blocks,
        dtype=torch.int32,
        device=device,
    ).view(1, 1, 1, selected_blocks)

    valid = (
        block_ids
        < valid_blocks.view(
            1,
            case.tokens,
            1,
            1,
        )
    )

    sentinel = torch.full_like(
        block_ids,
        case.tokens,
    )

    return torch.where(
        valid,
        block_ids,
        sentinel,
    ).expand(
        case.batch,
        case.tokens,
        case.kv_heads,
        selected_blocks,
    ).contiguous()


def measure(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    inner: int,
) -> list[float]:
    for _ in range(warmup):
        for _ in range(inner):
            function()
        synchronize()

    samples = []

    for iteration in range(1, repeat + 1):
        synchronize()
        start = time.perf_counter()

        for _ in range(inner):
            function()

        synchronize()
        latency_ms = (
            (time.perf_counter() - start)
            * 1000.0
            / inner
        )
        samples.append(latency_ms)

        print(
            "ITERATION",
            f"index={iteration}",
            f"inner={inner}",
            f"latency_ms={latency_ms:.6f}",
            flush=True,
        )

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(
            "selected",
            "compression",
            "full",
        ),
        default="selected",
    )
    parser.add_argument(
        "--case",
        choices=tuple(CASES),
        default="SMOKE",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP),
        default="bfloat16",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=int(
            os.environ.get(
                "S60_TEST_DEVICE",
                "0",
            )
        ),
    )
    parser.add_argument(
        "--selected-blocks",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--inner",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeat <= 0:
        raise ValueError("repeat must be positive")
    if args.inner <= 0:
        raise ValueError("inner must be positive")

    torch.gcu.set_device(args.device_index)
    torch.manual_seed(42)

    case = CASES[args.case]
    dtype = DTYPE_MAP[args.dtype]
    device = torch.device(
        f"gcu:{args.device_index}"
    )
    scale = case.head_dim ** -0.5

    q = torch.randn(
        (
            case.batch,
            case.tokens,
            case.query_heads,
            case.head_dim,
        ),
        dtype=dtype,
        device=device,
    )

    if args.mode == "compression":
        compressed_tokens = (
            case.tokens + args.block_size - 1
        ) // args.block_size

        k = torch.randn(
            (
                case.batch,
                compressed_tokens,
                case.kv_heads,
                case.head_dim,
            ),
            dtype=dtype,
            device=device,
        )
        v = torch.randn_like(k)

        @torch.no_grad()
        def run() -> torch.Tensor:
            output, _ = parallel_nsa_compression(
                q=q,
                k=k,
                v=v,
                block_size=args.block_size,
                scale=scale,
            )
            return output

        pipeline = "compression attention"

    else:
        k = torch.randn(
            (
                case.batch,
                case.tokens,
                case.kv_heads,
                case.head_dim,
            ),
            dtype=dtype,
            device=device,
        )
        v = torch.randn_like(k)

        if args.mode == "selected":
            block_indices = build_block_indices(
                case,
                args.selected_blocks,
                args.block_size,
                device,
            )

            @torch.no_grad()
            def run() -> torch.Tensor:
                return parallel_nsa(
                    q=q,
                    k=k,
                    v=v,
                    block_indices=block_indices,
                    block_counts=(
                        args.selected_blocks
                    ),
                    block_size=args.block_size,
                    scale=scale,
                )

            pipeline = "selected sparse attention"

        else:
            g_cmp = torch.sigmoid(
                torch.randn(
                    (
                        case.batch,
                        case.tokens,
                        case.query_heads,
                    ),
                    dtype=dtype,
                    device=device,
                )
            )
            g_slc = torch.sigmoid(
                torch.randn_like(g_cmp)
            )

            @torch.no_grad()
            def run() -> torch.Tensor:
                return parallel_nsa(
                    q=q,
                    k=k,
                    v=v,
                    g_cmp=g_cmp,
                    g_slc=g_slc,
                    block_counts=(
                        args.selected_blocks
                    ),
                    block_size=args.block_size,
                    scale=scale,
                )

            pipeline = (
                "compression + topk "
                "+ selected attention"
            )

    print("===== S60 NSA FORMAL BENCHMARK =====")
    print("mode:", args.mode)
    print("case:", args.case)
    print("device:", args.device_index)
    print(
        "shape:",
        (
            case.batch,
            case.tokens,
            case.kv_heads,
            case.query_heads,
            case.head_dim,
        ),
    )
    print(
        "selected_blocks:",
        args.selected_blocks,
    )
    print("block_size:", args.block_size)
    print("dtype:", dtype)
    print("pipeline:", pipeline)
    print("warmup:", args.warmup)
    print("repeat:", args.repeat)
    print("inner:", args.inner)

    output = run()
    synchronize()

    finite = bool(
        torch.isfinite(
            output.float()
        ).all()
    )

    print(
        "output_finite:",
        finite,
        flush=True,
    )

    if not finite:
        raise AssertionError(
            "NSA benchmark output is non-finite"
        )

    samples = measure(
        run,
        warmup=args.warmup,
        repeat=args.repeat,
        inner=args.inner,
    )

    median = statistics.median(samples)

    print(
        "RESULT",
        "operator=nsa",
        "stage=forward",
        f"mode={args.mode}",
        f"case={args.case}",
        f"dtype={args.dtype}",
        f"median_ms={median:.6f}",
        f"min_ms={min(samples):.6f}",
        f"max_ms={max(samples):.6f}",
        flush=True,
    )
    print(
        "S60 NSA FORMAL BENCHMARK PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
