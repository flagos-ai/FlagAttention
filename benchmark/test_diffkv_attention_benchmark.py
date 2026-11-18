# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""DiffKV Full Attention and SWA decode benchmarks, organized like chunk GLA.

Run with pytest -v -s -m diffkv_attention from the repository root. On NVIDIA,
the vLLM FA3 extension provides the local comparison baseline when available.

Generic 2D/3D paths retain preallocated buffers and the original timing
protocol. Registered vendor implementations use their public API and report
full-call latency separately. FLAG_ATTN_DIFFKV_BACKEND selects the generic
TLE/standard Triton implementation; runtime selects the device vendor.
"""

import math
import os
import site
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import triton

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:  # Execution from the benchmark directory.
    from recording import BenchmarkRecorder

from flag_attn.testing import backend as test_backend
from flag_attn.diffkv_attention import api as diffkv_impl
from flag_attn.diffkv_attention.api import DEFAULT_LAYOUT, LAUNCH, OP_NAME, SUPPORTED_PATHS


DEFAULT_WARMUP = 100
DEFAULT_REP = 500
DEFAULT_SAMPLES = 9
STABILITY_CV_THRESHOLD_PCT = 5.0
DEFAULT_SEED = 0
PAGE_TABLE = "random"
WINDOW_SIZE = 128
TABLE_WIDTH = 130

# Preserve the original Full Attention/SWA matrix and production head layout.
_SHAPES = [(batch, length) for batch in (1, 8, 16, 32) for length in (512, 2048, 8192, 32768)]
_DTYPES = [torch.bfloat16]
_MODES = ("full", "swa")
_PATHS = ("2d", "3d")


def _fa3_op_available() -> bool:
    try:
        namespace = getattr(torch.ops, "_vllm_fa3_C", None)
        return namespace is not None and hasattr(namespace, "fwd")
    except Exception:
        return False


def _load_fa3_provider() -> tuple[bool, str | None]:
    """Load the vLLM FA3 extension from the active Python environment."""
    if _fa3_op_available():
        return True, "already registered"

    environment = Path(sys.prefix).resolve()
    configured = os.environ.get("VLLM_FLASH_ATTN_EXTENSION_DIR")
    roots = (
        [Path(configured).expanduser()]
        if configured
        else [
            root
            for directory in site.getsitepackages()
            for root in (Path(directory) / "fa3_runtime", Path(directory))
        ]
    )
    errors = []
    for root in roots:
        root = root.resolve()
        if not root.is_relative_to(environment):
            continue
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = [
                root / "_vllm_fa3_C.abi3.so",
                *sorted(root.glob("*_vllm_fa3_C*.so")),
            ]
        else:
            continue
        for candidate in dict.fromkeys(candidates):
            if not candidate.is_file():
                continue
            try:
                torch.ops.load_library(str(candidate))
            except Exception as exc:  # The extension may have a different ABI.
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
                continue
            if _fa3_op_available():
                return True, str(candidate)
            errors.append(f"{candidate}: _vllm_fa3_C.fwd was not registered")
    reason = "; ".join(errors) or "FA3 extension not found in the active Python environment"
    return False, reason


def shape_seed(base_seed: int, mode: str, batch: int, seq_len: int, num_kv_heads: int) -> int:
    """Derive a stable per-shape seed without relying on Python's hash()."""
    mode_offset = 0 if mode == "full" else 1_000_003
    return (base_seed + mode_offset + batch * 1_009 + seq_len * 9_176 + num_kv_heads * 65_537) % (2**63 - 1)


def make_inputs(
    batch: int,
    seq_len: int,
    dtype: torch.dtype,
    num_query_heads: int,
    num_kv_heads: int,
    head_size_qk: int,
    head_size_v: int,
    block_size: int,
    *,
    seed: int,
    page_table: str,
    device: str,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    query = torch.randn(
        batch,
        num_query_heads,
        head_size_qk,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    # Preserve production packed HND storage, exposed as strided NHD views.
    kv_cache = torch.randn(
        batch * blocks_per_seq,
        num_kv_heads,
        block_size,
        head_size_qk + head_size_v,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    kv_cache_nhd = kv_cache.transpose(1, 2)
    key_cache = kv_cache_nhd[..., :head_size_qk]
    value_cache = kv_cache_nhd[..., head_size_qk:]
    if page_table == "identity":
        physical_pages = torch.arange(batch * blocks_per_seq, device=device, dtype=torch.int32)
    else:
        physical_pages = torch.randperm(
            batch * blocks_per_seq,
            device=device,
            dtype=torch.int64,
            generator=generator,
        ).to(torch.int32)
    block_tables = physical_pages.reshape(batch, blocks_per_seq)
    context_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    return query, key_cache, value_cache, context_lens, block_tables


def build_runner(inputs, path: str, window_size: int, diffkv_impl):
    """Build a runner with the unified TLE/standard backend dispatcher."""
    query, key_cache, value_cache, context_lens, block_tables = inputs
    if test_backend.has_specialization("diffkv_attention"):
        from flag_attn import diffkv_attention

        def run_public():
            return diffkv_attention(*inputs, window_size=window_size, path=path)

        return run_public, f"{test_backend.runtime.device.vendor_name} public API (full call)"
    batch, hq, dqk = query.shape
    hkv = key_cache.shape[2]
    dv = value_cache.shape[-1]
    seq_len = int(context_lens.max().item())
    block_size = key_cache.shape[1]
    cu_seqlens_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32)
    scale = dqk**-0.5
    if path not in SUPPORTED_PATHS:
        raise ValueError(f"unsupported path: {path}")
    use_3d = path == "3d"

    num_q_per_kv = hq // hkv
    block_m = LAUNCH.query_block_m if num_q_per_kv <= LAUNCH.query_block_m else triton.next_power_of_2(num_q_per_kv)
    block_q = block_m // num_q_per_kv
    total_num_q_blocks = query.shape[0] // block_q + batch
    properties = test_backend.device_fn.get_device_properties(query.device)
    num_sms = getattr(properties, "multi_processor_count", None)

    # Backend selection is environment-based: the benchmark keeps TLE and
    # standard Triton as explicit, reproducible comparison modes.
    use_optimized = diffkv_impl.USE_TLE
    impl_name = "TLE" if use_optimized else "non-TLE"
    selected_backend = diffkv_impl.SELECTED_BACKEND
    if use_3d:
        if use_optimized:
            num_segments = diffkv_impl.get_num_par_softmax_segments(
                seq_len,
                batch,
                True,
                total_num_q_blocks=total_num_q_blocks,
                num_kv_heads=hkv,
                num_sms=num_sms,
                block_size=block_size,
            )
        else:
            num_segments = diffkv_impl.get_num_par_softmax_segments(seq_len, batch, True)
        padded_v = triton.next_power_of_2(value_cache.shape[-1])
        segm_output = torch.empty(
            batch,
            hq,
            num_segments,
            padded_v,
            device=test_backend.device,
            dtype=query.dtype if use_optimized else torch.float32,
        )
        segm_max = torch.empty(batch, hq, num_segments, device=test_backend.device, dtype=torch.float32)
        segm_expsum = torch.empty_like(segm_max)
        threshold = batch
    else:
        num_segments = None
        segm_output = segm_max = segm_expsum = None
        threshold = None

    triton_window = (window_size - 1, 0) if window_size > 0 else (-1, -1)
    out = torch.empty(batch, hq, value_cache.shape[-1], device=test_backend.device, dtype=query.dtype)
    extra_kwargs = {}
    if (
        use_optimized
        and use_3d
        and diffkv_impl.should_use_tle_fused_reducer(dqk, dv, 1, seq_len, batch, block_size, True)
    ):
        extra_kwargs["fused_reducer_counter"] = torch.zeros(batch * hq, device=test_backend.device, dtype=torch.int32)

    def run():
        diffkv_impl.unified_attention_diffkv(
            q=query,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=context_lens,
            softmax_scale=scale,
            causal=True,
            window_size=triton_window,
            block_table=block_tables,
            softcap=0.0,
            max_seqlen_q=1,
            seq_threshold_3D=threshold,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=segm_output,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
            max_seqlen_k=seq_len,
            backend=selected_backend,
            path=path,
            **extra_kwargs,
        )

    return run, impl_name


def build_fa3_runner(inputs, window_size: int):
    """Build the direct FA3 paged-cache call with preallocated buffers."""
    query, key_cache, value_cache, context_lens, block_tables = inputs
    batch, hq, dqk = query.shape
    dv = value_cache.shape[-1]
    seq_len = int(context_lens.max().item())
    cu_seqlens_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32)
    out = torch.empty(batch, hq, dv, device=query.device, dtype=query.dtype)
    scale = dqk**-0.5
    window_left, window_right = (
        (window_size - 1, 0) if window_size > 0 else (-1, -1)
    )

    scheduler_metadata = None
    try:
        scheduler_metadata = torch.ops._vllm_fa3_C.get_scheduler_metadata(
            batch,
            1,
            seq_len,
            hq,
            key_cache.shape[2],
            dqk,
            dv,
            query.dtype,
            context_lens,
            cu_seqlens_q,
            None,
            None,
            None,
            None,
            key_cache.shape[1],
            0,
            True,
            window_left,
            window_right,
            False,
            0,
            None,
            0,
        )
    except (AttributeError, RuntimeError):
        # Older extensions may choose their schedule in fwd.
        pass

    def run():
        torch.ops._vllm_fa3_C.fwd(
            query,
            key_cache,
            value_cache,
            None,
            None,
            None,
            out,
            cu_seqlens_q,
            None,
            None,
            None,
            context_lens,
            1,
            seq_len,
            block_tables,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            scale,
            True,
            window_left,
            window_right,
            0.0,
            True,
            scheduler_metadata,
            0,
            None,
            0,
            None,
            1,
            0,
            None,
        )

    return run


def _bench_ms(fn, warmup_ms: int, rep_ms: int, samples: int):
    """Measure a callable with the selected device timing helper.

    ``do_bench`` owns warmup, CUDA-event placement, synchronization, adaptive
    repetition count, and benchmark-cache clearing.  We invoke it once per
    sample and reduce each returned list of per-repetition millisecond values
    to a sample median.  The outer sample distribution is retained for the
    stability CV reported by this benchmark.
    """
    values = []
    for _ in range(samples):
        timings_ms = test_backend.do_bench(
            fn,
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="all",
        )
        if not isinstance(timings_ms, (list, tuple)):
            timings_ms = [timings_ms]
        values.append(statistics.median(float(value) for value in timings_ms))
    median = statistics.median(values)
    mean = statistics.fmean(values)
    cv_pct = 100.0 * statistics.pstdev(values) / mean if mean else 0.0
    return median, min(values), max(values), cv_pct


def _print_header(
    mode: str, scope: str, warmup: int, rep: int, samples: int, fa3_available: bool
) -> None:
    title = "Full Attention" if mode == "full" else "SWA"
    print(f"\n{'=' * TABLE_WIDTH}")
    print(f"  diffkv_attention benchmark - {title}")
    print(f"  device={test_backend.get_device_name()} scope={scope}")
    baseline = "FA3" if fa3_available else "none"
    print(f"  warmup={warmup}ms rep={rep}ms samples={samples}; baseline: {baseline}")
    print(f"{'=' * TABLE_WIDTH}")
    print(
        f"{'B':>3} {'KV':>6} {'HQ':>4} {'HKV':>4} {'DQK':>4} {'DV':>4} "
        f"{'dtype':>8} {'path':>5} {'impl':>13} "
        f"{'flag_attn(ms)':>14} {'min(ms)':>10} {'max(ms)':>10} {'CV(%)':>8} "
        f"{'FA3(ms)':>10} {'FA3/flag':>10}"
    )


def _print_row(
    batch, length, kv_heads, dtype, path, implementation, measurement, fa3_latency
) -> None:
    latency, minimum, maximum, cv_pct = measurement
    fa3_text = f"{fa3_latency:>10.6f}" if fa3_latency is not None else f"{'n/a':>10}"
    speedup_text = (
        f"{fa3_latency / latency:>10.3f}"
        if fa3_latency is not None else f"{'n/a':>10}"
    )
    print(
        f"{batch:>3} {length:>6} {DEFAULT_LAYOUT.num_query_heads:>4} {kv_heads:>4} "
        f"{DEFAULT_LAYOUT.head_size_qk:>4} {DEFAULT_LAYOUT.head_size_v:>4} "
        f"{str(dtype).removeprefix('torch.'):>8} {path.upper():>5} {implementation:>13} "
        f"{latency:>14.6f} {minimum:>10.6f} {maximum:>10.6f} {cv_pct:>8.3f} "
        f"{fa3_text} {speedup_text}",
        flush=True,
    )
    if cv_pct > STABILITY_CV_THRESHOLD_PCT:
        print(f"[WARN unstable] {path.upper()} sample CV={cv_pct:.3f}% exceeds {STABILITY_CV_THRESHOLD_PCT:.1f}%")


@torch.inference_mode()
def run_benchmark(
    warmup: int = DEFAULT_WARMUP,
    rep: int = DEFAULT_REP,
    record_property: Callable[[str, object], None] | None = None,
    *,
    samples: int = DEFAULT_SAMPLES,
) -> None:
    if not test_backend.supports_operator(OP_NAME):
        raise RuntimeError(test_backend.skip_reason(OP_NAME))
    if warmup < 0 or rep <= 0 or samples <= 0:
        raise ValueError("warmup must be non-negative; rep and samples must be positive")

    specialized = test_backend.has_specialization(OP_NAME)
    scope = "public_api_forward" if specialized else "preallocated_forward_kernel"
    mode_name = "api" if specialized else "kernel"
    if specialized:
        implementation_name = test_backend.runtime.device.vendor_name
    else:
        implementation_name = diffkv_impl.SELECTED_BACKEND
        print(f"DiffKV implementation: {implementation_name}")

    fa3_available, fa3_info = (
        _load_fa3_provider() if test_backend.is_nvidia() else (False, None)
    )
    if fa3_info is not None:
        status = "provider" if fa3_available else "unavailable"
        print(f"FA3 baseline {status}: {fa3_info}")

    # Full and SWA are both decode workloads. Submit one record per dtype so
    # summary readers that group by op_name/dtype retain every measured row.
    recorders = {
        dtype: BenchmarkRecorder(
            record_property,
            op_name=OP_NAME,
            dtype=str(dtype),
            mode=mode_name,
            baseline="FA3" if fa3_available else None,
            phase="decode",
            benchmark_scope=scope,
            vendor=test_backend.runtime.device.vendor_name,
        )
        for dtype in _DTYPES
    }

    for mode in _MODES:
        window = -1 if mode == "full" else WINDOW_SIZE
        kv_heads = diffkv_impl.num_kv_heads_for_mode(mode)
        _print_header(mode, scope, warmup, rep, samples, fa3_available)
        for dtype in _DTYPES:
            recorder = recorders[dtype]
            for batch, length in _SHAPES:
                inputs = make_inputs(
                    batch,
                    length,
                    dtype,
                    DEFAULT_LAYOUT.num_query_heads,
                    kv_heads,
                    DEFAULT_LAYOUT.head_size_qk,
                    DEFAULT_LAYOUT.head_size_v,
                    DEFAULT_LAYOUT.block_size,
                    seed=shape_seed(DEFAULT_SEED, mode, batch, length, kv_heads),
                    page_table=PAGE_TABLE,
                    device=test_backend.device,
                )
                fa3_latency = None
                if fa3_available:
                    fa3_latency = _bench_ms(
                        build_fa3_runner(inputs, window), warmup, rep, samples
                    )[0]
                    if not math.isfinite(fa3_latency) or fa3_latency <= 0:
                        raise RuntimeError(
                            f"Invalid FA3 latency for {mode} B={batch} KV={length}"
                        )
                for path in _PATHS:
                    runner, _ = build_runner(inputs, path, window, diffkv_impl)
                    measurement = _bench_ms(runner, warmup, rep, samples)
                    latency, minimum, maximum, cv_pct = measurement
                    if not math.isfinite(latency) or latency <= 0:
                        raise RuntimeError(f"Invalid DiffKV latency for {mode} B={batch} KV={length} {path}")
                    _print_row(
                        batch, length, kv_heads, dtype, path,
                        implementation_name, measurement, fa3_latency,
                    )
                    recorder.add(
                        shape_detail={
                            "mode": mode,
                            "batch": batch,
                            "query_length": 1,
                            "kv_length": length,
                            "num_query_heads": DEFAULT_LAYOUT.num_query_heads,
                            "num_kv_heads": kv_heads,
                            "head_size_qk": DEFAULT_LAYOUT.head_size_qk,
                            "head_size_v": DEFAULT_LAYOUT.head_size_v,
                            "path": path,
                        },
                        latency=latency,
                        latency_base=fa3_latency,
                        min_ms=minimum,
                        max_ms=maximum,
                        sample_cv_pct=cv_pct,
                        implementation=implementation_name,
                    )
                del inputs, runner
        print("-" * TABLE_WIDTH)
    for recorder in recorders.values():
        recorder.record()
    print("\nDiffKV benchmark complete.")


@pytest.mark.diffkv_attention
@pytest.mark.skipif(not test_backend.supports_operator(OP_NAME), reason=test_backend.skip_reason(OP_NAME))
def test_diffkv_attention_benchmark(record_property) -> None:
    run_benchmark(record_property=record_property)
