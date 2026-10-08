# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Reproducible DiffKV decode benchmark with an FA3 baseline.

Run from the FlagAttention repository root with::

    pytest -q -s -m diffkv_attention \\
        benchmark/diffkv_attention_benchmark.py

The same named benchmark adapter can also be invoked explicitly::

    python -c "from benchmark.diffkv_attention_benchmark import DiffKVBenchmark; DiffKVBenchmark(op_name='diffkv_attention').run()"

The benchmark follows the GLA benchmark convention: edit the constants in
``DEFAULT_BENCHMARK_CONFIG`` below instead of passing command-line options.
The TLE and standard Triton paths should be run in separate Python processes
when compared; Triton caches JIT kernels per process and should not be mixed
in one timing loop.  The seed and page-table mode make input generation
reproducible.  Tail shapes can be enabled in the configuration to exercise
non-page-aligned masking.

The benchmark allocates all paged-cache/workspace tensors before timing and
uses ``triton.testing.do_bench`` for every provider.  ``do_bench`` records one
CUDA event pair per operator call and clears Triton's benchmark cache between
calls.  It reports one row per shape with locally measured FA3
(or a reference configured through ``BenchmarkConfig.fa3_csv``), TLE or
non-TLE Triton 2D/3D, best speedup, and best path.  Detailed timing
diagnostics are kept in the optional CSV output.  The FA3 extension is loaded
directly with
``torch.ops.load_library`` from the active Python environment's
``site-packages/fa3_runtime`` directory.  The benchmark does not search the
repository, checkout ancestors, or external overlay directories for FA3.

The FA3 provider is kept in this module, following the single-file provider
style used by the repository's FlashAttention benchmarks.  It is benchmark
support code only; the public DiffKV operator has no FA3 dependency.

The default configuration is suitable for formal performance reports and
requires a directly measured FA3 provider.  For an environment where the
extension cannot be loaded, set ``require_fa3=False`` in
``BenchmarkConfig`` for diagnostic runs, or set ``fa3_csv`` there to one or
more reference CSV paths.  This benchmark intentionally follows the GLA/MSA
style and does not expose a command-line argument parser.
"""

from __future__ import annotations

import importlib
import csv
import math
import os
import pathlib
import platform
import site
import statistics
import sys
from dataclasses import dataclass

import pytest
import torch
import triton

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


# Keep benchmark-domain options local, as in the MSA benchmark.  Model layout
# values are imported from the operator so the benchmark cannot silently use a
# different HQ/HKV/DQK/DV/block-size contract.
from flag_attn.diffkv_attention.api import (  # noqa: E402
    DEFAULT_LAYOUT,
    LAUNCH,
    OP_NAME,
    SUPPORTED_PATHS,
)

SUPPORTED_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
DEFAULT_DIAGNOSTIC_SEQ_LENS = (513, 2049)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fixed benchmark contract used by the pytest entry point.

    Head counts are mode-specific because the MiMo workload uses GQA for Full
    Attention and a wider KV grouping for SWA.  The remaining fields are
    shared by FA3 and both Triton paths so every provider receives identical
    tensors and timing settings.  This object deliberately has no large
    ``__post_init__`` validator: the production benchmark is a fixed module
    configuration, following the MSA/GLA benchmark style.  Public operator
    inputs are still validated in ``diffkv_attention.api``.
    """

    op_name: str = OP_NAME
    mode: str = "both"
    paths: tuple[str, ...] = ("2d", "3d")
    batches: tuple[int, ...] = (1, 8, 16, 32)
    seq_lens: tuple[int, ...] = (512, 2048, 8192, 32768)
    window_size: int = 128
    dtype: str = "bfloat16"
    num_query_heads: int = DEFAULT_LAYOUT.num_query_heads
    full_num_kv_heads: int = DEFAULT_LAYOUT.full_num_kv_heads
    swa_num_kv_heads: int = DEFAULT_LAYOUT.swa_num_kv_heads
    head_size_qk: int = DEFAULT_LAYOUT.head_size_qk
    head_size_v: int = DEFAULT_LAYOUT.head_size_v
    block_size: int = DEFAULT_LAYOUT.block_size
    # Triton's do_bench accepts warmup/rep in milliseconds, not iteration
    # counts.  Keep the units explicit so the benchmark cannot accidentally
    # pass an iteration count as a timing budget.
    warmup_ms: int = 100
    rep_ms: int = 500
    samples: int = 9
    stability_cv_pct: float = 5.0
    backend: str = "auto"
    # Optional vLLM benchmark CSV reference, configured in Python rather than
    # through a command-line option.
    fa3_csv: tuple[pathlib.Path, ...] | None = None
    fa3: str = "auto"
    csv: pathlib.Path | None = None
    seed: int = 0
    page_table: str = "random"
    include_tail_shapes: bool = False
    tail_seq_lens: tuple[int, ...] = DEFAULT_DIAGNOSTIC_SEQ_LENS
    # Formal runs must have a directly measured FA3 baseline (or an explicit
    # fa3_csv reference); diagnostic callers may opt out explicitly.
    require_fa3: bool = True
    device: str = "cuda:0"


def fa3_key(
    mode: str,
    batch: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size_qk: int,
    head_size_v: int,
) -> tuple[str, int, int, int, int, int, int]:
    return (
        mode,
        batch,
        seq_len,
        num_query_heads,
        num_kv_heads,
        head_size_qk,
        head_size_v,
    )


def load_fa3_reference(
    paths: list[pathlib.Path] | None,
    *,
    num_query_heads: int = DEFAULT_LAYOUT.num_query_heads,
    num_kv_heads: int = DEFAULT_LAYOUT.full_num_kv_heads,
    head_size_qk: int = DEFAULT_LAYOUT.head_size_qk,
    head_size_v: int = DEFAULT_LAYOUT.head_size_v,
):
    """Load FA3 p50 values from vLLM benchmark CSV files, if provided.

    Older vLLM CSVs do not contain all layout columns.  The explicit defaults
    keep those files usable while making the assumed model layout visible at
    the call site.
    """
    reference: dict[tuple[str, int, int, int, int, int, int], float] = {}
    for path in paths or []:
        with path.open(newline="", encoding="utf-8") as input_file:
            for row in csv.DictReader(input_file):
                if "fa3" not in row.get("provider", "").lower():
                    continue
                try:
                    key = fa3_key(
                        row.get("mode", "decode"),
                        int(row.get("batch_size", row.get("batch", "0"))),
                        int(row["seq_len"]),
                        int(row.get("num_query_heads", str(num_query_heads))),
                        int(row.get("num_kv_heads", str(num_kv_heads))),
                        int(row.get("head_size_qk", str(head_size_qk))),
                        int(row.get("head_size_v", str(head_size_v))),
                    )
                    value = float(row["latency_p50_us"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    reference[key] = value
    return reference


def fa3_op_available() -> bool:
    """Return whether the FA3 custom op is registered in this process."""
    try:
        namespace = getattr(torch.ops, "_vllm_fa3_C", None)
        return namespace is not None and hasattr(namespace, "fwd")
    except Exception:
        return False


def _fa3_environment_roots() -> list[pathlib.Path]:
    """Return FA3 search roots inside the active Python environment only."""
    environment = pathlib.Path(sys.prefix).resolve()
    roots: list[pathlib.Path] = []
    for directory in site.getsitepackages():
        site_packages = pathlib.Path(directory).resolve()
        try:
            site_packages.relative_to(environment)
        except ValueError:
            continue
        roots.extend((site_packages / "fa3_runtime", site_packages))
    return roots


def fa3_extension_candidates(path: str | None = None):
    """Yield FA3 shared libraries installed in the active environment."""
    configured = path or os.environ.get("VLLM_FLASH_ATTN_EXTENSION_DIR")
    roots = ([pathlib.Path(configured).expanduser()]
             if configured else _fa3_environment_roots())
    environment = pathlib.Path(sys.prefix).resolve()

    seen: set[pathlib.Path] = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        try:
            root.relative_to(environment)
        except ValueError:
            continue
        if root in seen:
            continue
        seen.add(root)
        if root.is_file():
            yield root
            continue
        if not root.is_dir():
            continue
        exact = root / "_vllm_fa3_C.abi3.so"
        if exact.is_file():
            yield exact
        for candidate in sorted(root.glob("*_vllm_fa3_C*.so")):
            if candidate != exact:
                yield candidate


def load_fa3_provider(requested: str = "auto", extension_path: str | None = None):
    """Load FA3 and return ``available/source/error`` status metadata."""
    if requested not in {"auto", "on", "off"}:
        raise ValueError(f"unsupported FA3 mode: {requested!r}")
    if requested == "off":
        return {"available": False, "source": "disabled", "error": None}
    if fa3_op_available():
        return {"available": True, "source": "already-registered", "error": None}

    errors: list[str] = []
    candidates = list(fa3_extension_candidates(extension_path))
    configured = extension_path or os.environ.get("VLLM_FLASH_ATTN_EXTENSION_DIR")
    if configured and not candidates:
        errors.append(f"no _vllm_fa3_C*.so found under {configured}")
    elif not configured and not candidates:
        errors.append(
            "no FA3 extension found in the active Python environment; "
            "install _vllm_fa3_C.abi3.so under site-packages/fa3_runtime"
        )
    for candidate in candidates:
        try:
            torch.ops.load_library(str(candidate))
        except Exception as exc:  # pragma: no cover - host ABI dependent
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        if fa3_op_available():
            return {"available": True, "source": str(candidate), "error": None}
        errors.append(f"{candidate}: loaded but _vllm_fa3_C.fwd was not registered")

    detail = "; ".join(errors) if errors else "FA3 extension unavailable"
    status = {"available": False, "source": "unavailable", "error": detail}
    if requested == "on":
        raise RuntimeError(
            "FA3 was requested but _vllm_fa3_C.fwd is unavailable: " + detail
        )
    return status


def shape_seed(base_seed: int, mode: str, batch: int, seq_len: int, num_kv_heads: int) -> int:
    """Derive a stable per-shape seed without relying on Python's hash()."""
    mode_offset = 0 if mode == "full" else 1_000_003
    return (
        base_seed
        + mode_offset
        + batch * 1_009
        + seq_len * 9_176
        + num_kv_heads * 65_537
    ) % (2**63 - 1)


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
    # Keep the same packed HND storage layout as vLLM's DiffKV backend, then
    # expose NHD K/V views.  FA3 is sensitive to cache strides (and this also
    # makes the Triton-vs-FA3 comparison use the production layout).
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
        physical_pages = torch.arange(
            batch * blocks_per_seq, device=device, dtype=torch.int32
        )
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
    batch, hq, dqk = query.shape
    hkv = key_cache.shape[2]
    dv = value_cache.shape[-1]
    seq_len = int(context_lens.max().item())
    block_size = key_cache.shape[1]
    cu_seqlens_q = torch.arange(
        batch + 1, device=query.device, dtype=torch.int32
    )
    scale = dqk**-0.5
    if path not in SUPPORTED_PATHS:
        raise ValueError(f"unsupported path: {path}")
    use_3d = path == "3d"

    num_q_per_kv = hq // hkv
    block_m = (
        LAUNCH.query_block_m
        if num_q_per_kv <= LAUNCH.query_block_m
        else triton.next_power_of_2(num_q_per_kv)
    )
    block_q = block_m // num_q_per_kv
    total_num_q_blocks = query.shape[0] // block_q + batch
    num_sms = torch.cuda.get_device_properties(query.device).multi_processor_count

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
            num_segments = diffkv_impl.get_num_par_softmax_segments(
                seq_len, batch, True
            )
        padded_v = triton.next_power_of_2(value_cache.shape[-1])
        segm_output = torch.empty(
            batch,
            hq,
            num_segments,
            padded_v,
            device="cuda",
            dtype=query.dtype if use_optimized else torch.float32,
        )
        segm_max = torch.empty(
            batch, hq, num_segments, device="cuda", dtype=torch.float32
        )
        segm_expsum = torch.empty_like(segm_max)
        threshold = batch
    else:
        num_segments = None
        segm_output = segm_max = segm_expsum = None
        threshold = None

    triton_window = (window_size - 1, 0) if window_size > 0 else (-1, -1)
    out = torch.empty(
        batch, hq, value_cache.shape[-1], device="cuda", dtype=query.dtype
    )
    extra_kwargs = {}
    if use_optimized and use_3d and diffkv_impl.should_use_tle_fused_reducer(
        dqk, dv, 1, seq_len, batch, block_size, True
    ):
        extra_kwargs["fused_reducer_counter"] = torch.zeros(
            batch * hq, device="cuda", dtype=torch.int32
        )

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


def build_fa3_runner(inputs: tuple[torch.Tensor, ...], window_size: int):
    """Build a direct FA3 paged-cache runner with preallocated buffers."""
    if not fa3_op_available():
        raise RuntimeError("_vllm_fa3_C.fwd is not registered")
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
        get_scheduler_metadata = getattr(
            torch.ops._vllm_fa3_C, "get_scheduler_metadata"
        )
        scheduler_metadata = get_scheduler_metadata(
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
        # Older overlays may not expose scheduler metadata; fwd remains valid
        # and selects its fallback schedule.
        scheduler_metadata = None

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


def measure(fn, warmup_ms: int, rep_ms: int, samples: int):
    """Measure a callable with Triton's standard ``do_bench`` protocol.

    ``do_bench`` owns warmup, CUDA-event placement, synchronization, adaptive
    repetition count, and benchmark-cache clearing.  We invoke it once per
    sample and reduce each returned list of per-repetition millisecond values
    to a sample median.  The outer sample distribution is retained for the
    stability CV reported by this benchmark.
    """
    values = []
    for _ in range(samples):
        timings_ms = triton.testing.do_bench(
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


def parse_dtype(name: str):
    try:
        return SUPPORTED_DTYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dtype {name!r}; choose from {sorted(SUPPORTED_DTYPES)}"
        ) from exc


def load_diffkv_backend(name: str):
    """Load exactly one backend before Triton JIT compilation starts."""
    os.environ["FLAG_ATTN_DIFFKV_BACKEND"] = name
    # Import the submodule explicitly so this benchmark remains independent
    # of the package's top-level exports.
    # Public API follows the same structure as GLA/MSA: dispatch lives in
    # ``api.py`` while Triton kernels remain private to ``diffkv.py``.
    diffkv_impl = importlib.import_module("flag_attn.diffkv_attention.api")

    if name == "tle" and not diffkv_impl.HAS_TLE:
        detail = diffkv_impl.get_diffkv_backend_info()["tle_error"]
        raise RuntimeError(
            "TLE backend was requested but is unavailable. Install a Triton "
            "build exposing triton.experimental.tle.language. "
            f"Import error: {detail}"
        )
    return diffkv_impl


def run_benchmark(config: BenchmarkConfig | None = None):
    # The default path is intentionally fixed, like the MSA/GLA benchmarks.
    # An explicit config remains useful for local diagnostics, while the
    # named adapter and pytest entry point keep the public op_name stable.
    config = DEFAULT_BENCHMARK_CONFIG if config is None else config
    if config.op_name != OP_NAME:
        raise ValueError(
            f"unsupported benchmark op_name={config.op_name!r}; expected {OP_NAME!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("DiffKV benchmark requires CUDA")
    torch_device = torch.device(config.device)
    if torch_device.index is not None:
        torch.cuda.set_device(torch_device)
    diffkv_impl = load_diffkv_backend(config.backend)
    dtype = parse_dtype(config.dtype)
    fa3_reference = load_fa3_reference(
        config.fa3_csv,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.full_num_kv_heads,
        head_size_qk=config.head_size_qk,
        head_size_v=config.head_size_v,
    )
    fa3_status = load_fa3_provider(config.fa3)
    if config.require_fa3 and not fa3_status["available"] and not fa3_reference:
        raise RuntimeError(
            "FA3 is required, but direct FA3 is unavailable and no usable "
            "BenchmarkConfig.fa3_csv reference was supplied. "
            f"Reason: {fa3_status['error']}"
        )
    modes = ["full", "swa"] if config.mode == "both" else [config.mode]
    seq_lens = list(config.seq_lens)
    if config.include_tail_shapes:
        seq_lens.extend(config.tail_seq_lens)
    seq_lens = list(dict.fromkeys(seq_lens))
    rows = []
    torch_version = torch.__version__
    triton_version = getattr(triton, "__version__", "unknown")
    cuda_runtime = torch.version.cuda or "unknown"
    device_name = torch.cuda.get_device_name(torch_device)
    capability = torch.cuda.get_device_capability(torch_device)
    print(
        f"Device: {device_name} ({config.device}) dtype={config.dtype} "
        f"compute_capability={capability[0]}.{capability[1]}"
    )
    print(
        f"Environment: torch={torch_version} triton={triton_version} "
        f"torch_cuda={cuda_runtime} seed={config.seed} "
        f"page_table={config.page_table} host={platform.node()}"
    )
    print(
        f"Workload: modes={','.join(modes)} paths={','.join(config.paths)} "
        f"batches={config.batches} seq_lens={tuple(seq_lens)} "
        f"warmup_ms={config.warmup_ms} rep_ms={config.rep_ms} "
        f"samples={config.samples}"
    )
    print(
        "DiffKV backend: "
        f"requested={config.backend}, selected={diffkv_impl.SELECTED_BACKEND}, "
        f"HAS_TLE={diffkv_impl.is_tle_available()}"
    )
    if not fa3_status["available"]:
        print(
            "FA3 direct measurement unavailable; "
            "BenchmarkConfig.fa3_csv may still provide a reference. "
            f"Reason: {fa3_status['error']}"
        )
    if not diffkv_impl.is_tle_available():
        print(
            "TLE unavailable; using standard non-TLE Triton path. "
            f"Reason: {diffkv_impl.get_diffkv_backend_info()['tle_error']}"
        )
    print(
        f"Operator: {config.op_name}\n"
        "Timing: triton.testing.do_bench around preallocated operators; "
        "adaptive per-call CUDA events and cache clear"
    )
    for mode in modes:
        window = -1 if mode == "full" else config.window_size
        hkv = (
            config.full_num_kv_heads
            if mode == "full"
            else config.swa_num_kv_heads
        )
        section_rows = []
        for batch in config.batches:
            for seq_len in seq_lens:
                inputs = make_inputs(
                    batch,
                    seq_len,
                    dtype,
                    config.num_query_heads,
                    hkv,
                    config.head_size_qk,
                    config.head_size_v,
                    config.block_size,
                    seed=shape_seed(
                        config.seed, mode, batch, seq_len, hkv
                    ),
                    page_table=config.page_table,
                    device=config.device,
                )
                measurements = {}
                path_impls = {}
                for path in config.paths:
                    runner, impl_name = build_runner(
                        inputs, path, window, diffkv_impl
                    )
                    p50, pmin, pmax, cv_pct = measure(
                        runner,
                        config.warmup_ms, config.rep_ms, config.samples,
                    )
                    measurements[path] = (p50, pmin, pmax, cv_pct)
                    path_impls[path] = impl_name
                fa3_measurement = None
                fa3_error = None
                if fa3_status["available"]:
                    try:
                        fa3_measurement = measure(
                            build_fa3_runner(inputs, window),
                            config.warmup_ms,
                            config.rep_ms,
                            config.samples,
                        )
                    except Exception as exc:  # pragma: no cover - host-specific
                        fa3_error = f"{type(exc).__name__}: {exc}"
                best_path = min(measurements, key=lambda path: measurements[path][0])
                best = measurements[best_path]
                reference_fa3_us = fa3_reference.get(
                    fa3_key(
                        "decode",
                        batch,
                        seq_len,
                        config.num_query_heads,
                        hkv,
                        config.head_size_qk,
                        config.head_size_v,
                    )
                )
                if fa3_measurement is not None:
                    fa3_us = fa3_measurement[0]
                    fa3_source = "local"
                else:
                    fa3_us = reference_fa3_us
                    fa3_source = "reference-csv" if fa3_us is not None else "n/a"
                if config.require_fa3 and fa3_us is None:
                    raise RuntimeError(
                        "FA3 is required but no measurement/reference exists for "
                        f"mode={mode}, batch={batch}, seq_len={seq_len}, "
                        f"HKV={hkv}"
                    )
                best_speedup = fa3_us / best[0] if fa3_us is not None else None
                shape_text = (
                    f"B={batch} Q=1 KV={seq_len} HQ={config.num_query_heads} "
                    f"HKV={hkv} DQK={config.head_size_qk} "
                    f"DV={config.head_size_v} {config.dtype}"
                )
                section_rows.append(
                    (
                        shape_text,
                        f"{fa3_us / 1000:.6f}" if fa3_us is not None else "n/a",
                        f"{measurements['2d'][0] / 1000:.6f}"
                        if "2d" in measurements else "n/a",
                        f"{measurements['3d'][0] / 1000:.6f}"
                        if "3d" in measurements else "n/a",
                        f"{best_speedup:.3f}x" if best_speedup is not None else "n/a",
                        best_path.upper(),
                    )
                )
                for path, measurement in measurements.items():
                    rows.append({
                        "op_name": config.op_name,
                        "mode": mode,
                        "batch": batch,
                        "seq_len": seq_len,
                        "path": path,
                        "implementation": path_impls[path],
                        "dtype": config.dtype,
                        "backend": diffkv_impl.SELECTED_BACKEND,
                        "fa3_source": fa3_source,
                        "fa3_p50_us": ""
                        if fa3_us is None else f"{fa3_us:.3f}",
                        "fa3_min_us": ""
                        if fa3_measurement is None else f"{fa3_measurement[1]:.3f}",
                        "fa3_max_us": ""
                        if fa3_measurement is None else f"{fa3_measurement[2]:.3f}",
                        "fa3_sample_cv_pct": ""
                        if fa3_measurement is None
                        else f"{fa3_measurement[3]:.3f}",
                        "fa3_error": fa3_error or "",
                        "p50_us": f"{measurement[0]:.3f}",
                        "min_us": f"{measurement[1]:.3f}",
                        "max_us": f"{measurement[2]:.3f}",
                        "sample_cv_pct": f"{measurement[3]:.3f}",
                        "best_path": best_path,
                        "best_speedup": ""
                        if best_speedup is None else f"{best_speedup:.6f}",
                        "device": device_name,
                        "device_capability": f"{capability[0]}.{capability[1]}",
                        "torch_version": torch_version,
                        "triton_version": triton_version,
                        "torch_cuda": cuda_runtime,
                        "seed": config.seed,
                        "shape_seed": shape_seed(
                            config.seed, mode, batch, seq_len, hkv
                        ),
                        "page_table": config.page_table,
                        "warmup_ms": config.warmup_ms,
                        "rep_ms": config.rep_ms,
                        "samples": config.samples,
                    })
        headers = (
            "shape (B,Q,KV,HQ,HKV,DQK,DV,dtype)",
            "FA3 p50(ms)",
            "Triton-2D p50(ms)",
            "Triton-3D p50(ms)",
            "best speedup",
            "best path",
        )
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in section_rows))
            for index in range(len(headers))
        ]
        separator_length = sum(widths) + 2 * (len(widths) - 1)
        title = "Full Attention" if mode == "full" else "SWA"
        print("\n" + "=" * separator_length)
        print(f"[{title}]")
        print("=" * separator_length)
        print("  ".join(value.ljust(widths[index])
                         for index, value in enumerate(headers)))
        print("  ".join("-" * width for width in widths))
        for row in section_rows:
            print("  ".join(value.ljust(widths[index])
                             for index, value in enumerate(row)))
        print("-" * separator_length)
    if config.csv is not None:
        config.csv.parent.mkdir(parents=True, exist_ok=True)
        with config.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {config.csv}")


DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig(
    # Edit this object to change the default benchmark, following the GLA
    # benchmark convention instead of passing command-line arguments.
    op_name=OP_NAME,
    mode="both",
    paths=("2d", "3d"),
    batches=(1, 8, 16, 32),
    seq_lens=(512, 2048, 8192, 32768),
    window_size=128,
    dtype="bfloat16",
    num_query_heads=DEFAULT_LAYOUT.num_query_heads,
    full_num_kv_heads=DEFAULT_LAYOUT.full_num_kv_heads,
    swa_num_kv_heads=DEFAULT_LAYOUT.swa_num_kv_heads,
    head_size_qk=DEFAULT_LAYOUT.head_size_qk,
    head_size_v=DEFAULT_LAYOUT.head_size_v,
    block_size=DEFAULT_LAYOUT.block_size,
    warmup_ms=100,
    rep_ms=500,
    samples=9,
    stability_cv_pct=5.0,
    backend="auto",
    fa3="auto",
    seed=0,
    page_table="random",
    include_tail_shapes=False,
    tail_seq_lens=DEFAULT_DIAGNOSTIC_SEQ_LENS,
    require_fa3=True,
    device="cuda:0",
)


class DiffKVBenchmark:
    """Named benchmark adapter following the repository benchmark pattern.

    The DiffKV benchmark compares multiple providers and launch paths, so it
    cannot use the unary ``Benchmark`` harness directly.  It still exposes
    the same explicit ``op_name`` contract used by the standard benchmarks.
    """

    def __init__(
        self,
        op_name: str = OP_NAME,
        config: BenchmarkConfig = DEFAULT_BENCHMARK_CONFIG,
    ) -> None:
        if op_name != OP_NAME:
            raise ValueError(
                f"unsupported benchmark op_name={op_name!r}; expected {OP_NAME!r}"
            )
        if config.op_name != op_name:
            raise ValueError(
                f"BenchmarkConfig.op_name={config.op_name!r} does not match "
                f"{op_name!r}"
            )
        self.op_name = op_name
        self.config = config

    def run(self):
        return run_benchmark(self.config)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DiffKV benchmark requires CUDA",
)
@pytest.mark.diffkv_attention
def test_perf_diffkv_attention():
    """Run the default DiffKV benchmark under pytest."""
    bench = DiffKVBenchmark(op_name=OP_NAME)
    bench.run()


if __name__ == "__main__":
    DiffKVBenchmark(op_name=OP_NAME).run()
