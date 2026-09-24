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

"""CUDA benchmark for the MiniMax M3 paged MSA kernels.

The benchmark compares the FlagAttention and vLLM implementations on the same
inputs.  ``fp8`` means FP8 index queries/index keys and FP8 main KV cache,
with scalar K/V dequantization scales passed to both implementations.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import warnings
from dataclasses import dataclass, replace
from typing import Callable

import pytest
import torch
import triton
import triton.knobs
import triton.testing as triton_testing

try:
    from benchmark.recording import benchmark_metric, record_benchmark_result
except ModuleNotFoundError:  # Direct script execution.
    from recording import benchmark_metric, record_benchmark_result

from flag_attn.minimax_sparse_attention import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)

try:
    from vllm.models.minimax_m3.common.ops.index_topk import (
        minimax_m3_index_decode as vllm_index_decode,
    )
    from vllm.models.minimax_m3.common.ops.index_topk import (
        minimax_m3_index_score as vllm_index_score,
    )
    from vllm.models.minimax_m3.common.ops.index_topk import (
        minimax_m3_index_topk as vllm_index_topk,
    )
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn as vllm_sparse_attn,
    )
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn_decode as vllm_sparse_attn_decode,
    )

    VLLM_AVAILABLE = True
    VLLM_IMPORT_ERROR = ""
except Exception as exc:  # vLLM is an optional benchmark baseline.
    VLLM_AVAILABLE = False
    VLLM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

warnings.filterwarnings("ignore", message="tl.make_block_ptr is deprecated")
triton.knobs.autotuning.adjust_block_size = False


class _CachedPlatform:
    """Return one cached PDL decision during benchmark iterations."""

    def __init__(self, supports_pdl: bool):
        self._supports_pdl = supports_pdl

    def is_arch_support_pdl(self) -> bool:
        return self._supports_pdl


_flag_attn_index_module = sys.modules[minimax_m3_index_decode.__module__]
_flag_attn_sparse_module = sys.modules[minimax_m3_sparse_attn_decode.__module__]
_flag_attn_platform = _CachedPlatform(
    _flag_attn_index_module.current_platform.is_arch_support_pdl()
)
_flag_attn_index_module.current_platform = _flag_attn_platform
_flag_attn_sparse_module.current_platform = _flag_attn_platform

if VLLM_AVAILABLE:
    _vllm_index_module = sys.modules[vllm_index_decode.__module__]
    _vllm_sparse_module = sys.modules[vllm_sparse_attn_decode.__module__]
    _vllm_platform = _CachedPlatform(
        _vllm_index_module.current_platform.is_arch_support_pdl()
    )
    _vllm_index_module.current_platform = _vllm_platform
    _vllm_sparse_module.current_platform = _vllm_platform


BLOCK = SPARSE_BLOCK_SIZE
HEAD_DIM = 128
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
DEFAULT_WARMUP = 200
DEFAULT_REP = 300
KV_SCALE = 0.5
BF16_ATOL = BF16_RTOL = 2e-2
FP8_ATOL, FP8_RTOL = 6e-2, 8e-2

PREFILL_SHAPES = [
    (1, 8192, 16, 96),
    (2, 16384, 8, 96),
    (1, 32768, 16, 96),
    (2, 8192, 8, 96),
    (4, 4096, 16, 384),
    (4, 4096, 16, 256),
]

DECODE_SHAPES = [
    (1, 4096, 16, 96),
    (1, 16384, 16, 96),
    (1, 65536, 16, 96),
    (4, 4096, 8, 96),
    (4, 16384, 8, 96),
    (16, 4096, 8, 96),
    (32, 2048, 4, 48),
    (64, 1024, 4, 48),
]


@dataclass
class MSABenchmarkArgs:
    dtype: str = "both"
    shape: str | None = None
    topk: int = 16
    init_blocks: int = 1
    local_blocks: int = 2
    all_shapes: bool = False
    per_step: bool = True
    identity_pages: bool = False
    prefill_only: bool = False
    decode_only: bool = False
    decode_qlen: int = 1
    warmup: int = DEFAULT_WARMUP
    rep: int = DEFAULT_REP
    seed: int = 0
    no_vllm: bool = False
    decode: bool = False


@dataclass
class MSAData:
    q: torch.Tensor
    idx_q: torch.Tensor
    kv_cache: torch.Tensor
    index_kv_cache: torch.Tensor
    block_table: torch.Tensor
    cu_q: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    sm_scale: float
    k_scale: torch.Tensor | None
    v_scale: torch.Tensor | None


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")


def _supports_fp8() -> bool:
    if FP8_DTYPE is None or not torch.cuda.is_available():
        return False
    # NVIDIA FP8 Tensor Core support starts with Ada (8.9) and Hopper (9.0).
    return torch.cuda.get_device_capability() >= (8, 9)


def _encode_fp8(value: torch.Tensor, scale: float) -> torch.Tensor:
    if FP8_DTYPE is None:
        raise RuntimeError("This PyTorch build does not provide float8_e4m3fn.")
    return (value / scale).to(FP8_DTYPE)


def _random_storage(
    shape: tuple[int, ...],
    device: torch.device,
    fp8: bool,
    scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # Generate in BF16 because torch.randn support for FP8 is version-dependent.
    value = (
        torch.randn(
            shape,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.5
    )
    return _encode_fp8(value, scale) if fp8 else value


def make_data(
    batch: int,
    seq_len: int,
    num_kv_heads: int,
    num_heads: int,
    device: torch.device,
    dtype_name: str,
    *,
    decode: bool,
    decode_qlen: int,
    randomize_pages: bool = True,
    generator: torch.Generator | None = None,
) -> MSAData:
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if dtype_name not in {"bf16", "fp8"}:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    if dtype_name == "fp8" and FP8_DTYPE is None:
        raise RuntimeError("FP8 was requested but float8_e4m3fn is unavailable.")

    storage_dtype = torch.bfloat16 if dtype_name == "bf16" else FP8_DTYPE
    blocks_per_request = (seq_len + BLOCK - 1) // BLOCK
    total_blocks = batch * blocks_per_request
    total_q = batch * decode_qlen if decode else batch * seq_len

    q = (
        torch.randn(
            (total_q, num_heads, HEAD_DIM),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.5
    )
    idx_q = _random_storage(
        (total_q, num_kv_heads, HEAD_DIM),
        device,
        dtype_name == "fp8",
        generator=generator,
    )
    k_cont = _random_storage(
        (total_blocks * BLOCK, num_kv_heads, HEAD_DIM),
        device,
        dtype_name == "fp8",
        KV_SCALE,
        generator,
    )
    v_cont = _random_storage(
        (total_blocks * BLOCK, num_kv_heads, HEAD_DIM),
        device,
        dtype_name == "fp8",
        KV_SCALE,
        generator,
    )
    index_k_cont = _random_storage(
        (total_blocks * BLOCK, HEAD_DIM),
        device,
        dtype_name == "fp8",
        generator=generator,
    )

    kv_cache = torch.empty(
        (total_blocks, num_kv_heads, BLOCK, 2 * HEAD_DIM),
        device=device,
        dtype=storage_dtype,
    )
    k_paged = k_cont.reshape(total_blocks, BLOCK, num_kv_heads, HEAD_DIM).permute(
        0, 2, 1, 3
    )
    v_paged = v_cont.reshape(total_blocks, BLOCK, num_kv_heads, HEAD_DIM).permute(
        0, 2, 1, 3
    )
    kv_cache[..., :HEAD_DIM] = k_paged
    kv_cache[..., HEAD_DIM:] = v_paged
    index_kv_cache = index_k_cont.reshape(total_blocks, BLOCK, HEAD_DIM)

    physical_pages = torch.randperm(total_blocks, device=device, generator=generator)
    if not randomize_pages:
        physical_pages = torch.arange(total_blocks, device=device)
    # Force identity page ordering for FP8 inputs.
    if dtype_name == "fp8":
        physical_pages = torch.arange(total_blocks, device=device)
        randomize_pages = False  # Skip the page-remapping step below.
    block_table = physical_pages.reshape(batch, blocks_per_request).to(torch.int32)
    if randomize_pages:
        kv_cache = kv_cache[physical_pages.argsort()].contiguous()
        index_kv_cache = index_kv_cache[physical_pages.argsort()].contiguous()

    q_stride = decode_qlen if decode else seq_len
    cu_q = torch.arange(
        0,
        (batch + 1) * q_stride,
        q_stride,
        device=device,
        dtype=torch.int32,
    )
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    prefix_lens = torch.zeros_like(seq_lens)
    if dtype_name == "fp8":
        k_scale = torch.tensor([KV_SCALE], device=device, dtype=torch.float32)
        v_scale = torch.tensor([KV_SCALE], device=device, dtype=torch.float32)
    else:
        k_scale = v_scale = None
    return MSAData(
        q,
        idx_q,
        kv_cache,
        index_kv_cache,
        block_table,
        cu_q,
        seq_lens,
        prefix_lens,
        HEAD_DIM**-0.5,
        k_scale,
        v_scale,
    )


def _call_sparse(
    fn: Callable,
    data: MSAData,
    topk_idx: torch.Tensor,
    output: torch.Tensor,
    *,
    decode: bool,
    max_query_len: int,
    num_kv_heads: int,
    decode_qlen: int,
) -> None:
    common = dict(
        q=data.q,
        kv_cache=data.kv_cache,
        topk_idx=topk_idx,
        block_table=data.block_table,
        sm_scale=data.sm_scale,
        output=output,
    )
    if decode:
        common.update(
            seq_lens=data.seq_lens,
            num_kv_heads=num_kv_heads,
            decode_query_len=decode_qlen,
        )
    else:
        common.update(
            cu_seqlens_q=data.cu_q,
            seq_lens=data.seq_lens,
            prefix_lens=data.prefix_lens,
            max_query_len=max_query_len,
            num_kv_heads=num_kv_heads,
        )
    if data.k_scale is not None:
        common.update(k_scale=data.k_scale, v_scale=data.v_scale)
    fn(**common)


def run_prefill(
    index_score: Callable,
    index_topk: Callable,
    sparse_attn: Callable,
    data: MSAData,
    seq_len: int,
    num_kv_heads: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    output: torch.Tensor,
) -> None:
    scores = index_score(
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        seq_len,
        seq_len,
        num_kv_heads,
    )
    topk_idx = index_topk(
        scores,
        data.cu_q,
        data.prefix_lens,
        seq_len,
        topk,
        init_blocks,
        local_blocks,
    )
    _call_sparse(
        sparse_attn,
        data,
        topk_idx,
        output,
        decode=False,
        max_query_len=seq_len,
        num_kv_heads=num_kv_heads,
        decode_qlen=1,
    )


def run_decode(
    index_decode: Callable,
    sparse_attn_decode: Callable,
    data: MSAData,
    seq_len: int,
    num_kv_heads: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    decode_qlen: int,
    output: torch.Tensor,
) -> None:
    topk_idx = index_decode(
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.seq_lens,
        seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_qlen,
        decode_qlen,
    )
    _call_sparse(
        sparse_attn_decode,
        data,
        topk_idx,
        output,
        decode=True,
        max_query_len=1,
        num_kv_heads=num_kv_heads,
        decode_qlen=decode_qlen,
    )


def bench_fn(fn: Callable, warmup: int, rep: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return float(
        triton_testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    )


def _supports_fp8_scales() -> bool:
    if not VLLM_AVAILABLE:
        return False
    try:
        prefill_params = inspect.signature(vllm_sparse_attn).parameters
        decode_params = inspect.signature(vllm_sparse_attn_decode).parameters
    except (TypeError, ValueError):
        return False
    return (
        "k_scale" in prefill_params
        and "v_scale" in prefill_params
        and ("k_scale" in decode_params and "v_scale" in decode_params)
    )


def _sparse_kv_layout(fn: Callable) -> str:
    """Infer the cache contract from the actual vLLM wrapper's stride accesses."""
    try:
        source = inspect.getsource(inspect.unwrap(fn))
        tree = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError(
            f"Cannot inspect vLLM {fn.__name__} KV cache layout; refusing to "
            "report an unverified speedup."
        ) from exc

    indices: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "kv_cache"
            and node.func.attr == "stride"
        ):
            continue
        if (
            len(node.args) != 1
            or node.keywords
            or not isinstance(node.args[0], ast.Constant)
            or type(node.args[0].value) is not int
        ):
            raise RuntimeError(
                f"Ambiguous vLLM {fn.__name__} KV cache stride access; "
                "cannot select a safe cache layout."
            )
        indices.add(node.args[0].value)

    layouts = {
        frozenset(range(4)): "packed4d",
        frozenset(range(5)): "split5d",
    }
    layout = layouts.get(frozenset(indices))
    if layout is None:
        raise RuntimeError(
            f"Unknown vLLM {fn.__name__} KV cache stride indices "
            f"{sorted(indices)}; cannot select a safe cache layout."
        )
    return layout


def _vllm_kv_layout(prefill: Callable, decode: Callable) -> str:
    prefill_layout = _sparse_kv_layout(prefill)
    decode_layout = _sparse_kv_layout(decode)
    if prefill_layout != decode_layout:
        raise RuntimeError(
            "vLLM prefill/decode KV cache layouts differ: "
            f"{prefill_layout} vs {decode_layout}."
        )
    return prefill_layout


def _vllm_data(data: MSAData, layout: str) -> MSAData:
    if layout == "packed4d":
        return data
    if layout != "split5d":
        raise ValueError(f"Unsupported vLLM KV cache layout: {layout}")
    cache = data.kv_cache
    if cache.ndim != 4 or cache.shape[-1] != 2 * HEAD_DIM:
        raise ValueError(f"Unexpected FlagAttention KV cache shape: {tuple(cache.shape)}")
    blocks, heads, tokens, _ = cache.shape
    # 4D: [block, head, token, K|V] -> 5D: [block, K/V, token, head, dim].
    # Convert after physical pages are randomized, preserving block_table indices.
    legacy_cache = (
        cache.reshape(blocks, heads, tokens, 2, HEAD_DIM)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
    )
    return replace(data, kv_cache=legacy_cache)


def _check_outputs(
    flag_output: torch.Tensor,
    vllm_output: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    dtype_name: str,
    mode: str,
    layout: str,
) -> float:
    context = f"{mode}, {dtype_name}, shape={shape}, vLLM layout={layout}"
    for provider, output in (("FlagAttention", flag_output), ("vLLM", vllm_output)):
        if not torch.isfinite(output).all().item():
            raise AssertionError(f"{provider} produced NaN or Inf ({context})")

    atol, rtol = (
        (FP8_ATOL, FP8_RTOL) if dtype_name == "fp8" else (BF16_ATOL, BF16_RTOL)
    )
    try:
        torch.testing.assert_close(flag_output, vllm_output, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = (flag_output.float() - vllm_output.float()).abs()
        raise AssertionError(
            f"FlagAttention/vLLM output mismatch ({context}); "
            f"max_abs_diff={diff.max().item():.6g}, atol={atol}, rtol={rtol}"
        ) from exc

    flag_flat = flag_output.float().flatten()
    vllm_flat = vllm_output.float().flatten()
    flag_norm = torch.linalg.vector_norm(flag_flat)
    vllm_norm = torch.linalg.vector_norm(vllm_flat)
    cosine = torch.where(
        (flag_norm == 0) & (vllm_norm == 0),
        torch.ones_like(flag_norm),
        torch.dot(flag_flat, vllm_flat)
        / (flag_norm * vllm_norm).clamp_min(torch.finfo(torch.float32).tiny),
    )
    return cosine.clamp(-1.0, 1.0).item()


def _bench_steps(
    data: MSAData,
    decode: bool,
    args: MSABenchmarkArgs,
    shape: tuple[int, int, int, int],
) -> dict[str, float]:
    batch, seq_len, num_kv_heads, _ = shape
    output = torch.empty_like(data.q)
    if decode:

        def index_decode() -> torch.Tensor:
            return minimax_m3_index_decode(
                data.idx_q,
                data.index_kv_cache,
                data.block_table,
                data.seq_lens,
                seq_len,
                args.topk,
                args.init_blocks,
                args.local_blocks,
                num_kv_heads,
                args.decode_qlen,
                args.decode_qlen,
            )

        topk_idx = index_decode()

        def attention() -> None:
            _call_sparse(
                minimax_m3_sparse_attn_decode,
                data,
                topk_idx,
                output,
                decode=True,
                max_query_len=1,
                num_kv_heads=num_kv_heads,
                decode_qlen=args.decode_qlen,
            )

        attention()
        torch.cuda.synchronize()
        return {
            "index_decode": bench_fn(index_decode, args.warmup, args.rep),
            "attention_decode": bench_fn(attention, args.warmup, args.rep),
        }

    scores: torch.Tensor | None = None
    topk_idx: torch.Tensor | None = None

    def index_score() -> None:
        nonlocal scores
        scores = minimax_m3_index_score(
            data.idx_q,
            data.index_kv_cache,
            data.block_table,
            data.cu_q,
            data.seq_lens,
            data.prefix_lens,
            seq_len,
            seq_len,
            num_kv_heads,
        )

    def index_topk() -> None:
        nonlocal topk_idx
        assert scores is not None
        topk_idx = minimax_m3_index_topk(
            scores,
            data.cu_q,
            data.prefix_lens,
            seq_len,
            args.topk,
            args.init_blocks,
            args.local_blocks,
        )

    def attention() -> None:
        assert topk_idx is not None
        _call_sparse(
            minimax_m3_sparse_attn,
            data,
            topk_idx,
            output,
            decode=False,
            max_query_len=seq_len,
            num_kv_heads=num_kv_heads,
            decode_qlen=1,
        )

    index_score()
    index_topk()
    attention()
    torch.cuda.synchronize()
    return {
        "index_score": bench_fn(index_score, args.warmup, args.rep),
        "index_topk": bench_fn(index_topk, args.warmup, args.rep),
        "attention": bench_fn(attention, args.warmup, args.rep),
    }


def _parse_shape(value: str) -> tuple[int, int, int, int]:
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(
            "--shape must contain four comma-separated integers: "
            "batch,seq_len,num_kv_heads,num_heads"
        ) from exc
    if len(shape) != 4 or any(item <= 0 for item in shape):
        raise ValueError(
            "--shape must contain four positive integers: "
            "batch,seq_len,num_kv_heads,num_heads"
        )
    return shape


def _get_shapes(args: MSABenchmarkArgs) -> list[tuple[int, int, int, int]]:
    if args.shape is not None and not args.all_shapes:
        return [_parse_shape(args.shape)]
    return DECODE_SHAPES if args.decode else PREFILL_SHAPES


def _format_columns(columns: list[tuple[str, int]]) -> str:
    return "  ".join(f"{value:>{width}s}" for value, width in columns)


def _run_dtype(
    args: MSABenchmarkArgs,
    dtype_name: str,
    record_property: Callable[[str, object], None] | None = None,
) -> None:
    run_vllm = VLLM_AVAILABLE and not args.no_vllm
    if dtype_name == "fp8" and run_vllm and not _supports_fp8_scales():
        print("[baseline] vLLM FP8 skipped: k_scale/v_scale are unavailable")
        run_vllm = False
    vllm_layout = (
        _vllm_kv_layout(vllm_sparse_attn, vllm_sparse_attn_decode)
        if run_vllm
        else None
    )

    mode = f"decode qlen={args.decode_qlen}" if args.decode else "prefill"
    use_identity_pages = args.identity_pages or dtype_name == "fp8"
    page_mode = "identity" if use_identity_pages else "random"
    print(f"\nMiniMax M3 paged sparse attention ({mode})")
    print(
        f"dtype={dtype_name}, topk={args.topk}, "
        f"init/local={args.init_blocks}/{args.local_blocks}, "
        f"pages={page_mode}, seed={args.seed}, "
        f"warmup={args.warmup}ms, rep={args.rep}ms"
    )
    print("Timing: eager execution with CUDA events")
    if run_vllm:
        print(f"vLLM KV cache layout: {vllm_layout}; outputs checked before timing")
        print("Provider order: alternates by shape")
    else:
        print("vLLM baseline: unavailable; FlagAttn only")

    headers = [("Shape [B,S,KVH,H]", 22), ("FlagAttn(ms)", 13)]
    if run_vllm:
        headers.extend(
            [("vLLM(ms)", 10), ("vLLM/ours", 10), ("Cosine", 9), ("Match", 6)]
        )
    if args.per_step:
        if args.decode:
            headers.extend([("IdxDec(ms)", 11), ("AttnDec(ms)", 11)])
        else:
            headers.extend([("Score(ms)", 10), ("TopK(ms)", 10), ("Attn(ms)", 10)])
    separator = "-" * len(_format_columns(headers))
    print(separator)
    print(_format_columns(headers))
    print(separator)

    device = torch.device("cuda")
    metrics = []
    for shape_index, shape in enumerate(_get_shapes(args)):
        batch, seq_len, num_kv_heads, num_heads = shape
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"Invalid shape {shape}: num_heads must be divisible by " "num_kv_heads"
            )
        if args.decode and args.decode_qlen > seq_len:
            raise ValueError(
                f"Invalid shape {shape}: decode_qlen={args.decode_qlen} "
                "cannot exceed seq_len"
            )

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + shape_index + (100_000 if args.decode else 0))
        data = make_data(
            batch,
            seq_len,
            num_kv_heads,
            num_heads,
            device,
            dtype_name,
            decode=args.decode,
            decode_qlen=args.decode_qlen,
            randomize_pages=not use_identity_pages,
            generator=generator,
        )
        vllm_data = _vllm_data(data, vllm_layout) if vllm_layout else None
        flag_attn_output = torch.empty_like(data.q)
        vllm_output = torch.empty_like(data.q) if run_vllm else None

        def flag_attn_run() -> None:
            if args.decode:
                run_decode(
                    minimax_m3_index_decode,
                    minimax_m3_sparse_attn_decode,
                    data,
                    seq_len,
                    num_kv_heads,
                    args.topk,
                    args.init_blocks,
                    args.local_blocks,
                    args.decode_qlen,
                    flag_attn_output,
                )
            else:
                run_prefill(
                    minimax_m3_index_score,
                    minimax_m3_index_topk,
                    minimax_m3_sparse_attn,
                    data,
                    seq_len,
                    num_kv_heads,
                    args.topk,
                    args.init_blocks,
                    args.local_blocks,
                    flag_attn_output,
                )

        def vllm_run() -> None:
            assert vllm_output is not None
            assert vllm_data is not None
            if args.decode:
                run_decode(
                    vllm_index_decode,
                    vllm_sparse_attn_decode,
                    vllm_data,
                    seq_len,
                    num_kv_heads,
                    args.topk,
                    args.init_blocks,
                    args.local_blocks,
                    args.decode_qlen,
                    vllm_output,
                )
            else:
                run_prefill(
                    vllm_index_score,
                    vllm_index_topk,
                    vllm_sparse_attn,
                    vllm_data,
                    seq_len,
                    num_kv_heads,
                    args.topk,
                    args.init_blocks,
                    args.local_blocks,
                    vllm_output,
                )

        vllm_ms = None
        accuracy = None
        if run_vllm:
            flag_attn_run()
            vllm_run()
            torch.cuda.synchronize()
            assert vllm_output is not None and vllm_layout is not None
            accuracy = _check_outputs(
                flag_attn_output,
                vllm_output,
                shape=shape,
                dtype_name=dtype_name,
                mode=mode,
                layout=vllm_layout,
            )
            providers = (
                (("flag_attn", flag_attn_run), ("vllm", vllm_run))
                if shape_index % 2 == 0
                else (("vllm", vllm_run), ("flag_attn", flag_attn_run))
            )
            timings = {
                name: bench_fn(fn, args.warmup, args.rep) for name, fn in providers
            }
            flag_attn_ms = timings["flag_attn"]
            vllm_ms = timings["vllm"]
        else:
            flag_attn_ms = bench_fn(flag_attn_run, args.warmup, args.rep)

        steps = _bench_steps(data, args.decode, args, shape) if args.per_step else {}
        row = [
            (f"{batch}x{seq_len}x{num_kv_heads}x{num_heads}", 22),
            (f"{flag_attn_ms:.4f}", 13),
        ]
        if run_vllm:
            row.extend(
                [
                    (f"{vllm_ms:.4f}", 10),
                    (f"{vllm_ms / flag_attn_ms:.2f}x", 10),
                    (f"{accuracy:.6f}", 9),
                    ("pass", 6),
                ]
            )
        if args.per_step:
            if args.decode:
                row.extend(
                    [
                        (f"{steps['index_decode']:.4f}", 11),
                        (f"{steps['attention_decode']:.4f}", 11),
                    ]
                )
            else:
                row.extend(
                    [
                        (f"{steps['index_score']:.4f}", 10),
                        (f"{steps['index_topk']:.4f}", 10),
                        (f"{steps['attention']:.4f}", 10),
                    ]
                )
        print(_format_columns(row))
        sys.stdout.flush()
        metrics.append(
            benchmark_metric(
                shape_detail=shape,
                latency_base=vllm_ms,
                latency=flag_attn_ms,
                speedup=(
                    vllm_ms / flag_attn_ms
                    if vllm_ms is not None and flag_attn_ms > 0
                    else None
                ),
                accuracy=accuracy,
                steps=steps,
            )
        )

    record_benchmark_result(
        record_property,
        op_name="minimax_m3_sparse_attn",
        dtype=str(torch.bfloat16 if dtype_name == "bf16" else FP8_DTYPE),
        result=metrics,
        baseline="vLLM" if run_vllm else None,
        vllm_kv_layout=vllm_layout,
        phase="decode" if args.decode else "prefill",
        topk=args.topk,
        decode_qlen=args.decode_qlen if args.decode else None,
    )


def run_benchmark(
    args: MSABenchmarkArgs,
    record_property: Callable[[str, object], None] | None = None,
) -> None:
    _require_cuda()
    if args.topk < 1:
        raise ValueError("--topk must be positive")
    if args.decode_qlen < 1:
        raise ValueError("--decode-qlen must be positive")
    if args.warmup < 0 or args.rep <= 0:
        raise ValueError("--warmup must be non-negative and --rep must be positive")
    capability = torch.cuda.get_device_capability()
    print(
        f"[device] {torch.cuda.get_device_name()} "
        f"capability={capability[0]}.{capability[1]}"
    )
    if args.prefill_only:
        modes = (False,)
    elif args.decode_only:
        modes = (True,)
    else:
        modes = (False, True)
    if args.dtype in {"fp8", "both"} and not _supports_fp8():
        print("[FP8] skipped: this GPU or PyTorch build does not support FP8")
    if args.dtype == "fp8" and not _supports_fp8():
        return
    dtypes = (
        ("bf16", "fp8")
        if args.dtype == "both" and _supports_fp8()
        else ("bf16",) if args.dtype == "both" else (args.dtype,)
    )

    if args.no_vllm:
        print("[baseline] vLLM skipped (--no-vllm)")
    elif VLLM_AVAILABLE:
        print("[baseline] vLLM enabled")
    else:
        print(f"[baseline] vLLM skipped ({VLLM_IMPORT_ERROR})")

    for mode_index, decode in enumerate(modes):
        args.decode = decode
        if mode_index:
            print()
        for dtype_name in dtypes:
            _run_dtype(args, dtype_name, record_property)


@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MiniMax M3 benchmark requires CUDA"
)
def test_msa_benchmark(request, record_property) -> None:
    """Run the MSA benchmark through pytest using benchmark CLI timing options."""
    args = MSABenchmarkArgs(
        topk=int(request.config.getoption("--topk", default=16)),
        warmup=int(request.config.getoption("--warmup", default=DEFAULT_WARMUP)),
        rep=int(request.config.getoption("--iter", default=DEFAULT_REP)),
    )
    run_benchmark(args, record_property)


if __name__ == "__main__":
    run_benchmark(MSABenchmarkArgs())
