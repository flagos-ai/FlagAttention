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

import math
from dataclasses import dataclass
from importlib import import_module
from statistics import median

import pytest
import torch

bsa_ops = import_module(
    "flag_attn.hpc_ops_attention.prefill.attention_blocksparse_prefill_fp8"
)
attention_with_kvcache_blocksparse_prefill_fp8 = (
    bsa_ops.attention_with_kvcache_blocksparse_prefill_fp8
)

try:
    import hpc
except (ImportError, OSError, RuntimeError) as exc:
    # hpc-ops is an optional CUDA baseline.  Its absence must not suppress the
    # TLE benchmark; when it imports successfully, the same test compares it.
    hpc = None
    HPC_IMPORT_ERROR = exc
else:
    required_hpc_apis = (
        "QuantType",
        "attention_with_kvcache_blocksparse_prefill_fp8",
    )
    missing_hpc_apis = tuple(
        name for name in required_hpc_apis if not hasattr(hpc, name)
    )
    if missing_hpc_apis:
        HPC_IMPORT_ERROR = RuntimeError(
            "hpc is missing required APIs: " + ", ".join(missing_hpc_apis)
        )
        hpc = None
    else:
        required_quant_types = (
            "QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD",
            "QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR",
        )
        missing_quant_types = tuple(
            name for name in required_quant_types if not hasattr(hpc.QuantType, name)
        )
        if missing_quant_types:
            HPC_IMPORT_ERROR = RuntimeError(
                "hpc.QuantType is missing: " + ", ".join(missing_quant_types)
            )
            hpc = None
        else:
            HPC_IMPORT_ERROR = None

CASES = [(127, 127), (512, 4096), (512, 32768)]
PERF_RESULTS = []
BLOCK = 128
PAGE_SIZE = 64
Q_HEADS = 32
KV_HEADS = 4
HEAD_DIM = 128
MASK_SKIP_RATIO = 0.75
SEED = 10086
DEFAULT_GRAPH_WARMUP = 5
DEFAULT_GRAPH_ITERS = 50


@dataclass(frozen=True)
class Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    cu_seqlens_q: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    max_q_len: int
    quant_type: int
    block_mask: torch.Tensor | None
    sparsity_bucket: int
    effective_skip_ratio: float


def _print_performance_table():
    if not PERF_RESULTS:
        return

    headers = (
        "Q Len",
        "KV Len",
        "Quant",
        "Mask",
        "Skip",
        "Layout",
        "FlagAttention Impl",
        "FlagAttention (ms)",
        "HPC CUDA (ms)",
        "HPC/FlagAttention",
    )
    rows = [headers]
    for result in PERF_RESULTS:
        rows.append(
            (
                str(result["q_len"]),
                str(result["kv_len"]),
                str(result["quant_type"]),
                str(result["masked"]),
                f'{result["effective_skip_ratio"]:.1%}',
                result["kv_layout"],
                result["flagattention_impl"],
                f'{result["flagattention_ms"]:.4f}',
                (f'{result["hpc_ms"]:.4f}' if result["hpc_ms"] is not None else "N/A"),
                (
                    f'{result["hpc_ms"] / result["flagattention_ms"]:.3f}x'
                    if result["hpc_ms"] is not None
                    else "N/A"
                ),
            )
        )

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row):
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    first = PERF_RESULTS[0]
    print("\n\nBSA FP8 prefill performance summary (CUDA Graph replay median)")
    print(
        f'Warmup replays: {first["warmup"]}; ' f'Timed samples: {first["repetitions"]}'
    )
    print(separator)
    print(format_row(rows[0]))
    print(separator)
    for row in rows[1:]:
        print(format_row(row))
    print(separator)
    print(
        "HPC/FlagAttention > 1: FlagAttention is faster; "
        "HPC/FlagAttention < 1: HPC CUDA is faster."
    )
    if hpc is None:
        print(f"HPC CUDA unavailable: {HPC_IMPORT_ERROR}")


@pytest.fixture(scope="module", autouse=True)
def report_performance_results():
    yield
    _print_performance_table()


def _sparsity_bucket(active_tiles, causal_tiles):
    active_ratio = active_tiles / causal_tiles if causal_tiles else 1.0
    if active_ratio >= 0.75:
        return 0
    if active_ratio >= 0.50:
        return 1
    if active_ratio >= 0.25:
        return 2
    return 3


def _make_block_mask(q_len, kv_len, masked, device):
    q_tiles = math.ceil(q_len / BLOCK)
    kv_tiles = math.ceil(kv_len / BLOCK)
    rows = torch.arange(q_tiles).view(q_tiles, 1)
    cols = torch.arange(kv_tiles).view(1, kv_tiles)
    causal_boundary = rows + (kv_tiles - q_tiles)
    causal = cols <= causal_boundary
    causal_tiles = int(causal.sum().item()) * Q_HEADS
    if not masked:
        return None, 0, 0.0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 17)
    block_mask = (
        torch.rand((1, Q_HEADS, q_tiles, kv_tiles), generator=generator)
        >= MASK_SKIP_RATIO
    )
    block_mask &= causal.view(1, 1, q_tiles, kv_tiles)

    # Keep the right-causal frontier and the tile containing each Q tile's
    # first query row.  This prevents all-masked softmax rows for ragged cases.
    frontier = torch.clamp(causal_boundary, max=kv_tiles - 1)
    block_mask |= (cols == frontier).view(1, 1, q_tiles, kv_tiles)
    first_q_positions = kv_len - q_len + rows * BLOCK
    first_q_tiles = torch.div(first_q_positions, BLOCK, rounding_mode="floor").clamp(
        min=0, max=kv_tiles - 1
    )
    block_mask |= (cols == first_q_tiles).view(1, 1, q_tiles, kv_tiles)

    active_tiles = int(block_mask.sum().item())
    effective_skip_ratio = 1.0 - active_tiles / causal_tiles
    bucket = _sparsity_bucket(active_tiles, causal_tiles)
    block_mask = block_mask.to(device=device, dtype=torch.uint8).contiguous()
    return block_mask, bucket, effective_skip_ratio


def _make_inputs(q_len, kv_len, quant_type, masked, kv_layout):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    requested_pages = math.ceil(kv_len / PAGE_SIZE)
    physical_pages = max(requested_pages * 2, requested_pages + 8)
    q = (
        torch.randn(q_len, Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        / math.sqrt(HEAD_DIM)
    ).to(fp8)
    raw_cache = torch.randn(
        physical_pages,
        2,
        PAGE_SIZE,
        KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    k = (raw_cache[:, 0] / math.sqrt(HEAD_DIM)).to(fp8)
    v = raw_cache[:, 1].to(fp8)
    if kv_layout == "hnd":
        k = k.transpose(1, 2).contiguous().transpose(1, 2)
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
    q_scale = torch.zeros(
        1,
        Q_HEADS,
        math.ceil(q_len / BLOCK) * BLOCK,
        device=device,
        dtype=torch.float32,
    )
    q_scale[:, :, :q_len] = torch.randn(1, Q_HEADS, q_len, device=device).abs() / 10.0
    if quant_type == 0:
        k_scale = (
            torch.randn(
                physical_pages,
                PAGE_SIZE // 32,
                KV_HEADS,
                HEAD_DIM // 4,
                device=device,
                dtype=torch.float32,
            )
            .abs()
            .clamp_min_(1e-6)
            .view(fp8)
        )
        v_scale = torch.randn(KV_HEADS, device=device).abs().clamp_min_(1e-6)
    else:
        k_scale = torch.rand(1, device=device, dtype=torch.float32) + 0.5
        v_scale = torch.rand(1, device=device, dtype=torch.float32) + 0.5
    cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    block_ids = torch.randperm(physical_pages, device=device)[:requested_pages]
    block_ids = block_ids.to(torch.int32).view(1, -1)
    kv_lens = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_mask, bucket, effective_skip_ratio = _make_block_mask(
        q_len, kv_len, masked, device
    )
    return Inputs(
        q=q,
        k_cache=k,
        v_cache=v,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cu_seqlens_q=cu_seqlens_q,
        block_ids=block_ids,
        kv_lens=kv_lens,
        max_q_len=q_len,
        quant_type=quant_type,
        block_mask=block_mask,
        sparsity_bucket=bucket,
        effective_skip_ratio=effective_skip_ratio,
    )


def _requested_count(request, option, default):
    invocation_args = tuple(map(str, request.config.invocation_params.args))
    explicitly_set = any(
        arg == option or arg.startswith(f"{option}=") for arg in invocation_args
    )
    return int(request.config.getoption(option)) if explicitly_set else default


def _selected_flagattention_impl():
    has_tle = bool(getattr(bsa_ops, "_HAS_TLE_HOPPER", False))
    if has_tle and torch.cuda.get_device_capability() == (9, 0):
        return "TLE"
    return "Triton"


def _bench_cuda_graph(call_fn, warmup, repetitions):
    """Measure CUDA Graph replay with CUDA events, matching hpc-ops-sc."""
    # Compile/autotune and initialize both provider dispatch paths before
    # capture.  None of this host-side setup belongs to kernel timing.
    call_fn()
    for _ in range(warmup):
        call_fn()
    torch.cuda.synchronize()

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            call_fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(repetitions)
    ]
    for start, end in events:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return float(median(start.elapsed_time(end) for start, end in events))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("q_len,kv_len", CASES)
@pytest.mark.parametrize("quant_type", [0, 1])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("kv_layout", ["nhd", "hnd"])
def test_attention_blocksparse_prefill_fp8_perf(
    request, q_len, kv_len, quant_type, masked, kv_layout
):
    inputs = _make_inputs(q_len, kv_len, quant_type, masked, kv_layout)
    warmup = _requested_count(request, "--warmup", DEFAULT_GRAPH_WARMUP)
    repetitions = _requested_count(request, "--iter", DEFAULT_GRAPH_ITERS)
    if warmup < 0 or repetitions < 1:
        raise ValueError("--warmup must be >= 0 and --iter must be >= 1")
    flagattention_impl = _selected_flagattention_impl()
    flagattention_output = torch.empty_like(inputs.q, dtype=torch.bfloat16)

    def run_flagattention():
        return attention_with_kvcache_blocksparse_prefill_fp8(
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.q_scale,
            inputs.k_scale,
            inputs.v_scale,
            inputs.cu_seqlens_q,
            inputs.block_ids,
            inputs.kv_lens,
            inputs.max_q_len,
            inputs.quant_type,
            block_mask=inputs.block_mask,
            output=flagattention_output,
            sparsity_bucket=inputs.sparsity_bucket,
        )

    flagattention_ms = _bench_cuda_graph(run_flagattention, warmup, repetitions)

    if hpc is None:
        PERF_RESULTS.append(
            {
                "q_len": q_len,
                "kv_len": kv_len,
                "quant_type": quant_type,
                "masked": masked,
                "effective_skip_ratio": inputs.effective_skip_ratio,
                "kv_layout": kv_layout,
                "flagattention_impl": flagattention_impl,
                "flagattention_ms": flagattention_ms,
                "hpc_ms": None,
                "warmup": warmup,
                "repetitions": repetitions,
            }
        )
        return

    hpc_quant_type = (
        hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD
        if quant_type == 0
        else hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR
    )
    hpc_output = torch.empty_like(inputs.q, dtype=torch.bfloat16)

    def run_hpc_cuda():
        return hpc.attention_with_kvcache_blocksparse_prefill_fp8(
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.q_scale,
            inputs.k_scale,
            inputs.v_scale,
            inputs.cu_seqlens_q,
            inputs.block_ids,
            inputs.kv_lens,
            inputs.max_q_len,
            quant_type=hpc_quant_type,
            block_mask=inputs.block_mask,
            output=hpc_output,
        )

    hpc_ms = _bench_cuda_graph(run_hpc_cuda, warmup, repetitions)
    PERF_RESULTS.append(
        {
            "q_len": q_len,
            "kv_len": kv_len,
            "quant_type": quant_type,
            "masked": masked,
            "effective_skip_ratio": inputs.effective_skip_ratio,
            "kv_layout": kv_layout,
            "flagattention_impl": flagattention_impl,
            "flagattention_ms": flagattention_ms,
            "hpc_ms": hpc_ms,
            "warmup": warmup,
            "repetitions": repetitions,
        }
    )
