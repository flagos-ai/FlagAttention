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


from itertools import accumulate

import pytest
import torch
import triton

from flag_attn.testing import backend as test_backend
from flag_attn import parallel_moba, runtime

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:  # Direct script execution.
    from recording import BenchmarkRecorder
from flag_attn.FLA.moba.routing import (
    _build_dense_chunk_metadata_triton,
    _compute_dense_chunk_means_triton,
    _compute_selected_chunks_batched_streaming_triton,
)

flash_moba_varlen_func = None
flash_moba_gpu = None
flash_topk_mean_pool = None
if test_backend.is_nvidia():
    try:
        from flash_moba import flash_moba_varlen_func
        from flash_moba.flash_moba_interface import flash_moba_gpu, flash_topk_mean_pool
    except ImportError:
        flash_moba_varlen_func = None
        flash_moba_gpu = None
        flash_topk_mean_pool = None

mate_parallel_moba = None
if runtime.device.vendor_name == "mthreads":
    try:
        from BaselineBenchmark.backends._mthreads.ops.moba import (
            parallel_moba as mate_parallel_moba,
        )
    except ImportError:
        pass



WARMUP_MS = 25
REPETITION_MS = 100
MAX_MEAN_ABS_DIFF = 3e-3
TABLE_WIDTH = 75
BENCHMARK_CASES = [
    # Match the official FlashMoBA benchmark's batch size while varying only
    # per-sequence length so the scaling comparison remains interpretable.
    ((4096, 4096), 16, 128, 128, 8, torch.bfloat16),
    ((8192, 8192), 16, 128, 128, 8, torch.bfloat16),
    ((16384, 16384), 16, 128, 128, 8, torch.bfloat16),
    ((32768, 32768), 16, 128, 128, 8, torch.bfloat16),
    ((65536, 65536), 16, 128, 128, 8, torch.bfloat16),
    ((131072, 131072), 16, 128, 128, 8, torch.bfloat16),
    ((262144, 262144), 16, 128, 128, 8, torch.bfloat16),
]


def generate_data(seq_lens, num_heads, head_dim, dtype):
    """Generate packed Q/K/V tensors for explicit per-sequence lengths."""
    if not seq_lens or any(length <= 0 for length in seq_lens):
        raise ValueError(f"seq_lens must contain positive lengths, got {seq_lens}")

    torch.manual_seed(42)
    device = torch.device(test_backend.device)
    total_tokens = sum(seq_lens)
    shape = (total_tokens, num_heads, head_dim)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    cu_seqlens = torch.tensor(
        [0, *accumulate(seq_lens)],
        dtype=torch.int32,
        device=device,
    )
    return q, k, v, cu_seqlens, max(seq_lens)


@torch.no_grad()
def _validate_routing_contract():
    """Verify that both providers implement the same causal top-k contract.

    Random routing scores can be nearly tied, so different BF16 reduction orders
    are allowed to choose different chunks.  This check instead uses exactly
    representable, well-separated chunk scores and requires the selected chunk
    sets and the number of valid routes to match exactly for every query/head.
    """
    seq_lens, num_heads, head_dim, chunk_size, topk, dtype = BENCHMARK_CASES[0]
    device = torch.device(test_backend.device)
    total_tokens = sum(seq_lens)
    batch_size = len(seq_lens)
    max_seqlen = max(seq_lens)
    max_chunks = triton.cdiv(max_seqlen, chunk_size)

    local_positions = torch.cat(
        [torch.arange(length, device=device) for length in seq_lens]
    )
    batch_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        torch.tensor(seq_lens, device=device),
    )
    local_chunks = local_positions // chunk_size
    q = torch.ones(
        (total_tokens, num_heads, head_dim), device=device, dtype=dtype
    )
    # Multiples of 1 / 128 are exact in BF16 and keep adjacent historical
    # chunks far enough apart that selection is independent of reduction order.
    chunk_scores = (local_chunks + 1).to(dtype) / 128
    k = chunk_scores[:, None, None].expand(-1, num_heads, head_dim).contiguous()
    cu_seqlens = torch.tensor(
        [0, *accumulate(seq_lens)], device=device, dtype=torch.int32
    )

    pooled_k, cu_pooled_k, _ = flash_topk_mean_pool(
        k, cu_seqlens, max_seqlen, chunk_size
    )
    _, _, _, _, flash_indices = flash_moba_gpu.moba_fused_topk(
        q,
        pooled_k,
        cu_seqlens,
        cu_seqlens,
        cu_pooled_k,
        max_seqlen,
        max_seqlen,
        topk,
        chunk_size,
        True,
    )
    flash_routes = flash_indices[..., :topk].to(torch.int64)

    (
        cu_chunk,
        chunk_end,
        _,
        valid_historical,
        max_chunks_per_sequence,
    ) = _build_dense_chunk_metadata_triton(
        cu_seqlens, max_seqlen, chunk_size
    )
    assert max_chunks_per_sequence == max_chunks
    chunk_means = _compute_dense_chunk_means_triton(
        k, cu_chunk, valid_historical, chunk_size
    )
    historical_routes = _compute_selected_chunks_batched_streaming_triton(
        q,
        chunk_means,
        chunk_end,
        cu_seqlens,
        max_seqlen,
        max_chunks_per_sequence,
        topk - 1,
    ).permute(2, 1, 0).to(torch.int64)

    chunk_bases = (batch_ids * max_chunks_per_sequence)[:, None, None]
    historical_routes = torch.where(
        historical_routes >= 0,
        historical_routes - chunk_bases,
        -1,
    )
    current_routes = local_chunks[:, None, None].expand(-1, num_heads, 1)
    triton_routes = torch.cat((current_routes, historical_routes), dim=-1)

    expected_route_counts = torch.minimum(
        local_chunks + 1,
        torch.full_like(local_chunks, topk),
    )[:, None].expand(-1, num_heads)
    torch.testing.assert_close(
        (triton_routes >= 0).sum(-1),
        expected_route_counts,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        (flash_routes >= 0).sum(-1),
        expected_route_counts,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        triton_routes.sort(dim=-1).values,
        flash_routes.sort(dim=-1).values,
        rtol=0,
        atol=0,
    )


def _triton_moba(q, k, v, cu_seqlens, max_seqlen, chunk_size, topk):
    return parallel_moba(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        moba_chunk_size=chunk_size,
        moba_topk=topk,
    )


def _flash_moba(q, k, v, cu_seqlens, max_seqlen, chunk_size, topk):
    return flash_moba_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        moba_chunk_size=chunk_size,
        moba_topk=topk,
        causal=True,
    )


def benchmark_kernel(func, args):
    """Return median device-event latency in milliseconds."""

    @torch.no_grad()
    def run():
        return func(*args)

    return test_backend.do_bench(
        run,
        warmup=WARMUP_MS,
        rep=REPETITION_MS,
        return_mode="median",
    )


def _format_shape(seq_lens, num_heads, head_dim, chunk_size, topk, dtype):
    if len(set(seq_lens)) == 1:
        sequence_shape = f"B{len(seq_lens)}x{seq_lens[0]}"
    else:
        sequence_shape = f"V{len(seq_lens)}x{sum(seq_lens)}"
    dtype_name = "bf16" if dtype == torch.bfloat16 else "f16"
    return (
        f"{sequence_shape} H{num_heads} D{head_dim} "
        f"C{chunk_size} K{topk} {dtype_name}"
    )


def _print_results(results, baseline_name):
    print(f"\n{'=' * TABLE_WIDTH}")
    print("MoBA inference benchmark")
    print(f"device: {test_backend.get_device_name()}")
    print(f"baseline: {baseline_name or 'unavailable'}")
    print(f"timing: warmup={WARMUP_MS}ms  rep={REPETITION_MS}ms")
    print(f"{'-' * TABLE_WIDTH}")
    print(f"{'Shape':<38} {'Flag ms':>8} {'Base ms':>8} {'Base/Flag':>10}")
    print(f"{'-' * TABLE_WIDTH}")

    for shape, flag_ms, baseline_ms in results:
        baseline_text = "N/A" if baseline_ms is None else f"{baseline_ms:.3f}"
        speedup = None if baseline_ms is None else baseline_ms / flag_ms
        speedup_text = "N/A" if speedup is None else f"{speedup:.2f}x"
        print(
            f"{shape:<38} {flag_ms:>8.3f} {baseline_text:>8} "
            f"{speedup_text:>10}"
        )

    print(f"{'-' * TABLE_WIDTH}")
    print("Base/Flag > 1.00x means FlagAttention is faster.")
    print(f"{'=' * TABLE_WIDTH}")


@pytest.mark.parallel_moba
def test_attn_varlen_moba_speed(record_property):
    if not test_backend.supports_operator("parallel_moba"):
        pytest.skip("MoBA benchmark requires an available accelerator")
    baseline_name = baseline_fn = None
    max_verified_seqlen = None
    if flash_moba_varlen_func is not None:
        baseline_name = "FlashMoBA"
        baseline_fn = _flash_moba
        _validate_routing_contract()
    elif mate_parallel_moba is not None:
        # The experimental MATE baseline is verified only through 65536 tokens.
        baseline_name = "MATE MoBA (experimental)"
        baseline_fn = mate_parallel_moba
        max_verified_seqlen = 65536

    recorder = BenchmarkRecorder(
        record_property,
        op_name="parallel_moba",
        dtype=str(BENCHMARK_CASES[0][-1]),
        baseline=baseline_name,
        phase="inference",
    )
    results = []
    for (
        seq_lens,
        num_heads,
        head_dim,
        chunk_size,
        topk,
        dtype,
    ) in BENCHMARK_CASES:
        q, k, v, cu_seqlens, max_seqlen = generate_data(
            seq_lens, num_heads, head_dim, dtype
        )
        args = (q, k, v, cu_seqlens, max_seqlen, chunk_size, topk)
        case_baseline_fn = baseline_fn
        if max_verified_seqlen is not None and max_seqlen > max_verified_seqlen:
            case_baseline_fn = None
        shape = _format_shape(
            seq_lens, num_heads, head_dim, chunk_size, topk, dtype
        )

        with torch.no_grad():
            flag_output = _triton_moba(*args)
            assert flag_output.shape == q.shape
            assert flag_output.dtype == dtype
            assert torch.isfinite(flag_output).all()
            mean_abs_diff = None
            if case_baseline_fn:
                base_output = case_baseline_fn(*args)
                assert base_output.shape == q.shape
                assert base_output.dtype == dtype
                assert torch.isfinite(base_output).all()
                mean_abs_diff = (
                    (flag_output.float() - base_output.float()).abs().mean().item()
                )
                assert mean_abs_diff < MAX_MEAN_ABS_DIFF, (
                    f"{shape}: {baseline_name} mean absolute difference "
                    f"{mean_abs_diff:.3e} exceeds {MAX_MEAN_ABS_DIFF:.1e}"
                )

        flag_ms = benchmark_kernel(_triton_moba, args)
        baseline_ms = benchmark_kernel(case_baseline_fn, args) if case_baseline_fn else None
        results.append((shape, flag_ms, baseline_ms))
        recorder.add(
            shape_detail=(seq_lens, num_heads, head_dim, chunk_size, topk),
            latency=flag_ms,
            latency_base=baseline_ms,
            mean_abs_diff=mean_abs_diff,
            baseline_used=baseline_name if case_baseline_fn else None,
        )
        del args, q, k, v, cu_seqlens

    _print_results(results, baseline_name)
    recorder.record()
