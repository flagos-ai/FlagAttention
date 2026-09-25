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

from flag_attn.FLA.moba.parallel import parallel_moba
from flag_attn.FLA.moba.routing import (
    _build_dense_chunk_metadata_triton,
    _compute_dense_chunk_means_triton,
    _compute_selected_chunks_batched_streaming_triton,
)

try:
    from flash_moba import flash_moba_varlen_func
    from flash_moba.flash_moba_interface import (
        flash_moba_gpu,
        flash_topk_mean_pool,
    )
except ImportError:
    flash_moba_varlen_func = None
    flash_moba_gpu = None
    flash_topk_mean_pool = None


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
    ((524288, 524288), 16, 128, 128, 8, torch.bfloat16),
]


def generate_data(seq_lens, num_heads, head_dim, dtype):
    """Generate packed Q/K/V tensors for explicit per-sequence lengths."""
    if not seq_lens or any(length <= 0 for length in seq_lens):
        raise ValueError(f"seq_lens must contain positive lengths, got {seq_lens}")

    torch.manual_seed(42)
    device = torch.device("cuda")
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
    device = torch.device("cuda")
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
    """Return median CUDA latency in milliseconds."""

    @torch.no_grad()
    def run():
        return func(*args)

    return triton.testing.do_bench(
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


def _print_results(results):
    print(f"\n{'=' * TABLE_WIDTH}")
    print("MoBA benchmark: Triton MoBA vs FlashMoBA CUDA")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"timing: warmup={WARMUP_MS}ms  rep={REPETITION_MS}ms")
    print(f"{'-' * TABLE_WIDTH}")
    print(
        f"{'Shape':<38} {'Tri ms':>8} {'CUDA ms':>7} {'Speedup':>8}"
    )
    print(f"{'-' * TABLE_WIDTH}")

    for shape, triton_ms, cuda_ms in results:
        triton_speedup = cuda_ms / triton_ms
        print(
            f"{shape:<38} {triton_ms:>8.3f} {cuda_ms:>7.3f} "
            f"{triton_speedup:>7.2f}x"
        )

    print(f"{'-' * TABLE_WIDTH}")
    print("Speedup = CUDA / Triton; > 1.00x means Triton wins.")
    print(f"{'=' * TABLE_WIDTH}")


@pytest.mark.parallel_moba
def test_attn_varlen_moba_speed():
    if not torch.cuda.is_available():
        pytest.skip("MoBA benchmark requires CUDA")
    if flash_moba_varlen_func is None:
        pytest.skip("FlashMoBA is required for the CUDA comparison")

    _validate_routing_contract()

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
            seq_lens,
            num_heads,
            head_dim,
            dtype,
        )
        args = (q, k, v, cu_seqlens, max_seqlen, chunk_size, topk)
        shape = _format_shape(
            seq_lens,
            num_heads,
            head_dim,
            chunk_size,
            topk,
            dtype,
        )

        # A fast result is meaningful only when both providers compute the same
        # operation. Mean error is used because near-tied routing scores can send
        # a handful of queries to different chunks and inflate max error.
        with torch.no_grad():
            triton_output = _triton_moba(*args)
            cuda_output = _flash_moba(*args)
            assert triton_output.shape == cuda_output.shape == q.shape
            assert triton_output.dtype == cuda_output.dtype == dtype
            assert torch.isfinite(triton_output).all()
            assert torch.isfinite(cuda_output).all()
            mean_abs_diff = (
                (triton_output.float() - cuda_output.float()).abs().mean().item()
            )
        assert mean_abs_diff < MAX_MEAN_ABS_DIFF, (
            f"{shape}: mean absolute difference {mean_abs_diff:.3e} exceeds "
            f"{MAX_MEAN_ABS_DIFF:.1e}"
        )

        triton_ms = benchmark_kernel(_triton_moba, args)
        cuda_ms = benchmark_kernel(_flash_moba, args)
        results.append((shape, triton_ms, cuda_ms))

    _print_results(results)
