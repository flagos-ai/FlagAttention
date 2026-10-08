"""Correctness tests for the Triton MoBA implementation."""

from itertools import accumulate

import pytest
import torch

from flag_attn.FLA.moba.naive_moba import moba_attn_varlen_naive
from flag_attn.FLA.moba.parallel import parallel_moba
from flag_attn.FLA.moba.routing import (
    _compute_chunk_means_triton,
    _compute_selected_chunks_streaming_triton,
    _exclusive_prefix_sum_triton,
    _max_counts_triton,
    calc_chunks,
)

try:
    from flash_moba import flash_moba_varlen_func
except ImportError:
    flash_moba_varlen_func = None


pytestmark = [
    pytest.mark.parallel_moba,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Triton MoBA requires CUDA",
    ),
]


CASES = [
    pytest.param(
        (64,), 1, 1, 64, 128, 1, torch.bfloat16, id="single-partial-k1"
    ),
    pytest.param(
        (512,), 4, 4, 64, 128, 1, torch.float16, id="multi-chunk-k1"
    ),
    pytest.param(
        (512,), 2, 2, 128, 128, 2, torch.bfloat16, id="basic-sparse"
    ),
    pytest.param(
        (513,), 4, 4, 64, 128, 4, torch.bfloat16, id="one-token-tail"
    ),
    pytest.param(
        (127, 128, 129), 4, 4, 64, 128, 2, torch.float16, id="chunk-boundaries"
    ),
    pytest.param(
        (200, 400, 648),
        4,
        4,
        128,
        256,
        3,
        torch.bfloat16,
        id="packed-varlen",
    ),
    pytest.param(
        (1024,), 8, 8, 128, 128, 8, torch.bfloat16, id="representative-k8"
    ),
    pytest.param(
        (1024,), 4, 4, 64, 256, 64, torch.float16, id="topk-saturates"
    ),
    pytest.param((300,), 3, 3, 80, 96, 3, torch.float16, id="non-power-of-two"),
    pytest.param((256,), 1, 1, 8, 128, 2, torch.float16, id="small-head-dim"),
    pytest.param((256,), 1, 1, 256, 128, 2, torch.float16, id="head-dim-256"),
    pytest.param((384,), 4, 2, 64, 128, 2, torch.float16, id="gqa-4q-2kv"),
    pytest.param((384,), 8, 1, 128, 128, 3, torch.bfloat16, id="mqa-8q-1kv"),
]


def generate_data(seq_lens, num_q_heads, num_kv_heads, head_dim, dtype):
    """Generate deterministic packed inputs for explicit sequence lengths."""
    torch.manual_seed(0)
    total_tokens = sum(seq_lens)
    q = torch.randn(
        (total_tokens, num_q_heads, head_dim),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        (total_tokens, num_kv_heads, head_dim),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        (total_tokens, num_kv_heads, head_dim),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    cu_seqlens = torch.tensor(
        [0, *accumulate(seq_lens)],
        device="cuda",
        dtype=torch.int32,
    )
    return q, k, v, cu_seqlens, max(seq_lens)


def _naive_with_gqa(q, k, v, cu_seqlens, max_seqlen, chunk_size, topk):
    repeats = q.shape[1] // k.shape[1]
    if repeats > 1:
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    return moba_attn_varlen_naive(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        moba_chunk_size=chunk_size,
        moba_topk=topk,
    )


@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "chunk_size",
        "topk",
        "dtype",
    ),
    [
        pytest.param(
            (384,), 4, 4, 64, 128, 2, torch.bfloat16, id="inference-mha"
        ),
        pytest.param(
            (384,), 4, 2, 64, 128, 2, torch.bfloat16, id="inference-gqa"
        ),
        pytest.param(
            (384,), 4, 1, 64, 128, 2, torch.bfloat16, id="inference-mqa"
        ),
        pytest.param(
            (127, 256, 385),
            8,
            2,
            128,
            128,
            4,
            torch.bfloat16,
            id="inference-packed-tail",
        ),
        pytest.param(
            (4096, 4096),
            4,
            4,
            64,
            128,
            8,
            torch.bfloat16,
            id="inference-b2-batch-aware",
        ),
        pytest.param(
            (1024, 1536, 2048, 2560),
            4,
            2,
            64,
            128,
            8,
            torch.bfloat16,
            id="inference-b4-varlen",
        ),
    ],
)
def test_inference_path_matches_autograd_path(
    seq_lens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    chunk_size,
    topk,
    dtype,
):
    q, k, v, cu_seqlens, max_seqlen = generate_data(
        seq_lens, num_q_heads, num_kv_heads, head_dim, dtype
    )
    regular = parallel_moba(
        q, k, v, cu_seqlens, max_seqlen, chunk_size, topk
    )
    with torch.no_grad():
        inference = parallel_moba(
            q, k, v, cu_seqlens, max_seqlen, chunk_size, topk
        )
    torch.testing.assert_close(
        inference,
        regular.detach(),
        atol=4e-3,
        rtol=1e-3,
    )


def test_inference_path_is_cuda_graph_capturable():
    q, k, v, cu_seqlens, max_seqlen = generate_data(
        (384,), 4, 2, 64, torch.bfloat16
    )
    with torch.no_grad():
        for _ in range(3):
            eager = parallel_moba(
                q, k, v, cu_seqlens, max_seqlen, 128, 2
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = parallel_moba(
                q, k, v, cu_seqlens, max_seqlen, 128, 2
            )
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, atol=4e-3, rtol=1e-3)


@pytest.mark.parametrize("num_experts", [1, 1024, 1025, 65537, 131072])
@torch.no_grad()
def test_hierarchical_exclusive_prefix_sum(num_experts):
    counts = (torch.arange(num_experts, device="cuda") % 7).to(torch.int32)
    offsets = torch.empty(
        (num_experts + 1,), device="cuda", dtype=torch.int32
    )
    _exclusive_prefix_sum_triton(counts, offsets)
    expected = torch.cat(
        [
            torch.zeros((1,), device="cuda", dtype=torch.int32),
            torch.cumsum(counts, dim=0, dtype=torch.int32),
        ]
    )
    torch.testing.assert_close(offsets, expected, rtol=0, atol=0)
    assert _max_counts_triton(counts).item() == counts.max().item()


def _full_causal_attention(q, k, v, cu_seqlens):
    """Independent full-attention reference for saturated-top-k properties."""
    repeats = q.shape[1] // k.shape[1]
    if repeats > 1:
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    outputs = []
    scale = q.shape[-1] ** -0.5
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        q_seq = q[start:end].float()
        k_seq = k[start:end].float()
        v_seq = v[start:end].float()
        scores = torch.einsum("thd,shd->hts", q_seq, k_seq) * scale
        causal = torch.ones(
            (end - start, end - start),
            dtype=torch.bool,
            device=q.device,
        ).tril_()
        probabilities = scores.masked_fill(~causal, -torch.inf).softmax(dim=-1)
        outputs.append(torch.einsum("hts,shd->thd", probabilities, v_seq))
    return torch.cat(outputs).to(q.dtype)


def _assert_close(
    name,
    actual,
    expected,
    mean_tolerance=4e-4,
    max_tolerance=4e-2,
):
    diff = (actual.float() - expected.float()).abs()
    torch.testing.assert_close(actual, expected, atol=max_tolerance, rtol=2e-2)
    assert diff.max().item() < max_tolerance, (
        f"{name} max diff: {diff.max().item()}"
    )
    assert diff.mean().item() < mean_tolerance, (
        f"{name} mean diff: {diff.mean().item()}"
    )


@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "chunk_size",
        "topk",
        "dtype",
    ),
    CASES,
)
def test_moba_matches_naive_forward_and_backward(
    seq_lens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    chunk_size,
    topk,
    dtype,
):
    q, k, v, cu_seqlens, max_seqlen = generate_data(
        seq_lens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        dtype,
    )
    output_grad = torch.randn_like(q)

    output = parallel_moba(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        moba_chunk_size=chunk_size,
        moba_topk=topk,
    )
    grads = torch.autograd.grad(output, (q, k, v), output_grad)

    reference = _naive_with_gqa(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        chunk_size,
        topk,
    )
    reference_grads = torch.autograd.grad(reference, (q, k, v), output_grad)

    _assert_close("output", output, reference)
    kv_group_size = num_q_heads // num_kv_heads
    for name, actual, expected in zip(("dq", "dk", "dv"), grads, reference_grads):
        is_kv_grad = name != "dq"
        mean_tolerance = 4e-4 * (kv_group_size if is_kv_grad else 1)
        max_tolerance = 4e-2 * (kv_group_size**0.5 if is_kv_grad else 1)
        _assert_close(
            name,
            actual,
            expected,
            mean_tolerance,
            max_tolerance,
        )


def test_moba_rejects_invalid_configuration():
    q, k, v, cu_seqlens, max_seqlen = generate_data(
        (128,), 4, 4, 64, torch.float16
    )

    with pytest.raises(ValueError, match="moba_topk"):
        parallel_moba(q, k, v, cu_seqlens, max_seqlen, 128, 0)

    with pytest.raises(ValueError, match="divisible"):
        parallel_moba(
            q,
            k[:, :3],
            v[:, :3],
            cu_seqlens,
            max_seqlen,
            128,
            2,
        )

    with pytest.raises(TypeError, match="same dtype"):
        parallel_moba(
            q,
            k.bfloat16(),
            v,
            cu_seqlens,
            max_seqlen,
            128,
            2,
        )

    with pytest.raises(ValueError, match="<= 64"):
        parallel_moba(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            128,
            65,
        )

    with pytest.raises(ValueError, match="smaller than the actual"):
        parallel_moba(q, k, v, cu_seqlens, 64, 128, 1)

    with pytest.raises(ValueError, match="start at 0"):
        bad_start = torch.tensor([1, 128], device="cuda", dtype=torch.int32)
        parallel_moba(q, k, v, bad_start, max_seqlen, 128, 1)

    with pytest.raises(ValueError, match="packed token count"):
        bad_end = torch.tensor([0, 127], device="cuda", dtype=torch.int32)
        parallel_moba(q, k, v, bad_end, max_seqlen, 128, 1)

    with pytest.raises(ValueError, match="non-empty increasing"):
        empty_sequence = torch.tensor(
            [0, 0, 128], device="cuda", dtype=torch.int32
        )
        parallel_moba(
            q, k, v, empty_sequence, max_seqlen, 128, 1
        )


def test_mutating_cu_seqlens_invalidates_cached_metadata():
    q, k, v, cu_seqlens, max_seqlen = generate_data(
        (128,), 1, 1, 16, torch.float16
    )
    parallel_moba(q, k, v, cu_seqlens, max_seqlen, 128, 1)
    cu_seqlens[-1] = 64

    with pytest.raises(ValueError, match="packed token count"):
        parallel_moba(
            q, k, v, cu_seqlens, max_seqlen, 128, 1
        )


@pytest.mark.parametrize(
    ("seq_lens", "num_heads", "head_dim", "chunk_size", "topk"),
    [
        pytest.param((512,), 4, 64, 128, 64, id="T512-H4-D64-C128-K64"),
        pytest.param((1024,), 8, 64, 256, 64, id="T1024-H8-D64-C256-K64"),
        pytest.param((2048,), 4, 128, 256, 64, id="T2048-H4-D128-C256-K64"),
        pytest.param((256, 512, 256), 4, 64, 128, 64, id="varlen-256-512-256"),
        pytest.param((512, 1024), 8, 64, 256, 64, id="varlen-512-1024"),
        pytest.param(
            (200, 400, 800, 648),
            4,
            128,
            128,
            64,
            id="varlen-200-400-800-648",
        ),
        pytest.param(
            (129, 256, 300), 2, 64, 128, 64, id="varlen-partial-chunks"
        ),
    ],
)
def test_saturated_topk_matches_independent_full_attention(
    seq_lens, num_heads, head_dim, chunk_size, topk
):
    q, k, v, cu_seqlens, max_seqlen = generate_data(
        seq_lens, num_heads, num_heads, head_dim, torch.float16
    )
    output_grad = torch.randn_like(q)
    output = parallel_moba(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        moba_chunk_size=chunk_size,
        moba_topk=topk,
    )
    grads = torch.autograd.grad(output, (q, k, v), output_grad)
    reference = _full_causal_attention(q, k, v, cu_seqlens)
    reference_grads = torch.autograd.grad(reference, (q, k, v), output_grad)

    _assert_close("output", output, reference)
    for name, actual, expected in zip(("dq", "dk", "dv"), grads, reference_grads):
        _assert_close(name, actual, expected)


def test_sparse_topk_does_not_fall_back_to_dense_attention():
    q, k, v, cu_seqlens, max_seqlen = generate_data(
        (512,), 2, 2, 64, torch.float16
    )
    sparse = parallel_moba(
        q, k, v, cu_seqlens, max_seqlen, moba_chunk_size=128, moba_topk=2
    )
    dense = _full_causal_attention(q, k, v, cu_seqlens)
    assert (sparse.float() - dense.float()).abs().mean().item() > 1e-3


def test_many_short_sequences_cap_topk_per_sequence():
    seq_lens = (256,) * 64
    q, k, v, cu_seqlens, max_seqlen = generate_data(
        seq_lens, 1, 1, 16, torch.float16
    )
    output = parallel_moba(
        q, k, v, cu_seqlens, max_seqlen, moba_chunk_size=128, moba_topk=64
    )
    reference = _full_causal_attention(q, k, v, cu_seqlens)
    _assert_close("output", output, reference)


def test_noncontiguous_inputs_match_naive():
    torch.manual_seed(42)
    seqlen, heads, head_dim = 300, 2, 64

    def make_input():
        base = torch.randn(
            (seqlen, heads, head_dim * 2),
            device="cuda",
            dtype=torch.float16,
        )
        return base[..., ::2].detach().requires_grad_(True)

    q, k, v = make_input(), make_input(), make_input()
    assert not q.is_contiguous()
    cu_seqlens = torch.tensor([0, seqlen], device="cuda", dtype=torch.int32)
    output = parallel_moba(q, k, v, cu_seqlens, seqlen, 96, 3)
    reference = moba_attn_varlen_naive(q, k, v, cu_seqlens, seqlen, 96, 3)
    output_grad = torch.randn_like(output)
    grads = torch.autograd.grad(output, (q, k, v), output_grad)
    reference_grads = torch.autograd.grad(reference, (q, k, v), output_grad)

    _assert_close("output", output, reference)
    for name, actual, expected in zip(("dq", "dk", "dv"), grads, reference_grads):
        _assert_close(name, actual, expected)


@pytest.mark.skipif(flash_moba_varlen_func is None, reason="FlashMoBA is required")
@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "chunk_size",
        "topk",
        "dtype",
    ),
    [
        pytest.param(
            (512,), 4, 4, 64, 128, 1, torch.float16, id="topk1-local-chunk"
        ),
        pytest.param(
            (127, 128, 129),
            4,
            4,
            64,
            128,
            2,
            torch.float16,
            id="topk2-varlen",
        ),
        pytest.param(
            (1024,), 8, 8, 128, 128, 8, torch.bfloat16, id="topk8-bf16"
        ),
        pytest.param(
            (512,), 4, 4, 64, 128, 64, torch.float16, id="topk64-saturated"
        ),
        pytest.param(
            (384,), 4, 2, 64, 128, 2, torch.float16, id="gqa-4q-2kv"
        ),
    ],
)
def test_matches_flash_moba_forward_and_backward(
    seq_lens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    chunk_size,
    topk,
    dtype,
):
    q, k, v, cu_seqlens, max_seqlen = generate_data(
        seq_lens, num_q_heads, num_kv_heads, head_dim, dtype
    )
    output_grad = torch.randn_like(q)
    output = parallel_moba(
        q, k, v, cu_seqlens, max_seqlen, chunk_size, topk
    )
    grads = torch.autograd.grad(output, (q, k, v), output_grad)
    cuda_output = flash_moba_varlen_func(
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
    cuda_grads = torch.autograd.grad(cuda_output, (q, k, v), output_grad)

    _assert_close("output", output, cuda_output)
    for name, actual, expected in zip(("dq", "dk", "dv"), grads, cuda_grads):
        is_kv_grad = name != "dq"
        kv_group_size = num_q_heads // num_kv_heads
        _assert_close(
            name,
            actual,
            expected,
            mean_tolerance=4e-4 * (kv_group_size if is_kv_grad else 1),
            max_tolerance=4e-2 * (kv_group_size**0.5 if is_kv_grad else 1),
        )


@pytest.mark.parametrize("routed_topk", [7, 33, 63])
@torch.no_grad()
def test_streaming_topk_across_chunk_tiles(routed_topk):
    """The global top-k must remain correct beyond one 128-chunk tile."""
    seqlen = 32768
    chunk_size = 128
    head_dim = 64
    q = torch.ones((seqlen, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    token_chunk = torch.arange(seqlen, device="cuda") // chunk_size + 1
    k = token_chunk[:, None, None].expand(-1, 1, head_dim).to(torch.bfloat16)
    cu_seqlens = torch.tensor([0, seqlen], device="cuda", dtype=torch.int32)

    cu_chunk, filtered_indices, num_filtered, _ = calc_chunks(
        cu_seqlens,
        chunk_size,
    )
    chunk_means = _compute_chunk_means_triton(
        k,
        cu_chunk,
        filtered_indices,
        chunk_size,
    )
    chunk_end = cu_chunk.index_select(0, filtered_indices + 1)
    batch_end = torch.full_like(chunk_end, seqlen)
    selected = _compute_selected_chunks_streaming_triton(
        q,
        chunk_means,
        chunk_end,
        batch_end,
        routed_topk,
    )

    for token in (0, 127, 128, 255, 16384, 16512, seqlen - 1):
        num_history = token // chunk_size
        selected_count = min(routed_topk, num_history)
        expected = list(range(num_history - 1, num_history - selected_count - 1, -1))
        expected.extend([-1] * (routed_topk - selected_count))
        assert selected[:, 0, token].tolist() == expected
