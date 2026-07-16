import pytest
import torch

import flag_attn

NUM_BLOCKS = 1000


def base_paged_attention(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    max_seq_len,
    num_splits=0,
    dtype=torch.float16,
    device="cuda",
):
    torch.set_default_dtype(dtype)
    torch.set_default_device(device=device)

    num_kv_heads = num_query_heads // query_group_size

    context_lens = torch.randint(1, max_seq_len, [num_seqs], dtype=torch.int32)
    context_lens[0] = max_seq_len
    max_context_len = context_lens.max().item()
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size

    attn_scale = head_size**-0.5
    q = torch.empty(num_seqs, num_query_heads, head_size)
    q.uniform_(-attn_scale, attn_scale)

    k_cache = torch.empty(NUM_BLOCKS, num_kv_heads, block_size, head_size)
    k_cache.uniform_(-attn_scale, attn_scale)
    v_cache = torch.empty_like(k_cache)
    v_cache.uniform_(-attn_scale, attn_scale)

    # (NUM_SEQS, MAX_NUM_BLOCKS_PER_SEQ)
    block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq))

    out = flag_attn.paged_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        num_splits,
    )

    ref_out = flag_attn.testing.paged_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        attn_scale,
    )
    print(torch.abs(out - ref_out).max())
    assert torch.allclose(out, ref_out, atol=2e-3, rtol=1e-5)


@pytest.mark.parametrize("num_seqs", [1, 32])
@pytest.mark.parametrize("num_query_heads", [64])
@pytest.mark.parametrize("query_group_size", [1, 8])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [16, 128, 256])
@pytest.mark.parametrize("max_seq_len", [512, 4096])
def test_paged_attention_default(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    max_seq_len,
    dtype=torch.float16,
    device="cuda",
):
    base_paged_attention(
        num_seqs,
        num_query_heads,
        query_group_size,
        head_size,
        block_size,
        max_seq_len,
    )


@pytest.mark.parametrize("num_seqs", [1, 16])
@pytest.mark.parametrize("num_query_heads", [64])
@pytest.mark.parametrize("query_group_size", [1, 8])
@pytest.mark.parametrize("head_size", [32, 64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_splits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_paged_attention_by_num_splits(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    max_seq_len,
    num_splits,
    dtype=torch.float16,
    device="cuda",
):
    base_paged_attention(
        num_seqs,
        num_query_heads,
        query_group_size,
        head_size,
        block_size,
        max_seq_len,
        num_splits=num_splits,
    )


@pytest.mark.parametrize(
    "num_seqs, num_query_heads, query_group_size, head_size, block_size, max_seq_len, num_splits",
    [
        (1, 12, 1, 64, 16, 2, 0),
        (16, 64, 8, 32, 16, 2048, 2),
        (16, 64, 1, 64, 16, 2048, 6),
    ],
)
def test_paged_attention_by_case(
    num_seqs,
    num_query_heads,
    query_group_size,
    head_size,
    block_size,
    max_seq_len,
    num_splits,
    dtype=torch.float16,
    device="cuda",
):
    base_paged_attention(
        num_seqs,
        num_query_heads,
        query_group_size,
        head_size,
        block_size,
        max_seq_len,
        num_splits,
    )
