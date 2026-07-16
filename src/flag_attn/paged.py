import torch
import triton
import triton.language as tl


# Requires triton 2.2.0
def attention(
    query: torch.Tensor,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: torch.Tensor,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE], required same stride with key_cache
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale: float,
    max_context_len: int,
    num_splits: int = 0,
) -> None:
    out = torch.empty_like(query)

    num_seqs = query.shape[0]
    num_kv_heads = key_cache.shape[1]
    kv_block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    query_group_size = query.shape[1] // num_kv_heads

    if query_group_size == 1:
        padded_group_size = 1
    elif query_group_size < 16:
        padded_group_size = 16
    else:
        padded_group_size = triton.next_power_of_2(query_group_size)

    assert head_size in (16, 32, 64, 128, 256, 512), f"head_size={head_size}"
    assert (
        padded_group_size == 1 or kv_block_size >= 16
    ), f"kv_block_size={kv_block_size}"
    # query_group_size in (1, 2, 4, 8, 16, 32, 64, 128, 256)
    # assert query_group_size > 0 and query_group_size & (query_group_size-1) == 0,
    # f"query_group_size={query_group_size}"

    # config for A100
    # TODO: support more devices and optimize
    device = torch.cuda.device_of(query)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_splits == 0:
        if num_seqs * num_kv_heads > 2 * num_sms:
            num_splits = 1
            if max_context_len >= 4096:
                partition_size = max(256, kv_block_size)
                num_splits = triton.cdiv(max_context_len, partition_size)
        else:
            partition_size = max(256, kv_block_size)
            num_splits = triton.cdiv(max_context_len, partition_size)
            if max_context_len <= 1024 or kv_block_size >= 256:
                num_splits = 1
    elif num_splits > 1:
        partition_size = triton.cdiv(max_context_len, num_splits)
        partition_size = triton.next_power_of_2(partition_size)

    with torch.cuda.device(device):
        if num_splits == 1:
            grid = (num_seqs, num_kv_heads, 1)
            _paged_attn_kernel[grid](
                out,  # dummy input
                out,  # dummy input
                out,
                query,
                key_cache,
                value_cache,
                context_lens,
                block_tables,
                attn_scale,
                block_tables.stride(0),
                block_tables.stride(1),
                query.stride(0),
                query.stride(1),
                query.stride(2),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(1),
                out.stride(1),
                out.stride(2),
                head_size,
                query_group_size,
                padded_group_size,
                num_kv_heads,
                kv_block_size,
                PARTITION_SIZE=0,
            )

        else:
            grid = (num_seqs, num_kv_heads, num_splits)
            m_i = torch.empty(
                size=(num_seqs, num_kv_heads, num_splits, query_group_size),
                dtype=torch.float32,
                device=query.device,
            )
            l_i = torch.empty_like(m_i)
            tmp_out = torch.empty(
                size=(
                    num_seqs,
                    num_kv_heads,
                    num_splits,
                    query_group_size,
                    head_size,
                ),
                dtype=out.dtype,
                device=out.device,
            )

            assert (partition_size >= kv_block_size) and (
                partition_size % kv_block_size == 0
            ), f"partition_size={partition_size}, kv_block_size={kv_block_size}"
            _paged_attn_kernel[grid](
                m_i,
                l_i,
                tmp_out,
                query,
                key_cache,
                value_cache,
                context_lens,
                block_tables,
                attn_scale,
                block_tables.stride(0),
                block_tables.stride(1),
                query.stride(0),
                query.stride(1),
                query.stride(2),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                tmp_out.stride(0),
                tmp_out.stride(1),
                tmp_out.stride(2),
                tmp_out.stride(3),
                tmp_out.stride(4),
                head_size,
                query_group_size,
                padded_group_size,
                num_kv_heads,
                kv_block_size,
                partition_size,
            )

            reduce_grid = (num_seqs, num_kv_heads)
            next_num_splits = triton.next_power_of_2(num_splits)

            _paged_attn_v2_reduce_kernel[reduce_grid](
                out,
                m_i,
                l_i,
                tmp_out,
                context_lens,
                num_splits,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                head_size,
                query_group_size,
                num_kv_heads,
                partition_size,
                next_num_splits,
            )
    return out


def get_num_warps(QUERY_GROUP_SIZE, HEAD_SIZE, KV_BLOCK_SIZE):
    if QUERY_GROUP_SIZE == 1:
        if HEAD_SIZE >= 128 and KV_BLOCK_SIZE >= 32:
            return 16
        else:
            return 8
    else:
        return 4


def get_num_stages(PARTITION_SIZE, KV_BLOCK_SIZE):
    if PARTITION_SIZE == 0:
        return 1
    else:
        if torch.cuda.get_device_capability() == (8, 0):
            if KV_BLOCK_SIZE < 256:
                return 3
            else:
                return 2
        elif torch.cuda.get_device_capability() == (8, 6):
            if KV_BLOCK_SIZE < 256:
                return 2
            else:
                return 1
        else:
            return 1


@triton.heuristics(
    {
        "num_warps": lambda args: get_num_warps(
            args["QUERY_GROUP_SIZE"], args["HEAD_SIZE"], args["KV_BLOCK_SIZE"]
        ),
        "num_stages": lambda args: get_num_stages(
            args["QUERY_GROUP_SIZE"], args["KV_BLOCK_SIZE"]
        ),
    }
)
@triton.jit
def _paged_attn_kernel(
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    q_ptr,  # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    k_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,  # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens_ptr,  # [num_seqs]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    attn_scale,
    stride_bt0,
    stride_bt1,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_kv0,
    stride_kv1,
    stride_kv2,
    stride_kv3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_o4,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    part_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634

    USE_PARTITIONING = PARTITION_SIZE > 0
    context_len = tl.load(context_lens_ptr + seq_idx)
    if USE_PARTITIONING:
        context_start_idx = part_idx * PARTITION_SIZE
        if context_start_idx >= context_len:
            return
        context_end_idx = tl.minimum(context_start_idx + PARTITION_SIZE, context_len)
        num_blocks = tl.cdiv(context_end_idx - context_start_idx, KV_BLOCK_SIZE)
    else:
        num_blocks = tl.cdiv(context_len, KV_BLOCK_SIZE)

    block_offset = tl.arange(0, KV_BLOCK_SIZE)
    head_offset = tl.arange(0, HEAD_SIZE)
    padding_group_offset = tl.arange(0, PADDED_QUERY_GROUP_SIZE)

    kv_offset = (
        kv_head_idx * stride_kv1
        + block_offset[:, None] * stride_kv2
        + head_offset[None, :] * stride_kv3
    )

    # Load queries.
    q_offset = (
        seq_idx * stride_q0
        + (kv_head_idx * QUERY_GROUP_SIZE + padding_group_offset[:, None]) * stride_q1
        + head_offset[None, :] * stride_q2
    )
    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    # q: [PADDED_QUERY_GROUP_SIZE, HEAD_SIZE]
    q = tl.load(q_ptr + q_offset, mask=group_mask, other=0.0)

    m_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32)
    acc = tl.zeros([PADDED_QUERY_GROUP_SIZE, HEAD_SIZE], dtype=tl.float32)

    num_prev_blocks = part_idx * (PARTITION_SIZE // KV_BLOCK_SIZE)
    for i in range(num_blocks):
        block_idx = num_prev_blocks + i
        block_number = tl.load(
            block_tables_ptr + seq_idx * stride_bt0 + block_idx * stride_bt1
        )

        # Load a key block.
        kv_block_offset = block_number * stride_kv0 + kv_offset
        mask_offset = block_idx * KV_BLOCK_SIZE + block_offset
        kv_mask = mask_offset[:, None] < context_len

        # k: [KV_BLOCK_SIZE, HEAD_SIZE]
        k = tl.load(k_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)

        # qk: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        if PADDED_QUERY_GROUP_SIZE == 1:
            qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
        else:
            qk = tl.dot(q, k.T, out_dtype=tl.float32)

        qk *= attn_scale
        qk = tl.where(mask_offset < context_len, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))

        # p: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        p = tl.math.exp2((qk - m_i_new[:, None]) * log2e)
        alpha = tl.math.exp2((m_i - m_i_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLOCK_SIZE, HEAD_SIZE]
        v = tl.load(v_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)

        if PADDED_QUERY_GROUP_SIZE == 1:
            acc += tl.sum(p.T[:, :, None] * v[:, None, :], axis=0)
        else:
            p = p.to(v.dtype)
            acc += tl.dot(p, v, out_dtype=tl.float32)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    acc = acc / l_i[:, None]

    if USE_PARTITIONING:
        part_offset = (
            (seq_idx * NUM_KV_HEADS + kv_head_idx)
            * max_num_partitions
            * QUERY_GROUP_SIZE
            + part_idx * QUERY_GROUP_SIZE
            + padding_group_offset
        )
        mask = padding_group_offset < QUERY_GROUP_SIZE
        tl.store(m_i_ptr + part_offset, m_i, mask=mask)
        tl.store(l_i_ptr + part_offset, l_i, mask=mask)

    out_offset = seq_idx * stride_o0
    if USE_PARTITIONING:
        out_offset += kv_head_idx * stride_o1
    else:
        out_offset += kv_head_idx * QUERY_GROUP_SIZE * stride_o1
    out_offset += (
        part_idx * stride_o2
        + padding_group_offset[:, None] * stride_o3
        + head_offset[None, :] * stride_o4
    )

    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    tl.store(out_ptr + out_offset, acc, mask=group_mask)


@triton.jit
def _paged_attn_v2_reduce_kernel(
    out_ptr,  # [num_seqs, NUM_KV_HEADS, QUERY_GROUP_SIZE, HEAD_SIZE]
    m_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    tmp_out_ptr,  # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    context_lens_ptr,  # [num_seqs]
    max_num_partitions,  # partition stride
    stride_o0,
    stride_o1,
    stride_o2,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + seq_idx)

    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)
    group_head_offset = (
        tl.arange(0, QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, :]
    )
    if num_partitions == 1:
        tmp_out_offset = (
            seq_idx * NUM_KV_HEADS + kv_head_idx
        ) * max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE + group_head_offset
        tmp_out = tl.load(tmp_out_ptr + tmp_out_offset)

        out_offset = (
            seq_idx * stride_o0
            + kv_head_idx * QUERY_GROUP_SIZE * stride_o1
            + group_head_offset * stride_o2
        )
        tl.store(out_ptr + out_offset, tmp_out)
        return

    # Get the global max logit.
    ml_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE
        + tl.arange(0, NUM_PARTITIONS)[:, None] * QUERY_GROUP_SIZE
        + tl.arange(0, QUERY_GROUP_SIZE)[None, :]
    )

    mask = tl.arange(0, NUM_PARTITIONS)[:, None] < num_partitions
    # m_i: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    m_i = tl.load(m_i_ptr + ml_offset, mask=mask, other=float("-inf"))
    # m: [QUERY_GROUP_SIZE]
    m = tl.max(m_i, axis=0)

    # Rescale the exp sums and compute the global sum.
    # l_i: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    l_i = tl.load(l_i_ptr + ml_offset, mask=mask, other=0.0)
    l_i *= tl.exp(m_i - m[None, :])
    # l: [QUERY_GROUP_SIZE]
    log_sum = tl.sum(l_i, axis=0)
    # r: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    r = l_i / log_sum[None, :]
    r = tl.reshape(r, (NUM_PARTITIONS, QUERY_GROUP_SIZE, 1))

    tmp_out_offset = (
        (seq_idx * NUM_KV_HEADS + kv_head_idx)
        * max_num_partitions
        * QUERY_GROUP_SIZE
        * HEAD_SIZE
        + tl.arange(0, NUM_PARTITIONS)[:, None, None] * QUERY_GROUP_SIZE * HEAD_SIZE
        + tl.arange(0, QUERY_GROUP_SIZE)[None, :, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, None, :]
    )
    # tmp_out: [NUM_PARTITIONS, QUERY_GROUP_SIZE, HEAD_SIZE]
    tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=mask[:, :, None], other=0.0)
    # out: [QUERY_GROUP_SIZE, HEAD_SIZE]
    out = tl.sum((tmp_out * r).to(tl.float32), axis=0)

    out_offset = (
        seq_idx * stride_o0
        + kv_head_idx * QUERY_GROUP_SIZE * stride_o1
        + group_head_offset * stride_o2
    )
    tl.store(out_ptr + out_offset, out)
