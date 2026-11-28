# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA PDL decode kernels and SM-dependent launch policies."""

import torch
import triton
import triton.language as tl
from flag_attn.utils import current_platform
from flag_attn.minimax_sparse_attention.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    _FP8_DTYPES,
    _KV_SCALE_NONE,
    _KV_SCALE_SCALAR,
    _KV_SCALE_PER_TOKEN_HEAD,
    _kv_scale_args,
)

_MSA_DECODE_GRID_BAND_LO = 96


_MSA_DECODE_GRID_BAND_HI = 160


_MSA_DECODE_TARGET_GRID = 256


_MSA_DECODE_FUSED_MIN_PARALLELISM = 512


_SM_COUNT_CACHE: dict[int, int] = {}


_PDL_SUPPORTED: bool | None = None


def _pdl_supported() -> bool:
    """Whether this platform supports programmatic dependent launch.

    Measured at 4.7 us per call on this box, against a decode kernel of
    17-25 us at small batch -- so querying it per launch is a real cost, not a
    rounding error. The answer is fixed for the process.
    """
    global _PDL_SUPPORTED
    if _PDL_SUPPORTED is None:
        _PDL_SUPPORTED = current_platform.is_arch_support_pdl()
    return _PDL_SUPPORTED


def _sm_count(device) -> int:
    """SM count, cached per device.

    `torch.cuda.get_device_properties` is not free, and the decode wrapper runs
    it on every call while the decode kernel itself is only ~37 us at small
    batch -- so the uncached query showed up as a ~0.9x regression on this
    repo's small-batch benchmark shapes.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = _SM_COUNT_CACHE.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT_CACHE[index] = count
    return count


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # Valid block count from seq_len (no sentinel): min(topk, cdiv(kv_len, blk)).
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        k = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[None, :] * stride_kv_pos
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
            if KV_SCALE_MODE == 1:
                k = (k * tl.load(k_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                k_scale = tl.load(
                    k_scale_ptr
                    + pid_kh * stride_ks_h
                    + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                    mask=pos_mask,
                    other=1.0,
                )
                k = (k * k_scale[None, :]).to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[:, None] * stride_kv_pos
            + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
            if KV_SCALE_MODE == 1:
                v = (v * tl.load(v_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                v_scale = tl.load(
                    v_scale_ptr
                    + pid_kh * stride_vs_h
                    + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                    mask=pos_mask,
                    other=1.0,
                )
                v = (v * v_scale[:, None]).to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    # The lse store is unconditional even though the merge-skip path never
    # reads it back. Gating it on a constexpr looks like free dead-code
    # removal and is not: dropping it removes a register pressure point late
    # in the kernel, the scheduler then picks a more aggressive allocation
    # (168 -> 190 regs), and at 73732 B of smem per CTA that crosses the
    # occupancy cliff from 3 CTAs/SM to 2 (65536 / (190*4*32) = 2). Measured
    # at (b=512, s=3072) fp8: 291.4 us with the store, 330.4 us without --
    # 13% slower to skip ~5 KB of writes. An explicit maxnreg=168 cap only
    # recovers half of it (316.3 us) because the cap spills instead.
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    # NOTE: assume seq_lens is safe to load before gdc_wait()
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp2(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = (
        out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    )
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_fused_kernel(
    q_ptr,
    kv_cache_ptr,
    t_ptr,
    o_ptr,
    lse_ptr,
    block_table_ptr,
    seq_lens,
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
    SINGLE_CHUNK: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    # Bound by the constexpr BLOCK_SIZE_H rather than the runtime
    # gqa_group_size so Triton can prove the tile is fully in range and drop
    # the boundary masks (registers 213 -> 168).
    #
    # REQUIRES gqa_group_size == BLOCK_SIZE_H. Since
    # BLOCK_SIZE_H = max(16, next_pow2(gqa_group_size)), any gqa_group_size
    # below 16 makes BLOCK_SIZE_H larger than the number of heads that actually
    # exist, and these block pointers then run past this kv_head into the next
    # one -- measured max_diff 7.8e-02 at (nkv=8, gqa=8). The wrapper enforces
    # the equality before selecting this kernel; do not relax it.
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(BLOCK_SIZE_H, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        # One base address shared by the K and V loads.
        kv_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        k = tl.load(
            kv_base + off_n[None, :] * stride_kv_pos + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            kv_base
            + off_n[:, None] * stride_kv_pos
            + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(BLOCK_SIZE_H, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    # With a single chunk there is no merge to consume lse: the store is dead.
    if not SINGLE_CHUNK:
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
            shape=(BLOCK_SIZE_H,),
            strides=(stride_l_h,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_H,),
            order=(0,),
        )
        tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (output, output, 0, 0, 0, 0, _KV_SCALE_NONE)
    )
    use_pdl = _pdl_supported()
    pdl_launch = {"launch_pdl": True} if use_pdl else {}

    # split-K over the selected blocks; chunk count is shape-constant (cuda
    # graph). 3*SM pushes a narrow band down to chunks == 1, which activates the
    # merge-skip; outside it the fixed 256 target is better.
    #
    # The band was derived for 78 SM, where 3*SM == 234 reaches chunks == 1 for
    # P in 118..128. At 132 SM it cannot: 3*SM == 396 yields chunks == 2 there,
    # and for P in 129..198 that is strictly worse than the fixed 256 target,
    # which already gives 1. Measured on H100, the shipped band and no band at
    # all are indistinguishable at every default and user shape (0.0647 vs
    # 0.0647 ms at P=128) -- it is a dead configuration here.
    #
    # Reaching chunks == 1 at P == 128 needs a target in [128, 255], and the
    # payoff splits by dtype, because halving the grid also halves occupancy
    # (0.97 -> 0.48 waves/SM) and only bf16 is bandwidth-bound enough to come
    # out ahead. Measured at P=128, n=5, GQA 16:
    #            bf16                      fp8
    #   chunks=2 (shipped)  0.0647 ms      0.0593 ms
    #   chunks=1 (tg 255)   0.0626 ms      0.0740 ms
    #                       1.034x faster  0.801x slower
    # So the merge-skip is taken on the bf16 path only. P=64 and P=256 are
    # unaffected either way (1.000-1.001x), confirming this is scoped to the band.
    p = max(1, total_q * num_kv_heads)
    if _MSA_DECODE_GRID_BAND_LO <= p <= _MSA_DECODE_GRID_BAND_HI:
        target_grid = 3 * _sm_count(q.device)
    else:
        target_grid = _MSA_DECODE_TARGET_GRID
    if not use_fp8 and p <= _MSA_DECODE_TARGET_GRID:
        # bf16: if a target just under the fixed one would collapse the chunk
        # count to 1 while the selected target does not, take it -- that is the
        # merge-skip the band was meant to reach. On 78 SM the band already
        # gets there, so this changes nothing; it only fires where 3*SM
        # overshoots. max_topk is in the min() above, so compare post-clamp.
        skip_grid = _MSA_DECODE_TARGET_GRID - 1
        if max(1, min(max_topk, skip_grid // p)) == 1 and (
            max(1, min(max_topk, target_grid // p)) > 1
        ):
            target_grid = skip_grid
    target = max(1, min(max_topk, target_grid // p))
    num_topk_chunks = 1 << (target.bit_length() - 1)

    # chunks == 1 makes the merge a pure copy (single-chunk softmax weights are
    # identically 1.0): allocate no partials and write final output directly.
    # The constexpr-bounded block pointers in the bf16 kernel are only in
    # range when the GQA group exactly fills BLOCK_SIZE_H; see the comment
    # there. Any other head configuration falls through to the general kernel.
    # Ordered so the cheap integer tests short-circuit before the
    # next_power_of_2 call: at small batch this wrapper's Python cost is
    # comparable to the kernel itself (~17 us), so it has to stay lean.
    # Cheap integer tests first; next_power_of_2 is a Python call and this
    # wrapper runs against a ~17 us kernel, so it must not be evaluated on the
    # fallback path.
    use_fused_decode = (
        (
            not use_fp8
            and p >= _MSA_DECODE_FUSED_MIN_PARALLELISM
            and gqa_group_size == max(16, triton.next_power_of_2(gqa_group_size))
        )
        if (not use_fp8 and p >= _MSA_DECODE_FUSED_MIN_PARALLELISM)
        else False
    )

    # Fast path: neither optimization applies when there are multiple chunks
    # (nothing to merge-skip) and the bf16 kernel is ineligible. The decode
    # kernel is only ~17 us at small batch, so this wrapper's own Python cost
    # is a measurable share of wall time -- taking the shortest route here is
    # worth more than any kernel tweak. Measured: without this, small-batch
    # shapes ran 0.89-0.95x of upstream purely on host overhead.
    # `single_chunk` decides where the kernel writes and whether a merge pass
    # runs. It is deliberately NOT folded into a constexpr of the shared
    # decode kernel: that forks the kernel so one shape resolves to different
    # compiled variants depending on the call site (measured as a systematic
    # 1.4-3.8% loss on the chunks>1 shapes), and the one dead store it would
    # let the compiler remove costs an occupancy tier (see that store).
    single_chunk = num_topk_chunks == 1
    if not single_chunk and not use_fused_decode:
        o_partial = torch.empty(
            num_topk_chunks,
            total_q,
            num_heads,
            head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        lse_partial = torch.empty(
            num_topk_chunks,
            total_q,
            num_heads,
            dtype=torch.float32,
            device=q.device,
        )
        _gqa_sparse_decode_kernel[(total_q * num_topk_chunks, num_kv_heads)](
            q,
            kv_cache,
            k_scale_arg,
            v_scale_arg,
            topk_idx,
            o_partial,
            lse_partial,
            block_table,
            seq_lens,
            total_q,
            gqa_group_size,
            head_dim,
            max_topk,
            sm_scale,
            decode_query_len,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            stride_ks_h,
            stride_ks_t,
            stride_vs_h,
            stride_vs_t,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_FP8=use_fp8,
            KV_SCALE_MODE=kv_scale_mode,
            USE_PDL=use_pdl,
            **pdl_launch,
        )
        _merge_topk_attn_out_kernel[(total_q, num_heads)](
            o_partial,
            lse_partial,
            output,
            head_dim,
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_PDL=use_pdl,
            **pdl_launch,
        )
        return

    if single_chunk:
        o_partial = output
        stride_o_c = 0
        stride_o_b, stride_o_h, stride_o_d = (
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )
        if use_fused_decode:
            # The fused kernel drops its lse store under SINGLE_CHUNK without
            # losing occupancy (its constexpr-bounded pointers already hold it
            # at 168 regs), so no buffer is needed.
            lse_partial = output  # unused; keeps the launch signature uniform
            stride_l_c = stride_l_b = stride_l_h = 0
        else:
            # The shared kernel stores lse unconditionally; see the note at
            # that store for why removing it costs more than the write.
            lse_partial = torch.empty(
                num_topk_chunks,
                total_q,
                num_heads,
                dtype=torch.float32,
                device=q.device,
            )
            stride_l_c, stride_l_b, stride_l_h = (
                lse_partial.stride(0),
                lse_partial.stride(1),
                lse_partial.stride(2),
            )
        pdl_launch = {}
    else:
        o_partial = torch.empty(
            num_topk_chunks,
            total_q,
            num_heads,
            head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        lse_partial = torch.empty(
            num_topk_chunks,
            total_q,
            num_heads,
            dtype=torch.float32,
            device=q.device,
        )
        stride_o_c, stride_o_b, stride_o_h, stride_o_d = (
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
        )
        stride_l_c, stride_l_b, stride_l_h = (
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
        )

    grid = (total_q * num_topk_chunks, num_kv_heads)
    # The bf16-specialized kernel bounds its Q/O block pointers by the
    # constexpr BLOCK_SIZE_H, which lets Triton drop the boundary masks and
    # brings registers 213 -> 168, i.e. the occupancy limiter from 2 to 3
    # blocks/SM. That is a large win where there are enough CTAs to use the
    # extra concurrency and a small LOSS where there are not: measured
    # 1.09-1.16x for P >= 512 but 0.97-1.00x at P = 128..256, where the chunk
    # count is already 1-2 and the shapes are latency- rather than
    # occupancy-limited. So it is scoped to the range where it was measured
    # to win, exactly like the TARGET_GRID band above.
    if use_fused_decode:
        _gqa_sparse_decode_fused_kernel[grid](
            q,
            kv_cache,
            topk_idx,
            o_partial,
            lse_partial,
            block_table,
            seq_lens,
            total_q,
            gqa_group_size,
            head_dim,
            max_topk,
            sm_scale,
            decode_query_len,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            stride_o_c,
            stride_o_b,
            stride_o_h,
            stride_o_d,
            stride_l_c,
            stride_l_b,
            stride_l_h,
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_PDL=use_pdl,
            SINGLE_CHUNK=single_chunk,
            **pdl_launch,
        )
    else:
        # Upstream's kernel verbatim for both chunk counts: it carries no
        # merge-skip constexpr, so a given shape always resolves to the same
        # compiled variant regardless of call site. The merge-skip is purely
        # a host-side decision (where o_partial points, and whether the merge
        # launches below).
        _gqa_sparse_decode_kernel[grid](
            q,
            kv_cache,
            k_scale_arg,
            v_scale_arg,
            topk_idx,
            o_partial,
            lse_partial,
            block_table,
            seq_lens,
            total_q,
            gqa_group_size,
            head_dim,
            max_topk,
            sm_scale,
            decode_query_len,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            stride_ks_h,
            stride_ks_t,
            stride_vs_h,
            stride_vs_t,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            stride_o_c,
            stride_o_b,
            stride_o_h,
            stride_o_d,
            stride_l_c,
            stride_l_b,
            stride_l_h,
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_FP8=use_fp8,
            KV_SCALE_MODE=kv_scale_mode,
            USE_PDL=use_pdl,
            **pdl_launch,
        )
    if not single_chunk:
        merge_grid = (total_q, num_heads)
        _merge_topk_attn_out_kernel[merge_grid](
            o_partial,
            lse_partial,
            output,
            head_dim,
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_PDL=use_pdl,
            **pdl_launch,
        )
