import math

import torch
import triton
import triton.language as tl

"""
This file implements flash decoding, flash attention with split_kv, which exposes another
dimension of parallelism when batch_size * num_heads * blocks_along_seqlen_q cannot saturate
the gpu's SM's.

For more details, refer to https://princeton-nlp.github.io/flash-decoding/.
"""


@triton.jit
def _fwd_split_kv_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    O,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_os,
    stride_om,
    stride_ok,
    Z,
    H,
    M,
    N,
    P_SEQ,
    N_SPLIT_SIZE,
    S,
    num_groups,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    n_split_id = tl.program_id(1)
    off_zh = tl.program_id(2)
    off_h = off_zh % H
    off_z = off_zh // H
    off_hk = off_h // num_groups

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch & head)
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh

    # offset pointers for (batch & head, split)
    O += (
        off_z * stride_oz + off_h * stride_oh + n_split_id * stride_os
    )  # o's shape is (B, H, S, M, D)
    L += ((off_z * H + off_h) * S + n_split_id) * M  # l's shape is (B, H, S, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (
        offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    )  # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (
        offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    )  # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None])

    # Dot I trick: to place q in registers, it saves shared memory
    if BLOCK_DMODEL < 128:
        identity = tl.where(
            offs_k[:, None] == offs_k,
            tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
            tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype),
        )
        q = tl.dot(q, identity).to(input_dtype)
    # else:
    #     I = tl.where(offs_m_base[:, None] == offs_m_base,
    #                  tl.full((BLOCK_M, BLOCK_M), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_M, BLOCK_M), 0.0, dtype=input_dtype))
    #     q = tl.dot(I, q).to(input_dtype)

    # NOTE: Loop-Bound-For-N
    # The indices in m-dimension that this block may access is in `[start_m * BLOCK_M, (start_m + 1) * BLOCK_M)`.
    # According to the rule of causal masking, then max index in n-dimension that this block may access
    # is `P_SEQ + (start_m + 1) * BLOCK_M`.
    # However, the upper bound of index in n-dimension should never exceed the sequence length of k/v(`P_SEQ + N_CTX`).
    # `P_SEQ + (start_m + 1) * BLOCK_M` may be larger than `N`.
    # At this case, there would be illegal memory access when loading k & v tiles
    # if mask_n is not applied for loading(only when `DIVISIBLE_N`` is true).
    # See also https://github.com/FlagOpen/FlagAttention/pull/8
    N_LEFT = n_split_id * N_SPLIT_SIZE
    N_RIGHT = tl.minimum(N_LEFT + N_SPLIT_SIZE, N)
    if IS_CAUSAL:
        hi = tl.minimum(N_RIGHT, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(N_LEFT, hi)
    else:
        hi = N_RIGHT

    # loop over k, v and update accumulators
    offs_n_init = N_LEFT + offs_n_base
    k_ptrs = K + (
        offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn
    )  # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (
        offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk
    )  # (BLOCK_N, BLOCK_DMODEL)
    for start_n in range(N_LEFT, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # write back l & o
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        log_sum = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        log_sum = m_i * sm_scale + tl.log(l_i)  # log(normalizer)

    if DIVISIBLE_M:
        tl.store(l_ptrs, log_sum, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        tl.store(l_ptrs, log_sum, mask=mask_m, cache_modifier=".cg")
        tl.store(
            o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg"
        )


@triton.jit
def _fwd_combine_kv_splits(
    multiple_o,
    multiple_l,
    final_o,
    final_l,
    stride_mul_oz,
    stride_mul_oh,
    stride_mul_os,
    stride_mul_om,
    stride_mul_ok,
    stride_fin_oz,
    stride_fin_oh,
    stride_fin_om,
    stride_fin_ok,
    Z,
    H,
    M,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_h = tl.program_id(1)
    offs_z = tl.program_id(2)

    # offset
    multiple_o += offs_z * stride_mul_oz + offs_h * stride_mul_oh  # (B, H, S, M, D)
    multiple_l += (offs_z * H + offs_h) * S * M  # (B, H, S, M)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not DIVISIBLE_M:
        mask_m = offs_m < M

    # 1st loop: online logsumexp to save a swipe
    m = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    acc = tl.full([BLOCK_M], value=float(0.0), dtype=tl.float32)
    l_ptrs = multiple_l + offs_m
    for _ in range(0, S):
        if DIVISIBLE_M:
            log_sum = tl.load(l_ptrs)
        else:
            log_sum = tl.load(l_ptrs, mask=mask_m)
        m_new = tl.maximum(m, log_sum)
        acc = acc * tl.exp(m - m_new) + tl.exp(log_sum - m_new)
        m = m_new
        l_ptrs += M
    l_acc = m + tl.log(acc)

    # 2rd loop to rescale and accumulate o
    o_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_ptrs = multiple_l + offs_m
    offs_k = tl.arange(0, BLOCK_DMODEL)
    o_ptrs = (
        multiple_o + offs_m[:, None] * stride_mul_om + offs_k[None, :] * stride_mul_ok
    )
    for _ in range(0, S):
        log_sum = tl.load(l_ptrs, mask=offs_m < M)
        rescale = tl.exp(log_sum - l_acc)
        if DIVISIBLE_M:
            o = tl.load(
                o_ptrs,
            )
        else:
            o = tl.load(o_ptrs, mask=mask_m[:, None])
        o_acc += o * rescale[:, None]

        l_ptrs += M
        o_ptrs += stride_mul_os

    # write back
    final_o += offs_z * stride_fin_oz + offs_h * stride_fin_oh
    final_l += (offs_z * H + offs_h) * M
    a_ptrs = final_o + offs_m[:, None] * stride_fin_om + offs_k * stride_fin_ok
    b_ptrs = final_l + offs_m

    if DIVISIBLE_M:
        tl.store(a_ptrs, o_acc)
        tl.store(b_ptrs, l_acc)
    else:
        tl.store(a_ptrs, o_acc, mask=mask_m[:, None])
        tl.store(b_ptrs, l_acc, mask=mask_m)


def get_fwd_config(B, H, M, N, D, causal):
    # BLOCK_M, BLOCK_N, num_stages, num_warps
    return (16, 128, 1, 4)


# this function is adapted from:
# https://github.com/Dao-AILab/flash-attention/blob/61a777247900f6c2a37376f3ffd7134385fdc95c/csrc/flash_attn/flash_api.cpp#L235
def num_splits_herustic(B, H, M, N, BLOCK_M, BLOCK_N, num_sms, max_splits):
    num_blocks_without_split_kv = B * H * triton.cdiv(M, BLOCK_M)
    if num_blocks_without_split_kv >= 0.8 * num_sms:
        return 1

    num_n_blocks = triton.cdiv(N, BLOCK_N)

    def num_split_avaiable(s):
        blocks_per_split = triton.cdiv(num_n_blocks, s)
        return s == 1 or (blocks_per_split * s - num_n_blocks < blocks_per_split)

    def efficiency(s):
        n_waves = (num_blocks_without_split_kv * s) / num_sms
        eff = n_waves / math.ceil(n_waves)
        return eff

    max_efficiency = 0.0
    plans = []  # (num_split, efficiency)
    max_splits = min(num_sms, num_n_blocks, max_splits)

    for num_split in range(1, max_splits + 1):
        if num_split_avaiable(num_split):
            eff = efficiency(num_split)
            plans.append((num_split, eff))
            max_efficiency = max(eff, max_efficiency)

    for num_split, eff in plans:
        if eff >= 0.85 * max_efficiency:
            return num_split
    return 1


# flash decoding
def attention(q, k, v, causal=False, sm_scale=None):
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv
    assert Dk in {16, 32, 64, 128}

    B, H, M, D = q.shape
    N = k.shape[2]
    Hk, Hv = k.shape[1], v.shape[1]
    assert Hk == Hv, "num of heads in k and v should be equal"
    assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
    num_groups = H // Hk
    P_SEQ = N - M
    larger_m = M > N

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # to work around https://github.com/openai/triton/issues/2441
    device = torch.cuda.device_of(q)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    with torch.cuda.device(device):
        config = get_fwd_config(B, H, M, N, D, causal)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config
        S = num_splits_herustic(B, H, M, N, BLOCK_M, BLOCK_N, num_sms, 128)

        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0

        # consider using 3d grid to avoid div & rem
        multiple_l = torch.empty((B, H, S, M), dtype=torch.float32, device="cuda")
        multiple_o = torch.empty((B, H, S, M, D), dtype=torch.float16, device="cuda")
        grid = (triton.cdiv(M, BLOCK_M), S, H * B)
        N_SPLIT_SIZE = triton.cdiv(triton.cdiv(N, BLOCK_N), S) * BLOCK_N
        _fwd_split_kv_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            multiple_l,
            multiple_o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            multiple_o.stride(0),
            multiple_o.stride(1),
            multiple_o.stride(2),
            multiple_o.stride(3),
            multiple_o.stride(4),
            B,
            H,
            M,
            N,
            P_SEQ,
            N_SPLIT_SIZE,
            S,
            num_groups,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=D,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal,
            LARGER_M=larger_m,
            DIVISIBLE_M=divisible_m,
            DIVISIBLE_N=divisible_n,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        if S == 1:
            return multiple_o.squeeze(2)

        final_l = torch.empty((B, H, M), dtype=torch.float32, device="cuda")
        final_o = torch.empty_like(q)
        grid = (triton.cdiv(M, BLOCK_M), H, B)
        _fwd_combine_kv_splits[grid](
            multiple_o,
            multiple_l,
            final_o,
            final_l,
            multiple_o.stride(0),
            multiple_o.stride(1),
            multiple_o.stride(2),
            multiple_o.stride(3),
            multiple_o.stride(4),
            final_o.stride(0),
            final_o.stride(1),
            final_o.stride(2),
            final_o.stride(3),
            B,
            H,
            M,
            S,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=D,
            DIVISIBLE_M=divisible_m,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    return final_o
