import math

import torch
import triton
import triton.language as tl


def get_fwd_config(B, H, M, N, D, causal):
    return (32, 32, 1, 4)


def total_attention(q, k, log_sum, causal=False, sm_scale=None):
    Dq, Dk = q.shape[-1], k.shape[-1]
    assert Dq == Dk
    assert Dk in {16, 32, 64, 128}
    # assert L is contiguous

    B, H, M, D = q.shape
    N = k.shape[2]
    Hk = k.shape[1]
    assert H % Hk == 0, "number of heads in q must be a multiple of that in k"
    num_groups = H // Hk

    P_SEQ = N - M

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # to work around https://github.com/openai/triton/issues/2441
    device = torch.cuda.device_of(q)
    with torch.cuda.device(device):
        config = get_fwd_config(B, H, M, N, D, causal)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config

        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0
        # consider using 3d grid to avoid div & rem
        grid = (triton.cdiv(N, BLOCK_N), H, B)
        tot_attn = torch.empty((B, H, N), dtype=torch.float32, device=q.device)
        _total_attention_kernel[grid](
            q,
            k,
            log_sum,
            tot_attn,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            B,
            H,
            M,
            N,
            P_SEQ,
            num_groups,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=D,
            BLOCK_N=BLOCK_N,
            CAUSAL=causal,
            DIVISIBLE_M=divisible_m,
            DIVISIBLE_N=divisible_n,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    return tot_attn


@triton.jit
def _total_attention_kernel(
    Q,
    K,
    L,
    TA,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    Z,
    H,
    M,
    N,
    P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    # -- grid id --
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    L += (off_z * H + off_h) * M
    TA += (off_z * H + off_h) * N  # (b, h, n)

    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (
        offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk
    )  # (BLOCK_M, BLOCK_DMODEL)
    k_ptrs = K + (
        offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    )  # (BLOCK_N, BLOCK_DMODEL)
    ta_ptrs = TA + offs_n  # (BLOCK_N, )

    # k and v stay in SRAM throughout
    if DIVISIBLE_N:
        k = tl.load(k_ptrs)
    else:
        mask_n = offs_n < N
        k = tl.load(k_ptrs, mask=mask_n[:, None])

    # initialize total attention
    tot_attn = tl.zeros([BLOCK_N], dtype=tl.float32)

    # loop over a col
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = (P_SEQ + offs_m[:, None]) >= (
            offs_n[None, :]
        )  # (BLOCK_M, BLOCK_N)

        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
        else:
            mask_m = offs_m < M
            valid_mask = mask_m[:, None]  # & mask_n
            q = tl.load(q_ptrs, mask=mask_m[:, None])
        # recompute p = softmax(qk * sm_scale, dim=-1)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # s = tl.where(valid_mask, s , float("-inf"))
        # if CAUSAL:
        #     s = tl.where(causal_mask, s, float("-inf"))

        # -- recompute p ---
        if DIVISIBLE_M:
            log_sum = tl.load(L + offs_m)
        else:
            log_sum = tl.load(L + offs_m, mask=mask_m)
        p = tl.math.exp2(s * qk_scale - log_sum[:, None] * log2e)  # (BLOCK_M, BLOCK_N)

        if not DIVISIBLE_M:
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        tot_attn += tl.sum(p, 0)
        # increment pointers
        q_ptrs += BLOCK_M * stride_qm

    if DIVISIBLE_N:
        tl.store(ta_ptrs, tot_attn)  # (BLOCK_N,)
    else:
        tl.store(ta_ptrs, tot_attn, mask=mask_n)  # (BLOCK_N, )
