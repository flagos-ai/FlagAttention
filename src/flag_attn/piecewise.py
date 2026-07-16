"""
Piecewise Attention
====================

This is a extension to Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf) that performs piecewise computation
of attention scores(The scores to which softmax is applied). This design originates from
the need to make better predictions when the predicted sequence is longer than sequences
in the training set.

It takes as input two q's and two k's as inputs. The attention score is the dot product
of (q1, k1) or (q2, k2) depending on whether the distance between q & k exceeds a threshold.

The code is adapted from triton's tutorial:
https://github.com/openai/triton/blob/5162871c6cae01a8508a309cf21a8e6b68a4c091/python/tutorials/06-fused-attention.py
"""

import math

import torch
import triton
import triton.language as tl

__all__ = ["attention"]


# --------------------------- public API ---------------------------
class PiecewiseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q1, k1, q2, k2, v, w, causal, sm_scale):
        # shape constraints
        Dq1, Dk1, Dq2, Dk2, Dv = (
            q1.shape[-1],
            k1.shape[-1],
            q2.shape[-1],
            k2.shape[-1],
            v.shape[-1],
        )
        assert Dq1 == Dk1 == Dq2 == Dk2 == Dv
        assert Dk1 in {16, 32, 64, 128}

        B, H, M, D = q1.shape
        N = k1.shape[2]
        P_SEQ = N - M
        larger_m = M > N

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q1)
        with torch.cuda.device(device):
            config = get_fwd_config(B, H, M, N, D, causal)
            BLOCK_M, BLOCK_N, num_stages, num_warps = config

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0

            grid = (triton.cdiv(M, BLOCK_M), H, B)
            o = torch.empty_like(q1)
            L = torch.empty((B, H, M), device=q1.device, dtype=torch.float32)

            _fwd_kernel[grid](
                q1,
                k1,
                q2,
                k2,
                v,
                sm_scale,
                L,
                o,
                q1.stride(0),
                q1.stride(1),
                q1.stride(2),
                q1.stride(3),
                k1.stride(0),
                k1.stride(1),
                k1.stride(2),
                k1.stride(3),
                q2.stride(0),
                q2.stride(1),
                q2.stride(2),
                q2.stride(3),
                k2.stride(0),
                k2.stride(1),
                k2.stride(2),
                k2.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                B,
                H,
                M,
                N,
                P_SEQ,
                w=w,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=D,
                IS_CAUSAL=causal,
                LARGER_M=larger_m,
                DIVISIBLE_M=divisible_m,
                DIVISIBLE_N=divisible_n,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        ctx.save_for_backward(q1, k1, q2, k2, v, o, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.w = w
        return o

    @staticmethod
    def backward(ctx, do):
        q1, k1, q2, k2, v, o, L = ctx.saved_tensors
        w = ctx.w
        causal = ctx.causal
        sm_scale = ctx.sm_scale

        B, H, M, D = q1.shape
        N = k1.shape[2]
        P_SEQ = N - M
        larger_m = M > N

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q1)
        with torch.cuda.device(device):
            config = get_bwd_config(B, H, M, N, D, causal)
            BLOCK_M, BLOCK_N, num_stages, num_warps = config

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0

            delta = torch.empty((B, H, M), device=q1.device, dtype=torch.float32)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_preprocess[grid](
                o,
                do,
                delta,
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                delta.stride(0),
                delta.stride(1),
                delta.stride(2),
                M,
                BLOCK_M=BLOCK_M,
                D_HEAD=D,
                DIVISIBLE_M=divisible_m,
            )

            dk1 = torch.empty_like(k1)
            dk2 = torch.empty_like(k2)
            dv = torch.empty_like(v)
            grid = (triton.cdiv(N, BLOCK_N), H, B)
            _bwd_kv_kernel[grid](
                q1,
                k1,
                q2,
                k2,
                v,
                sm_scale,
                do,
                dk1,
                dk2,
                dv,
                L,
                delta,
                q1.stride(0),
                q1.stride(1),
                q1.stride(2),
                q1.stride(3),
                k1.stride(0),
                k1.stride(1),
                k1.stride(2),
                k1.stride(3),
                q2.stride(0),
                q2.stride(1),
                q2.stride(2),
                q2.stride(3),
                k2.stride(0),
                k2.stride(1),
                k2.stride(2),
                k2.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                dk1.stride(0),
                dk1.stride(1),
                dk1.stride(2),
                dk1.stride(3),
                dk2.stride(0),
                dk2.stride(1),
                dk2.stride(2),
                dk2.stride(3),
                dv.stride(0),
                dv.stride(1),
                dv.stride(2),
                dv.stride(3),
                B,
                H,
                M,
                N,
                P_SEQ,
                w=w,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=D,
                BLOCK_N=BLOCK_N,
                CAUSAL=causal,
                DIVISIBLE_M=divisible_m,
                DIVISIBLE_N=divisible_n,
                num_stages=num_stages,
                num_warps=num_warps,
            )

            dq1 = torch.zeros_like(q1)
            dq2 = torch.zeros_like(q2)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_q_kernel[grid](
                q1,
                k1,
                q2,
                k2,
                v,
                sm_scale,
                do,
                dq1,
                dq2,
                L,
                delta,
                q1.stride(0),
                q1.stride(1),
                q1.stride(2),
                q1.stride(3),
                k1.stride(0),
                k1.stride(1),
                k1.stride(2),
                k1.stride(3),
                q2.stride(0),
                q2.stride(1),
                q2.stride(2),
                q2.stride(3),
                k2.stride(0),
                k2.stride(1),
                k2.stride(2),
                k2.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                dq1.stride(0),
                dq1.stride(1),
                dq1.stride(2),
                dq1.stride(3),
                dq2.stride(0),
                dq2.stride(1),
                dq2.stride(2),
                dq2.stride(3),
                B,
                H,
                M,
                N,
                P_SEQ,
                w=w,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=D,
                BLOCK_N=BLOCK_N,
                CAUSAL=causal,
                LARGER_M=larger_m,
                DIVISIBLE_M=divisible_m,
                DIVISIBLE_N=divisible_n,
                num_stages=num_stages,
                num_warps=num_warps,
            )

        return dq1, dk1, dq2, dk2, dv, None, None, None


def attention(q1, k1, q2, k2, v, dist_threshold, causal=False, sm_scale=None):
    """
    PiecewiseAttention

    Piecewise deviates from standard scaled dot product attention in that takes
    as inputs two q's and two k's as inputs. The attention score is dot product
    of (q1, k1) or (q2, k2) depending on whether the distance between q & k
    exceeds a threshold.

    Arguments:
        q1(torch.Tensor): The first queries. The shape is (batch_size, nheads, seqlen_q, headdim).
        k1(torch.Tensor): The first keys. The shape is (batch_size, nheads, seqlen_k, headdim).
        q2(torch.Tensor): The second queries. The shape is (batch_size, nheads, seqlen_q, headdim).
        k2(torch.Tensor): The second keys. The shape is (batch_size, nheads, seqlen_k, headdim).
        v(torch.Tensor): The values. The shape is (batch_size, nheads, seqlen_k, headdim).
        dist_threshold(int): The threshold of distance between q and k. When the distance
            is not greater than w, the attention score is dot(q1, k1); otherwise dot(q2, k2).
        causal(bool): Whether causal masking is applied to attention scores before applying softmax.
        sm_scale(float): The scaling of attention scores before applying softmax.

    Returns:
        out: (torch.Tensor): The output. The shape is (batch_size, nheads, seqlen_q, headdim).
    """
    return PiecewiseAttention.apply(q1, k1, q2, k2, v, dist_threshold, causal, sm_scale)


# --------------------------- Forward ---------------------------
def get_fwd_config(B, H, M, N, D, causal):
    # A100
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 8
    # RTX-3090, ...
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 8
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return BLOCK_M, BLOCK_N, num_stages, num_warps


@triton.jit
def _fwd_kernel(
    Q1,
    K1,
    Q2,
    K2,
    V,
    sm_scale,
    L,
    O,
    stride_q1z,
    stride_q1h,
    stride_q1m,
    stride_q1k,
    stride_k1z,
    stride_k1h,
    stride_k1n,
    stride_k1k,
    stride_q2z,
    stride_q2h,
    stride_q2m,
    stride_q2k,
    stride_k2z,
    stride_k2h,
    stride_k2n,
    stride_k2k,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    M,
    N,
    P_SEQ,
    w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q1.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M  # L: shape(B, H, N_CTX), C-contiguous

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to v alue-like data
    q1_ptrs = Q1 + (
        offs_m[:, None] * stride_q1m + offs_k[None, :] * stride_q1k
    )  # (BLOCK_M, BLOCK_DMODEL)
    q2_ptrs = Q2 + (
        offs_m[:, None] * stride_q2m + offs_k[None, :] * stride_q2k
    )  # (BLOCK_M, BLOCK_DMODEL)
    k1_ptrs = K1 + (
        offs_n_init[:, None] * stride_k1n + offs_k[None, :] * stride_k1k
    )  # (BLOCK_N, BLOCK_DMODEL)
    k2_ptrs = K2 + (
        offs_n_init[:, None] * stride_k2n + offs_k[None, :] * stride_k2k
    )  # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (
        offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk
    )  # (BLOCK_N, BLOCK_DMODEL)
    o_ptrs = O + (
        offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    )  # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q: it will stay in SRAM throughout
    if DIVISIBLE_M:
        q1 = tl.load(q1_ptrs)
        q2 = tl.load(q2_ptrs)
    else:
        mask_m = offs_m < M
        q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
        q2 = tl.load(q2_ptrs, mask=mask_m[:, None])

    # Dot I trick: it converts q1, q2 into mma layout and saves shared memory
    # better way to generate a eye matrix. avoid casting from bool
    identity = tl.where(
        offs_k[:, None] == offs_k,
        tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
        tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype),
    )
    q1 = tl.dot(q1, identity).to(input_dtype)
    q2 = tl.dot(q2, identity).to(input_dtype)

    # loop over k, v and update accumulator
    # see note "Loop-Bound-For-N"
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        # -- offsets & masking --
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        piecewise_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :] + w)

        # -- load k, v --
        if DIVISIBLE_N:
            k1 = tl.load(k1_ptrs)
            k2 = tl.load(k2_ptrs)
            v = tl.load(v_ptrs)
        else:
            mask_n = offs_n < N
            k1 = tl.load(k1_ptrs, mask=mask_n[:, None])
            k2 = tl.load(k2_ptrs, mask=mask_n[:, None])
            v = tl.load(v_ptrs, mask=mask_n[:, None])

        # -- compute s = qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # TODO: more careful masking
        s += tl.where(
            piecewise_mask, tl.dot(q2, tl.trans(k2)), tl.dot(q1, tl.trans(k1))
        )
        if not DIVISIBLE_N:
            s = tl.where(mask_n, s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        # loop l2r, so no extra handling of inf is needed
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        # update pointers
        k1_ptrs += BLOCK_N * stride_k1n
        k2_ptrs += BLOCK_N * stride_k2n
        v_ptrs += BLOCK_N * stride_vn

    # write back l & o
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l_i = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l_i = m_i * sm_scale + tl.log(l_i)

    if DIVISIBLE_M:
        tl.store(l_ptrs, l_i)
        tl.store(o_ptrs, acc.to(input_dtype))
    else:
        tl.store(l_ptrs, l_i, mask=mask_m)
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None])


# --------------------------- Backward ---------------------------
def get_bwd_config(B, H, M, N, D, causal):
    # A100
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 2, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 64, 2, 4

        # BLOCK_M = 64 if D<=64 else 128
        # BLOCK_N = 64
        # num_stages = 1 if D<=64 else (2 if not causal else 1)
        # num_warps = 4 if D <=64 else 8
    # RTX-3090, ...
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 64, 2, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 64, 2, 8
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return BLOCK_M, BLOCK_N, num_stages, num_warps


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dz,
    stride_dh,
    stride_dm,
    M,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh

    # compute (Out * Dout).sum() for vector interpretation
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)

    # load
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok

    if DIVISIBLE_M:
        o = tl.load(o_ptrs).to(tl.float32)
        do = tl.load(do_ptrs).to(tl.float32)
    else:
        mask_m = off_m < M
        o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)

    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kv_kernel(
    Q1,
    K1,
    Q2,
    K2,
    V,
    sm_scale,
    DO,
    DK1,
    DK2,
    DV,
    L,
    D,
    stride_q1z,
    stride_q1h,
    stride_q1m,
    stride_q1k,
    stride_k1z,
    stride_k1h,
    stride_k1n,
    stride_k1k,
    stride_q2z,
    stride_q2h,
    stride_q2m,
    stride_q2k,
    stride_k2z,
    stride_k2h,
    stride_k2n,
    stride_k2k,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dk1z,
    stride_dk1h,
    stride_dk1n,
    stride_dk1k,
    stride_dk2z,
    stride_dk2h,
    stride_dk2n,
    stride_dk2k,
    stride_dvz,
    stride_dvh,
    stride_dvn,
    stride_dvk,
    Z,
    H,
    M,
    N,
    P_SEQ,
    w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q1.dtype.element_ty
    # -- grid id --
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    # offset pointers for batch/head
    DK1 += off_z * stride_dk1z + off_h * stride_dk1h
    DK2 += off_z * stride_dk2z + off_h * stride_dk2h
    DV += off_z * stride_dvz + off_h * stride_dvh

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
    q1_ptrs = Q1 + (
        offs_m_init[:, None] * stride_q1m + offs_k[None, :] * stride_q1k
    )  # (BLOCK_M, BLOCK_DMODEL)
    q2_ptrs = Q2 + (
        offs_m_init[:, None] * stride_q2m + offs_k[None, :] * stride_q2k
    )  # (BLOCK_M, BLOCK_DMODEL)
    k1_ptrs = K1 + (
        offs_k[:, None] * stride_k1k + offs_n[None, :] * stride_k1n
    )  # (BLOCK_DMODEL, BLOCK_N)
    k2_ptrs = K2 + (
        offs_k[:, None] * stride_k2k + offs_n[None, :] * stride_k2n
    )  # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (
        offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    )  # (BLOCK_N, BLOCK_DMODEL)
    do_ptrs = DO + (
        offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok
    )  # (BLOCK_M, BLOCK_DMODEL)

    dv_ptrs = DV + (
        offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk
    )  # (BLOCK_N, BLOCK_DMODEL)
    dk1_ptrs = DK1 + (
        offs_n[:, None] * stride_dk1n + offs_k[None, :] * stride_dk1k
    )  # (BLOCK_N, BLOCK_DMODEL)
    dk2_ptrs = DK2 + (
        offs_n[:, None] * stride_dk2n + offs_k[None, :] * stride_dk2k
    )  # (BLOCK_N, BLOCK_DMODEL)

    # k and v stay in SRAM throughout
    if DIVISIBLE_N:
        k1 = tl.load(k1_ptrs)
        k2 = tl.load(k2_ptrs)
        v = tl.load(v_ptrs)
    else:
        mask_n = offs_n < N
        k1 = tl.load(k1_ptrs, mask=mask_n[None, :])
        k2 = tl.load(k2_ptrs, mask=mask_n[None, :])
        v = tl.load(v_ptrs, mask=mask_n[:, None])

    # initialize dk amd dv
    dk1 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # loop over a column
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base

        # load q1, k1, q2, k2, v, do on-chip
        if DIVISIBLE_M:
            q1 = tl.load(q1_ptrs)
            q2 = tl.load(q2_ptrs)
            do = tl.load(do_ptrs)  # (BLOCK_M, BLOCK_DMODEL)
            delta = tl.load(D + offs_m)
            log_sum = tl.load(L + offs_m)
        else:
            mask_m = offs_m < M
            q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
            q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
            do = tl.load(do_ptrs, mask=mask_m[:, None])  # (BLOCK_M, BLOCK_DMODEL)
            delta = tl.load(D + offs_m, mask=mask_m)
            log_sum = tl.load(L + offs_m, mask=mask_m)

        # recompute p = softmax(qk, dim=-1).T
        piecewise_mask = (P_SEQ + offs_m[:, None]) >= (
            offs_n[None, :] + w
        )  # (BLOCK_M, BLOCK_N)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.where(piecewise_mask, tl.dot(q2, k2), tl.dot(q1, k1))

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # if CAUSAL:
        #     s = tl.where(causal_mask & valid_mask, s, float("-inf"))
        # else:
        #     s = tl.where(valid_mask, s, float("-inf"))

        # -- recompute p ---
        # l = tl.load(L + offs_m, mask=mask_m)
        p = tl.math.exp2(s * qk_scale - log_sum[:, None] * log2e)  # (BLOCK_M, BLOCK_N)
        if not DIVISIBLE_M:
            valid_mask = mask_m[:, None]  # & mask_n
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (
                offs_n[None, :]
            )  # (BLOCK_M, BLOCK_N)
            p = tl.where(causal_mask, p, 0.0)

        # compute dv = dot(p, do)
        # do = tl.load(do_ptrs, mask=mask_m[:, None]) # (BLOCK_M, BLOCK_DMODEL)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)  # (BLOCK_N, BLOCK_DMODEL)

        # compute dp = dot(v, do)
        # delta = tl.load(D + offs_m, mask=mask_m)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))
        # no need to mask dp
        # if CAUSAL:
        #     dp = tl.where(causal_mask & valid_mask, dp, 0.0)
        # else:
        #     dp = tl.where(valid_mask, dp, 0.0)

        # compute ds = p * (dp - delta[:, None])
        # move scale out to dk at last
        ds = p * (dp - delta[:, None])  # (BLOCK_M, BLOCK_N)

        # mask ds To ensure no small values
        if not DIVISIBLE_M:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        ds2 = tl.where(piecewise_mask, ds, 0.0).to(input_dtype)
        ds1 = tl.where(piecewise_mask, 0.0, ds).to(input_dtype)

        # compute dk = dot(ds.T, q) masking
        dk1 += tl.dot(tl.trans(ds1), q1)
        dk2 += tl.dot(tl.trans(ds2), q2)

        # increment pointers
        q1_ptrs += BLOCK_M * stride_q1m
        q2_ptrs += BLOCK_M * stride_q2m
        do_ptrs += BLOCK_M * stride_dom

    dk1 *= sm_scale
    dk2 *= sm_scale

    if DIVISIBLE_N:
        tl.store(dk1_ptrs, dk1.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dk2_ptrs, dk2.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL)
    else:
        tl.store(
            dk1_ptrs, dk1.to(input_dtype), mask=mask_n[:, None]
        )  # (BLOCK_N, BLOCK_DMODEL)
        tl.store(
            dk2_ptrs, dk2.to(input_dtype), mask=mask_n[:, None]
        )  # (BLOCK_N, BLOCK_DMODEL)
        tl.store(
            dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None]
        )  # (BLOCK_N, BLOCK_DMODEL)


@triton.jit
def _bwd_q_kernel(
    Q1,
    K1,
    Q2,
    K2,
    V,
    sm_scale,
    DO,
    DQ1,
    DQ2,
    L,
    D,
    stride_q1z,
    stride_q1h,
    stride_q1m,
    stride_q1k,
    stride_k1z,
    stride_k1h,
    stride_k1n,
    stride_k1k,
    stride_q2z,
    stride_q2h,
    stride_q2m,
    stride_q2k,
    stride_k2z,
    stride_k2h,
    stride_k2n,
    stride_k2k,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dq1z,
    stride_dq1h,
    stride_dq1m,
    stride_dq1k,
    stride_dq2z,
    stride_dq2h,
    stride_dq2m,
    stride_dq2k,
    Z,
    H,
    M,
    N,
    P_SEQ,
    w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q1.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    # offset pointers for batch/head
    DQ1 += off_z * stride_dq1z + off_h * stride_dq1h
    DQ2 += off_z * stride_dq2z + off_h * stride_dq2h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q1_ptrs = Q1 + (
        offs_m[:, None] * stride_q1m + offs_k[None, :] * stride_q1k
    )  # (BLOCK_M, BLOCK_DMODEL)
    q2_ptrs = Q2 + (
        offs_m[:, None] * stride_q2m + offs_k[None, :] * stride_q2k
    )  # (BLOCK_M, BLOCK_DMODEL)
    k1_ptrs = K1 + (
        offs_n_init[:, None] * stride_k1n + offs_k[None, :] * stride_k1k
    )  # (BLOCK_N, BLOCK_DMODEL)
    k2_ptrs = K2 + (
        offs_n_init[:, None] * stride_k2n + offs_k[None, :] * stride_k2k
    )  # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (
        offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk
    )  # (BLOCK_N, BLOCK_DMODEL)

    dq1_ptrs = DQ1 + (
        offs_m[:, None] * stride_dq1m + offs_k[None, :] * stride_dq1k
    )  # (BLOCK_M, BLOCK_DMODEL)
    dq2_ptrs = DQ2 + (
        offs_m[:, None] * stride_dq2m + offs_k[None, :] * stride_dq2k
    )  # (BLOCK_M, BLOCK_DMODEL)
    do_ptrs = DO + (
        offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    )  # (BLOCK_M, BLOCK_DMODEL)

    # pointer to row-wise quantities in value-like data
    d_ptrs = D + offs_m
    l_ptrs = L + offs_m

    # load q: it will stay in SRAM throughout
    if DIVISIBLE_M:
        q1 = tl.load(q1_ptrs)
        q2 = tl.load(q2_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        log_sum = tl.load(l_ptrs)
    else:
        mask_m = offs_m < M
        q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
        q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        log_sum = tl.load(l_ptrs, mask=mask_m)

    # initialize dq
    dq1 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dq2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # loop over k, v and update accumulator
    # see note "Loop-Bound-For-N"
    if CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    # loop over a row
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_base

        # load k1, k2, v on chip
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k1 = tl.load(k1_ptrs)
            k2 = tl.load(k2_ptrs)
        else:
            mask_n = offs_n < N
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k1 = tl.load(k1_ptrs, mask=mask_n[:, None])
            k2 = tl.load(k2_ptrs, mask=mask_n[:, None])

        # recompute p = softmax(qk * sm_scale, dim=-1)
        piecewise_mask = (P_SEQ + offs_m[:, None]) >= (
            offs_n[None, :] + w
        )  # (BLOCK_M, BLOCK_N)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.where(
            piecewise_mask, tl.dot(q2, tl.trans(k2)), tl.dot(q1, tl.trans(k1))
        )
        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # if CAUSAL:
        #     s = tl.where(causal_mask & valid_mask, s, float("-inf"))
        # else:
        #     s = tl.where(valid_mask, s, float("-inf"))
        p = tl.math.exp2(s * qk_scale - log_sum[:, None] * log2e)  # (BLOCK_M, BLOCK_N)

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))
        # no need to mask dp
        # if CAUSAL:
        #     dp = tl.where(causal_mask & valid_mask, dp, 0.0)
        # else:
        #     dp = tl.where(valid_mask, dp, 0.0)

        # compute ds = p * (dp - delta[:, None])
        # move scale out to dq at last
        ds = p * (dp - delta[:, None])  # (BLOCK_M, BLOCK_N)

        # mask ds to ensure no small values
        if not DIVISIBLE_N:
            ds = tl.where(mask_n, ds, 0.0)
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (
                offs_n[None, :]
            )  # (BLOCK_M, BLOCK_N)
            ds = tl.where(causal_mask, ds, 0.0)

        ds2 = tl.where(piecewise_mask, ds, 0.0).to(input_dtype)
        ds1 = tl.where(piecewise_mask, 0.0, ds).to(input_dtype)

        dq1 += tl.dot(ds1, k1)
        dq2 += tl.dot(ds2, k2)

        # increment pointers
        k1_ptrs += BLOCK_N * stride_k1n
        k2_ptrs += BLOCK_N * stride_k2n
        v_ptrs += BLOCK_N * stride_vn

    dq1 *= sm_scale
    dq2 *= sm_scale
    if DIVISIBLE_M:
        tl.store(dq1_ptrs, dq1.to(input_dtype))
        tl.store(dq2_ptrs, dq2.to(input_dtype))
    else:
        tl.store(dq1_ptrs, dq1.to(input_dtype), mask=mask_m[:, None])
        tl.store(dq2_ptrs, dq2.to(input_dtype), mask=mask_m[:, None])
