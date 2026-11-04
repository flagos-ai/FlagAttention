# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code derived from flash-linear-attention.
# The original source code is licensed under the MIT license.

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices


@triton.jit
def exp2(x):
    return tl.exp2(x)


@triton.jit
def _store_gated_block(
    AG,
    block,
    prefix,
    start,
    row: tl.constexpr,
    col: tl.constexpr,
    HV: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    T,
):
    ri = row * BC + tl.arange(0, BC)
    ci = col * BC + tl.arange(0, BC)
    gr = tl.gather(prefix, ri, 0)
    gc = tl.gather(prefix, ci, 0)
    lower = ri[:, None] >= ci[None, :]
    difference = tl.where(lower, gr[:, None] - gc[None, :], 0.0)
    gated = tl.where(lower, block * tl.exp(difference), 0.0)
    tl.store(AG + (start + ri[:, None]) * HV * BT + ci[None, :], gated, mask=(start + ri[:, None] < T))


@triton.heuristics(
    {
        "USE_G": lambda args: not args["OUTPUT_UNGATED"],
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BK": BK}, num_warps=num_warps) for BK in [32, 64] for num_warps in [1, 2, 4, 8]],
    key=["T", "H", "HV", "K", "BC", "OUTPUT_GATED", "OUTPUT_UNGATED", "MERGE_PRECISION", "FULL_CHUNKS"],
)
@triton.jit(do_not_specialize=["T"])
def _prepare_kernel(
    k,
    g,
    g_out,
    beta,
    A,
    AG,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    OUTPUT_GATED: tl.constexpr,
    OUTPUT_UNGATED: tl.constexpr,
    MERGE_PRECISION: tl.constexpr,
    FULL_CHUNKS: tl.constexpr,
):
    """Compute chunk gate prefixes and the requested triangular inverse in registers."""
    tl.static_assert(USE_G == (not OUTPUT_UNGATED))
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    gg = tl.arange(0, BT)
    gv = tl.load(g + (bos + i_t * BT + gg) * HV + i_h, mask=i_t * BT + gg < T, other=0).to(tl.float32)
    prefix = tl.cumsum(gv)
    tl.store(g_out + (bos + i_t * BT + gg) * HV + i_h, prefix, mask=i_t * BT + gg < T)

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_h // (HV // H)) * K

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    p_b0 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,), padding_option="zero").to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,), padding_option="zero").to(tl.float32)

    if USE_G:
        # Gated-only output applies the prefix diagonal before inversion.
        prefix_log2 = prefix * 1.4426950408889634
        b_g0 = tl.gather(prefix_log2, o_i, 0)
        b_g1 = tl.gather(prefix_log2, BC + o_i, 0)
        b_g2 = tl.gather(prefix_log2, 2 * BC + o_i, 0)
        b_g3 = tl.gather(prefix_log2, 3 * BC + o_i, 0)

    # Four 16-token diagonal blocks and six lower off-diagonal KKT blocks.

    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)

    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1), padding_option="zero")
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1), padding_option="zero")
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                b_k2 = tl.load(p_k2, boundary_check=(0, 1), padding_option="zero")
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1), padding_option="zero")
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    # Beta scales the rows of the strictly lower KKT matrix.

    # Padded KKT entries stay zero even when a gate factor overflows.
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    if USE_G:
        b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.0)
        b_A11 *= tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], exp2(b_g1[:, None] - b_g1[None, :]), 0.0)
        b_A22 *= tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], exp2(b_g2[:, None] - b_g2[None, :]), 0.0)
        b_A33 *= tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], exp2(b_g3[:, None] - b_g3[None, :]), 0.0)

        b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :], exp2(b_g1[:, None] - b_g0[None, :]), 0.0)
        b_A20 *= tl.where(m_tc2[:, None] & m_tc0[None, :], exp2(b_g2[:, None] - b_g0[None, :]), 0.0)
        b_A21 *= tl.where(m_tc2[:, None] & m_tc1[None, :], exp2(b_g2[:, None] - b_g1[None, :]), 0.0)
        b_A30 *= tl.where(m_tc3[:, None] & m_tc0[None, :], exp2(b_g3[:, None] - b_g0[None, :]), 0.0)
        b_A31 *= tl.where(m_tc3[:, None] & m_tc1[None, :], exp2(b_g3[:, None] - b_g1[None, :]), 0.0)
        b_A32 *= tl.where(m_tc3[:, None] & m_tc2[None, :], exp2(b_g3[:, None] - b_g2[None, :]), 0.0)
    else:
        b_A00 = tl.where(m_d, b_A00, 0.0)
        b_A11 = tl.where(m_d, b_A11, 0.0)
        b_A22 = tl.where(m_d, b_A22, 0.0)
        b_A33 = tl.where(m_d, b_A33, 0.0)

    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]

    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    # Extract substitution rows from registers instead of reading KKT from HBM.

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    if FULL_CHUNKS:
        # Interleave independent diagonal solves to hide reduction latency.
        for i in range(2, BC):
            b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
            b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
            b_a11 = tl.where(o_i < i, b_a11, 0.0)
            b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
            b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
            b_a22 = tl.where(o_i < i, b_a22, 0.0)
            b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
            b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
            b_a33 = tl.where(o_i < i, b_a33, 0.0)
            b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)
    else:
        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(2, min(BC, T - i_tc1)):
            b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
            b_a11 = tl.where(o_i < i, b_a11, 0.0)
            b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
        for i in range(2, min(BC, T - i_tc2)):
            b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
            b_a22 = tl.where(o_i < i, b_a22, 0.0)
            b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
        for i in range(2, min(BC, T - i_tc3)):
            b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
            b_a33 = tl.where(o_i < i, b_a33, 0.0)
            b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    # Merge the inverse blocks using the requested FP32 dot precision.

    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_A10, input_precision=MERGE_PRECISION), b_Ai00, input_precision=MERGE_PRECISION)
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_A21, input_precision=MERGE_PRECISION), b_Ai11, input_precision=MERGE_PRECISION)
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_A32, input_precision=MERGE_PRECISION), b_Ai22, input_precision=MERGE_PRECISION)

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=MERGE_PRECISION) + tl.dot(b_A21, b_Ai10, input_precision=MERGE_PRECISION),
        input_precision=MERGE_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=MERGE_PRECISION) + tl.dot(b_A32, b_Ai21, input_precision=MERGE_PRECISION),
        input_precision=MERGE_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=MERGE_PRECISION)
        + tl.dot(b_A31, b_Ai10, input_precision=MERGE_PRECISION)
        + tl.dot(b_A32, b_Ai20, input_precision=MERGE_PRECISION),
        input_precision=MERGE_PRECISION,
    )

    if USE_G:
        A = AG
    if OUTPUT_UNGATED or USE_G:
        A += (bos * HV + i_h) * BT
        p_A00 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
        p_A10 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        p_A11 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
        p_A20 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_A21 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        p_A22 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
        p_A30 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_A31 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_A32 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        p_A33 = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))

        tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), boundary_check=(0, 1))

    if OUTPUT_GATED and not USE_G:
        # Restore gate factors before rounding the FP32 inverse for storage.
        AG += (bos * HV + i_h) * BT
        _store_gated_block(AG, b_Ai00, prefix, i_tc0, 0, 0, HV, BT, BC, T)
        _store_gated_block(AG, b_Ai10, prefix, i_tc0, 1, 0, HV, BT, BC, T)
        _store_gated_block(AG, b_Ai11, prefix, i_tc0, 1, 1, HV, BT, BC, T)
        _store_gated_block(AG, b_Ai20, prefix, i_tc0, 2, 0, HV, BT, BC, T)
        _store_gated_block(AG, b_Ai21, prefix, i_tc0, 2, 1, HV, BT, BC, T)
        _store_gated_block(AG, b_Ai22, prefix, i_tc0, 2, 2, HV, BT, BC, T)
        _store_gated_block(AG, b_Ai30, prefix, i_tc0, 3, 0, HV, BT, BC, T)
        _store_gated_block(AG, b_Ai31, prefix, i_tc0, 3, 1, HV, BT, BC, T)
        _store_gated_block(AG, b_Ai32, prefix, i_tc0, 3, 2, HV, BT, BC, T)
        _store_gated_block(AG, b_Ai33, prefix, i_tc0, 3, 3, HV, BT, BC, T)

    zero = tl.full((BC, BC), 0, k.dtype.element_ty)
    for row in tl.static_range(0, 4):
        for col in tl.static_range(row + 1, 4):
            if OUTPUT_UNGATED or USE_G:
                upper = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_tc0 + row * BC, col * BC), (BC, BC), (1, 0))
                tl.store(upper, zero, boundary_check=(0, 1))
            if OUTPUT_GATED and not USE_G:
                upper_g = tl.make_block_ptr(AG, (T, BT), (HV * BT, 1), (i_tc0 + row * BC, col * BC), (BC, BC), (1, 0))
                tl.store(upper_g, zero, boundary_check=(0, 1))


def prepare_fused_tle_inputs(
    k, g, beta, output_gated=False, cu_seqlens=None, merge_precision="ieee", output_ungated=True
):
    """Shared chunk preparation; keep KKT and inverse subtiles in registers."""
    if merge_precision not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("unsupported inverse merge precision")
    if not (output_gated or output_ungated):
        raise ValueError("at least one inverse output is required")
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    gc = torch.empty_like(g, dtype=torch.float32)
    a = torch.zeros((B, T, H, 64), device=k.device, dtype=k.dtype) if output_ungated else None
    ag = torch.zeros((B, T, H, 64), device=k.device, dtype=k.dtype) if output_gated else None
    kernel_gc, kernel_a, kernel_ag = gc, a, ag
    if (
        cu_seqlens is None
        and B > 1
        and T % 64 == 0
        and B * T < 2**31
        and k.is_contiguous()
        and g.is_contiguous()
        and beta.is_contiguous()
    ):
        # Aligned dense chunks share a token axis without crossing chunk boundaries.
        tokens = B * T
        k = k.view(1, tokens, Hg, K)
        g, beta = g.view(1, tokens, H), beta.view(1, tokens, H)
        kernel_gc = gc.view(1, tokens, H)
        kernel_a = a.view(1, tokens, H, 64) if a is not None else None
        kernel_ag = ag.view(1, tokens, H, 64) if ag is not None else None
        B, T = 1, tokens
    indices = prepare_chunk_indices(cu_seqlens, 64) if cu_seqlens is not None else None
    chunks = triton.cdiv(T, 64) if indices is None else indices.shape[0]
    if chunks:
        _prepare_kernel[(chunks, B * H if indices is None else H)](
            k=k,
            g=g,
            g_out=kernel_gc,
            beta=beta,
            A=kernel_a,
            AG=kernel_ag,
            cu_seqlens=cu_seqlens,
            chunk_indices=indices,
            T=T,
            H=Hg,
            HV=H,
            K=K,
            BT=64,
            BC=16,
            USE_G=not output_ungated,
            OUTPUT_GATED=output_gated,
            OUTPUT_UNGATED=output_ungated,
            MERGE_PRECISION=merge_precision,
            FULL_CHUNKS=cu_seqlens is None and T % 64 == 0,
        )
    return (gc, a, ag) if output_gated else (gc, a)
