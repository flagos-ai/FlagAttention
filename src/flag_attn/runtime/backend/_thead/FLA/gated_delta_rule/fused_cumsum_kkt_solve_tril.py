# Copyright 2026 FlagOS Contributors

# Copyright (c) 2025 FlagGems. All rights reserved.
# Fused cumsum + KKT + solve_tril for chunk_gated_delta_rule. Returns g_out, A_inv; w_u is separate.
# License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from flag_attn.FLA.compat import libentry, libtuner
from flag_attn.FLA.index import prepare_chunk_indices
from flag_attn.FLA.utils import exp


FLA_TRIL_PRECISION = os.environ.get("FLAG_ATTN_TRIL_PRECISION", "ieee")
BLOCK16_TRIL_PRECISION = os.environ.get("FLAG_ATTN_TRIL_PRECISION", "tf32x3")
ALLOWED_TRIL_PRECISIONS = {"ieee", "tf32", "tf32x3"}
if FLA_TRIL_PRECISION not in ALLOWED_TRIL_PRECISIONS:
    raise ValueError(
        "FLAG_ATTN_TRIL_PRECISION must be one of "
        f"{sorted(ALLOWED_TRIL_PRECISIONS)}, got {FLA_TRIL_PRECISION}"
    )


@libentry()
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel(
    g_in,
    g_out,
    k,
    beta,
    A,
    A_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # ---------- cumsum ----------
    p_g_in = tl.make_block_ptr(
        g_in + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g_out = tl.make_block_ptr(
        g_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g = tl.load(p_g_in, boundary_check=(0,)).to(tl.float32)
    b_g = tl.cumsum(b_g, axis=0)
    tl.store(p_g_out, b_g.to(p_g_out.dtype.element_ty), boundary_check=(0,))

    # ---------- KKT (write L to A) ----------
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
    b_g_diff = b_g[:, None] - b_g[None, :]
    b_A = b_A * exp(b_g_diff)
    m_A_kkt = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A_kkt, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    # ---------- solve_tril (read A, write A_inv) ----------
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A_base = A + (bos * H + i_h) * BT
    A_inv_base = A_inv + (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p_A_22 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_A_33 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_A_44 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_A_31 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_A_32 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_A_41 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_A_42 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_A_43 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
    )
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),
        b_Ai_22,
        input_precision=DOT_PRECISION,
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),
        b_Ai_33,
        input_precision=DOT_PRECISION,
    )
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    p_Ai_11 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        A_inv_base,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + 16, 16),
        (16, 16),
        (1, 0),
    )
    p_Ai_33 = tl.make_block_ptr(
        A_inv_base,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + 32, 32),
        (16, 16),
        (1, 0),
    )
    p_Ai_44 = tl.make_block_ptr(
        A_inv_base,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + 48, 48),
        (16, 16),
        (1, 0),
    )
    p_Ai_21 = tl.make_block_ptr(
        A_inv_base,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + 16, 0),
        (16, 16),
        (1, 0),
    )
    p_Ai_31 = tl.make_block_ptr(
        A_inv_base,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + 32, 0),
        (16, 16),
        (1, 0),
    )
    p_Ai_32 = tl.make_block_ptr(
        A_inv_base,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + 32, 16),
        (16, 16),
        (1, 0),
    )
    p_Ai_41 = tl.make_block_ptr(
        A_inv_base,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + 48, 0),
        (16, 16),
        (1, 0),
    )
    p_Ai_42 = tl.make_block_ptr(
        A_inv_base,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + 48, 16),
        (16, 16),
        (1, 0),
    )
    p_Ai_43 = tl.make_block_ptr(
        A_inv_base,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + 48, 32),
        (16, 16),
        (1, 0),
    )
    tl.store(
        p_Ai_11,
        b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_33,
        b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_44,
        b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_31,
        b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_32,
        b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_41,
        b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_42,
        b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_43,
        b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused kernel: cumsum(g) + KKT(L) + solve_tril(L -> inv). Returns (g_out, A_inv).
    w_u stays a separate kernel (e.g. recompute_w_u_fwd) for HGMMA."""
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    output_dtype = output_dtype or k.dtype
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g_out = torch.empty_like(g)
    A_inv = torch.zeros(B, T, H, BT, device=g.device, dtype=output_dtype)

    def grid(meta):
        # Varlen kernels derive the sequence from chunk_indices. The batch id
        # from i_bh is unused there, so H programs are sufficient for B > 1 too.
        return (NT, H if cu_seqlens is not None else B * H)

    if BT == 64 and k.dtype in (torch.float16, torch.bfloat16):
        chunk_gated_delta_rule_block16_kernel[grid](
            g_in=g, g_out=g_out, k=k, beta=beta, A_inv=A_inv,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, Hg=Hg, K=K, BK=64,
            DOT_PRECISION=BLOCK16_TRIL_PRECISION,
            num_warps=1, num_stages=1,
        )
        return g_out, A_inv
    A = torch.empty(B, T, H, BT, device=g.device, dtype=torch.float32)
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel[grid](
        g_in=g,
        g_out=g_out,
        k=k,
        beta=beta,
        A=A,
        A_inv=A_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
        DOT_PRECISION=FLA_TRIL_PRECISION,
    )
    return g_out, A_inv


@triton.jit
def _kkt16_ppu(
    k, beta, gate, bos, T, i_t, i_h,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr,
    ROW: tl.constexpr, COL: tl.constexpr, BK: tl.constexpr,
):
    qh = i_h // (H // Hg)
    key_base = k + (bos * Hg + qh).to(tl.int64) * K
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,),
        (i_t * 64 + ROW,), (16,), (0,),
    )
    strength = tl.load(p_beta, boundary_check=(0,))
    lower = tl.zeros((16, 16), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        left = tl.make_block_ptr(
            key_base, (T, K), (Hg * K, 1),
            (i_t * 64 + ROW, block * BK), (16, BK), (1, 0),
        )
        right = tl.make_block_ptr(
            key_base, (K, T), (1, Hg * K),
            (block * BK, i_t * 64 + COL), (BK, 16), (0, 1),
        )
        keys_left = tl.load(left, boundary_check=(0, 1))
        keys_right = tl.load(right, boundary_check=(0, 1))
        weighted = (keys_left * strength[:, None]).to(keys_left.dtype)
        lower += tl.dot(weighted, keys_right)
    offsets = tl.arange(0, 16)
    gate_rows = tl.gather(gate, ROW + offsets, axis=0)
    gate_cols = tl.gather(gate, COL + offsets, axis=0)
    lower *= exp(gate_rows[:, None] - gate_cols[None, :])
    if ROW == COL:
        lower = tl.where(offsets[:, None] > offsets[None, :], lower, 0.0)
    return lower


@triton.jit
def _invert16_ppu(lower, DOT_PRECISION: tl.constexpr):
    # Strictly lower 16x16 matrices satisfy L**16 = 0.
    # Three doublings form I-L+L**2-...-L**15 with FP32 accumulation.
    offsets = tl.arange(0, 16)
    power = -lower
    inverse = power + (offsets[:, None] == offsets[None, :])
    for _ in tl.static_range(3):
        power = tl.dot(power, power, input_precision=DOT_PRECISION)
        inverse += tl.dot(inverse, power, input_precision=DOT_PRECISION)
    return inverse


@triton.jit
def _store_inverse16_ppu(
    output, block, T, i_t,
    H: tl.constexpr, ROW: tl.constexpr, COL: tl.constexpr,
):
    p_out = tl.make_block_ptr(
        output, (T, 64), (H * 64, 1),
        (i_t * 64 + ROW, COL), (16, 16), (1, 0),
    )
    tl.store(p_out, block.to(p_out.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@libentry()
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_block16_kernel(
    g_in, g_out, k, beta, A_inv, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr,
    BK: tl.constexpr, IS_VARLEN: tl.constexpr, DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        sequence = tl.load(chunk_indices + 2 * i_t).to(tl.int32)
        i_t = tl.load(chunk_indices + 2 * i_t + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + sequence).to(tl.int32)
        eos = tl.load(cu_seqlens + sequence + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    p_gate = tl.make_block_ptr(g_in + bos * H + i_h, (T,), (H,), (i_t * 64,), (64,), (0,))
    gate = tl.cumsum(tl.load(p_gate, boundary_check=(0,)).to(tl.float32), 0)
    p_gate_out = tl.make_block_ptr(g_out + bos * H + i_h, (T,), (H,), (i_t * 64,), (64,), (0,))
    tl.store(p_gate_out, gate.to(p_gate_out.dtype.element_ty), boundary_check=(0,))
    # A native 16x16 Tensor Cell tile stays warp-local throughout each solve.
    d11 = _invert16_ppu(_kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 0, 0, BK), DOT_PRECISION)
    d22 = _invert16_ppu(_kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 16, 16, BK), DOT_PRECISION)
    d33 = _invert16_ppu(_kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 32, 32, BK), DOT_PRECISION)
    d44 = _invert16_ppu(_kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 48, 48, BK), DOT_PRECISION)
    l21 = _kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 16, 0, BK)
    d21 = -tl.dot(tl.dot(d22, l21, input_precision=DOT_PRECISION), d11, input_precision=DOT_PRECISION)
    l32 = _kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 32, 16, BK)
    d32 = -tl.dot(tl.dot(d33, l32, input_precision=DOT_PRECISION), d22, input_precision=DOT_PRECISION)
    l43 = _kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 48, 32, BK)
    d43 = -tl.dot(tl.dot(d44, l43, input_precision=DOT_PRECISION), d33, input_precision=DOT_PRECISION)
    l31 = _kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 32, 0, BK)
    d31 = -tl.dot(
        d33,
        tl.dot(l31, d11, input_precision=DOT_PRECISION) + tl.dot(l32, d21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    l42 = _kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 48, 16, BK)
    d42 = -tl.dot(
        d44,
        tl.dot(l42, d22, input_precision=DOT_PRECISION) + tl.dot(l43, d32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    l41 = _kkt16_ppu(k, beta, gate, bos, T, i_t, i_h, H, Hg, K, 48, 0, BK)
    d41 = -tl.dot(
        d44,
        tl.dot(l41, d11, input_precision=DOT_PRECISION)
        + tl.dot(l42, d21, input_precision=DOT_PRECISION)
        + tl.dot(l43, d31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    output_base = A_inv + (bos * H + i_h).to(tl.int64) * 64
    _store_inverse16_ppu(output_base, d11, T, i_t, H, 0, 0)
    _store_inverse16_ppu(output_base, d22, T, i_t, H, 16, 16)
    _store_inverse16_ppu(output_base, d33, T, i_t, H, 32, 32)
    _store_inverse16_ppu(output_base, d44, T, i_t, H, 48, 48)
    _store_inverse16_ppu(output_base, d21, T, i_t, H, 16, 0)
    _store_inverse16_ppu(output_base, d31, T, i_t, H, 32, 0)
    _store_inverse16_ppu(output_base, d32, T, i_t, H, 32, 16)
    _store_inverse16_ppu(output_base, d41, T, i_t, H, 48, 0)
    _store_inverse16_ppu(output_base, d42, T, i_t, H, 48, 16)
    _store_inverse16_ppu(output_base, d43, T, i_t, H, 48, 32)
