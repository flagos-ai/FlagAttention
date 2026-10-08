# Enflame GDN2 Triton implementation.
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

"""BT=64 fused Triton inference kernels for GDN2 prefill."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# 此仓库的pre index的入口参数没有BT，不支持varlen
from ..compat import exp, exp2, has_triton_tle
from ..index import prepare_chunk_indices, prepare_chunk_offsets


@triton.jit
def softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + exp(x)))

_GRID_MODE_ORIGINAL = 0
_GRID_MODE_SWAP_BH = 1
_GRID_MODE_SPLIT_BH = 2
_ENFLAME_GRID_YZ_LIMIT = 255

LN2 = 0.6931471805599453
RCP_LN2 = 1.4426950408889634

# Fixed S60 configurations.  Triton-GCU stops the whole autotune when any
# candidate fails to compile, so the Enflame inference path deliberately uses
# one configuration per kernel instead of probing multiple candidates.
_S60_INTRA_BK = 256
_S60_INTRA_BV = 256
_S60_INTRA_NUM_WARPS = 1
_S60_INTRA_NUM_STAGES = 2

# Keep the already-fast K=64 path unchanged.  Wider K2 tiles reduce the number
# of state/output dot blocks for K=128 and K=256 on S60.
_S60_STATE_BK_SMALL = 64
_S60_STATE_BK_MEDIUM = 128
_S60_STATE_BK_LARGE = 256
_S60_STATE_BV = 256
_S60_STATE_NUM_WARPS = 1
_S60_STATE_NUM_STAGES = 2
_S60_FUSED_BT = 64

# Retained as capability metadata for existing benchmark/test imports. Runtime
# dispatch does not depend on it: the fused Triton kernels are always defined.
HAS_TLE_GDN2 = has_triton_tle(3, 6, 0)

__all__ = ["chunk_gdn2"]


# triton实现

# 3. Fused Triton kernels for GDN-2 prefill, BT=64, inference path.

# =============================================================================
# K1: fused intra-chunk kernel, parallel over (chunk, batch*head)
# =============================================================================

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "USE_GATE_IN_KERNEL": lambda args: args["A_log"] is not None,
        "USE_QK_L2NORM": lambda args: args["use_qk_l2norm"],
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _chunk_gdn2_fwd_intra_infer_kernel(
    q,  # [B, T, H, K]
    k,  # [B, T, H, K]
    v,  # [B, T, H, V]
    g,  # [B, T, H, K], raw gate if USE_GATE_IN_KERNEL else log-decay
    b,  # [B, T, H, K], GDN-2 erase gate on K axis
    write_w_gate,  # [B, T, H, V], GDN-2 write gate on V axis
    w_wy,  # [B, T, H, K], WY erase auxiliary
    u_wy,  # [B, T, H, V], WY write/value auxiliary
    qg,  # [B, T, H, K]
    kg,  # [B, T, H, K]
    Aqk,  # [B, T, H, BT]
    Akk,  # [B, T, H, BT]
    g_out,  # [B, T, H, K], gate cumsum in base-2 log space
    A_log,  # [H], optional if gate-in-kernel
    dt_bias,  # [H*K], optional if gate-in-kernel
    lower_bound,  # optional safe lower bound in log space, e.g. -5.0
    scale: float,
    g_scale: float,
    l2norm_eps: float,
    use_qk_l2norm: tl.constexpr,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    GRID_MODE: tl.constexpr,
):
    if GRID_MODE == 1:
        i_bh = tl.program_id(0)
        i_t = tl.program_id(1)
    elif GRID_MODE == 2:
        i_t = tl.program_id(0)
        i_b = tl.program_id(1)
        i_h = tl.program_id(2)
    else:
        i_t = tl.program_id(0)
        i_bh = tl.program_id(1)
    if GRID_MODE != 2:
        i_b = i_bh // H
        i_h = i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
        eos = i_b * T + T

    if i_t * BT >= T:
        return

    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    g += (bos * H + i_h).to(tl.int64) * K
    b += (bos * H + i_h).to(tl.int64) * K
    g_out += (bos * H + i_h).to(tl.int64) * K
    v += (bos * H + i_h).to(tl.int64) * V
    write_w_gate += (bos * H + i_h).to(tl.int64) * V
    Aqk += (bos * H + i_h).to(tl.int64) * BT
    Akk += (bos * H + i_h).to(tl.int64) * BT
    w_wy += (bos * H + i_h).to(tl.int64) * K
    u_wy += (bos * H + i_h).to(tl.int64) * V

    if STORE_QG:
        qg += (bos * H + i_h).to(tl.int64) * K
    if STORE_KG:
        kg += (bos * H + i_h).to(tl.int64) * K

    o_i = tl.arange(0, BT)
    o_c = i_t * BT + o_i
    m_c = o_c < T

    # Phase 0: optional L2 norm of q/k inside the kernel.
    if USE_QK_L2NORM:
        b_q_ss = tl.zeros([BT], dtype=tl.float32)
        b_k_ss = tl.zeros([BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_k = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_q_blk = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_k_blk = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_q_ss += tl.sum(b_q_blk * b_q_blk, 1)
            b_k_ss += tl.sum(b_k_blk * b_k_blk, 1)
        b_q_rstd = 1.0 / tl.sqrt(b_q_ss + l2norm_eps)
        b_k_rstd = 1.0 / tl.sqrt(b_k_ss + l2norm_eps)
    else:
        b_q_rstd = tl.full([BT], 1.0, dtype=tl.float32)
        b_k_rstd = tl.full([BT], 1.0, dtype=tl.float32)

    # Phase 1: base-2 gate cumsum and intra-chunk score matrices.
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Akk = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_GATE_IN_KERNEL:
        # exp(A_log) in base 2: exp(A_log) = exp2(A_log / ln(2))
        b_A = exp2(tl.load(A_log + i_h).to(tl.float32) * g_scale)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_b = tl.make_block_ptr(
            b, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )

        b_q_blk = (
            tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * b_q_rstd[:, None]
        )
        b_k_blk = (
            tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * b_k_rstd[:, None]
        )
        b_g_blk = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_erase_blk = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)

        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                p_dt = tl.make_block_ptr(
                    dt_bias + i_h * K, (K,), (1,), (i_k * BK,), (BK,), (0,)
                )
                b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
                b_g_blk += b_bias[None, :]
            if USE_LOWER_BOUND:
                b_g_blk = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g_blk)
            else:
                b_g_blk = -b_A * softplus(b_g_blk) * g_scale
        else:
            b_g_blk *= g_scale

        b_g_blk = tl.cumsum(b_g_blk, axis=0)
        p_g_out = tl.make_block_ptr(
            g_out,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        tl.store(
            p_g_out,
            b_g_blk.to(g_out.dtype.element_ty),
            boundary_check=(0, 1),
        )

        b_gq = tl.where(m_c[:, None], exp2(b_g_blk), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-b_g_blk), 0.0)

        # Keep dot operands in the input dtype so S60 can use its
        # low-precision matrix path while retaining FP32 accumulation.
        b_kgt = tl.trans((b_k_blk * b_gk).to(k.dtype.element_ty))
        b_q_dot = (b_q_blk * b_gq).to(q.dtype.element_ty)
        b_Aqk += tl.dot(b_q_dot, b_kgt, out_dtype=tl.float32)
        # GDN-2 erase gate is vector-valued on K; fold it into the current-row
        # key factor before the dot instead of applying a scalar beta afterwards.
        b_k_erase_dot = (b_k_blk * b_erase_blk * b_gq).to(
            k.dtype.element_ty
        )
        b_Akk += tl.dot(b_k_erase_dot, b_kgt, out_dtype=tl.float32)

    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    # Phase 2: 64x64 finite Neumann-series inverse. b_L is strictly lower
    # triangular, so b_L**64 is zero. Include factors through L32 to cover
    # every term through L63.
    b_L = b_Akk.to(tl.float16)
    b_Ai = m_I.to(tl.float16) - b_L
    b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
    b_Ai += tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
    b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
    b_Ai += tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
    b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
    b_Ai += tl.dot(b_Ai, b_L8, out_dtype=tl.float16)
    b_L16 = tl.dot(b_L8, b_L8, out_dtype=tl.float16)
    b_Ai += tl.dot(b_Ai, b_L16, out_dtype=tl.float16)
    b_L32 = tl.dot(b_L16, b_L16, out_dtype=tl.float16)
    b_Ai += tl.dot(b_Ai, b_L32, out_dtype=tl.float16)

    p_Akk = tl.make_block_ptr(
        Akk, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    # Phase 3a: u_wy = inv(Akk) @ (write_w_gate * v)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_write = tl.make_block_ptr(
            write_w_gate, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u_wy, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v_blk = tl.load(p_v, boundary_check=(0, 1))
        b_write_blk = tl.load(p_write, boundary_check=(0, 1)).to(tl.float32)
        b_v_write = (b_v_blk.to(tl.float32) * b_write_blk).to(b_v_blk.dtype)
        b_u_blk = tl.dot(b_Ai.to(b_v_write.dtype), b_v_write)
        tl.store(p_u, b_u_blk.to(u_wy.dtype.element_ty), boundary_check=(0, 1))

    # Phase 3b: qg, kg and w_wy = inv(Akk) @ (erase_b * k * exp2(g_cumsum)).
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_b = tl.make_block_ptr(
            b, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_gk = tl.make_block_ptr(
            g_out, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )

        b_k_blk = (
            tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * b_k_rstd[:, None]
        )
        b_erase_blk = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
        b_gk_blk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_k_erase = b_k_blk * b_erase_blk * exp2(b_gk_blk)

        if STORE_QG:
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_qg = tl.make_block_ptr(
                qg, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_q_blk = (
                tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
                * b_q_rstd[:, None]
            )
            b_qg_val = b_q_blk * exp2(b_gk_blk)
            tl.store(p_qg, b_qg_val.to(qg.dtype.element_ty), boundary_check=(0, 1))

        if STORE_KG:
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            last_idx = tl.minimum(i_t * BT + BT, T) - 1
            b_gn = tl.load(g_out + last_idx * H * K + o_k, mask=m_k, other=0.0).to(
                tl.float32
            )
            b_kg_val = b_k_blk * tl.where(
                m_c[:, None], exp2(b_gn[None, :] - b_gk_blk), 0.0
            )
            p_kg = tl.make_block_ptr(
                kg, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            tl.store(p_kg, b_kg_val.to(kg.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w_wy, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_w_blk = tl.dot(
            b_Ai.to(k.dtype.element_ty),
            b_k_erase.to(k.dtype.element_ty),
            out_dtype=tl.float32,
        )
        tl.store(p_w, b_w_blk.to(w_wy.dtype.element_ty), boundary_check=(0, 1))

def chunk_gdn2_fwd_intra_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = _S60_FUSED_BT,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fused intra-chunk GDN-2 inference preprocessing for BT=64.

    Returns: (w_wy, u_wy, qg, kg, Aqk, Akk_inv, gate_base2).
    """
    B, T_len, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size

    if BT != _S60_FUSED_BT:
        raise ValueError(
            "Flash-style GDN-2 fused inference requires "
            f"chunk_size={_S60_FUSED_BT}, got {BT}."
        )

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)

    grid_mode = _GRID_MODE_ORIGINAL
    bh = B * H
    if bh > _ENFLAME_GRID_YZ_LIMIT:
        if NT > _ENFLAME_GRID_YZ_LIMIT:
            grid_mode = _GRID_MODE_SPLIT_BH
        else:
            grid_mode = _GRID_MODE_SWAP_BH
    if grid_mode == _GRID_MODE_SWAP_BH:
        grid = (bh, NT)
    elif grid_mode == _GRID_MODE_SPLIT_BH:
        grid = (NT, B, H)
    else:
        grid = (NT, bh)

    g_out = torch.empty(B, T_len, H, K, device=q.device, dtype=torch.float32)
    w_wy = torch.empty(B, T_len, H, K, device=q.device, dtype=q.dtype)
    u_wy = torch.empty(B, T_len, H, V, device=q.device, dtype=v.dtype)
    qg = torch.empty(B, T_len, H, K, device=q.device, dtype=q.dtype)
    kg = torch.empty(B, T_len, H, K, device=q.device, dtype=k.dtype)
    Aqk = torch.empty(B, T_len, H, BT, device=q.device, dtype=q.dtype)
    Akk = torch.empty(B, T_len, H, BT, device=q.device, dtype=q.dtype)

    _chunk_gdn2_fwd_intra_infer_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        write_w_gate=w_gate,
        w_wy=w_wy,
        u_wy=u_wy,
        qg=qg,
        kg=kg,
        Aqk=Aqk,
        Akk=Akk,
        g_out=g_out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        scale=scale,
        g_scale=RCP_LN2,
        l2norm_eps=1e-12,
        use_qk_l2norm=use_qk_l2norm,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T_len,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=_S60_INTRA_BK,
        BV=_S60_INTRA_BV,
        GRID_MODE=grid_mode,
        num_warps=_S60_INTRA_NUM_WARPS,
        num_stages=_S60_INTRA_NUM_STAGES,
    )
    return w_wy, u_wy, qg, kg, Aqk, Akk, g_out

# =============================================================================
# K2: fused state propagation + output, sequential over chunks per (batch, head, V-tile)
# =============================================================================

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "STORE_H": lambda args: args["h"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _chunk_gdn2_fwd_h_o_infer_kernel(
    kg,  # [B, T, H, K]
    w_wy,  # [B, T, H, K]
    u_wy,  # [B, T, H, V]
    gk,  # [B, T, H, K], local gate cumsum
    qg,  # [B, T, H, K]
    Aqk,  # [B, T, H, BT]
    o,  # [B, T, H, V]
    h,  # optional [N, NT, H, K, V] or [N, NT, H, V, K]
    h0,  # optional initial state
    ht,  # optional final state
    cu_seqlens,
    chunk_offsets,
    scale: float,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_H: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GRID_MODE: tl.constexpr,
):
    if GRID_MODE == 1:
        i_nh = tl.program_id(0)
        i_v = tl.program_id(1)
    else:
        i_v = tl.program_id(0)
        i_nh = tl.program_id(1)

    if IS_VARLEN:
        i_n = i_nh // H
        i_h = i_nh % H
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        i_n = i_nh // H
        i_h = i_nh % H
        bos = i_n * T
        eos = i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    kg += (bos * H + i_h).to(tl.int64) * K
    w_wy += (bos * H + i_h).to(tl.int64) * K
    u_wy += (bos * H + i_h).to(tl.int64) * V
    gk += (bos * H + i_h).to(tl.int64) * K
    qg += (bos * H + i_h).to(tl.int64) * K
    Aqk += (bos * H + i_h).to(tl.int64) * BT
    o += (bos * H + i_h).to(tl.int64) * V

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, BK], dtype=tl.float32)
        if K > BK:
            b_h2 = tl.zeros([BV, BK], dtype=tl.float32)
        if K > 2 * BK:
            b_h3 = tl.zeros([BV, BK], dtype=tl.float32)
        if K > 3 * BK:
            b_h4 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > BK:
            b_h2 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > 2 * BK:
            b_h3 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > 3 * BK:
            b_h4 = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0_1 = tl.make_block_ptr(
                h0 + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, 0),
                (BV, BK),
                (1, 0),
            )
        else:
            p_h0_1 = tl.make_block_ptr(
                h0 + i_nh * K * V,
                (K, V),
                (V, 1),
                (0, i_v * BV),
                (BK, BV),
                (1, 0),
            )
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)

        if K > BK:
            if STATE_V_FIRST:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, BK),
                    (BV, BK),
                    (1, 0),
                )
            else:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V,
                    (K, V),
                    (V, 1),
                    (BK, i_v * BV),
                    (BK, BV),
                    (1, 0),
                )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

        if K > 2 * BK:
            if STATE_V_FIRST:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 2 * BK),
                    (BV, BK),
                    (1, 0),
                )
            else:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V,
                    (K, V),
                    (V, 1),
                    (2 * BK, i_v * BV),
                    (BK, BV),
                    (1, 0),
                )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)

        if K > 3 * BK:
            if STATE_V_FIRST:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 3 * BK),
                    (BV, BK),
                    (1, 0),
                )
            else:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V,
                    (K, V),
                    (V, 1),
                    (3 * BK, i_v * BV),
                    (BK, BV),
                    (1, 0),
                )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    if STORE_H:
        h += (boh * H + i_h).to(tl.int64) * K * V

    for i_t in range(NT):
        # Optional: store the pre-chunk state.
        if STORE_H:
            i_t64 = i_t.to(tl.int64)
            if STATE_V_FIRST:
                p_h1 = tl.make_block_ptr(
                    h + i_t64 * H * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 0),
                    (BV, BK),
                    (1, 0),
                )
            else:
                p_h1 = tl.make_block_ptr(
                    h + i_t64 * H * K * V,
                    (K, V),
                    (V, 1),
                    (0, i_v * BV),
                    (BK, BV),
                    (1, 0),
                )
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))

            if K > BK:
                if STATE_V_FIRST:
                    p_h2 = tl.make_block_ptr(
                        h + i_t64 * H * K * V,
                        (V, K),
                        (K, 1),
                        (i_v * BV, BK),
                        (BV, BK),
                        (1, 0),
                    )
                else:
                    p_h2 = tl.make_block_ptr(
                        h + i_t64 * H * K * V,
                        (K, V),
                        (V, 1),
                        (BK, i_v * BV),
                        (BK, BV),
                        (1, 0),
                    )
                tl.store(
                    p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1)
                )

            if K > 2 * BK:
                if STATE_V_FIRST:
                    p_h3 = tl.make_block_ptr(
                        h + i_t64 * H * K * V,
                        (V, K),
                        (K, 1),
                        (i_v * BV, 2 * BK),
                        (BV, BK),
                        (1, 0),
                    )
                else:
                    p_h3 = tl.make_block_ptr(
                        h + i_t64 * H * K * V,
                        (K, V),
                        (V, 1),
                        (2 * BK, i_v * BV),
                        (BK, BV),
                        (1, 0),
                    )
                tl.store(
                    p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1)
                )

            if K > 3 * BK:
                if STATE_V_FIRST:
                    p_h4 = tl.make_block_ptr(
                        h + i_t64 * H * K * V,
                        (V, K),
                        (K, 1),
                        (i_v * BV, 3 * BK),
                        (BV, BK),
                        (1, 0),
                    )
                else:
                    p_h4 = tl.make_block_ptr(
                        h + i_t64 * H * K * V,
                        (K, V),
                        (V, 1),
                        (3 * BK, i_v * BV),
                        (BK, BV),
                        (1, 0),
                    )
                tl.store(
                    p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1)
                )

        # v_new = u_wy - w_wy @ h
        p_w = tl.make_block_ptr(
            w_wy, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
        )
        b_w_blk = tl.load(p_w, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_v_new = tl.dot(b_w_blk, tl.trans(b_h1).to(b_w_blk.dtype))
        else:
            b_v_new = tl.dot(b_w_blk, b_h1.to(b_w_blk.dtype))

        if K > BK:
            p_w = tl.make_block_ptr(
                w_wy,
                (T, K),
                (H * K, 1),
                (i_t * BT, BK),
                (BT, BK),
                (1, 0),
            )
            b_w_blk = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v_new += tl.dot(b_w_blk, tl.trans(b_h2).to(b_w_blk.dtype))
            else:
                b_v_new += tl.dot(b_w_blk, b_h2.to(b_w_blk.dtype))

        if K > 2 * BK:
            p_w = tl.make_block_ptr(
                w_wy,
                (T, K),
                (H * K, 1),
                (i_t * BT, 2 * BK),
                (BT, BK),
                (1, 0),
            )
            b_w_blk = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v_new += tl.dot(b_w_blk, tl.trans(b_h3).to(b_w_blk.dtype))
            else:
                b_v_new += tl.dot(b_w_blk, b_h3.to(b_w_blk.dtype))

        if K > 3 * BK:
            p_w = tl.make_block_ptr(
                w_wy,
                (T, K),
                (H * K, 1),
                (i_t * BT, 3 * BK),
                (BT, BK),
                (1, 0),
            )
            b_w_blk = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v_new += tl.dot(b_w_blk, tl.trans(b_h4).to(b_w_blk.dtype))
            else:
                b_v_new += tl.dot(b_w_blk, b_h4.to(b_w_blk.dtype))

        p_u = tl.make_block_ptr(
            u_wy, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v_new = tl.load(p_u, boundary_check=(0, 1)) - b_v_new

        # o = scale * qg @ h + Aqk @ v_new
        p_qg = tl.make_block_ptr(
            qg, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
        )
        b_qg_blk = tl.load(p_qg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_o = tl.dot(b_qg_blk, tl.trans(b_h1).to(b_qg_blk.dtype))
        else:
            b_o = tl.dot(b_qg_blk, b_h1.to(b_qg_blk.dtype))

        if K > BK:
            p_qg = tl.make_block_ptr(
                qg,
                (T, K),
                (H * K, 1),
                (i_t * BT, BK),
                (BT, BK),
                (1, 0),
            )
            b_qg_blk = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg_blk, tl.trans(b_h2).to(b_qg_blk.dtype))
            else:
                b_o += tl.dot(b_qg_blk, b_h2.to(b_qg_blk.dtype))

        if K > 2 * BK:
            p_qg = tl.make_block_ptr(
                qg,
                (T, K),
                (H * K, 1),
                (i_t * BT, 2 * BK),
                (BT, BK),
                (1, 0),
            )
            b_qg_blk = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg_blk, tl.trans(b_h3).to(b_qg_blk.dtype))
            else:
                b_o += tl.dot(b_qg_blk, b_h3.to(b_qg_blk.dtype))

        if K > 3 * BK:
            p_qg = tl.make_block_ptr(
                qg,
                (T, K),
                (H * K, 1),
                (i_t * BT, 3 * BK),
                (BT, BK),
                (1, 0),
            )
            b_qg_blk = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg_blk, tl.trans(b_h4).to(b_qg_blk.dtype))
            else:
                b_o += tl.dot(b_qg_blk, b_h4.to(b_qg_blk.dtype))

        b_o *= scale
        p_Aqk = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
        # FP32 remains the resource-safe fused path for K=256.
        b_o += tl.dot(
            b_Aqk.to(tl.float32),
            b_v_new,
            allow_tf32=False,
            out_dtype=tl.float32,
        )

        p_o = tl.make_block_ptr(
            o,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # The recurrent state update uses low-precision dot inputs.
        b_v_cast = b_v_new.to(kg.dtype.element_ty)

        # State decay: h *= exp2(g_last) per K channel.
        last_idx = tl.minimum(i_t * BT + BT, T) - 1
        o_k = tl.arange(0, BK)
        b_g_last1 = tl.load(
            gk + last_idx * H * K + o_k, mask=o_k < K, other=0.0
        ).to(tl.float32)
        if STATE_V_FIRST:
            b_h1 *= exp2(b_g_last1)[None, :]
        else:
            b_h1 *= exp2(b_g_last1)[:, None]

        if K > BK:
            o_k2 = BK + o_k
            b_g_last2 = tl.load(
                gk + last_idx * H * K + o_k2, mask=o_k2 < K, other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h2 *= exp2(b_g_last2)[None, :]
            else:
                b_h2 *= exp2(b_g_last2)[:, None]

        if K > 2 * BK:
            o_k3 = 2 * BK + o_k
            b_g_last3 = tl.load(
                gk + last_idx * H * K + o_k3, mask=o_k3 < K, other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h3 *= exp2(b_g_last3)[None, :]
            else:
                b_h3 *= exp2(b_g_last3)[:, None]

        if K > 3 * BK:
            o_k4 = 3 * BK + o_k
            b_g_last4 = tl.load(
                gk + last_idx * H * K + o_k4, mask=o_k4 < K, other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h4 *= exp2(b_g_last4)[None, :]
            else:
                b_h4 *= exp2(b_g_last4)[:, None]

        # State update: h += kg^T @ v_new.  The GDN-2 write gate has already
        # been folded into u_wy and therefore into v_new.
        p_kg = tl.make_block_ptr(
            kg, (K, T), (1, H * K), (0, i_t * BT), (BK, BT), (0, 1)
        )
        b_kg_blk = tl.load(p_kg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_kg_blk, b_v_cast))
        else:
            b_h1 += tl.dot(b_kg_blk, b_v_cast)

        if K > BK:
            p_kg = tl.make_block_ptr(
                kg,
                (K, T),
                (1, H * K),
                (BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            b_kg_blk = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_kg_blk, b_v_cast))
            else:
                b_h2 += tl.dot(b_kg_blk, b_v_cast)

        if K > 2 * BK:
            p_kg = tl.make_block_ptr(
                kg,
                (K, T),
                (1, H * K),
                (2 * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            b_kg_blk = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_kg_blk, b_v_cast))
            else:
                b_h3 += tl.dot(b_kg_blk, b_v_cast)

        if K > 3 * BK:
            p_kg = tl.make_block_ptr(
                kg,
                (K, T),
                (1, H * K),
                (3 * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            b_kg_blk = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_kg_blk, b_v_cast))
            else:
                b_h4 += tl.dot(b_kg_blk, b_v_cast)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, 0),
                (BV, BK),
                (1, 0),
            )
        else:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V,
                (K, V),
                (V, 1),
                (0, i_v * BV),
                (BK, BV),
                (1, 0),
            )
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

        if K > BK:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, BK),
                    (BV, BK),
                    (1, 0),
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V,
                    (K, V),
                    (V, 1),
                    (BK, i_v * BV),
                    (BK, BV),
                    (1, 0),
                )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

        if K > 2 * BK:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 2 * BK),
                    (BV, BK),
                    (1, 0),
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V,
                    (K, V),
                    (V, 1),
                    (2 * BK, i_v * BV),
                    (BK, BV),
                    (1, 0),
                )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

        if K > 3 * BK:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 3 * BK),
                    (BV, BK),
                    (1, 0),
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V,
                    (K, V),
                    (V, 1),
                    (3 * BK, i_v * BV),
                    (BK, BV),
                    (1, 0),
                )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

def chunk_gdn2_fwd_h_o_infer(
    kg: torch.Tensor,
    w_wy: torch.Tensor,
    u_wy: torch.Tensor,
    gk: torch.Tensor,
    qg: torch.Tensor,
    Aqk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = _S60_FUSED_BT,
) -> (
    tuple[torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
):
    """Fused GDN-2 state propagation + output for BT=64 inference."""
    B, T, H, K = kg.shape
    V = u_wy.shape[-1]
    BT = chunk_size
    if K <= 64:
        state_bk = _S60_STATE_BK_SMALL
    elif K <= 128:
        state_bk = _S60_STATE_BK_MEDIUM
    else:
        state_bk = _S60_STATE_BK_LARGE

    if BT != _S60_FUSED_BT:
        raise ValueError(
            "Flash-style GDN-2 fused inference requires "
            f"chunk_size={_S60_FUSED_BT}, got {BT}."
        )

    if cu_seqlens is None:
        N = B
        NT = triton.cdiv(T, BT)
        chunk_offsets = None
    else:
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices) if chunk_indices is not None else None
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        if NT is None:
            # This is only used for h allocation.  If the caller needs h in
            # varlen mode, pass chunk_indices from prepare_chunk_indices.
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
            NT = len(chunk_indices)

    final_state = None
    if output_final_state:
        if state_v_first:
            final_state = kg.new_zeros(N, H, V, K, dtype=torch.float32)
        else:
            final_state = kg.new_zeros(N, H, K, V, dtype=torch.float32)

    h = None
    if return_intermediate_states:
        if state_v_first:
            h = kg.new_empty(N, NT, H, V, K, dtype=torch.bfloat16)
        else:
            h = kg.new_empty(N, NT, H, K, V, dtype=torch.bfloat16)

    o = torch.empty(B, T, H, V, device=kg.device, dtype=u_wy.dtype)

    grid_mode = _GRID_MODE_ORIGINAL
    nh = N * H
    if nh > _ENFLAME_GRID_YZ_LIMIT:
        grid_mode = _GRID_MODE_SWAP_BH

    def grid(meta):
        n_v_tiles = triton.cdiv(V, meta["BV"])
        if grid_mode == _GRID_MODE_SWAP_BH:
            return (N * H, n_v_tiles)
        return (n_v_tiles, N * H)

    _chunk_gdn2_fwd_h_o_infer_kernel[grid](
        kg=kg,
        w_wy=w_wy,
        u_wy=u_wy,
        gk=gk,
        qg=qg,
        Aqk=Aqk,
        o=o,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=state_bk,
        BV=_S60_STATE_BV,
        STATE_V_FIRST=state_v_first,
        GRID_MODE=grid_mode,
        num_warps=_S60_STATE_NUM_WARPS,
    )

    if return_intermediate_states:
        return o, final_state, h
    return o, final_state

def chunk_gdn2_fwd_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = _S60_FUSED_BT,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    cp_context: "None" = None,
    **kwargs,
) -> (
    tuple[torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
):
    """FlashKDA-style fused inference forward for GDN-2.

    Args follow ``fla.ops.gdn2.chunk_gdn2`` where possible.

    Important differences from the training path:
    - Only inference forward is implemented.
    - ``chunk_size`` must be 64.
    - ``b`` is the post-activation erase gate [B,T,H,K].
    - ``w`` is the post-activation write gate [B,T,H,V].
    - ``safe_gate`` only affects validation; the actual gated activation path
    is selected by passing ``lower_bound`` with ``use_gate_in_kernel=True``.
    """
    if cp_context is not None:
        raise NotImplementedError(
            "chunk_gdn2_fwd_infer currently does not support context parallelism."
        )
    if disable_recompute:
        # Kept for signature compatibility; inference does not use recompute.
        pass
    if chunk_size != _S60_FUSED_BT:
        raise ValueError(
            "Flash-style GDN-2 fused inference requires "
            f"chunk_size={_S60_FUSED_BT}, got {chunk_size}."
        )
    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError(
            f"q, k, g, b must all have shape [B,T,H,K]; got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, g={tuple(g.shape)}, b={tuple(b.shape)}."
        )
    if v.shape != w.shape:
        raise ValueError(
            f"v and w must both have shape [B,T,H,V]; got v={tuple(v.shape)}, w={tuple(w.shape)}."
        )
    if q.shape[:3] != v.shape[:3]:
        raise ValueError(
            f"q/k/g/b and v/w must match on [B,T,H]; got q={tuple(q.shape)}, v={tuple(v.shape)}."
        )
    if q.shape[-1] > 256:
        raise ValueError(
            f"GDN-2 fused inference supports K <= 256, got K={q.shape[-1]}."
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be float32.")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log must be provided when use_gate_in_kernel=True.")
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError(
                "lower_bound must be set when safe_gate=True and use_gate_in_kernel=True."
            )
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound}.")

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    w_wy, u_wy, qg, kg, Aqk, _Akk, g_cumsum = chunk_gdn2_fwd_intra_infer(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w_gate=w,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound if use_gate_in_kernel else None,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
    )

    return chunk_gdn2_fwd_h_o_infer(
        kg=kg,
        w_wy=w_wy,
        u_wy=u_wy,
        gk=g_cumsum,
        qg=qg,
        Aqk=Aqk,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )


# 4. 唯一公开入口
def chunk_gdn2(
    q,
    k,
    v,
    g,
    b,
    w,
    *,
    A_log=None,
    dt_bias=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
    chunk_size=_S60_FUSED_BT,
    return_intermediate_states=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
):
    inputs = {"q": q, "k": k, "v": v, "g": g, "b": b, "w": w}
    if torch.is_grad_enabled():
        raise RuntimeError("chunk_gdn2 is inference-only")
    if any(not isinstance(tensor, torch.Tensor) for tensor in inputs.values()):
        raise TypeError("q, k, v, g, b, and w must be torch.Tensor instances")
    if q.device.type != "gcu":
        raise ValueError(f"chunk_gdn2 requires Enflame GCU inputs, got {q.device}")
    if any(tensor.device != q.device for tensor in inputs.values()):
        raise ValueError("q, k, v, g, b, and w must be on the same GCU device")

    # Older public wrappers and tests explicitly selected the fused path by
    # passing 16. The S60 fused implementation operates on BT=64, so map
    # that legacy selector to the new fixed chunk size before dispatch. Packed
    # sequence metadata must be regenerated because its chunk numbering depends
    # on BT.
    if chunk_size == 16:
        chunk_size = _S60_FUSED_BT
        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    if chunk_size != _S60_FUSED_BT:
        raise ValueError(
            f"chunk_gdn2 fused Triton path requires chunk_size={_S60_FUSED_BT}, "
            f"got {chunk_size}"
        )

    return chunk_gdn2_fwd_infer(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        return_intermediate_states=return_intermediate_states,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
    )
