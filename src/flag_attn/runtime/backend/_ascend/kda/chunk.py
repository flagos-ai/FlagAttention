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


"""Ascend 910B KDA with fused intra-chunk and recurrent/output kernels.

Preserves the validated BN normalization tile and matrix configurations.
This implementation uses ordinary Triton; it does not claim TLE pipelining.
"""
import os

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices, prepare_chunk_offsets

RCP_LN2 = 1.4426950216
_FP16_DOT_PRECISION = tl.constexpr("ieee")

@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))



@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)



@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "USE_GATE_IN_KERNEL": lambda args: args["A_log"] is not None,
        "USE_QK_L2NORM": lambda args: args["use_qk_l2norm"],
        "APPLY_BETA_SIGMOID": lambda args: args["apply_beta_sigmoid"],
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BK": 16, "BV": 16}, num_warps=1, num_stages=4)],
    key=["H", "HV", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_intra_triton_kernel(
    q,
    k,
    v,
    g,
    beta,
    w,
    u,
    qg,
    kg,
    Aqk,
    Akk,
    g_out,
    A_log,
    dt_bias,
    lower_bound,
    scale,
    g_scale,
    l2norm_eps,
    use_qk_l2norm,
    apply_beta_sigmoid,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BN: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    g_out += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    w += (bos * HV + i_hv) * K
    u += (bos * HV + i_hv) * V
    beta += bos * HV + i_hv
    if STORE_QG:
        qg += (bos * HV + i_hv) * K
    if STORE_KG:
        kg += (bos * HV + i_hv) * K

    o_i = tl.arange(0, BT)
    o_c = i_t * BT + o_i
    m_c = o_c < T

    # Phase 0: L2 norm on q/k (optional) + beta sigmoid (optional)
    if USE_QK_L2NORM:
        b_q_ss = tl.zeros([BT], dtype=tl.float32)
        b_k_ss = tl.zeros([BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BN)):
            p_q = tl.make_block_ptr(
                q,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BN),
                (BT, BN),
                (1, 0),
            )
            p_k = tl.make_block_ptr(
                k,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BN),
                (BT, BN),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_q_ss += tl.sum(b_q * b_q, 1)
            b_k_ss += tl.sum(b_k * b_k, 1)

        b_q_rstd = 1.0 / tl.sqrt(b_q_ss + l2norm_eps)
        b_k_rstd = 1.0 / tl.sqrt(b_k_ss + l2norm_eps)

    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    if APPLY_BETA_SIGMOID:
        b_beta = tl.sigmoid(b_beta)

    # Phase 1: cumsum(g) + intra-chunk Aqk/Akk
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Akk = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_GATE_IN_KERNEL:
        b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )

        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * b_q_rstd[:, None]
            b_k = b_k * b_k_rstd[:, None]
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                p_dt = tl.make_block_ptr(
                    dt_bias + i_hv * K, (K,), (1,), (i_k * BK,), (BK,), (0,)
                )
                b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
                b_g = b_g + b_bias[None, :]
            if USE_LOWER_BOUND:
                b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
            else:
                b_g = -b_A * _softplus(b_g) * g_scale
        else:
            b_g = b_g * g_scale
        b_g = tl.cumsum(b_g, axis=0)

        p_g_out = tl.make_block_ptr(
            g_out, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        tl.store(p_g_out, b_g.to(g_out.dtype.element_ty), boundary_check=(0, 1))

        if BT == 16:
            b_gq = tl.where(m_c[:, None], exp2(b_g), 0.0)
            b_gk = tl.where(m_c[:, None], exp2(-b_g), 0.0)
        else:
            # The dot-product factorization only needs the product
            # exp2(g_i - g_j).  Center both factors so neither side alone
            # overflows for BT=32 while preserving that product exactly.
            b_g_shift = 0.5 * (
                tl.max(b_g, axis=0) + tl.min(b_g, axis=0)
            )
            b_gq = tl.where(
                m_c[:, None], exp2(b_g - b_g_shift[None, :]), 0.0
            )
            b_gk = tl.where(
                m_c[:, None], exp2(b_g_shift[None, :] - b_g), 0.0
            )

        b_kgt = tl.trans(b_k * b_gk)
        b_Aqk += tl.dot(b_q * b_gq, b_kgt)
        b_Akk += tl.dot(b_k * b_gq, b_kgt)

    # Causal mask
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    # Phase 2: Solve (I + L)^{-1} via parallel prefix.  Gate centering above
    # keeps b_Akk finite for BT=32, so retain the fast fp16 dot path and add
    # only the extra nilpotent-matrix powers required by a larger chunk.
    b_L = b_Akk.to(tl.float16)
    b_Ai = m_I.to(tl.float16) - b_L
    b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
    b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
    b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float16)
    if BT > 16:
        b_L16 = tl.dot(b_L8, b_L8, out_dtype=tl.float16)
        b_Ai = b_Ai + tl.dot(b_Ai, b_L16, out_dtype=tl.float16)
    if BT > 32:
        b_L32 = tl.dot(b_L16, b_L16, out_dtype=tl.float16)
        b_Ai = b_Ai + tl.dot(b_Ai, b_L32, out_dtype=tl.float16)

    p_Akk_out = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_Akk_out, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    # Phase 3: w, u, qg, kg
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_Ai.to(b_vb.dtype), b_vb)
        tl.store(p_u, b_u.to(u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_gk = tl.make_block_ptr(
            g_out, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * b_k_rstd[:, None]
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb = b_k * b_beta[:, None] * exp2(b_gk)

        if STORE_QG:
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_qg_out = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * b_q_rstd[:, None]
            b_qg_val = b_q * exp2(b_gk)
            tl.store(p_qg_out, b_qg_val.to(qg.dtype.element_ty), boundary_check=(0, 1))

        if STORE_KG:
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            last_idx = tl.minimum(i_t * BT + BT, T) - 1
            b_gn = tl.load(g_out + last_idx * HV * K + o_k, mask=m_k, other=0.0).to(
                tl.float32
            )
            b_kg_val = b_k * tl.where(m_c[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg_out = tl.make_block_ptr(
                kg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            tl.store(p_kg_out, b_kg_val.to(kg.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_w = tl.dot(b_Ai.to(b_kb.to(b_k.dtype).dtype), b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(w.dtype.element_ty), boundary_check=(0, 1))



def _kda_fwd_intra_triton(
    q,
    k,
    v,
    g,
    beta,
    scale,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=16,
    lower_bound=None,
    A_log=None,
    dt_bias=None,
    use_qk_l2norm=True,
    apply_beta_sigmoid=True,
):
    """Fused intra-chunk computation. Returns (w, u, qg, kg, Aqk, Akk, g_cumsum)."""
    B, T_len, H, K = q.shape
    HV = g.shape[2]
    V = v.shape[-1]
    BT = chunk_size
    norm_bk_env = os.environ.get("FLAG_ATTN_ASCEND_KDA_NORM_BK")
    norm_bk = (
        min(128, 1 << (int(K) - 1).bit_length())
        if norm_bk_env is None
        else int(norm_bk_env)
    )
    if norm_bk not in (16, 32, 64, 128):
        raise ValueError(
            "FLAG_ATTN_ASCEND_KDA_NORM_BK must be one of {16, 32, 64, 128}, "
            f"got {norm_bk}"
        )

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * HV)

    g_out = torch.empty(B, T_len, HV, K, device=q.device, dtype=torch.float32)
    w = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    u = torch.empty(B, T_len, HV, V, device=q.device, dtype=q.dtype)
    qg = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    kg = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    Aqk = torch.empty(B, T_len, HV, BT, device=q.device, dtype=q.dtype)
    Akk = torch.zeros(B, T_len, HV, BT, device=q.device, dtype=q.dtype)

    _kda_fwd_intra_triton_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        w=w,
        u=u,
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
        l2norm_eps=1e-6,
        use_qk_l2norm=use_qk_l2norm,
        apply_beta_sigmoid=apply_beta_sigmoid,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T_len,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BN=norm_bk,
    )
    return w, u, qg, kg, Aqk, Akk, g_out



@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BV": 64}, num_warps=4)],
    key=["HV", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_h_o_triton_kernel(
    kg,
    w,
    u,
    gk,
    qg,
    Aqk,
    o,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_nh // HV
        i_h = i_nh % HV
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        i_n = i_nh // HV
        i_h = i_nh % HV
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    kg += (bos * HV + i_h).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    u += (bos * HV + i_h).to(tl.int64) * V
    gk += (bos * HV + i_h).to(tl.int64) * K
    qg += (bos * HV + i_h).to(tl.int64) * K
    Aqk += (bos * HV + i_h).to(tl.int64) * BT
    o += (bos * HV + i_h).to(tl.int64) * V

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0_1 = tl.make_block_ptr(
                h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_h0_1 = tl.make_block_ptr(
                h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        # v_new = u - w @ h
        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_u = tl.make_block_ptr(
            u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v

        # output = scale * qg @ h + Aqk @ v_new
        p_qg = tl.make_block_ptr(
            qg, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_o = tl.dot(b_qg, tl.trans(b_h1).to(b_qg.dtype))
        else:
            b_o = tl.dot(b_qg, b_h1.to(b_qg.dtype))
        if K > 64:
            p_qg = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h2).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h2.to(b_qg.dtype))
        if K > 128:
            p_qg = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h3).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h3.to(b_qg.dtype))
        if K > 192:
            p_qg = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h4).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h4.to(b_qg.dtype))
        b_o *= scale

        p_Aqk = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
        b_o += tl.dot(b_Aqk.to(b_v.dtype), b_v)

        p_o = tl.make_block_ptr(
            o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # decay: h *= exp2(gk_last)
        last_idx = tl.minimum(i_t * BT + BT, T) - 1
        o_k1 = tl.arange(0, 64)
        b_gk_last1 = tl.load(
            gk + last_idx * HV * K + o_k1, mask=(o_k1 < K), other=0.0
        ).to(tl.float32)
        if STATE_V_FIRST:
            b_h1 *= exp2(b_gk_last1)[None, :]
        else:
            b_h1 *= exp2(b_gk_last1)[:, None]
        if K > 64:
            o_k2 = 64 + o_k1
            b_gk_last2 = tl.load(
                gk + last_idx * HV * K + o_k2, mask=(o_k2 < K), other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h2 *= exp2(b_gk_last2)[None, :]
            else:
                b_h2 *= exp2(b_gk_last2)[:, None]
        if K > 128:
            o_k3 = 128 + o_k1
            b_gk_last3 = tl.load(
                gk + last_idx * HV * K + o_k3, mask=(o_k3 < K), other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h3 *= exp2(b_gk_last3)[None, :]
            else:
                b_h3 *= exp2(b_gk_last3)[:, None]
        if K > 192:
            o_k4 = 192 + o_k1
            b_gk_last4 = tl.load(
                gk + last_idx * HV * K + o_k4, mask=(o_k4 < K), other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h4 *= exp2(b_gk_last4)[None, :]
            else:
                b_h4 *= exp2(b_gk_last4)[:, None]

        # state update: h += kg^T @ v_new
        b_v = b_v.to(kg.dtype.element_ty)
        p_kg = tl.make_block_ptr(
            kg, (K, T), (1, HV * K), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_kg = tl.load(p_kg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_kg, b_v))
        else:
            b_h1 += tl.dot(b_kg, b_v)
        if K > 64:
            p_kg = tl.make_block_ptr(
                kg, (K, T), (1, HV * K), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h2 += tl.dot(b_kg, b_v)
        if K > 128:
            p_kg = tl.make_block_ptr(
                kg, (K, T), (1, HV * K), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h3 += tl.dot(b_kg, b_v)
        if K > 192:
            p_kg = tl.make_block_ptr(
                kg, (K, T), (1, HV * K), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h4 += tl.dot(b_kg, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))



def _kda_fwd_h_o_triton(
    kg,
    w,
    u,
    gk,
    qg,
    Aqk,
    scale,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=16,
):
    """Fused state propagation + output."""
    B, T, HV, K = kg.shape
    V = u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N = B
        chunk_offsets = None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    final_state = None
    if output_final_state:
        if state_v_first:
            final_state = kg.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = kg.new_zeros(N, HV, K, V, dtype=torch.float32)

    o = torch.zeros(B, T, HV, V, device=kg.device, dtype=u.dtype)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * HV)

    _kda_fwd_h_o_triton_kernel[grid](
        kg=kg,
        w=w,
        u=u,
        gk=gk,
        qg=qg,
        Aqk=Aqk,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return o, final_state



def chunk_kda_fwd_infer_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the two fused Ascend KDA phases for one segment."""
    BT = chunk_size

    # Segmenting T leaves gaps between batches when B > 1. These kernels
    # address packed B/T/H storage, so normalize only noncontiguous segments.
    # Contiguous inputs retain their existing storage and launch schedule.
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    if A_log is not None:
        A_log = A_log.contiguous()
    if dt_bias is not None:
        dt_bias = dt_bias.contiguous()

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    w, u, qg, kg, Aqk, Akk, g_cumsum = _kda_fwd_intra_triton(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
        lower_bound=lower_bound,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        apply_beta_sigmoid=use_beta_sigmoid_in_kernel,
    )
    # Ascend's Triton 3.5.1 backend produces incorrect results in the
    # STATE_V_FIRST=True branch of ``_kda_fwd_h_o_triton_kernel`` when the
    # sequence spans multiple chunks.  The equivalent K/V-layout branch is
    # numerically correct on the same device.  Keep the native V/K branch for
    # CUDA, and use explicit state transposes only for the Ascend workaround.
    use_npu_state_layout_workaround = state_v_first and q.device.type == "npu"
    triton_state_v_first = False if use_npu_state_layout_workaround else state_v_first
    triton_initial_state = initial_state
    if use_npu_state_layout_workaround and initial_state is not None:
        triton_initial_state = initial_state.transpose(-1, -2).contiguous()

    o, final_state = _kda_fwd_h_o_triton(
        kg=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        qg=qg,
        Aqk=Aqk,
        scale=scale,
        initial_state=triton_initial_state,
        output_final_state=output_final_state,
        state_v_first=triton_state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
    )

    if use_npu_state_layout_workaround and final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    return o, final_state



def _validate_chunk_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    chunk_size: int,
    state_v_first: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: float | None,
) -> None:
    if torch.is_grad_enabled():
        raise RuntimeError("Ascend chunk_kda only supports inference/no-grad mode")
    if chunk_size != 16:
        raise ValueError(f"Ascend chunk_kda requires chunk_size=16, got {chunk_size}")
    supported_dtypes = (torch.bfloat16, torch.float16)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if tensor.dtype not in supported_dtypes:
            raise ValueError(
                f"Ascend chunk_kda requires {name} dtype to be bf16 or fp16, "
                f"got {tensor.dtype}"
            )
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 4:
        raise ValueError("q, k, v, and g must be 4D tensors in [B, T, H, D] layout")
    if beta.ndim != 3:
        raise ValueError("beta must be a 3D tensor in [B, T, HV] layout")

    B, T, H, K = q.shape
    Bk, Tk, Hk, Kk = k.shape
    Bv, Tv, HV, V = v.shape
    if (Bk, Tk, Hk, Kk) != (B, T, H, K):
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}")
    if (Bv, Tv) != (B, T):
        raise ValueError("v must share B and T dimensions with q/k")
    if g.shape != (B, T, HV, K):
        raise ValueError(f"g must have shape {(B, T, HV, K)}, got {tuple(g.shape)}")
    if beta.shape != (B, T, HV):
        raise ValueError(f"beta must have shape {(B, T, HV)}, got {tuple(beta.shape)}")
    if K not in (64, 128, 192, 256):
        raise ValueError(
            f"Ascend chunk_kda requires K in {{64, 128, 192, 256}}, got {K}"
        )
    if V <= 0:
        raise ValueError(f"Ascend chunk_kda requires V > 0, got {V}")
    if HV < H or HV % H != 0:
        raise ValueError(f"Ascend chunk_kda requires HV % H == 0, got H={H}, HV={HV}")

    if not use_qk_l2norm_in_kernel:
        raise ValueError("Ascend chunk_kda requires use_qk_l2norm_in_kernel=True")
    if not use_gate_in_kernel:
        raise ValueError("Ascend chunk_kda requires use_gate_in_kernel=True")
    if not use_beta_sigmoid_in_kernel:
        raise ValueError("Ascend chunk_kda requires use_beta_sigmoid_in_kernel=True")
    if allow_neg_eigval:
        raise ValueError("Ascend chunk_kda does not support allow_neg_eigval=True")
    if not safe_gate:
        raise ValueError("Ascend chunk_kda requires safe_gate=True")
    if lower_bound is None:
        raise ValueError("Ascend chunk_kda requires lower_bound")
    if A_log is None:
        raise ValueError("Ascend chunk_kda requires A_log")
    if A_log.device != q.device:
        raise ValueError("A_log must be on the same device as q")
    if A_log.numel() != HV:
        raise ValueError(f"A_log.numel() must be HV={HV}, got {A_log.numel()}")
    if dt_bias is None:
        raise ValueError("Ascend chunk_kda requires dt_bias")
    if dt_bias.device != q.device:
        raise ValueError("dt_bias must be on the same device as q")
    if dt_bias.numel() != HV * K:
        raise ValueError(
            f"dt_bias.numel() must be HV*K={HV * K}, got {dt_bias.numel()}"
        )

    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1D tensor")
        if cu_seqlens.dtype != torch.long:
            raise ValueError("cu_seqlens must have dtype torch.long")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same device as q")
        if B != 1:
            raise ValueError("cu_seqlens packed varlen inputs must use B=1")

    if initial_state is not None:
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
        expected_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
        if tuple(initial_state.shape) != expected_shape:
            raise ValueError(
                f"initial_state must have shape {expected_shape}, "
                f"got {tuple(initial_state.shape)}"
            )



def _ascend_native_forward_single(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    state_v_first,
    chunk_size,
    safe_gate,
    lower_bound,
    A_log,
    dt_bias,
    forward_impl,
):
    """Run one fixed-layout sequence, splitting long T to avoid 910B timeout."""
    max_t = int(os.environ.get("FLAG_ATTN_ASCEND_KDA_MAX_T", "1024"))
    if max_t < chunk_size or max_t % chunk_size:
        raise ValueError(
            "FLAG_ATTN_ASCEND_KDA_MAX_T must be a multiple of chunk_size "
            f"({chunk_size}), got {max_t}"
        )

    # The 910B workaround computes recurrent states in K/V layout because the
    # V/K kernel branch is numerically incorrect.  Keep that internal layout
    # across all long-sequence segments instead of transposing the full state
    # at every segment boundary.
    use_npu_state_layout_workaround = state_v_first and q.device.type == "npu"
    internal_state_v_first = (
        False if use_npu_state_layout_workaround else state_v_first
    )
    state = initial_state
    if use_npu_state_layout_workaround and state is not None:
        state = state.transpose(-1, -2).contiguous()

    if q.shape[1] <= max_t:
        output, state = forward_impl(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            state_v_first=internal_state_v_first,
            cu_seqlens=None,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        if use_npu_state_layout_workaround and state is not None:
            state = state.transpose(-1, -2).contiguous()
        return output, state

    outputs = []
    for start in range(0, q.shape[1], max_t):
        end = min(start + max_t, q.shape[1])
        output, state = forward_impl(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            scale=scale,
            initial_state=state,
            # A state is needed to carry the recurrence between sub-segments.
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            state_v_first=internal_state_v_first,
            cu_seqlens=None,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        outputs.append(output)
    if use_npu_state_layout_workaround and state is not None:
        state = state.transpose(-1, -2).contiguous()
    return torch.cat(outputs, dim=1), state if output_final_state else None
