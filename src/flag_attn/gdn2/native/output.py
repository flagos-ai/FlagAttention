# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GLA-style output composition for the native GDN2 fallback."""

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices

from ..triton_ops_helper import exp2


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config(
            {"BK": block_k, "BV": block_v},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_k in [32, 64]
        for block_v in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "HV", "STATE_V_FIRST"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        num_chunks = tl.cdiv(T, BT)
        i_tg = (i_b * num_chunks + i_t).to(tl.int64)
        bos = (i_b * T).to(tl.int64)

    causal_mask = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    output = tl.zeros([BT, BV], dtype=tl.float32)
    # Multi-stage pipelining of this reduction produces NaNs on Hopper with
    # upstream Triton 3.7 (for example BK=32, 4 warps and 3 stages).
    for i_k in tl.range(tl.cdiv(K, BK), num_stages=1):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(
                h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
            )
        else:
            p_h = tl.make_block_ptr(
                h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
            )

        q_block = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        g_block = tl.load(p_g, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        qg = (q_block * exp2(g_block)).to(q_block.dtype)
        state = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        if STATE_V_FIRST:
            output += tl.dot(qg, tl.trans(state).to(qg.dtype))
        else:
            output += tl.dot(qg, state.to(qg.dtype))

    output *= scale
    p_v = tl.make_block_ptr(
        v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_A = tl.make_block_ptr(
        A, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    values = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    scores = tl.load(p_A, boundary_check=(0, 1), padding_option="zero")
    scores = tl.where(causal_mask, scores, 0.0).to(values.dtype)
    output += tl.dot(scores, values)
    tl.store(p_o, output.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    num_chunks = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    output = torch.zeros_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), num_chunks, B * HV)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=output,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        STATE_V_FIRST=state_v_first,
    )
    return output
