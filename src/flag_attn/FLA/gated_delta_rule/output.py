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

"""Output stage, in portable Triton.

Each program computes one ``[BT, BV]`` output tile from the chunk-entry state
``h`` and the delta values ``v_new``:

    o_i = scale * gamma_i * (q_i @ h_c)                      inter-chunk
        + scale * sum_{j<=i} (q_i . k_j) * exp(G_i - G_j) * v_new_j

The causal mask is inclusive (``j <= i``) here, unlike the strictly lower
mask of the UT transform, because the current token's own delta contributes
to its output. Addressing is explicit pointer arithmetic with masks only.
"""

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices


@triton.jit(do_not_specialize=["T"])
def _gdn_fwd_output_kernel(
    q,
    k,
    v_new,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_hg = i_h // (H // Hg)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_tg = tl.program_id(1)
    else:
        bos = i_b * T
        i_tg = i_b * tl.cdiv(T, BT) + i_t

    o_t = tl.arange(0, BT)
    pos = i_t * BT + o_t
    m_t = pos < T
    row = bos + pos
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    # Walk K: accumulate both the inter-chunk projection and the q k^T scores.
    for i_k in range(tl.cdiv(K, BK)):
        off_k = i_k * BK + tl.arange(0, BK)
        m_k = off_k < K
        p_q = q + row[:, None] * (Hg * K) + i_hg * K + off_k[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        p_h = h + (i_tg.to(tl.int64) * H + i_h) * K * V + off_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_o += tl.dot(b_q, b_h)
        # [BK, BT] key tile, loaded transposed by swapping index broadcasting.
        p_kt = k + row[None, :] * (Hg * K) + i_hg * K + off_k[:, None]
        b_kt = tl.load(p_kt, mask=m_t[None, :] & m_k[:, None], other=0.0)
        b_A += tl.dot(b_q, b_kt)

    # Gate both terms: the inter-chunk state by gamma_i, the scores by
    # exp(G_i - G_j) under an inclusive causal mask.
    b_g = tl.load(g + row * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_o *= tl.exp(b_g)[:, None]
    m_causal = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = tl.where(m_causal, b_A * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)

    p_vn = v_new + row[:, None] * (H * V) + i_h * V + o_v[None, :]
    m_tv = m_t[:, None] & m_v[None, :]
    b_v = tl.load(p_vn, mask=m_tv, other=0.0)
    b_o = (b_o + tl.dot(b_A.to(b_v.dtype), b_v)) * scale

    p_o = o + row[:, None] * (H * V) + i_h * V + o_v[None, :]
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=m_tv)


def gdn_fwd_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the operator output ``o`` with the same layout as ``v_new``."""
    B, T, Hg, K = q.shape
    H, V = v_new.shape[2], v_new.shape[3]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v_new)
    BV = min(64, triton.next_power_of_2(V))
    grid = (triton.cdiv(V, BV), NT, (H if cu_seqlens is not None else B * H))
    _gdn_fwd_output_kernel[grid](
        q=q,
        k=k,
        v_new=v_new,
        h=h,
        g=g,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=min(64, triton.next_power_of_2(K)),
        BV=BV,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
    )
    return o
