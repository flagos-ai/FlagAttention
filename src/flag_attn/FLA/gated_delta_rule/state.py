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

"""Inter-chunk state recurrence, in portable Triton.

One program owns the full ``K`` extent and a ``BV`` slice of the value
dimension for one (sequence, value head), and walks the chunks in order, so
the sequential dependency stays in registers and the ``v_new`` reduction over
``K`` is local to the program (no cross-program ordering is required). For
chunk ``c`` with gate prefix sums ``G`` and chunk-final gate ``G_last``:

    h_c     = state entering the chunk                 (published for stage 3)
    v_new   = u - w @ h_c                              (the delta values)
    h_{c+1} = exp(G_last) * h_c + k^T @ (v_new * exp(G_last - G))

``v_new`` is published too, because stage 3 needs it for the intra-chunk term.
Addressing is explicit pointer arithmetic with masks only.
"""

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_offsets


@triton.jit(do_not_specialize=["T"])
def _gdn_fwd_state_kernel(
    k,
    w,
    u,
    g,
    h,
    v_new,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    i_hg = i_h // (H // Hg)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos = i_n * T
        boh = i_n * tl.cdiv(T, BT)
    NT = tl.cdiv(T, BT)

    # BK spans the whole key dimension, so the w @ h reduction is program-local.
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh.to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_kv, other=0.0).to(tl.float32)

    o_t = tl.arange(0, BT)
    for i_t in range(NT):
        pos = i_t * BT + o_t
        m_t = pos < T
        row = bos + pos
        m_tv = m_t[:, None] & m_v[None, :]

        # Publish the state entering this chunk for the output stage.
        p_h = h + ((boh + i_t).to(tl.int64) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_h.to(h.dtype.element_ty), mask=m_kv)

        # v_new = u - w @ h_c
        p_w = w + row[:, None] * (H * K) + i_h * K + o_k[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_acc = tl.dot(b_w, b_h.to(w.dtype.element_ty))

        p_u = u + row[:, None] * (H * V) + i_h * V + o_v[None, :]
        b_v = tl.load(p_u, mask=m_tv, other=0.0).to(tl.float32) - b_acc
        p_vn = v_new + row[:, None] * (H * V) + i_h * V + o_v[None, :]
        tl.store(p_vn, b_v.to(v_new.dtype.element_ty), mask=m_tv)

        # Gate the carried state, then fold in this chunk's contribution.
        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_g = tl.load(g + row * H + i_h, mask=m_t, other=0.0).to(tl.float32)
        b_g_last = tl.load(g + (bos + last) * H + i_h).to(tl.float32)
        b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0.0)[:, None]
        b_h *= tl.exp(b_g_last)

        # [BK, BT] key tile, loaded transposed by swapping index broadcasting.
        p_kt = k + row[None, :] * (Hg * K) + i_hg * K + o_k[:, None]
        b_kt = tl.load(p_kt, mask=m_t[None, :] & m_k[:, None], other=0.0)
        b_h += tl.dot(b_kt, b_v.to(k.dtype.element_ty))

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh.to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(ht.dtype.element_ty), mask=m_kv)


def gdn_fwd_state(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return ``(h, v_new, final_state)``.

    ``h[:, c]`` is the state entering chunk ``c``, ``v_new`` holds the delta
    values, and ``final_state`` is the FP32 state after the last token.
    """
    B, T, Hg, K = k.shape
    H, V = u.shape[2], u.shape[3]
    BT = chunk_size

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        NT = int(chunk_offsets[-1])

    h = k.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    # The state tile is [BK, BV] in registers with BK covering all of K, so
    # keep BV modest for wide keys to bound register pressure.
    BK = triton.next_power_of_2(K)
    BV = 64 if K <= 128 else 32
    BV = min(BV, triton.next_power_of_2(V))
    grid = (triton.cdiv(V, BV), N * H)
    _gdn_fwd_state_kernel[grid](
        k=k,
        w=w,
        u=u,
        g=g,
        h=h,
        v_new=v_new,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4 if K <= 128 else 8,
    )
    return h, v_new, final_state
