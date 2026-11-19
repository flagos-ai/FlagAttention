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

"""WY-representation projections from a materialized UT transform.

The portable forward folds this work into its prepare kernel, which keeps the
transform in registers. This module recomputes the same projections from an
explicit ``A`` matrix for callers that hold one:

    w = A @ (beta * exp(G) * k)        u = A @ (beta * v)

Portable Triton only: explicit pointer arithmetic with masks.
"""

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices


@triton.jit(do_not_specialize=["T"])
def _recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_hg = i_h // (H // Hg)

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    o_t = tl.arange(0, BT)
    pos = i_t * BT + o_t
    m_t = pos < T
    row = bos + pos

    b_beta = tl.load(beta + row * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_g = tl.load(g + row * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    p_A = A + row[:, None] * (H * BT) + i_h * BT + o_t[None, :]
    b_A = tl.load(p_A, mask=m_t[:, None] & m_t[None, :], other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        off_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (off_v < V)[None, :]
        p_v = v + row[:, None] * (H * V) + i_h * V + off_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0).to(tl.float32)
        b_u = tl.dot(b_A, (b_v * b_beta[:, None]).to(b_A.dtype))
        p_u = u + row[:, None] * (H * V) + i_h * V + off_v[None, :]
        tl.store(p_u, b_u.to(u.dtype.element_ty), mask=m_tv)

    b_scale = b_beta * tl.exp(b_g)
    for i_k in range(tl.cdiv(K, BK)):
        off_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (off_k < K)[None, :]
        p_k = k + row[:, None] * (Hg * K) + i_hg * K + off_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0).to(tl.float32)
        b_w = tl.dot(b_A, (b_k * b_scale[:, None]).to(b_A.dtype))
        p_w = w + row[:, None] * (H * K) + i_h * K + off_k[None, :]
        tl.store(p_w, b_w.to(w.dtype.element_ty), mask=m_tk)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    block_v: int | None = None,
    pipe_stages: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(w, u)`` from a materialized UT transform ``A``."""
    del pipe_stages  # Scheduling hint of the Hopper route; unused here.
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = A.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    grid = (NT, (H if cu_seqlens is not None else B * H))
    _recompute_w_u_fwd_kernel[grid](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=min(64, triton.next_power_of_2(K)),
        BV=min(block_v or 64, triton.next_power_of_2(V)),
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
    )
    return w, u
