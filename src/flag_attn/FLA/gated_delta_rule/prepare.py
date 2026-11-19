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

"""Chunk prologue: gate cumsum, the UT transform, and the w/u projections.

Portable Triton only. Addressing is explicit pointer arithmetic with masks
rather than block pointers or tensor descriptors, so the kernel compiles on
any Triton backend that provides ``tl.load``/``tl.store``, ``tl.dot``,
``tl.trans``, ``tl.cumsum`` and the usual elementwise and reduction builtins.

For one chunk of ``BT`` tokens (local index ``i``, gate prefix sum
``G_i = sum_{j<=i} g_j``, ``gamma_i = exp(G_i)``) the strictly lower matrix

    M[i, j] = beta_i * (k_i . k_j) * exp(G_i - G_j)        for j < i

defines the in-chunk coupling, and ``Tm = (I + M)^-1`` is the UT transform.
Because ``M`` is nilpotent (``M^BT = 0``), the inverse is the terminating
product

    (I + M)^-1 = (I - N)^-1 = prod_{p=0}^{log2(BT)-1} (I + N^(2^p)),  N = -M

which this kernel evaluates with ``log2(BT)`` squaring steps instead of a
serial ``BT``-step forward substitution. It then applies the transform to
produce the two projections consumed downstream:

    w = Tm @ (beta * gamma * k)        u = Tm @ (beta * v)
"""

import os

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices

# Precision of the squaring chain that builds the triangular inverse. Plain
# "tf32" drops to ~4e-5 relative error here, which the solve amplifies; "ieee"
# holds ~1e-8 but has no tensor-core path and measured ~36x slower on sm_100.
# "tf32x3" reaches the same ~1e-8 at tensor-core speed, so it is the default,
# with the env var left as an escape hatch for backends that lack it.
_TRIL_PRECISION = os.environ.get("FLAG_ATTN_GDN_TRIL_PRECISION", "tf32x3")
if _TRIL_PRECISION not in ("ieee", "tf32", "tf32x3"):
    raise ValueError(
        "FLAG_ATTN_GDN_TRIL_PRECISION must be one of 'ieee', 'tf32', 'tf32x3'; "
        f"got {_TRIL_PRECISION!r}"
    )
_DOT_PRECISION: tl.constexpr = tl.constexpr(_TRIL_PRECISION)


@triton.jit
def _invert_unit_lower(b_M, m_diag, LOG_BT: tl.constexpr):
    """Return ``(I + b_M)^-1`` for a strictly lower triangular ``b_M``.

    Uses the terminating Neumann product ``prod_p (I + N^(2^p))`` with
    ``N = -b_M``, which costs ``LOG_BT`` squaring steps rather than ``BT``
    serial substitutions.
    """
    b_n = -b_M
    b_inv = m_diag + b_n
    b_pow = tl.dot(b_n, b_n, input_precision=_DOT_PRECISION)
    for _ in tl.static_range(LOG_BT - 1):
        b_inv = b_inv + tl.dot(b_inv, b_pow, input_precision=_DOT_PRECISION)
        b_pow = tl.dot(b_pow, b_pow, input_precision=_DOT_PRECISION)
    return b_inv


@triton.jit(do_not_specialize=["T"])
def _gdn_fwd_prepare_kernel(
    g_in,
    k,
    v,
    beta,
    g_out,
    w,
    u,
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
    LOG_BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    i_hg = i_h // (H // Hg)

    o_t = tl.arange(0, BT)
    pos = i_t * BT + o_t
    m_t = pos < T
    row = bos + pos

    # Gate prefix sum inside the chunk, kept and published in FP32.
    b_g = tl.load(g_in + row * H + i_h, mask=m_t, other=0.0).to(tl.float32)
    b_g = tl.cumsum(b_g, axis=0)
    tl.store(g_out + row * H + i_h, b_g, mask=m_t)
    b_beta = tl.load(beta + row * H + i_h, mask=m_t, other=0.0).to(tl.float32)

    # M = tril_strict(beta_i * (k_i . k_j) * exp(G_i - G_j)); the beta and gate
    # factors are rank-one in (i, j), so they apply after the K reduction.
    b_M = tl.zeros([BT, BT], dtype=tl.float32)
    o_k = tl.arange(0, BK)
    for i_k in range(tl.cdiv(K, BK)):
        off_k = i_k * BK + o_k
        m_kv = m_t[:, None] & (off_k < K)[None, :]
        b_k = tl.load(k + row[:, None] * (Hg * K) + i_hg * K + off_k[None, :], mask=m_kv, other=0.0)
        b_M += tl.dot(b_k, tl.trans(b_k))

    b_M = b_M * b_beta[:, None] * tl.exp(b_g[:, None] - b_g[None, :])
    m_strict = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_M = tl.where(m_strict, b_M, 0.0)

    m_diag = ((o_t[:, None] == o_t[None, :]) & m_t[:, None]).to(tl.float32)
    b_Tm = _invert_unit_lower(b_M, m_diag, LOG_BT)
    b_Tm = tl.where(m_t[:, None] & m_t[None, :], b_Tm, 0.0)

    # u = Tm @ (beta * v)
    b_gamma = tl.exp(b_g)
    o_v = tl.arange(0, BV)
    b_Tm_lp = b_Tm.to(v.dtype.element_ty)
    for i_v in range(tl.cdiv(V, BV)):
        off_v = i_v * BV + o_v
        mask_v = m_t[:, None] & (off_v < V)[None, :]
        p_v = v + row[:, None] * (H * V) + i_h * V + off_v[None, :]
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_u = tl.dot(b_Tm_lp, (b_v * b_beta[:, None]).to(v.dtype.element_ty))
        tl.store(u + row[:, None] * (H * V) + i_h * V + off_v[None, :], b_u.to(u.dtype.element_ty), mask=mask_v)

    # w = Tm @ (beta * gamma * k)
    b_Tm_k = b_Tm.to(k.dtype.element_ty)
    for i_k in range(tl.cdiv(K, BK)):
        off_k = i_k * BK + o_k
        mask_k = m_t[:, None] & (off_k < K)[None, :]
        p_k = k + row[:, None] * (Hg * K) + i_hg * K + off_k[None, :]
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_kb = (b_k * (b_beta * b_gamma)[:, None]).to(k.dtype.element_ty)
        b_w = tl.dot(b_Tm_k, b_kb)
        tl.store(w + row[:, None] * (H * K) + i_h * K + off_k[None, :], b_w.to(w.dtype.element_ty), mask=mask_k)


def gdn_fwd_prepare(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(g_cumsum, w, u)`` for the chunked forward.

    ``g_cumsum`` is the within-chunk prefix sum of the log gates in FP32;
    ``w`` and ``u`` are the UT-transformed key and value projections.
    """
    B, T, Hg, K = k.shape
    H, V = v.shape[2], v.shape[3]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g_cumsum = torch.empty(B, T, H, device=g.device, dtype=torch.float32)
    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)

    # One program per (chunk, batch*head); varlen packs the batch into T, so
    # chunk_indices already carries the sequence id and i_b is unused there.
    grid = (NT, (H if cu_seqlens is not None else B * H))
    _gdn_fwd_prepare_kernel[grid](
        g_in=g,
        k=k,
        v=v,
        beta=beta,
        g_out=g_cumsum,
        w=w,
        u=u,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=min(64, triton.next_power_of_2(K)),
        BV=min(64, triton.next_power_of_2(V)),
        LOG_BT=BT.bit_length() - 1,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
    )
    return g_cumsum, w, u
