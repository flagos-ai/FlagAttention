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

"""BT=16 inference kernels for KDA prefill.

Three forward paths with identical semantics share one public entry, ``chunk_kda``:

* Strict TLE path (``chunk_kda_fwd_infer_strict_tle``): optimized for the
  production KDA shape and its narrower input constraints.
* Generic TLE path (``chunk_kda_fwd_infer``): TMA-accelerated,
  warp-specialized fused kernels for the wider supported input set.
* Triton fallback (``chunk_kda_fwd_infer_triton``): portable plain-Triton
  kernels used when TLE is unavailable (e.g. CI).

``chunk_kda`` validates inputs first, then dispatches: strict TLE when its
constraints are satisfied, generic TLE otherwise, and finally Triton. Set
``FLAGGEMS_CHUNK_KDA_BACKEND`` to ``strict_tle``, ``tle``, or ``triton`` to
force a backend; the default is ``auto``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from flag_attn.gated_delta_rule.compat import has_triton_tle

from ..index import prepare_chunk_indices, prepare_chunk_offsets

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE_KDA = True
    except ImportError:
        tle = None
        HAS_TLE_KDA = False
else:
    tle = None
    HAS_TLE_KDA = False

# =============================================================================
# nv_mma_shared_layout: CUDA uses NV MMA shared layout; MUSA does not support it.
# =============================================================================

_NV_MMA_SHARED_LAYOUT: bool | None = None


def _nv_mma_shared_layout() -> bool:
    """True on CUDA devices, False on MUSA (and any non-NV device).

    Resolved lazily without depending on the old repository's global device.
    """
    global _NV_MMA_SHARED_LAYOUT
    if _NV_MMA_SHARED_LAYOUT is None:
        musa_available = hasattr(torch, "musa") and torch.musa.is_available()
        _NV_MMA_SHARED_LAYOUT = torch.cuda.is_available() and not musa_available
    return _NV_MMA_SHARED_LAYOUT


__all__ = ["chunk_kda"]

# =============================================================================
# Shared helpers
# =============================================================================

RCP_LN2 = 1.4426950216
_BACKEND_ENV = "FLAGGEMS_CHUNK_KDA_BACKEND"
_BACKEND_ALIASES = {
    "auto": "auto",
    "strict": "strict_tle",
    "strict_tle": "strict_tle",
    "tle": "tle",
    "generic_tle": "tle",
    "triton": "triton",
    "triton_fuse": "triton",
}


def _chunk_kda_backend() -> str:
    value = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    try:
        return _BACKEND_ALIASES[value]
    except KeyError as exc:
        choices = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(f"invalid {_BACKEND_ENV}={value!r}; expected one of: {choices}") from exc


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


_FP16_DOT_PRECISION = tl.constexpr("ieee")
_FP16_DOT_PRECISION_REFRESHED = False


def _refresh_fp16_dot_precision() -> None:
    global _FP16_DOT_PRECISION, _FP16_DOT_PRECISION_REFRESHED
    if _FP16_DOT_PRECISION_REFRESHED:
        return
    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
            if major >= 8:
                _FP16_DOT_PRECISION = tl.constexpr("tf32")
    except Exception:
        _FP16_DOT_PRECISION = tl.constexpr("ieee")
    _FP16_DOT_PRECISION_REFRESHED = True


def _allocate_triton_workspace(size: int, _alignment: int, _stream) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


# =============================================================================
# TLE path (fused, TMA + warp-specialized) -- default when available
# =============================================================================

if HAS_TLE_KDA:

    @triton.heuristics(
        {
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=num_warps, num_stages=num_stages, maxnreg=maxnreg)
            for num_warps in [2, 4, 8]
            for num_stages in [2, 4, 8]
            for maxnreg in [None, 32, 64, 72]
        ],
        key=["H", "HV", "K", "BT"],
    )
    @triton.jit(do_not_specialize=["T"])
    def _kda_fwd_intra_kernel(
        q,
        k,
        g,
        beta,
        ws,
        Aqk,
        Akk,
        g_out,
        A_log,
        dt_bias,
        lower_bound,
        scale,
        g_scale,
        l2norm_eps,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        NV_MMA_SHARED_LAYOUT: tl.constexpr,
    ):
        i_t, i_bh = tl.program_id(0), tl.program_id(1)
        i_hv = i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
        else:
            bos = i_bh // HV * T

        if i_t * BT >= T:
            return

        q += (bos * H + i_h) * K
        k += (bos * H + i_h) * K
        g += (bos * HV + i_hv) * K
        g_out += (bos * HV + i_hv) * K
        Aqk += (bos * HV + i_hv) * BT
        Akk += (bos * HV + i_hv) * BT
        ws += (bos * HV + i_hv) * 3 * K
        beta += bos * HV + i_hv

        o_i = tl.arange(0, BT)
        o_c = i_t * BT + o_i
        m_c = o_c < T

        # Reuse q/k/g cumsum from shared memory across the intra-chunk phases.
        q_buf = tle.gpu.alloc([BT, K], dtype=q.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        k_buf = tle.gpu.alloc([BT, K], dtype=k.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        gc_buf = tle.gpu.alloc([BT, K], dtype=tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)

        rows = tl.broadcast_to(tl.arange(0, BT)[:, None], (BT, K))
        cols = tl.broadcast_to(tl.arange(0, K)[None, :], (BT, K))
        q_sp = tle.gpu.local_ptr(q_buf, (rows, cols))
        k_sp = tle.gpu.local_ptr(k_buf, (rows, cols))
        gc_sp = tle.gpu.local_ptr(gc_buf, (rows, cols))

        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        b_q = tle.load(p_q, boundary_check=(0, 1), is_async=True)
        b_k = tle.load(p_k, boundary_check=(0, 1), is_async=True)
        tl.store(q_sp, b_q)
        tl.store(k_sp, b_k)

        b_qf = b_q.to(tl.float32)
        b_kf = b_k.to(tl.float32)

        b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
        b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

        b_g = tle.load(p_g, boundary_check=(0, 1), is_async=True).to(tl.float32)
        b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)
        p_dt = tl.make_block_ptr(dt_bias + i_hv * K, (K,), (1,), (0,), (K,), (0,))
        b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
        b_g = b_g + b_bias[None, :]
        # FlashKDA-compatible safe gate; lower_bound is required by the verifier.
        b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
        tl.store(gc_sp, b_g)
        one_row = tl.broadcast_to(tl.arange(0, 1)[:, None], (1, K))
        col_row = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
        b_acc = tl.zeros([1, K], dtype=tl.float32)
        for r in tl.static_range(BT):
            rp = tle.gpu.local_ptr(gc_buf, (tl.broadcast_to(one_row + r, (1, K)), col_row))
            b_acc = b_acc + tl.load(rp)
            tl.store(rp, b_acc)

        p_g_out = tl.make_block_ptr(g_out, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        tl.store(p_g_out, tl.load(gc_sp).to(g_out.dtype.element_ty), boundary_check=(0, 1))

        # Intra-chunk Aqk/Akk plus triangular solve.
        b_gq = tl.where(m_c[:, None], exp2(tl.load(gc_sp)), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-tl.load(gc_sp)), 0.0)

        # Keep b_gq/b_gk in fp32: exp2(±cumsum) can exceed fp16 max (65504), casting would overflow.
        # For bfloat16, bf16 range (3.4e38) is sufficient, so cast is safe.
        if q.dtype.element_ty == tl.float16:
            b_kgt = tl.trans(b_kf * b_gk)
            b_Aqk = tl.dot(
                b_qf * b_gq,
                b_kgt,
                input_precision=_FP16_DOT_PRECISION,
                out_dtype=tl.float32,
            )
            b_Akk = tl.dot(
                b_kf * b_gq,
                b_kgt,
                input_precision=_FP16_DOT_PRECISION,
                out_dtype=tl.float32,
            )
        else:
            b_kgt = tl.trans(b_kf * b_gk).to(b_k.dtype)
            b_Aqk = tl.dot(
                (b_qf * b_gq).to(b_q.dtype),
                b_kgt,
                input_precision=_FP16_DOT_PRECISION,
                out_dtype=tl.float32,
            )
            b_Akk = tl.dot(
                (b_kf * b_gq).to(b_k.dtype),
                b_kgt,
                input_precision=_FP16_DOT_PRECISION,
                out_dtype=tl.float32,
            )

        b_Aqk = b_Aqk * b_q_rstd[:, None] * b_k_rstd[None, :]
        b_Akk = b_Akk * b_k_rstd[:, None] * b_k_rstd[None, :]

        p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_beta = tl.sigmoid(tl.load(p_beta, boundary_check=(0,)).to(tl.float32))

        m_Aqk = o_i[:, None] >= o_i[None, :]
        m_Akk = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
        b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

        p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        b_L = b_Akk.to(tl.float16)
        b_Ai = m_I.to(tl.float16) - b_L
        b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
        b_Ai = b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
        b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
        b_Ai = b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
        b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
        b_Ai = b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float16)

        p_Akk_out = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_Akk_out, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))

        # Pack w, qg, and kg into one workspace at columns 0, K, and 2*K.
        b_k3 = tl.load(k_sp).to(tl.float32) * b_k_rstd[:, None]
        b_gk3 = tl.load(gc_sp)
        b_kb = b_k3 * b_beta[:, None] * exp2(b_gk3)
        p_w = tl.make_block_ptr(ws, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        tl.store(p_w, b_kb.to(ws.dtype.element_ty), boundary_check=(0, 1))

        b_q3 = tl.load(q_sp).to(tl.float32) * b_q_rstd[:, None]
        b_qg_val = b_q3 * exp2(b_gk3)
        p_qg = tl.make_block_ptr(ws, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, K), (BT, K), (1, 0))
        tl.store(p_qg, b_qg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))

        last_local = tl.minimum(BT, T - i_t * BT) - 1
        gn_rows = tl.broadcast_to(last_local + tl.zeros([1, K], dtype=tl.int32), (1, K))
        gn_cols = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
        b_gn = tl.load(tle.gpu.local_ptr(gc_buf, (gn_rows, gn_cols)))
        b_kg_val = b_k3 * tl.where(m_c[:, None], exp2(b_gn - b_gk3), 0)
        p_kg = tl.make_block_ptr(ws, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 2 * K), (BT, K), (1, 0))
        tl.store(p_kg, b_kg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))

    def _kda_fwd_intra(
        q,
        k,
        g,
        beta,
        scale,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=16,
        lower_bound=None,
        A_log=None,
        dt_bias=None,
    ):
        B, T_len, H, K = q.shape
        HV = g.shape[2]
        BT = chunk_size

        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)
        grid = (NT, B * HV)

        # Pad the workspace T dimension so TMA descriptors can read full BT tiles.
        T_padded = NT * BT
        g_out = torch.empty(B, T_padded, HV, K, device=q.device, dtype=torch.float32)
        ws = torch.empty(B, T_padded, HV, 3 * K, device=q.device, dtype=q.dtype)
        Aqk = torch.empty(B, T_padded, HV, BT, device=q.device, dtype=q.dtype)
        Akk = torch.zeros(B, T_padded, HV, BT, device=q.device, dtype=q.dtype)

        _kda_fwd_intra_kernel[grid](
            q=q,
            k=k,
            g=g,
            beta=beta,
            ws=ws,
            Aqk=Aqk,
            Akk=Akk,
            g_out=g_out,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            scale=scale,
            g_scale=RCP_LN2,
            l2norm_eps=1e-6,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T_len,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            NV_MMA_SHARED_LAYOUT=_nv_mma_shared_layout(),
        )
        return ws, Aqk, Akk, g_out

    # -----------------------------------------------------------------------------
    # Kernel 2: state propagation + output
    # -----------------------------------------------------------------------------

    # @triton.jit
    # def _kda_state_output_load_producer(
    #     writer,
    #     ws_desc,
    #     v_ptr,
    #     beta_ptr,
    #     gk_desc,
    #     Aqk_desc,
    #     Akk_desc,
    #     K,
    #     T,
    #     HV: tl.constexpr,
    #     V: tl.constexpr,
    #     BT: tl.constexpr,
    #     BV: tl.constexpr,
    #     NT,
    #     i_v,
    # ):
    #     for i_t in tl.range(NT):
    #         slot = writer.acquire(i_t)
    #         last_idx = tl.minimum(i_t * BT + BT, T) - 1

    #         # The workspace packs w, qg, and kg at offsets 0, K, and 2*K.
    #         tle.gpu.copy(ws_desc, slot.w1, [BT, 64], [i_t * BT, 0])
    #         tle.gpu.copy(ws_desc, slot.qg1, [BT, 64], [i_t * BT, K])
    #         tle.gpu.copy(ws_desc, slot.kg1, [BT, 64], [i_t * BT, 2 * K])
    #         tle.gpu.copy(Aqk_desc, slot.Aqk, [BT, BT], [i_t * BT, 0])
    #         tle.gpu.copy(Akk_desc, slot.Akk, [BT, BT], [i_t * BT, 0])
    #         tle.gpu.copy(gk_desc, slot.gk1, [1, 64], [last_idx, 0])
    #         if K > 64:
    #             tle.gpu.copy(ws_desc, slot.w2, [BT, 64], [i_t * BT, 64])
    #             tle.gpu.copy(ws_desc, slot.qg2, [BT, 64], [i_t * BT, K + 64])
    #             tle.gpu.copy(ws_desc, slot.kg2, [BT, 64], [i_t * BT, 2 * K + 64])
    #             tle.gpu.copy(gk_desc, slot.gk2, [1, 64], [last_idx, 64])
    #         if K > 128:
    #             tle.gpu.copy(ws_desc, slot.w3, [BT, 64], [i_t * BT, 128])
    #             tle.gpu.copy(ws_desc, slot.qg3, [BT, 64], [i_t * BT, K + 128])
    #             tle.gpu.copy(ws_desc, slot.kg3, [BT, 64], [i_t * BT, 2 * K + 128])
    #             tle.gpu.copy(gk_desc, slot.gk3, [1, 64], [last_idx, 128])
    #         if K > 192:
    #             tle.gpu.copy(ws_desc, slot.w4, [BT, 64], [i_t * BT, 192])
    #             tle.gpu.copy(ws_desc, slot.qg4, [BT, 64], [i_t * BT, K + 192])
    #             tle.gpu.copy(ws_desc, slot.kg4, [BT, 64], [i_t * BT, 2 * K + 192])
    #             tle.gpu.copy(gk_desc, slot.gk4, [1, 64], [last_idx, 192])

    #         # v and beta need elementwise work, so keep them as regular loads.
    #         p_v = tl.make_block_ptr(
    #             v_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    #         )
    #         p_beta = tl.make_block_ptr(beta_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    #         b_v = tl.load(p_v, boundary_check=(0, 1))
    #         b_beta = tl.load(p_beta, boundary_check=(0,))
    #         b_beta_f = tl.sigmoid(b_beta.to(tl.float32))
    #         b_vb = (b_v.to(tl.float32) * b_beta_f[:, None]).to(b_v.dtype)
    #         tl.store(tle.gpu.local_ptr(slot.vb), b_vb)

    #         writer.commit(i_t)

    # @triton.jit
    # def _kda_state_output_mma_consumer(
    #     load_reader,
    #     store_writer,
    #     h0,
    #     ht,
    #     gk,
    #     scale,
    #     i_v,
    #     i_nh,
    #     T,
    #     HV: tl.constexpr,
    #     K: tl.constexpr,
    #     V: tl.constexpr,
    #     BT: tl.constexpr,
    #     BV: tl.constexpr,
    #     NT,
    #     kg_dtype: tl.constexpr,
    #     USE_INITIAL_STATE: tl.constexpr,
    #     STORE_FINAL_STATE: tl.constexpr,
    #     STATE_V_FIRST: tl.constexpr,
    # ):
    #     # Initial state.
    #     if USE_INITIAL_STATE:
    #         if STATE_V_FIRST:
    #             p_h0_1 = tl.make_block_ptr(
    #                 h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
    #             )
    #             b_h1 = tl.trans(tl.load(p_h0_1, boundary_check=(0, 1))).to(tl.float32)
    #             if K > 64:
    #                 p_h0_2 = tl.make_block_ptr(
    #                     h0 + i_nh * K * V,
    #                     (V, K),
    #                     (K, 1),
    #                     (i_v * BV, 64),
    #                     (BV, 64),
    #                     (1, 0),
    #                 )
    #                 b_h2 = tl.trans(tl.load(p_h0_2, boundary_check=(0, 1))).to(
    #                     tl.float32
    #                 )
    #             if K > 128:
    #                 p_h0_3 = tl.make_block_ptr(
    #                     h0 + i_nh * K * V,
    #                     (V, K),
    #                     (K, 1),
    #                     (i_v * BV, 128),
    #                     (BV, 64),
    #                     (1, 0),
    #                 )
    #                 b_h3 = tl.trans(tl.load(p_h0_3, boundary_check=(0, 1))).to(
    #                     tl.float32
    #                 )
    #             if K > 192:
    #                 p_h0_4 = tl.make_block_ptr(
    #                     h0 + i_nh * K * V,
    #                     (V, K),
    #                     (K, 1),
    #                     (i_v * BV, 192),
    #                     (BV, 64),
    #                     (1, 0),
    #                 )
    #                 b_h4 = tl.trans(tl.load(p_h0_4, boundary_check=(0, 1))).to(
    #                     tl.float32
    #                 )
    #         else:
    #             p_h0_1 = tl.make_block_ptr(
    #                 h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
    #             )
    #             b_h1 = tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
    #             if K > 64:
    #                 p_h0_2 = tl.make_block_ptr(
    #                     h0 + i_nh * K * V,
    #                     (K, V),
    #                     (V, 1),
    #                     (64, i_v * BV),
    #                     (64, BV),
    #                     (1, 0),
    #                 )
    #                 b_h2 = tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
    #             if K > 128:
    #                 p_h0_3 = tl.make_block_ptr(
    #                     h0 + i_nh * K * V,
    #                     (K, V),
    #                     (V, 1),
    #                     (128, i_v * BV),
    #                     (64, BV),
    #                     (1, 0),
    #                 )
    #                 b_h3 = tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
    #             if K > 192:
    #                 p_h0_4 = tl.make_block_ptr(
    #                     h0 + i_nh * K * V,
    #                     (K, V),
    #                     (V, 1),
    #                     (192, i_v * BV),
    #                     (64, BV),
    #                     (1, 0),
    #                 )
    #                 b_h4 = tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
    #     else:
    #         b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    #         if K > 64:
    #             b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    #         if K > 128:
    #             b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    #         if K > 192:
    #             b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    #     # Sequential recurrence across chunks.
    #     for i_t in tl.range(NT):
    #         wait = load_reader.wait(i_t)
    #         slot = wait.slot

    #         b_w1 = tl.load(tle.gpu.local_ptr(slot.w1))
    #         b_vb = tl.load(tle.gpu.local_ptr(slot.vb))
    #         b_qg1 = tl.load(tle.gpu.local_ptr(slot.qg1))
    #         b_kg1 = tl.load(tle.gpu.local_ptr(slot.kg1))
    #         b_Aqk = tl.load(tle.gpu.local_ptr(slot.Aqk))
    #         b_Akk = tl.load(tle.gpu.local_ptr(slot.Akk))
    #         b_gk1 = tl.load(tle.gpu.local_ptr(slot.gk1)).reshape([64])
    #         if K > 64:
    #             b_w2 = tl.load(tle.gpu.local_ptr(slot.w2))
    #             b_qg2 = tl.load(tle.gpu.local_ptr(slot.qg2))
    #             b_kg2 = tl.load(tle.gpu.local_ptr(slot.kg2))
    #             b_gk2 = tl.load(tle.gpu.local_ptr(slot.gk2)).reshape([64])
    #         if K > 128:
    #             b_w3 = tl.load(tle.gpu.local_ptr(slot.w3))
    #             b_qg3 = tl.load(tle.gpu.local_ptr(slot.qg3))
    #             b_kg3 = tl.load(tle.gpu.local_ptr(slot.kg3))
    #             b_gk3 = tl.load(tle.gpu.local_ptr(slot.gk3)).reshape([64])
    #         if K > 192:
    #             b_w4 = tl.load(tle.gpu.local_ptr(slot.w4))
    #             b_qg4 = tl.load(tle.gpu.local_ptr(slot.qg4))
    #             b_kg4 = tl.load(tle.gpu.local_ptr(slot.kg4))
    #             b_gk4 = tl.load(tle.gpu.local_ptr(slot.gk4)).reshape([64])

    #         b_h1_bf = b_h1.to(kg_dtype)
    #         if K > 64:
    #             b_h2_bf = b_h2.to(kg_dtype)
    #         if K > 128:
    #             b_h3_bf = b_h3.to(kg_dtype)
    #         if K > 192:
    #             b_h4_bf = b_h4.to(kg_dtype)

    #         # v_new = Akk_inv @ (v*beta - w @ h)
    #         b_kh = tl.dot(b_w1, b_h1_bf).to(tl.float32)
    #         if K > 64:
    #             b_kh += tl.dot(b_w2, b_h2_bf).to(tl.float32)
    #         if K > 128:
    #             b_kh += tl.dot(b_w3, b_h3_bf).to(tl.float32)
    #         if K > 192:
    #             b_kh += tl.dot(b_w4, b_h4_bf).to(tl.float32)
    #         b_diff = b_vb.to(tl.float32) - b_kh
    #         b_v = tl.dot(b_Akk, b_diff.to(kg_dtype)).to(tl.float32)

    #         # output = scale * qg @ h + Aqk @ v_new
    #         b_qh = tl.dot(b_qg1, b_h1_bf).to(tl.float32)
    #         if K > 64:
    #             b_qh += tl.dot(b_qg2, b_h2_bf).to(tl.float32)
    #         if K > 128:
    #             b_qh += tl.dot(b_qg3, b_h3_bf).to(tl.float32)
    #         if K > 192:
    #             b_qh += tl.dot(b_qg4, b_h4_bf).to(tl.float32)
    #         b_o = scale * b_qh
    #         b_v_cast = b_v.to(kg_dtype)
    #         b_o += tl.dot(b_Aqk, b_v_cast).to(tl.float32)

    #         out_slot = store_writer.acquire(i_t)
    #         tl.store(tle.gpu.local_ptr(out_slot.output), b_o)
    #         store_writer.commit(i_t)

    #         load_reader.release(i_t)

    #         # state decay + update: h = h * exp2(gk_last) + kg^T @ v_new
    #         b_h1 = b_h1 * exp2(b_gk1)[:, None] + tl.dot(tl.trans(b_kg1), b_v_cast).to(
    #             tl.float32
    #         )
    #         if K > 64:
    #             b_h2 = b_h2 * exp2(b_gk2)[:, None] + tl.dot(
    #                 tl.trans(b_kg2), b_v_cast
    #             ).to(tl.float32)
    #         if K > 128:
    #             b_h3 = b_h3 * exp2(b_gk3)[:, None] + tl.dot(
    #                 tl.trans(b_kg3), b_v_cast
    #             ).to(tl.float32)
    #         if K > 192:
    #             b_h4 = b_h4 * exp2(b_gk4)[:, None] + tl.dot(
    #                 tl.trans(b_kg4), b_v_cast
    #             ).to(tl.float32)

    #     # Final state.
    #     if STORE_FINAL_STATE:
    #         if STATE_V_FIRST:
    #             p_ht1 = tl.make_block_ptr(
    #                 ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
    #             )
    #             tl.store(
    #                 p_ht1,
    #                 tl.trans(b_h1).to(p_ht1.dtype.element_ty),
    #                 boundary_check=(0, 1),
    #             )
    #             if K > 64:
    #                 p_ht2 = tl.make_block_ptr(
    #                     ht + i_nh * K * V,
    #                     (V, K),
    #                     (K, 1),
    #                     (i_v * BV, 64),
    #                     (BV, 64),
    #                     (1, 0),
    #                 )
    #                 tl.store(
    #                     p_ht2,
    #                     tl.trans(b_h2).to(p_ht2.dtype.element_ty),
    #                     boundary_check=(0, 1),
    #                 )
    #             if K > 128:
    #                 p_ht3 = tl.make_block_ptr(
    #                     ht + i_nh * K * V,
    #                     (V, K),
    #                     (K, 1),
    #                     (i_v * BV, 128),
    #                     (BV, 64),
    #                     (1, 0),
    #                 )
    #                 tl.store(
    #                     p_ht3,
    #                     tl.trans(b_h3).to(p_ht3.dtype.element_ty),
    #                     boundary_check=(0, 1),
    #                 )
    #             if K > 192:
    #                 p_ht4 = tl.make_block_ptr(
    #                     ht + i_nh * K * V,
    #                     (V, K),
    #                     (K, 1),
    #                     (i_v * BV, 192),
    #                     (BV, 64),
    #                     (1, 0),
    #                 )
    #                 tl.store(
    #                     p_ht4,
    #                     tl.trans(b_h4).to(p_ht4.dtype.element_ty),
    #                     boundary_check=(0, 1),
    #                 )
    #         else:
    #             p_ht1 = tl.make_block_ptr(
    #                 ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
    #             )
    #             tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
    #             if K > 64:
    #                 p_ht2 = tl.make_block_ptr(
    #                     ht + i_nh * K * V,
    #                     (K, V),
    #                     (V, 1),
    #                     (64, i_v * BV),
    #                     (64, BV),
    #                     (1, 0),
    #                 )
    #                 tl.store(
    #                     p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1)
    #                 )
    #             if K > 128:
    #                 p_ht3 = tl.make_block_ptr(
    #                     ht + i_nh * K * V,
    #                     (K, V),
    #                     (V, 1),
    #                     (128, i_v * BV),
    #                     (64, BV),
    #                     (1, 0),
    #                 )
    #                 tl.store(
    #                     p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1)
    #                 )
    #             if K > 192:
    #                 p_ht4 = tl.make_block_ptr(
    #                     ht + i_nh * K * V,
    #                     (K, V),
    #                     (V, 1),
    #                     (192, i_v * BV),
    #                     (64, BV),
    #                     (1, 0),
    #                 )
    #                 tl.store(
    #                     p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1)
    #                 )

    # @triton.jit
    # def _kda_state_output_store_consumer(
    #     store_reader,
    #     o_ptr,
    #     T,
    #     HV: tl.constexpr,
    #     V: tl.constexpr,
    #     BT: tl.constexpr,
    #     BV: tl.constexpr,
    #     NT,
    #     i_v,
    # ):
    #     for i_t in tl.range(NT):
    #         store_wait = store_reader.wait(i_t)
    #         slot = store_wait.slot
    #         b_o = tl.load(tle.gpu.local_ptr(slot.output))
    #         p_o = tl.make_block_ptr(
    #             o_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    #         )
    #         tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    #         store_reader.release(i_t)

    # PIPE_STAGES = tl.constexpr(4)

    # @triton.heuristics(
    #     {
    #         "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
    #         "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
    #         "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    #     }
    # )
    # @triton.autotune(
    #     configs=[triton.Config({"BV": BV}, num_warps=4) for BV in [32, 64, 128]],
    #     key=[
    #         "HV",
    #         "K",
    #         "V",
    #         "BT",
    #     ],  # warp_specialize fixes num_warps=4, only BV is tuned
    # )
    # @triton.jit(do_not_specialize=["T"])
    # def _kda_fwd_state_output_kernel(
    #     kg,
    #     v,
    #     beta,
    #     gk,
    #     Aqk,
    #     Akk,
    #     o,
    #     ws,
    #     h0,
    #     ht,
    #     cu_seqlens,
    #     scale,
    #     T,
    #     HV: tl.constexpr,
    #     K: tl.constexpr,
    #     V: tl.constexpr,
    #     BT: tl.constexpr,
    #     BV: tl.constexpr,
    #     STATE_V_FIRST: tl.constexpr,
    #     USE_INITIAL_STATE: tl.constexpr,
    #     STORE_FINAL_STATE: tl.constexpr,
    #     IS_VARLEN: tl.constexpr,
    #     NV_MMA_SHARED_LAYOUT: tl.constexpr,
    # ):
    #     i_v, i_nh = tl.program_id(0), tl.program_id(1)

    #     if IS_VARLEN:
    #         i_n = i_nh // HV
    #         i_h = i_nh % HV
    #         bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    #         eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    #         T = eos - bos
    #         NT = tl.cdiv(T, BT)
    #     else:
    #         i_n = i_nh // HV
    #         i_h = i_nh % HV
    #         bos = i_n * T
    #         NT = tl.cdiv(T, BT)

    #     v += (bos * HV + i_h).to(tl.int64) * V
    #     beta += bos * HV + i_h
    #     gk += (bos * HV + i_h).to(tl.int64) * K
    #     Aqk += (bos * HV + i_h).to(tl.int64) * BT
    #     Akk += (bos * HV + i_h).to(tl.int64) * BT
    #     o += (bos * HV + i_h).to(tl.int64) * V
    #     ws_base = ws + (bos * HV + i_h).to(tl.int64) * 3 * K

    #     # TMA descriptors.
    #     ws_desc = tl.make_tensor_descriptor(
    #         ws_base, shape=[T, 3 * K], strides=[HV * 3 * K, 1], block_shape=[BT, 64]
    #     )
    #     gk_desc = tl.make_tensor_descriptor(
    #         gk, shape=[T, K], strides=[HV * K, 1], block_shape=[1, 64]
    #     )
    #     Aqk_desc = tl.make_tensor_descriptor(
    #         Aqk, shape=[T, BT], strides=[HV * BT, 1], block_shape=[BT, BT]
    #     )
    #     Akk_desc = tl.make_tensor_descriptor(
    #         Akk, shape=[T, BT], strides=[HV * BT, 1], block_shape=[BT, BT]
    #     )

    #     # Allocate only the K blocks needed by this specialization.
    #     w1_smem = tle.gpu.alloc(
    #         [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     vb_smem = tle.gpu.alloc(
    #         [PIPE_STAGES, BT, BV], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     qg1_smem = tle.gpu.alloc(
    #         [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     kg1_smem = tle.gpu.alloc(
    #         [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     Aqk_smem = tle.gpu.alloc(
    #         [PIPE_STAGES, BT, BT], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     Akk_smem = tle.gpu.alloc(
    #         [PIPE_STAGES, BT, BT], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     gk1_smem = tle.gpu.alloc(
    #         [PIPE_STAGES, 1, 64], dtype=tl.float32, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     if K > 64:
    #         w2_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #         qg2_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #         kg2_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #         gk2_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, 1, 64], dtype=tl.float32, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #     if K > 128:
    #         w3_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #         qg3_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #         kg3_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #         gk3_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, 1, 64], dtype=tl.float32, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #     if K > 192:
    #         w4_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #         qg4_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #         kg4_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #         gk4_smem = tle.gpu.alloc(
    #             [PIPE_STAGES, 1, 64], dtype=tl.float32, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #         )
    #     out_smem = tle.gpu.alloc(
    #         [PIPE_STAGES, BT, BV], dtype=tl.float32, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )

    #     # Build pipe slots that match the allocated blocks.
    #     if K <= 64:
    #         load_pipe = tle.pipe(
    #             capacity=PIPE_STAGES,
    #             scope="cta",
    #             name="kda_load",
    #             w1=w1_smem,
    #             vb=vb_smem,
    #             qg1=qg1_smem,
    #             kg1=kg1_smem,
    #             Aqk=Aqk_smem,
    #             Akk=Akk_smem,
    #             gk1=gk1_smem,
    #         )
    #     elif K <= 128:
    #         load_pipe = tle.pipe(
    #             capacity=PIPE_STAGES,
    #             scope="cta",
    #             name="kda_load",
    #             w1=w1_smem,
    #             w2=w2_smem,
    #             vb=vb_smem,
    #             qg1=qg1_smem,
    #             qg2=qg2_smem,
    #             kg1=kg1_smem,
    #             kg2=kg2_smem,
    #             Aqk=Aqk_smem,
    #             Akk=Akk_smem,
    #             gk1=gk1_smem,
    #             gk2=gk2_smem,
    #         )
    #     elif K <= 192:
    #         load_pipe = tle.pipe(
    #             capacity=PIPE_STAGES,
    #             scope="cta",
    #             name="kda_load",
    #             w1=w1_smem,
    #             w2=w2_smem,
    #             w3=w3_smem,
    #             vb=vb_smem,
    #             qg1=qg1_smem,
    #             qg2=qg2_smem,
    #             qg3=qg3_smem,
    #             kg1=kg1_smem,
    #             kg2=kg2_smem,
    #             kg3=kg3_smem,
    #             Aqk=Aqk_smem,
    #             Akk=Akk_smem,
    #             gk1=gk1_smem,
    #             gk2=gk2_smem,
    #             gk3=gk3_smem,
    #         )
    #     else:
    #         load_pipe = tle.pipe(
    #             capacity=PIPE_STAGES,
    #             scope="cta",
    #             name="kda_load",
    #             w1=w1_smem,
    #             w2=w2_smem,
    #             w3=w3_smem,
    #             w4=w4_smem,
    #             vb=vb_smem,
    #             qg1=qg1_smem,
    #             qg2=qg2_smem,
    #             qg3=qg3_smem,
    #             qg4=qg4_smem,
    #             kg1=kg1_smem,
    #             kg2=kg2_smem,
    #             kg3=kg3_smem,
    #             kg4=kg4_smem,
    #             Aqk=Aqk_smem,
    #             Akk=Akk_smem,
    #             gk1=gk1_smem,
    #             gk2=gk2_smem,
    #             gk3=gk3_smem,
    #             gk4=gk4_smem,
    #         )
    #     store_pipe = tle.pipe(
    #         capacity=PIPE_STAGES,
    #         scope="cta",
    #         name="kda_store",
    #         output=out_smem,
    #     )
    #     tle.gpu.warp_specialize(
    #         [
    #             (
    #                 _kda_state_output_load_producer,
    #                 (
    #                     load_pipe.writer(),
    #                     ws_desc,
    #                     v,
    #                     beta,
    #                     gk_desc,
    #                     Aqk_desc,
    #                     Akk_desc,
    #                     K,
    #                     T,
    #                     HV,
    #                     V,
    #                     BT,
    #                     BV,
    #                     NT,
    #                     i_v,
    #                 ),
    #             ),
    #             (
    #                 _kda_state_output_mma_consumer,
    #                 (
    #                     load_pipe.reader(),
    #                     store_pipe.writer(),
    #                     h0,
    #                     ht,
    #                     gk,
    #                     scale,
    #                     i_v,
    #                     i_nh,
    #                     T,
    #                     HV,
    #                     K,
    #                     V,
    #                     BT,
    #                     BV,
    #                     NT,
    #                     kg.dtype.element_ty,
    #                     USE_INITIAL_STATE,
    #                     STORE_FINAL_STATE,
    #                     STATE_V_FIRST,
    #                 ),
    #             ),
    #             (
    #                 _kda_state_output_store_consumer,
    #                 (store_pipe.reader(), o, T, HV, V, BT, BV, NT, i_v),
    #             ),
    #         ],
    #         [4, 1],
    #         [240, 32],
    #     )

    # def _kda_fwd_state_output(
    #     kg: torch.Tensor,
    #     v: torch.Tensor,
    #     beta: torch.Tensor,
    #     Akk: torch.Tensor,
    #     gk: torch.Tensor,
    #     Aqk: torch.Tensor,
    #     scale: float | None,
    #     ws: torch.Tensor | None = None,
    #     initial_state: torch.Tensor | None = None,
    #     output_final_state: bool = False,
    #     state_v_first: bool = True,
    #     cu_seqlens: torch.LongTensor | None = None,
    #     chunk_size: int = 16,
    # ) -> tuple[torch.Tensor, torch.Tensor | None]:
    #     B, _, HV, K = kg.shape
    #     T_actual = v.shape[1]
    #     V = v.shape[-1]
    #     BT = chunk_size

    #     if K > 256:
    #         raise ValueError(f"KDA K must be <= 256, got {K}")

    #     if cu_seqlens is None:
    #         N = B
    #     else:
    #         N = len(cu_seqlens) - 1

    #     final_state = None
    #     if output_final_state:
    #         if state_v_first:
    #             final_state = kg.new_zeros(N, HV, V, K, dtype=torch.float32)
    #         else:
    #             final_state = kg.new_zeros(N, HV, K, V, dtype=torch.float32)

    #     o = torch.zeros(B, T_actual, HV, V, device=kg.device, dtype=v.dtype)

    #     h0_arg = (
    #         initial_state
    #         if initial_state is not None
    #         else kg.new_empty(1, dtype=torch.float32)
    #     )
    #     ht_arg = (
    #         final_state
    #         if final_state is not None
    #         else kg.new_empty(1, dtype=torch.float32)
    #     )

    #     grid = lambda meta: (triton.cdiv(V, meta["BV"]), N * HV)
    #     _kda_fwd_state_output_kernel[grid](
    #         kg=kg,
    #         v=v,
    #         beta=beta,
    #         gk=gk,
    #         Aqk=Aqk,
    #         Akk=Akk,
    #         o=o,
    #         ws=ws,
    #         h0=h0_arg,
    #         ht=ht_arg,
    #         cu_seqlens=cu_seqlens,
    #         scale=scale,
    #         T=T_actual,
    #         HV=HV,
    #         K=K,
    #         V=V,
    #         BT=BT,
    #         STATE_V_FIRST=state_v_first,
    #         NV_MMA_SHARED_LAYOUT=_nv_mma_shared_layout(),
    #     )

    #     return o, final_state

    # =============================================================================
    # Rewritten: state propagation + output (no TMA, no pipe, no warp_specialize)
    # Uses plain tl.load/tl.store instead of tle.gpu.copy / make_tensor_descriptor
    # =============================================================================
    @triton.heuristics(
        {
            "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
            "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.autotune(
        configs=[triton.Config({"BV": BV}, num_warps=4) for BV in [32, 64, 128]],
        key=["HV", "K", "V", "BT"],
    )
    @triton.jit(do_not_specialize=["T"])
    def _kda_fwd_state_output_kernel(
        kg,
        v,
        beta,
        gk,
        Aqk,
        Akk,
        o,
        ws,
        h0,
        ht,
        cu_seqlens,
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
        NV_MMA_SHARED_LAYOUT: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)

        if IS_VARLEN:
            i_n = i_nh // HV
            i_h = i_nh % HV
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
            NT = tl.cdiv(T, BT)
        else:
            i_n = i_nh // HV
            i_h = i_nh % HV
            bos = i_n * T
            NT = tl.cdiv(T, BT)

        v += (bos * HV + i_h).to(tl.int64) * V
        beta += bos * HV + i_h
        gk += (bos * HV + i_h).to(tl.int64) * K
        Aqk += (bos * HV + i_h).to(tl.int64) * BT
        Akk += (bos * HV + i_h).to(tl.int64) * BT
        o += (bos * HV + i_h).to(tl.int64) * V
        ws_base = ws + (bos * HV + i_h).to(tl.int64) * 3 * K

        # Shared memory allocation: single stage (no pipe buffering needed)
        w1_smem = tle.gpu.alloc([BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        vb_smem = tle.gpu.alloc([BT, BV], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        qg1_smem = tle.gpu.alloc([BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        kg1_smem = tle.gpu.alloc([BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        Aqk_smem = tle.gpu.alloc([BT, BT], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        Akk_smem = tle.gpu.alloc([BT, BT], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        gk1_smem = tle.gpu.alloc([1, 64], dtype=tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        if K > 64:
            w2_smem = tle.gpu.alloc([BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
            qg2_smem = tle.gpu.alloc(
                [BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            kg2_smem = tle.gpu.alloc(
                [BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            gk2_smem = tle.gpu.alloc([1, 64], dtype=tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        if K > 128:
            w3_smem = tle.gpu.alloc([BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
            qg3_smem = tle.gpu.alloc(
                [BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            kg3_smem = tle.gpu.alloc(
                [BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            gk3_smem = tle.gpu.alloc([1, 64], dtype=tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        if K > 192:
            w4_smem = tle.gpu.alloc([BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
            qg4_smem = tle.gpu.alloc(
                [BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            kg4_smem = tle.gpu.alloc(
                [BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            gk4_smem = tle.gpu.alloc([1, 64], dtype=tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)

        # Load initial state
        if USE_INITIAL_STATE:
            if STATE_V_FIRST:
                p_h0_1 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
                b_h1 = tl.trans(tl.load(p_h0_1, boundary_check=(0, 1))).to(tl.float32)
                if K > 64:
                    p_h0_2 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
                    b_h2 = tl.trans(tl.load(p_h0_2, boundary_check=(0, 1))).to(tl.float32)
                if K > 128:
                    p_h0_3 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
                    b_h3 = tl.trans(tl.load(p_h0_3, boundary_check=(0, 1))).to(tl.float32)
                if K > 192:
                    p_h0_4 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
                    b_h4 = tl.trans(tl.load(p_h0_4, boundary_check=(0, 1))).to(tl.float32)
            else:
                p_h0_1 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
                b_h1 = tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
                if K > 64:
                    p_h0_2 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                    b_h2 = tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
                if K > 128:
                    p_h0_3 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                    b_h3 = tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
                if K > 192:
                    p_h0_4 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                    b_h4 = tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_h1 = tl.zeros([64, BV], dtype=tl.float32)
            if K > 64:
                b_h2 = tl.zeros([64, BV], dtype=tl.float32)
            if K > 128:
                b_h3 = tl.zeros([64, BV], dtype=tl.float32)
            if K > 192:
                b_h4 = tl.zeros([64, BV], dtype=tl.float32)

        # Sequential recurrence across chunks (merged producer + consumer + store)
        for i_t in tl.range(NT):
            last_idx = tl.minimum(i_t * BT + BT, T) - 1

            # ---- Load phase: plain tl.load into smem (replaces tle.gpu.copy) ----
            # ws layout: [T, 3*K] with stride [HV*3*K, 1]
            # w at col offset 0, qg at K, kg at 2*K
            p_w1 = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            tl.store(tle.gpu.local_ptr(w1_smem), tl.load(p_w1, boundary_check=(0, 1)))
            p_qg1 = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, K), (BT, 64), (1, 0))
            tl.store(tle.gpu.local_ptr(qg1_smem), tl.load(p_qg1, boundary_check=(0, 1)))
            p_kg1 = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 2 * K), (BT, 64), (1, 0))
            tl.store(tle.gpu.local_ptr(kg1_smem), tl.load(p_kg1, boundary_check=(0, 1)))

            # Aqk, Akk: [T, BT] with stride [HV*BT, 1]
            p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
            tl.store(tle.gpu.local_ptr(Aqk_smem), tl.load(p_Aqk, boundary_check=(0, 1)))
            p_Akk = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
            tl.store(tle.gpu.local_ptr(Akk_smem), tl.load(p_Akk, boundary_check=(0, 1)))

            # gk last row: [1, 64] from gk[T, K] stride [HV*K, 1]
            p_gk1 = tl.make_block_ptr(gk, (T, K), (HV * K, 1), (last_idx, 0), (1, 64), (1, 0))
            tl.store(tle.gpu.local_ptr(gk1_smem), tl.load(p_gk1, boundary_check=(0, 1)))

            if K > 64:
                p_w2 = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                tl.store(tle.gpu.local_ptr(w2_smem), tl.load(p_w2, boundary_check=(0, 1)))
                p_qg2 = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, K + 64), (BT, 64), (1, 0))
                tl.store(tle.gpu.local_ptr(qg2_smem), tl.load(p_qg2, boundary_check=(0, 1)))
                p_kg2 = tl.make_block_ptr(
                    ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 2 * K + 64), (BT, 64), (1, 0)
                )
                tl.store(tle.gpu.local_ptr(kg2_smem), tl.load(p_kg2, boundary_check=(0, 1)))
                p_gk2 = tl.make_block_ptr(gk, (T, K), (HV * K, 1), (last_idx, 64), (1, 64), (1, 0))
                tl.store(tle.gpu.local_ptr(gk2_smem), tl.load(p_gk2, boundary_check=(0, 1)))
            if K > 128:
                p_w3 = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                tl.store(tle.gpu.local_ptr(w3_smem), tl.load(p_w3, boundary_check=(0, 1)))
                p_qg3 = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, K + 128), (BT, 64), (1, 0))
                tl.store(tle.gpu.local_ptr(qg3_smem), tl.load(p_qg3, boundary_check=(0, 1)))
                p_kg3 = tl.make_block_ptr(
                    ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 2 * K + 128), (BT, 64), (1, 0)
                )
                tl.store(tle.gpu.local_ptr(kg3_smem), tl.load(p_kg3, boundary_check=(0, 1)))
                p_gk3 = tl.make_block_ptr(gk, (T, K), (HV * K, 1), (last_idx, 128), (1, 64), (1, 0))
                tl.store(tle.gpu.local_ptr(gk3_smem), tl.load(p_gk3, boundary_check=(0, 1)))
            if K > 192:
                p_w4 = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                tl.store(tle.gpu.local_ptr(w4_smem), tl.load(p_w4, boundary_check=(0, 1)))
                p_qg4 = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, K + 192), (BT, 64), (1, 0))
                tl.store(tle.gpu.local_ptr(qg4_smem), tl.load(p_qg4, boundary_check=(0, 1)))
                p_kg4 = tl.make_block_ptr(
                    ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 2 * K + 192), (BT, 64), (1, 0)
                )
                tl.store(tle.gpu.local_ptr(kg4_smem), tl.load(p_kg4, boundary_check=(0, 1)))
                p_gk4 = tl.make_block_ptr(gk, (T, K), (HV * K, 1), (last_idx, 192), (1, 64), (1, 0))
                tl.store(tle.gpu.local_ptr(gk4_smem), tl.load(p_gk4, boundary_check=(0, 1)))

            # v and beta: elementwise, keep as regular loads
            p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_beta = tl.load(p_beta, boundary_check=(0,))
            b_beta_f = tl.sigmoid(b_beta.to(tl.float32))
            b_vb = (b_v.to(tl.float32) * b_beta_f[:, None]).to(b_v.dtype)
            tl.store(tle.gpu.local_ptr(vb_smem), b_vb)

            # ---- Compute phase (formerly the MMA consumer) ----
            b_w1 = tl.load(tle.gpu.local_ptr(w1_smem))
            b_vb = tl.load(tle.gpu.local_ptr(vb_smem))
            b_qg1 = tl.load(tle.gpu.local_ptr(qg1_smem))
            b_kg1 = tl.load(tle.gpu.local_ptr(kg1_smem))
            b_Aqk = tl.load(tle.gpu.local_ptr(Aqk_smem))
            b_Akk = tl.load(tle.gpu.local_ptr(Akk_smem))
            b_gk1 = tl.load(tle.gpu.local_ptr(gk1_smem)).reshape([64])
            if K > 64:
                b_w2 = tl.load(tle.gpu.local_ptr(w2_smem))
                b_qg2 = tl.load(tle.gpu.local_ptr(qg2_smem))
                b_kg2 = tl.load(tle.gpu.local_ptr(kg2_smem))
                b_gk2 = tl.load(tle.gpu.local_ptr(gk2_smem)).reshape([64])
            if K > 128:
                b_w3 = tl.load(tle.gpu.local_ptr(w3_smem))
                b_qg3 = tl.load(tle.gpu.local_ptr(qg3_smem))
                b_kg3 = tl.load(tle.gpu.local_ptr(kg3_smem))
                b_gk3 = tl.load(tle.gpu.local_ptr(gk3_smem)).reshape([64])
            if K > 192:
                b_w4 = tl.load(tle.gpu.local_ptr(w4_smem))
                b_qg4 = tl.load(tle.gpu.local_ptr(qg4_smem))
                b_kg4 = tl.load(tle.gpu.local_ptr(gk4_smem)).reshape([64])
                b_gk4 = tl.load(tle.gpu.local_ptr(gk4_smem)).reshape([64])

            kg_dtype = kg.dtype.element_ty
            b_h1_bf = b_h1.to(kg_dtype)
            if K > 64:
                b_h2_bf = b_h2.to(kg_dtype)
            if K > 128:
                b_h3_bf = b_h3.to(kg_dtype)
            if K > 192:
                b_h4_bf = b_h4.to(kg_dtype)

            # v_new = Akk_inv @ (v*beta - w @ h)
            b_kh = tl.dot(b_w1, b_h1_bf).to(tl.float32)
            if K > 64:
                b_kh += tl.dot(b_w2, b_h2_bf).to(tl.float32)
            if K > 128:
                b_kh += tl.dot(b_w3, b_h3_bf).to(tl.float32)
            if K > 192:
                b_kh += tl.dot(b_w4, b_h4_bf).to(tl.float32)
            b_diff = b_vb.to(tl.float32) - b_kh
            b_v_new = tl.dot(b_Akk, b_diff.to(kg_dtype)).to(tl.float32)

            # output = scale * qg @ h + Aqk @ v_new
            b_qh = tl.dot(b_qg1, b_h1_bf).to(tl.float32)
            if K > 64:
                b_qh += tl.dot(b_qg2, b_h2_bf).to(tl.float32)
            if K > 128:
                b_qh += tl.dot(b_qg3, b_h3_bf).to(tl.float32)
            if K > 192:
                b_qh += tl.dot(b_qg4, b_h4_bf).to(tl.float32)
            b_o = scale * b_qh
            b_v_cast = b_v_new.to(kg_dtype)
            b_o += tl.dot(b_Aqk, b_v_cast).to(tl.float32)

            # ---- Store phase: plain tl.store to global memory ----
            p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

            # ---- State decay + update ----
            b_h1 = b_h1 * exp2(b_gk1)[:, None] + tl.dot(tl.trans(b_kg1), b_v_cast).to(tl.float32)
            if K > 64:
                b_h2 = b_h2 * exp2(b_gk2)[:, None] + tl.dot(tl.trans(b_kg2), b_v_cast).to(tl.float32)
            if K > 128:
                b_h3 = b_h3 * exp2(b_gk3)[:, None] + tl.dot(tl.trans(b_kg3), b_v_cast).to(tl.float32)
            if K > 192:
                b_h4 = b_h4 * exp2(b_gk4)[:, None] + tl.dot(tl.trans(b_kg4), b_v_cast).to(tl.float32)

        # Store final state
        if STORE_FINAL_STATE:
            if STATE_V_FIRST:
                p_ht1 = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
                tl.store(p_ht1, tl.trans(b_h1).to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
                if K > 64:
                    p_ht2 = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
                    tl.store(p_ht2, tl.trans(b_h2).to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
                if K > 128:
                    p_ht3 = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
                    tl.store(p_ht3, tl.trans(b_h3).to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
                if K > 192:
                    p_ht4 = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
                    tl.store(p_ht4, tl.trans(b_h4).to(p_ht4.dtype.element_ty), boundary_check=(0, 1))
            else:
                p_ht1 = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
                if K > 64:
                    p_ht2 = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                    tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
                if K > 128:
                    p_ht3 = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                    tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
                if K > 192:
                    p_ht4 = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                    tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))

    def _kda_fwd_state_output(
        kg: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        Akk: torch.Tensor,
        gk: torch.Tensor,
        Aqk: torch.Tensor,
        scale: float | None,
        ws: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        state_v_first: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, _, HV, K = kg.shape
        T_actual = v.shape[1]
        V = v.shape[-1]
        BT = chunk_size

        if K > 256:
            raise ValueError(f"KDA K must be <= 256, got {K}")

        if cu_seqlens is None:
            N = B
        else:
            N = len(cu_seqlens) - 1

        final_state = None
        if output_final_state:
            if state_v_first:
                final_state = kg.new_zeros(N, HV, V, K, dtype=torch.float32)
            else:
                final_state = kg.new_zeros(N, HV, K, V, dtype=torch.float32)

        o = torch.zeros(B, T_actual, HV, V, device=kg.device, dtype=v.dtype)

        h0_arg = initial_state if initial_state is not None else kg.new_empty(1, dtype=torch.float32)
        ht_arg = final_state if final_state is not None else kg.new_empty(1, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(V, meta["BV"]), N * HV)  # noqa: E731

        _kda_fwd_state_output_kernel[grid](
            kg=kg,
            v=v,
            beta=beta,
            gk=gk,
            Aqk=Aqk,
            Akk=Akk,
            o=o,
            ws=ws,
            h0=h0_arg,
            ht=ht_arg,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T_actual,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            STATE_V_FIRST=state_v_first,
            NV_MMA_SHARED_LAYOUT=_nv_mma_shared_layout(),
        )

        return o, final_state


def chunk_kda_fwd_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
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
    triton.set_allocator(_allocate_triton_workspace)
    _refresh_fp16_dot_precision()

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    ws, Aqk, Akk, g_cumsum = _kda_fwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    K = q.shape[-1]
    return _kda_fwd_state_output(
        kg=ws[:, :, :, 2 * K :],
        v=v,
        beta=beta,
        Akk=Akk,
        gk=g_cumsum,
        Aqk=Aqk,
        ws=ws,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


# =============================================================================
# Strict TLE fast path (narrower constraints, higher performance)
# =============================================================================

HAS_STRICT_TLE_KDA = HAS_TLE_KDA


def strict_tle_input_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    cu_seqlens: torch.LongTensor | None,
    safe_gate: bool,
    lower_bound: float | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    chunk_size: int,
) -> str | None:
    if not HAS_STRICT_TLE_KDA:
        return "strict TLE KDA requires Triton TLE >= 3.6.0"

    inputs = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    invalid_dtypes = {name: tensor.dtype for name, tensor in inputs.items() if tensor.dtype != torch.bfloat16}
    if invalid_dtypes:
        details = ", ".join(f"{name}={dtype}" for name, dtype in invalid_dtypes.items())
        return f"strict TLE KDA requires bfloat16 inputs, got {details}"
    if any(not tensor.is_cuda for tensor in inputs.values()):
        return "strict TLE KDA requires CUDA inputs"
    if any(tensor.device != q.device for tensor in inputs.values()):
        return "strict TLE KDA requires all inputs on the same device"
    if any(not tensor.is_contiguous() for tensor in inputs.values()):
        return "strict TLE KDA requires contiguous inputs"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 4 or beta.ndim != 3:
        return "strict TLE KDA expects q/k/v/g with rank 4 and beta with rank 3"

    B, T, H, D = q.shape
    if T == 0:
        return "strict TLE KDA requires a non-empty sequence"
    if D != 128:
        return f"strict TLE KDA requires K=128, got {D}"
    if v.shape[-1] != 128:
        return f"strict TLE KDA requires V=128, got {v.shape[-1]}"
    if v.shape[2] != H:
        return f"strict TLE KDA does not support GVA (HV={v.shape[2]} != H={H})"
    if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        return "strict TLE KDA requires q, k, v, and g shape [B, T, H, 128]"
    if beta.shape != (B, T, H):
        return f"strict TLE KDA requires beta shape {(B, T, H)}, got {tuple(beta.shape)}"
    if not use_qk_l2norm_in_kernel:
        return "strict TLE KDA requires use_qk_l2norm_in_kernel=True"
    if not use_gate_in_kernel:
        return "strict TLE KDA requires use_gate_in_kernel=True"
    if not use_beta_sigmoid_in_kernel:
        return "strict TLE KDA requires use_beta_sigmoid_in_kernel=True"
    if allow_neg_eigval:
        return "strict TLE KDA does not support allow_neg_eigval=True"
    if not safe_gate:
        return "strict TLE KDA requires safe_gate=True"
    if lower_bound is None or not -5 <= lower_bound < 0:
        return f"strict TLE KDA requires -5 <= lower_bound < 0, got {lower_bound}"
    if not state_v_first:
        return "strict TLE KDA requires state_v_first=True"
    if chunk_size != 16:
        return f"strict TLE KDA requires chunk_size=16, got {chunk_size}"

    if A_log is None or A_log.dtype != torch.float32 or A_log.shape != (H,):
        actual = None if A_log is None else (tuple(A_log.shape), A_log.dtype)
        return f"strict TLE KDA requires float32 A_log with shape {(H,)}, got {actual}"
    if dt_bias is None or dt_bias.dtype != torch.float32 or dt_bias.shape != (H, D):
        actual = None if dt_bias is None else (tuple(dt_bias.shape), dt_bias.dtype)
        return f"strict TLE KDA requires float32 dt_bias with shape {(H, D)}, got {actual}"
    if A_log.device != q.device or dt_bias.device != q.device:
        return "strict TLE KDA requires A_log and dt_bias on the input device"
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        return "strict TLE KDA requires contiguous A_log and dt_bias"

    N = B
    if cu_seqlens is not None:
        if B != 1:
            return "strict TLE KDA requires B=1 when cu_seqlens is provided"
        if cu_seqlens.device != q.device or cu_seqlens.dtype != torch.long or cu_seqlens.ndim != 1:
            return "strict TLE KDA requires a 1D int64 cu_seqlens tensor on the input device"
        if cu_seqlens.numel() < 2:
            return "strict TLE KDA requires cu_seqlens to contain at least two elements"
        N = cu_seqlens.numel() - 1

    if initial_state is not None:
        expected = (N, H, D, D)
        if initial_state.dtype != torch.float32 or initial_state.shape != expected:
            return f"strict TLE KDA requires float32 initial_state with shape {expected}"
        if initial_state.device != q.device or not initial_state.is_contiguous():
            return "strict TLE KDA requires contiguous initial_state on the input device"
    return None


if HAS_STRICT_TLE_KDA:

    @triton.heuristics(
        {
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=num_warps, num_stages=num_stages, maxnreg=maxnreg)
            for num_warps in [2, 4, 8]
            for num_stages in [2, 4, 8]
            for maxnreg in [None, 32, 64, 72]
        ],
        key=["H", "HV", "K", "BT", "IS_VARLEN"],
    )
    @triton.jit(do_not_specialize=["T"])
    def _strict_kda_fwd_intra_kernel(
        q,
        k,
        g,
        beta,
        ws,
        Aqk,
        Akk,
        g_last,
        A_log,
        dt_bias,
        lower_bound,
        scale,
        g_scale,
        l2norm_eps,
        cu_seqlens,
        chunk_indices,
        T,
        NT_TOTAL,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        NV_MMA_SHARED_LAYOUT: tl.constexpr,
    ):
        i_tg, i_bh = tl.program_id(0), tl.program_id(1)
        i_t = i_tg
        i_hv = i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_b = 0
            i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
        else:
            i_b = i_bh // HV
            bos = i_b.to(tl.int64) * T
            i_tg = i_b * tl.cdiv(T, BT) + i_t

        if i_t * BT >= T:
            return

        q += (bos * H + i_h) * K
        k += (bos * H + i_h) * K
        g += (bos * HV + i_hv) * K
        g_last += (i_tg * HV + i_hv).to(tl.int64) * K
        if IS_VARLEN:
            a_chunk = i_hv * NT_TOTAL + i_tg
        else:
            a_chunk = (i_b * HV + i_hv) * NT_TOTAL + i_t
        Aqk += a_chunk.to(tl.int64) * BT * BT
        Akk += a_chunk.to(tl.int64) * BT * BT
        ws += (bos * HV + i_hv) * 3 * K
        beta += bos * HV + i_hv

        o_i = tl.arange(0, BT)
        token_start = i_t * BT
        o_c = token_start + o_i
        m_c = o_c < T

        q_buf = tle.gpu.alloc([BT, K], dtype=q.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        k_buf = tle.gpu.alloc([BT, K], dtype=k.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        gc_buf = tle.gpu.alloc([BT, K], dtype=tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)

        rows = tl.broadcast_to(tl.arange(0, BT)[:, None], (BT, K))
        cols = tl.broadcast_to(tl.arange(0, K)[None, :], (BT, K))
        q_sp = tle.gpu.local_ptr(q_buf, (rows, cols))
        k_sp = tle.gpu.local_ptr(k_buf, (rows, cols))
        gc_sp = tle.gpu.local_ptr(gc_buf, (rows, cols))

        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (token_start, 0), (BT, K), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (token_start, 0), (BT, K), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (token_start, 0), (BT, K), (1, 0))
        b_q = tle.load(p_q, boundary_check=(0, 1), is_async=True)
        b_k = tle.load(p_k, boundary_check=(0, 1), is_async=True)
        tl.store(q_sp, b_q)
        tl.store(k_sp, b_k)

        b_qf = b_q.to(tl.float32)
        b_kf = b_k.to(tl.float32)

        b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
        b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

        b_g = tle.load(p_g, boundary_check=(0, 1), is_async=True).to(tl.float32)
        b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)
        p_dt = tl.make_block_ptr(dt_bias + i_hv * K, (K,), (1,), (0,), (K,), (0,))
        b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
        b_g = b_g + b_bias[None, :]
        b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
        tl.store(gc_sp, b_g)
        one_row = tl.broadcast_to(tl.arange(0, 1)[:, None], (1, K))
        col_row = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
        b_acc = tl.zeros([1, K], dtype=tl.float32)
        for r in tl.static_range(BT):
            rp = tle.gpu.local_ptr(gc_buf, (tl.broadcast_to(one_row + r, (1, K)), col_row))
            b_acc = b_acc + tl.load(rp)
            tl.store(rp, b_acc)

        b_gq = tl.where(m_c[:, None], exp2(tl.load(gc_sp)), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-tl.load(gc_sp)), 0.0)

        b_kgt = tl.trans(b_kf * b_gk).to(b_k.dtype)
        b_Aqk = tl.dot((b_qf * b_gq).to(b_q.dtype), b_kgt, out_dtype=tl.float32)
        b_Akk = tl.dot((b_kf * b_gq).to(b_k.dtype), b_kgt, out_dtype=tl.float32)

        b_Aqk = b_Aqk * b_q_rstd[:, None] * b_k_rstd[None, :]
        b_Akk = b_Akk * b_k_rstd[:, None] * b_k_rstd[None, :]

        p_beta = tl.make_block_ptr(beta, (T,), (HV,), (token_start,), (BT,), (0,))
        b_beta = tl.sigmoid(tl.load(p_beta, boundary_check=(0,)).to(tl.float32))

        m_Aqk = o_i[:, None] >= o_i[None, :]
        m_Akk = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
        b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

        p_Aqk = tl.make_block_ptr(Aqk, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
        tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty))

        b_L = b_Akk.to(tl.float16)
        b_Ai = m_I.to(tl.float16) - b_L
        b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
        b_Ai = b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
        b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
        b_Ai = b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
        b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
        b_Ai = b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float16)

        p_Akk_out = tl.make_block_ptr(Akk, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
        tl.store(p_Akk_out, b_Ai.to(Akk.dtype.element_ty))

        b_k3 = tl.load(k_sp).to(tl.float32) * b_k_rstd[:, None]
        b_gk3 = tl.load(gc_sp)
        b_kb = b_k3 * b_beta[:, None] * exp2(b_gk3)
        p_w = tl.make_block_ptr(ws, (T, 3 * K), (HV * 3 * K, 1), (token_start, 0), (BT, K), (1, 0))
        tl.store(p_w, b_kb.to(ws.dtype.element_ty), boundary_check=(0, 1))

        b_q3 = tl.load(q_sp).to(tl.float32) * b_q_rstd[:, None]
        b_qg_val = b_q3 * exp2(b_gk3)
        p_qg = tl.make_block_ptr(ws, (T, 3 * K), (HV * 3 * K, 1), (token_start, K), (BT, K), (1, 0))
        tl.store(p_qg, b_qg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))

        last_local = (tl.minimum(BT, T - token_start) - 1).to(tl.int32)
        gn_rows = tl.broadcast_to(last_local + tl.zeros([1, K], dtype=tl.int32), (1, K))
        gn_cols = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
        b_gn = tl.load(tle.gpu.local_ptr(gc_buf, (gn_rows, gn_cols)))
        p_g_last = tl.make_block_ptr(g_last, (1, K), (K, 1), (0, 0), (1, K), (1, 0))
        tl.store(p_g_last, b_gn.to(g_last.dtype.element_ty), boundary_check=(0, 1))
        b_kg_val = b_k3 * tl.where(m_c[:, None], exp2(b_gn - b_gk3), 0)
        p_kg = tl.make_block_ptr(
            ws,
            (T, 3 * K),
            (HV * 3 * K, 1),
            (token_start, 2 * K),
            (BT, K),
            (1, 0),
        )
        tl.store(p_kg, b_kg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))

    def _strict_kda_fwd_intra(
        q,
        k,
        g,
        beta,
        scale,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=16,
        lower_bound=None,
        A_log=None,
        dt_bias=None,
    ):
        B, T_len, H, K = q.shape
        HV = g.shape[2]
        BT = chunk_size

        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)
        grid = (NT, B * HV)

        T_padded = NT * BT
        g_last = torch.empty(B * NT, HV, K, device=q.device, dtype=torch.float32)
        ws = torch.empty(B, T_padded, HV, 3 * K, device=q.device, dtype=q.dtype)
        Aqk = torch.empty(B, HV, NT, BT, BT, device=q.device, dtype=q.dtype)
        Akk = torch.empty(B, HV, NT, BT, BT, device=q.device, dtype=q.dtype)

        _strict_kda_fwd_intra_kernel[grid](
            q=q,
            k=k,
            g=g,
            beta=beta,
            ws=ws,
            Aqk=Aqk,
            Akk=Akk,
            g_last=g_last,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            scale=scale,
            g_scale=RCP_LN2,
            l2norm_eps=1e-6,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T_len,
            NT_TOTAL=NT,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            NV_MMA_SHARED_LAYOUT=_nv_mma_shared_layout(),
        )
        return ws, Aqk, Akk, g_last

    # @triton.jit
    # def _strict_kda_state_output_load_producer(
    #     writer,
    #     ws_desc,
    #     v_ptr,
    #     beta_ptr,
    #     gk_desc,
    #     Aqk_desc,
    #     Akk_desc,
    #     K: tl.constexpr,
    #     T,
    #     HV: tl.constexpr,
    #     V: tl.constexpr,
    #     BT: tl.constexpr,
    #     BV: tl.constexpr,
    #     NT,
    #     i_v,
    #     USE_HOST_DESCRIPTORS: tl.constexpr,
    #     state_row_start,
    #     chunk_start,
    #     i_h,
    # ):
    #     for i_t in tl.range(NT):
    #         slot = writer.acquire(i_t)

    #         ws_row = i_t * BT
    #         ws_col = 0
    #         A_row = i_t * BT
    #         A_col = 0
    #         gk_row = i_t
    #         gk_col = 0
    #         if USE_HOST_DESCRIPTORS:
    #             ws_row += state_row_start
    #             ws_col = i_h * 3 * K
    #             A_row = state_row_start * HV + i_h * NT * BT + i_t * BT
    #             gk_row += chunk_start
    #             gk_col = i_h * K

    #         ws_row = ws_row.to(tl.int32)
    #         ws_col = ws_col.to(tl.int32)
    #         A_row = A_row.to(tl.int32)
    #         A_col = A_col.to(tl.int32)
    #         gk_row = gk_row.to(tl.int32)
    #         gk_col = gk_col.to(tl.int32)

    #         tle.gpu.copy(ws_desc, slot.w, [BT, K], [ws_row, ws_col])
    #         tle.gpu.copy(ws_desc, slot.qg, [BT, K], [ws_row, ws_col + K])
    #         tle.gpu.copy(ws_desc, slot.kg, [BT, K], [ws_row, ws_col + 2 * K])
    #         tle.gpu.copy(Aqk_desc, slot.Aqk, [BT, BT], [A_row, A_col])
    #         tle.gpu.copy(Akk_desc, slot.Akk, [BT, BT], [A_row, A_col])
    #         tle.gpu.copy(gk_desc, slot.gk, [1, K], [gk_row, gk_col])

    #         p_v = tl.make_block_ptr(
    #             v_ptr,
    #             (T, V),
    #             (HV * V, 1),
    #             (i_t * BT, i_v * BV),
    #             (BT, BV),
    #             (1, 0),
    #         )
    #         b_v_raw = tl.load(p_v, boundary_check=(0, 1))
    #         p_beta = tl.make_block_ptr(beta_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    #         b_beta = tl.load(p_beta, boundary_check=(0,))
    #         b_beta_f = tl.sigmoid(b_beta.to(tl.float32))
    #         b_v = (b_v_raw.to(tl.float32) * b_beta_f[:, None]).to(b_v_raw.dtype)
    #         tl.store(tle.gpu.local_ptr(slot.v), b_v)

    #         writer.commit(i_t)

    # @triton.jit
    # def _strict_kda_state_output_mma_consumer(
    #     load_reader,
    #     store_writer,
    #     h0,
    #     ht,
    #     scale,
    #     i_v,
    #     i_nh,
    #     K: tl.constexpr,
    #     V: tl.constexpr,
    #     BV: tl.constexpr,
    #     NT,
    #     USE_INITIAL_STATE: tl.constexpr,
    #     STORE_FINAL_STATE: tl.constexpr,
    # ):
    #     state_dtype: tl.constexpr = tl.bfloat16
    #     if USE_INITIAL_STATE:
    #         p_h0 = tl.make_block_ptr(
    #             h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0)
    #         )
    #         b_h = tl.trans(tl.load(p_h0, boundary_check=(0, 1))).to(tl.float32)
    #     else:
    #         b_h = tl.zeros([K, BV], dtype=tl.float32)

    #     for i_t in tl.range(NT):
    #         wait = load_reader.wait(i_t)
    #         slot = wait.slot

    #         b_w = tl.load(tle.gpu.local_ptr(slot.w))
    #         b_v_raw = tl.load(tle.gpu.local_ptr(slot.v))
    #         b_qg = tl.load(tle.gpu.local_ptr(slot.qg))
    #         b_kg = tl.load(tle.gpu.local_ptr(slot.kg))
    #         b_Aqk = tl.load(tle.gpu.local_ptr(slot.Aqk))
    #         b_Akk = tl.load(tle.gpu.local_ptr(slot.Akk))
    #         b_gk = tl.load(tle.gpu.local_ptr(slot.gk)).reshape([K])

    #         b_h_bf = b_h.to(state_dtype)

    #         b_kh = tl.dot(b_w, b_h_bf).to(tl.float32)
    #         b_diff = b_v_raw.to(tl.float32) - b_kh
    #         b_v = tl.dot(b_Akk, b_diff.to(state_dtype)).to(tl.float32)

    #         b_qh = tl.dot(b_qg, b_h_bf).to(tl.float32)
    #         b_o = scale * b_qh
    #         b_v_cast = b_v.to(state_dtype)
    #         b_o += tl.dot(b_Aqk, b_v_cast).to(tl.float32)

    #         out_slot = store_writer.acquire(i_t)
    #         tl.store(tle.gpu.local_ptr(out_slot.output), b_o)
    #         store_writer.commit(i_t)

    #         load_reader.release(i_t)

    #         b_h = b_h * exp2(b_gk)[:, None] + tl.dot(tl.trans(b_kg), b_v_cast).to(
    #             tl.float32
    #         )

    #     if STORE_FINAL_STATE:
    #         p_ht = tl.make_block_ptr(
    #             ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0)
    #         )
    #         tl.store(
    #             p_ht, tl.trans(b_h).to(p_ht.dtype.element_ty), boundary_check=(0, 1)
    #         )

    # @triton.jit
    # def _strict_kda_state_output_store_consumer(
    #     store_reader,
    #     store_target,
    #     BT: tl.constexpr,
    #     BV: tl.constexpr,
    #     NT,
    #     i_v,
    #     USE_HOST_DESCRIPTORS: tl.constexpr,
    #     output_row_start,
    #     output_col_start,
    # ):
    #     for i_t in tl.range(NT):
    #         store_wait = store_reader.wait(i_t)
    #         slot = store_wait.slot
    #         output_row = i_t * BT
    #         output_col = i_v * BV
    #         if USE_HOST_DESCRIPTORS:
    #             output_row += output_row_start
    #             output_col += output_col_start
    #         output_row = output_row.to(tl.int32)
    #         output_col = output_col.to(tl.int32)
    #         tle.gpu.copy(slot.output, store_target, [BT, BV], [output_row, output_col])
    #         store_reader.release(i_t)

    # STRICT_PIPE_STAGES = tl.constexpr(4)

    # @triton.heuristics(
    #     {
    #         "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
    #         "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
    #         "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    #     }
    # )
    # @triton.jit(do_not_specialize=["T"])
    # def _strict_kda_fwd_state_output_kernel(
    #     v,
    #     beta,
    #     gk,
    #     Aqk,
    #     Akk,
    #     o,
    #     ws,
    #     h0,
    #     ht,
    #     ws_host_desc,
    #     gk_host_desc,
    #     Aqk_host_desc,
    #     Akk_host_desc,
    #     output_host_desc,
    #     cu_seqlens,
    #     chunk_offsets,
    #     scale,
    #     T,
    #     NT_TOTAL,
    #     HV: tl.constexpr,
    #     K: tl.constexpr,
    #     V: tl.constexpr,
    #     BT: tl.constexpr,
    #     BV: tl.constexpr,
    #     USE_HOST_DESCRIPTORS: tl.constexpr,
    #     USE_INITIAL_STATE: tl.constexpr,
    #     STORE_FINAL_STATE: tl.constexpr,
    #     IS_VARLEN: tl.constexpr,
    #     NV_MMA_SHARED_LAYOUT: tl.constexpr,
    # ):
    #     i_v, i_nh = tl.program_id(0), tl.program_id(1)

    #     if IS_VARLEN:
    #         i_n = i_nh // HV
    #         i_h = i_nh % HV
    #         bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    #         eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    #         T = (eos - bos).to(tl.int32)
    #         NT = tl.cdiv(T, BT)
    #         chunk_start = tl.load(chunk_offsets + i_n).to(tl.int32)
    #     else:
    #         i_n = i_nh // HV
    #         i_h = i_nh % HV
    #         bos = i_n.to(tl.int64) * T
    #         NT = tl.cdiv(T, BT)
    #         chunk_start = i_n * NT

    #     v += (bos * HV + i_h) * V
    #     beta += bos * HV + i_h
    #     gk += (chunk_start * HV + i_h).to(tl.int64) * K
    #     o += (bos * HV + i_h) * V
    #     ws_base = ws + (bos * HV + i_h) * 3 * K

    #     if IS_VARLEN:
    #         state_row_start = bos.to(tl.int64)
    #     else:
    #         state_row_start = i_n.to(tl.int64) * NT * BT
    #     if USE_HOST_DESCRIPTORS:
    #         ws_desc = ws_host_desc
    #         gk_desc = gk_host_desc
    #         Aqk_desc = Aqk_host_desc
    #         Akk_desc = Akk_host_desc
    #     else:
    #         if IS_VARLEN:
    #             a_chunk = (i_h * NT_TOTAL + chunk_start) * BT * BT
    #         else:
    #             a_chunk = (i_n * HV + i_h) * NT_TOTAL * BT * BT
    #         Aqk += a_chunk.to(tl.int64)
    #         Akk += a_chunk.to(tl.int64)
    #         ws_desc = tl.make_tensor_descriptor(
    #             ws_base,
    #             shape=[T, 3 * K],
    #             strides=[HV * 3 * K, 1],
    #             block_shape=[BT, K],
    #         )
    #         gk_desc = tl.make_tensor_descriptor(
    #             gk, shape=[NT, K], strides=[HV * K, 1], block_shape=[1, K]
    #         )
    #         Aqk_desc = tl.make_tensor_descriptor(
    #             Aqk,
    #             shape=[NT * BT, BT],
    #             strides=[BT, 1],
    #             block_shape=[BT, BT],
    #         )
    #         Akk_desc = tl.make_tensor_descriptor(
    #             Akk,
    #             shape=[NT * BT, BT],
    #             strides=[BT, 1],
    #             block_shape=[BT, BT],
    #         )

    #     w_smem = tle.gpu.alloc(
    #         [STRICT_PIPE_STAGES, BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     v_smem = tle.gpu.alloc(
    #         [STRICT_PIPE_STAGES, BT, BV], dtype=tl.bfloat16, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     qg_smem = tle.gpu.alloc(
    #         [STRICT_PIPE_STAGES, BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     kg_smem = tle.gpu.alloc(
    #         [STRICT_PIPE_STAGES, BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     Aqk_smem = tle.gpu.alloc(
    #         [STRICT_PIPE_STAGES, BT, BT], dtype=tl.bfloat16, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     Akk_smem = tle.gpu.alloc(
    #         [STRICT_PIPE_STAGES, BT, BT], dtype=tl.bfloat16, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     gk_smem = tle.gpu.alloc(
    #         [STRICT_PIPE_STAGES, 1, K], dtype=tl.float32, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     out_smem = tle.gpu.alloc(
    #         [STRICT_PIPE_STAGES, BT, BV], dtype=tl.bfloat16, scope=tle.gpu.smem,nv_mma_shared_layout=False
    #     )
    #     if USE_HOST_DESCRIPTORS:
    #         output_store_target = output_host_desc
    #     else:
    #         output_store_target = tl.make_tensor_descriptor(
    #             o, shape=[T, V], strides=[HV * V, 1], block_shape=[BT, BV]
    #         )

    #     load_pipe = tle.pipe(
    #         capacity=STRICT_PIPE_STAGES,
    #         scope="cta",
    #         name="kda_load",
    #         w=w_smem,
    #         v=v_smem,
    #         qg=qg_smem,
    #         kg=kg_smem,
    #         Aqk=Aqk_smem,
    #         Akk=Akk_smem,
    #         gk=gk_smem,
    #     )
    #     store_pipe = tle.pipe(
    #         capacity=STRICT_PIPE_STAGES,
    #         scope="cta",
    #         name="kda_store",
    #         output=out_smem,
    #     )
    #     tle.gpu.warp_specialize(
    #         [
    #             (
    #                 _strict_kda_state_output_load_producer,
    #                 (
    #                     load_pipe.writer(),
    #                     ws_desc,
    #                     v,
    #                     beta,
    #                     gk_desc,
    #                     Aqk_desc,
    #                     Akk_desc,
    #                     K,
    #                     T,
    #                     HV,
    #                     V,
    #                     BT,
    #                     BV,
    #                     NT,
    #                     i_v,
    #                     USE_HOST_DESCRIPTORS,
    #                     state_row_start,
    #                     chunk_start,
    #                     i_h,
    #                 ),
    #             ),
    #             (
    #                 _strict_kda_state_output_mma_consumer,
    #                 (
    #                     load_pipe.reader(),
    #                     store_pipe.writer(),
    #                     h0,
    #                     ht,
    #                     scale,
    #                     i_v,
    #                     i_nh,
    #                     K,
    #                     V,
    #                     BV,
    #                     NT,
    #                     USE_INITIAL_STATE,
    #                     STORE_FINAL_STATE,
    #                 ),
    #             ),
    #             (
    #                 _strict_kda_state_output_store_consumer,
    #                 (
    #                     store_pipe.reader(),
    #                     output_store_target,
    #                     BT,
    #                     BV,
    #                     NT,
    #                     i_v,
    #                     USE_HOST_DESCRIPTORS,
    #                     bos.to(tl.int64),
    #                     i_h.to(tl.int64) * V,
    #                 ),
    #             ),
    #         ],
    #         [4, 1],
    #         [240, 32],
    #     )

    # def _strict_kda_fwd_state_output(
    #     v: torch.Tensor,
    #     beta: torch.Tensor,
    #     Akk: torch.Tensor,
    #     gk: torch.Tensor,
    #     Aqk: torch.Tensor,
    #     ws: torch.Tensor,
    #     scale: float,
    #     initial_state: torch.Tensor | None = None,
    #     output_final_state: bool = False,
    #     cu_seqlens: torch.LongTensor | None = None,
    #     chunk_size: int = 16,
    # ) -> tuple[torch.Tensor, torch.Tensor | None]:
    #     B, _, HV, packed_K = ws.shape
    #     K = packed_K // 3
    #     T_actual = v.shape[1]
    #     V = v.shape[-1]
    #     BT = chunk_size

    #     if cu_seqlens is None:
    #         N = B
    #         chunk_offsets = None
    #     else:
    #         N = len(cu_seqlens) - 1
    #         chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    #     final_state = None
    #     if output_final_state:
    #         final_state = ws.new_empty(N, HV, V, K, dtype=torch.float32)

    #     o = torch.empty(B, T_actual, HV, V, device=ws.device, dtype=v.dtype)

    #     h0_arg = (
    #         initial_state
    #         if initial_state is not None
    #         else ws.new_empty(1, dtype=torch.float32)
    #     )
    #     ht_arg = (
    #         final_state
    #         if final_state is not None
    #         else ws.new_empty(1, dtype=torch.float32)
    #     )

    #     use_host_descriptors = (
    #         cu_seqlens is None and T_actual % BT == 0 and TensorDescriptor is not None
    #     )
    #     if use_host_descriptors:
    #         NT_total = ws.shape[1] // BT
    #         descriptor_rows = B * HV * NT_total * BT
    #         ws_desc_arg = TensorDescriptor(
    #             ws,
    #             shape=[descriptor_rows, HV * 3 * K],
    #             strides=[HV * 3 * K, 1],
    #             block_shape=[BT, K],
    #         )
    #         gk_desc_arg = TensorDescriptor(
    #             gk,
    #             shape=[gk.shape[0], HV * K],
    #             strides=[HV * K, 1],
    #             block_shape=[1, K],
    #         )
    #         Aqk_desc_arg = TensorDescriptor(
    #             Aqk,
    #             shape=[descriptor_rows, BT],
    #             strides=[BT, 1],
    #             block_shape=[BT, BT],
    #         )
    #         Akk_desc_arg = TensorDescriptor(
    #             Akk,
    #             shape=[descriptor_rows, BT],
    #             strides=[BT, 1],
    #             block_shape=[BT, BT],
    #         )
    #         output_desc_arg = TensorDescriptor(
    #             o,
    #             shape=[B * T_actual, HV * V],
    #             strides=[HV * V, 1],
    #             block_shape=[BT, 128],
    #         )
    #     else:
    #         ws_desc_arg = ws
    #         gk_desc_arg = gk
    #         Aqk_desc_arg = Aqk
    #         Akk_desc_arg = Akk
    #         output_desc_arg = o

    #     _strict_kda_fwd_state_output_kernel[(1, N * HV)](
    #         v=v,
    #         beta=beta,
    #         gk=gk,
    #         Aqk=Aqk,
    #         Akk=Akk,
    #         o=o,
    #         ws=ws,
    #         h0=h0_arg,
    #         ht=ht_arg,
    #         ws_host_desc=ws_desc_arg,
    #         gk_host_desc=gk_desc_arg,
    #         Aqk_host_desc=Aqk_desc_arg,
    #         Akk_host_desc=Akk_desc_arg,
    #         output_host_desc=output_desc_arg,
    #         cu_seqlens=cu_seqlens,
    #         chunk_offsets=chunk_offsets,
    #         scale=scale,
    #         T=T_actual,
    #         NT_TOTAL=ws.shape[1] // BT,
    #         HV=HV,
    #         K=K,
    #         V=V,
    #         BT=BT,
    #         BV=128,
    #         USE_HOST_DESCRIPTORS=use_host_descriptors,
    #         num_warps=4,
    #         NV_MMA_SHARED_LAYOUT=_nv_mma_shared_layout(),
    #     )

    #     return o, final_state

    # =============================================================================
    # Rewritten: strict TLE state output (no tle.pipe / no warp_specialize)
    # =============================================================================

    @triton.heuristics(
        {
            "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
            "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.jit(do_not_specialize=["T"])
    def _strict_kda_fwd_state_output_kernel(
        v,
        beta,
        gk,
        Aqk,
        Akk,
        o,
        ws,
        h0,
        ht,
        cu_seqlens,
        chunk_offsets,
        scale,
        T,
        NT_TOTAL,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
        STORE_FINAL_STATE: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        NV_MMA_SHARED_LAYOUT: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)

        if IS_VARLEN:
            i_n = i_nh // HV
            i_h = i_nh % HV
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = (eos - bos).to(tl.int32)
            NT = tl.cdiv(T, BT)
            chunk_start = tl.load(chunk_offsets + i_n).to(tl.int32)
        else:
            i_n = i_nh // HV
            i_h = i_nh % HV
            bos = i_n.to(tl.int64) * T
            NT = tl.cdiv(T, BT)
            chunk_start = i_n * NT

        v += (bos * HV + i_h) * V
        beta += bos * HV + i_h
        gk += (chunk_start * HV + i_h).to(tl.int64) * K
        o += (bos * HV + i_h) * V
        ws_base = ws + (bos * HV + i_h) * 3 * K

        # Shared memory: single stage
        w_smem = tle.gpu.alloc([BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        v_smem = tle.gpu.alloc([BT, BV], dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        qg_smem = tle.gpu.alloc([BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        kg_smem = tle.gpu.alloc([BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        Aqk_smem = tle.gpu.alloc([BT, BT], dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        Akk_smem = tle.gpu.alloc([BT, BT], dtype=tl.bfloat16, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        gk_smem = tle.gpu.alloc([1, K], dtype=tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)

        # Load initial state
        state_dtype: tl.constexpr = tl.bfloat16
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0))
            b_h = tl.trans(tl.load(p_h0, boundary_check=(0, 1))).to(tl.float32)
        else:
            b_h = tl.zeros([K, BV], dtype=tl.float32)

        # Sequential recurrence across chunks
        for i_t in tl.range(NT):
            # ---- Load phase: plain tl.load into smem ----
            # ws: [B, T_padded, HV, 3*K], accessed via ws_base pointing to [T, 3*K] region
            p_w = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
            tl.store(tle.gpu.local_ptr(w_smem), tl.load(p_w, boundary_check=(0, 1)))
            p_qg = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, K), (BT, K), (1, 0))
            tl.store(tle.gpu.local_ptr(qg_smem), tl.load(p_qg, boundary_check=(0, 1)))
            p_kg = tl.make_block_ptr(ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 2 * K), (BT, K), (1, 0))
            tl.store(tle.gpu.local_ptr(kg_smem), tl.load(p_kg, boundary_check=(0, 1)))

            # Aqk/Akk: stored as [B, HV, NT, BT, BT] in strict path
            # For non-varlen: offset = (i_n * HV + i_h) * NT_TOTAL * BT * BT + i_t * BT * BT
            # For varlen: offset = (i_h * NT_TOTAL + chunk_start + i_t) * BT * BT
            if IS_VARLEN:
                a_offset = (i_h * NT_TOTAL + chunk_start + i_t).to(tl.int64) * BT * BT
            else:
                a_offset = ((i_n * HV + i_h) * NT_TOTAL + i_t).to(tl.int64) * BT * BT
            p_Aqk = tl.make_block_ptr(Aqk + a_offset, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
            tl.store(tle.gpu.local_ptr(Aqk_smem), tl.load(p_Aqk, boundary_check=(0, 1)))
            p_Akk = tl.make_block_ptr(Akk + a_offset, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
            tl.store(tle.gpu.local_ptr(Akk_smem), tl.load(p_Akk, boundary_check=(0, 1)))

            # gk: [NT_total, HV, K] or similar; last row of current chunk
            # gk pointer already offset to (chunk_start * HV + i_h) * K
            p_gk = tl.make_block_ptr(gk, (NT, K), (HV * K, 1), (i_t, 0), (1, K), (1, 0))
            tl.store(tle.gpu.local_ptr(gk_smem), tl.load(p_gk, boundary_check=(0, 1)))

            # v and beta
            p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v_raw = tl.load(p_v, boundary_check=(0, 1))
            p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))
            b_beta_f = tl.sigmoid(b_beta.to(tl.float32))
            b_v = (b_v_raw.to(tl.float32) * b_beta_f[:, None]).to(b_v_raw.dtype)
            tl.store(tle.gpu.local_ptr(v_smem), b_v)

            # ---- Compute phase ----
            b_w = tl.load(tle.gpu.local_ptr(w_smem))
            b_v_raw = tl.load(tle.gpu.local_ptr(v_smem))
            b_qg = tl.load(tle.gpu.local_ptr(qg_smem))
            b_kg = tl.load(tle.gpu.local_ptr(kg_smem))
            b_Aqk = tl.load(tle.gpu.local_ptr(Aqk_smem))
            b_Akk = tl.load(tle.gpu.local_ptr(Akk_smem))
            b_gk = tl.load(tle.gpu.local_ptr(gk_smem)).reshape([K])

            b_h_bf = b_h.to(state_dtype)

            # v_new = Akk_inv @ (v*beta - w @ h)
            b_kh = tl.dot(b_w, b_h_bf).to(tl.float32)
            b_diff = b_v_raw.to(tl.float32) - b_kh
            b_v_new = tl.dot(b_Akk, b_diff.to(state_dtype)).to(tl.float32)

            # output = scale * qg @ h + Aqk @ v_new
            b_qh = tl.dot(b_qg, b_h_bf).to(tl.float32)
            b_o = scale * b_qh
            b_v_cast = b_v_new.to(state_dtype)
            b_o += tl.dot(b_Aqk, b_v_cast).to(tl.float32)

            # ---- Store phase: plain tl.store to global memory ----
            p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

            # ---- State decay + update ----
            b_h = b_h * exp2(b_gk)[:, None] + tl.dot(tl.trans(b_kg), b_v_cast).to(tl.float32)

        # Store final state
        if STORE_FINAL_STATE:
            p_ht = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0))
            tl.store(p_ht, tl.trans(b_h).to(p_ht.dtype.element_ty), boundary_check=(0, 1))

    def _strict_kda_fwd_state_output(
        v: torch.Tensor,
        beta: torch.Tensor,
        Akk: torch.Tensor,
        gk: torch.Tensor,
        Aqk: torch.Tensor,
        ws: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, _, HV, packed_K = ws.shape
        K = packed_K // 3
        T_actual = v.shape[1]
        V = v.shape[-1]
        BT = chunk_size

        if cu_seqlens is None:
            N = B
            chunk_offsets = None
        else:
            N = len(cu_seqlens) - 1
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

        final_state = None
        if output_final_state:
            final_state = ws.new_empty(N, HV, V, K, dtype=torch.float32)

        o = torch.empty(B, T_actual, HV, V, device=ws.device, dtype=v.dtype)

        h0_arg = initial_state if initial_state is not None else ws.new_empty(1, dtype=torch.float32)
        ht_arg = final_state if final_state is not None else ws.new_empty(1, dtype=torch.float32)

        _strict_kda_fwd_state_output_kernel[(1, N * HV)](
            v=v,
            beta=beta,
            gk=gk,
            Aqk=Aqk,
            Akk=Akk,
            o=o,
            ws=ws,
            h0=h0_arg,
            ht=ht_arg,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            scale=scale,
            T=T_actual,
            NT_TOTAL=ws.shape[1] // BT,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BV=128,
            num_warps=4,
            NV_MMA_SHARED_LAYOUT=_nv_mma_shared_layout(),
        )

        return o, final_state


def chunk_kda_fwd_infer_strict_tle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    reason = strict_tle_input_error(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
    )
    if reason is not None:
        raise ValueError(reason)

    triton.set_allocator(_allocate_triton_workspace)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    ws, Aqk, Akk, g_last = _strict_kda_fwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    return _strict_kda_fwd_state_output(
        v=v,
        beta=beta,
        Akk=Akk,
        gk=g_last,
        Aqk=Aqk,
        ws=ws,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


# =============================================================================
# Triton fallback path (portable, used when TLE is unavailable)
# =============================================================================


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
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [16, 32, 64]
        for BV in [16, 32, 64]
        for num_warps in [1, 2, 4]
        for num_stages in [1, 2, 4]
    ],
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
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
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
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
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
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * b_q_rstd[:, None]
            b_k = b_k * b_k_rstd[:, None]
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                p_dt = tl.make_block_ptr(dt_bias + i_hv * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
                b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
                b_g = b_g + b_bias[None, :]
            if USE_LOWER_BOUND:
                b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
            else:
                b_g = -b_A * _softplus(b_g) * g_scale
        else:
            b_g = b_g * g_scale
        b_g = tl.cumsum(b_g, axis=0)

        p_g_out = tl.make_block_ptr(g_out, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_g_out, b_g.to(g_out.dtype.element_ty), boundary_check=(0, 1))

        b_gq = tl.where(m_c[:, None], exp2(b_g), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-b_g), 0.0)

        b_kgt = tl.trans(b_k * b_gk)
        b_Aqk += tl.dot(b_q * b_gq, b_kgt)
        b_Akk += tl.dot(b_k * b_gq, b_kgt)

    # Causal mask
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    # Phase 2: Solve (I + L)^{-1} via parallel prefix
    b_L = b_Akk.to(tl.float16)
    b_Ai = m_I.to(tl.float16) - b_L
    b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
    b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
    b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float16)

    p_Akk_out = tl.make_block_ptr(Akk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_Akk_out, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    # Phase 3: w, u, qg, kg
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_Ai.to(b_vb.dtype), b_vb)
        tl.store(p_u, b_u.to(u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gk = tl.make_block_ptr(g_out, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * b_k_rstd[:, None]
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb = b_k * b_beta[:, None] * exp2(b_gk)

        if STORE_QG:
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_qg_out = tl.make_block_ptr(qg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * b_q_rstd[:, None]
            b_qg_val = b_q * exp2(b_gk)
            tl.store(p_qg_out, b_qg_val.to(qg.dtype.element_ty), boundary_check=(0, 1))

        if STORE_KG:
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            last_idx = tl.minimum(i_t * BT + BT, T) - 1
            b_gn = tl.load(g_out + last_idx * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_kg_val = b_k * tl.where(m_c[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg_out = tl.make_block_ptr(kg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_kg_out, b_kg_val.to(kg.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
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
    configs=[triton.Config({"BV": BV}, num_warps=num_warps) for BV in [32, 64] for num_warps in [2, 4]],
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
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
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
            p_h0_1 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h0_2 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h0_3 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h0_4 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
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
            p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_u = tl.make_block_ptr(u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v

        # output = scale * qg @ h + Aqk @ v_new
        p_qg = tl.make_block_ptr(qg, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_o = tl.dot(b_qg, tl.trans(b_h1).to(b_qg.dtype))
        else:
            b_o = tl.dot(b_qg, b_h1.to(b_qg.dtype))
        if K > 64:
            p_qg = tl.make_block_ptr(qg, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h2).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h2.to(b_qg.dtype))
        if K > 128:
            p_qg = tl.make_block_ptr(qg, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h3).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h3.to(b_qg.dtype))
        if K > 192:
            p_qg = tl.make_block_ptr(qg, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h4).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h4.to(b_qg.dtype))
        b_o *= scale

        p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
        b_o += tl.dot(b_Aqk.to(b_v.dtype), b_v)

        p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # decay: h *= exp2(gk_last)
        last_idx = tl.minimum(i_t * BT + BT, T) - 1
        o_k1 = tl.arange(0, 64)
        b_gk_last1 = tl.load(gk + last_idx * HV * K + o_k1, mask=(o_k1 < K), other=0.0).to(tl.float32)
        if STATE_V_FIRST:
            b_h1 *= exp2(b_gk_last1)[None, :]
        else:
            b_h1 *= exp2(b_gk_last1)[:, None]
        if K > 64:
            o_k2 = 64 + o_k1
            b_gk_last2 = tl.load(gk + last_idx * HV * K + o_k2, mask=(o_k2 < K), other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                b_h2 *= exp2(b_gk_last2)[None, :]
            else:
                b_h2 *= exp2(b_gk_last2)[:, None]
        if K > 128:
            o_k3 = 128 + o_k1
            b_gk_last3 = tl.load(gk + last_idx * HV * K + o_k3, mask=(o_k3 < K), other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                b_h3 *= exp2(b_gk_last3)[None, :]
            else:
                b_h3 *= exp2(b_gk_last3)[:, None]
        if K > 192:
            o_k4 = 192 + o_k1
            b_gk_last4 = tl.load(gk + last_idx * HV * K + o_k4, mask=(o_k4 < K), other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                b_h4 *= exp2(b_gk_last4)[None, :]
            else:
                b_h4 *= exp2(b_gk_last4)[:, None]

        # state update: h += kg^T @ v_new
        b_v = b_v.to(kg.dtype.element_ty)
        p_kg = tl.make_block_ptr(kg, (K, T), (1, HV * K), (0, i_t * BT), (64, BT), (0, 1))
        b_kg = tl.load(p_kg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_kg, b_v))
        else:
            b_h1 += tl.dot(b_kg, b_v)
        if K > 64:
            p_kg = tl.make_block_ptr(kg, (K, T), (1, HV * K), (64, i_t * BT), (64, BT), (0, 1))
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h2 += tl.dot(b_kg, b_v)
        if K > 128:
            p_kg = tl.make_block_ptr(kg, (K, T), (1, HV * K), (128, i_t * BT), (64, BT), (0, 1))
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h3 += tl.dot(b_kg, b_v)
        if K > 192:
            p_kg = tl.make_block_ptr(kg, (K, T), (1, HV * K), (192, i_t * BT), (64, BT), (0, 1))
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h4 += tl.dot(b_kg, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
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
    """Plain-Triton inference forward for chunk KDA (TLE-free fallback)."""
    BT = chunk_size

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

    return _kda_fwd_h_o_triton(
        kg=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        qg=qg,
        Aqk=Aqk,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
    )


# =============================================================================
# Input validation (shared by both paths)
# =============================================================================


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
        raise RuntimeError("chunk_kda TLE path only supports inference/no-grad mode")
    if chunk_size != 16:
        raise ValueError(f"chunk_kda TLE path requires chunk_size=16, got {chunk_size}")
    supported_dtypes = (torch.bfloat16, torch.float16)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if tensor.dtype not in supported_dtypes:
            raise ValueError(f"chunk_kda TLE path requires {name} dtype to be bf16 or fp16, got {tensor.dtype}")
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
        raise ValueError(f"chunk_kda TLE path requires K in {{64, 128, 192, 256}}, got {K}")
    if V <= 0:
        raise ValueError(f"chunk_kda TLE path requires V > 0, got {V}")
    if HV < H or HV % H != 0:
        raise ValueError(f"chunk_kda TLE path requires HV % H == 0, got H={H}, HV={HV}")

    if not use_qk_l2norm_in_kernel:
        raise ValueError("chunk_kda TLE path requires use_qk_l2norm_in_kernel=True")
    if not use_gate_in_kernel:
        raise ValueError("chunk_kda TLE path requires use_gate_in_kernel=True")
    if not use_beta_sigmoid_in_kernel:
        raise ValueError("chunk_kda TLE path requires use_beta_sigmoid_in_kernel=True")
    if allow_neg_eigval:
        raise ValueError("chunk_kda TLE path does not support allow_neg_eigval=True")
    if not safe_gate:
        raise ValueError("chunk_kda TLE path requires safe_gate=True")
    if lower_bound is None:
        raise ValueError("chunk_kda TLE path requires lower_bound")
    if A_log is None:
        raise ValueError("chunk_kda TLE path requires A_log")
    if A_log.device != q.device:
        raise ValueError("A_log must be on the same device as q")
    if A_log.numel() != HV:
        raise ValueError(f"A_log.numel() must be HV={HV}, got {A_log.numel()}")
    if dt_bias is None:
        raise ValueError("chunk_kda TLE path requires dt_bias")
    if dt_bias.device != q.device:
        raise ValueError("dt_bias must be on the same device as q")
    if dt_bias.numel() != HV * K:
        raise ValueError(f"dt_bias.numel() must be HV*K={HV * K}, got {dt_bias.numel()}")

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
            raise ValueError(f"initial_state must have shape {expected_shape}, got {tuple(initial_state.shape)}")


# =============================================================================
# Public entry: validate, then dispatch (TLE if available, else Triton)
# =============================================================================


def chunk_kda(
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
    allow_neg_eigval: bool = False,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Inference-only implementation of chunk Kimi Delta Attention.

    Inputs use seq-first layout: q/k ``[B, T, H, K]``, v/g ``[B, T, HV, *]``,
    and beta ``[B, T, HV]``. q/k L2 norm, gate activation, and beta sigmoid are
    computed inside the kernels. Backend selection defaults to automatic and
    can be forced with ``FLAGGEMS_CHUNK_KDA_BACKEND``.
    """
    A_log = kwargs.get("A_log")
    dt_bias = kwargs.get("dt_bias")

    _validate_chunk_kda_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    backend = _chunk_kda_backend()
    strict_tle_reason = strict_tle_input_error(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
    )
    if backend == "strict_tle" and strict_tle_reason is not None:
        raise RuntimeError(f"{_BACKEND_ENV}=strict_tle was requested, but {strict_tle_reason}")
    if backend == "tle" and not HAS_TLE_KDA:
        raise RuntimeError(f"{_BACKEND_ENV}=tle requires Triton TLE >= 3.6.0")

    if backend in {"auto", "strict_tle"} and strict_tle_reason is None:
        return chunk_kda_fwd_infer_strict_tle(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    if backend in {"auto", "tle"} and HAS_TLE_KDA:
        return chunk_kda_fwd_infer(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    return chunk_kda_fwd_infer_triton(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )
