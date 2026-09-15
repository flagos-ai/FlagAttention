# This file contains code derived from the flash-linear-attention project.
# The original source code is licensed under the MIT license.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

import os

import torch
import triton
import triton.language as tl

from flag_attn.FLA.compat import libentry, libtuner
from flag_attn.utils import has_triton_tle

TWO_KERNEL_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_TWO_KERNEL_TLE"
if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TWO_KERNEL_TLE = True
    except ImportError:
        tle = None
        HAS_TWO_KERNEL_TLE = False
else:
    tle = None
    HAS_TWO_KERNEL_TLE = False


def can_use_two_kernel_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    enabled = os.environ.get(TWO_KERNEL_TLE_ENV, "1").lower()
    if not (HAS_TWO_KERNEL_TLE and enabled not in {"0", "false", "off", "no"}):
        return False
    if initial_state is not None or not output_final_state or cu_seqlens is not None:
        return False
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not all(x.device == q.device and x.dtype == q.dtype for x in (k, v, beta, g)):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if beta.ndim != 3 or g.ndim != 3:
        return False

    B, T, Hg, K = q.shape
    H, V = v.shape[2:]
    return (
        k.shape == (B, T, Hg, K)
        and v.shape[:2] == (B, T)
        and beta.shape == (B, T, H)
        and g.shape == (B, T, H)
        and H % Hg == 0
        and B * H >= 64
        and K in {64, 128, 256}
        and V > 0
        and T > 0
        and T % 64 == 0
    )


if HAS_TWO_KERNEL_TLE:

    @libentry()
    @libtuner(
        configs=[
            triton.Config({"BV": 64}, num_warps=4, num_stages=1),
            triton.Config({"BV": 64}, num_warps=4, num_stages=3),
            triton.Config({"BV": 32}, num_warps=2, num_stages=2),
        ],
        key=["H", "Hg", "K", "V", "BT"],
    )
    @triton.jit(do_not_specialize=["T"])
    def _chunk_gdn_two_kernel_fwd_kernel(
        q,
        k,
        v,
        beta,
        A,
        g,
        o,
        final_state,
        scale,
        T,
        H: tl.constexpr,
        Hg: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_v, i_bh = tl.program_id(0), tl.program_id(1)
        i_b, i_h = i_bh // H, i_bh % H
        i_qh = i_h // (H // Hg)
        bos = i_b * T
        NT = tl.cdiv(T, BT)

        q += (bos * Hg + i_qh).to(tl.int64) * K
        k += (bos * Hg + i_qh).to(tl.int64) * K
        v += (bos * H + i_h).to(tl.int64) * V
        beta += bos * H + i_h
        A += (bos * H + i_h).to(tl.int64) * BT
        g += bos * H + i_h
        o += (bos * H + i_h).to(tl.int64) * V

        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T

            p_A = tl.make_block_ptr(
                A,
                (T, BT),
                (H * BT, 1),
                (i_t * BT, 0),
                (BT, BT),
                (1, 0),
            )
            b_inv = tl.load(p_A, boundary_check=(0, 1))
            p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))

            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + last_idx * H).to(tl.float32)
            p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tle.load(p_g, boundary_check=(0,), is_async=True).to(tl.float32)
            b_g_exp = tl.math.exp2(b_g * 1.4426950408889634)
            b_g_last_exp = tl.math.exp2(b_g_last * 1.4426950408889634)

            p_v = tl.make_block_ptr(
                v,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            b_v_input = tle.load(p_v, boundary_check=(0, 1), is_async=True)
            b_v_beta = (b_v_input * b_beta[:, None]).to(b_v_input.dtype)
            b_u = tl.dot(b_inv, b_v_beta, allow_tf32=False)
            b_v_new = b_u.to(v.dtype.element_ty).to(tl.float32)

            p_k1 = tl.make_block_ptr(
                k, (T, K), (Hg * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
            )
            b_k1 = tle.load(p_k1, boundary_check=(0, 1), is_async=True)
            b_kb1 = b_k1 * b_beta[:, None] * b_g_exp[:, None]
            b_w1 = tl.dot(b_inv, b_kb1.to(b_k1.dtype))
            b_v_new -= tl.dot(b_w1.to(b_k1.dtype), b_h1.to(b_k1.dtype))

            if K > 64:
                p_k2 = tl.make_block_ptr(
                    k,
                    (T, K),
                    (Hg * K, 1),
                    (i_t * BT, 64),
                    (BT, 64),
                    (1, 0),
                )
                b_k2 = tle.load(p_k2, boundary_check=(0, 1), is_async=True)
                b_kb2 = b_k2 * b_beta[:, None] * b_g_exp[:, None]
                b_w2 = tl.dot(b_inv, b_kb2.to(b_k2.dtype))
                b_v_new -= tl.dot(b_w2.to(b_k2.dtype), b_h2.to(b_k2.dtype))

            if K > 128:
                p_k3 = tl.make_block_ptr(
                    k,
                    (T, K),
                    (Hg * K, 1),
                    (i_t * BT, 128),
                    (BT, 64),
                    (1, 0),
                )
                b_k3 = tle.load(p_k3, boundary_check=(0, 1), is_async=True)
                b_kb3 = b_k3 * b_beta[:, None] * b_g_exp[:, None]
                b_w3 = tl.dot(b_inv, b_kb3.to(b_k3.dtype))
                b_v_new -= tl.dot(b_w3.to(b_k3.dtype), b_h3.to(b_k3.dtype))

            if K > 192:
                p_k4 = tl.make_block_ptr(
                    k,
                    (T, K),
                    (Hg * K, 1),
                    (i_t * BT, 192),
                    (BT, 64),
                    (1, 0),
                )
                b_k4 = tle.load(p_k4, boundary_check=(0, 1), is_async=True)
                b_kb4 = b_k4 * b_beta[:, None] * b_g_exp[:, None]
                b_w4 = tl.dot(b_inv, b_kb4.to(b_k4.dtype))
                b_v_new -= tl.dot(b_w4.to(b_k4.dtype), b_h4.to(b_k4.dtype))

            b_v_decay = (
                b_v_new
                * tl.where(
                    m_t,
                    tl.math.exp2((b_g_last - b_g) * 1.4426950408889634),
                    0,
                )[:, None]
            )
            b_v_state = b_v_decay.to(k.dtype.element_ty)
            b_v_o = b_v_new.to(k.dtype.element_ty)
            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            b_qk = tl.zeros([BT, BT], dtype=tl.float32)

            p_q1 = tl.make_block_ptr(
                q, (T, K), (Hg * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
            )
            p_kt1 = tl.make_block_ptr(
                k, (K, T), (1, Hg * K), (0, i_t * BT), (64, BT), (0, 1)
            )
            b_q1 = tle.load(p_q1, boundary_check=(0, 1), is_async=True)
            b_kt1 = tle.load(p_kt1, boundary_check=(0, 1), is_async=True)
            b_o += tl.dot(b_q1, b_h1.to(b_q1.dtype))
            b_qk += tl.dot(b_q1, b_kt1)
            b_h1 *= b_g_last_exp
            b_h1 += tl.dot(b_kt1, b_v_state)

            if K > 64:
                p_q2 = tl.make_block_ptr(
                    q,
                    (T, K),
                    (Hg * K, 1),
                    (i_t * BT, 64),
                    (BT, 64),
                    (1, 0),
                )
                p_kt2 = tl.make_block_ptr(
                    k,
                    (K, T),
                    (1, Hg * K),
                    (64, i_t * BT),
                    (64, BT),
                    (0, 1),
                )
                b_q2 = tle.load(p_q2, boundary_check=(0, 1), is_async=True)
                b_kt2 = tle.load(p_kt2, boundary_check=(0, 1), is_async=True)
                b_o += tl.dot(b_q2, b_h2.to(b_q2.dtype))
                b_qk += tl.dot(b_q2, b_kt2)
                b_h2 *= b_g_last_exp
                b_h2 += tl.dot(b_kt2, b_v_state)

            if K > 128:
                p_q3 = tl.make_block_ptr(
                    q,
                    (T, K),
                    (Hg * K, 1),
                    (i_t * BT, 128),
                    (BT, 64),
                    (1, 0),
                )
                p_kt3 = tl.make_block_ptr(
                    k,
                    (K, T),
                    (1, Hg * K),
                    (128, i_t * BT),
                    (64, BT),
                    (0, 1),
                )
                b_q3 = tle.load(p_q3, boundary_check=(0, 1), is_async=True)
                b_kt3 = tle.load(p_kt3, boundary_check=(0, 1), is_async=True)
                b_o += tl.dot(b_q3, b_h3.to(b_q3.dtype))
                b_qk += tl.dot(b_q3, b_kt3)
                b_h3 *= b_g_last_exp
                b_h3 += tl.dot(b_kt3, b_v_state)

            if K > 192:
                p_q4 = tl.make_block_ptr(
                    q,
                    (T, K),
                    (Hg * K, 1),
                    (i_t * BT, 192),
                    (BT, 64),
                    (1, 0),
                )
                p_kt4 = tl.make_block_ptr(
                    k,
                    (K, T),
                    (1, Hg * K),
                    (192, i_t * BT),
                    (64, BT),
                    (0, 1),
                )
                b_q4 = tle.load(p_q4, boundary_check=(0, 1), is_async=True)
                b_kt4 = tle.load(p_kt4, boundary_check=(0, 1), is_async=True)
                b_o += tl.dot(b_q4, b_h4.to(b_q4.dtype))
                b_qk += tl.dot(b_q4, b_kt4)
                b_h4 *= b_g_last_exp
                b_h4 += tl.dot(b_kt4, b_v_state)

            b_o *= b_g_exp[:, None]
            b_qk *= tl.math.exp2((b_g[:, None] - b_g[None, :]) * 1.4426950408889634)
            m_qk = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
            b_qk = tl.where(m_qk, b_qk, 0)
            b_o = (b_o + tl.dot(b_qk.to(b_v_o.dtype), b_v_o)) * scale

            p_o = tl.make_block_ptr(
                o,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        o_k = tl.arange(0, 64)
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        state_base = final_state + (i_bh * K * V).to(tl.int64)
        tl.store(
            state_base + o_k[:, None] * V + o_v[None, :],
            b_h1,
            mask=m_v[None, :],
        )
        if K > 64:
            tl.store(
                state_base + (o_k[:, None] + 64) * V + o_v[None, :],
                b_h2,
                mask=m_v[None, :],
            )
        if K > 128:
            tl.store(
                state_base + (o_k[:, None] + 128) * V + o_v[None, :],
                b_h3,
                mask=m_v[None, :],
            )
        if K > 192:
            tl.store(
                state_base + (o_k[:, None] + 192) * V + o_v[None, :],
                b_h4,
                mask=m_v[None, :],
            )


def chunk_gdn_two_kernel_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K = q.shape
    H, V = v.shape[2:]
    BT = A.shape[-1]
    o = torch.empty_like(v)
    final_state = torch.empty((B, H, K, V), dtype=torch.float32, device=v.device)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), B * H)

    _chunk_gdn_two_kernel_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        o=o,
        final_state=final_state,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return o, final_state
