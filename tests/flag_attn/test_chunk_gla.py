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

# Self-contained fused_recurrent_gla implementation
# Inlined from:
#   fla/ops/gla/fused_recurrent.py
#   fla/ops/common/fused_recurrent.py
#   fla/ops/utils/op.py (exp only)
#   fla/utils.py (autocast, input_guard, autotune_cache_kwargs)

import contextlib
import functools
import inspect
import inspect as _inspect
import os
import os as _os

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_attn.FLA.gated_linear_attention import chunk_gla
from flag_attn.FLA.index import (
    prepare_chunk_indices as _prepare_chunk_indices,
)


# --- From fla/ops/utils/op.py: exp ---
@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


# --- From fla/utils.py: autotune_cache_kwargs ---
if "cache_results" in _inspect.signature(triton.autotune).parameters:
    _autotune_cache_kwargs = {
        "cache_results": _os.getenv("FLA_CACHE_RESULTS", "1") == "1"
    }
else:
    _autotune_cache_kwargs = {}

# --- From fla/utils.py: autocast helpers ---
_autocast_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
_autocast_custom_bwd = torch.amp.custom_bwd(device_type="cuda")


# --- From fla/utils.py: input_guard (minimal) ---
def _input_guard(fn=None, *, no_guard_contiguous=False):
    def _decorator(fn):
        sig = _inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        skip_params = set()
        if isinstance(no_guard_contiguous, list):
            skip_params = set(no_guard_contiguous)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            processed_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    pname = param_names[i]
                else:
                    pname = "__arg_{}".format(i)
                if isinstance(arg, torch.Tensor):
                    if no_guard_contiguous is True or pname in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    if no_guard_contiguous is True or k in skip_params:
                        processed_kwargs[k] = v
                    else:
                        processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor):
                        tensor = v
                        break

            if tensor is not None and tensor.device.type == "cuda":
                ctx = torch.cuda.device(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    if fn is not None:
        return _decorator(fn)
    return _decorator


# --- From fla/ops/common/fused_recurrent.py: fwd kernel ---
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [4, 8]],
    key=["BK", "BV", "USE_G", "USE_G_GAMMA", "USE_GK", "USE_GV", "STATE_V_FIRST"],
    **_autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["B", "T"])
def _fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr = False,
):
    i_v, i_k, i_nh = (
        tl.program_id(0).to(tl.int64),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_o = o + ((i_k * all + bos) + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    if USE_G:
        p_g = g + (bos + ((T - 1) if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    if USE_GV:
        p_gv = gv + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)

    m_k = o_k < K
    m_v = o_v < V
    if STATE_V_FIRST:
        m_h = m_v[:, None] & m_k[None, :]
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        m_h = m_k[:, None] & m_v[None, :]
        b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = h0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h = b_h * exp(b_gk[None, :])
            else:
                b_h = b_h * exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h = b_h * exp(b_gv[:, None])
            else:
                b_h = b_h * exp(b_gv[None, :])
        if STATE_V_FIRST:
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], axis=1)
        else:
            b_h += b_k[:, None] * b_v[None, :]
            b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
        p_q += (-1 if REVERSE else 1) * H * K
        p_k += (-1 if REVERSE else 1) * H * K
        p_v += (-1 if REVERSE else 1) * H * V
        p_o += (-1 if REVERSE else 1) * H * V
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * H * K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * H * V

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


# --- From fla/ops/common/fused_recurrent.py: bwd kernel ---
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_INITIAL_STATE_GRADIENT": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [4]],
    key=["BK", "BV", "USE_G", "USE_G_GAMMA", "USE_GK", "USE_GV", "STATE_V_FIRST"],
    **_autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["B", "T"])
def _fused_recurrent_bwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    o,
    h0,
    do,
    dq,
    dk,
    dv,
    dg,
    dgk,
    dgv,
    dht,
    dh0,
    cu_seqlens,
    scale,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr = False,
):
    i_v, i_k, i_nh = (
        tl.program_id(0).to(tl.int64),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NV = tl.cdiv(V, BV)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    if STATE_V_FIRST:
        m_h = m_v[:, None] & m_k[None, :]
    else:
        m_h = m_k[:, None] & m_v[None, :]

    p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_do = do + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_dq = (
        dq + ((i_v * all + bos) + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    )
    if USE_G:
        p_g = g + (bos + ((T - 1) if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    if USE_GV:
        p_gv = gv + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)

    if STATE_V_FIRST:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = h0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h = b_h * exp(b_gk[None, :])
            else:
                b_h = b_h * exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h = b_h * exp(b_gv[:, None])
            else:
                b_h = b_h * exp(b_gv[None, :])
        if STATE_V_FIRST:
            b_h += b_v[:, None] * b_k[None, :]
            b_dq = tl.sum(b_h * b_do[:, None], axis=0) * scale
        else:
            b_h += b_k[:, None] * b_v[None, :]
            b_dq = tl.sum(b_h * b_do[None, :], axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_k)

        p_k += (-1 if REVERSE else 1) * H * K
        p_v += (-1 if REVERSE else 1) * H * V
        p_do += (-1 if REVERSE else 1) * H * V
        p_dq += (-1 if REVERSE else 1) * H * K
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * H * K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * H * V

    # sync threads
    tl.debug_barrier()

    p_q = q + (bos + ((T - 1) if not REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if not REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if not REVERSE else 0)) * H * V + i_h * V + o_v

    p_do = do + (bos + ((T - 1) if not REVERSE else 0)) * H * V + i_h * V + o_v
    p_dq = (
        dq
        + ((i_v * all + bos) + ((T - 1) if not REVERSE else 0)) * H * K
        + i_h * K
        + o_k
    )
    p_dk = (
        dk
        + ((i_v * all + bos) + ((T - 1) if not REVERSE else 0)) * H * K
        + i_h * K
        + o_k
    )
    p_dv = (
        dv
        + ((i_k * all + bos) + ((T - 1) if not REVERSE else 0)) * H * V
        + i_h * V
        + o_v
    )
    if USE_G:
        p_g = g + (bos + ((T - 1) if not REVERSE else 0)) * H + i_h
        p_dg = (
            dg
            + ((i_k * NV + i_v) * all + bos + ((T - 1) if not REVERSE else 0)) * H
            + i_h
        )
    if USE_GK:
        p_gk = gk + (bos + ((T - 1) if not REVERSE else 0)) * H * K + i_h * K + o_k
        p_dgk = (
            dgk
            + ((i_v * all + bos) + ((T - 1) if not REVERSE else 0)) * H * K
            + i_h * K
            + o_k
        )
    if USE_GV:
        p_o = o + (bos + ((T - 1) if not REVERSE else 0)) * H * V + i_h * V + o_v
        p_gv = gv + (bos + ((T - 1) if not REVERSE else 0)) * H * V + i_h * V + o_v
        p_dgv = (
            dgv
            + ((i_k * all + bos) + ((T - 1) if not REVERSE else 0)) * H * V
            + i_h * V
            + o_v
        )

    if STATE_V_FIRST:
        b_dh = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dht = dht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_dht = dht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_dh += tl.load(p_dht, mask=m_h, other=0).to(tl.float32)

    if USE_G:
        b_dg = tl.sum(b_h * b_dh)
    if USE_GK:
        if STATE_V_FIRST:
            b_dgk = tl.sum(b_h * b_dh, 0)
        else:
            b_dgk = tl.sum(b_h * b_dh, 1)
    if USE_GV:
        if STATE_V_FIRST:
            b_dgv = tl.sum(b_h * b_dh, 1)
        else:
            b_dgv = tl.sum(b_h * b_dh, 0)

    for _ in range(T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        if STATE_V_FIRST:
            b_dh += b_do[:, None] * (b_q * scale)[None, :]
            b_dk = tl.sum(b_dh * b_v[:, None], axis=0)
            b_dv = tl.sum(b_dh * b_k[None, :], axis=1)
        else:
            b_dh += (b_q * scale)[:, None] * b_do[None, :]
            b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
            b_dv = tl.sum(b_dh * b_k[:, None], axis=0)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_dq = tl.load(p_dq, mask=m_k, other=0).to(tl.float32)
            b_dg += tl.sum(b_q * b_dq - b_k * b_dk)
            b_dh *= exp(b_g)
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty))
        if USE_G_GAMMA:
            b_dh *= exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_dq = tl.load(p_dq, mask=m_k, other=0).to(tl.float32)
            b_dgk += b_q * b_dq - b_k * b_dk
            if STATE_V_FIRST:
                b_dh *= exp(b_gk)[None, :]
            else:
                b_dh *= exp(b_gk)[:, None]
            tl.store(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), mask=m_k)
        if USE_GV:
            b_o = tl.load(p_o, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            if i_k == 0:
                b_dgv += b_o * b_do
            b_dgv -= b_v * b_dv
            if STATE_V_FIRST:
                b_dh *= exp(b_gv)[:, None]
            else:
                b_dh *= exp(b_gv)[None, :]
            tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), mask=m_v)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)

        p_q += (1 if REVERSE else -1) * H * K
        p_k += (1 if REVERSE else -1) * H * K
        p_v += (1 if REVERSE else -1) * H * V

        p_do += (1 if REVERSE else -1) * H * V
        p_dq += (1 if REVERSE else -1) * H * K
        p_dk += (1 if REVERSE else -1) * H * K
        p_dv += (1 if REVERSE else -1) * H * V
        if USE_G:
            p_g += (1 if REVERSE else -1) * H
            p_dg += (1 if REVERSE else -1) * H
        if USE_GK:
            p_gk += (1 if REVERSE else -1) * H * K
            p_dgk += (1 if REVERSE else -1) * H * K
        if USE_GV:
            p_o += (1 if REVERSE else -1) * H * V
            p_gv += (1 if REVERSE else -1) * H * V
            p_dgv += (1 if REVERSE else -1) * H * V

    if STORE_INITIAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dh0 = dh0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_dh0 = dh0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=m_h)


# --- From fla/ops/common/fused_recurrent.py: Python wrappers ---


def _fused_recurrent_fwd(
    q,
    k,
    v,
    g=None,
    g_gamma=None,
    gk=None,
    gv=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    reverse=False,
    state_v_first=False,
    cu_seqlens=None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    if output_final_state:
        if state_v_first:
            ht = q.new_empty(N, H, V, K, dtype=torch.float32)
        else:
            ht = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        ht = None
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    _fused_recurrent_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        o=o,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
        STATE_V_FIRST=state_v_first,
    )
    o = o.sum(0)
    return o, ht


def _fused_recurrent_bwd(
    q,
    k,
    v,
    g=None,
    g_gamma=None,
    gk=None,
    gv=None,
    o=None,
    do=None,
    dht=None,
    scale=None,
    initial_state=None,
    reverse=False,
    state_v_first=False,
    cu_seqlens=None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    dq = q.new_empty(NV, *q.shape, dtype=torch.float32)
    dk = q.new_empty(NV, *k.shape, dtype=torch.float32)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float32)
    dh0 = torch.empty_like(h0) if h0 is not None else None

    dg, dgk, dgv = None, None, None
    if g is not None:
        dg = g.new_empty(NK * NV, *g.shape, dtype=torch.float32)
    if gk is not None:
        dgk = gk.new_empty(NV, *gk.shape, dtype=torch.float32)
    if gv is not None:
        dgv = gv.new_empty(NK, *gv.shape, dtype=torch.float32)

    grid = (NV, NK, N * H)
    _fused_recurrent_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        o=o,
        h0=h0,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        dgk=dgk,
        dgv=dgv,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
        STATE_V_FIRST=state_v_first,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    if g is not None:
        dg = dg.sum(0).to(g)
    if gk is not None:
        dgk = dgk.sum(0).to(gk)
    if gv is not None:
        dgv = dgv.sum(0).to(gv)

    return dq, dk, dv, dg, dgk, dgv, dh0


# --- From fla/ops/common/fused_recurrent.py: autograd Function ---
class _FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @_input_guard
    @_autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g=None,
        g_gamma=None,
        gk=None,
        gv=None,
        scale=None,
        initial_state=None,
        output_final_state=False,
        reverse=False,
        state_v_first=False,
        cu_seqlens=None,
    ):
        o, ht = _fused_recurrent_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            gk=gk,
            gv=gv,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
        )
        ctx.save_for_backward(q, k, v, g, g_gamma, gk, gv, initial_state, o)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        ctx.state_v_first = state_v_first
        return o.to(q.dtype), ht

    @staticmethod
    @_input_guard
    @_autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, g_gamma, gk, gv, initial_state, o = ctx.saved_tensors
        dq, dk, dv, dg, dgk, dgv, dh0 = _fused_recurrent_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            gk=gk,
            gv=gv,
            o=o,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            reverse=ctx.reverse,
            cu_seqlens=ctx.cu_seqlens,
            state_v_first=ctx.state_v_first,
        )
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg,
            None,
            dgk,
            dgv,
            None,
            dh0,
            None,
            None,
            None,
            None,
        )


# --- From fla/ops/common/fused_recurrent.py: fused_recurrent ---


def _fused_recurrent(
    q,
    k,
    v,
    g=None,
    g_gamma=None,
    gk=None,
    gv=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    reverse=False,
    state_v_first=False,
    cu_seqlens=None,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return _FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        g_gamma,
        gk,
        gv,
        scale,
        initial_state,
        output_final_state,
        reverse,
        state_v_first,
        cu_seqlens,
    )


# --- From fla/ops/gla/fused_recurrent.py: fused_recurrent_gla ---


def fused_recurrent_gla(
    q,
    k,
    v,
    gk=None,
    gv=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    reverse=False,
    state_v_first=False,
    cu_seqlens=None,
    **kwargs,
):
    if "transpose_state_layout" in kwargs:
        if state_v_first:
            raise ValueError(
                "Cannot pass both state_v_first and the deprecated transpose_state_layout."
            )
        import warnings as _w

        _w.warn(
            "transpose_state_layout is deprecated and renamed to state_v_first.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop("transpose_state_layout")

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                "The batch size is expected to be 1 rather than {} when using cu_seqlens."
                "Please flatten variable-length inputs before processing.".format(
                    q.shape[0]
                ),
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "The number of initial states is expected to be equal to the number of input sequences, "
                "i.e., {} rather than {}.".format(
                    len(cu_seqlens) - 1, initial_state.shape[0]
                ),
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = _fused_recurrent(
        q=q,
        k=k,
        v=v,
        g=None,
        g_gamma=None,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        state_v_first=state_v_first,
    )
    return o, final_state


def _cuda_available() -> bool:
    return torch.cuda.is_available()


pytestmark = [
    pytest.mark.skipif(not _cuda_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module", autouse=True)
def _compat_prepare_chunk_indices_kwarg():
    sig = inspect.signature(_prepare_chunk_indices)
    if "cu_seqlens_cpu" in sig.parameters:
        yield
        return
    import sys

    mod = sys.modules["flag_attn.FLA.gated_linear_attention.chunk_gla"]
    old = mod.prepare_chunk_indices

    def _wrapped(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
        del cu_seqlens_cpu
        return _prepare_chunk_indices(cu_seqlens, chunk_size)

    mod.prepare_chunk_indices = _wrapped
    try:
        yield
    finally:
        mod.prepare_chunk_indices = old


def _assert_close(name, actual, expected, ratio, err_atol=1e-6):
    abs_atol = (actual.detach() - expected.detach()).flatten().abs().max().item()
    error_rate = (
        (actual.detach() - expected.detach()).flatten().square().mean().sqrt()
        / actual.detach().flatten().square().mean().sqrt().clamp_min(1e-8)
    ).item()
    print(f"[{name}] diff: {abs_atol:.6f} ratio: {error_rate:.6f}")
    if abs_atol <= err_atol:
        return
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in expected"
    assert error_rate < ratio, f"{name}: diff: {abs_atol:.6f} ratio: {error_rate:.6f}"


@pytest.mark.chunk_gla_chunk
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(*c, id="B{}-T{}-H{}-D{}-gate_logit_normalizer{}-{}".format(*c))
        for c in [
            (1, 63, 1, 64, 1.0, torch.float16),
            (2, 1024, 4, 60, 1.0, torch.float16),
            (2, 1024, 8, 128, 0.1, torch.float16),
            (2, 1024, 8, 128, 1.0, torch.float16),
            (2, 1024, 8, 128, 10.0, torch.float16),
            (4, 2048, 8, 64, 1.0, torch.float16),
        ]
    ],
)
def test_chunk(B, T, H, D, dtype, gate_logit_normalizer):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    device = torch.device("cuda")

    q = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    k = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    v = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    g = (
        F.logsigmoid(torch.rand(B, T, H, D, dtype=dtype, device=device))
        / gate_logit_normalizer
    ).requires_grad_()
    h0 = torch.rand(B, H, D, D, dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn(B, H, D, D, dtype=torch.float32, device=device)

    tri, tri_ht = chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=D**-0.5,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=None,
    )
    ((tri * do).sum() + (tri_ht * dht).sum().to(do.dtype)).backward()
    tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    tri_dg, tri_dh0 = g.grad.clone(), h0.grad.clone()
    q.grad = k.grad = v.grad = g.grad = h0.grad = None

    ref, ref_ht = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    ref_dg, ref_dh0 = g.grad.clone(), h0.grad.clone()

    _assert_close("o", ref, tri, 0.004)
    _assert_close("ht", ref_ht, tri_ht, 0.005)
    _assert_close("dq", ref_dq, tri_dq, 0.005)
    _assert_close("dk", ref_dk, tri_dk, 0.005)
    _assert_close("dv", ref_dv, tri_dv, 0.005)
    _assert_close("dg", ref_dg, tri_dg, 0.005)
    _assert_close("dh0", ref_dh0, tri_dh0, 0.005)


@pytest.mark.chunk_gla_state_v_first
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*c, id="B{}-T{}-H{}-D{}-{}".format(*c))
        for c in [
            (2, 256, 4, 64, torch.float),
            (2, 1024, 4, 128, torch.float16),
        ]
    ],
)
def test_chunk_state_v_first(B, T, H, D, dtype):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    device = torch.device("cuda")

    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand(B, T, H, D, dtype=dtype, device=device))
    h0 = torch.rand(B, H, D, D, dtype=torch.float32, device=device)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run(state_v_first):
        q_ = q.detach().clone().requires_grad_()
        k_ = k.detach().clone().requires_grad_()
        v_ = v.detach().clone().requires_grad_()
        g_ = g.detach().clone().requires_grad_()
        h0_in = h0.transpose(-1, -2).contiguous() if state_v_first else h0.clone()
        dht_in = dht.transpose(-1, -2).contiguous() if state_v_first else dht
        h0_in = h0_in.requires_grad_()
        out, ht = chunk_gla(
            q=q_,
            k=k_,
            v=v_,
            g=g_,
            scale=D**-0.5,
            initial_state=h0_in,
            output_final_state=True,
            state_v_first=state_v_first,
        )
        ((out * do).sum() + (ht * dht_in).sum()).backward()
        return out, ht, q_.grad, k_.grad, v_.grad, g_.grad, h0_in.grad

    ref_o, ref_ht, ref_dq, ref_dk, ref_dv, ref_dg, ref_dh0 = run(False)
    tri_o, tri_ht, tri_dq, tri_dk, tri_dv, tri_dg, tri_dh0 = run(True)

    _assert_close("o", ref_o, tri_o, 0.005)
    _assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 0.005)
    _assert_close("dq", ref_dq, tri_dq, 0.005)
    _assert_close("dk", ref_dk, tri_dk, 0.005)
    _assert_close("dv", ref_dv, tri_dv, 0.005)
    _assert_close("dg", ref_dg, tri_dg, 0.005)
    _assert_close("dh0", ref_dh0, tri_dh0.transpose(-1, -2), 0.005)


@pytest.mark.chunk_gla_varlen
@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens", "dtype"),
    [
        pytest.param(*c, id="H{}-D{}-cu_seqlens{}-{}".format(*c))
        for c in [
            (4, 64, [0, 15], torch.float16),
            (4, 64, [0, 256, 500, 1000], torch.float16),
            (4, 100, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
def test_chunk_varlen(H, D, cu_seqlens, dtype):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    device = torch.device("cuda")

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.rand(1, T, H, D, dtype=dtype, device=device).requires_grad_()
    k = torch.rand(1, T, H, D, dtype=dtype, device=device).requires_grad_()
    v = torch.rand(1, T, H, D, dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(
        torch.rand(1, T, H, D, dtype=dtype, device=device)
    ).requires_grad_()
    h0 = torch.rand(N, H, D, D, dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.rand(N, H, D, D, dtype=torch.float32, device=device)

    ref, ref_ht = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    ((ref * do).sum() + (ref_ht * dht).sum().to(do.dtype)).backward()
    ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    ref_dg, ref_dh0 = g.grad.clone(), h0.grad.clone()
    q.grad = k.grad = v.grad = g.grad = h0.grad = None

    tri, tri_ht = chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=D**-0.5,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    tri_dg, tri_dh0 = g.grad.clone(), h0.grad.clone()

    _assert_close("o", ref, tri, 0.004)
    _assert_close("ht", ref_ht, tri_ht, 0.005)
    _assert_close("dq", ref_dq, tri_dq, 0.005)
    _assert_close("dk", ref_dk, tri_dk, 0.005)
    _assert_close("dv", ref_dv, tri_dv, 0.005)
    _assert_close("dg", ref_dg, tri_dg, 0.005)
    _assert_close("dh0", ref_dh0, tri_dh0, 0.005)
