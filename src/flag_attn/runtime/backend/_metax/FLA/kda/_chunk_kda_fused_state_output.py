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

"""Narrow dense, initial-state-free fused consumer for MetaX KDA generic TLE."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl


__all__ = ["fused_kda_fwd_state_output"]


# The following cache/index helpers retain the notices and exact
# behavior of the existing FlagAttention FLA compatibility sources.
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries: tuple[tuple | None, dict | None, Any] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:i]
                    + cache_entries[i + 1 :]
                    + [(args, kwargs, last_result)]
                )
                return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper

@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            triton.cdiv(prepare_lens(cu_seqlens), chunk_size),
        ]
    ).cumsum(-1)


@triton.jit
def _exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
        "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _kda_tle_fused_state_output_vmajor_kernel(
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
    o += (bos * HV + i_h) * V
    beta += bos * HV + i_h
    gk += (chunk_start * HV + i_h).to(tl.int64) * K
    ws_base = ws + (bos * HV + i_h) * 3 * K

    if IS_VARLEN:
        a_chunk = (i_h * NT_TOTAL + chunk_start) * BT * BT
    else:
        a_chunk = (i_n * HV + i_h) * NT_TOTAL * BT * BT

    Aqk += a_chunk.to(tl.int64)
    Akk += a_chunk.to(tl.int64)

    state_dtype: tl.constexpr = ws.dtype.element_ty

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0 + i_nh * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, K),
            (1, 0),
        )
        b_h0_vk = tl.load(
            p_h0,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)

        # Retain the frozen canonicalization in source. Production dispatch
        # hard-gates this kernel to the initial_state=None specialization.
        h0_buf = tle.gpu.alloc(
            [K, BV],
            dtype=tl.float32,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        h0_rows = tl.broadcast_to(
            tl.arange(0, K)[:, None],
            (K, BV),
        )
        h0_cols = tl.broadcast_to(
            tl.arange(0, BV)[None, :],
            (K, BV),
        )
        h0_sp = tle.gpu.local_ptr(
            h0_buf,
            (h0_rows, h0_cols),
        )

        b_h0_kv = tl.trans(b_h0_vk)
        tl.store(h0_sp, b_h0_kv)
        tl.debug_barrier()

        b_h = tl.load(h0_sp).to(tl.float32)
        tl.debug_barrier()
        b_ht = tl.trans(b_h)
    else:
        b_ht = tl.zeros([BV, K], dtype=tl.float32)

    # One V-major BF16 pre-state tile: [BV, K].
    h_pre_buf = tle.gpu.alloc(
        [BV, K],
        dtype=ws.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    h_rows = tl.broadcast_to(
        tl.arange(0, BV)[:, None],
        (BV, K),
    )
    h_cols = tl.broadcast_to(
        tl.arange(0, K)[None, :],
        (BV, K),
    )
    h_pre_sp = tle.gpu.local_ptr(
        h_pre_buf,
        (h_rows, h_cols),
    )

    for i_t in tl.range(NT):
        p_w = tl.make_block_ptr(
            ws_base,
            (T, 3 * K),
            (HV * 3 * K, 1),
            (i_t * BT, 0),
            (BT, K),
            (1, 0),
        )
        p_qg = tl.make_block_ptr(
            ws_base,
            (T, 3 * K),
            (HV * 3 * K, 1),
            (i_t * BT, K),
            (BT, K),
            (1, 0),
        )
        p_kg = tl.make_block_ptr(
            ws_base,
            (T, 3 * K),
            (HV * 3 * K, 1),
            (i_t * BT, 2 * K),
            (BT, K),
            (1, 0),
        )
        p_v = tl.make_block_ptr(
            v,
            (T, V),
            (HV * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_beta = tl.make_block_ptr(
            beta,
            (T,),
            (HV,),
            (i_t * BT,),
            (BT,),
            (0,),
        )
        p_Aqk = tl.make_block_ptr(
            Aqk,
            (NT * BT, BT),
            (BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
        p_Akk = tl.make_block_ptr(
            Akk,
            (NT * BT, BT),
            (BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
        p_gk = tl.make_block_ptr(
            gk,
            (NT, K),
            (HV * K, 1),
            (i_t, 0),
            (1, K),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o,
            (T, V),
            (HV * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )

        # Materialize the exact V-major chunk pre-state.
        h_pre_bf = b_ht.to(state_dtype)
        tl.store(h_pre_sp, h_pre_bf)
        tl.debug_barrier()

        # (w @ h).T == h.T @ w.T
        h_for_state = tl.load(h_pre_sp)
        b_w = tl.load(
            p_w,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        b_Akk = tl.load(
            p_Akk,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        b_v_raw = tl.load(
            p_v,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)
        b_beta = tl.load(
            p_beta,
            boundary_check=(0,),
            padding_option="zero",
        ).to(tl.float32)
        b_v_raw = (b_v_raw * tl.sigmoid(b_beta)[:, None]).to(state_dtype)

        b_w_t = tl.trans(b_w)
        b_kht = tl.dot(
            h_for_state,
            b_w_t,
        ).to(tl.float32)

        b_v_raw_t = tl.trans(b_v_raw)
        b_diff_t = b_v_raw_t.to(tl.float32) - b_kht
        b_Akk_t = tl.trans(b_Akk)
        b_vt = tl.dot(
            b_diff_t.to(state_dtype),
            b_Akk_t,
        ).to(tl.float32)

        tl.debug_barrier()

        # qh and output still use the pre-update state.
        h_for_output = tl.load(h_pre_sp)
        b_qg = tl.load(
            p_qg,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        b_Aqk = tl.load(
            p_Aqk,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        b_v_for_output_t = b_vt.to(state_dtype)
        b_qg_t = tl.trans(b_qg)
        b_Aqk_t = tl.trans(b_Aqk)

        b_ot = scale * tl.dot(
            h_for_output,
            b_qg_t,
        ).to(tl.float32)
        b_ot += tl.dot(
            b_v_for_output_t,
            b_Aqk_t,
        ).to(tl.float32)
        tl.store(
            p_o,
            tl.trans(b_ot).to(p_o.dtype.element_ty),
            boundary_check=(0, 1),
        )

        tl.debug_barrier()

        b_kg = tl.load(
            p_kg,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        b_gk = tl.load(
            p_gk,
            boundary_check=(0, 1),
            padding_option="zero",
        ).reshape([K])
        b_v_for_state_t = b_vt.to(state_dtype)

        b_ht = b_ht * _exp2(b_gk)[None, :]
        b_ht += tl.dot(
            b_v_for_state_t,
            b_kg,
        ).to(tl.float32)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_nh * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, K),
            (1, 0),
        )
        tl.store(
            p_ht,
            b_ht.to(p_ht.dtype.element_ty),
            boundary_check=(0, 1),
        )


def _launch_vmajor_state_output(
    *,
    v: torch.Tensor,
    beta: torch.Tensor,
    Akk: torch.Tensor,
    gk: torch.Tensor,
    Aqk: torch.Tensor,
    ws: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if initial_state is not None:
        raise RuntimeError("V-major BV128 only supports initial_state=None")

    B, _, HV, packed_K = ws.shape
    K = packed_K // 3
    T_actual = v.shape[1]
    V = v.shape[-1]
    BT = chunk_size

    if packed_K != 3 * K:
        raise RuntimeError("invalid packed KDA workspace")
    if K != 128 or V != 128 or BT != 16:
        raise RuntimeError(
            "fused KDA state/output only supports K=V=128 and chunk_size=16"
        )
    if v.dtype is not torch.bfloat16 or ws.dtype is not torch.bfloat16:
        raise RuntimeError("fused KDA state/output currently supports BF16 only")

    if cu_seqlens is None:
        N = B
        chunk_offsets = None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(
            cu_seqlens,
            BT,
        )

    final_state = None
    if output_final_state:
        final_state = ws.new_empty(
            N,
            HV,
            V,
            K,
            dtype=torch.float32,
        )

    o = torch.empty(
        B,
        T_actual,
        HV,
        V,
        device=ws.device,
        dtype=v.dtype,
    )

    h0_arg = (
        initial_state
        if initial_state is not None
        else ws.new_empty(1, dtype=torch.float32)
    )
    ht_arg = (
        final_state if final_state is not None else ws.new_empty(1, dtype=torch.float32)
    )

    grid = (triton.cdiv(V, 128), N * HV)
    _kda_tle_fused_state_output_vmajor_kernel[grid](
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
        num_warps=8,
        num_stages=3,
    )
    return o, final_state


def fused_kda_fwd_state_output(
    v: torch.Tensor,
    beta: torch.Tensor,
    Akk: torch.Tensor,
    gk: torch.Tensor,
    Aqk: torch.Tensor,
    scale: float | None,
    ws: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the proven narrow fused consumer on V-first state tensors."""

    if ws is None:
        raise ValueError("MetaX KDA TLE state/output requires the packed workspace")
    K = ws.shape[-1] // 3
    if scale is None:
        scale = K**-0.5

    if initial_state is not None:
        raise ValueError("fused KDA state/output only supports initial_state=None")
    if not output_final_state:
        raise ValueError("fused KDA state/output requires output_final_state=True")
    if cu_seqlens is not None:
        raise ValueError("fused KDA state/output only supports cu_seqlens=None")

    return _launch_vmajor_state_output(
        v=v,
        beta=beta,
        Akk=Akk,
        gk=gk,
        Aqk=Aqk,
        ws=ws,
        scale=scale,
        initial_state=None,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
