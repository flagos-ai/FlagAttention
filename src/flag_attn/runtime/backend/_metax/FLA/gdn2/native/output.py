# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN2-local output tuning, reusing the unchanged shared GLA JIT kernel.

The copied host launch follows the shared helper. Only the GDN2 caller uses
this module's independent autotuner; shared GLA tuners are never modified.
Unobserved layouts/dtypes/head-value combinations retain the original route.
FLAG_ATTN_GDN2_FULL_TUNING=1 restores the original 108 candidates at import.
"""

import copy
import importlib
import os

import torch
import triton
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction

from ...compat import autotune_cache_kwargs
from ..tuning import get_tuned_config

_shared = importlib.import_module("flag_attn.gated_linear_attention.chunk_gla")
HAS_TLE = _shared.HAS_TLE
prepare_chunk_indices = _shared.prepare_chunk_indices
chunk_gla_fwd_kernel_o = _shared.chunk_gla_fwd_kernel_o
_PROFILED_HV_V = {'torch.bfloat16': ((2, 64),
                    (8, 64),
                    (8, 512),
                    (16, 64),
                    (16, 128),
                    (16, 512),
                    (32, 256),
                    (64, 128),
                    (96, 128)),
 'torch.float16': ((8, 64),
                   (8, 512),
                   (16, 64),
                   (16, 128),
                   (16, 512),
                   (32, 256),
                   (64, 128),
                   (96, 128))}


def _unwrap_shared_tuner(kernel):
    seen = set()
    tuner = None
    while id(kernel) not in seen:
        seen.add(id(kernel))
        if isinstance(kernel, Autotuner):
            if tuner is not None:
                raise RuntimeError("Nested shared output autotuners need review")
            tuner = kernel
        if isinstance(kernel, JITFunction):
            if tuner is None:
                raise RuntimeError("Shared output autotuner is missing")
            return tuner, kernel
        kernel = getattr(kernel, "fn", None)
        if kernel is None:
            break
    raise RuntimeError("Unrecognized shared output kernel wrapper")


if HAS_TLE:
    _shared_tuner, _shared_jit = _unwrap_shared_tuner(
        _shared.chunk_gla_fwd_kernel_o_tle
    )
    # Own Config objects and runtime state; only the JIT kernel is shared.
    _output_configs = get_tuned_config(
        "gdn2_native_output_tle",
        fallback=copy.deepcopy(_shared_tuner.configs),
    )
    chunk_gla_fwd_kernel_o_tle = triton.heuristics(
        {"IS_VARLEN": lambda args: args["cu_seqlens"] is not None}
    )(
        triton.autotune(
            configs=_output_configs,
            key=["BT", "HV", "STATE_V_FIRST", "V"],
            **autotune_cache_kwargs,
        )(_shared_jit)
    )


def _uses_profile(q, v, g, A, h, state_v_first, cu_seqlens, chunk_size):
    if not HAS_TLE or os.environ.get("FLAG_ATTN_GLA_TLE", "1") == "0":
        return False
    if chunk_size != 64 or state_v_first or cu_seqlens is not None:
        return False
    if g.dtype != torch.float32 or any(t.dtype != q.dtype for t in (v, A, h)):
        return False
    return (v.shape[2], v.shape[-1]) in _PROFILED_HV_V.get(str(q.dtype), ())


def chunk_gdn2_fwd_o(
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
):
    if not _uses_profile(q, v, g, A, h, state_v_first, cu_seqlens, chunk_size):
        return _shared.chunk_gla_fwd_o_gk(
            q=q, v=v, g=g, A=A, h=h, scale=scale,
            state_v_first=state_v_first, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size, chunk_indices=chunk_indices,
        )

    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Please ensure zeros, since vllm will use padding v
    o = torch.zeros_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * HV)

    def grid_tle(meta):
        return (triton.cdiv(V, meta["BV"] * meta.get("B_VCHUNK", 1)), NT, B * HV)

    _use_tle = HAS_TLE and os.environ.get("FLAG_ATTN_GLA_TLE", "1") != "0"
    if _use_tle:
        kernel = chunk_gla_fwd_kernel_o_tle
        grid_fn = grid_tle
    else:
        kernel = chunk_gla_fwd_kernel_o
        grid_fn = grid
    kernel[grid_fn](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return o
