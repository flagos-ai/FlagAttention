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

"""Hygon Triton implementation of the chunked GDN-2 operator.

This module currently keeps the implementation kernels and their orchestration
together to preserve the established Hygon dispatch. The sections below are
grouped by responsibility: public API adapter, H-state recurrence, gate
transforms, WY recomputation, intra-chunk solve, and the forward pipeline.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_attn.FLA.compat import libentry, libtuner
from flag_attn.runtime.backend._hygon import device
from flag_attn.runtime.backend._hygon.FLA.cumsum import chunk_local_cumsum
from flag_attn.runtime.backend._hygon.FLA.gdn2.triton_ops_helper import (
    autotune_cache_kwargs,
    exp,
    exp2,
)
from flag_attn.runtime.backend._hygon.FLA.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from flag_attn.runtime.backend._hygon.FLA.utils import use_cuda_graph


__all__ = ["chunk_gdn2"]


_GDN2_GATE_CUMSUM_INTER_ENV = "FLAG_ATTN_GDN2_FUSE_CUMSUM_INTER"
_GDN2_INTER_ROW_SPLIT_ENV = "FLAG_ATTN_GDN2_INTER_ROW_SPLIT"
_GDN2_H_W_PREFETCH_ENV = "FLAG_ATTN_GDN2_H_W_PREFETCH"
_GDN2_H_SHORT_BATCH_S1_ENV = "FLAG_ATTN_GDN2_H_SHORT_BATCH_S1"

# Exact BF16 configurations validated for moving base-2 gate cumsum into the
# inter-solve kernel. Keep this allowlist conservative until nearby shapes have
# their own correctness and end-to-end measurements.
_GDN2_GATE_CUMSUM_INTER_SHAPES = frozenset(
    {
        (1, 512, 8, 64, 64),
        (2, 512, 8, 64, 64),
        (4, 512, 8, 64, 64),
        (8, 512, 8, 64, 64),
        (2, 512, 8, 128, 128),
        (4, 512, 8, 128, 128),
    }
)

_GDN2_FUSE_HO_ENV = "FLAG_ATTN_GDN2_FUSE_HO"
_GDN2_KG_HEAD_MAJOR_ENV = "FLAG_ATTN_GDN2_KG_HEAD_MAJOR"
_GDN2_AKK_HEAD_MAJOR_ENV = "FLAG_ATTN_GDN2_AKK_HEAD_MAJOR"
_GDN2_SAFE_DIAG_REUSE_ENV = "FLAG_ATTN_GDN2_SAFE_DIAG_REUSE"


def _inter_row_split_enabled() -> bool:
    """Return whether the schedule flag is on and safe-diagonal reuse is off."""
    if os.environ.get(_GDN2_INTER_ROW_SPLIT_ENV, "1") != "1":
        return False
    if os.environ.get(_GDN2_SAFE_DIAG_REUSE_ENV, "0") != "0":
        return False
    return True


def _gdn2_safe_diag_reuse_mode(
    *, safe_gate: bool, fuse_gate_cumsum: bool
) -> int:
    """Return the compile-time safe-diagonal reuse mode for the intra solve."""
    if fuse_gate_cumsum:
        return 4
    if not safe_gate:
        return 0

    mode = os.environ.get(_GDN2_SAFE_DIAG_REUSE_ENV, "0")
    if mode in ("1", "4"):
        return 4
    if mode == "2":
        return 2
    return 0


def _gdn2_gate_cumsum_inter_shape_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    seq_len: int,
    key_dim: int,
) -> bool:
    """Return whether this exact BF16 shape is in the measured allowlist."""
    shape = (q.shape[0], seq_len, q.shape[-2], key_dim, v.shape[-1])
    return shape in _GDN2_GATE_CUMSUM_INTER_SHAPES and all(
        tensor.dtype == torch.bfloat16 for tensor in (q, k, b, v)
    )


def _should_fuse_gate_cumsum_inter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    *,
    seq_len: int,
    key_dim: int,
    chunk_size: int,
    safe_gate: bool,
    use_gate_in_kernel: bool,
    disable_recompute: bool,
    return_intermediate_states: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
) -> bool:
    """Check runtime prerequisites and shape evidence for cumsum fusion.

    The optimization changes the inter-solve dataflow: it reads raw gate,
    builds the base-2 prefix sum there, and writes the cumulative gate for
    later stages. It is currently validated only for fixed-length, chunk-64,
    safe-gate BF16 inference calls in the allowlist above. In particular,
    varlen, ``use_gate_in_kernel`` calls, recompute-disabled,
    intermediate-state, and unmeasured shapes retain the separate-cumsum
    fallback.
    """
    if os.environ.get(_GDN2_GATE_CUMSUM_INTER_ENV, "1") != "1":
        return False
    if use_gate_in_kernel or not safe_gate or disable_recompute:
        return False
    if (
        return_intermediate_states
        or cu_seqlens is not None
        or chunk_indices is not None
    ):
        return False
    if chunk_size != 64:
        return False
    return _gdn2_gate_cumsum_inter_shape_supported(
        q=q,
        k=k,
        v=v,
        b=b,
        seq_len=seq_len,
        key_dim=key_dim,
    )


def _gdn2_h_o_shape_supported(
    q: torch.Tensor,
    *,
    seq_len: int,
    key_dim: int,
    value_dim: int,
) -> bool:
    """Return whether a measured shape family supports H-to-O fusion."""
    head_count = q.shape[-2]
    batch_size = q.shape[0]

    # Short K/V=64 shapes were consistently useful in paired measurements.
    if seq_len <= 1024:
        if key_dim <= 64:
            if value_dim <= 64:
                return True

    # The T=2048, H=8 extension is batch-limited by the measured regression at B=4.
    if batch_size <= 2:
        if seq_len == 2048:
            if head_count == 8:
                if key_dim == 64:
                    if value_dim == 64:
                        return True

    if head_count >= 64:
        return True

    # Long H=32 is useful; shorter H=32 and long H=16 are excluded by A/B data.
    if head_count == 32:
        if seq_len >= 16384:
            return True

    return False


def _should_fuse_gdn2_h_o(
    q: torch.Tensor,
    *,
    seq_len: int,
    key_dim: int,
    value_dim: int,
    cu_seqlens: torch.LongTensor | None,
    return_intermediate_states: bool,
    disable_recompute: bool,
) -> bool:
    """Check H-to-O prerequisites before consulting measured shape families."""
    if os.environ.get(_GDN2_FUSE_HO_ENV, "1") != "1":
        return False
    if cu_seqlens is not None:
        return False
    if return_intermediate_states:
        return False
    if disable_recompute:
        return False
    if key_dim > 128:
        return False
    if value_dim > 128:
        return False

    return _gdn2_h_o_shape_supported(
        q,
        seq_len=seq_len,
        key_dim=key_dim,
        value_dim=value_dim,
    )


def _gdn2_h96_row_split_layout_shape_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    *,
    seq_len: int,
    key_dim: int,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None,
) -> bool:
    """Match the measured BF16 H=96 layout-only row-split exception."""
    if chunk_size != 64:
        return False
    if q.shape[0] != 1:
        return False
    if seq_len < 8192 or seq_len > 16384:
        return False
    if q.shape[-2] != 96:
        return False
    if key_dim != 128:
        return False
    if v.shape[-1] != 128:
        return False
    if q.dtype != torch.bfloat16:
        return False
    if k.dtype != q.dtype:
        return False
    if b.dtype != q.dtype:
        return False
    if v.dtype != q.dtype:
        return False
    if g.dtype != torch.float32:
        return False
    if chunk_indices is not None:
        return False
    return True


def _should_override_gdn2_h96_row_split_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    *,
    seq_len: int,
    key_dim: int,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None,
    fuse_h_o: bool,
    safe_gate: bool,
    inter_row_split_requested: bool,
) -> bool:
    """Disable head-major intermediates only for the active H96 row-split path."""
    if not fuse_h_o:
        return False
    if not safe_gate:
        return False
    if not inter_row_split_requested:
        return False
    return _gdn2_h96_row_split_layout_shape_supported(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        seq_len=seq_len,
        key_dim=key_dim,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )


def _should_store_gdn2_kg_head_major(
    q: torch.Tensor,
    *,
    key_dim: int,
    fuse_h_o: bool,
    h96_row_split_layout_override: bool,
) -> bool:
    """Select the measured high-head, K=128 KG layout when H-to-O is active."""
    if not fuse_h_o:
        return False
    if h96_row_split_layout_override:
        return False
    if os.environ.get(_GDN2_KG_HEAD_MAJOR_ENV, "0") != "1":
        return False
    if q.shape[-2] < 64:
        return False
    if key_dim < 128:
        return False
    return True


def _should_store_gdn2_akk_head_major(
    q: torch.Tensor,
    *,
    seq_len: int,
    key_dim: int,
    value_dim: int,
    fuse_h_o: bool,
    safe_gate: bool,
    h96_row_split_layout_override: bool,
) -> bool:
    """Select Akk head-major except for its measured long H=32 regression."""
    if not fuse_h_o:
        return False
    if h96_row_split_layout_override:
        return False
    if not safe_gate:
        return False
    if os.environ.get(_GDN2_AKK_HEAD_MAJOR_ENV, "1") != "1":
        return False
    if key_dim < 128:
        if q.shape[-2] < 16:
            return False

    # For long H=32/K=V=128, HO-only wins; retain contiguous Akk.
    if q.shape[-2] == 32:
        if seq_len >= 16384:
            if key_dim == 128:
                if value_dim == 128:
                    return False
    return True


def chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    return_intermediate_states: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    """Run Hygon's Triton GDN-2 implementation with the public API shape."""
    if chunk_size != 64:
        # The Hygon inter-solve kernel is built around four 16-token sub-chunks.
        # Preserve the public API for other chunk sizes with the generic path.
        from flag_attn.gdn2.chunk import chunk_gdn2 as generic_chunk_gdn2

        return generic_chunk_gdn2(
            q,
            k,
            v,
            g,
            b,
            w,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            chunk_size=chunk_size,
            return_intermediate_states=return_intermediate_states,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
        )

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(q.dtype)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(k.dtype)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    (
        o,
        final_state,
        _g,
        _Aqk,
        _Akk,
        _w_wy,
        _u_wy,
        _qg,
        _kg,
        _v_new,
        h,
        _initial_state,
    ) = chunk_gdn2_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w_gate=w,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        return_intermediate_states=return_intermediate_states,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
    )
    if return_intermediate_states:
        return o, final_state, h
    return o, final_state


# ---- H-state recurrence kernels ----

# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

NUM_WARPS = [2, 4, 8, 16]

IS_HIP_BACKEND = device.vendor_name in {"amd", "hygon"}

H_KERNEL_NUM_STAGES = [1, 2, 3] if IS_HIP_BACKEND else [2, 3, 4]
# gfx936 benefits from a wider CTA for large V tiles. Keep the existing
# 2/4-warps candidates and let autotune select 8 only when the workload
# amortizes the extra parallelism; this avoids forcing a larger register
# footprint on the short, launch-bound cases.
H_KERNEL_NUM_WARPS = [2, 4, 8] if IS_HIP_BACKEND else [2, 4]

@libentry()
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        # A narrower H-state tile raises CTA count for low-head workloads;
        # keep it as an autotune candidate so larger batches/heads can retain
        # the existing wider configuration when that is faster.
        triton.Config({"BV": 16}, num_warps=4, num_stages=2),
        *[
            triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in H_KERNEL_NUM_WARPS
            for num_stages in H_KERNEL_NUM_STAGES
            for BV in [32, 64]
        ],
    ],
    key=[
        "H",
        "K",
        "V",
        "BT",
        "IS_VARLEN",
        "USE_INITIAL_STATE",
        "STORE_FINAL_STATE",
        "GK_IS_BASE2",
        "KG_HEAD_MAJOR",
    ],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    q,
    A,
    o,
    scale,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    GK_IS_BASE2: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    FUSE_O: tl.constexpr,
    KG_HEAD_MAJOR: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    if KG_HEAD_MAJOR:
        k += ((i_n * H + i_h) * T * K).to(tl.int64)
    else:
        k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if FUSE_O:
        q += ((bos * H + i_h) * K).to(tl.int64)
        gq = gk + ((bos * H + i_h) * K).to(tl.int64)
        A += ((bos * H + i_h) * BT).to(tl.int64)
        o += ((bos * H + i_h) * V).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = K if KG_HEAD_MAJOR else Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        if not FUSE_O:
            p_h1 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(
                    h + i_t * stride_h,
                    (K, V),
                    (V, 1),
                    (64, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(
                    h + i_t * stride_h,
                    (K, V),
                    (V, 1),
                    (128, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(
                    h + i_t * stride_h,
                    (K, V),
                    (V, 1),
                    (192, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v


        if FUSE_O:
            # The state in b_h* is the state at the beginning of this chunk,
            # which is exactly the state consumed by the regular O kernel.
            # Compute q*exp2(g) @ h and Aqk @ v_new before advancing the
            # recurrence, avoiding the materialized h and v_new tensors.
            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
            )
            p_gq = tl.make_block_ptr(
                gq,
                (T, K),
                (H * K, 1),
                (i_t * BT, 0),
                (BT, 64),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.load(p_gq, boundary_check=(0, 1)).to(tl.float32)
            b_o += tl.dot((b_q * exp2(b_gq)).to(b_q.dtype), b_h1.to(b_q.dtype))
            if K > 64:
                p_q = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
                )
                p_gq = tl.make_block_ptr(
                    gq,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT, 64),
                    (BT, 64),
                    (1, 0),
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_gq = tl.load(p_gq, boundary_check=(0, 1)).to(tl.float32)
                b_o += tl.dot((b_q * exp2(b_gq)).to(b_q.dtype), b_h2.to(b_q.dtype))
            if K > 128:
                p_q = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
                )
                p_gq = tl.make_block_ptr(
                    gq,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT, 128),
                    (BT, 64),
                    (1, 0),
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_gq = tl.load(p_gq, boundary_check=(0, 1)).to(tl.float32)
                b_o += tl.dot((b_q * exp2(b_gq)).to(b_q.dtype), b_h3.to(b_q.dtype))
            if K > 192:
                p_q = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
                )
                p_gq = tl.make_block_ptr(
                    gq,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT, 192),
                    (BT, 64),
                    (1, 0),
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_gq = tl.load(p_gq, boundary_check=(0, 1)).to(tl.float32)
                b_o += tl.dot((b_q * exp2(b_gq)).to(b_q.dtype), b_h4.to(b_q.dtype))
            p_A = tl.make_block_ptr(
                A,
                (T, BT),
                (H * BT, 1),
                (i_t * BT, 0),
                (BT, BT),
                (1, 0),
            )
            m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
            b_A = tl.where(m_s, tl.load(p_A, boundary_check=(0, 1)), 0.0)
            b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v)
            p_o = tl.make_block_ptr(
                o,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            ).to(tl.float32)
            if GK_IS_BASE2:
                b_h1 *= exp2(b_gk_last1)[:, None]
            else:
                b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                ).to(tl.float32)
                if GK_IS_BASE2:
                    b_h2 *= exp2(b_gk_last2)[:, None]
                else:
                    b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                ).to(tl.float32)
                if GK_IS_BASE2:
                    b_h3 *= exp2(b_gk_last3)[:, None]
                else:
                    b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                ).to(tl.float32)
                if GK_IS_BASE2:
                    b_h4 *= exp2(b_gk_last4)[:, None]
                else:
                    b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@libentry()
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in H_KERNEL_NUM_WARPS
        for num_stages in H_KERNEL_NUM_STAGES
        for BV in [32, 64]
    ],
    key=[
        "H",
        "K",
        "V",
        "BT",
        "IS_VARLEN",
        "USE_INITIAL_STATE",
        "STORE_FINAL_STATE",
        "GK_IS_BASE2",
        "KG_HEAD_MAJOR",
    ],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_lookahead(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    q,
    A,
    o,
    scale,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    GK_IS_BASE2: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    FUSE_O: tl.constexpr,
    KG_HEAD_MAJOR: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    if KG_HEAD_MAJOR:
        k += ((i_n * H + i_h) * T * K).to(tl.int64)
    else:
        k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if FUSE_O:
        q += ((bos * H + i_h) * K).to(tl.int64)
        gq = gk + ((bos * H + i_h) * K).to(tl.int64)
        A += ((bos * H + i_h) * BT).to(tl.int64)
        o += ((bos * H + i_h) * V).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = K if KG_HEAD_MAJOR else Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # Preload chunk zero, then carry one W tile ahead across the recurrence.
    p_w = tl.make_block_ptr(
        w, (T, K), (stride_w, 1), (0, 0), (BT, 64), (1, 0)
    )
    b_w = tl.load(p_w, boundary_check=(0, 1))
    for i_t in range(NT):
        if not FUSE_O:
            p_h1 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(
                    h + i_t * stride_h,
                    (K, V),
                    (V, 1),
                    (64, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(
                    h + i_t * stride_h,
                    (K, V),
                    (V, 1),
                    (128, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(
                    h + i_t * stride_h,
                    (K, V),
                    (V, 1),
                    (192, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        # b_w was prefetched at the previous recurrence boundary.
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        # Start the next chunk's first W tile while current-chunk O/gate/H
        # work runs.
        # The last iteration loads a fully masked tile that is not consumed.
        p_w_next = tl.make_block_ptr(
            w,
            (T, K),
            (stride_w, 1),
            ((i_t + 1) * BT, 0),
            (BT, 64),
            (1, 0),
        )
        b_w_next = tl.load(p_w_next, boundary_check=(0, 1))


        if FUSE_O:
            # The state in b_h* is the state at the beginning of this chunk,
            # which is exactly the state consumed by the regular O kernel.
            # Compute q*exp2(g) @ h and Aqk @ v_new before advancing the
            # recurrence, avoiding the materialized h and v_new tensors.
            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
            )
            p_gq = tl.make_block_ptr(
                gq,
                (T, K),
                (H * K, 1),
                (i_t * BT, 0),
                (BT, 64),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.load(p_gq, boundary_check=(0, 1)).to(tl.float32)
            b_o += tl.dot((b_q * exp2(b_gq)).to(b_q.dtype), b_h1.to(b_q.dtype))
            if K > 64:
                p_q = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
                )
                p_gq = tl.make_block_ptr(
                    gq,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT, 64),
                    (BT, 64),
                    (1, 0),
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_gq = tl.load(p_gq, boundary_check=(0, 1)).to(tl.float32)
                b_o += tl.dot((b_q * exp2(b_gq)).to(b_q.dtype), b_h2.to(b_q.dtype))
            if K > 128:
                p_q = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
                )
                p_gq = tl.make_block_ptr(
                    gq,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT, 128),
                    (BT, 64),
                    (1, 0),
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_gq = tl.load(p_gq, boundary_check=(0, 1)).to(tl.float32)
                b_o += tl.dot((b_q * exp2(b_gq)).to(b_q.dtype), b_h3.to(b_q.dtype))
            if K > 192:
                p_q = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
                )
                p_gq = tl.make_block_ptr(
                    gq,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT, 192),
                    (BT, 64),
                    (1, 0),
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_gq = tl.load(p_gq, boundary_check=(0, 1)).to(tl.float32)
                b_o += tl.dot((b_q * exp2(b_gq)).to(b_q.dtype), b_h4.to(b_q.dtype))
            p_A = tl.make_block_ptr(
                A,
                (T, BT),
                (H * BT, 1),
                (i_t * BT, 0),
                (BT, BT),
                (1, 0),
            )
            m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
            b_A = tl.where(m_s, tl.load(p_A, boundary_check=(0, 1)), 0.0)
            b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v)
            p_o = tl.make_block_ptr(
                o,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            ).to(tl.float32)
            if GK_IS_BASE2:
                b_h1 *= exp2(b_gk_last1)[:, None]
            else:
                b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                ).to(tl.float32)
                if GK_IS_BASE2:
                    b_h2 *= exp2(b_gk_last2)[:, None]
                else:
                    b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                ).to(tl.float32)
                if GK_IS_BASE2:
                    b_h3 *= exp2(b_gk_last3)[:, None]
                else:
                    b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                ).to(tl.float32)
                if GK_IS_BASE2:
                    b_h4 *= exp2(b_gk_last4)[:, None]
                else:
                    b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)
        b_w = b_w_next

    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    gk_base2: bool = False,
    scale: float = 1.0,
    q: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    o: torch.Tensor | None = None,
    fuse_o: bool = False,
    kg_head_major: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This kernel differs slightly from FLA to support different Q/K head counts.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    # N: the number of sequences in the batch, with equal or variable lengths.
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if fuse_o:
        if q is None or A is None or o is None or gk is None:
            raise ValueError("fuse_o requires q, A, o and gk tensors")
    # The fused O kernel does not materialize the per-chunk state history.
    # A one-element placeholder keeps the pointer ABI valid; the FUSE_O
    # specialization removes all h stores.
    if fuse_o:
        h = k.new_empty(1)
    else:
        h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value and not fuse_o else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    # Keep the default JIT entrypoint and its autotune key untouched. The
    # separate W-lookahead kernel is limited to the one paired-validated
    # fixed-length shape; this environment override exists only for reproducible
    # A/B checks (auto/on/off) and does not alter the public operator API.
    prefetch_mode = os.environ.get(_GDN2_H_W_PREFETCH_ENV, "auto").lower()
    if prefetch_mode not in {"auto", "on", "off"}:
        raise ValueError(
            f"{_GDN2_H_W_PREFETCH_ENV} must be one of: auto, on, off"
        )
    prefetch_eligible = (
        cu_seqlens is None
        and B == 2
        and T == 16384
        and H == 16
        and K == 128
        and V == 128
        and not fuse_o
    )
    if prefetch_mode == "on" and not prefetch_eligible:
        raise ValueError(
            "H-kernel W lookahead is only validated for fixed-length "
            "B=2,T=16384,H=16,K=V=128, non-fused recurrences"
        )
    h_kernel = chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    if prefetch_eligible and prefetch_mode != "off":
        h_kernel = chunk_gated_delta_rule_fwd_kernel_h_blockdim64_lookahead

    # On short, high-batch H=8/K=V=64 workloads, BV=16/W4/S1 reduces the
    # H-kernel resource footprint and wins over the default autotuned BV=16/W4/S2.
    # Keep this narrow dispatch opt-out-able: the H autotune key omits B/T, so
    # reusing its cached choice across low- and high-batch shapes hides this case.
    short_high_batch_h_s1 = (
        os.environ.get(_GDN2_H_SHORT_BATCH_S1_ENV, "1") == "1"
        and cu_seqlens is None
        and fuse_o
        and B in (8, 16)
        and T in (512, 1024)
        and H == 8
        and K == 64
        and V == 64
        and k.dtype in (torch.float16, torch.bfloat16)
    )

    h_launch_args = dict(
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        GK_IS_BASE2=gk_base2,
        h=h,
        h0=initial_state,
        ht=final_state,
        q=q,
        A=A,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        FUSE_O=fuse_o,
        KG_HEAD_MAJOR=kg_head_major,
    )
    if short_high_batch_h_s1:
        # Bypass the shared autotuner cache only for this measured class.
        # Supply the compile-time heuristic values normally added by the
        # Heuristics wrapper and launch the same JIT body with the validated
        # low-stage config; all other calls retain the existing autotuner.
        h_kernel_direct = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn.fn
        h_kernel_direct[grid](
            **h_launch_args,
            BV=16,
            USE_G=g is not None,
            USE_GK=gk is not None,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            SAVE_NEW_VALUE=v_new is not None,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
            num_stages=1,
        )
    else:
        h_kernel[grid](**h_launch_args)
    return h, v_new, final_state


# ---- Gate transforms and chunk-local cumsum ----

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# This file is modified and supported by the Moonshot AI Team

def input_guard(fn):
    return fn


def autocast_custom_fwd(fn):
    return fn


def autocast_custom_bwd(fn):
    return fn


@triton.jit
def softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + exp(x)))


BS_LIST = [32, 64]
BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [4, 8, 16, 32]


def naive_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Torch reference implementation for KDA gate computation.

    Computes: g = -A_log.exp().unsqueeze(-1) * softplus(g + dt_bias.view(g.shape[-2:]))

    Args:
        g (torch.Tensor):
            Input tensor of shape `[..., H, K]`.
        A_log (torch.Tensor):
            Parameter tensor with `H` elements.
        dt_bias (torch.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[H * K]`.

    Returns:
        Output tensor of shape `[..., H, K]` .
    """
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)

    g = (-A_log.view(H, 1).float().exp() * F.softplus(g.float())).to(output_dtype)
    return g


def naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = lower_bound * F.sigmoid(A_log.view(H, 1).exp() * g)
    return g.to(output_dtype)


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_BETA": lambda args: args["beta"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in BT_LIST_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    beta,
    yg,
    yb,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(
        g + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    p_yg = tl.make_block_ptr(
        yg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    # [BT, BD]
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias, (H * D,), (1,), (i_h * D,), (BD,), (0,))
        b_g = b_g + tl.load(p_b, boundary_check=(0,)).to(tl.float32)
    if not USE_LOWER_BOUND:
        b_yg = -exp(b_A) * softplus(b_g)
    else:
        b_yg = lower_bound * tl.sigmoid(exp(b_A) * b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0, 1))

    if HAS_BETA:
        p_b = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_yb = tl.make_block_ptr(yb + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_yb = tl.sigmoid(tl.load(p_b, boundary_check=(0,)).to(tl.float32))
        tl.store(p_yb, b_yb.to(p_yb.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_BETA": lambda args: args["beta"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    beta,
    dyg,
    dyb,
    dg,
    dA,
    dbeta,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(
        g + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    p_dg = tl.make_block_ptr(
        dg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    p_dyg = tl.make_block_ptr(
        dyg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )

    # [BT, BD]
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_dyg = tl.load(p_dyg, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias, (H * D,), (1,), (i_h * D,), (BD,), (0,))
        b_g = b_g + tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    # [BT, BD]
    if not USE_LOWER_BOUND:
        b_A = -exp(b_A)
        b_yg = b_A * softplus(b_g)
        b_dg = b_A * (b_dyg * tl.sigmoid(b_g))
        b_dA = tl.sum(tl.sum(b_dyg * b_yg, 1), 0)
    else:
        b_A = exp(b_A)
        b_inner = b_A * b_g
        b_sig = tl.sigmoid(b_inner)
        b_dsig = b_sig * (1.0 - b_sig)
        # Common term: dy * (LB * dsig)
        b_d_inner_term = b_dyg * (lower_bound * b_dsig)
        # dg = d_inner_term * A
        b_dg = b_d_inner_term * b_A
        b_dA = tl.sum(tl.sum(b_dg * b_g, 1), 0)

    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dA + i_t * H + i_h, b_dA)

    if HAS_BETA:
        p_b = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_db = tl.make_block_ptr(dbeta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_dyb = tl.make_block_ptr(dyb + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

        b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_db = tl.load(p_dyb, boundary_check=(0,)).to(tl.float32) * b_b * (1.0 - b_b)
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


def kda_gate_fwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)

    yg = torch.empty_like(g, dtype=output_dtype)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), H)

    kda_gate_fwd_kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        beta=None,
        yg=yg,
        yb=None,
        T=T,
        H=H,
        D=K,
        BD=triton.next_power_of_2(K),
        lower_bound=lower_bound,
    )
    return yg


def kda_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    dyg: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)
    BT = 32
    NT = triton.cdiv(T, BT)

    dg = torch.empty_like(g, dtype=torch.float32)
    dA = A_log.new_empty(NT, H, dtype=torch.float32)

    grid = (triton.cdiv(T, BT), H)
    kda_gate_bwd_kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        beta=None,
        dyg=dyg,
        dyb=None,
        dg=dg,
        dA=dA,
        dbeta=None,
        T=T,
        H=H,
        D=K,
        BT=BT,
        BD=triton.next_power_of_2(K),
        lower_bound=lower_bound,
    )

    dg = dg.view_as(g).type_as(g)
    dA = dA.sum(0).view_as(A_log).type_as(A_log)
    dbias = dg.view(-1, H * K).sum(0).to(dt_bias) if dt_bias is not None else None

    return dg, dA, dbias


class KDAGateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = None,
        lower_bound: float | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        yg = kda_gate_fwd(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            output_dtype=output_dtype,
        )
        ctx.save_for_backward(g, A_log, dt_bias)
        ctx.lower_bound = lower_bound
        return yg

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dyg: torch.Tensor):
        g, A_log, dt_bias = ctx.saved_tensors
        dg, dA, dbias = kda_gate_bwd(
            g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg, lower_bound=ctx.lower_bound
        )
        return dg, dA, dbias, None, None


@torch.compiler.disable
def fused_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Fused KDA gate computation with autograd support.

    Computes: g = -A_log.exp().unsqueeze(-1) * softplus(g + dt_bias.view(g.shape[-2:]))

    Args:
        g (torch.Tensor):
            Input tensor of shape `[..., H, K]`.
        A_log (torch.Tensor):
            Parameter tensor with `H` elements.
        dt_bias (torch.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[H * K]`.

    Returns:
        Output tensor of shape `[..., H, K]`.
    """
    return KDAGateFunction.apply(g, A_log, dt_bias, lower_bound, output_dtype)


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["H", "S", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

    # Apply dt_bias if exists
    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias + i_h * S, (S,), (1,), (i_s * BS,), (BS,), (0,))
        b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        # Apply gate: -exp(A_log) * softplus(g + bias)
        b_gate = -exp(b_A) * softplus(b_s)
    else:
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    # Apply chunk local cumsum
    if REVERSE:
        b_o = tl.cumsum(b_gate, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    lower_bound: float | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert (
            g.shape[0] == 1
        ), "Only batch size 1 is supported when cu_seqlens are provided"
    assert len(g.shape) == 4
    B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    kda_gate_chunk_cumsum_vector_kernel[grid](
        s=g_org,
        A_log=A_log,
        dt_bias=dt_bias,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        T=T,
        H=H,
        S=S,
        BT=BT,
        REVERSE=False,
    )
    return g


# ---- WY auxiliary recomputation ----

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# WY-representation recompute for GDN-2.
#
# Given the solved chunk inverse ``A = (I + tril(T,-1))^{-1}``, this kernel
# produces the two within-chunk auxiliaries consumed by the inter-chunk state
# recurrence:
#
#   u = A @ (w * v)            # write-side, V-axis
#   w = A @ (b * exp(gk) * k)  # erase-side, K-axis
#
# It optionally emits ``qg = q * exp(gk)`` and the tail-decayed key
# ``kg = k * exp(gn - gk)`` used by the inter-chunk update.
#
# Difference vs the KDA WY recompute: KDA uses a single scalar ``beta`` for both
# the u and w blocks. GDN-2 uses the channel-wise ``w_gate`` on the V axis for u
# and the channel-wise ``b`` on the K axis for w. These two gates live on
# different axes and cannot be reduced to a shared scalar; the kernel loads
# them per BV / BK block respectively.

@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_gdn2_kernel(
    q,
    k,
    qg,
    kg,
    v,
    b,  # channel-wise erase gate, [B, T, H, K]
    w_gate,  # channel-wise write gate, [B, T, H, V]
    w,  # output: A @ (b * exp(gk) * k), [B, T, H, K]
    u,  # output: A @ (w_gate * v), [B, T, H, V]
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    KG_HEAD_MAJOR: tl.constexpr,
    A_HEAD_MAJOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if A_HEAD_MAJOR:
        a_base = (i_b * H + i_h) * T * BT
        a_stride = BT
    else:
        a_base = (bos * H + i_h) * BT
        a_stride = H * BT
    p_A = tl.make_block_ptr(
        A + a_base,
        (T, BT),
        (a_stride, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
                v + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
        p_u = tl.make_block_ptr(
                u + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
        p_wg = tl.make_block_ptr(
                w_gate + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_wg = tl.load(p_wg, boundary_check=(0, 1))
        b_vb = (b_v * b_wg).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_b = tl.make_block_ptr(
            b + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_kb = b_k * b_b

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb *= exp2(b_gk)

        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k,
                mask=m_k,
                other=0.0,
            ).to(tl.float32)
            b_kg = b_k * tl.where(
                (i_t * BT + tl.arange(0, BT) < T)[:, None],
                exp2(b_gn[None, :] - b_gk),
                0,
            )
            if KG_HEAD_MAJOR:
                p_kg = tl.make_block_ptr(
                    kg + (i_b * H + i_h) * T * K,
                    (T, K),
                    (K, 1),
                    (i_t * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
            else:
                p_kg = tl.make_block_ptr(
                    kg + (bos * H + i_h) * K,
                    (T, K),
                    (H * K, 1),
                    (i_t * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd_gdn2(
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    kg_head_major: bool = False,
    a_head_major: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Produce GDN-2 WY auxiliaries from the solved chunk inverse ``A``.

    Returns ``(w, u, qg, kg)`` with shapes matching the inputs. ``qg`` is
    produced only when ``q`` is provided; ``kg`` only when ``gk`` is provided.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    # For the large-V class, halve the number of V tiles.  Keep BV=64 for
    # smaller workloads because the wider tile increases register pressure
    # without providing enough reuse to amortize it.
    BV = 128 if V >= 512 else 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if q is not None else None
    if gk is not None and kg_head_major:
        kg_storage = torch.empty((B, H, T, K), device=k.device, dtype=k.dtype)
        kg = kg_storage.as_strided(
            (B, T, H, K),
            (H * T * K, K, T * K, 1),
        )
    else:
        kg = torch.empty_like(k) if gk is not None else None
    recompute_w_u_fwd_gdn2_kernel[(NT, B * H)](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=v,
        b=b,
        w_gate=w_gate,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        KG_HEAD_MAJOR=kg_head_major,
        A_HEAD_MAJOR=a_head_major,
    )
    return w, u, qg, kg


# ---- Intra-chunk token-parallel score construction ----

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Token-parallel intra-chunk kernel for GDN-2.
#
# Builds the diagonal blocks of the within-chunk score matrices used by the WY
# representation:
#   Aqk - causal query-key scores, decay-weighted, for the output path.
#   Akk - gated key-key scores (the strictly-lower matrix T whose inverse
#         (I + T)^{-1} defines the WY representation).
#
# Difference vs KDA: the channel-wise erase gate ``b`` is folded into the key
# tile before the dot product, instead of multiplying the row by a scalar
# ``beta`` after the dot. The rest is identical.

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BH": BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "N"])
def chunk_gdn2_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    g_out,
    b,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    q += bos * H * K
    k += bos * H * K
    g += bos * H * K
    b += bos * H * K
    Aqk += bos * H * BT
    Akk += bos * H * BC

    BK: tl.constexpr = triton.next_power_of_2(K)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K

    p_q = tl.make_block_ptr(
        q + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
    )
    p_g = tl.make_block_ptr(
        g + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
    )
    p_b = tl.make_block_ptr(
        b + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_b = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)

    # Fold channel-wise erase gate into the key tile.
    b_k = b_k * b_b

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kj = tl.make_block_ptr(
            k + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
        )
        p_gj = tl.make_block_ptr(
            g + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
        )
        b_kj = tl.load(p_kj, boundary_check=(0, 1)).to(tl.float32)
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        b_kgj = b_kj * exp2(b_g - b_gj)
        b_kgj = tl.where(m_k[None, :], b_kgj, 0.0)
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        tl.store(
            Aqk + i_t * H * BT + (i_hg * BH + o_h) * BT + j % BT,
            b_Aqk.to(Aqk.dtype.element_ty),
            mask=m_h,
        )
        tl.store(
            Akk + i_t * H * BC + (i_hg * BH + o_h) * BC + j - i_ts,
            b_Akk.to(Akk.dtype.element_ty),
            mask=m_h,
        )


def chunk_gdn2_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    b: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
):
    """Token-parallel diagonal kernel: builds Aqk plus diagonal Akk in fp32."""
    B, T, H, K = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    def grid(meta):
        return (B * T, triton.cdiv(H, meta["BH"]))

    chunk_gdn2_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        b=b,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
    return Aqk, Akk


# ---- Intra-chunk sub-chunk and triangular-solve kernels ----

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Intra-chunk Triton kernels for GDN-2.
#
# Three kernels live here:
#   * chunk_gdn2_fwd_kernel_intra_sub_chunk   - sub-chunk diagonal builder used
#     by the safe-gate path (alternative to the token-parallel diagonal kernel).
#   * chunk_gdn2_fwd_kernel_inter_solve_fused - off-diagonal Akk blocks + the
#     blocked WY triangular solve, fused. Reads diagonal blocks from Akkd, emits
#     the full lower-triangular inverse to Akk.
#   * chunk_gdn2_bwd_kernel_intra             - backward over the within-chunk
#     score matrices, producing dq, dk, db, dg contributions.
#
# Compared with KDA, the GDN-2 variant takes a channel-wise erase gate ``b`` on
# the K axis instead of a per-head scalar ``beta``; the b tensor is folded into
# the relevant tiles directly rather than applied as a row-scalar multiply.

IS_TF32_SUPPORTED = (
    device.vendor_name == "nvidia"
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 8
)

IS_GATHER_SUPPORTED = hasattr(tl, "gather")

if IS_GATHER_SUPPORTED:
    gather = tl.gather
else:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        # 仅用于让 Triton 解析 inactive branch；
        # 实际运行时 USE_GATHER=False，不会执行这里。
        return None


if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


@triton.jit
def _build_safe_diag_in_registers(
    q,
    k,
    g,
    b,
    Aqk,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    i_ti,
    i_col: tl.constexpr,
):
    """Build one 16-token diagonal tile and keep its inverse in FP32 registers.

    This is the producer half of the safe-gate fused path.  It deliberately
    matches ``chunk_gdn2_fwd_kernel_intra_sub_chunk``: dot operands retain the
    input dtype, while the triangular inverse stays FP32 until the enclosing
    inter-subchunk solve consumes it.  Consequently no FP32 ``Akkd`` tile is
    materialized in global memory and no early BF16 truncation is introduced.
    """
    o_i = tl.arange(0, BC)
    m_c = (i_ti + o_i) < T
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0)
        )
        p_b = tl.make_block_ptr(
            b, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        i_mid = i_ti + min(BC // 2, T - i_ti - 1)
        b_gn = tl.load(g + i_mid * H * K + o_k, mask=m_k, other=0.0)[None, :]
        b_gm = (b_g - b_gn).to(tl.float32)
        b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)
        b_kgt = tl.trans(b_k * b_gk)
        b_bk = (b_b.to(tl.float32) * b_k.to(tl.float32)).to(b_k.dtype)
        b_Aqk += tl.dot(b_q * b_gq, b_kgt)
        b_Akk += tl.dot(b_bk * b_gq, b_kgt)

    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)
    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_ti, i_col), (BC, BC), (1, 0)
    )
    tl.store(p_Aqk, (b_Aqk * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    b_Ai = -b_Akk
    for i in range(2, BC):
        # Select row i without spilling the FP32 tile to global memory.
        b_a = -tl.sum(tl.where((o_i == i)[:, None], b_Akk, 0.0), axis=0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    return b_Ai + m_I


@triton.jit
def _accumulate_safe_diag_from_loaded_tiles(
    b_q,
    b_k,
    b_g,
    b_b,
    b_gn,
    b_Aqk,
    b_Akk,
    m_c,
):
    """Accumulate one safe diagonal tile from tiles already loaded by inter-solve."""
    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)
    b_kgt = tl.trans(b_k * b_gk)
    b_bk = (b_b.to(tl.float32) * b_k.to(tl.float32)).to(b_k.dtype)
    b_Aqk += tl.dot(b_q * b_gq, b_kgt)
    b_Akk += tl.dot(b_bk * b_gq, b_kgt)
    return b_Aqk, b_Akk


@triton.jit
def _finish_safe_diag_from_registers(
    b_Aqk,
    b_Akk,
    Aqk,
    scale,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    i_ti,
    i_col: tl.constexpr,
):
    """Store a safe diagonal Aqk tile and return its FP32 triangular inverse."""
    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)
    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_ti, i_col), (BC, BC), (1, 0)
    )
    tl.store(p_Aqk, (b_Aqk * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    b_Ai = -b_Akk
    for i in range(2, BC):
        b_a = -tl.sum(tl.where((o_i == i)[:, None], b_Akk, 0.0), axis=0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    return b_Ai + m_I


@triton.jit
def _gather_cumsum_rows(
    b_g_cum,
    START: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.arange(0, BC) + START
    indices = tl.broadcast_to(rows[:, None], (BC, BK))
    return tl.gather(b_g_cum, indices, axis=0)


@triton.jit
def _gather_cumsum_row(b_g_cum, ROW: tl.constexpr, BK: tl.constexpr):
    indices = tl.full((1, BK), ROW, tl.int32)
    return tl.reshape(tl.gather(b_g_cum, indices, axis=0), (BK,))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gdn2_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    b,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * H + i_h) * K
    b = b + (bos * H + i_h) * K
    Aqk = Aqk + (bos * H + i_h) * BT
    Akk = Akk + (bos * H + i_h) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))

    if USE_GATHER:
        b_gn = gather(
            b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0
        )
    else:
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

    b_kgt = tl.trans(b_k * b_gk)

    b_bk = (b_b.to(tl.float32) * b_k.to(tl.float32)).to(b_k.dtype)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_bk * b_gq, b_kgt)

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
    )
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * H * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gdn2_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    g_out,
    b,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
    FUSE_SAFE_DIAG: tl.constexpr,
    SAFE_DIAG_REUSE: tl.constexpr,
    FUSE_GATE_CUMSUM: tl.constexpr,
    GATE_SCALE: tl.constexpr,
    AKK_HEAD_MAJOR: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    g_out += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    if AKK_HEAD_MAJOR:
        AKK_STRIDE = BT
        akk_base = (i_b * H + i_h) * T * BT
    else:
        AKK_STRIDE = H * BT
        akk_base = (bos * H + i_h) * BT
    Akk += akk_base
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    if FUSE_GATE_CUMSUM:
        o_g = tl.arange(0, BT)
        m_g_cum = (o_g[:, None] >= o_g[None, :]).to(tl.float32)

    if FUSE_SAFE_DIAG:
        m_I = (o_i[:, None] == o_i[None, :]).to(tl.float32)
        if SAFE_DIAG_REUSE >= 2:
            b_Aqk_diag0 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk_diag0 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Aqk_diag1 = tl.zeros([BC, BC], dtype=tl.float32)
            b_Akk_diag1 = tl.zeros([BC, BC], dtype=tl.float32)
            if SAFE_DIAG_REUSE >= 4:
                b_Aqk_diag2 = tl.zeros([BC, BC], dtype=tl.float32)
                b_Akk_diag2 = tl.zeros([BC, BC], dtype=tl.float32)
                b_Aqk_diag3 = tl.zeros([BC, BC], dtype=tl.float32)
                b_Akk_diag3 = tl.zeros([BC, BC], dtype=tl.float32)
            else:
                b_Ai22 = m_I
                b_Ai33 = m_I
                if i_tc2 < T:
                    b_Ai22 = _build_safe_diag_in_registers(
                        q, k, g, b, Aqk, scale, T, H, K, BT, BC, BK, i_tc2, 2 * BC
                    )
                if i_tc3 < T:
                    b_Ai33 = _build_safe_diag_in_registers(
                        q, k, g, b, Aqk, scale, T, H, K, BT, BC, BK, i_tc3, 3 * BC
                    )
        else:
            b_Ai00 = _build_safe_diag_in_registers(
                q, k, g, b, Aqk, scale, T, H, K, BT, BC, BK, i_tc0, 0
            )
            b_Ai11 = m_I
            b_Ai22 = m_I
            b_Ai33 = m_I
            if i_tc1 < T:
                b_Ai11 = _build_safe_diag_in_registers(
                    q, k, g, b, Aqk, scale, T, H, K, BT, BC, BK, i_tc1, BC
                )
            if i_tc2 < T:
                b_Ai22 = _build_safe_diag_in_registers(
                    q, k, g, b, Aqk, scale, T, H, K, BT, BC, BK, i_tc2, 2 * BC
                )
            if i_tc3 < T:
                b_Ai33 = _build_safe_diag_in_registers(
                    q, k, g, b, Aqk, scale, T, H, K, BT, BC, BK, i_tc3, 3 * BC
                )

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0_raw = tl.load(p_k0, boundary_check=(0, 1))
        if FUSE_GATE_CUMSUM:
            p_g_all = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BT, BK), (1, 0)
            )
            b_g_raw_all = tl.load(p_g_all, boundary_check=(0, 1)).to(tl.float32)
            b_g_cum = tl.dot(
                m_g_cum, b_g_raw_all * GATE_SCALE, allow_tf32=False
            )
            p_g_out = tl.make_block_ptr(
                g_out, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BT, BK), (1, 0)
            )
            tl.store(p_g_out, b_g_cum, boundary_check=(0, 1))
            b_g0_raw = _gather_cumsum_rows(b_g_cum, 0, BC, BK)
            b_g1_raw = _gather_cumsum_rows(b_g_cum, BC, BC, BK)
            b_g2_raw = _gather_cumsum_rows(b_g_cum, 2 * BC, BC, BK)
            b_g3_raw = _gather_cumsum_rows(b_g_cum, 3 * BC, BC, BK)
        else:
            p_g0 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            b_g0_raw = tl.load(p_g0, boundary_check=(0, 1))
        b_k0 = b_k0_raw.to(tl.float32)
        b_g0 = b_g0_raw.to(tl.float32)

        if SAFE_DIAG_REUSE >= 2:
            p_q0 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            p_b0 = tl.make_block_ptr(
                b, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            b_q0_raw = tl.load(p_q0, boundary_check=(0, 1))
            b_b0_raw = tl.load(p_b0, boundary_check=(0, 1))
            if FUSE_GATE_CUMSUM:
                b_gn_diag0 = _gather_cumsum_row(b_g_cum, BC // 2, BK)[None, :]
            else:
                i_mid0 = i_tc0 + min(BC // 2, T - i_tc0 - 1)
                b_gn_diag0 = tl.load(
                    g + i_mid0 * H * K + o_k, mask=m_k, other=0.0
                )[None, :]
            b_Aqk_diag0, b_Akk_diag0 = _accumulate_safe_diag_from_loaded_tiles(
                b_q0_raw, b_k0_raw, b_g0_raw, b_b0_raw,
                b_gn_diag0, b_Aqk_diag0, b_Akk_diag0, m_tc0
            )

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_g1 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_b1 = tl.make_block_ptr(
                b, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            b_q1_raw = tl.load(p_q1, boundary_check=(0, 1))
            b_k1_raw = tl.load(p_k1, boundary_check=(0, 1))
            if not FUSE_GATE_CUMSUM:
                b_g1_raw = tl.load(p_g1, boundary_check=(0, 1))
            b_b1_raw = tl.load(p_b1, boundary_check=(0, 1))
            b_q1 = b_q1_raw.to(tl.float32)
            b_k1 = b_k1_raw.to(tl.float32)
            b_g1 = b_g1_raw.to(tl.float32)
            b_b1 = b_b1_raw.to(tl.float32)
            if FUSE_GATE_CUMSUM:
                b_gn1 = _gather_cumsum_row(b_g_cum, BC, BK)
            else:
                b_gn1 = tl.load(
                    g + i_tc1 * H * K + o_k,
                    mask=m_k,
                    other=0,
                ).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_bk1 = b_b1 * b_k1
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_bk1 * b_gqn, b_kgt)

            if SAFE_DIAG_REUSE >= 2:
                if FUSE_GATE_CUMSUM:
                    b_gn_diag1 = _gather_cumsum_row(b_g_cum, BC + BC // 2, BK)[None, :]
                else:
                    i_mid1 = i_tc1 + min(BC // 2, T - i_tc1 - 1)
                    b_gn_diag1 = tl.load(
                        g + i_mid1 * H * K + o_k, mask=m_k, other=0.0
                    )[None, :]
                b_Aqk_diag1, b_Akk_diag1 = _accumulate_safe_diag_from_loaded_tiles(
                    b_q1_raw, b_k1_raw, b_g1_raw, b_b1_raw,
                    b_gn_diag1, b_Aqk_diag1, b_Akk_diag1, m_tc1
                )

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_g2 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_b2 = tl.make_block_ptr(
                    b, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                b_q2_raw = tl.load(p_q2, boundary_check=(0, 1))
                b_k2_raw = tl.load(p_k2, boundary_check=(0, 1))
                if not FUSE_GATE_CUMSUM:
                    b_g2_raw = tl.load(p_g2, boundary_check=(0, 1))
                b_b2_raw = tl.load(p_b2, boundary_check=(0, 1))
                b_q2 = b_q2_raw.to(tl.float32)
                b_k2 = b_k2_raw.to(tl.float32)
                b_g2 = b_g2_raw.to(tl.float32)
                b_b2 = b_b2_raw.to(tl.float32)
                if FUSE_GATE_CUMSUM:
                    b_gn2 = _gather_cumsum_row(b_g_cum, 2 * BC, BK)
                else:
                    b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(
                        tl.float32
                    )
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_bkg2 = (b_b2 * b_k2) * b_gqn2
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_bkg2, b_kgt)

                if SAFE_DIAG_REUSE >= 4:
                    if FUSE_GATE_CUMSUM:
                        b_gn_diag2 = _gather_cumsum_row(
                            b_g_cum,
                            2 * BC + BC // 2,
                            BK,
                        )[None, :]
                    else:
                        i_mid2 = i_tc2 + min(BC // 2, T - i_tc2 - 1)
                        b_gn_diag2 = tl.load(
                            g + i_mid2 * H * K + o_k, mask=m_k, other=0.0
                        )[None, :]
                    b_Aqk_diag2, b_Akk_diag2 = _accumulate_safe_diag_from_loaded_tiles(
                        b_q2_raw, b_k2_raw, b_g2_raw, b_b2_raw,
                        b_gn_diag2, b_Aqk_diag2, b_Akk_diag2, m_tc2
                    )
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_bkg2, b_kgt)

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(
                        q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_g3 = tl.make_block_ptr(
                        g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_b3 = tl.make_block_ptr(
                        b, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    b_q3_raw = tl.load(p_q3, boundary_check=(0, 1))
                    b_k3_raw = tl.load(p_k3, boundary_check=(0, 1))
                    if not FUSE_GATE_CUMSUM:
                        b_g3_raw = tl.load(p_g3, boundary_check=(0, 1))
                    b_b3_raw = tl.load(p_b3, boundary_check=(0, 1))
                    b_q3 = b_q3_raw.to(tl.float32)
                    b_k3 = b_k3_raw.to(tl.float32)
                    b_g3 = b_g3_raw.to(tl.float32)
                    b_b3 = b_b3_raw.to(tl.float32)
                    if FUSE_GATE_CUMSUM:
                        b_gn3 = _gather_cumsum_row(b_g_cum, 3 * BC, BK)
                    else:
                        b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(
                            tl.float32
                        )
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_bkg3 = (b_b3 * b_k3) * b_gqn3
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_bkg3, b_kgt)
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_bkg3, b_kgt)
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_bkg3, b_kgt)

                    if SAFE_DIAG_REUSE >= 4:
                        if FUSE_GATE_CUMSUM:
                            b_gn_diag3 = _gather_cumsum_row(
                                b_g_cum,
                                3 * BC + BC // 2,
                                BK,
                            )[None, :]
                        else:
                            i_mid3 = i_tc3 + min(BC // 2, T - i_tc3 - 1)
                            b_gn_diag3 = tl.load(
                                g + i_mid3 * H * K + o_k, mask=m_k, other=0.0
                            )[None, :]
                        b_Aqk_diag3, b_Akk_diag3 = (
                            _accumulate_safe_diag_from_loaded_tiles(
                                b_q3_raw,
                                b_k3_raw,
                                b_g3_raw,
                                b_b3_raw,
                                b_gn_diag3,
                                b_Aqk_diag3,
                                b_Akk_diag3,
                                m_tc3,
                            )
                        )

    if FUSE_SAFE_DIAG and SAFE_DIAG_REUSE >= 2:
        b_Ai00 = _finish_safe_diag_from_registers(
            b_Aqk_diag0, b_Akk_diag0, Aqk, scale, T, H, BT, BC, i_tc0, 0
        )
        b_Ai11 = m_I
        if i_tc1 < T:
            b_Ai11 = _finish_safe_diag_from_registers(
                b_Aqk_diag1, b_Akk_diag1, Aqk, scale, T, H, BT, BC, i_tc1, BC
            )
        if SAFE_DIAG_REUSE >= 4:
            b_Ai22 = m_I
            b_Ai33 = m_I
            if i_tc2 < T:
                b_Ai22 = _finish_safe_diag_from_registers(
                    b_Aqk_diag2, b_Akk_diag2, Aqk, scale, T, H, BT, BC, i_tc2, 2 * BC
                )
            if i_tc3 < T:
                b_Ai33 = _finish_safe_diag_from_registers(
                    b_Aqk_diag3, b_Akk_diag3, Aqk, scale, T, H, BT, BC, i_tc3, 3 * BC
                )

    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Aqk21 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        p_Aqk31 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
        )
        p_Aqk32 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

    if not FUSE_SAFE_DIAG:
        p_Akk00 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akk11 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akk22 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akk33 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
        b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

        if not USE_SAFE_GATE:
            m_A = o_i[:, None] > o_i[None, :]
            m_I = o_i[:, None] == o_i[None, :]

            b_Ai00 = -tl.where(m_A, b_Ai00, 0)
            b_Ai11 = -tl.where(m_A, b_Ai11, 0)
            b_Ai22 = -tl.where(m_A, b_Ai22, 0)
            b_Ai33 = -tl.where(m_A, b_Ai33, 0)

            for i in range(2, min(BC, T - i_tc0)):
                b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
                b_a00 = tl.where(o_i < i, b_a00, 0.0)
                b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
                b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
            for i in range(BC + 2, min(2 * BC, T - i_tc0)):
                b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
                b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
                b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
                b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
            for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
                b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
                b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
                b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
                b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
            for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
                b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
                b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
                b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
                b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

            b_Ai00 += m_I
            b_Ai11 += m_I
            b_Ai22 += m_I
            b_Ai33 += m_I

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    p_Akk00 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc0, 0), (BC, BC), (1, 0)
    )
    p_Akk10 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc1, 0), (BC, BC), (1, 0)
    )
    p_Akk11 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc1, BC), (BC, BC), (1, 0)
    )
    p_Akk20 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc2, 0), (BC, BC), (1, 0)
    )
    p_Akk21 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc2, BC), (BC, BC), (1, 0)
    )
    p_Akk22 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk30 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc3, 0), (BC, BC), (1, 0)
    )
    p_Akk31 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc3, BC), (BC, BC), (1, 0)
    )
    p_Akk32 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk33 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
    )

    # Akk is consumed as a dense [BT, BT] tile by the following WY stage.
    # Materialize its mathematically-zero upper blocks here so the caller can
    # allocate Akk with torch.empty instead of launching a separate memset.
    p_Akk01 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc0, BC), (BC, BC), (1, 0)
    )
    p_Akk02 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc0, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk03 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc0, 3 * BC), (BC, BC), (1, 0)
    )
    p_Akk12 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc1, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk13 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc1, 3 * BC), (BC, BC), (1, 0)
    )
    p_Akk23 = tl.make_block_ptr(
        Akk, (T, BT), (AKK_STRIDE, 1), (i_tc2, 3 * BC), (BC, BC), (1, 0)
    )

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    b_Akk_zero = tl.zeros([BC, BC], dtype=Akk.dtype.element_ty)
    tl.store(p_Akk01, b_Akk_zero, boundary_check=(0, 1))
    tl.store(p_Akk02, b_Akk_zero, boundary_check=(0, 1))
    tl.store(p_Akk03, b_Akk_zero, boundary_check=(0, 1))
    tl.store(p_Akk12, b_Akk_zero, boundary_check=(0, 1))
    tl.store(p_Akk13, b_Akk_zero, boundary_check=(0, 1))
    tl.store(p_Akk23, b_Akk_zero, boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BK", "NC", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["B", "T"])
def chunk_gdn2_bwd_kernel_intra(
    q,
    k,
    g,
    b,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K

    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K
    db += (i_k * all + bos) * H * BK + i_h * BK

    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    p_b = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_b = tl.load(p_b, boundary_check=(0, 1))

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

    if i_i > 0:
        p_gn = g + i_ti * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0)
            )
            p_gk = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0)
            )
            p_dAqk = tl.make_block_ptr(
                dAqk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0)
            )
            p_dAkk = tl.make_block_ptr(
                dAkk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0)
            )
            b_kj = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_kj * exp2(b_gn - b_gk)
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            b_dq2 += tl.dot(b_dAqk, b_kg)
            b_dk2 += tl.dot(b_dAkk, b_kg)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * H * BT + i_i * BC
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(
                b_g,
                tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16),
                axis=0,
            )
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]

        p_dAqk = tl.make_block_ptr(
            dAqk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
        )
        p_dAkk = tl.make_block_ptr(
            dAkk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
        )
        b_dAqk_diag_qk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_qk = (
            (o_i[:, None] >= o_i[None, :])
            & ((i_ti + o_i[:, None]) < T)
            & ((i_ti + o_i[None, :]) < T)
        )
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.0)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.0)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.0)
        exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.0)
        exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.0)

        b_k_exp_diag_qk = b_k * exp_neg_b_g_diag_qk
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] >= j
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.0)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.0)

            p_kj += H * K
            p_gkj += H * K

    # db gets the elementwise b * k contribution from dk2 (before b is folded
    # back into dk2 itself); then dk2 is rescaled by the channel-wise gate.
    b_db_tile = b_dk2 * b_k
    b_dk2 = b_dk2 * b_b

    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(
        dq2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0)
    )
    p_db = tl.make_block_ptr(db, (T, BK), (H * BK, 1), (i_ti, 0), (BC, BK), (1, 0))

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db_tile.to(p_db.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0)
            )
            p_k = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0)
            )
            p_gk = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0)
            )
            p_bj = tl.make_block_ptr(
                b, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0)
            )
            p_dAqk = tl.make_block_ptr(
                dAqk,
                (BT, T),
                (1, H * BT),
                (i_i * BC, i_t * BT + i_j * BC),
                (BC, BC),
                (0, 1),
            )
            p_dAkk = tl.make_block_ptr(
                dAkk,
                (BT, T),
                (1, H * BT),
                (i_i * BC, i_t * BT + i_j * BC),
                (BC, BC),
                (0, 1),
            )
            b_bj = tl.load(p_bj, boundary_check=(0, 1))
            b_qj = tl.load(p_q, boundary_check=(0, 1))
            b_kbj = tl.load(p_k, boundary_check=(0, 1)) * b_bj
            b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kbj * tl.where(m_j[:, None], b_gkn, 0)
            b_dkt += tl.dot(b_dAqk, b_qg)
            b_dkt += tl.dot(b_dAkk, b_kbg)
        b_dkt *= exp2(b_gn - b_g)

    o_dA = i_ti * H * BT + i_i * BC + o_i
    p_qj = q + i_ti * H * K + o_k
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k
    p_bj_ptr = b + i_ti * H * K + o_k

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(
                b_g,
                tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16),
                axis=0,
            )
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        p_dAqk = tl.make_block_ptr(
            dAqk, (BT, T), (1, H * BT), (i_i * BC, i_ti), (BC, BC), (0, 1)
        )
        p_dAkk = tl.make_block_ptr(
            dAkk, (BT, T), (1, H * BT), (i_i * BC, i_ti), (BC, BC), (0, 1)
        )
        b_dAqk_diag_kk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_kk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_kk = (
            (o_i[:, None] <= o_i[None, :])
            & ((i_ti + o_i[:, None]) < T)
            & ((i_ti + o_i[None, :]) < T)
        )
        m_j_diag_kk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.0)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.0)
        b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.0)
        exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.0)
        exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.0)

        b_q_exp = b_q * exp_b_g_diag_kk
        b_kb_exp = b_k * b_b * exp_b_g_diag_kk

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp) * exp_neg_b_g_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp) * exp_neg_b_g_diag_kk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j * H * BT)
            b_dAkk = tl.load(dAkk + o_dA + j * H * BT)
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_bj = tl.load(p_bj_ptr, mask=m_k, other=0).to(tl.float32)
            b_kbj = b_kj * b_bj
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.0)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.0)

            p_qj += H * K
            p_kj += H * K
            p_gkj += H * K
            p_bj_ptr += H * K

    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(
        dk2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0)
    )
    p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(
        dg2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0)
    )

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _gdn2_inter_row_split_producer(
    q,
    k,
    g,
    b,
    Aqk,
    raw,
    diag,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Build one 16-token row of a 64-token sub-chunk tile per program."""
    pid0 = tl.program_id(0)
    bh = tl.program_id(1)
    tile, row = pid0 // 4, pid0 % 4
    ib, ih = bh // H, bh % H
    qkv_base = ((ib * T) * H + ih) * K
    aqk_base = ((ib * T) * H + ih) * BT
    row_start = tile * BT + row * BC
    oi = tl.arange(0, BC)
    ok = tl.arange(0, BK)

    # Diagonal accumulators and three possible lower off-diagonal blocks.
    aq_diag = tl.zeros((BC, BC), tl.float32)
    ak_diag = tl.zeros((BC, BC), tl.float32)
    aq0 = tl.zeros((BC, BC), tl.float32)
    ak0 = tl.zeros((BC, BC), tl.float32)
    aq1 = tl.zeros((BC, BC), tl.float32)
    ak1 = tl.zeros((BC, BC), tl.float32)
    aq2 = tl.zeros((BC, BC), tl.float32)
    ak2 = tl.zeros((BC, BC), tl.float32)

    for kk in range(tl.cdiv(K, BK)):
        ko = kk * BK + ok
        p_q = tl.make_block_ptr(
            q + qkv_base,
            (T, K),
            (H * K, 1),
            (row_start, kk * BK),
            (BC, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + qkv_base,
            (T, K),
            (H * K, 1),
            (row_start, kk * BK),
            (BC, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + qkv_base,
            (T, K),
            (H * K, 1),
            (row_start, kk * BK),
            (BC, BK),
            (1, 0),
        )
        p_b = tl.make_block_ptr(
            b + qkv_base,
            (T, K),
            (H * K, 1),
            (row_start, kk * BK),
            (BC, BK),
            (1, 0),
        )
        q_i = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        k_i = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        g_i = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_i = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)

        mid = row_start + min(BC // 2, T - row_start - 1)
        g_diag = (
            tl.load(
                g + qkv_base + mid * H * K + ko,
                mask=ko < K,
                other=0.0,
            )
            .to(tl.float32)[None, :]
        )
        g_origin = (
            tl.load(
                g + qkv_base + row_start * H * K + ko,
                mask=ko < K,
                other=0.0,
            )
            .to(tl.float32)[None, :]
        )
        q_diag = q_i * tl.exp2(g_i - g_diag)
        k_diag = k_i * tl.exp2(g_diag - g_i)
        b_diag = (b_i * k_i) * tl.exp2(g_i - g_diag)
        aq_diag += tl.dot(q_diag, tl.trans(k_diag))
        ak_diag += tl.dot(b_diag, tl.trans(k_diag))

        if row > 0:
            p_k0 = tl.make_block_ptr(
                k + qkv_base,
                (T, K),
                (H * K, 1),
                (tile * BT, kk * BK),
                (BC, BK),
                (1, 0),
            )
            p_g0 = tl.make_block_ptr(
                g + qkv_base,
                (T, K),
                (H * K, 1),
                (tile * BT, kk * BK),
                (BC, BK),
                (1, 0),
            )
            k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
            g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)
            aq0 += tl.dot(
                q_i * tl.exp2(g_i - g_origin),
                tl.trans(k0 * tl.exp2(g_origin - g0)),
            )
            ak0 += tl.dot(
                (b_i * k_i) * tl.exp2(g_i - g_origin),
                tl.trans(k0 * tl.exp2(g_origin - g0)),
            )
        if row > 1:
            p_k1 = tl.make_block_ptr(
                k + qkv_base,
                (T, K),
                (H * K, 1),
                (tile * BT + BC, kk * BK),
                (BC, BK),
                (1, 0),
            )
            p_g1 = tl.make_block_ptr(
                g + qkv_base,
                (T, K),
                (H * K, 1),
                (tile * BT + BC, kk * BK),
                (BC, BK),
                (1, 0),
            )
            k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            aq1 += tl.dot(
                q_i * tl.exp2(g_i - g_origin),
                tl.trans(k1 * tl.exp2(g_origin - g1)),
            )
            ak1 += tl.dot(
                (b_i * k_i) * tl.exp2(g_i - g_origin),
                tl.trans(k1 * tl.exp2(g_origin - g1)),
            )
        if row > 2:
            p_k2 = tl.make_block_ptr(
                k + qkv_base,
                (T, K),
                (H * K, 1),
                (tile * BT + 2 * BC, kk * BK),
                (BC, BK),
                (1, 0),
            )
            p_g2 = tl.make_block_ptr(
                g + qkv_base,
                (T, K),
                (H * K, 1),
                (tile * BT + 2 * BC, kk * BK),
                (BC, BK),
                (1, 0),
            )
            k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
            g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            aq2 += tl.dot(
                q_i * tl.exp2(g_i - g_origin),
                tl.trans(k2 * tl.exp2(g_origin - g2)),
            )
            ak2 += tl.dot(
                (b_i * k_i) * tl.exp2(g_i - g_origin),
                tl.trans(k2 * tl.exp2(g_origin - g2)),
            )

    lower = oi[:, None] >= oi[None, :]
    strict = oi[:, None] > oi[None, :]
    eye = (oi[:, None] == oi[None, :]).to(tl.float32)
    aq_diag = tl.where(lower, aq_diag, 0.0)
    ak_diag = tl.where(strict, ak_diag, 0.0)
    ai = -ak_diag
    for r in range(2, BC):
        ar = -tl.sum(tl.where((oi == r)[:, None], ak_diag, 0.0), axis=0)
        ar = tl.where(oi < r, ar, 0.0)
        ar += tl.sum(ar[:, None] * ai, axis=0)
        ai = tl.where((oi == r)[:, None], ar, ai)
    ai += eye
    diag_ptr = diag + (((ib * NT + tile) * H + ih) * 4 + row) * BC * BC
    tl.store(diag_ptr + oi[:, None] * BC + oi[None, :], ai)

    p_diag = tl.make_block_ptr(
        Aqk + aqk_base,
        (T, BT),
        (H * BT, 1),
        (row_start, row * BC),
        (BC, BC),
        (1, 0),
    )
    tl.store(p_diag, (aq_diag * SCALE).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    raw_base = (((ib * NT + tile) * H + ih) * 4 + row) * 4 * BC * BC
    if row > 0:
        p0 = tl.make_block_ptr(
            Aqk + aqk_base,
            (T, BT),
            (H * BT, 1),
            (row_start, 0),
            (BC, BC),
            (1, 0),
        )
        tl.store(p0, (aq0 * SCALE).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(
            raw + raw_base + oi[:, None] * BC + oi[None, :],
            ak0.to(raw.dtype.element_ty),
        )
    if row > 1:
        p1 = tl.make_block_ptr(
            Aqk + aqk_base,
            (T, BT),
            (H * BT, 1),
            (row_start, BC),
            (BC, BC),
            (1, 0),
        )
        tl.store(p1, (aq1 * SCALE).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(
            raw + raw_base + BC * BC + oi[:, None] * BC + oi[None, :],
            ak1.to(raw.dtype.element_ty),
        )
    if row > 2:
        p2 = tl.make_block_ptr(
            Aqk + aqk_base,
            (T, BT),
            (H * BT, 1),
            (row_start, 2 * BC),
            (BC, BC),
            (1, 0),
        )
        tl.store(p2, (aq2 * SCALE).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(
            raw + raw_base + 2 * BC * BC + oi[:, None] * BC + oi[None, :],
            ak2.to(raw.dtype.element_ty),
        )

    zero = tl.zeros((BC, BC), tl.float32).to(Aqk.dtype.element_ty)
    if row < 1:
        tl.store(
            tl.make_block_ptr(
                Aqk + aqk_base,
                (T, BT),
                (H * BT, 1),
                (row_start, BC),
                (BC, BC),
                (1, 0),
            ),
            zero,
            boundary_check=(0, 1),
        )
    if row < 2:
        tl.store(
            tl.make_block_ptr(
                Aqk + aqk_base,
                (T, BT),
                (H * BT, 1),
                (row_start, 2 * BC),
                (BC, BC),
                (1, 0),
            ),
            zero,
            boundary_check=(0, 1),
        )
    if row < 3:
        tl.store(
            tl.make_block_ptr(
                Aqk + aqk_base,
                (T, BT),
                (H * BT, 1),
                (row_start, 3 * BC),
                (BC, BC),
                (1, 0),
            ),
            zero,
            boundary_check=(0, 1),
        )


@triton.jit
def _gdn2_inter_row_split_solve(
    raw,
    diag,
    akk,
    T: tl.constexpr,
    H: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // H
    head = batch_head % H
    block_idx = tl.arange(0, BC)

    diag_base = (((batch * NT + tile) * H + head) * 4) * BC * BC
    diag_stride = BC * BC
    diag_row = block_idx[:, None] * BC
    diag_col = block_idx[None, :]
    d0 = tl.load(diag + diag_base + diag_row + diag_col)
    d1 = tl.load(diag + diag_base + diag_stride + diag_row + diag_col)
    d2 = tl.load(diag + diag_base + 2 * diag_stride + diag_row + diag_col)
    d3 = tl.load(diag + diag_base + 3 * diag_stride + diag_row + diag_col)

    raw_base = (((batch * NT + tile) * H + head) * 4) * 4 * BC * BC
    raw_stride = BC * BC
    raw_row = block_idx[:, None] * BC
    raw_col = block_idx[None, :]
    a10 = tl.load(
        raw + raw_base + (1 * 4 + 0) * raw_stride + raw_row + raw_col
    ).to(tl.float32)
    a20 = tl.load(
        raw + raw_base + (2 * 4 + 0) * raw_stride + raw_row + raw_col
    ).to(tl.float32)
    a21 = tl.load(
        raw + raw_base + (2 * 4 + 1) * raw_stride + raw_row + raw_col
    ).to(tl.float32)
    a30 = tl.load(
        raw + raw_base + (3 * 4 + 0) * raw_stride + raw_row + raw_col
    ).to(tl.float32)
    a31 = tl.load(
        raw + raw_base + (3 * 4 + 1) * raw_stride + raw_row + raw_col
    ).to(tl.float32)
    a32 = tl.load(
        raw + raw_base + (3 * 4 + 2) * raw_stride + raw_row + raw_col
    ).to(tl.float32)

    i10 = -tl.dot(tl.dot(d1, a10), d0)
    i21 = -tl.dot(tl.dot(d2, a21), d1)
    i32 = -tl.dot(tl.dot(d3, a32), d2)
    i20 = -tl.dot(d2, tl.dot(a20, d0) + tl.dot(a21, i10))
    i31 = -tl.dot(d3, tl.dot(a31, d1) + tl.dot(a32, i21))
    i30 = -tl.dot(
        d3,
        tl.dot(a30, d0) + tl.dot(a31, i10) + tl.dot(a32, i20),
    )

    output_base = ((batch * T) * H + head) * BT
    zero_block = tl.zeros((BC, BC), tl.float32).to(akk.dtype.element_ty)
    output_shape = (T, BT)
    output_stride = (H * BT, 1)

    p00 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT, 0),
        (BC, BC),
        (1, 0),
    )
    p10 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + BC, 0),
        (BC, BC),
        (1, 0),
    )
    p11 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + BC, BC),
        (BC, BC),
        (1, 0),
    )
    p20 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + 2 * BC, 0),
        (BC, BC),
        (1, 0),
    )
    p21 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + 2 * BC, BC),
        (BC, BC),
        (1, 0),
    )
    p22 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + 2 * BC, 2 * BC),
        (BC, BC),
        (1, 0),
    )
    p30 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + 3 * BC, 0),
        (BC, BC),
        (1, 0),
    )
    p31 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + 3 * BC, BC),
        (BC, BC),
        (1, 0),
    )
    p32 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + 3 * BC, 2 * BC),
        (BC, BC),
        (1, 0),
    )
    p33 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + 3 * BC, 3 * BC),
        (BC, BC),
        (1, 0),
    )
    p01 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT, BC),
        (BC, BC),
        (1, 0),
    )
    p02 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT, 2 * BC),
        (BC, BC),
        (1, 0),
    )
    p03 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT, 3 * BC),
        (BC, BC),
        (1, 0),
    )
    p12 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + BC, 2 * BC),
        (BC, BC),
        (1, 0),
    )
    p13 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + BC, 3 * BC),
        (BC, BC),
        (1, 0),
    )
    p23 = tl.make_block_ptr(
        akk + output_base,
        output_shape,
        output_stride,
        (tile * BT + 2 * BC, 3 * BC),
        (BC, BC),
        (1, 0),
    )

    output_dtype = akk.dtype.element_ty
    tl.store(p00, d0.to(output_dtype))
    tl.store(p10, i10.to(output_dtype))
    tl.store(p11, d1.to(output_dtype))
    tl.store(p20, i20.to(output_dtype))
    tl.store(p21, i21.to(output_dtype))
    tl.store(p22, d2.to(output_dtype))
    tl.store(p30, i30.to(output_dtype))
    tl.store(p31, i31.to(output_dtype))
    tl.store(p32, i32.to(output_dtype))
    tl.store(p33, d3.to(output_dtype))
    tl.store(p01, zero_block)
    tl.store(p02, zero_block)
    tl.store(p03, zero_block)
    tl.store(p12, zero_block)
    tl.store(p13, zero_block)
    tl.store(p23, zero_block)


def chunk_gdn2_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
    store_kg_head_major: bool = False,
    store_akk_head_major: bool = False,
    fuse_gate_cumsum: bool = False,
    gk_out: torch.Tensor | None = None,
):
    """Intra-chunk forward: build Aqk, Akk_inv, and the WY auxiliaries (w, u)."""
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16
    if fuse_gate_cumsum:
        if gk_out is None:
            raise ValueError(
                "gk_out is required when fusing gate cumsum into inter-solve"
            )
    else:
        gk_out = gk
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    akk_head_major = store_akk_head_major and cu_seqlens is None
    if akk_head_major:
        Akk_storage = torch.empty(B, H, T, BT, device=k.device, dtype=k.dtype)
        Akk = Akk_storage.as_strided(
            (B, T, H, BT), (H * T * BT, BT, T * BT, 1)
        )
        Akk_kernel = Akk_storage
    else:
        # The fused inter-solve kernel writes both the lower-triangular values
        # and the structurally-zero upper blocks, so no separate zero-fill is needed.
        Akk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
        Akk_kernel = Akk
    # The safe path produces diagonal inverses inside the inter-solve kernel,
    # so it does not need the B*T*H*BC FP32 global workspace.
    Akkd = (
        None
        if safe_gate
        else torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)
    )

    # The row-split schedule replaces one large 4-row inter-solve CTA with
    # producer CTAs per 16-token row plus a compact 4x4 solve. Measurements
    # show a stable end-to-end gain for this fixed-length BF16 family. Keep a
    # runtime off switch and leave every unvalidated case on the original path.
    inter_row_split_requested = _inter_row_split_enabled()
    fixed_length_input = cu_seqlens is None and chunk_indices is None
    row_split_shape_supported = (
        chunk_size == 64
        and B in (1, 2)
        and 8192 <= T <= 32768
        and T % 64 == 0
        and (H in (8, 16, 32) or (H == 96 and T <= 16384))
        and K == 128
        and v.shape[-1] == 128
    )
    row_split_dtype_supported = (
        q.dtype == torch.bfloat16
        and k.dtype == q.dtype
        and b.dtype == q.dtype
        and v.dtype == q.dtype
        and gk.dtype == torch.float32
    )
    use_inter_row_split = (
        not fuse_gate_cumsum
        and inter_row_split_requested
        and safe_gate
        and not disable_recompute
        and fixed_length_input
        and row_split_shape_supported
        and row_split_dtype_supported
        and not store_kg_head_major
        and not akk_head_major
    )

    if use_inter_row_split:
        row_raw = torch.empty(
            B, NT, H, 4, 4, BC, BC, device=k.device, dtype=k.dtype
        )
        row_diag = torch.empty(
            B, NT, H, 4, BC, BC, device=k.device, dtype=torch.float32
        )
        _gdn2_inter_row_split_producer[(NT * 4, B * H)](
            q,
            k,
            gk,
            b,
            Aqk,
            row_raw,
            row_diag,
            T,
            H,
            K,
            NT,
            BT,
            BC,
            64,
            scale,
            num_warps=1,
            num_stages=1,
        )
        _gdn2_inter_row_split_solve[(NT, B * H)](
            row_raw,
            row_diag,
            Akk_kernel,
            T,
            H,
            NT,
            BT,
            BC,
            num_warps=1,
            num_stages=1,
        )
    else:
        if not safe_gate:
            chunk_gdn2_fwd_intra_token_parallel(
                q=q,
                k=k,
                gk=gk,
                b=b,
                Aqk=Aqk,
                Akk=Akkd,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_size=BT,
                sub_chunk_size=BC,
            )

        # Fused inter-subchunk Akk + solve_tril → full Akk_inv.
        grid = (NT, B * H)
        chunk_gdn2_fwd_kernel_inter_solve_fused[grid](
            q=q,
            k=k,
            g=gk,
            g_out=gk_out,
            b=b,
            Aqk=Aqk,
            # Pointer arguments cannot be None; the safe specialization removes
            # every Akkd access at compile time, so Akk is a harmless placeholder.
            Akkd=Akk if Akkd is None else Akkd,
            Akk=Akk_kernel,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            USE_SAFE_GATE=safe_gate,
            FUSE_SAFE_DIAG=safe_gate,
            SAFE_DIAG_REUSE=_gdn2_safe_diag_reuse_mode(
                safe_gate=safe_gate,
                fuse_gate_cumsum=fuse_gate_cumsum,
            ),
            FUSE_GATE_CUMSUM=fuse_gate_cumsum,
            GATE_SCALE=RCP_LN2,
            AKK_HEAD_MAJOR=akk_head_major,
        )

    gk = gk_out

    w, u, qg, kg = recompute_w_u_fwd_gdn2(
        k=k,
        v=v,
        b=b,
        w_gate=w_gate,
        A=Akk_kernel,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        kg_head_major=store_kg_head_major,
        a_head_major=akk_head_major,
    )
    return w, u, qg, kg, Aqk, Akk


def chunk_gdn2_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    """Intra-chunk backward: q, k, g, b contributions from dAqk, dAkk.

    Adds the intra-chunk piece of db (channel-wise, K-dim) to the running ``db``
    accumulator passed in.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)
    db2 = q.new_empty(NK, B, T, H, BK, dtype=torch.float32)
    dg2 = torch.empty_like(dg, dtype=torch.float32)

    grid = (NK * NC, NT, B * H)
    chunk_gdn2_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        b=b,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg,
        dg2=dg2,
        db=db2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        SAFE_GATE=safe_gate,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    dq = dq2
    dk = dk2
    db2_combined = (
        db2.permute(1, 2, 3, 0, 4).contiguous().reshape(B, T, H, NK * BK)[..., :K]
    )
    db = db.add_(db2_combined)
    dg = dg2
    return dq, dk, db, dg


# ---- Forward pipeline and GDN2-local output kernel ----

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Forward orchestration for GDN-2 chunkwise training.

RCP_LN2 = 1.4426950408889634


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"BK": BK, "BV": BV},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "H", "K", "HV", "STATE_V_FIRST", "V"],
)
@triton.jit(do_not_specialize=["T"])
def _gdn2_fwd_kernel_o(
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
    """GDN2-local output kernel; deliberately independent of the GLA module."""
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
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

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
        else:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

    b_o *= scale
    p_v = tl.make_block_ptr(
        v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_A = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _chunk_gdn2_fwd_o_gk(
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
    """Launch GDN2's standalone output stage with GDN2-owned autotuning."""
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * HV)

    _gdn2_fwd_kernel_o[grid](
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


def chunk_gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
):
    """Top-level GDN-2 forward pipeline.

    The pipeline is:
      1. Compute the base-2 log-decay cumsum within each chunk
         (``kda_gate_chunk_cumsum`` if ``use_gate_in_kernel`` else
         ``chunk_local_cumsum``).
      2. Build the intra-chunk score matrices (Aqk, Akk_inv) and the WY
         auxiliaries (w_wy, u_wy, qg, kg) via ``chunk_gdn2_fwd_intra``.
      3. Run the inter-chunk state recurrence (shared with KDA / GDN v1).
      4. Compose the output with the GDN2-local Triton output kernel.

    Returns ``(o, final_state, g_cumsum, Aqk, Akk, w_wy, u_wy, qg, kg, v_new,
    h, initial_state)``.
    """
    T = q.shape[1]
    K = k.shape[-1]
    # The Hygon sub-chunk path reuses the K/gate tile across four BC=16
    # sub-chunks.  It is beneficial for large head dimensions or long
    # sequences.  Keep the token-parallel fallback for small K/short T, where
    # the extra sub-chunk scheduling cost can dominate.  Callers can still
    # force either schedule with an explicit bool.
    if safe_gate is None:
        safe_gate = True
    fuse_gate_cumsum = _should_fuse_gate_cumsum_inter(
        q=q,
        k=k,
        v=v,
        b=b,
        seq_len=T,
        key_dim=K,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        use_gate_in_kernel=use_gate_in_kernel,
        disable_recompute=disable_recompute,
        return_intermediate_states=return_intermediate_states,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    g_cumsum_out = None
    if use_gate_in_kernel:
        g = kda_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    elif fuse_gate_cumsum:
        # The inter-solve experiment consumes raw gate values and writes the
        # cumulative base-2 gate tensor for downstream H/O kernels.
        g_cumsum_out = torch.empty_like(g, dtype=torch.float32)
    else:
        g = chunk_local_cumsum(
            # Fuse the base-2 conversion into the cumsum kernel.  The old
            # path materialized ``g.float() * RCP_LN2`` before launching the
            # prefix-sum kernel, creating an avoidable full-size intermediate.
            g=g,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
            scale=RCP_LN2,
        )

    # Keep H->O eligibility separate from the measured shape-family evidence.
    fuse_h_o = _should_fuse_gdn2_h_o(
        q=q,
        seq_len=T,
        key_dim=K,
        value_dim=v.shape[-1],
        cu_seqlens=cu_seqlens,
        return_intermediate_states=return_intermediate_states,
        disable_recompute=disable_recompute,
    )
    # The row-split schedule itself is selected in chunk_gdn2_fwd_intra.
    # H=8/16 cases already have contiguous intermediates because H->O is off,
    # while long H=32 explicitly avoids head-major Akk. H=96 is different:
    # H->O and both head-major layouts are otherwise selected, so its measured
    # BF16 T=8192..16384 family needs this layout-only override. This predicate
    # does not enable row-split; it only prevents incompatible KG/Akk layouts.
    inter_row_split_requested = _inter_row_split_enabled()
    h96_row_split_layout_override = _should_override_gdn2_h96_row_split_layout(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        seq_len=T,
        key_dim=K,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        fuse_h_o=fuse_h_o,
        safe_gate=safe_gate,
        inter_row_split_requested=inter_row_split_requested,
    )
    kg_head_major = _should_store_gdn2_kg_head_major(
        q=q,
        key_dim=K,
        fuse_h_o=fuse_h_o,
        h96_row_split_layout_override=h96_row_split_layout_override,
    )
    akk_head_major = _should_store_gdn2_akk_head_major(
        q=q,
        seq_len=T,
        key_dim=K,
        value_dim=v.shape[-1],
        fuse_h_o=fuse_h_o,
        safe_gate=safe_gate,
        h96_row_split_layout_override=h96_row_split_layout_override,
    )
    w_wy, u_wy, qg, kg, Aqk, Akk = chunk_gdn2_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        b=b,
        w_gate=w_gate,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
        store_kg_head_major=kg_head_major,
        store_akk_head_major=akk_head_major,
        fuse_gate_cumsum=fuse_gate_cumsum,
        gk_out=g_cumsum_out,
    )
    if fuse_gate_cumsum:
        g = g_cumsum_out

    # In forward-only fixed-length inference, H and O consume the same
    # chunk-start state.  Stream O directly from the H recurrence to avoid
    # materializing the full [B, num_chunks, H, K, V] history and v_new.
    # Keep the fallback for backward, varlen, and intermediate-state callers.
    o = torch.empty_like(v) if fuse_h_o else None
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w_wy,
        u=u_wy,
        # ``g`` is already a base-2 cumulative gate.  Let the Hygon state
        # kernel consume it directly with exp2, avoiding a full g*ln(2)
        # intermediate tensor and its global write/read.
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        gk_base2=True,
        scale=scale,
        q=q if fuse_h_o else None,
        A=Aqk if fuse_h_o else None,
        o=o,
        fuse_o=fuse_h_o,
        kg_head_major=kg_head_major,
    )
    if state_v_first:
        if h is not None:
            h = h.transpose(-1, -2).contiguous()
        if final_state is not None:
            final_state = final_state.transpose(-1, -2).contiguous()

    if not fuse_h_o:
        o = _chunk_gdn2_fwd_o_gk(
            q=q,
            v=v_new,
            g=g,
            A=Aqk,
            h=h,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )

    if disable_recompute is False:
        # Free intermediates that the backward will recompute.
        w_wy, u_wy, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            h = None
        if use_gate_in_kernel:
            g = None
    return o, final_state, g, Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h, initial_state
