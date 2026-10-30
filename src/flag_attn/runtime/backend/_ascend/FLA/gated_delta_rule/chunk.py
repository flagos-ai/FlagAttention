# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend orchestration for the GDN v1 chunk forward.

Three paths:

* **fused path** (bf16/fp16, ``K <= 128``, ``V <= 128``, default chunk 64): three
  Triton launches -- ``cumsum(+fp32 beta)`` -> ``prep_w_u`` -> ``fwd_h_o``.
  ``fused_ascend.py`` merges the KKT/Newton inverse with ``recompute_w_u``
  (``A_inv`` never hits GM) and the state scan with the output stage (``h`` and
  ``v_new`` never hit GM).  This is the default because the staged pipeline is
  launch/host bound on this backend, not compute bound.
  ``FLAG_ATTN_ASCEND_GDN_FUSE=0`` selects the staged fast path (5 launches)
  instead.
* **staged fast path**: the same math in five Ascend kernels
  (``wy_fast_ascend.py``, ``fwd_h_ascend.py``, ``chunk_o_ascend.py``), kept as
  an explicit alternative.
* **fallback path** (``K``/``V`` > 128, or
  ``FLAG_ATTN_ASCEND_GDN_FAST=0``): the generic implementation -- ``kkt.py``
  prep plus the generic ``wy_fast``/``chunk_delta_h``/``chunk_o`` kernels at
  ``BT=32``, so fp16 and K=256 keep working.

All Ascend kernels here share the same backend rules (see the module
docstrings): no autotune, K in a single tile, and per-token decay applied
through *2D factor tiles gathered from fp32 buffers* -- a 1D->2D broadcast
multiply is slow on this backend, and bf16 2D gathers are much slower than
fp32 gathers.

``FLAG_ATTN_ASCEND_GDN_BT`` selects the internal chunk size (default 64, the
same chunk CANN's AscendC fused op uses); BT=32 remains available as an
explicit request and as the fallback chunk size.
"""

from __future__ import annotations

import os

import torch
import triton

from flag_attn.FLA.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from flag_attn.FLA.chunk_o import chunk_fwd_o
from flag_attn.FLA.gated_delta_rule.wy_fast import recompute_w_u_fwd

from .chunk_o_ascend import chunk_fwd_o_ascend
from .fused_ascend import (
    fold_ascend,
    fold_headfirst_strided_trim,
    fold_trim_enabled,
    fwd_h_o_2pass_ascend,
    fwd_h_o_ascend,
    prep_affine_ascend,
    prep_w_u_ascend,
)
from .fast_launch import begin_forward, end_forward
from .fwd_h_ascend import chunk_gated_delta_rule_fwd_h_ascend
from .kkt import chunk_gated_delta_rule_prep
from .wy_fast_ascend import (
    chunk_gated_delta_rule_prep_ascend,
    chunk_local_cumsum_ascend,
    recompute_w_u_fwd_ascend,
)

_BT_ENV = "FLAG_ATTN_ASCEND_GDN_BT"
_FAST_ENV = "FLAG_ATTN_ASCEND_GDN_FAST"
_FUSE_ENV = "FLAG_ATTN_ASCEND_GDN_FUSE"
_RAW_INVERSE_ENV = "FLAG_ATTN_ASCEND_GDN_RAW_INVERSE"
_ONEPASS_SMALL_ENV = "FLAG_ATTN_ASCEND_GDN_ONEPASS_SMALL"
# Job-count threshold for the Raw inverse prep route (an empirical value for
# this backend): once the chunk/head grid reaches 1024 jobs the exact two-pass
# scan plus the Raw inverse pays off even for short sequences (NT >= 2).  Below
# it the Raw prep adds inverse materialisation work that the grid cannot hide.
_RAW_INVERSE_MIN_JOBS = 1024
# v1 fold trim (only materialise k_e/k_r) needs the Raw prep path and a grid
# large enough that the extra qe/vb recompute in the affine kernel is
# amortised; again an empirical job count for this backend.
_FOLD_TRIM_MIN_JOBS = 2048

_OFF = {"0", "false", "off", "no"}

# Chunk count from which the exact two-pass state scan is preferred over the
# fused sequential one (an empirical value for this backend; see
# fused_ascend._affine_kernel).  For NT < 32 the two-pass scan can still pay
# off on large grids, so the jobs threshold below is OR-ed in; NT=1 keeps the
# one-pass path because a second pass cannot pay for itself there.
_TWOPASS_MIN_CHUNKS = 32
_TWOPASS_MIN_JOBS = 1024

def _use_fused() -> bool:
    """Fused (fold+prep_w_u+fwd_h_o) is the default; the staged 5-launch path
    remains available via ``FLAG_ATTN_ASCEND_GDN_FUSE=0`` as a fallback."""
    forced = os.environ.get(_FUSE_ENV, "").lower()
    return forced not in _OFF if forced else True


def ascend_chunk_size() -> int:
    return int(os.environ.get(_BT_ENV, "64"))


def _use_fast_path(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if os.environ.get(_FAST_ENV, "1").lower() in _OFF:
        return False
    # The fused fold kernel tiles v_beta with the K slice, so it requires
    # K == V; other widths use the staged fast path.
    return (
        q.dtype in (torch.bfloat16, torch.float16)
        and k.shape[-1] <= 128
        and v.shape[-1] <= 128
        and k.shape[-1] == v.shape[-1]
    )


def _ascend_wu_tile(dim: int) -> int:
    """Return a dot-compatible power-of-two tile, capped at 128."""
    return min(128, max(16, 1 << (dim - 1).bit_length()))


def _fwd_legacy(q, k, v, g, beta, scale, initial_state, output_final_state,
                cu_seqlens, bt):
    g_cum, A = chunk_gated_delta_rule_prep(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=bt,
        output_dtype=k.dtype,
    )
    block_v = 128 if v.shape[-1] == 128 else 64
    pipe_stages = 5 if v.shape[-1] >= 256 else (3 if v.shape[-1] == 128 else 2)
    wu_ascend = (
        q.dtype in (torch.bfloat16, torch.float16)
        and max(k.shape[-1], v.shape[-1]) > 128
        and k.shape[-1] in (64, 128, 256)
        and 0 < v.shape[-1] <= 256
    )
    if wu_ascend:
        # This only replaces legacy W/U with an API key width accepted by the
        # full-K prep dot and dimensions <=256. The kernel supports fixed and
        # packed-varlen W/U (including chunk-index
        # lookup) and independent K/V tails; round each tile up to a valid
        # power of two (capped at 128) so e.g. V=192 uses masked 128-wide
        # tiles.  The <=128 bf16 fused path never reaches here.
        w, u = recompute_w_u_fwd_ascend(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g_cumsum=g_cum,
            cu_seqlens=cu_seqlens,
            bk=_ascend_wu_tile(k.shape[-1]),
            bv=_ascend_wu_tile(v.shape[-1]),
            num_warps=4,
            num_stages=2,
        )
    else:
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g_cumsum=g_cum,
            cu_seqlens=cu_seqlens,
            block_v=block_v,
            pipe_stages=pipe_stages,
        )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_cum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=bt,
        cu_seqlens=cu_seqlens,
    )
    output_bv = min(128, max(16, triton.next_power_of_2(v.shape[-1])))
    fixed_output_supported = (
        q.dtype in (torch.bfloat16, torch.float16)
        and k.shape[-1] in (64, 128, 256)
        and 0 < v.shape[-1] <= 256
        and cu_seqlens is None
    )
    # Use the fixed one-warp output kernel for fixed-length FP16 shapes in this
    # supported width range.  Besides staying within ACL's coreDim limit on
    # large grids, this avoids the generic autotuner's much slower output
    # configs on this backend.  The fixed route is also used for K=V=256 bf16.
    needs_fixed_output = q.dtype == torch.float16 or (
        k.shape[-1] == 256 and v.shape[-1] == 256
    )
    if fixed_output_supported and needs_fixed_output:
        # The fixed-config output kernel bounds each launch to the ACL budget;
        # oversized supported grids are covered by multiple offset flat-PID
        # launches.  A single full-K dot avoids the generic loop-carried fp32
        # dot accumulator, while V is split into 128-wide tiles.  All
        # preceding legacy stages remain unchanged.
        o = chunk_fwd_o_ascend(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g_cum,
            scale=scale,
            cu_seqlens=None,
            chunk_size=bt,
            bv=output_bv,
            num_warps=1,
        )
    else:
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g_cum,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=bt,
        )
    return g_cum, o, A, final_state


def _fwd_fast(q, k, v, g, beta, scale, initial_state, output_final_state,
              cu_seqlens, bt, head_first_strided=False):
    # One vector-only pass produces the chunk-local cumsum and an fp32 copy of
    # ``beta``; the fp32 copy matters because 2D gathers of bf16 data are much
    # slower than fp32 gathers on this backend (see wy_fast_ascend.py).
    if _use_fused():
        # The normal path uses vector folds -> prep (KKT/Newton+w/u+A) ->
        # state scan + output. Long fixed sequences fuse prep with the affine
        # precompute below, removing the W/U GM round trip, the A reload, and
        # one launch. A is still written because this function returns it.
        #
        # v1 fold trim: on BF16 fixed-length Raw-prep grids (jobs >= 2048) the
        # strided fold only materialises k_e/k_r; qe/vb are recomputed inside
        # the Raw affine kernel from q/v and the small fp32 factor buffers,
        # which removes the GM traffic of writing and reloading them.  FP16
        # keeps the full fold.
        nt = -(-g.shape[1] // bt)
        jobs = q.shape[0] * nt * beta.shape[2]
        two_pass_env = os.environ.get("FLAG_ATTN_ASCEND_GDN_TWOPASS", "")
        if two_pass_env:
            expect_two_pass = two_pass_env.lower() not in _OFF
        else:
            expect_two_pass = cu_seqlens is None and (
                nt >= _TWOPASS_MIN_CHUNKS
                or (nt >= 2 and jobs >= _TWOPASS_MIN_JOBS)
            )
        expect_raw = (
            expect_two_pass
            and cu_seqlens is None
            and os.environ.get(_RAW_INVERSE_ENV, "1").lower() not in _OFF
            and bt == 64
            and jobs >= _RAW_INVERSE_MIN_JOBS
            and k.shape[-1] == v.shape[-1]
            and 0 < k.shape[-1] <= 128
            and q.dtype in (torch.bfloat16, torch.float16)
        )
        trim = (
            expect_raw
            and head_first_strided
            and fold_trim_enabled()
            and q.dtype == torch.bfloat16
            and jobs >= _FOLD_TRIM_MIN_JOBS
        )
        trim_ctx = None
        if trim:
            g_cum, ke, kr, trim_ctx = fold_headfirst_strided_trim(
                g, q, k, v, beta, bt,
            )
            qe = vb = None
        else:
            g_cum, qe, ke, kr, vb = fold_ascend(
                g, q, k, v, beta, cu_seqlens, bt,
                head_first_strided=head_first_strided,
            )
        # Optional exact two-pass state scan (pass1 chunk-parallel affine
        # precompute, pass2 with two dots per chunk; see _affine_kernel in
        # fused_ascend.py).  It pays off once the sequential chain is long
        # (NT >= 32) or the chunk/head grid is large enough (jobs >= 1024,
        # NT >= 2), where the pass-1 parallelism dominates the cost of the
        # extra pass.  Override with FLAG_ATTN_ASCEND_GDN_TWOPASS=1/0.
        two_pass_env = os.environ.get("FLAG_ATTN_ASCEND_GDN_TWOPASS", "")
        if two_pass_env:
            two_pass = two_pass_env.lower() not in _OFF
        else:
            nt = -(-g.shape[1] // bt)
            jobs = q.shape[0] * nt * beta.shape[2]
            two_pass = cu_seqlens is None and (
                nt >= _TWOPASS_MIN_CHUNKS
                or (nt >= 2 and jobs >= _TWOPASS_MIN_JOBS)
            )
            # The one-pass scan unrolls two chunks per iteration and wins on
            # tiny head grids where the two-pass pass1 cannot pay for itself.
            # Keep two-pass for B*H >= 16, where the pass-1 parallelism
            # outweighs the extra pass.
            if (
                two_pass
                and os.environ.get(_ONEPASS_SMALL_ENV, "1").lower() not in _OFF
                and bt == 64
                and q.dtype == torch.bfloat16
                and k.shape[-1] == v.shape[-1]
                and 0 < k.shape[-1] <= 128
                and nt >= 8
                and nt % 2 == 0
                and q.shape[0] * beta.shape[2] == 8
            ):
                two_pass = False
        if two_pass and cu_seqlens is None:
            jobs = q.shape[0] * triton.cdiv(q.shape[1], bt) * beta.shape[2]
            raw_inverse = (
                os.environ.get(_RAW_INVERSE_ENV, "1").lower() not in _OFF
                and bt == 64
                and jobs >= _RAW_INVERSE_MIN_JOBS
                and k.shape[-1] == v.shape[-1]
                and 0 < k.shape[-1] <= 128
                and q.dtype in (torch.bfloat16, torch.float16)
            )
            if raw_inverse:
                # This route is chosen for workloads with at least 1024
                # independent chunk/head jobs (see _RAW_INVERSE_MIN_JOBS).
                # Below that the fused prep wins because Raw adds inverse
                # materialization work; one-pass and varlen keep fused prep.
                if trim_ctx is not None:
                    from .raw_inverse_prep import (
                        raw_inverse_prep_affine_trim_ascend,
                    )

                    qe2, au, cu, mw, a_masked = (
                        raw_inverse_prep_affine_trim_ascend(
                            *trim_ctx, ke, kr, bt, output_dtype=q.dtype
                        )
                    )
                else:
                    from .raw_inverse_prep import (
                        raw_inverse_prep_affine_ascend,
                    )

                    qe2, au, cu, mw, a_masked = raw_inverse_prep_affine_ascend(
                        qe, ke, kr, vb, bt, output_dtype=q.dtype
                    )
            else:
                qe2, au, cu, mw, a_masked = prep_affine_ascend(
                    qe, ke, kr, vb, bt, output_dtype=q.dtype
                )
            o, final_state = fwd_h_o_2pass_ascend(
                qe2, au, cu, mw, g_cum, scale, initial_state,
                output_final_state, bt,
            )
            return g_cum, o, a_masked, final_state
        w, u, a_masked = prep_w_u_ascend(
            qe, ke, kr, vb, cu_seqlens, bt, output_dtype=q.dtype
        )
        o, final_state = fwd_h_o_ascend(
            qe, a_masked, kr, w, u, g_cum, scale, initial_state,
            output_final_state, cu_seqlens, bt,
        )
        return g_cum, o, a_masked, final_state
    g_cum, beta32 = chunk_local_cumsum_ascend(
        g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=bt
    )
    g_cum, A = chunk_gated_delta_rule_prep_ascend(
        g=g,
        k=k,
        beta=beta32,
        cu_seqlens=cu_seqlens,
        chunk_size=bt,
        output_dtype=k.dtype,
        g_cum=g_cum,
    )
    w, u = recompute_w_u_fwd_ascend(
        k=k,
        v=v,
        beta=beta32,
        A=A,
        g_cumsum=g_cum,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_ascend(
        k=k,
        w=w,
        u=u,
        g=g_cum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=bt,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o_ascend(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=bt,
    )
    return g_cum, o, A, final_state


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int | None = None,
    head_first_strided: bool = False,
) -> tuple:
    """Forward pass; returns the generic ``(g, o, A, final_state, w, h, v_new)``."""
    chunk_size = int(chunk_size or ascend_chunk_size())
    if _use_fast_path(q, k, v):
        # One device + stream lookup per forward; every fast-path launch is on
        # this stream (see fast_launch.begin_forward).
        begin_forward()
        try:
            g_cum, o, A, final_state = _fwd_fast(
                q, k, v, g, beta, scale, initial_state, output_final_state,
                cu_seqlens, chunk_size, head_first_strided,
            )
        finally:
            end_forward()
    else:
        # fp16 / wide head dims: the generic path is only UB-safe at BT<=32.
        g_cum, o, A, final_state = _fwd_legacy(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, min(chunk_size, 32),
        )
    return g_cum, o, A, final_state, None, None, None


__all__ = ["chunk_gated_delta_rule_fwd", "ascend_chunk_size"]
