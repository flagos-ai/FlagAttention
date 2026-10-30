# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Public GDN v1 entry point for the Ascend backend.

Signature and layout conversions mirror ``flag_attn.FLA.gated_delta_rule.api``;
the NVIDIA-only TLE/direct fast paths are intentionally not used here.  ``BT``
keeps the upstream contract (only 16/32/64 are meaningful); the internal chunk
length is ``ascend_chunk_size()`` for the default BT=64.

The fast path is host-bound: a handful of kernel launches per call, where the
Python/Triton launch side costs more than the device work, so this wrapper keeps
the per-call Python as lean as possible while preserving exactly the same
validation semantics as the generic entry point:

* the five ``_as_seq_first`` calls and the ``_validate_inputs`` dict/loop are
  inlined into straight-line checks (same exceptions and messages);
* ``is_contiguous()`` is checked directly instead of through the helper, so a
  contiguous input skips the call entirely;
* the orchestration call is positional (no kwargs dict).
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401

from flag_attn.FLA.gated_delta_rule.api import _l2_normalize_last_dim

from .chunk import (
    _use_fast_path,
    _use_fused,
    ascend_chunk_size,
    chunk_gated_delta_rule_fwd,
)
from .fused_ascend import strided_fold_supported

_ALLOWED_BT = (16, 32, 64)


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = True,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if BT not in _ALLOWED_BT:
        raise ValueError(
            f"chunk_gated_delta_rule supports BT in {_ALLOWED_BT}, got {BT}"
        )

    # --- inlined _as_seq_first (layout conversion to sequence-first) --------
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if not isinstance(k, torch.Tensor):
        raise TypeError("k must be a torch.Tensor")
    if not isinstance(v, torch.Tensor):
        raise TypeError("v must be a torch.Tensor")
    if not isinstance(beta, torch.Tensor):
        raise TypeError("beta must be a torch.Tensor")
    if not isinstance(g, torch.Tensor):
        raise TypeError("g must be a torch.Tensor")
    if q.ndim != 4:
        raise ValueError(f"q must be 4D, got shape {tuple(q.shape)}")
    if k.ndim != 4:
        raise ValueError(f"k must be 4D, got shape {tuple(k.shape)}")
    if v.ndim != 4:
        raise ValueError(f"v must be 4D, got shape {tuple(v.shape)}")
    if beta.ndim != 3:
        raise ValueError(f"beta must be 3D, got shape {tuple(beta.shape)}")
    if g.ndim != 3:
        raise ValueError(f"g must be 3D, got shape {tuple(g.shape)}")
    if head_first:
        q_seq = q.transpose(1, 2)
        k_seq = k.transpose(1, 2)
        v_seq = v.transpose(1, 2)
        beta_seq = beta.transpose(1, 2)
        g_seq = g.transpose(1, 2)
    else:
        q_seq, k_seq, v_seq, beta_seq, g_seq = q, k, v, beta, g

    # --- inlined _validate_inputs (same checks and messages) ----------------
    B, T, Hg, K = q_seq.shape
    Bk, Tk, Hk, Kk = k_seq.shape
    Bv, Tv, H, V = v_seq.shape
    q_dev, q_dtype = q_seq.device, q_seq.dtype
    if k_seq.device != q_dev:
        raise ValueError("k must be on the same device as q")
    if v_seq.device != q_dev:
        raise ValueError("v must be on the same device as q")
    if beta_seq.device != q_dev:
        raise ValueError("beta must be on the same device as q")
    if g_seq.device != q_dev:
        raise ValueError("g must be on the same device as q")
    if k_seq.dtype != q_dtype:
        raise ValueError("k must have the same dtype as q")
    if v_seq.dtype != q_dtype:
        raise ValueError("v must have the same dtype as q")
    if beta_seq.dtype != q_dtype:
        raise ValueError("beta must have the same dtype as q")
    if g_seq.dtype != q_dtype:
        raise ValueError("g must have the same dtype as q")
    if (Bk, Tk, Hk, Kk) != (B, T, Hg, K):
        raise ValueError(
            "q and k must have matching [B, T, Hq, K] shapes after layout conversion"
        )
    if (Bv, Tv) != (B, T):
        raise ValueError("v must have matching B and T dimensions with q/k")
    if H % Hg != 0:
        raise ValueError("the q/k head count must divide the v head count")
    if beta_seq.shape != (B, T, H):
        raise ValueError(
            f"beta must have shape {(B, T, H)} after layout conversion, "
            f"got {tuple(beta_seq.shape)}"
        )
    if g_seq.shape != (B, T, H):
        raise ValueError(
            f"g must have shape {(B, T, H)} after layout conversion, "
            f"got {tuple(g_seq.shape)}"
        )
    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1D tensor")
        if cu_seqlens.dtype != torch.long:
            raise ValueError("cu_seqlens must have dtype torch.long")
        if cu_seqlens.device != q_dev:
            raise ValueError("cu_seqlens must be on the same device as q")
        if B != 1:
            raise ValueError("cu_seqlens packed varlen inputs must use B=1")
    if initial_state is not None:
        if initial_state.device != q_dev:
            raise ValueError("initial_state must be on the same device as q")
        if initial_state.dtype != q_dtype:
            raise ValueError("initial_state must have the same dtype as q")
        expected_n = B if cu_seqlens is None else cu_seqlens.numel() - 1
        expected_shape = (expected_n, H, K, V)
        if initial_state.shape != expected_shape:
            raise ValueError(
                f"initial_state must have shape {expected_shape}, "
                f"got {tuple(initial_state.shape)}"
            )

    if scale is None:
        scale = K**-0.5

    # The stride-aware fold is restricted to the reviewed fixed-length fast
    # path. Head-first inputs may be noncontiguous after the layout transpose;
    # sequence-first inputs use it only when all five public tensors are
    # already contiguous. The workload-dependent decision lives in
    # ``fused_ascend.strided_fold_supported`` so this wrapper and
    # ``fold_ascend`` cannot disagree on which shapes take which kernel.
    # Other calls take the materialised path and keep its public
    # validation/norm behavior.
    chunk_size = ascend_chunk_size() if BT == 64 else BT
    strided_candidate = strided_fold_supported(
        B, T, H, Hg, K, V, q_seq.dtype, cu_seqlens, chunk_size
    )
    sequence_first_contiguous = False
    if strided_candidate and not head_first:
        # Only pay for the contiguity probes when the strided path is actually
        # a candidate; small shapes keep the materialised path with no extra
        # per-call host work (the fast path is host-bound).
        sequence_first_contiguous = (
            q_seq.is_contiguous()
            and k_seq.is_contiguous()
            and v_seq.is_contiguous()
            and beta_seq.is_contiguous()
            and g_seq.is_contiguous()
        )
    head_first_strided = (
        strided_candidate
        and (head_first or sequence_first_contiguous)
        and not use_qk_l2norm_in_kernel
        and _use_fast_path(q_seq, k_seq, v_seq)
        and _use_fused()
    )
    if not head_first_strided:
        # Most chunk kernels index sequence-first tensors with contiguous
        # strides, so unsupported inputs take the materialised path.
        if not q_seq.is_contiguous():
            q_seq = q_seq.contiguous()
        if not k_seq.is_contiguous():
            k_seq = k_seq.contiguous()
        if not v_seq.is_contiguous():
            v_seq = v_seq.contiguous()
        if not beta_seq.is_contiguous():
            beta_seq = beta_seq.contiguous()
        if not g_seq.is_contiguous():
            g_seq = g_seq.contiguous()

    if use_qk_l2norm_in_kernel:
        q_seq = _l2_normalize_last_dim(q_seq)
        k_seq = _l2_normalize_last_dim(k_seq)

    _, o, _, final_state, _, _, _ = chunk_gated_delta_rule_fwd(
        q_seq,
        k_seq,
        v_seq,
        g_seq,
        beta_seq,
        float(scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_size,
        head_first_strided,
    )
    if head_first:
        o = o.transpose(1, 2)
    return o, final_state if output_final_state else None


__all__ = ["chunk_gated_delta_rule"]
