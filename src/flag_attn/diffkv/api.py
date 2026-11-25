# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""Public API and host-side dispatch for DiffKV attention.

This module follows the same boundary used by the GLA and MSA operators:
input validation and backend selection stay in the API layer, while the
actual Triton/TLE kernels and launchers live in :mod:`.diffkv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Literal, cast

import torch
import triton

DiffKVMode = Literal["full", "swa"]
DiffKVBackend = Literal["auto", "tle", "triton"]
DiffKVPath = Literal["2d", "3d"]

# Stable operator identifier shared by the public API, pytest marker,
# benchmark adapter, and ``conf/operators.yaml`` inventory.  Keep this
# separate from Triton kernel function names so callers do not depend on
# implementation details.
OP_NAME: Final = "diffkv_attention"


@dataclass(frozen=True)
class DiffKVLayout:
    """Production DiffKV head layout shared by Full Attention and SWA."""

    num_query_heads: int = 64
    full_num_kv_heads: int = 4
    swa_num_kv_heads: int = 8
    head_size_qk: int = 192
    head_size_v: int = 128
    block_size: int = 16


DEFAULT_LAYOUT: Final = DiffKVLayout()
SUPPORTED_MODES: Final = frozenset(("full", "swa", "both"))
SUPPORTED_PATHS: Final = frozenset(("2d", "3d"))
SUPPORTED_BACKENDS: Final = frozenset(("auto", "tle", "triton"))


def num_kv_heads_for_mode(
    mode: DiffKVMode, layout: DiffKVLayout = DEFAULT_LAYOUT
) -> int:
    """Return the model KV-head count without a device-side branch."""
    if mode == "full":
        return layout.full_num_kv_heads
    if mode == "swa":
        return layout.swa_num_kv_heads
    raise ValueError(f"mode must be 'full' or 'swa', got {mode!r}")


DEFAULT_BLOCK_SIZE: Final = DEFAULT_LAYOUT.block_size
OPTIMIZED_HEAD_SIZE_QK: Final = DEFAULT_LAYOUT.head_size_qk
OPTIMIZED_HEAD_SIZE_V: Final = DEFAULT_LAYOUT.head_size_v
QK_TILE_LO: Final = 128
QK_TILE_HI: Final = 64


@dataclass(frozen=True)
class LaunchDefaults:
    """Small stable defaults shared by the launch heuristics and kernels.

    Shape-specific values are derived properties rather than independent
    configuration knobs.  This keeps the launch policy from accumulating a
    separate constant for every historical benchmark shape.
    """

    warps: int = 4
    stages: int = 3
    query_block_m: int = 16
    default_tile: int = 32
    fused_segments: int = 32

    @property
    def split_head_block_m_2d(self) -> int:
        return max(self.query_block_m // 2, 1)

    @property
    def split_head_block_m_3d(self) -> int:
        return max(self.query_block_m // 4, 1)

    @property
    def split_head_block_m_fused(self) -> int:
        return max(self.query_block_m // 8, 1)

    @property
    def short_2d_wide_tile(self) -> int:
        return self.default_tile * 4

    @property
    def short_2d_batch_tile(self) -> int:
        return self.default_tile * 2

    @property
    def short_3d_tile(self) -> int:
        return max(self.default_tile // 2, 1)

    @property
    def wide_3d_tile(self) -> int:
        return self.default_tile * 2

    @property
    def short_2d_stages(self) -> int:
        return self.stages + 2

    @property
    def compact_fused_segments(self) -> int:
        return max(self.fused_segments // 2, 1)

    @property
    def fused_warps(self) -> int:
        return max(self.warps // 2, 1)

    @property
    def fused_stages(self) -> int:
        return max(self.stages - 1, 1)


@dataclass(frozen=True)
class WorkloadPolicy:
    """KV-length boundaries used to classify launch workloads."""

    short_k: int = 1024
    medium_k: int = 8192
    short_2d_max_k: int = 1536
    short_2d_wide_tile_k: int = 512


@dataclass(frozen=True)
class ResourcePolicy:
    """Small geometry targets shared by launch heuristics."""

    target_cta_waves_per_sm: int = 2
    dedup_programs_per_sm_limit: int = 8


LAUNCH = LaunchDefaults()
WORKLOAD = WorkloadPolicy()
RESOURCE = ResourcePolicy()


@lru_cache(maxsize=1)
def _implementation():
    """Load the kernel module lazily to keep API/config imports acyclic."""
    from . import diffkv

    return diffkv


_IMPLEMENTATION_EXPORTS = frozenset(
    {
        "HAS_TLE",
        "USE_TLE",
        "REQUESTED_BACKEND",
        "SELECTED_BACKEND",
        "get_diffkv_backend_info",
        "get_num_par_softmax_segments",
        "is_tle_available",
        "tle_import_error",
        "should_use_tle_fused_reducer",
    }
)


def __getattr__(name: str):
    """Lazily expose implementation diagnostics without an import cycle."""
    if name in _IMPLEMENTATION_EXPORTS:
        return getattr(_implementation(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_backend(backend: DiffKVBackend | str | None) -> DiffKVBackend:
    """Resolve a per-call backend override before launching a kernel."""
    selected = (
        _implementation()._default_backend()
        if backend is None
        else backend.strip().lower()
    )
    if selected == "auto":
        selected = _implementation()._default_backend()
    if selected not in {"tle", "triton"}:
        raise ValueError(
            "backend must be one of auto, tle, or triton; "
            f"got {backend!r}"
        )
    if selected == "tle" and not _implementation().HAS_TLE:
        raise RuntimeError(
            "TLE backend requested but "
            "triton.experimental.tle.language is unavailable: "
            f"{_implementation().tle_import_error()}"
        )
    return cast(DiffKVBackend, selected)


def _validate_public_inputs(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    window_size: int,
) -> tuple[int, int, int, int]:
    """Validate the lightweight public paged-cache contract.

    Like the reference FlagAttention operators, this checks only structural
    constraints needed to select a kernel.  The caller owns the paged-cache
    contract; scanning ``context_lens`` or ``block_tables`` here would add a
    GPU-to-host synchronization to every public call.
    """
    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError(
            "expected query [B,Hq,Dqk] and paged key/value "
            "[num_blocks,block_size,Hkv,D]"
        )
    if context_lens.ndim != 1 or block_tables.ndim != 2:
        raise ValueError(
            "context_lens must be [B] and block_tables must be [B,max_blocks]"
        )
    tensors = (query, key_cache, value_cache, context_lens, block_tables)
    if query.device.type != "cuda":
        raise ValueError("DiffKV requires all inputs on CUDA")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all DiffKV inputs must be on the same CUDA device")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("DiffKV supports float16 and bfloat16 query tensors")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise TypeError("query, key_cache, and value_cache must have the same dtype")

    batch, num_query_heads, head_size_qk = query.shape
    num_blocks, block_size, num_kv_heads, key_head_size = key_cache.shape
    if (
        value_cache.shape[:3] != key_cache.shape[:3]
        or key_head_size != head_size_qk
        or num_query_heads % num_kv_heads
        or context_lens.numel() != batch
        or block_tables.shape[0] != batch
    ):
        raise ValueError("invalid DiffKV cache or batch layout")
    if window_size == 0 or window_size < -1:
        raise ValueError("window_size must be -1 or a positive token count")
    return batch, num_query_heads, head_size_qk, value_cache.shape[3]


def unified_attention_diffkv(
    *args: Any,
    backend: DiffKVBackend | str | None = None,
    path: DiffKVPath | str = "2d",
    **kwargs: Any,
):
    """Run DiffKV through the selected TLE or standard Triton path."""
    selected = _resolve_backend(backend)
    # Keep the legacy vLLM threshold argument accepted by existing callers;
    # path selection is now centralized in the launchers.
    kwargs.pop("seq_threshold_3D", None)
    if selected == "tle":
        return _implementation()._unified_attention_diffkv_tle(
            *args, path=path, **kwargs
        )
    # The synchronous Triton launcher has no fused-reducer argument.
    kwargs.pop("fused_reducer_counter", None)
    return _implementation()._unified_attention_diffkv_fallback(
        *args, path=path, **kwargs
    )


def unified_attention_diffkv_tle(*args: Any, **kwargs: Any):
    """Explicit TLE implementation entry point."""
    return unified_attention_diffkv(*args, backend="tle", **kwargs)


def unified_attention_diffkv_fallback(*args: Any, **kwargs: Any):
    """Explicit standard non-TLE Triton entry point."""
    return unified_attention_diffkv(*args, backend="triton", **kwargs)


@torch.no_grad()
def diffkv_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    attn_scale: float | None = None,
    window_size: int = -1,
    path: DiffKVPath = "2d",
    num_segments: int | None = None,
    backend: DiffKVBackend | None = None,
) -> torch.Tensor:
    """Run paged DiffKV attention for decode workloads.

    Args:
        query: Query tensor with shape ``[B, Hq, Dqk]``.
        key_cache: Paged key cache ``[NB, BS, Hkv, Dqk]``.
        value_cache: Paged value cache ``[NB, BS, Hkv, Dv]``.
        context_lens: Cached token count for each sequence, shape ``[B]``.
        block_tables: Physical KV block indices, shape ``[B, max_blocks]``.
        attn_scale: Optional softmax scale; defaults to ``Dqk ** -0.5``.
        window_size: Number of recent KV tokens, or ``-1`` for full context.
        path: Launch path, either ``"2d"`` or ``"3d"``.
        num_segments: Optional split-KV segment count for the 3D path.
        backend: Optional backend override: ``"auto"``, ``"tle"`` or
            ``"triton"``.

    Returns:
        Attention output with shape ``[B, Hq, Dv]``.
    """
    selected = _resolve_backend(backend)
    batch, num_query_heads, head_size_qk, head_size_v = _validate_public_inputs(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        window_size,
    )
    num_kv_heads = key_cache.shape[2]
    if attn_scale is None:
        attn_scale = head_size_qk**-0.5
    context_lens = context_lens.to(device=query.device, dtype=torch.int32)
    block_tables = block_tables.to(device=query.device, dtype=torch.int32)
    out = torch.empty(
        (batch, num_query_heads, head_size_v),
        device=query.device,
        dtype=query.dtype,
    )
    cu_seqlens_q = torch.arange(
        batch + 1, device=query.device, dtype=torch.int32
    )
    max_seqlen_k = int(context_lens.max().item())
    path = _implementation()._normalize_path(path)
    use_3d = path == "3d"
    if use_3d:
        if num_segments is None:
            num_segments = _implementation().get_num_par_softmax_segments(
                max_seqlen_k,
                batch,
                True,
                total_num_q_blocks=2 * batch,
                num_kv_heads=num_kv_heads,
                num_sms=_implementation()._device_num_sms(query.device),
                block_size=key_cache.shape[1],
            )
        padded_v = triton.next_power_of_2(head_size_v)
        segm_output = torch.empty(
            (batch, num_query_heads, num_segments, padded_v),
            device=query.device,
            dtype=query.dtype if selected == "tle" else torch.float32,
        )
        segm_max = torch.empty(
            (batch, num_query_heads, num_segments),
            device=query.device,
            dtype=torch.float32,
        )
        segm_expsum = torch.empty_like(segm_max)
        seq_threshold = batch
    else:
        num_segments = None
        segm_output = segm_max = segm_expsum = None
        seq_threshold = None
    fused_reducer_counter = None
    if selected == "tle" and use_3d and _implementation().should_use_tle_fused_reducer(
        head_size_qk,
        head_size_v,
        1,
        max_seqlen_k,
        batch,
        key_cache.shape[1],
        use_3d,
    ):
        fused_reducer_counter = torch.zeros(
            batch * num_query_heads,
            device=query.device,
            dtype=torch.int32,
        )
    triton_window = (
        (window_size - 1, 0) if window_size > 0 else (-1, -1)
    )
    unified_attention_diffkv(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=context_lens,
        softmax_scale=float(attn_scale),
        causal=True,
        window_size=triton_window,
        block_table=block_tables,
        softcap=0.0,
        max_seqlen_q=1,
        seq_threshold_3D=seq_threshold,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        max_seqlen_k=max_seqlen_k,
        fused_reducer_counter=fused_reducer_counter,
        path=path,
        backend=selected,
    )
    return out


__all__ = [
    "OP_NAME",
    "diffkv_attention",
    "unified_attention_diffkv",
    "unified_attention_diffkv_tle",
    "unified_attention_diffkv_fallback",
    "DiffKVBackend",
    "DiffKVPath",
    "_resolve_backend",
]
