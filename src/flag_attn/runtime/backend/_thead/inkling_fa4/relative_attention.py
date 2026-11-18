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

"""T-Head paged relative attention with Triton and stable Split-KV merging.

Queries and relative logits are packed by sequence. K/V caches have layout
[num_pages, page_size, num_kv_heads, head_dim]. This inference-only operator
supports GQA, causal/local attention, and caller-provided output buffers.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import triton
from triton import knobs
from triton.knobs import HookChain
from triton.runtime.jit import JITFunction

from flag_attn.runtime.backend._thead.inkling_fa4.kernels import _attention, _combine
from flag_attn.runtime.backend._thead.inkling_fa4.launch_configs import get_config

# Bound the process-local compiled-launch cache. Entries contain no tensors,
# tensor addresses, outputs or GPU length contents.
_LAUNCH_CACHE = {}
_LAUNCH_CACHE_LIMIT = 128


def _launch(kernel, grid, *args, **kwargs):
    # Preserve JIT hook/instrumentation behavior and inspection wrappers.
    if (not isinstance(kernel, JITFunction) or kernel.pre_run_hooks
            or knobs.runtime.debug or knobs.compilation.instrumentation_mode
            or (knobs.runtime.launch_enter_hook is not None and
                (not isinstance(knobs.runtime.launch_enter_hook, HookChain) or knobs.runtime.launch_enter_hook.calls))
            or (knobs.runtime.launch_exit_hook is not None and
                (not isinstance(knobs.runtime.launch_exit_hook, HookChain) or knobs.runtime.launch_exit_hook.calls))):
        return kernel[grid](*args, **kwargs)
    tensor_count = 10 if kernel is _attention else 3
    tensor_key = tuple((x.dtype, x.data_ptr() % 16) for x in args[:tensor_count])
    scalar_key = tuple((type(x), x) for x in args[tensor_count:])
    const_key = tuple((name, type(value), value) for name, value in kwargs.items())
    key = (kernel, args[0].device, grid, tensor_key, scalar_key, const_key)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        compiled = kernel[grid](*args, **kwargs)
        tail = tuple(kwargs[name] for name in kernel.arg_names[len(args):])
        entry = (compiled[grid], tail)
        if len(_LAUNCH_CACHE) >= _LAUNCH_CACHE_LIMIT:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[key] = entry
        return compiled
    runner, tail = entry
    runner(*args, *tail)


def _window(
    causal: bool, window: tuple[int, int]
) -> tuple[bool, bool, int | None, int | None]:
    """Canonicalize windows using the upstream FA4 boundary convention."""
    left, right = (None, None) if window == (-1, -1) else window
    if causal:
        right = 0
    if left is not None and right is not None and left + right < 0:
        left, right = None, None
    if left is None and right == 0:
        return True, False, None, None
    if left is not None or right is not None:
        return False, True, left, right
    return causal, False, None, None


def bucket_max_seqlen_q(length: int) -> int:
    return 1 << max(0, length - 1).bit_length()


def inkling_fa4_num_splits(
    *,
    is_local: bool,
    batch_size: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    """Upstream non-Hopper split cap, exposed for explicit experiments only.

    The public kernel respects its explicit num_splits, as upstream .kernel
    does. It does not apply this serving dispatch policy behind the caller.
    """
    if is_local or batch_size == 0 or max_query_len == 0:
        return 1
    q_tiles = (max_query_len * (num_heads // num_kv_heads) + 255) // 256
    base_ctas = batch_size * num_kv_heads * q_tiles
    target = 256 if q_tiles == 1 and batch_size == 1 else (128 if q_tiles == 1 else 64)
    cap = 128
    if q_tiles == 1 and batch_size == 1:
        if num_kv_heads == 8:
            cap = 16
        elif num_kv_heads == 4 or max_kv_len <= 8192:
            cap = 32
        elif max_kv_len <= 65536:
            cap = 64
    return max(1, min(target // base_ctas, cap, (max_kv_len + 127) // 128))


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bt: torch.Tensor,
    kl: torch.Tensor,
    cu: torch.Tensor,
    maxq: int,
    window: tuple[int, int],
    extent: int,
    rel: torch.Tensor,
    splits: int,
    out: torch.Tensor | None,
) -> None:
    """Validate tensor metadata without synchronizing GPU length contents."""
    device = q.device
    for tensor in (q, k, v, bt, kl, cu, rel):
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError("all inputs must be on the same PPU device")
    if q.ndim != 3 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected packed Q and paged K/V tensors")
    qshape, kshape, vshape = q.shape, k.shape, v.shape
    if kshape[:3] != vshape[:3] or qshape[2] != kshape[3]:
        raise ValueError("incompatible Q/K/V shapes")
    if not qshape[1] or not kshape[2] or qshape[1] % kshape[2]:
        raise ValueError("Q heads must be divisible by KV heads")
    dtype = q.dtype
    if dtype not in (torch.float16, torch.bfloat16) or k.dtype != dtype or v.dtype != dtype or rel.dtype != dtype:
        raise TypeError("Q/K/V/relative logits must share FP16 or BF16 dtype")
    if not (8 <= qshape[2] <= 512 and 8 <= vshape[3] <= 512) or qshape[2] % 8 or vshape[3] % 8:
        raise ValueError("head dimensions must be aligned to 8, in [8,512]")
    if kshape[1] <= 0 or extent <= 0:
        raise ValueError("page size and relative extent must be positive")
    if bt.ndim != 2 or kl.ndim != 1 or cu.ndim != 1:
        raise ValueError("invalid sequence metadata ranks")
    batch_size = kl.numel()
    if bt.shape[0] != batch_size or cu.numel() != batch_size + 1:
        raise ValueError("invalid sequence metadata shapes")
    if bt.dtype != torch.int32 or kl.dtype != torch.int32 or cu.dtype != torch.int32:
        raise TypeError("sequence metadata must use int32")
    if kl.stride(0) != 1 or cu.stride(0) != 1:
        raise ValueError("length metadata must be contiguous")
    if rel.shape != (qshape[0], qshape[1], extent):
        raise ValueError("relative logits have an incompatible shape")
    if not isinstance(window, tuple) or len(window) != 2 or not (
        isinstance(window[0], int) and window[0] >= -1
        and isinstance(window[1], int) and window[1] >= -1
    ):
        raise ValueError("window_size must be a pair of integers >= -1")
    if splits <= 0:
        raise ValueError("num_splits must be positive")
    if maxq < 0:
        raise ValueError("max_seqlen_q must be non-negative")
    if out is not None and (
        out.shape != (qshape[0], qshape[1], vshape[3])
        or out.dtype != dtype
        or out.device != device
    ):
        raise ValueError("incompatible output buffer")


@torch.no_grad()
def inkling_fa4_rel_attention_triton(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute packed paged attention, optionally using explicit KV splits.

    Relative logits are added after QK scaling. The output shape is
    [total_q, num_heads, value_head_dim], with the same dtype as q.
    """
    _validate(q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
              max_seqlen_q, window_size, rel_extent, rel_logits, num_splits, out)
    ca, local, left, right = _window(causal, window_size)
    rel_logits = rel_logits.contiguous()
    h, dq = q.shape[1:]
    hk = key_cache.shape[2]
    dv = value_cache.shape[3]
    if out is None:
        out = torch.empty((q.shape[0], h, dv), dtype=q.dtype, device=q.device)
    if cache_seqlens.shape[0] == 0 or q.shape[0] == 0:
        return out.zero_()
    if max_seqlen_q == 0:
        raise ValueError("max_seqlen_q must be positive for non-empty q")
    cfg = get_config(dq, dv, ca, local)
    if num_splits > 1:
        partial = torch.empty((num_splits, q.shape[0], h, dv), dtype=torch.float32, device=q.device)
        lse = torch.empty((num_splits, q.shape[0], h), dtype=torch.float32, device=q.device)
    else:
        partial, lse = out, out
    grid = (triton.cdiv(max_seqlen_q * (h // hk), cfg["BLOCK_M"]),
            cache_seqlens.numel() * hk, num_splits)
    _launch(_attention, grid,
        q, key_cache, value_cache, rel_logits, out, partial, lse,
        block_table, cache_seqlens, cu_seqlens_q,
        *q.stride(), *key_cache.stride(), *value_cache.stride(),
        *rel_logits.stride(), *out.stride(), *block_table.stride(),
        softmax_scale, TOTAL_Q=q.shape[0], H=h, HK=hk, DQ=dq, DV=dv,
        # PPU dot requires K >= 16; masks zero-fill padded channels.
        BDQ=max(16, triton.next_power_of_2(dq)), BDV=max(16, triton.next_power_of_2(dv)),
        PAGE=key_cache.shape[1], EXTENT=rel_extent, G=h // hk,
        BM=cfg["BLOCK_M"], BN=cfg["BLOCK_N"], SPLITS=num_splits, STAGES=cfg["num_stages"],
        CAUSAL=ca, LOCAL=local, HAS_LEFT=left is not None, HAS_RIGHT=right is not None,
        LEFT=left or 0, RIGHT=right or 0,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    if num_splits > 1:
        _launch(_combine, (q.shape[0] * h, 1, 1),
            partial, lse, out, q.shape[0], h, dv, *out.stride(),
            num_splits, triton.next_power_of_2(num_splits), triton.next_power_of_2(dv),
            num_warps=4,
        )
    return out


def get_backend(name: str | None = None) -> Callable[..., torch.Tensor]:
    """Resolve the supported auto/Triton aliases without launching a kernel."""
    selected = (name or os.getenv("INKLING_FA4_BACKEND", "triton")).strip().lower()
    if selected in ("auto", "triton", "base"):
        return inkling_fa4_rel_attention_triton
    raise ValueError(f"unknown T-Head Inkling FA4 backend: {selected}")


def inkling_fa4_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int = 1,
    out: torch.Tensor | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Run the T-Head relative-attention backend through the public API."""
    return get_backend(backend)(
        q, key_cache, value_cache, block_table=block_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, softmax_scale=softmax_scale,
        causal=causal, window_size=window_size, rel_extent=rel_extent,
        rel_logits=rel_logits, num_splits=num_splits, out=out,
    )
