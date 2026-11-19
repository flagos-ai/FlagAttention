# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Ascend launchers for packed InfLLM-V2."""

import math
from functools import lru_cache
from pathlib import Path
import torch
import triton
from triton.compiler import errors as compiler_errors
from flag_attn.infllmv2 import forward as kernels
from .kernels import stage1_kernel, pool_scores_kernel, attention_kernel, select_blocks_persistent_kernel

_RAW_COMPILE_ERRORS = (
    compiler_errors.CompilationError,
    getattr(compiler_errors, "MLIRCompilationError", compiler_errors.CompilationError),
)


def _check(*tensors):
    for tensor in tensors:
        if tensor.device.type != "npu":
            raise ValueError("Ascend InfLLM-V2 requires NPU tensors")
        if not tensor.is_contiguous():
            raise ValueError("InfLLM-V2 Triton kernels require contiguous tensors")


def _query_grid(max_q, heads, batch, vector=False):
    utility = triton.runtime.driver.active.utils
    cores = utility.get_aivector_core_num() if vector else utility.get_aicore_num()
    return max(1, min(max_q, cores // (heads * batch))), heads, batch


def compress_k(k, cu_seqlens_k, kernel_size, stride):
    _check(k, cu_seqlens_k)
    if cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_k must be int32")
    if k.numel() > 2**31 - 1:
        raise ValueError("Ascend InfLLM-V2 generic compression requires 32-bit element offsets")
    cu_out = kernels.compressed_lengths(cu_seqlens_k, kernel_size, stride)
    total = int(cu_out[-1].item())
    out = torch.empty((total, k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device)
    if total:
        count = int((cu_out[1:] - cu_out[:-1]).max().item())
        kernels._compress_k_kernel[count, k.shape[1], cu_seqlens_k.numel() - 1](
            k,
            out,
            cu_seqlens_k,
            cu_out,
            hkv=k.shape[1],
            d=k.shape[2],
            kernel_size=kernel_size,
            stride=stride,
            block_window=triton.next_power_of_2(kernel_size),
            block_d=triton.next_power_of_2(k.shape[2]),
            num_warps=4,
        )
    return out, cu_out


def stage1(
    q,
    k1,
    k2,
    cu_seqlens_q,
    cu_seqlens_k1,
    cu_seqlens_k2,
    max_seqlen_q,
    k1_stride=16,
    k2_stride=64,
    softmax_scale=None,
    causal=True,
    cu_seqlens_k=None,
    *,
    _allow_tle=True,
    _max_k1=None,
):
    cu_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    _check(q, k1, k2, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k2, cu_k)
    total_q, hq, d = q.shape
    hkv = k1.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    n1 = int((cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).max().item()) if _max_k1 is None else _max_k1
    out = torch.zeros((hkv, total_q, n1), dtype=q.dtype, device=q.device)
    if n1:
        grid = _query_grid(max_seqlen_q, hkv, cu_seqlens_q.numel() - 1)
        stage1_kernel[grid](
            q,
            k1,
            k2,
            out,
            cu_seqlens_q,
            cu_seqlens_k1,
            cu_seqlens_k2,
            cu_k,
            hq=hq,
            hkv=hkv,
            group=hq // hkv,
            d=d,
            total_q=total_q,
            max_k1=n1,
            scale=d ** (-0.5) if softmax_scale is None else softmax_scale,
            k1_stride=k1_stride,
            k2_stride=k2_stride,
            causal=causal,
            block_q=1,
            block_g=max(16, triton.next_power_of_2(hq // hkv)),
            block_d=triton.next_power_of_2(d),
            block_n=128,
            num_warps=4,
            num_stages=2,
            max_q=max_seqlen_q,
        )
    return out


def pool_scores(
    score,
    cu_seqlens_q,
    cu_seqlens_k1,
    max_seqlen_q,
    kernel_size=32,
    kernel_stride=16,
    block_size=64,
    init_blocks=1,
    local_blocks=32,
    cu_seqlens_k=None,
    max_seqlen_k=None,
    *,
    _valid_only=False,
):
    cu_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    max_k = max_seqlen_q if max_seqlen_k is None else max_seqlen_k
    _check(score, cu_seqlens_q, cu_seqlens_k1, cu_k)
    hkv, total_q, n1 = score.shape
    nb = math.ceil(max_k / block_size)
    out = torch.empty((hkv, total_q, nb), dtype=score.dtype, device=score.device)
    if nb:
        grid = _query_grid(max_seqlen_q, hkv, cu_seqlens_q.numel() - 1, vector=True)
        pool_scores_kernel[grid](
            score,
            out,
            cu_seqlens_q,
            cu_seqlens_k1,
            cu_k,
            total_q,
            n1,
            nb,
            hkv=hkv,
            block_size=block_size,
            block_stride=block_size // kernel_stride,
            pool_window=kernel_size // kernel_stride + block_size // kernel_stride - 1,
            pad=kernel_size // kernel_stride - 1,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            block_b=triton.next_power_of_2(nb),
            valid_only=_valid_only,
            num_warps=4,
            max_q=max_seqlen_q,
        )
    return out


@lru_cache(maxsize=1)
def _raw_selector():
    if not Path(__file__).with_name("infllmv2_top64_raw.cpp").is_file():
        return None
    try:
        from .raw_topk import select_blocks_raw_kernel
    except ImportError:
        return None
    return select_blocks_raw_kernel


def select_blocks(block_score, cu_seqlens_q, topk, block_size=64, cu_seqlens_k=None, *, _allow_tle=True, _max_q=None):
    cu_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    _check(block_score, cu_seqlens_q, cu_k)
    hkv, total_q, nb = block_score.shape
    if nb > 4096:
        raise ValueError("TopK currently supports at most 4096 KV blocks")
    if topk > 2048 or topk & topk - 1 != 0:
        raise ValueError("TopK must be a power of two no greater than 2048")
    out = torch.full((hkv, total_q, topk), -1, dtype=torch.int32, device=block_score.device)
    if nb:
        max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()) if _max_q is None else _max_q
        tile = max(64, triton.next_power_of_2(nb))
        raw = None
        if (
            _allow_tle
            and max_q >= 1024
            and nb >= 128
            and topk == 64
            and block_size == 64
            and block_score.dtype == torch.bfloat16
            and "910B" in torch.npu.get_device_name(block_score.device)
        ):
            raw = _raw_selector()
        grid = _query_grid(max_q, hkv, cu_seqlens_q.numel() - 1, vector=True)
        args = (block_score, out, cu_seqlens_q, cu_k, total_q, nb)
        options = dict(
            block_size=block_size,
            topk=topk,
            block_b=tile,
            num_warps=4,
            max_q=max_q,
        )
        if raw is not None:
            try:
                raw[grid](*args, **options, num_stages=1, multibuffer=False)
                return out
            except _RAW_COMPILE_ERRORS:
                pass
        select_blocks_persistent_kernel[grid](*args, **options, num_stages=2)
    return out


def sparse_attention_forward(
    q,
    k,
    v,
    selected_blocks,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    block_size=64,
    softmax_scale=None,
    causal=True,
    store_lse=True,
    *,
    dense=False,
    out=None,
    exception_flags=None,
    exception_pages=None,
    exception_max_blocks=0,
    selection_is_compact=False,
):
    _check(q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k)
    total_q, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    if selected_blocks.ndim != 3 or selected_blocks.shape[:2] != (hkv, total_q):
        raise ValueError("selected_blocks must have [Hkv, total_q_tokens, topk] layout")
    if not q.numel() or not k.numel():
        out = torch.zeros_like(q)
        lse = torch.full((total_q, hq), -torch.inf, dtype=torch.float32, device=q.device) if store_lse else out
        return out, lse
    if out is None:
        out = torch.empty_like(q)
    lse = torch.empty((total_q, hq), dtype=torch.float32, device=q.device) if store_lse else out
    # Batch selected ranks from the SAME query. Bound each K/V tile at 16K
    # elements so the enlarged Cube tile fits the local-memory budget.
    ranks_per_tile = 1
    # Short selections do not amortize the batched gather's wider vectors.
    if selected_blocks.shape[-1] >= 64 and hq // hkv <= 16:
        ranks_per_tile = max(1, min(4 if d <= 64 else 2, 16384 // (block_size * triton.next_power_of_2(d))))
    value_scale = triton.next_power_of_2(max(1, block_size * selected_blocks.shape[-1])) if v.dtype == torch.bfloat16 else 1
    passes = ((value_scale, False),)
    if value_scale > 1 and exception_flags is None:
        # Retry nonfinite BF16 output on device after the unscaled pass.
        passes = ((1, False), (value_scale, True))
    grid = _query_grid(max_seqlen_q, hkv, cu_seqlens_q.numel() - 1)
    for scale_v, replay in passes:
        attention_kernel[grid](
            q,
            k,
            v,
            selected_blocks,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            hq=hq,
            hkv=hkv,
            group=hq // hkv,
            d=d,
            total_q=total_q,
            scale=d ** (-0.5) if softmax_scale is None else softmax_scale,
            topk=selected_blocks.shape[-1],
            sparse_block=block_size,
            causal=causal,
            block_m=max(16, triton.next_power_of_2(hq // hkv)),
            block_n=block_size * ranks_per_tile,
            block_d=triton.next_power_of_2(d),
            store_lse=store_lse and not replay,
            ranks_per_tile=ranks_per_tile,
            dense=dense,
            selection_is_compact=selection_is_compact,
            wide_offsets=max(q.numel(), k.numel(), selected_blocks.numel()) > 2**31 - 1,
            exception_flags=exception_flags,
            flag_count=0 if exception_flags is None else exception_flags.numel(),
            exception_pages=exception_pages,
            exception_max_blocks=exception_max_blocks,
            value_scale=scale_v,
            replay=replay,
            num_warps=4,
            num_stages=2,
            max_q=max_seqlen_q,
        )
    return out, lse


def _attached_forward(ctx, q, k, v, selected, cu_q, cu_k, max_q, block_size, scale, causal, **options):
    out, lse = sparse_attention_forward(
        q, k, v, selected, cu_q, cu_k, max_q, block_size, scale, causal, **options
    )
    ctx.save_for_backward(q, k, v, selected, out, lse, cu_q, cu_k)
    ctx.max_seqlen_q = max_q
    ctx.block_size = block_size
    ctx.softmax_scale = scale
    ctx.causal = causal
    return out


class _DenseAttentionFunction(kernels._SparseAttentionFunction):
    @staticmethod
    def forward(ctx, q, k, v, selected, cu_q, cu_k, max_q, block_size, scale, causal):
        return _attached_forward(ctx, q, k, v, selected, cu_q, cu_k, max_q, block_size, scale, causal, dense=True)


class _CompactAttentionFunction(kernels._SparseAttentionFunction):
    @staticmethod
    def forward(ctx, q, k, v, selected, cu_q, cu_k, max_q, block_size, scale, causal):
        return _attached_forward(
            ctx, q, k, v, selected, cu_q, cu_k, max_q, block_size, scale, causal, selection_is_compact=True
        )


def sparse_attention(
    q,
    k,
    v,
    selected_blocks,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    block_size=64,
    softmax_scale=None,
    causal=True,
    *,
    _allow_tle=True,
    dense=False,
    selection_is_compact=False,
):
    if not q.numel() or not k.numel():
        return empty_attention(q, k, v)
    args = (q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, block_size, softmax_scale, causal)
    if torch.is_grad_enabled() and any((t.requires_grad for t in (q, k, v))):
        if max(q.numel(), k.numel(), selected_blocks.numel()) > 2**31 - 1:
            raise ValueError("Ascend InfLLM-V2 backward requires 32-bit element offsets")
        if dense:
            function = _DenseAttentionFunction
        elif selection_is_compact:
            function = _CompactAttentionFunction
        else:
            function = kernels._SparseAttentionFunction
        return function.apply(*args)
    return sparse_attention_forward(
        *args, store_lse=False, dense=dense, selection_is_compact=selection_is_compact
    )[0]


def empty_attention(q, k, v):
    out = torch.zeros_like(q)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        # Empty reductions preserve zero gradients without reading padded data.
        return out + q[:0].sum() + k[:0].sum() + v[:0].sum()
    return out
