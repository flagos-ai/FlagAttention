# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Packed Ascend prefill/decode with an exact all-block dense fallback."""

import torch
import triton

from flag_attn.infllmv2.operator import InfLLMV2Config, _validate_packed_inputs
from .forward import sparse_attention
from . import forward
from .compression import compress_pair
from .fused_stage1_pool import stage1_pool


def _can_use_paged_inference(q, k, v, max_q, config):
    if not (
        config.causal
        and config.block_size == 64
        and min(q.shape[0], max_q) >= 128
        and k.shape[1] <= 2
        and q.shape[1] // k.shape[1] <= 16
        and q.shape[-1] in (64, 128)
        and q.dtype in (torch.float16, torch.bfloat16)
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)))
    ):
        return False

    import torch_npu

    return hasattr(torch_npu, "npu_fused_infer_attention_score") and "910B" in torch.npu.get_device_name(q.device)


def _sparse_pipeline(q, k, v, cu_q, cu_k, max_q, max_k, config, scale):
    with torch.no_grad():
        geometry = (config.k1_kernel_size, config.k1_stride, config.k2_kernel_size, config.k2_stride)
        fast = geometry == (32, 16, 128, 64)
        if fast:
            k1, cu1, k2, cu2 = compress_pair(k, cu_k, max_k, single_token=max_q == 1)
        else:
            k1, cu1 = forward.compress_k(k, cu_k, config.k1_kernel_size, config.k1_stride)
            k2, cu2 = forward.compress_k(k, cu_k, config.k2_kernel_size, config.k2_stride)
        if (
            fast
            and config.block_size == 64
            and q.shape[1] // k.shape[1] <= 16
            and q.shape[-1] in (64, 128)
            and q.dtype in (torch.bfloat16, torch.float16)
        ):
            pooled = stage1_pool(
                q,
                k1,
                k2,
                cu_q,
                cu1,
                cu2,
                cu_k,
                max_q,
                max_k,
                scale,
                config.causal,
                config.init_blocks,
                config.local_blocks,
            )
        else:
            max_k1 = max(0, (max_k - config.k1_kernel_size) // config.k1_stride + 1)
            score = forward.stage1(
                q,
                k1,
                k2,
                cu_q,
                cu1,
                cu2,
                max_q,
                config.k1_stride,
                config.k2_stride,
                scale,
                config.causal,
                cu_k,
                _max_k1=max_k1,
            )
            pooled = forward.pool_scores(
                score,
                cu_q,
                cu1,
                max_q,
                config.k1_kernel_size,
                config.k1_stride,
                config.block_size,
                config.init_blocks,
                config.local_blocks,
                cu_k,
                max_k,
                _valid_only=True,
            )
        selected = forward.select_blocks(pooled, cu_q, config.topk, config.block_size, cu_k, _max_q=max_q)
    if config.topk == 64 and max_k >= 4096 and _can_use_paged_inference(q, k, v, max_q, config):
        from .paged import sparse_paged_inference

        return sparse_paged_inference(q, k, v, selected, cu_q, cu_k, max_q, max_k, scale)
    return sparse_attention(
        q, k, v, selected, cu_q, cu_k, max_q, config.block_size, scale, config.causal,
        selection_is_compact=True,
    )


def infllmv2_attention(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *, config=None, softmax_scale=None
):
    config = config or InfLLMV2Config()
    config.validate()
    _validate_packed_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
    if not q.numel() or not k.numel():
        return forward.empty_attention(q, k, v)
    if max_seqlen_k < config.dense_len:
        blocks = triton.cdiv(max_seqlen_k, config.block_size)
        paged = blocks <= 64 and _can_use_paged_inference(q, k, v, max_seqlen_q, config)
        width = max(16, triton.next_power_of_2(blocks)) if paged else blocks
        selected = torch.arange(width, dtype=torch.int32, device=q.device)
        if paged:
            selected = torch.where(selected < blocks, selected, -1)
        selected = selected.view(1, 1, width).expand(k.shape[1], q.shape[0], width).contiguous()
        if paged:
            from .paged import sparse_paged_inference

            return sparse_paged_inference(
                q, k, v, selected, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, dense=True
            )
        return sparse_attention(
            q, k, v, selected, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, config.block_size, softmax_scale, config.causal, dense=True
        )
    return _sparse_pipeline(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, config, softmax_scale)


def infllmv2_decode(q, k_cache, v_cache, cu_seqlens_k, max_seqlen_k, *, config=None, softmax_scale=None):
    if q.ndim != 3 or q.shape[0] != cu_seqlens_k.numel() - 1:
        raise ValueError("decode q must have [batch, Hq, D] layout")
    cu_q = torch.arange(cu_seqlens_k.numel(), device=q.device, dtype=torch.int32)
    return infllmv2_attention(
        q, k_cache, v_cache, cu_q, cu_seqlens_k, 1, max_seqlen_k, config=config, softmax_scale=softmax_scale
    )
