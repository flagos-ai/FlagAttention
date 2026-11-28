# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""NVIDIA policy for InfLLM-V2; unsupported shapes use standard Triton."""

from functools import lru_cache
import os

import torch

from flag_attn.infllmv2 import forward as generic


def _force_generic():
    return os.getenv("FLAG_ATTN_FORCE_GENERIC", "0").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _hopper_kernels():
    # Stock Triton and older TLE installations retain the complete fallback.
    try:
        import triton.experimental.tle.language as tle
        from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401
    except (ImportError, AttributeError):
        return None
    gpu = getattr(tle, "gpu", None)
    if not all(hasattr(gpu, name) for name in (
        "alloc", "local_ptr", "alloc_barriers", "barrier_wait", "barrier_arrive",
        "copy", "wgmma", "wgmma_wait", "smem", "READY",
    )):
        return None
    from .hopper.infllmv2 import forward

    return forward


def _use_hopper(tensor):
    return (
        not _force_generic()
        and tensor.is_cuda
        and torch.cuda.get_device_capability(tensor.device) == (9, 0)
        and _hopper_kernels() is not None
    )


def stage1(
    q, k1, k2, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k2, max_seqlen_q,
    k1_stride=16, k2_stride=64, softmax_scale=None, causal=True, cu_seqlens_k=None,
    *, _allow_tle=True,
):
    implementation = generic.stage1
    # H100 only amortizes the double-buffered TMA/WGMMA setup at long prefill.
    if (
        _allow_tle and max_seqlen_q >= 65536
        and q.dtype == k1.dtype == k2.dtype == torch.bfloat16
        and q.shape[1] == 16 * k1.shape[1] and q.shape[2] in (64, 128)
        # Empty compressed K2 cannot be described by a CUDA TensorMap.
        and k1.shape[0] > 0 and k2.shape[0] > 0 and _use_hopper(q)
    ):
        implementation = _hopper_kernels().stage1_tle_hopper
    return implementation(
        q, k1, k2, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k2, max_seqlen_q,
        k1_stride, k2_stride, softmax_scale, causal, cu_seqlens_k,
    )


def select_blocks(
    block_score, cu_seqlens_q, topk, block_size=64, cu_seqlens_k=None,
    *, _allow_tle=True,
):
    if topk <= 0 or topk > 2048 or topk & (topk - 1):
        raise ValueError("TopK must be a power of two no greater than 2048")
    implementation = generic.select_blocks
    if (
        _allow_tle and block_score.shape[1] >= 1024
        and 256 < block_score.shape[-1] <= 4096 and topk <= 64
        and block_score.dtype == torch.bfloat16 and _use_hopper(block_score)
        and int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()) >= 1024
    ):
        implementation = _hopper_kernels().select_blocks_tle_radix
    return implementation(block_score, cu_seqlens_q, topk, block_size, cu_seqlens_k)


def sparse_attention(
    q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    block_size=64, softmax_scale=None, causal=True, *, _allow_tle=True,
):
    implementation = generic.sparse_attention
    if (
        _allow_tle and (not torch.is_grad_enabled() or not any(t.requires_grad for t in (q, k, v)))
        and max_seqlen_q >= 1024 and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape[1] == 16 * k.shape[1] and q.shape[2] == 128 and block_size == 64
        and _use_hopper(q)
    ):
        implementation = _hopper_kernels().sparse_attention_tle_hopper
    return implementation(
        q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        block_size, softmax_scale, causal,
    )


def _dense_packed(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale):
    if _force_generic():
        from flag_attn.infllmv2.operator import _dense_packed as portable

        return portable(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    from flag_attn.flash import attention as flash_attention

    chunks = []
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    k_cu = cu_seqlens_k.detach().cpu().tolist()
    for qs, qe, ks, ke in zip(q_cu[:-1], q_cu[1:], k_cu[:-1], k_cu[1:]):
        q_seq = q[qs:qe].transpose(0, 1).unsqueeze(0)
        k_seq = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_seq = v[ks:ke].transpose(0, 1).unsqueeze(0)
        out = flash_attention(q_seq, k_seq, v_seq, causal=True, sm_scale=softmax_scale)
        chunks.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(chunks, dim=0)
