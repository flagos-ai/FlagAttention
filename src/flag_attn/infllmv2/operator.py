# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""InfLLM-V2 operator configuration, public entry points, and dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .forward import compress_k, pool_scores, select_blocks, sparse_attention, stage1


@dataclass(frozen=True)
class InfLLMV2Config:
    """Algorithm parameters for continuous-packed causal InfLLM-V2 attention."""

    k1_kernel_size: int = 32
    k1_stride: int = 16
    k2_kernel_size: int = 128
    k2_stride: int = 64
    block_size: int = 64
    topk: int = 64
    init_blocks: int = 1
    local_blocks: int = 32
    dense_len: int = 8192
    causal: bool = True

    def validate(self) -> None:
        """Reject only modes that change this operator's causal semantics.

        Shape-specific requirements are enforced by the forward kernels that
        actually consume them, instead of duplicating a large policy matrix at
        the public boundary.
        """
        if not self.causal:
            raise NotImplementedError("InfLLM-V2 attention currently supports causal mode only")

def _validate_packed_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or k.shape != v.shape:
        raise ValueError("q, k and v must use packed [tokens, heads, head_dim] layout")
    if q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError("Q/K/V head dimensions must match and Hq must be divisible by Hkv")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Q/K/V must have the same dtype")
    if not (q.device == k.device == v.device == cu_seqlens_q.device == cu_seqlens_k.device):
        raise ValueError("all inputs must be on the same device")
    if (
        cu_seqlens_q.ndim != 1
        or cu_seqlens_k.ndim != 1
        or cu_seqlens_q.dtype != torch.int32
        or cu_seqlens_k.dtype != torch.int32
        or cu_seqlens_q.numel() != cu_seqlens_k.numel()
    ):
        raise ValueError("Q/KV cumulative lengths must be matching one-dimensional int32 tensors")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("maximum sequence lengths must be positive")


def _dense_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Reuse FlagAttention FlashAttention for each continuous sequence."""
    from flag_attn.flash import attention as flash_attention

    chunks: list[torch.Tensor] = []
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    k_cu = cu_seqlens_k.detach().cpu().tolist()
    for qs, qe, ks, ke in zip(q_cu[:-1], q_cu[1:], k_cu[:-1], k_cu[1:]):
        q_seq = q[qs:qe].transpose(0, 1).unsqueeze(0)
        k_seq = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_seq = v[ks:ke].transpose(0, 1).unsqueeze(0)
        out = flash_attention(q_seq, k_seq, v_seq, causal=True, sm_scale=softmax_scale)
        chunks.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(chunks, dim=0)


def _run_sparse_pipeline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    config: InfLLMV2Config,
    softmax_scale: float | None,
) -> torch.Tensor:
    # Top-k selection is discrete. Gradients flow through Stage-2 Q/K/V only.
    with torch.no_grad():
        k1, cu_k1 = compress_k(k, cu_seqlens_k, config.k1_kernel_size, config.k1_stride)
        k2, cu_k2 = compress_k(k, cu_seqlens_k, config.k2_kernel_size, config.k2_stride)
        token_score = stage1(
            q, k1, k2, cu_seqlens_q, cu_k1, cu_k2, max_seqlen_q,
            config.k1_stride, config.k2_stride, softmax_scale, config.causal, cu_seqlens_k,
        )
        block_score = pool_scores(
            token_score, cu_seqlens_q, cu_k1, max_seqlen_q,
            config.k1_kernel_size, config.k1_stride, config.block_size,
            config.init_blocks, config.local_blocks, cu_seqlens_k, max_seqlen_k,
            _valid_only=True,
        )
        selected = select_blocks(
            block_score, cu_seqlens_q, config.topk, config.block_size, cu_seqlens_k
        )
    return sparse_attention(
        q, k, v, selected, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        config.block_size, softmax_scale, config.causal,
    )


def _dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    config: InfLLMV2Config,
    softmax_scale: float | None,
) -> torch.Tensor:
    # Batch-wide decision from the LONGEST per-sequence KV, never sum(KV).
    # Match the model's strict boundary: KV < dense_len is dense; equality sparse.
    if max_seqlen_k < config.dense_len:
        return _dense_packed(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    return _run_sparse_pipeline(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        config, softmax_scale,
    )


def infllmv2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    config: InfLLMV2Config | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Continuous packed full, prefix-cache, or chunked causal prefill.

    Each Q sequence is the suffix of its KV sequence: query i has absolute
    position ``Lkv - Lq + i``. KV must include the cached prefix AND the current
    query chunk. Pass only the current Q chunk; previously computed outputs
    are not recomputed. The caller manages cache append/storage across calls.
    Unequal per-sample lengths and gradients through all supplied Q/K/V are
    supported. No paged cache or in-place cache updates are performed here.
    """
    config = config or InfLLMV2Config()
    config.validate()
    _validate_packed_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
    return _dispatch(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        config, softmax_scale,
    )


def infllmv2_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    *,
    config: InfLLMV2Config | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """One-token decode over a continuous packed KV cache.

    ``q`` is ``[batch, Hq, D]``. ``k_cache`` and ``v_cache`` are packed
    ``[total_cached_tokens, Hkv, D]`` and already include the current token.
    This API intentionally excludes paged KV and in-place cache updates.
    """
    config = config or InfLLMV2Config()
    config.validate()
    batch = cu_seqlens_k.numel() - 1
    if q.ndim != 3 or q.shape[0] != batch:
        raise ValueError("decode q must have [batch, Hq, D] layout")
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    _validate_packed_inputs(q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, 1, max_seqlen_k)
    q, k_cache, v_cache = q.contiguous(), k_cache.contiguous(), v_cache.contiguous()
    cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
    return _dispatch(
        q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, 1, max_seqlen_k,
        config, softmax_scale,
    )
