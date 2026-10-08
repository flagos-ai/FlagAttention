# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Readable PyTorch references for the InfLLM-V2 algorithm.

These functions favor clarity over speed. They define the tensor layouts and
edge semantics used by the Triton implementation and are intended for tests,
validation, and algorithm study. The production public APIs do not dispatch to
this module.
"""

from __future__ import annotations

import math

import torch

from .forward import compressed_lengths


def compress_k_ref(
    k: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kernel_size: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool packed K windows; output remains ``[total, Hkv, D]``."""
    cu_out = compressed_lengths(cu_seqlens_k, kernel_size, stride)
    chunks: list[torch.Tensor] = []
    cu_cpu = cu_seqlens_k.detach().cpu().tolist()
    for start, end in zip(cu_cpu[:-1], cu_cpu[1:]):
        length = end - start
        for offset in range(0, max(0, length - kernel_size + 1), stride):
            chunks.append(k[start + offset : start + offset + kernel_size].float().mean(dim=0))
    if not chunks:
        return k.new_empty((0, k.shape[1], k.shape[2])), cu_out
    return torch.stack(chunks).to(k.dtype), cu_out


def stage1_ref(
    q: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k1: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    k1_stride: int = 16,
    k2_stride: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-pass Stage-1 semantics, returning ``[Hkv, Tq, max_K1]``."""
    total_q, hq, d = q.shape
    hkv = k1.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    max_k1 = int(torch.max(cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).item())
    out = torch.zeros((hkv, total_q, max_k1), dtype=q.dtype, device=q.device)
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    original_k_cu = (
        cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    ).detach().cpu().tolist()
    k1_cu = cu_seqlens_k1.detach().cpu().tolist()
    k2_cu = cu_seqlens_k2.detach().cpu().tolist()

    for qs, qe, ks, ke, a1, z1, a2, z2 in zip(
        q_cu[:-1],
        q_cu[1:],
        original_k_cu[:-1],
        original_k_cu[1:],
        k1_cu[:-1],
        k1_cu[1:],
        k2_cu[:-1],
        k2_cu[1:],
    ):
        q_seq = q[qs:qe].float().view(qe - qs, hkv, group, d)
        for hk in range(hkv):
            qh = q_seq[:, hk]
            coarse = torch.einsum("qgd,kd->qgk", qh, k2[a2:z2, hk].float()) * scale
            fine = torch.einsum("qgd,kd->qgk", qh, k1[a1:z1, hk].float()) * scale
            if causal:
                qpos = torch.arange(qe - qs, device=q.device)[:, None]
                cpos = torch.arange(z2 - a2, device=q.device)[None, :]
                fpos = torch.arange(z1 - a1, device=q.device)[None, :]
                coarse_stride = k1_stride if z1 - a1 == z2 - a2 else k2_stride
                coarse_q_len = max(0, qe - qs - coarse_stride + 1) // coarse_stride
                fine_q_len = max(0, qe - qs - k1_stride + 1) // k1_stride
                coarse_valid = cpos < torch.clamp(
                    (qpos + 1) // coarse_stride - 1 + z2 - a2 - coarse_q_len,
                    min=0,
                )
                fine_valid = fpos < torch.clamp(
                    (qpos + 1) // k1_stride - 1 + z1 - a1 - fine_q_len,
                    min=0,
                )
                coarse = coarse.masked_fill(~coarse_valid[:, None, :], -torch.inf)
                fine = fine.masked_fill(~fine_valid[:, None, :], -torch.inf)
            lse = torch.logsumexp(coarse, dim=-1)
            probs = torch.exp(fine - lse[..., None])
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            out[hk, qs:qe, : z1 - a1] = probs.sum(dim=1).to(q.dtype).float()
    return out


def pool_scores_ref(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k1: torch.Tensor,
    max_seqlen_q: int,
    kernel_size: int = 32,
    kernel_stride: int = 16,
    block_size: int = 64,
    init_blocks: int = 1,
    local_blocks: int = 32,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Map K1 scores to block scores using the operator pooling rules."""
    hkv, total_q, _ = score.shape
    max_seqlen_k = max_seqlen_q if max_seqlen_k is None else max_seqlen_k
    max_blocks = math.ceil(max_seqlen_k / block_size)
    result = torch.full(
        (hkv, total_q, max_blocks),
        -torch.inf,
        device=score.device,
        dtype=score.dtype,
    )
    window = kernel_size // kernel_stride + block_size // kernel_stride - 1
    pad = kernel_size // kernel_stride - 1
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    original_k_cu = (
        cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    ).detach().cpu().tolist()
    k1_cu = cu_seqlens_k1.detach().cpu().tolist()
    for qs, qe, ks, ke, k1s, k1e in zip(
        q_cu[:-1],
        q_cu[1:],
        original_k_cu[:-1],
        original_k_cu[1:],
        k1_cu[:-1],
        k1_cu[1:],
    ):
        k1_len = k1e - k1s
        q_position_offset = (ke - ks) - (qe - qs)
        for q_local in range(qe - qs):
            q_block = (q_position_offset + q_local) // block_size
            num_k_blocks = math.ceil((ke - ks) / block_size)
            for kb in range(num_k_blocks):
                if kb < init_blocks or kb <= q_block <= kb + local_blocks:
                    result[:, qs + q_local, kb] = torch.inf
                    continue
                begin = kb * (block_size // kernel_stride) - pad
                ids = torch.arange(begin, begin + window, device=score.device)
                valid = (ids >= 0) & (ids < k1_len)
                if valid.any():
                    result[:, qs + q_local, kb] = score[
                        :, qs + q_local, ids[valid]
                    ].amax(dim=-1)
    return result


def select_blocks_ref(
    block_score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    block_size: int = 64,
    cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select causal blocks; invalid slots are ``-1`` and rows are sorted."""
    hkv, total_q, _ = block_score.shape
    result = torch.full(
        (hkv, total_q, topk),
        -1,
        dtype=torch.int32,
        device=block_score.device,
    )
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    original_k_cu = (
        cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    ).detach().cpu().tolist()
    for qs, qe, ks, ke in zip(
        q_cu[:-1], q_cu[1:], original_k_cu[:-1], original_k_cu[1:]
    ):
        q_position_offset = (ke - ks) - (qe - qs)
        for q_local in range(qe - qs):
            q_position = q_position_offset + q_local
            valid_blocks = min(
                math.ceil((ke - ks) / block_size),
                (q_position + block_size) // block_size,
            )
            count = min(topk, valid_blocks)
            if count:
                idx = torch.argsort(
                    block_score[:, qs + q_local, :valid_blocks],
                    dim=-1,
                    descending=True,
                    stable=True,
                )[:, :count]
                result[:, qs + q_local, :count] = idx.sort(dim=-1).values.to(torch.int32)
    return result


def sparse_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_blocks: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Attention over the union of selected original-KV blocks."""
    _, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    q_fp32, k_fp32, v_fp32 = q.float(), k.float(), v.float()
    out = torch.zeros_like(q)
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    k_cu = cu_seqlens_k.detach().cpu().tolist()
    for qs, qe, ks, ke in zip(q_cu[:-1], q_cu[1:], k_cu[:-1], k_cu[1:]):
        q_len, k_len = qe - qs, ke - ks
        for qi in range(q_len):
            causal_limit = k_len - q_len + qi + 1
            for h in range(hq):
                hk = h // group
                positions: list[int] = []
                for block in selected_blocks[hk, qs + qi].detach().cpu().tolist():
                    if block < 0:
                        continue
                    lo = block * block_size
                    positions.extend(range(lo, min(lo + block_size, k_len)))
                if causal:
                    positions = [pos for pos in positions if pos < causal_limit]
                if not positions:
                    continue
                ids = torch.tensor(positions, device=q.device, dtype=torch.long)
                logits = (k_fp32[ks + ids, hk] @ q_fp32[qs + qi, h]) * scale
                probs = torch.softmax(logits, dim=0)
                out[qs + qi, h] = (
                    probs[:, None] * v_fp32[ks + ids, hk]
                ).sum(dim=0).to(q.dtype)
    return out


def dense_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Readable packed dense attention with bottom-right causal alignment."""
    _, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    outputs: list[torch.Tensor] = []
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    k_cu = cu_seqlens_k.detach().cpu().tolist()
    for qs, qe, ks, ke in zip(q_cu[:-1], q_cu[1:], k_cu[:-1], k_cu[1:]):
        q_seq = q[qs:qe].float().transpose(0, 1)
        k_seq = k[ks:ke].float().repeat_interleave(group, dim=1).transpose(0, 1)
        v_seq = v[ks:ke].float().repeat_interleave(group, dim=1).transpose(0, 1)
        logits = torch.matmul(q_seq, k_seq.transpose(-1, -2)) * scale
        if causal:
            q_len, k_len = qe - qs, ke - ks
            q_pos = (k_len - q_len) + torch.arange(q_len, device=q.device)[:, None]
            k_pos = torch.arange(k_len, device=q.device)[None, :]
            logits = logits.masked_fill((k_pos > q_pos)[None, :, :], -torch.inf)
        probs = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        outputs.append(torch.matmul(probs, v_seq).transpose(0, 1).to(q.dtype))
    return torch.cat(outputs, dim=0)
