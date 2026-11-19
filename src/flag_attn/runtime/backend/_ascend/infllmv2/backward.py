# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""KV-owned reverse index with no contended atomic return values."""

import torch
import triton

from flag_attn.infllmv2 import backward as kernels
from .kernels import reverse_index_kernel, make_tasks_kernel
from .owned_backward import dkdv_owned_kernel


def sparse_backward(
    q, k, v, out, dout, lse, selected, cu_q, cu_k, max_q, block_size=64, softmax_scale=None, causal=True
):
    if not q.numel() or not k.numel():
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    if max(q.numel(), k.numel(), selected.numel()) > 2**31 - 1:
        raise ValueError("Ascend InfLLM-V2 backward requires 32-bit element offsets")
    total_q, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    # A single packed sequence spans K, so its length is host-visible.
    max_k = k.shape[0] if cu_k.numel() == 2 else int((cu_k[1:] - cu_k[:-1]).max().item())
    max_blocks = triton.cdiv(max_k, block_size)
    owners = (cu_q.numel() - 1) * hkv * max_blocks
    query_capacity = triton.cdiv(max_q, 8) * 8
    query_tile = triton.next_power_of_2(max_q)
    topk = selected.shape[-1]
    if max(owners * query_capacity, query_tile * topk - 1) > 2**31 - 1:
        raise ValueError("Ascend InfLLM-V2 reverse index requires 32-bit element offsets")
    counts = torch.empty(owners, device=q.device, dtype=torch.int32)
    queries = torch.empty((owners, query_capacity), device=q.device, dtype=torch.int32)
    reverse_index_kernel[(owners,)](
        selected,
        cu_q,
        counts,
        queries,
        TOTAL_Q=total_q,
        TOPK=topk,
        HKV=hkv,
        MAX_Q=query_capacity,
        MAX_BLOCKS=max_blocks,
        # Bounded live tile: a full max_q x topk block overflows AIV UB.
        BQ=min(query_tile, 128),
        BK=min(triton.next_power_of_2(topk), 64),
        # The decoded multiplicity base stays the whole-sequence tile.
        ENC=query_tile,
        num_warps=4,
    )
    if max_q > 128:
        query_split = 64
        task_cu = torch.empty(owners + 1, device=q.device, dtype=torch.int32)
        task_cu[0] = 0
        torch.cumsum((counts + query_split - 1) // query_split, 0, out=task_cu[1:])
        num_tasks = int(task_cu[-1].item())
        tasks = torch.empty((num_tasks, 2), device=q.device, dtype=torch.int32)
        if num_tasks:
            make_tasks_kernel[(triton.cdiv(num_tasks * 2, 512),)](
                task_cu,
                tasks,
                OWNERS=owners,
                NUM_TASKS=num_tasks,
                QUERY_SPLIT=query_split,
                TILE=512,
                num_warps=4,
            )
    dq = torch.empty_like(q)
    delta = torch.empty((total_q, hq), device=q.device, dtype=torch.float32)
    if max_q <= 128:
        dk, dv = torch.empty_like(k), torch.empty_like(v)
    else:
        dk, dv = torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32)
    scale = d**-0.5 if softmax_scale is None else softmax_scale
    kernels._dq_kernel[(max_q, hkv, cu_q.numel() - 1)](
        q,
        k,
        v,
        out,
        dout,
        lse,
        selected,
        cu_q,
        cu_k,
        dq,
        delta,
        HQ=hq,
        HKV=hkv,
        G=group,
        D=d,
        TOTAL_Q=total_q,
        TOPK=topk,
        BS=block_size,
        SCALE=scale,
        CAUSAL=causal,
        M=max(16, triton.next_power_of_2(group)),
        BD=triton.next_power_of_2(d),
        CHECK_BLOCK_BOUNDS=True,
        num_warps=4,
        num_stages=2,
    )
    if max_q <= 128:
        gp = triton.next_power_of_2(group)
        dkdv_owned_kernel[(owners,)](
            q,
            k,
            v,
            dout,
            lse,
            delta,
            cu_q,
            cu_k,
            counts,
            queries,
            dk,
            dv,
            HQ=hq,
            HKV=hkv,
            G=group,
            GP=gp,
            D=d,
            MAX_Q=query_capacity,
            QUERY_TILE=query_tile,
            MAX_BLOCKS=max_blocks,
            BS=block_size,
            BD=triton.next_power_of_2(d),
            # One query's GQA rows keep Q/DO address bases constant.
            M=max(16, gp),
            SCALE=scale,
            CAUSAL=causal,
            num_warps=8,
            num_stages=2,
        )
        return dq, dk, dv
    if num_tasks:
        gp = triton.next_power_of_2(group)
        kernels._dkdv_kernel[(num_tasks,)](
            q,
            k,
            v,
            dout,
            lse,
            delta,
            cu_q,
            cu_k,
            counts,
            queries,
            tasks,
            dk,
            dv,
            HQ=hq,
            HKV=hkv,
            G=group,
            GP=gp,
            D=d,
            MAX_Q=query_capacity,
            MAX_BLOCKS=max_blocks,
            BS=block_size,
            BD=triton.next_power_of_2(d),
            M=max(16, 4 * gp),
            QUERY_SPLIT=query_split,
            QUERY_TILE=query_tile,
            SCALE=scale,
            CAUSAL=causal,
            num_warps=8,
            num_stages=2,
        )
    return dq, dk.to(k.dtype), dv.to(v.dtype)
