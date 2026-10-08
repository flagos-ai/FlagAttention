# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Sparse backward with a query dQ pass and KV-owned grouped reductions.

The reverse index is built in backward and its cost is part of backward timing.
Each query keeps its own selection. Only contributions to the SAME KV block
are grouped; no query tokens are forced to share a TopK decision.

This is not a deterministic backward: split tasks still use FP32 atomic add.
The reverse index reserves O(B * Hkv * ceil(max_K / BS) * max_Q) int32 entries.
The selected list must contain unique valid block IDs per query (plus -1 pads),
as guaranteed by this module's selector.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _build_reverse_index(
    Selected,
    CuQ,
    Counts,
    Queries,
    TOTAL_Q: tl.constexpr,
    TOPK: tl.constexpr,
    HKV: tl.constexpr,
    MAX_Q: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    TILE: tl.constexpr,
):
    tile, hk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    edge = tile * TILE + tl.arange(0, TILE)
    qi, rank = edge // TOPK, edge % TOPK
    block = tl.load(Selected + (hk * TOTAL_Q + qs + qi) * TOPK + rank, mask=qi < qe - qs, other=-1)
    valid = (qi < qe - qs) & (block >= 0) & (block < MAX_BLOCKS)
    owner = (batch * HKV + hk) * MAX_BLOCKS + tl.maximum(block, 0)
    slot = tl.atomic_add(Counts + owner, 1, mask=valid, sem="relaxed")
    tl.store(Queries + owner * MAX_Q + slot, qi, mask=valid)


@triton.jit
def _make_tasks(Counts, TaskCu, Tasks, MAX_TASKS: tl.constexpr, QUERY_SPLIT: tl.constexpr):
    owner = tl.program_id(0)
    offsets = tl.arange(0, MAX_TASKS)
    count = tl.load(Counts + owner)
    start = tl.load(TaskCu + owner)
    valid = offsets * QUERY_SPLIT < count
    tl.store(Tasks + (start + offsets) * 2, owner, mask=valid)
    tl.store(Tasks + (start + offsets) * 2 + 1, offsets * QUERY_SPLIT, mask=valid)


@triton.jit
def _dq_kernel(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    Selected,
    CuQ,
    CuK,
    DQ,
    Delta,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    TOPK: tl.constexpr,
    BS: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    M: tl.constexpr,
    BD: tl.constexpr,
):
    qi, hk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    ks, ke = tl.load(CuK + batch), tl.load(CuK + batch + 1)
    if qi >= qe - qs:
        return
    g, d, n = tl.arange(0, M), tl.arange(0, BD), tl.arange(0, BS)
    head = hk * G + g
    qoff = (qs + qi) * HQ * D + head[:, None] * D + d[None, :]
    mask = (g[:, None] < G) & (d[None, :] < D)
    q, o, do = tl.load(Q + qoff, mask, 0), tl.load(O + qoff, mask, 0), tl.load(DO + qoff, mask, 0)
    lse = tl.load(LSE + (qs + qi) * HQ + head, g < G, other=float("-inf"))
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    tl.store(Delta + (qs + qi) * HQ + head, delta, g < G)
    dq = tl.zeros((M, BD), tl.float32)
    limit = ke - ks - (qe - qs) + qi + 1
    for rank in tl.range(0, TOPK):
        block = tl.load(Selected + (hk * TOTAL_Q + qs + qi) * TOPK + rank)
        if block >= 0:
            pos = block * BS + n
            valid = pos < ke - ks
            if CAUSAL:
                valid = valid & (pos < limit)
            kv = (ks + pos[:, None]) * HKV * D + hk * D + d[None, :]
            k = tl.load(K + kv, valid[:, None] & (d[None, :] < D), 0)
            v = tl.load(V + kv, valid[:, None] & (d[None, :] < D), 0)
            logits = tl.dot(q, tl.trans(k)) * SCALE
            p = tl.where(
                (g[:, None] < G) & valid[None, :] & (lse[:, None] != float("-inf")), tl.exp(logits - lse[:, None]), 0.0
            )
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None]) * SCALE
            dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ + qoff, dq, mask)


@triton.jit
def _dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    CuQ,
    CuK,
    Counts,
    Queries,
    Tasks,
    DK,
    DV,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    MAX_Q: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    M: tl.constexpr,
    QUERY_SPLIT: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    task = tl.program_id(0)
    owner, start = tl.load(Tasks + task * 2), tl.load(Tasks + task * 2 + 1)
    block = owner % MAX_BLOCKS
    hk = owner // MAX_BLOCKS % HKV
    batch = owner // (MAX_BLOCKS * HKV)
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    ks, ke = tl.load(CuK + batch), tl.load(CuK + batch + 1)
    count = tl.load(Counts + owner)
    end = tl.minimum(count, start + QUERY_SPLIT)
    m, n, d = tl.arange(0, M), tl.arange(0, BS), tl.arange(0, BD)
    head = hk * G + m % GP
    pos = block * BS + n
    kv = (ks + pos[:, None]) * HKV * D + hk * D + d[None, :]
    kv_mask = (pos[:, None] < ke - ks) & (d[None, :] < D)
    k, v = tl.load(K + kv, kv_mask, 0), tl.load(V + kv, kv_mask, 0)
    dk, dv = tl.zeros((BS, BD), tl.float32), tl.zeros((BS, BD), tl.float32)
    for offset in tl.range(start, end, M // GP):
        slot = offset + m // GP
        qi = tl.load(Queries + owner * MAX_Q + slot, slot < end, 0)
        valid_m = (slot < end) & (m % GP < G)
        qoff = (qs + qi[:, None]) * HQ * D + head[:, None] * D + d[None, :]
        q = tl.load(Q + qoff, valid_m[:, None] & (d[None, :] < D), 0)
        do = tl.load(DO + qoff, valid_m[:, None] & (d[None, :] < D), 0)
        lse = tl.load(LSE + (qs + qi) * HQ + head, valid_m, other=float("-inf"))
        delta = tl.load(Delta + (qs + qi) * HQ + head, valid_m, 0)
        valid = valid_m[:, None] & (pos[None, :] < ke - ks) & (lse[:, None] != float("-inf"))
        if CAUSAL:
            valid = valid & (pos[None, :] < ke - ks - (qe - qs) + qi[:, None] + 1)
        logits = tl.dot(q, tl.trans(k)) * SCALE
        p = tl.where(valid, tl.exp(logits - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * SCALE
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
    # One update per element per QUERY_SPLIT contributors, not per query.
    tl.atomic_add(DK + kv, dk, kv_mask, sem="relaxed")
    tl.atomic_add(DV + kv, dv, kv_mask, sem="relaxed")


def sparse_backward(
    q, k, v, out, dout, lse, selected, cu_q, cu_k, max_q, block_size=64, softmax_scale=None, causal=True
):
    """Compute dQ/dK/dV, including fresh reverse-index construction each call.

    Host reads for max_K/task count synchronize CUDA. Timing must include the
    whole function, not only the two attention-gradient kernel launches.
    """
    total_q, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    max_k = int((cu_k[1:] - cu_k[:-1]).max().item())
    max_blocks = triton.cdiv(max_k, block_size)
    owners = (cu_q.numel() - 1) * hkv * max_blocks
    counts = torch.zeros(owners, device=q.device, dtype=torch.int32)
    # Capacity <= one entry per query per KV block (selected indices are unique).
    queries = torch.empty((owners, max_q), device=q.device, dtype=torch.int32)
    topk = selected.shape[-1]
    _build_reverse_index[(triton.cdiv(max_q * topk, 256), hkv, cu_q.numel() - 1)](
        selected,
        cu_q,
        counts,
        queries,
        TOTAL_Q=total_q,
        TOPK=topk,
        HKV=hkv,
        MAX_Q=max_q,
        MAX_BLOCKS=max_blocks,
        TILE=256,
        num_warps=4,
    )
    query_split = 64
    task_cu = torch.empty(owners + 1, device=q.device, dtype=torch.int32)
    task_cu[0] = 0
    torch.cumsum((counts + query_split - 1) // query_split, 0, out=task_cu[1:])
    num_tasks = int(task_cu[-1].item())
    tasks = torch.empty((num_tasks, 2), device=q.device, dtype=torch.int32)
    _make_tasks[(owners,)](
        counts,
        task_cu,
        tasks,
        MAX_TASKS=triton.next_power_of_2(triton.cdiv(max_q, query_split)),
        QUERY_SPLIT=query_split,
        num_warps=4,
    )
    dq = torch.empty_like(q)
    delta = torch.empty((total_q, hq), device=q.device, dtype=torch.float32)
    dk, dv = torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32)
    scale = d**-0.5 if softmax_scale is None else softmax_scale
    _dq_kernel[(max_q, hkv, cu_q.numel() - 1)](
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
        num_warps=4,
        num_stages=2,
    )
    if num_tasks:
        gp = triton.next_power_of_2(group)
        _dkdv_kernel[(num_tasks,)](
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
            MAX_Q=max_q,
            MAX_BLOCKS=max_blocks,
            BS=block_size,
            BD=triton.next_power_of_2(d),
            M=max(16, 4 * gp),
            QUERY_SPLIT=query_split,
            SCALE=scale,
            CAUSAL=causal,
            num_warps=8,
            num_stages=2,
        )
    return dq, dk.to(k.dtype), dv.to(v.dtype)
