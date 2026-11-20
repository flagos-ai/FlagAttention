# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""One complete KV owner reduction, with no conflicting gradient writers."""

import triton
import triton.language as tl


@triton.jit
def dkdv_owned_kernel(
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
    DK,
    DV,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    MAX_Q: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    M: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    owner = tl.program_id(0)
    block = owner % MAX_BLOCKS
    hk = owner // MAX_BLOCKS % HKV
    batch = owner // (MAX_BLOCKS * HKV)
    qs, qe = (tl.load(CuQ + batch), tl.load(CuQ + batch + 1))
    ks, ke = (tl.load(CuK + batch), tl.load(CuK + batch + 1))
    count = tl.load(Counts + owner)
    end = count
    m, n, d = (tl.arange(0, M), tl.arange(0, BS), tl.arange(0, BD))
    head = hk * G + m % GP
    pos = block * BS + n
    kv = (ks + pos[:, None]) * HKV * D + hk * D + d[None, :]
    kv_mask = (pos[:, None] < ke - ks) & (d[None, :] < D)
    k, v = (tl.load(K + kv, kv_mask, 0), tl.load(V + kv, kv_mask, 0))
    dk, dv = (tl.zeros((BS, BD), tl.float32), tl.zeros((BS, BD), tl.float32))
    for offset in tl.range(0, end, M // GP):
        slot = offset + m // GP
        contributor = tl.load(Queries + owner * MAX_Q + slot, slot < end, 0)
        qi = contributor % QUERY_TILE
        multiplicity = contributor // QUERY_TILE + 1
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
        p = tl.where(valid, tl.exp(logits - lse[:, None]) * multiplicity[:, None], 0.0)
        dp = tl.dot(do, tl.trans(v))
        # A future V may be infinite even when its attention probability is zero.
        ds = tl.where(valid, p * (dp - delta[:, None]) * SCALE, 0.0)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
    tl.store(DK + kv, dk, kv_mask)
    tl.store(DV + kv, dv, kv_mask)
