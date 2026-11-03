# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TLE full-score/full-D decode paths for supported BF16 MHA ranges."""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401; initialize Ascend runtime before importing TLE
import triton
import triton.language as tl
import triton.experimental.tle as tle  # noqa: F401
from triton.experimental.tle.language.dsa import tile_alloc, tile_copy, tile_to_tensor, extract_slice


_LOG2E = 1.4426950408889634
_BN = 512
_V_TILE = 1024


@triton.jit
def _prepack_qr_m2(Q, R, Packed, D: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    q = tl.load(Q + bh * D + d)
    r = tl.load(R + bh * D + d)
    rows = tl.arange(0, 2)
    packed = tl.where(rows[:, None] == 0, q[None, :], r[None, :])
    tl.store(Packed + bh * 2 * D + rows[:, None] * D + d[None, :], packed)


@triton.jit
def _score_qkrk_full(
    QRPacked, KCache, Score1, Score2, ScaleLog2, Skv,
    HQ: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BN: tl.constexpr, NUM_SPLITS: tl.constexpr,
    MAX_BLOCKS_PER_SPLIT: tl.constexpr,
):
    """Write QK*scale*log2(e) and RK to separate FP32 [BH,T] score planes."""
    bh = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    batch, head = bh // HQ, bh % HQ
    num_blocks = tl.cdiv(Skv, BN)
    first_block = (split * num_blocks) // NUM_SPLITS
    n, d = tl.arange(0, BN), tl.arange(0, D)

    q_ptr = tl.make_block_ptr(
        base=QRPacked + bh * 2 * D, shape=(2, D), strides=(D, 1),
        offsets=(0, 0), block_shape=(2, D), order=(1, 0),
    )
    q_l1 = tile_alloc([2, D], QRPacked.dtype.element_ty, tle.language.dsa.ascend.L1)
    tile_copy(q_ptr, q_l1, [2, D], inter_no_alias=True)
    qr = tile_to_tensor(q_l1, writable=False)

    base = (batch * Skv * H + head) * D
    for block_offset in tl.range(0, MAX_BLOCKS_PER_SPLIT, loop_unroll_factor=1):
        block = first_block + block_offset
        k_ptr = tl.make_block_ptr(
            base=KCache + base, shape=(Skv, D), strides=(H * D, 1),
            offsets=(block * BN, 0), block_shape=(BN, D), order=(1, 0),
        )
        k_l1 = tile_alloc([BN, D], KCache.dtype.element_ty, tle.language.dsa.ascend.L1)
        tile_copy(k_ptr, k_l1, [BN, D], inter_no_alias=True)
        k = tile_to_tensor(k_l1, writable=False)
        scores = tl.dot(qr, tl.trans(k)).to(tl.float32)
        qk = tl.reshape(extract_slice(scores, [0, 0], [1, BN], [1, 1]), (BN,))
        rk = tl.reshape(extract_slice(scores, [1, 0], [1, BN], [1, 1]), (BN,))
        linear_offset = bh * Skv + block * BN + n
        tl.store(Score1 + linear_offset, qk * ScaleLog2)
        tl.store(Score2 + linear_offset, rk)


@triton.jit
def _split_pv_full_d(
    Score1, Score2, VCache,
    PM, PD1, PD2, PO1, PO2,
    Skv,
    HQ: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    TOK_PER_SPLIT: tl.constexpr, TOK_PER_SPLIT_PADDED: tl.constexpr,
    NUM_SPLITS: tl.constexpr, V_TILE: tl.constexpr,
    NUM_V_BLOCKS: tl.constexpr,
):
    """One CTA reduces a split once and performs sequential full-D V slabs."""
    bh = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    batch, head = bh // HQ, bh % HQ
    tok = tl.arange(0, TOK_PER_SPLIT_PADDED)
    mask_tok = tok < TOK_PER_SPLIT
    rows = tl.arange(0, 2)

    score_offset = bh * Skv + split * TOK_PER_SPLIT + tok
    s1 = tl.load(Score1 + score_offset, mask=mask_tok, other=-float("inf")).to(tl.float32)
    s2 = tl.load(Score2 + score_offset, mask=mask_tok, other=0.0).to(tl.float32)
    m = tl.max(s1, axis=0)
    p1 = tl.exp2(s1 - m)
    p2 = p1 * s2
    d1 = tl.sum(p1, axis=0)
    d2 = tl.sum(p2, axis=0)
    p_full = tl.where(
        rows[:, None] == 0,
        p1.to(tl.bfloat16)[None, :],
        p2.to(tl.bfloat16)[None, :],
    )

    base = (batch * Skv * H + head) * D
    v_l1 = tile_alloc([V_TILE, 128], VCache.dtype.element_ty, tle.language.dsa.ascend.L1)
    pv_acc = tl.zeros((2, 128), tl.float32)
    for block in tl.range(0, NUM_V_BLOCKS, loop_unroll_factor=1):
        v_ptr = tl.make_block_ptr(
            base=VCache + base, shape=(Skv, D), strides=(H * D, 1),
            offsets=(split * TOK_PER_SPLIT + block * V_TILE, 0),
            block_shape=(V_TILE, 128), order=(1, 0),
        )
        tile_copy(v_ptr, v_l1, [V_TILE, 128], inter_no_alias=True)
        v = tile_to_tensor(v_l1, writable=False)
        p_block = extract_slice(p_full, [0, block * V_TILE], [2, V_TILE], [1, 1])
        pv_acc = tl.dot(p_block, v, acc=pv_acc, out_dtype=tl.float32)

    pv1 = tl.reshape(extract_slice(pv_acc, [0, 0], [1, 128], [1, 1]), (128,))
    pv2 = tl.reshape(extract_slice(pv_acc, [1, 0], [1, 128], [1, 1]), (128,))
    state = bh * NUM_SPLITS + split
    tl.store(PM + state, m)
    tl.store(PD1 + state, d1)
    tl.store(PD2 + state, d2)
    vector = state * D + tl.arange(0, 128)
    tl.store(PO1 + vector, pv1)
    tl.store(PO2 + vector, pv2)


@triton.jit
def _corrected_fullcache_pv(
    Score1, Score2, VCache, Out,
    T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    PV_TILE: tl.constexpr, TOK_PADDED: tl.constexpr,
):
    """Reduce whole-cache corrected weights, then stream full-D V slabs."""
    bh = tl.program_id(0).to(tl.int64)
    batch, head = bh // HQ, bh % HQ
    tok = tl.arange(0, TOK_PADDED)
    feat = tl.arange(0, D)

    mask_tok = tok < T
    s1 = tl.load(Score1 + bh * T + tok, mask=mask_tok, other=-float("inf")).to(tl.float32)
    s2 = tl.load(Score2 + bh * T + tok, mask=mask_tok, other=0.0).to(tl.float32)
    m = tl.max(s1, axis=0)
    p = tl.exp2(s1 - m)
    d1 = tl.sum(p, axis=0)
    d2 = tl.sum(p * s2, axis=0)
    corrected = (p / d1) * (1.0 + d2 / d1 - s2)
    corrected = tl.reshape(corrected.to(tl.bfloat16), (1, TOK_PADDED))

    base = (batch * T * H + head) * D
    v_l1 = tile_alloc([PV_TILE, D], VCache.dtype.element_ty, tle.language.dsa.ascend.L1)
    acc = tl.zeros((1, D), tl.float32)
    for block in tl.range(0, T // PV_TILE, loop_unroll_factor=1):
        v_ptr = tl.make_block_ptr(
            base=VCache + base, shape=(T, D), strides=(H * D, 1),
            offsets=(block * PV_TILE, 0), block_shape=(PV_TILE, D), order=(1, 0),
        )
        tile_copy(v_ptr, v_l1, [PV_TILE, D], inter_no_alias=True)
        v = tile_to_tensor(v_l1, writable=False)
        p_block = extract_slice(
            corrected, [0, block * PV_TILE], [1, PV_TILE], [1, 1]
        )
        acc = tl.dot(p_block, v, acc=acc, out_dtype=tl.float32)

    row = tl.reshape(extract_slice(acc, [0, 0], [1, D], [1, 1]), (D,))
    tl.store(Out + bh * D + feat, row)


def decode_large_mha_fullscore(q, r, k, v, *, scale: float, out, merge_kernel, num_splits: int):
    """Run shared full-score setup with corrected whole-cache or split PV."""
    batch, _, heads, dim = q.shape
    length = k.shape[1]
    if dim != 128 or k.shape[2] != heads or v.shape != k.shape:
        raise ValueError("full-score decode expects MHA with D=128 and matching K/V")
    if q.dtype != torch.bfloat16 or r.dtype != q.dtype or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("full-score decode expects BF16 Q/R/K/V")
    if length < 1024 or length % 1024:
        raise ValueError(f"unsupported sequence length {length}; expected a multiple of 1024")
    bh = batch * heads
    num_tiles = length // _V_TILE
    if num_splits < 1 or num_tiles % num_splits:
        raise ValueError(f"split count {num_splits} must divide {num_tiles} V tiles")
    tok_per_split = length // num_splits
    if num_splits == 1:
        if length > 8192:
            raise ValueError("whole-cache corrected PV is limited to 8192 tokens")
    elif num_splits > 8 or tok_per_split > 8192:
        raise ValueError(f"unsupported split working set {tok_per_split} with {num_splits} splits")
    num_v_blocks = tok_per_split // _V_TILE

    packed = torch.empty((bh, 2, dim), device=q.device, dtype=q.dtype)
    scores1 = torch.empty((bh, length), device=q.device, dtype=torch.float32)
    scores2 = torch.empty_like(scores1)

    _prepack_qr_m2[(bh,)](q, r, packed, dim, num_warps=4, num_stages=1)
    _score_qkrk_full[(bh, num_splits)](
        packed, k, scores1, scores2, scale * _LOG2E, length,
        HQ=heads, H=heads, D=dim, BN=_BN, NUM_SPLITS=num_splits,
        MAX_BLOCKS_PER_SPLIT=triton.cdiv(triton.cdiv(length, _BN), num_splits),
        num_warps=4, num_stages=2,
    )
    if num_splits == 1:
        _corrected_fullcache_pv[(bh,)](
            scores1, scores2, v, out, length, HQ=heads, H=heads, D=dim,
            PV_TILE=_V_TILE, TOK_PADDED=triton.next_power_of_2(length),
            num_warps=4, num_stages=2, multibuffer=False,
        )
        return out

    partial_m = torch.empty((bh, num_splits), device=q.device, dtype=torch.float32)
    partial_d1 = torch.empty_like(partial_m)
    partial_d2 = torch.empty_like(partial_m)
    partial_o1 = torch.empty((bh, num_splits, dim), device=q.device, dtype=torch.float32)
    partial_o2 = torch.empty_like(partial_o1)
    _split_pv_full_d[(bh, num_splits)](
        scores1, scores2, v, partial_m, partial_d1, partial_d2, partial_o1, partial_o2,
        length, HQ=heads, H=heads, D=dim, TOK_PER_SPLIT=tok_per_split,
        TOK_PER_SPLIT_PADDED=triton.next_power_of_2(tok_per_split),
        NUM_SPLITS=num_splits, NUM_V_BLOCKS=num_v_blocks, V_TILE=_V_TILE,
        num_warps=4, num_stages=2, multibuffer=False,
    )
    merge_kernel[(bh,)](
        partial_m, partial_d1, partial_d2, partial_o1, partial_o2, out,
        K=dim, BK=dim, NUM_SPLITS=num_splits,
        BSPLIT=triton.next_power_of_2(num_splits),
        num_warps=4, num_stages=1,
    )
    return out
