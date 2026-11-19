# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Fused coalesced means with device-resident packed metadata."""

import torch
import triton
import triton.language as tl


@triton.jit
def compression_metadata_kernel(CuIn, Cu1, Cu2, BATCH: tl.constexpr, BB: tl.constexpr):
    b = tl.arange(0, BB)
    begin = tl.load(CuIn + b, b < BATCH, other=0)
    end = tl.load(CuIn + b + 1, b < BATCH, other=0)
    length = end - begin
    fine = tl.where((b < BATCH) & (length >= 32), (length - 32) // 16 + 1, 0)
    coarse = tl.where((b < BATCH) & (length >= 128), (length - 128) // 64 + 1, 0)
    tl.store(Cu1 + b + 1, tl.cumsum(fine, 0), b < BATCH)
    tl.store(Cu2 + b + 1, tl.cumsum(coarse, 0), b < BATCH)
    tl.store(Cu1, 0)
    tl.store(Cu2, 0)


@triton.jit
def paired_compression_kernel(
    K, Out1, Out2, CuIn, Cu1, Cu2, HKV: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
    INLINE_METADATA: tl.constexpr = False,
    WIDE_OFFSETS: tl.constexpr = False,
):
    index_ty: tl.constexpr = tl.int64 if WIDE_OFFSETS else tl.int32
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, end = tl.load(CuIn + batch), tl.load(CuIn + batch + 1)
    if INLINE_METADATA:
        s1, s2 = 0, 0
        length = end - start
        e1 = tl.where(length >= 32, (length - 32) // 16 + 1, 0)
        e2 = tl.where(length >= 128, (length - 128) // 64 + 1, 0)
        if tile == 0 and head == 0:
            index = tl.arange(0, 2)
            tl.store(Cu1 + index, tl.where(index == 0, 0, e1))
            tl.store(Cu2 + index, tl.where(index == 0, 0, e2))
    else:
        s1, e1 = tl.load(Cu1 + batch), tl.load(Cu1 + batch + 1)
        s2, e2 = tl.load(Cu2 + batch), tl.load(Cu2 + batch + 1)
    if tile * 4 < e1 - s1:
        row, dim = tl.arange(0, 128), tl.arange(0, BD)
        pos = start.to(index_ty) + tile * 64 + row
        values = tl.load(
            K + pos[:, None] * HKV * D + head * D + dim[None, :], (pos[:, None] < end) & (dim[None, :] < D), other=0.0
        ).to(tl.float32)
        coarse = tl.sum(values, 0) / 128
        tl.store(Out2 + (s2.to(index_ty) + tile) * HKV * D + head * D + dim, coarse, (tile < e2 - s2) & (dim < D))
        for offset in tl.static_range(0, 4):
            # A register gather of the shared 128-row tile lowers to scalar
            # indexed work on this backend. Coalesced reloads retain the exact
            # original 32-row reduction and reuse the immediately preceding K
            # tile through the cache without keeping the whole tile live.
            fine_pos = start.to(index_ty) + tile * 64 + offset * 16 + tl.arange(0, 32)
            window = tl.load(K + fine_pos[:, None] * HKV * D + head * D + dim[None, :],
                             (fine_pos[:, None] < end) & (dim[None, :] < D), other=0.0).to(tl.float32)
            fine = tl.sum(window, 0) / 32
            index = tile * 4 + offset
            tl.store(Out1 + (s1.to(index_ty) + index) * HKV * D + head * D + dim, fine, (index < e1 - s1) & (dim < D))


def compress_pair(k, cu_seqlens_k, max_seqlen_k, *, single_token=False):
    """Private capacity buffers; Cu1/Cu2, not capacity, define packed lengths."""
    batch = cu_seqlens_k.numel() - 1
    n1 = max(0, (max_seqlen_k - 32) // 16 + 1)
    n2 = max(0, (max_seqlen_k - 128) // 64 + 1)
    hkv, d = k.shape[1:]
    k1 = torch.empty((batch * n1, hkv, d), dtype=k.dtype, device=k.device)
    k2 = torch.empty((batch * n2, hkv, d), dtype=k.dtype, device=k.device)
    cu1, cu2 = torch.empty_like(cu_seqlens_k), torch.empty_like(cu_seqlens_k)
    # A single decode sequence needs no prefix scan. One compression
    # program publishes both lengths for the subsequent kernels.
    inline_metadata = single_token and batch == 1 and n1 > 0
    if not inline_metadata:
        compression_metadata_kernel[(1,)](
            cu_seqlens_k, cu1, cu2, BATCH=batch, BB=triton.next_power_of_2(batch), num_warps=4
        )
    if n1:
        paired_compression_kernel[(triton.cdiv(n1, 4), hkv, batch)](
            k, k1, k2, cu_seqlens_k, cu1, cu2, HKV=hkv, D=d, BD=triton.next_power_of_2(d),
            INLINE_METADATA=inline_metadata,
            WIDE_OFFSETS=max(k.numel(), k1.numel(), k2.numel()) > 2**31 - 1,
            num_warps=4,
        )
    return k1, cu1, k2, cu2
