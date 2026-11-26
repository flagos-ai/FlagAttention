# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Optional Blackwell Gluon kernels; imported only by eligible SM100 launches.

The public entry points, validation and tuning policy remain in sparse_attn.
Importing ordinary/Hopper MSA does not require Gluon or Blackwell language APIs.
"""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
    mbarrier,
    fence_async_shared,
)

@gluon.jit
def _gqa_sparse_prefill_persistent_q_fp8_kv_fp8_kernel(
    Q,
    KV,
    Pages,
    CuQ,
    Seq,
    Prefix,
    RowPtr,
    TilePtr,
    QueryIds,
    Partial,
    LSE,
    KG,
    VG,
    qs0: gl.constexpr,
    qs1: gl.constexpr,
    kvs0: gl.constexpr,
    kvs1: gl.constexpr,
    kvs2: gl.constexpr,
    ps0: gl.constexpr,
    HEADS_Q: gl.constexpr,
    BATCH: gl.constexpr,
    PAGES: gl.constexpr,
    TOPK: gl.constexpr,
    CAPACITY: gl.constexpr,
    GQA: gl.constexpr,
    SCALE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    TILES_PER_CTA: gl.constexpr,
    GLOBAL_SCALE: gl.constexpr,
    NARROW_OUTPUT: gl.constexpr = True,
):
    gl.static_assert(Q.dtype.element_ty == gl.float8e4nv and KV.dtype.element_ty == gl.float8e4nv)
    gid, hb = gl.program_id(0), gl.program_id(1)
    head, batch = hb // BATCH, hb % BATCH
    tiles = TilePtr + hb * (PAGES + 1)
    if gid >= gl.load(tiles + PAGES):
        return
    lo, hi = 0, PAGES
    while lo < hi:
        mid = (lo + hi) // 2
        right = gid >= gl.load(tiles + mid + 1)
        lo, hi = gl.where(right, mid + 1, lo), gl.where(right, hi, mid)
    block = lo
    row_start = gl.load(RowPtr + hb * (PAGES + 1) + block)
    row_end = gl.load(RowPtr + hb * (PAGES + 1) + block + 1)
    group_start = gid - gl.load(tiles + block)
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    page = gl.load(Pages + batch * ps0 + block).to(gl.int64)
    kr = gl.arange(0, 128, gl.SliceLayout(1, L))
    kd = gl.arange(0, 128, gl.SliceLayout(0, L))
    offs = page * kvs0 + head * kvs1 + kr[:, None] * kvs2 + kd[None, :]
    value = gl.load(KV + offs + 128)
    PDT: gl.constexpr = gl.float8e4nv
    T: gl.constexpr = TensorMemoryLayout([BLOCK_M, 128], col_stride=1)
    R: gl.constexpr = get_tmem_reg_layout(gl.float32, [BLOCK_M, 128], T, 4)
    PR: gl.constexpr = get_tmem_reg_layout(PDT, [BLOCK_M, 128], T, 4)
    SL: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2, transposed=False)
    qbufs = gl.allocate_shared_memory(gl.float8e4nv, [2, BLOCK_M, 128], SL)
    ksm = gl.allocate_shared_memory(gl.float8e4nv, [128, 128], SL)
    VSL: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=PDT.primitive_bitwidth, rank=2, transposed=False
    )
    vsm = gl.allocate_shared_memory(PDT, [128, 128], VSL)
    acc = allocate_tensor_memory(gl.float32, [BLOCK_M, 128], T)
    phsm = allocate_tensor_memory(PDT, [BLOCK_M, 128], T)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    ksm.store(gl.load(KV + offs))
    vsm.store(value.to(PDT))
    fence_async_shared()
    qstart, prefix, seq = gl.load(CuQ + batch), gl.load(Prefix + batch), gl.load(Seq + batch)
    qks, vs = SCALE * 1.4426950408889634, 1.0
    if GLOBAL_SCALE:
        qks *= gl.load(KG)
        vs = gl.load(VG)
    ms = gl.arange(0, BLOCK_M, gl.SliceLayout(1, L))
    ds = gl.arange(0, 128, gl.SliceLayout(0, L))
    nr = gl.arange(0, 128, gl.SliceLayout(0, R))
    group_row = row_start + group_start * TILES_PER_CTA * (BLOCK_M // GQA)
    ntile = gl.minimum(TILES_PER_CTA, gl.cdiv(row_end - group_row, BLOCK_M // GQA))
    first_row = group_row + ms // GQA
    encoded = gl.load(QueryIds + hb * CAPACITY + first_row, first_row < row_end, 0)
    qhead = head * GQA + ms % GQA
    cp.async_copy_global_to_shared(
        qbufs.index(0),
        Q + (encoded // TOPK)[:, None] * qs0 + qhead[:, None] * qs1 + ds[None, :],
        mask=(first_row < row_end)[:, None],
    )
    cp.commit_group()
    for tile in range(ntile):
        row = group_row + tile * (BLOCK_M // GQA) + ms // GQA
        active = row < row_end
        token, slot = encoded // TOPK, encoded % TOPK
        cp.wait_group(0)
        fence_async_shared()
        tcgen05_mma(qbufs.index(tile % 2), ksm.permute((1, 0)), acc, use_acc=False)
        tcgen05_commit(bar)
        next_row = row + BLOCK_M // GQA
        next_active = (next_row < row_end) & (tile + 1 < ntile)
        next_encoded = gl.load(QueryIds + hb * CAPACITY + next_row, next_active, 0)
        cp.async_copy_global_to_shared(
            qbufs.index((tile + 1) % 2),
            Q + (next_encoded // TOPK)[:, None] * qs0 + qhead[:, None] * qs1 + ds[None, :],
            mask=next_active[:, None],
        )
        cp.commit_group()
        mbarrier.wait(bar, phase=0)
        scores = acc.load(R) * qks
        qpos = gl.convert_layout(prefix + token - qstart, gl.SliceLayout(1, R))
        valid = gl.convert_layout(active, gl.SliceLayout(1, R))
        pos = block * 128 + nr
        scores = gl.where(
            valid[:, None] & (pos[None, :] <= qpos[:, None]) & (pos[None, :] < seq), scores, float("-inf")
        )
        maximum = gl.where(valid, gl.max(scores, 1), 0.0)
        prob = gl.exp2(scores - maximum[:, None])
        denom = gl.sum(prob, 1)
        phsm.store(gl.convert_layout(prob.to(PDT), PR))
        fence_async_shared()
        tcgen05_mma(phsm, vsm, acc, use_acc=False)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=1)
        result = acc.load(R) * (vs / gl.where(denom > 0, denom, 1.0))[:, None]
        # Both paths store BF16. Conversion order trades layout-transfer cost
        # against register pressure; the host selects it by query-length regime.
        if NARROW_OUTPUT:
            result = result.to(gl.bfloat16)
        result = gl.convert_layout(result, L)
        lse = gl.convert_layout(maximum + gl.log2(denom), gl.SliceLayout(1, L))
        outrow = (token * HEADS_Q + qhead) * TOPK + slot
        gl.store(Partial + outrow[:, None] * 128 + ds[None, :], result, active[:, None])
        gl.store(LSE + outrow, lse, active)
        encoded = next_encoded
    cp.wait_group(0)
    mbarrier.invalidate(bar)

@gluon.jit
def _gqa_sparse_prefill_persistent_q_fp8_kv_nvfp4_kernel(
    Q,
    KV,
    Pages,
    CuQ,
    Seq,
    Prefix,
    RowPtr,
    TilePtr,
    QueryIds,
    Partial,
    LSE,
    KG,
    VG,
    qs0: gl.constexpr,
    qs1: gl.constexpr,
    kvs0: gl.constexpr,
    kvs1: gl.constexpr,
    kvs2: gl.constexpr,
    ps0: gl.constexpr,
    HEADS_Q: gl.constexpr,
    BATCH: gl.constexpr,
    PAGES: gl.constexpr,
    TOPK: gl.constexpr,
    CAPACITY: gl.constexpr,
    GQA: gl.constexpr,
    SCALE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    TILES_PER_CTA: gl.constexpr,
    GLOBAL_SCALE: gl.constexpr,
    DENSE_BUCKETS: gl.constexpr = False,
    MAX_QUERY: gl.constexpr = 0,
    VS=None,
):
    # KV/VS contain E4M3 K/V normalized by /8 after NVFP4 block scales.
    # They are regenerated on every host invocation.
    gl.static_assert(Q.dtype.element_ty == gl.float8e4nv)
    gl.static_assert(KV.dtype.element_ty == gl.float8e4nv and VS.dtype.element_ty == gl.float8e4nv)
    gid, hb = gl.program_id(0), gl.program_id(1)
    head, batch = hb // BATCH, hb % BATCH
    if DENSE_BUCKETS:
        groups: gl.constexpr = triton.cdiv(MAX_QUERY, TILES_PER_CTA * (BLOCK_M // GQA))
        block, group_start = gid // groups, gid % groups
        row_start = block * MAX_QUERY
        row_end = row_start + gl.load(RowPtr + hb * PAGES + block)
        if group_start * TILES_PER_CTA * (BLOCK_M // GQA) >= row_end - row_start:
            return
    else:
        tiles = TilePtr + hb * (PAGES + 1)
        if gid >= gl.load(tiles + PAGES):
            return
        lo, hi = 0, PAGES
        while lo < hi:
            mid = (lo + hi) // 2
            right = gid >= gl.load(tiles + mid + 1)
            lo = gl.where(right, mid + 1, lo)
            hi = gl.where(right, hi, mid)
        block = lo
        row_start = gl.load(RowPtr + hb * (PAGES + 1) + block)
        row_end = gl.load(RowPtr + hb * (PAGES + 1) + block + 1)
        group_start = gid - gl.load(tiles + block)
    # Wider input vectors amortize staged-cache loads; the output layout
    # retains coalesced BF16 stores without changing four-warp MMA ownership.
    L: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    OL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    T: gl.constexpr = TensorMemoryLayout([BLOCK_M, 128], col_stride=1)
    R: gl.constexpr = get_tmem_reg_layout(gl.float32, [BLOCK_M, 128], T, 4)
    QDT: gl.constexpr = gl.float8e4nv
    QSL: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=QDT.primitive_bitwidth, rank=2, transposed=False
    )
    VSL: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2, transposed=False)
    QSL3: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=QDT.primitive_bitwidth, rank=2, transposed=False
    )
    qbufs = gl.allocate_shared_memory(QDT, [2, BLOCK_M, 128], QSL3)
    KDT: gl.constexpr = gl.float8e4nv
    ksm = gl.allocate_shared_memory(KDT, [128, 128], QSL)
    PDT: gl.constexpr = gl.float8e4nv
    vsm = gl.allocate_shared_memory(PDT, [128, 128], VSL)
    acc = allocate_tensor_memory(gl.float32, [BLOCK_M, 128], T)
    PT: gl.constexpr = TensorMemoryLayout([BLOCK_M, 128], col_stride=1)
    PR: gl.constexpr = get_tmem_reg_layout(PDT, [BLOCK_M, 128], PT, 4)
    phsm = allocate_tensor_memory(PDT, [BLOCK_M, 128], PT)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    page = gl.load(Pages + batch * ps0 + block).to(gl.int64)
    kr = gl.arange(0, 128, gl.SliceLayout(1, L))
    kd = gl.arange(0, 128, gl.SliceLayout(0, L))
    offs = page * kvs0 + head * kvs1 + kr[:, None] * kvs2 + kd[None, :]
    ksm.store(gl.load(KV + offs))
    v = gl.load(VS + offs)
    vsm.store(v.to(PDT))
    fence_async_shared()
    qstart = gl.load(CuQ + batch)
    prefix, seq = gl.load(Prefix + batch), gl.load(Seq + batch)
    # Compensate _nvfp4_to_fp8's fixed /8 in FP32, not in FP8 operands.
    qks = SCALE * 1.4426950408889634 * 8.0
    vs = 8.0
    if GLOBAL_SCALE:
        qks *= gl.load(KG)
        vs *= gl.load(VG)
    ms = gl.arange(0, BLOCK_M, gl.SliceLayout(1, L))
    ds = gl.arange(0, 128, gl.SliceLayout(0, L))
    nr = gl.arange(0, 128, gl.SliceLayout(0, R))
    group_row = row_start + group_start * TILES_PER_CTA * (BLOCK_M // GQA)
    ntile = gl.minimum(TILES_PER_CTA, gl.cdiv(row_end - group_row, BLOCK_M // GQA))
    first_row = group_row + ms // GQA
    encoded = gl.load(QueryIds + hb * CAPACITY + first_row, first_row < row_end, 0)
    qhead = head * GQA + ms % GQA
    cp.async_copy_global_to_shared(
        qbufs.index(0),
        Q + (encoded // TOPK)[:, None] * qs0 + qhead[:, None] * qs1 + ds[None, :],
        mask=(first_row < row_end)[:, None],
        cache_modifier=".ca",
    )
    cp.commit_group()
    for tile in range(ntile):
        row = group_row + tile * (BLOCK_M // GQA) + ms // GQA
        active = row < row_end
        token, slot = encoded // TOPK, encoded % TOPK
        cp.wait_group(0)
        fence_async_shared()
        qsm = qbufs.index(tile % 2)
        tcgen05_mma(qsm, ksm.permute((1, 0)), acc, use_acc=False)
        tcgen05_commit(bar)
        next_row = row + BLOCK_M // GQA
        next_active = (next_row < row_end) & (tile + 1 < ntile)
        next_encoded = gl.load(QueryIds + hb * CAPACITY + next_row, next_active, 0)
        cp.async_copy_global_to_shared(
            qbufs.index((tile + 1) % 2),
            Q + (next_encoded // TOPK)[:, None] * qs0 + qhead[:, None] * qs1 + ds[None, :],
            mask=next_active[:, None],
            cache_modifier=".ca",
        )
        cp.commit_group()
        mbarrier.wait(bar, phase=0)
        scores = acc.load(R)
        scores *= qks
        qpos = gl.convert_layout(prefix + token - qstart, gl.SliceLayout(1, R))
        valid = gl.convert_layout(active, gl.SliceLayout(1, R))
        pos = block * 128 + nr
        scores = gl.where(
            valid[:, None] & (pos[None, :] <= qpos[:, None]) & (pos[None, :] < seq), scores, float("-inf")
        )
        maximum = gl.where(valid, gl.max(scores, 1), 0.0)
        prob = gl.exp2(scores - maximum[:, None])
        denom = gl.sum(prob, 1)
        ph = prob.to(PDT)
        phsm.store(gl.convert_layout(ph, PR))
        fence_async_shared()
        tcgen05_mma(phsm, vsm, acc, use_acc=False)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=1)
        out = acc.load(R) * (vs / gl.where(denom > 0, denom, 1.0))[:, None]
        # Match Partial's storage dtype before the cross-warp conversion.
        out = gl.convert_layout(out.to(gl.bfloat16), OL)
        lse = gl.convert_layout(maximum + gl.log2(denom), gl.SliceLayout(1, OL))
        outrow = (token * HEADS_Q + qhead) * TOPK + slot
        outrow = gl.convert_layout(outrow, gl.SliceLayout(1, OL))
        out_active = gl.convert_layout(active, gl.SliceLayout(1, OL))
        out_ds = gl.arange(0, 128, gl.SliceLayout(0, OL))
        gl.store(Partial + outrow[:, None] * 128 + out_ds[None, :], out, out_active[:, None], cache_modifier=".cs")
        gl.store(LSE + outrow, lse, out_active)
        encoded = next_encoded
    cp.wait_group(0)
    mbarrier.invalidate(bar)
