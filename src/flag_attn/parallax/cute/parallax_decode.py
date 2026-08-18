# Copyright (c) 2026 Zhichen Zeng.
# Reference: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd.py
# SPDX-License-Identifier: MIT
"""Parallax decoding kernel on NVIDIA Hopper (SM90).

Persistent split-K, warp-specialized streaming CuTeDSL kernel that
implements Algorithm 1 of the Parallax paper (https://arxiv.org/abs/2605.29157).
The per-tile online softmax, the per-tile composite-score state,
and the cross-split log-sum-exp merge all run inside a single kernel
launch via an atomic-last-CTA-wins finalize, with no separate
reduction-epilogue kernel.

Warp specialization (one producer warpgroup issuing TMA loads of
K_c / V_c tiles into shared memory, one consumer warpgroup driving
the QK and PV WGMMA pipeline plus the online softmax) follows the
FlashAttention 3 CuTeDSL kernel referenced above. Parallax packs
R_r alongside Q_r as row 1 of the shared-memory A operand, so the
same QK WGMMA emits both S_1 = Q_r * K_c^T * s (row 0) and
S_2 = R_r * K_c^T (row 1) into acc_QR; the PV WGMMA likewise emits
both O_1 = sum_j P_1_j * V_j (row 0) and O_2 = sum_j P_2_j * V_j
(row 1) into acc_O. The two branches share K_c / V_c reads, the
online-softmax max, and the rescaling factor — zero extra HBM
traffic and one extra register-accumulator row per CTA versus FA.

Public entry point (canonical, FA-style):
``parallax_attn_with_kvcache(q, r, k_cache, v_cache, *, page_table=None,
seqused_k=None, window_size=None, scale=None, out=None)``. Buffer ownership
is explicit (``out=None`` allocates fresh — no silent reuse), and the
finite-padding + stable-buffer (CUDA-graph) contracts are documented on that
function. ``parallax_decode(q, r, k, v, qk_scale, *, window_size_left=-1,
out=None)`` is a deprecated back-compat alias.

``window_size_left`` follows the FA2 convention: ``-1`` (default)
disables SWA; ``>= 0`` restricts the decode query to the most recent
``window_size_left`` keys. When SWA is active the split-K range is
re-allocated over the in-window tiles only, so out-of-window tiles
are never loaded by any CTA.

Restrictions:
  * SM90 (H100 / H200) only
  * bf16 or fp16 input
  * seqlen_q = 1
  * head_dim in {64, 128}
  * kv_len can be any positive integer
"""

from __future__ import annotations

import math
import operator
import os
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import nvvm
from cutlass._mlir.dialects import llvm as _mlir_llvm

import cutlass.utils.hopper_helpers as sm90_utils_basic
from flag_attn.parallax.cute._vendor import hopper_helpers as sm90_utils
from flag_attn.parallax.cute._vendor import pipeline
from flag_attn.parallax.cute._vendor import utils


@cute.jit
def _atom_acq_rel_gpu_add_u32(counter_ptr: cute.Pointer) -> Int32:
    """`atom.acq_rel.gpu.global.add.u32` — returns OLD pre-increment value.

    Uses acq_rel ordering at GPU scope: this RMW is both a release of
    prior-program-order memory ops AND an acquire of any happens-before
    release by another GPU thread. That makes it the proper sync point
    for atomic-last-CTA-wins fan-in.
    """
    ptr_i64 = counter_ptr.toint().ir_value()
    res = _mlir_llvm.inline_asm(
        Int32.mlir_type,
        [ptr_i64],
        "atom.acq_rel.gpu.global.add.u32 $0, [$1], 1;",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,  # AT&T
    )
    return Int32(res)


@cute.jit
def _st_global_cg_f32(gmem_ptr: cute.Pointer, val: Float32) -> None:
    """`st.global.cg.f32 [ptr], val` — cache-global write, bypasses L1.

    Goes directly to L2 so peer CTAs reading via `ld.global.cg` see the
    write without depending on stale L1 lines being evicted.
    """
    ptr_i64 = gmem_ptr.toint().ir_value()
    _mlir_llvm.inline_asm(
        None,
        [ptr_i64, val.ir_value()],
        "st.global.cg.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )


@cute.jit
def _ld_global_cv_f32(gmem_ptr: cute.Pointer) -> Float32:
    """`ld.global.cv.f32 dst, [ptr]` — cache-volatile load: never cached,
    always reads from L2 (or memory). Strongest hint to avoid stale
    cached values. We use .cv on the reader side (not .cg) to make
    absolutely sure peer-CTA writes are not shadowed by an L1-cached
    line from a prior call."""
    ptr_i64 = gmem_ptr.toint().ir_value()
    res = _mlir_llvm.inline_asm(
        Float32.mlir_type,
        [ptr_i64],
        "ld.global.cv.f32 $0, [$1];",
        "=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )
    return Float32(res)


_compile_cache: dict[tuple, Callable] = {}
_cute_input_cache: dict[tuple, tuple] = {}


class ParallaxDecodePersistentSplit:
    def __init__(self, dtype, head_dim: int, *, n_block_size: int = 64,
                 num_stages: int | None = None, pack_n: int = 1):
        if head_dim > 128:
            raise ValueError("SM90 TMA/WGMMA prototype requires head_dim <= 128")
        if n_block_size != 64:
            raise ValueError("SM90 TMA/WGMMA prototype currently hard-codes N=64")
        if pack_n not in (1, 2, 4, 8):
            raise ValueError(f"pack_n must be in {{1, 2, 4, 8}}, got {pack_n}")
        self.dtype = dtype
        self.head_dim = head_dim
        self.head_dim_padded = int(math.ceil(head_dim / 16) * 16)
        self.m_block_size = 64
        self.n_block_size = n_block_size
        # pack_n head-packing factor: 1 = pure MHA (one head per CTA), 2/4/8 = GQA
        # where one CTA serves pack_n query heads sharing a single KV head. The
        # head-packed layout uses rows 0..pack_n-1 (r_i=0) for S1_h and rows
        # 8..8+pack_n-1 (r_i=1) for S2_h, occupying the SAME lanes 4h..4h+3 of
        # warp 0 per head. pack_n is part of the compile-cache key.
        self.pack_n = pack_n
        # num_stages=2 (default). The pipeline race was fixed by removing
        # the deferred PV/V-release WGMMA overlap (see the serial tile loop).
        # num_stages>2 is supported but untested. PARALLAX_NUM_STAGES
        # overrides the default; num_stages is part of the compile-cache key.
        if num_stages is None:
            num_stages = int(os.environ.get("PARALLAX_NUM_STAGES", "2"))
        if num_stages < 2:
            raise ValueError(f"num_stages must be >= 2, got {num_stages}")
        self.num_stages = num_stages
        self.num_threads = 256
        self.num_threads_per_warp_group = 128

    def _get_layouts(self):
        qk_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.dtype,
                self.head_dim_padded,
            ),
            self.dtype,
        )
        v_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.dtype,
                self.head_dim_padded,
            ),
            self.dtype,
        )
        sQ_layout = cute.tile_to_shape(qk_atom, (self.m_block_size, self.head_dim_padded), (0, 1))
        sK_layout = cute.tile_to_shape(qk_atom, (self.n_block_size, self.head_dim_padded, self.num_stages), (0, 1, 2))
        sV_layout = cute.tile_to_shape(v_atom, (self.n_block_size, self.head_dim_padded, self.num_stages), (0, 1, 2))
        return sQ_layout, sK_layout, sV_layout

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            cutlass.Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self.n_block_size),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            cutlass.Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self.head_dim_padded),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mR: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        mWs_counter: cute.Tensor,
        mSeqlenK: cute.Tensor,
        softmax_scale_log2: Float32,
        stream: cuda.CUstream,
        num_k_splits: cutlass.Constexpr[int] = 1,
        window_size_left: cutlass.Constexpr[int] = -1,
        max_tiles_total: cutlass.Constexpr[int] = 0,
        cache_len: cutlass.Constexpr[int] = 0,
    ):
        mK_tma, mV_tma = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0]))
            for t in (mK, mV)
        ]

        sQ_layout, sK_layout, sV_layout = self._get_layouts()
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()

        copy_atom_kv = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            copy_atom_kv,
            mK_tma,
            cute.select(sK_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_padded),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            copy_atom_kv,
            mV_tma,
            cute.select(sV_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_padded),
        )

        self.tma_copy_k_bytes = cute.size_in_bytes(mK.element_type, cute.select(sK_layout, mode=[0, 1]))
        self.tma_copy_v_bytes = cute.size_in_bytes(mV.element_type, cute.select(sV_layout, mode=[0, 1]))

        sQ_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)], 128]
        sK_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)], 128]
        sV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sV_layout)], 128]
        p_row_struct = cute.struct.Align[cute.struct.MemRange[Float32, 128], 128]
        stats_struct = cute.struct.Align[cute.struct.MemRange[Float32, 128], 128]
        mbar_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorage:
            mbar_ptr_K: mbar_struct
            mbar_ptr_V: mbar_struct
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct
            p_row: p_row_struct
            stats: stats_struct

        self.kernel(
            mQ,
            mR,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            mWs_m,
            mWs_d1,
            mWs_d2,
            mWs_O1,
            mWs_O2,
            mWs_counter,
            mSeqlenK,
            tma_atom_K,
            tma_atom_V,
            max_tiles_total,
            softmax_scale_log2,
            sQ_layout,
            sK_layout,
            sV_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            num_k_splits,
            window_size_left,
            cache_len,
        ).launch(
            # Grid head axis = H_kv (one CTA per KV head). For pack_n==1 this is
            # the same as H_q. For pack_n>1 (GQA), one CTA emits pack_n query-head
            # rows from a single shared KV head load.
            grid=[cute.size(mQ.shape[0]), cute.size(mK.shape[2]), num_k_splits],
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mR: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        mWs_counter: cute.Tensor,
        mSeqlenK: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        max_tiles_total: cutlass.Constexpr[int],
        softmax_scale_log2: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr[Callable],
        num_k_splits: cutlass.Constexpr[int],
        window_size_left: cutlass.Constexpr[int],
        cache_len: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        batch_idx, head_idx, k_split_id = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Runtime kv_len, read per-batch from a (B,) Int32 device tensor. This
        # is the FA seqlen idiom (`SeqlenInfo.seqlen_k`): a per-batch index
        # prevents cutlass-dsl 4.1 from const-folding the load to the trace
        # value (a scalar (1,) tensor with index [0] folds; a (B,) tensor
        # indexed by runtime batch_idx does not), so one compiled kernel serves
        # every runtime kv_len — the compile-cache key (see
        # parallax_decode_cutedsl_sm90) is keyed on shapes, not active length.
        #
        # max_tiles_total is the constexpr upper bound on tiles_total used to
        # bound the inner loop (unroll=1 so the bound is a runtime
        # comparison anyway). The dispatcher rounds the launch kv_len up to a
        # bucket (typically pow2) so a serving session compiles a handful of
        # max_tiles_total variants, not one per kv_len.
        #
        # Clamp kv_len to [1, cache_len] to enforce the seqused_k contract
        # under CUDA-graph replay. cache_len is the true K/V extent
        # (k.shape[1], a constexpr), not the pow2 bucket ceiling — so
        # out-of-range seqused_k in (k.shape[1], bucket] is also clamped,
        # preventing silent attention to zero-padded rows above the real
        # cache. Use ternaries rather than cutlass.min/max — the lowered ops
        # are not signed-clean for negative operands (see the SWA pattern
        # above).
        kv_len_raw = Int32(mSeqlenK[batch_idx])
        kv_len_at_least_1 = kv_len_raw if kv_len_raw > Int32(1) else Int32(1)
        kv_len = kv_len_at_least_1 if kv_len_at_least_1 < Int32(cache_len) else Int32(cache_len)
        # Use plain Python ints for n_block_size in the arithmetic; cutlass
        # promotes them. Avoid the explicit Int32(...) wrappers — they trigger
        # a "derefine" type-narrowing the cute pipeline can't legalize.
        tiles_total = (kv_len + (self.n_block_size - 1)) // self.n_block_size
        valid_n_last_tile = kv_len - (tiles_total - 1) * self.n_block_size
        if const_expr(window_size_left >= 0):
            # NOTE: cutlass.max(Int32(0), Int32(-x)) silently returns the
            # negative value (the lowered max is not signed-clean for negative
            # operands). Use a ternary to compute max(0, kv_len - ws_left)
            # when kv_len < window_size_left.
            diff = kv_len - window_size_left
            window_start_kv = diff if diff > Int32(0) else Int32(0)
        else:
            window_start_kv = Int32(0)
        first_valid_tile = window_start_kv // self.n_block_size
        first_tile_skip = window_start_kv - first_valid_tile * self.n_block_size
        valid_tiles_total = tiles_total - first_valid_tile
        # tiles_per_split is computed from the constexpr max_tiles_total so it
        # stays constexpr → drives the grid-Z (num_k_splits) and the
        # in-merger unroll. A bucket whose actual tiles_total <= max yields
        # idle CTAs in the last split — handled by the cap below.
        max_valid_tiles: cutlass.Constexpr[int] = max_tiles_total
        tiles_per_split: cutlass.Constexpr[int] = (max_valid_tiles + num_k_splits - 1) // num_k_splits
        k_start_tile = first_valid_tile + k_split_id * tiles_per_split
        k_end_tile_uncapped = k_start_tile + tiles_per_split
        # Cap the last split at the runtime tiles_total. For kv_len shorter
        # than the bucket maximum, splits past valid_tiles_total are no-ops
        # (k_start_tile >= k_end_tile → empty range).
        k_end_tile = cutlass.min(k_end_tile_uncapped, tiles_total)
        k_start_tile = cutlass.min(k_start_tile, k_end_tile)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP_row = storage.p_row.get_tensor(cute.make_layout(64))
        sStats = storage.stats.get_tensor(cute.make_layout(128))
        sVt = utils.transpose_view(sV)

        pipeline_group_producer = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_group_consumer = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)
        pipeline_k = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_group_producer,
            consumer_group=pipeline_group_consumer,
            tx_count=self.tma_copy_k_bytes,
            init_wait=False,
        )
        pipeline_v = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_group_producer,
            consumer_group=pipeline_group_consumer,
            tx_count=self.tma_copy_v_bytes,
        )

        # Q/R fill + barrier live inside the consumer branch so the producer
        # warpgroup can start TMA loads immediately rather than waiting on an
        # all-block barrier.
        mK_cur = mK[None, None, head_idx, batch_idx]
        mV_cur = mV[None, None, head_idx, batch_idx]
        gK = cute.local_tile(mK_cur, (self.n_block_size, self.head_dim_padded), (None, 0))
        gV = cute.local_tile(mV_cur, (self.n_block_size, self.head_dim_padded), (None, 0))
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_V,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )

        # head_idx is the grid head dim, which now indexes H_kv (one CTA per kv
        # head). q_head_base = kv_head_idx * pack_n is the base offset into the
        # H_q axis of mQ / mR / mO. For pack_n=1 q_head_base == head_idx == the
        # single query head.
        q_head_base = head_idx * Int32(self.pack_n)

        if warp_idx < 4:
            cute.arch.warpgroup_reg_dealloc(24)
            producer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.num_stages)
            if warp_idx == 0:
                for n_tile in cutlass.range(k_start_tile, k_end_tile, unroll=1):
                    self._load_tile(tma_atom_K, tKgK, tKsK, pipeline_k, n_tile, producer_state)
                    self._load_tile(tma_atom_V, tVgV, tVsV, pipeline_v, n_tile, producer_state)
                    producer_state.advance()
        else:
            cute.arch.warpgroup_reg_alloc(240)
            tidx_mma = tidx - self.num_threads_per_warp_group
            self._fill_qr_smem(mQ, mR, sQ, batch_idx, q_head_base, tidx_mma)
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads_per_warp_group)
            self._mma_consumer(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mWs_m,
                mWs_d1,
                mWs_d2,
                mWs_O1,
                mWs_O2,
                mWs_counter,
                sQ,
                sK,
                sVt,
                sP_row,
                sStats,
                pipeline_k,
                pipeline_v,
                batch_idx,
                head_idx,
                q_head_base,
                k_split_id,
                tidx_mma,
                tiles_total,
                valid_n_last_tile,
                first_valid_tile,
                first_tile_skip,
                softmax_scale_log2,
                num_k_splits,
                window_size_left,
                k_start_tile,
                k_end_tile,
            )

    @cute.jit
    def _fill_qr_smem(
        self,
        mQ: cute.Tensor,
        mR: cute.Tensor,
        sQ: cute.Tensor,
        batch_idx: Int32,
        q_head_base: Int32,
        tidx: Int32,
    ) -> None:
        # Unified pack_n layout: each head h ∈ [0, pack_n) goes into rows h
        # (r_i=0) and h+8 (r_i=1) of acc — same 4 lanes 4h..4h+3 of warp 0 per
        # head. The WGMMA m64 fp32 C layout maps r_i=1 to thread-row+8, so the
        # (h, h+8) row pair lives on the SAME lane → no cross-lane shuffle in
        # the softmax / finalize. For pack_n=1 the h=0 specialization writes
        # rows 0 (Q) and 8 (R); for pack_n=8 rows 0..7 (Q_h) and 8..15 (R_h).
        # ``q_head_base = kv_head_idx * pack_n`` is the dispatcher-supplied
        # base offset into the H_q axis of mQ/mR/mO.
        for h in cutlass.range_constexpr(self.pack_n):
            if tidx < self.head_dim:
                sQ[h, tidx] = mQ[batch_idx, 0, q_head_base + h, tidx]
                sQ[h + 8, tidx] = mR[batch_idx, 0, q_head_base + h, tidx]
            if tidx >= self.head_dim and tidx < self.head_dim_padded:
                sQ[h, tidx] = self.dtype(0.0)
                sQ[h + 8, tidx] = self.dtype(0.0)
        # Other sQ rows are intentionally left uninitialized. The QK WGMMA writes
        # all 64 acc_QR rows, but only rows 0..pack_n-1 (S1 at r_i=0) and rows
        # 8..8+pack_n-1 (S2 at r_i=1 of the same lanes) are read downstream.
        # All other acc_O rows are discarded by _finalize_and_store.

    @cute.jit
    def _load_tile(
        self,
        tma_atom: cute.CopyAtom,
        tG: cute.Tensor,
        tS: cute.Tensor,
        pipe: cutlass.pipeline.PipelineAsync,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
    ) -> None:
        pipe.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            tG[None, block],
            tS[None, producer_state.index],
            tma_bar_ptr=pipe.producer_get_barrier(producer_state),
        )

    @cute.jit
    def _mma_consumer(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        mWs_counter: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP_row: cute.Tensor,
        sStats: cute.Tensor,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        batch_idx: Int32,
        head_idx: Int32,
        q_head_base: Int32,
        k_split_id: Int32,
        tidx: Int32,
        tiles_total: Int32,
        valid_n_last_tile: Int32,
        first_valid_tile: Int32,
        first_tile_skip: Int32,
        softmax_scale_log2: Float32,
        num_k_splits: cutlass.Constexpr[int],
        window_size_left: cutlass.Constexpr[int],
        k_start_tile: Int32,
        k_end_tile: Int32,
    ) -> None:
        # head_idx is the kv head index (grid head axis). q_head_base =
        # head_idx * pack_n is the output (H_q) base offset; the helpers write
        # rows q_head_base + h for h in [0, pack_n). For pack_n=1 they are
        # equal and this collapses to the original single-head path.
        # tiles_total / valid_n_last_tile / first_valid_tile / first_tile_skip
        # are runtime Int32 (derived from the per-batch kv_len). The last-tile
        # and SWA-skip masks now run unconditionally — the helpers internally
        # no-op when the runtime predicate (col >= valid_n / col < skip_n) is
        # always false, which is what happens on a clean-multiple kv_len or a
        # no-SWA / aligned-SWA call.

        wg_layout = cute.make_layout(1, stride=self.num_threads_per_warp_group)
        wg_mma_qk = tiled_mma_qk.get_slice(wg_layout(0))
        wg_mma_pv = tiled_mma_pv.get_slice(wg_layout(0))
        tSrQ = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sK))
        tOrVt = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sVt))
        acc_S_shape = tiled_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size))
        acc_O_shape = tiled_mma_pv.partition_shape_C((self.m_block_size, self.head_dim_padded))
        tOrP = cute.make_fragment(utils.convert_layout_acc_frgA(cute.make_layout(acc_S_shape)), self.dtype)
        acc_O = cute.make_fragment(acc_O_shape, Float32)
        # Empty-CTA handling: when k_start_tile == k_end_tile (a split that
        # lands beyond the runtime tiles_total, possible when kv_len's actual
        # tiles_total < the bucket ceiling), the tile loop runs zero times.
        # Without explicit init, acc_O carries register garbage that
        # _store_split_partials would emit to the workspace vector slots,
        # poisoning the merger via `0 * NaN = NaN`. Zero it only on empty
        # iterations; on non-empty iterations the PV gemm's zero_init=True
        # on the first tile overwrites acc_O — pre-filling there would create
        # an artificial read-modify-write dep that interferes with the WGMMA
        # zero_init codegen.
        if k_start_tile >= k_end_tile:
            acc_O.fill(0.0)

        m_r = -Float32.inf
        d1 = Float32(0.0)
        d2 = Float32(0.0)
        consumer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.num_stages)

        # Strictly serial WGMMA tile loop: each iteration retires both QK and
        # PV before the next tile begins, and releases K/V immediately after
        # the WGMMA that consumed them. The earlier deferred-release variant
        # (PV_{t-1} overlapped with QK_t and a joint wait_group with deferred
        # V release) was a real, latent race at large B*H — corrupted ~1-4 /
        # 180 output rows on launches that fill ≥1 SM wave, regardless of
        # num_stages. The WGMMA scheduler still pipelines QK/PV without the
        # explicit overlap, so the serial form has no measurable cost.
        for t in cutlass.range(k_start_tile, k_end_tile, unroll=1):
            acc_QR = cute.make_fragment(acc_S_shape, Float32)
            pipeline_k.consumer_wait(consumer_state, pipeline_k.consumer_try_wait(consumer_state))
            sm90_utils.gemm(
                tiled_mma_qk,
                acc_QR,
                tSrQ,
                tSrK[None, None, None, consumer_state.index],
                zero_init=True,
                wg_wait=0,  # Must be >=0: warpgroup-collective wait before release
            )
            pipeline_k.consumer_release(consumer_state)

            # Runtime mask predicates: pass valid_n_last_tile (could be == B_c
            # on a clean-multiple kv_len → mask is a no-op since
            # col >= n_block_size is impossible) and first_tile_skip (zero on
            # no-SWA / aligned-SWA → mask no-op). The const_expr guards used
            # to elide these calls entirely; with runtime kv_len they always
            # fire but their bodies short-circuit cheaply.
            valid_n_t = valid_n_last_tile if t == tiles_total - 1 else Int32(self.n_block_size)
            self._apply_last_tile_mask(acc_QR, tiled_mma_qk, valid_n_t)
            if const_expr(window_size_left >= 0):
                skip_n_t = first_tile_skip if t == first_valid_tile else Int32(0)
                self._apply_first_tile_skip_mask(acc_QR, tiled_mma_qk, skip_n_t)

            m_r, d1, d2, alpha = self._row0_online_softmax_and_make_p(
                acc_QR,
                tiled_mma_qk,
                sP_row,
                sStats,
                m_r,
                d1,
                d2,
                softmax_scale_log2,
            )
            if t > k_start_tile:
                self._scale_output_rows01(acc_O, alpha, tiled_mma_pv)
            tOrP_acc = cute.make_tensor(acc_QR.iterator, utils.convert_layout_acc_frgA(acc_QR.layout))
            utils.cvt_f16(tOrP_acc, tOrP)

            pipeline_v.consumer_wait(consumer_state, pipeline_v.consumer_try_wait(consumer_state))
            sm90_utils.gemm(
                tiled_mma_pv,
                acc_O,
                tOrP,
                tOrVt[None, None, None, consumer_state.index],
                zero_init=(t == k_start_tile),
                wg_wait=0,  # Must be >=0: warpgroup-collective wait before release
            )
            pipeline_v.consumer_release(consumer_state)
            consumer_state.advance()

        # Finalize: two paths, picked at JIT time on num_k_splits.
        #
        # num_k_splits == 1: this CTA owns the entire (B, H) row, so it
        # casts in-register and writes the bf16/fp16 row directly to mO.
        # No HBM workspace round-trip, no fence, no atomic, no merge.
        #
        # num_k_splits > 1: every CTA writes per-CTA un-normalized fp32
        # partials to HBM workspace, then atomic-last-CTA-wins picks the
        # merger:
        #   1. consumer-warpgroup-barrier (all 128 threads finished writes)
        #   2. fence_acq_rel_gpu — publish partial writes to peer CTAs
        #   3. tidx==0 atomic_add(counter[B, H], 1) returns OLD; the CTA
        #      that observes OLD == num_k_splits - 1 is the last arriver.
        #      Broadcast that via sStats[0].
        #   4. consumer-warpgroup-barrier (every thread sees the broadcast)
        #   5. last CTA only: 128 threads each handle one output column,
        #      read partials from HBM, run LSE merge in fp32, cast + store
        #      to mO. Then reset counter[B, H] = 0 for the next call.
        if const_expr(num_k_splits == 1):
            self._finalize_and_store(
                acc_O,
                d2,
                d1,
                tiled_mma_pv,
                mO,
                sP_row,
                batch_idx,
                q_head_base,
            )
        else:
            # Split-K partials: each CTA writes (m, d1, d2, O) to per-head
            # workspace slots. For pack_n>1 the CTA loops over pack_n heads
            # (slots head_idx..head_idx+pack_n-1; q_head_base=kv_head*pack_n
            # keeps them collision-free). The merger loops likewise.
            # pack_n>1 + num_k_splits>1 IS supported.
            self._store_split_partials(
                acc_O,
                d2,
                m_r,
                d1,
                softmax_scale_log2,
                tiled_mma_pv,
                mWs_m,
                mWs_d1,
                mWs_d2,
                mWs_O1,
                mWs_O2,
                batch_idx,
                q_head_base,
                k_split_id,
            )
            self._fused_epilogue_atomic_last_wins(
                mO,
                mWs_m,
                mWs_d1,
                mWs_d2,
                mWs_O1,
                mWs_O2,
                mWs_counter,
                sStats,
                batch_idx,
                q_head_base,
                tidx,
                num_k_splits,
            )

    @cute.jit
    def _row0_online_softmax_and_make_p(
        self,
        acc_qr: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        sP_row: cute.Tensor,
        sStats: cute.Tensor,
        m_r: Float32,
        d1: Float32,
        d2: Float32,
        softmax_scale_log2: Float32,
    ) -> tuple[Float32, Float32, Float32, Float32]:
        acc_mn = utils.make_acc_tensor_mn_view(acc_qr)
        thr_mma = tiled_mma_qk.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        tidx = cute.arch.thread_idx()[0] - self.num_threads_per_warp_group
        lane = tidx % 32
        warp = tidx // 32

        # Unified pack_n layout: S1 lives at r_i=0 (rows 0..pack_n-1) and S2 lives
        # at r_i=1 (rows 8..8+pack_n-1) of the SAME 4 lanes per head in warp 0.
        # For pack_n=1, only h=0 is live: S1 at r_i=0 lanes 0..3, S2 at r_i=1 of
        # the same 4 lanes — no cross-lane shuffle needed (replaces the old
        # lane-4 shuffle). width=4 warp_reduce keeps each head's 4 lanes in
        # agreement; dead lanes (warp 0 lanes >= 4*pack_n and warps 1..3) carry
        # stale m_r/d1/d2 that the reduce ignores by construction.
        alpha = Float32(1.0)
        m_r_new = m_r
        d1_new = d1
        d2_new = d2
        m_cur = m_r
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            # cS_mn[0, c_i][0] is this lane's m-coordinate at r_i=0. For lanes
            # that carry a live head, this is in [0, pack_n) (one of the S1
            # rows). For pack_n=1 this is just == 0.
            if cS_mn[0, c_i][0] < self.pack_n:
                qk_val = acc_mn[0, c_i]
                m_cur = qk_val if qk_val > m_cur else m_cur
        m_cur = utils.warp_reduce(m_cur, cute.arch.fmax, width=4)
        m_r_safe = Float32(0.0) if m_cur == -Float32.inf else m_cur
        m_r_new = m_r_safe
        alpha = utils.exp2f((m_r - m_r_safe) * softmax_scale_log2)

        tile_sum = Float32(0.0)
        tile_c = Float32(0.0)
        scaled_max = m_r_safe * softmax_scale_log2
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            if cS_mn[0, c_i][0] < self.pack_n:
                # S1 → P1 = exp(S1 * scale_log2 - scaled_max), in-place.
                qk_exp = utils.exp2f(acc_mn[0, c_i] * softmax_scale_log2 - scaled_max)
                tile_sum += qk_exp
                acc_mn[0, c_i] = qk_exp
                # S2 → P2 = P1 * S2. r_i=1 of the SAME lane → no shuffle.
                wr = qk_exp * acc_mn[1, c_i]
                tile_c += wr
                acc_mn[1, c_i] = wr
        tile_sum = utils.warp_reduce(tile_sum, operator.add, width=4)
        d1_new = d1 * alpha + tile_sum
        tile_c = utils.warp_reduce(tile_c, operator.add, width=4)
        d2_new = d2 * alpha + tile_c

        return m_r_new, d1_new, d2_new, alpha

    @cute.jit
    def _apply_last_tile_mask(
        self,
        acc_qr: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        valid_n: Int32,
    ) -> None:
        """Set S_1 to -inf for KV columns beyond ``valid_n`` (row 0 only).

        Used to support kv_len that is not a multiple of B_c: the last
        tile may cover fewer than B_c valid KV positions, and the
        excess columns must be masked out before softmax so they
        contribute 0 to P_1, d_1, d_2, O_1, O_2. TMA's default OOB
        behaviour delivers zeros for the out-of-bounds K/V slots, but
        zero attention logits would still receive a non-zero softmax
        weight, so we explicitly drive S_1 to -inf instead.

        Only row 0 (S_1 = Q_r * K_c^T) is masked. Row 1 (S_2 =
        R_r * K_c^T) is left as TMA delivered it (zeros for OOB K),
        which gives P_2 = P_1 * S_2 = 0 for masked columns. Forcing
        S_2 to -inf as well would produce 0 * -inf = NaN inside the
        composite-score accumulation.

        Predicate is per-c_i (constexpr column coordinate) compared to
        the runtime ``valid_n``; when ``valid_n == B_c`` the check is
        statically false everywhere and the compiler folds it away.
        """
        # Mask S1 (r_i=0 rows 0..pack_n-1) for all live heads; S2 untouched
        # (see the docstring above for the row-only / 0*-inf=NaN contract).
        acc_mn = utils.make_acc_tensor_mn_view(acc_qr)
        thr_mma = tiled_mma_qk.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            if cS_mn[0, c_i][0] < self.pack_n and cS_mn[0, c_i][1] >= valid_n:
                acc_mn[0, c_i] = -Float32.inf

    @cute.jit
    def _apply_first_tile_skip_mask(
        self,
        acc_qr: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        skip_n: Int32,
    ) -> None:
        """Set S_1 to -inf for the first ``skip_n`` columns of the tile (row 0 only).

        Sliding-window dual of ``_apply_last_tile_mask``: the lowest
        valid tile may cover positions both inside and outside the
        sliding window, with the in-window positions at the *high* end
        of the tile. We mask the low end (positions before the window)
        and let the softmax see the rest. Same row-0-only contract as
        the last-tile mask to avoid 0 * -inf NaNs in the composite
        branch.
        """
        # Mask S1 (r_i=0 rows 0..pack_n-1) for all live heads. See
        # _apply_last_tile_mask for the row-only contract rationale.
        acc_mn = utils.make_acc_tensor_mn_view(acc_qr)
        thr_mma = tiled_mma_qk.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            if cS_mn[0, c_i][0] < self.pack_n and cS_mn[0, c_i][1] < skip_n:
                acc_mn[0, c_i] = -Float32.inf

    @cute.jit
    def _scale_output_rows01(self, acc: cute.Tensor, alpha: Float32, tiled_mma: cute.TiledMma) -> None:
        # Unified pack_n layout: O1_h (= Σ P_1 V) lives at r_i=0 rows 0..pack_n-1,
        # O2_h (= Σ P_2 V) lives at r_i=1 rows 8..8+pack_n-1, on the SAME lanes
        # 4h..4h+3 of warp 0 per head. Per-lane alpha is each head's alpha (the
        # softmax kept the 4 lanes of a head in agreement via width=4 reduce).
        acc_mn = utils.make_acc_tensor_mn_view(acc)
        thr_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        cO_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            if cO_mn[0, c_i][0] < self.pack_n:
                # Same lane: r_i=0 is O1_h, r_i=1 is O2_h. Scale both.
                acc_mn[0, c_i] = acc_mn[0, c_i] * alpha
                acc_mn[1, c_i] = acc_mn[1, c_i] * alpha

    @cute.jit
    def _store_split_partials(
        self,
        acc_o: cute.Tensor,
        d2: Float32,
        m_r: Float32,
        d1: Float32,
        softmax_scale_log2: Float32,
        tiled_mma: cute.TiledMma,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        k_split_id: Int32,
    ) -> None:
        # Workspace tensors are shaped (B, H, num_k_splits) for scalars and
        # (B, H, num_k_splits, head_dim) for vector partials, so the kernel
        # can index them with (batch_idx, head_idx, k_split_id[, d]) directly.
        # The underlying memory layout is identical to the dispatcher's
        # (num_bh, num_k_splits[, head_dim]) flat allocation -- the .view()
        # is purely a CuTe-side reshape for clean indexing.
        acc_mn = utils.make_acc_tensor_mn_view(acc_o)
        thr_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        cO_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
        tidx = cute.arch.thread_idx()[0] - self.num_threads_per_warp_group
        # Scalar partials: one writer per CTA (lane 0 of warp 0 holds the
        # canonical reduced value because width=8 warp_reduce broadcasts
        # within the first 8 lanes of warp 0). The kernel carries m_r in
        # raw QK units (unscaled); rescale into natural-base (matching
        # the fp32 reference's m = (qk * qk_scale).max()) so the
        # cross-split exp() merge in `_merge_and_store_inkernel` uses the
        # same units as the per-split d1/d2 partials.
        _LN2 = Float32(0.6931471805599453)
        # Partials writes use `st.global.cg.f32` (bypass L1 -> L2); the merger
        # reads them with `ld.global.cv.f32` (cache-volatile) so it never sees
        # a stale L1 line.
        # Unified pack_n layout: m_r/d1/d2 are per-lane scalars that the
        # softmax helper's width=4 warp_reduce broadcasts within each head's
        # 4 lanes (lanes 4h..4h+3 of warp 0 share head h's reduced state).
        # So lane 4h is the canonical writer for head h's scalar partials at
        # workspace slot (batch, head_idx + h, k_split_id). For pack_n=1 this
        # collapses to the original `tidx == 0` path.
        for h in cutlass.range_constexpr(self.pack_n):
            if tidx == 4 * h:
                _st_global_cg_f32(utils.elem_pointer(mWs_m,  (batch_idx, head_idx + h, k_split_id)), m_r * softmax_scale_log2 * _LN2)
                _st_global_cg_f32(utils.elem_pointer(mWs_d1, (batch_idx, head_idx + h, k_split_id)), d1)
                _st_global_cg_f32(utils.elem_pointer(mWs_d2, (batch_idx, head_idx + h, k_split_id)), d2)
        # Vector partials: O1 lives at r_i=0 (rows 0..pack_n-1), O2 at r_i=1
        # (rows 8..8+pack_n-1), on the same lanes. For pack_n=1 only row 0
        # contributes.
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            col = cO_mn[0, c_i][1]
            row = cO_mn[0, c_i][0]
            if row < self.pack_n and col < self.head_dim:
                _st_global_cg_f32(utils.elem_pointer(mWs_O1, (batch_idx, head_idx + row, k_split_id, col)), acc_mn[0, c_i])
                _st_global_cg_f32(utils.elem_pointer(mWs_O2, (batch_idx, head_idx + row, k_split_id, col)), acc_mn[1, c_i])

    @cute.jit
    def _finalize_and_store(
        self,
        acc_o: cute.Tensor,
        d2: Float32,
        denom: Float32,
        tiled_mma: cute.TiledMma,
        mO: cute.Tensor,
        sP_row: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
    ) -> None:
        # Unified pack_n layout: for each lane in warp 0 with row position r =
        # cO_mn[0, c_i][0] in [0, pack_n), O1_r is at r_i=0 (acc_mn[0, c_i]) and
        # O2_r is at r_i=1 of the SAME lane (acc_mn[1, c_i]). Each head's d1/d2
        # are the per-lane scalars (consistent across the 4 lanes of the head).
        # ``head_idx`` here is already the *output* head index (kv_head * pack_n
        # + h for pack_n>=2; just head_idx for pack_n=1). The dispatcher passes
        # head_base; this helper writes head_base + row.
        acc_mn = utils.make_acc_tensor_mn_view(acc_o)
        thr_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0] - self.num_threads_per_warp_group)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        cO_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
        inv_d1 = cute.arch.rcp_approx(denom)
        c_norm = d2 * inv_d1
        for c_i in cutlass.range(cute.size(acc_mn, mode=[1]), unroll_full=True):
            row = cO_mn[0, c_i][0]
            col = cO_mn[0, c_i][1]
            if row < self.pack_n and col < self.head_dim:
                O1_d = acc_mn[0, c_i] * inv_d1
                O2_d = acc_mn[1, c_i] * inv_d1
                o_fp32 = O1_d + c_norm * O1_d - O2_d
                # Run the composite cancellation in fp32, cast at the store —
                # the bf16/fp16 cast on (O1 + c*O1 - O2) keeps more precision
                # than casting the per-term inputs first.
                mO[batch_idx, 0, head_idx + row, col] = mO.element_type(o_fp32)

    @cute.jit
    def _fused_epilogue_atomic_last_wins(
        self,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        mWs_counter: cute.Tensor,
        sStats: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        tidx: Int32,
        num_k_splits: cutlass.Constexpr[int],
    ) -> None:
        # Runs on the consumer warpgroup only (128 threads, tidx in [0, 128)).
        # Pre-condition: _store_split_partials just wrote this CTA's partials
        # to the HBM workspace.
        #
        # Protocol:
        #   1. Consumer-wg barrier to ensure all 128 threads finished their
        #      per-CTA partial writes.
        #   2. fence_acq_rel_gpu so other CTAs observe our partials before
        #      our atomic-inc.
        #   3. tidx==0 atomic-adds 1 to counter[batch, head] (i32 in HBM),
        #      reads back OLD value. If OLD == num_k_splits-1 we're the
        #      last arriver for this (B, H). Broadcast that decision via
        #      sStats[0] (1.0=last, 0.0=not).
        #   4. Consumer-wg barrier so all threads see the broadcast.
        #   5. If last: every thread does the LSE-style merge in fp32 (each
        #      thread handles one output column when tidx < head_dim), then
        #      writes mO[batch, 0, head, :] and resets the counter so the
        #      next call starts from 0.
        # Release-side fence: every consumer thread publishes its partial
        # writes (mWs_O1/Rv, plus tidx==0's scalars) so peer CTAs that
        # later acquire from our atomic-inc observe those writes. The
        # barrier first ensures every thread has issued its store; the
        # fence then makes those stores GPU-visible.
        cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads_per_warp_group)
        cute.arch.fence_acq_rel_gpu()
        if tidx == 0:
            counter_ptr = utils.elem_pointer(mWs_counter, (batch_idx, head_idx))
            # Inline PTX `atom.acq_rel.gpu.global.add.u32`: the acq_rel
            # ordering establishes a sync-with relationship with the
            # prior fence so peer CTAs that observe our increment also
            # observe our partials.
            old = _atom_acq_rel_gpu_add_u32(counter_ptr)
            is_last = old == Int32(num_k_splits - 1)
            sStats[0] = Float32(1.0) if is_last else Float32(0.0)
        cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads_per_warp_group)
        is_last_cta = sStats[0] > Float32(0.5)
        if is_last_cta:
            # Acquire-side fence: every consumer thread is about to read
            # mWs_O1/Rv[:, :, s, tidx] for s in [0, num_k_splits) and
            # needs its own acquire to see the prior CTAs' released writes.
            # The CTA-level barrier above propagates sStats[0] but not
            # global-memory acquire semantics for arbitrary peer-CTA stores.
            cute.arch.fence_acq_rel_gpu()
            self._merge_and_store_inkernel(
                mO,
                mWs_m,
                mWs_d1,
                mWs_d2,
                mWs_O1,
                mWs_O2,
                batch_idx,
                head_idx,
                tidx,
                num_k_splits,
            )
            # Reset counter so the next call sees 0 without needing a
            # separate reset kernel. One thread does the regular store —
            # we're the only CTA still touching this counter (we're the
            # last for this (B, H)).
            if tidx == 0:
                mWs_counter[batch_idx, head_idx] = Int32(0)

    @cute.jit
    def _merge_and_store_inkernel(
        self,
        mO: cute.Tensor,
        mWs_m: cute.Tensor,
        mWs_d1: cute.Tensor,
        mWs_d2: cute.Tensor,
        mWs_O1: cute.Tensor,
        mWs_O2: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        tidx: Int32,
        num_k_splits: cutlass.Constexpr[int],
    ) -> None:
        # Streaming final reduction (Algorithm 1, cross-split). Each of
        # the 128 consumer threads handles ONE output column (if
        # tidx < head_dim). Scalars (m, d1, d2) are read by every thread
        # — that's num_k_splits * 3 redundant HBM reads but they're all
        # L2-cached (we just wrote them) and broadcast-friendly.
        #
        # Math:
        #   m_global  = max_s m_s                       (natural-base)
        #   w[s]      = exp(m_s - m_global)
        #   d1_global = Σ_s d1_s * w[s]
        #   d2_global = Σ_s d2_s * w[s]
        #   O1_d      = (Σ_s O1_{s,d} * w[s]) / d1_global
        #   O2_d      = (Σ_s O2_{s,d} * w[s]) / d1_global
        #   c_norm    = d2_global / d1_global
        #   out[d]    = (1 + c_norm) * O1_d - O2_d
        _LOG2_E: cutlass.Constexpr[float] = 1.4426950408889634
        # Partials reads use `ld.global.cv.f32` (cache-volatile) so we bypass
        # stale L1 and pick up the L2-resident writes peer CTAs published
        # through the acq_rel atomic + their `st.global.cg` stores.
        # Loop over the pack_n heads this CTA group merged. For pack_n=1 this
        # is one iteration at head_idx; for pack_n=8 it merges all eight
        # query heads associated with a single kv_head into mO.
        for h in cutlass.range_constexpr(self.pack_n):
            head_h = head_idx + h
            m_global = -Float32.inf
            for s in cutlass.range(num_k_splits, unroll_full=True):
                m_s = _ld_global_cv_f32(utils.elem_pointer(mWs_m, (batch_idx, head_h, s)))
                m_global = m_s if m_s > m_global else m_global

            d1_global = Float32(0.0)
            d2_global = Float32(0.0)
            O1_acc = Float32(0.0)
            O2_acc = Float32(0.0)
            for s in cutlass.range(num_k_splits, unroll_full=True):
                m_s = _ld_global_cv_f32(utils.elem_pointer(mWs_m,  (batch_idx, head_h, s)))
                d1_s = _ld_global_cv_f32(utils.elem_pointer(mWs_d1, (batch_idx, head_h, s)))
                d2_s = _ld_global_cv_f32(utils.elem_pointer(mWs_d2, (batch_idx, head_h, s)))
                # exp(m_s - m_global) = exp2((m_s - m_global) * log2(e)).
                # m is stored in natural base (* ln2), see _store_split_partials.
                w = utils.exp2f((m_s - m_global) * Float32(_LOG2_E))
                d1_global += d1_s * w
                d2_global += d2_s * w
                if tidx < self.head_dim:
                    O1_s = _ld_global_cv_f32(utils.elem_pointer(mWs_O1, (batch_idx, head_h, s, tidx)))
                    O2_s = _ld_global_cv_f32(utils.elem_pointer(mWs_O2, (batch_idx, head_h, s, tidx)))
                    O1_acc += O1_s * w
                    O2_acc += O2_s * w
            inv_d1 = cute.arch.rcp_approx(d1_global)
            c_norm = d2_global * inv_d1
            if tidx < self.head_dim:
                O1_d = O1_acc * inv_d1
                O2_d = O2_acc * inv_d1
                o_fp32 = O1_d + c_norm * O1_d - O2_d
                mO[batch_idx, 0, head_h, tidx] = mO.element_type(o_fp32)


def _to_cute_tensor(t: torch.Tensor):
    return from_dlpack(t.detach(), assumed_align=16)


def _cached_cute_tensor(t: torch.Tensor):
    """from_dlpack wrapper + memoization keyed on (data_ptr, shape, dtype).

    The first-seen torch.Tensor is strong-ref'd in the cache so the
    underlying memory outlives the returned cute tensor view.
    """
    key = (t.data_ptr(), tuple(t.shape), t.dtype)
    entry = _cute_input_cache.get(key)
    if entry is not None:
        return entry[1]
    ct = _to_cute_tensor(t)
    _cute_input_cache[key] = (t, ct)
    return ct


def parallax_decode_cutedsl_sm90(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float,
    *,
    ws: dict | None = None,
    num_k_splits: int = 1,
    window_size_left: int = -1,
    pack_n: int = 1,
    max_tiles_total: int | None = None,
    seqused_k: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.ndim != 4 or q.shape[1] != 1:
        raise ValueError("expected q/r shape (B, 1, H, D)")
    if q.shape != r.shape or k.shape != v.shape:
        raise ValueError("q/r or k/v shape mismatch")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError(f"incompatible q/k shapes: {q.shape} vs {k.shape}")
    if pack_n not in (1, 2, 4, 8):
        raise ValueError(f"pack_n must be 1, 2, 4, or 8 (got {pack_n})")
    if q.shape[2] != k.shape[2] * pack_n:
        raise ValueError(
            f"GQA shape mismatch: H_q={q.shape[2]} must equal H_kv*pack_n="
            f"{k.shape[2] * pack_n}"
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("SM90 TMA/WGMMA CuTe backend requires fp16/bf16 inputs")
    if not q.is_cuda:
        raise ValueError("SM90 TMA/WGMMA CuTe backend requires CUDA tensors")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise RuntimeError("SM90 TMA/WGMMA CuTe backend requires compute capability 9.x")
    if q.shape[-1] > 128:
        raise ValueError("SM90 TMA/WGMMA CuTe backend currently supports head_dim <= 128")
    if k.shape[1] <= 0:
        raise ValueError("kv_len must be positive")
    if ws is None:
        raise ValueError(
            "Parallax always uses the in-kernel fused epilogue and requires the "
            "caller to supply a full workspace including the atomic counter. "
            "Use parallax_decode(...) (the public dispatcher) which builds the "
            "workspace via _get_workspace()."
        )

    q = q.contiguous()
    r = r.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    head_dim = q.shape[-1]
    kv_len = k.shape[1]
    # max_tiles_total: constexpr bucket bound for the runtime tile loop. Defaults
    # to ceil(kv_len / 64), so non-bucketed callers still get a (kv_len-keyed)
    # compile — but the runtime path is correct for any kv_len <= bucket.
    # Callers that want compile-once-across-lengths pre-compute a stable
    # max_tiles_total (e.g., the bucket ceiling) and pass it explicitly.
    if max_tiles_total is None:
        max_tiles_total = (kv_len + 63) // 64
    if kv_len > max_tiles_total * 64:
        raise ValueError(
            f"kv_len={kv_len} exceeds max_tiles_total*64={max_tiles_total*64}; "
            "pick a larger bucket"
        )
    # The fused path writes the merged bf16/fp16 row to mO from the last
    # CTA. mO MUST be the caller's output tensor (correct dtype + layout).
    #
    # Buffer ownership is explicit (no silent module-level reuse):
    #   * out=None  -> allocate a fresh tensor, returned to the caller who
    #     then owns it. Wrapped with the *uncached* _to_cute_tensor so we do
    #     not pin every transient output forever (that cache would leak on
    #     the hot path).
    #   * out given -> caller-owned. Wrapped via _cached_cute_tensor, which
    #     pins the buffer and memoizes its cute view, giving a stable address
    #     suitable for CUDA-graph capture and serving (see the stable-buffer
    #     contract in parallax_attn_with_kvcache's docstring).
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        out_t_cached = _to_cute_tensor(out)
    else:
        out_t_cached = _cached_cute_tensor(out)

    dtype = cutlass.BFloat16 if q.dtype is torch.bfloat16 else cutlass.Float16
    kernel = ParallaxDecodePersistentSplit(dtype, head_dim, pack_n=pack_n)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    scale_log2 = float(qk_scale) * math.log2(math.e)

    q_t, r_t, k_t, v_t = [_cached_cute_tensor(t) for t in (q, r, k, v)]
    out_t = out_t_cached

    # Workspace plumbing: reshape (num_bh, S[, D]) → (B, H_q, S[, D]) so the
    # kernel can index with (batch_idx, q_head_base + h, k_split_id[, d]). The
    # counter is reshaped to (B, H_q) i32 (over-allocated by pack_n× for GQA;
    # the kernel only writes the strided subset at slots kv_head*pack_n).
    B, H_q = q.shape[0], q.shape[2]
    ws_use = {
        "m":       ws["m"].view(B, H_q, num_k_splits),
        "d1":      ws["d1"].view(B, H_q, num_k_splits),
        "d2":      ws["d2"].view(B, H_q, num_k_splits),
        "O1":      ws["O1"].view(B, H_q, num_k_splits, head_dim),
        "O2":      ws["O2"].view(B, H_q, num_k_splits, head_dim),
        "counter": ws["counter"].view(B, H_q),
    }
    ws_m_t       = _cached_cute_tensor(ws_use["m"])
    ws_d1_t      = _cached_cute_tensor(ws_use["d1"])
    ws_d2_t      = _cached_cute_tensor(ws_use["d2"])
    ws_O1_t      = _cached_cute_tensor(ws_use["O1"])
    ws_O2_t      = _cached_cute_tensor(ws_use["O2"])
    ws_counter_t = _cached_cute_tensor(ws_use["counter"])

    # Per-batch seqlen tensor (B,) Int32. Copy caller-provided seqused_k
    # into the internal buffer to avoid unbounded _cute_input_cache leakage
    # (each fresh tensor has a new data_ptr and pins its cute view forever).
    if seqused_k is not None:
        seqlen_buf = _get_seqlen_buf(B, q.device)
        seqlen_buf.copy_(seqused_k)
    else:
        seqlen_buf = _get_seqlen_buf(B, q.device, kv_len)
    seqlen_t = _cached_cute_tensor(seqlen_buf)

    # The kernel reads per-batch kv_len at runtime from seqlen_t, so the
    # active length is NOT in the cache key. The K/V tensor shape IS in the
    # key, however — the TMA descriptors built inside `__call__` bake the
    # tensor extents (specifically k.shape[1]) into the compiled kernel.
    # Serving workflow: pre-allocate K/V at the cache ceiling, pass
    # seqused_k for the per-step active length. Same K shape → one compile
    # → kv_len varies for free.
    # max_tiles_total is also in the key — it bounds the constexpr
    # tiles_per_split / merger unroll.
    key = (q.dtype, out.dtype, head_dim, q.shape[0], q.shape[2], k.shape[1],
           num_k_splits, window_size_left, kernel.num_stages,
           pack_n, max_tiles_total)
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(
            kernel,
            q_t, r_t, k_t, v_t, out_t,
            ws_m_t, ws_d1_t, ws_d2_t, ws_O1_t, ws_O2_t, ws_counter_t,
            seqlen_t, scale_log2, stream,
            num_k_splits,
            window_size_left,
            max_tiles_total,
            k.shape[1],
        )
    _compile_cache[key](
        q_t, r_t, k_t, v_t, out_t,
        ws_m_t, ws_d1_t, ws_d2_t, ws_O1_t, ws_O2_t, ws_counter_t,
        seqlen_t,
        scale_log2, stream,
    )
    return out


# Per-batch seqlen buffer cache: one (B,) Int32 tensor per (B, device). Reused
# across calls — only `fill_` is needed each call, no allocation. Stable
# address makes the buffer CUDA-graph-friendly (mirrors _WORKSPACE_CACHE).
_SEQLEN_BUF_CACHE: dict[tuple, torch.Tensor] = {}

def _get_seqlen_buf(B: int, device: torch.device, kv_len: int | None = None) -> torch.Tensor:
    device_index = device.index if device.index is not None else (
        torch.cuda.current_device() if device.type == "cuda" else -1
    )
    key = (B, device_index)
    t = _SEQLEN_BUF_CACHE.get(key)
    if t is None:
        t = torch.empty(B, dtype=torch.int32, device=device)
        _SEQLEN_BUF_CACHE[key] = t
    if kv_len is not None:
        t.fill_(kv_len)
    return t


# Wave-aware split-count rounding (the in-kernel merge unrolls over the
# split dim, so we need a power-of-two count).
def _round_to_pow2_wave_aware(s: int, num_bh: int, num_sms: int) -> int:
    if s <= 1:
        return 1
    if (s & (s - 1)) == 0:
        return s
    next_pow2 = 1
    while next_pow2 < s:
        next_pow2 <<= 1
    prev_pow2 = next_pow2 >> 1
    waves_prev = (num_bh * prev_pow2 + num_sms - 1) // num_sms
    waves_next = (num_bh * next_pow2 + num_sms - 1) // num_sms
    if waves_prev < waves_next:
        return prev_pow2
    return next_pow2


def _choose_num_k_splits(num_bh: int, kv_len: int, num_sms: int) -> int:
    """Pick a split count S over the L axis so the (B, H, S) grid fits one wave."""
    if num_bh >= num_sms:
        return 1
    K_SEG = 64  # smallest L slice we will ever assign to a single CTA
    MAX_K_SPLITS = 256
    if kv_len < K_SEG:
        return 1
    max_splits = min(kv_len // K_SEG, MAX_K_SPLITS)
    needed = math.ceil(num_sms / num_bh)
    num_k_splits = max(1, min(needed, max_splits))
    tiles_total = (kv_len + 63) // 64
    return min(num_k_splits, max(1, tiles_total))


# Module-level cache for the per-split HBM workspace. Keyed by the launch
# shape so distinct (num_bh, S, head_dim, device) configurations get their
# own tensors and we never reallocate on the hot path.
_WORKSPACE_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def _get_workspace(num_bh: int,
                   num_k_splits: int,
                   head_dim: int,
                   device: torch.device,
                   dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """fp32 workspace for the cross-split log-sum-exp merge.

    Layout (per-split partials of Algorithm 1):
      m       : (num_bh, S)              per-split running max
      d1      : (num_bh, S)              per-split d_1 = sum_j P_1_j
      d2      : (num_bh, S)              per-split d_2 = sum_j P_2_j
      O1      : (num_bh, S, head_dim)    per-split O_1 = sum_j P_1_j v_j
      O2      : (num_bh, S, head_dim)    per-split O_2 = sum_j P_2_j v_j
      counter : (num_bh,) i32            atomic last-CTA detector

    The counter starts at zero; every CTA atomic-adds 1 once its partials are
    published; the CTA that reads OLD == S - 1 is elected the merger. It runs
    the log-sum-exp merge plus the (1 + d_2/d_1) O_1/d_1 - O_2/d_1
    cancellation in fp32 and writes the bf16/fp16 output row. The merger
    also resets the counter to zero so the next call starts clean.
    """
    device_index = device.index if device.index is not None else (
        torch.cuda.current_device() if device.type == "cuda" else -1
    )
    key = (num_bh, num_k_splits, head_dim, device_index)
    cached = _WORKSPACE_CACHE.get(key)
    if cached is not None:
        return cached
    ws = {
        "m":       torch.full((num_bh, num_k_splits),            -float("inf"), dtype=dtype, device=device),
        "d1":      torch.zeros((num_bh, num_k_splits),           dtype=dtype, device=device),
        "d2":      torch.zeros((num_bh, num_k_splits),           dtype=dtype, device=device),
        "O1":      torch.zeros((num_bh, num_k_splits, head_dim), dtype=dtype, device=device),
        "O2":      torch.zeros((num_bh, num_k_splits, head_dim), dtype=dtype, device=device),
        "counter": torch.zeros((num_bh,),                        dtype=torch.int32, device=device),
    }
    _WORKSPACE_CACHE[key] = ws
    return ws


def _decode_core(q: torch.Tensor,
                 r: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 scale: float,
                 *,
                 window_size_left: int = -1,
                 seqused_k: torch.Tensor | None = None,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """Shared decode core: validation + split-K selection + dispatch.

    Both public entries (:func:`parallax_attn_with_kvcache` and the deprecated
    :func:`parallax_decode`) funnel through here, so there is exactly one
    launch path. Buffer ownership is explicit: ``out=None`` causes the
    dispatcher to allocate a fresh tensor (no silent module-level reuse);
    a provided ``out`` must match ``(B, 1, H, D)`` and ``q.dtype``.
    """
    assert q.is_contiguous() and r.is_contiguous() and k.is_contiguous() and v.is_contiguous(), (
        "q/r/k/v must be contiguous"
    )
    assert q.dtype in (torch.bfloat16, torch.float16), (
        f"parallax decode requires bf16 or fp16 input, got {q.dtype}"
    )
    assert q.ndim == 4 and q.shape[1] == 1, (
        f"expected q/r shape (B, 1, H, D), got {tuple(q.shape)}"
    )
    B, _, H, D = q.shape
    assert r.shape == (B, 1, H, D), f"r shape {tuple(r.shape)} != q shape {tuple(q.shape)}"
    assert k.shape[0] == B and k.shape[3] == D, (
        f"k shape {tuple(k.shape)} incompatible with q {tuple(q.shape)}"
    )
    H_kv = k.shape[2]
    if H % H_kv != 0:
        raise ValueError(
            f"H_q={H} is not a multiple of H_kv={H_kv}; GQA requires "
            f"H_q % H_kv == 0."
        )
    pack_n = H // H_kv
    if pack_n not in (1, 2, 4, 8):
        raise NotImplementedError(
            f"GQA pack_n={pack_n} (= H_q/H_kv = {H}/{H_kv}) is not yet "
            "supported. Supported values are 1 (MHA) and 2/4/8 (GQA). For an "
            "intermediate ratio expand K/V on the caller side with "
            "repeat_interleave."
        )
    assert v.shape == k.shape, f"v shape {tuple(v.shape)} != k shape {tuple(k.shape)}"
    assert D in (64, 128), f"head_dim must be 64 or 128, got {D}"

    if out is not None:
        assert out.shape == (B, 1, H, D), (
            f"out shape mismatch: expected {(B, 1, H, D)}, got {tuple(out.shape)}"
        )
        assert out.dtype == q.dtype, (
            f"out dtype mismatch: expected {q.dtype}, got {out.dtype}"
        )

    # kv_len semantics:
    #  - seqused_k given → per-batch active length; the K/V tensor is the
    #    full-size cache and `k.shape[1]` is the cache ceiling. The bucket
    #    derives from the cache ceiling so the compiled kernel is the same
    #    for any active length ≤ cache.shape[1]. This is the serving pattern
    #    (compile-once across all decode steps).
    #  - seqused_k None → treat k.shape[1] as the active length; bucket
    #    rounds it to a power of 2 so a handful of distinct sequence lengths
    #    still share a kernel (still better than per-length compiles, but a
    #    different K-shape per call will rebind the TMA descriptor).
    cache_len = k.shape[1]
    if seqused_k is not None:
        # Active length used only for split-K sizing; bucket is the cache
        # ceiling so all calls hit the same compile.
        kv_len_for_split = int(cache_len)
    else:
        kv_len_for_split = int(cache_len)
    # Grid is (B, H_kv, num_k_splits): one CTA per (batch, kv_head) emits
    # pack_n query-head rows. num_bh = B * H_kv for split-K wave math.
    num_bh = B * H_kv
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count

    # Power-of-2 bucket kv_len → max_tiles_total. This is the constexpr the
    # kernel binds at compile time; bucketing keeps the compile-cache from
    # exploding with one entry per unique kv_len. With kv_len bucketed to
    # the next pow2, a serving session compiles ~log2(max_len) variants — for
    # max_len=16k that's ~14 kernels for the whole length range.
    tiles_exact = (kv_len_for_split + 63) // 64
    max_tiles_total = 1
    while max_tiles_total < tiles_exact:
        max_tiles_total <<= 1
    bucket_kv_len = max_tiles_total * 64

    # SWA reduces the effective tile range to [first_valid_tile, tiles_total).
    # Use the bucket ceiling for split-count picking so num_k_splits stays
    # stable across kv_len values within the same bucket.
    if window_size_left >= 0:
        window_start_kv = max(0, bucket_kv_len - window_size_left)
        first_valid_tile = window_start_kv // 64
    else:
        first_valid_tile = 0
    valid_tiles_total = max(1, max_tiles_total - first_valid_tile)
    effective_kv_len = valid_tiles_total * 64

    num_k_splits = _choose_num_k_splits(num_bh, effective_kv_len, num_sms)
    if num_k_splits > 1:
        num_k_splits = _round_to_pow2_wave_aware(num_k_splits, num_bh, num_sms)
    num_k_splits = min(num_k_splits, valid_tiles_total)

    # Workspace is sized over H_q (every Q head has its own partial slot, even
    # though pack_n heads share a kv_head launch). For pack_n=1 this is the
    # original layout; for pack_n>1 each CTA writes pack_n head slots and the
    # merger loops pack_n times over the cross-split reduction.
    ws = _get_workspace(B * H, num_k_splits, D, q.device)
    return parallax_decode_cutedsl_sm90(
        q, r, k, v, scale, ws=ws, num_k_splits=num_k_splits,
        window_size_left=window_size_left, pack_n=pack_n,
        max_tiles_total=max_tiles_total, seqused_k=seqused_k, out=out,
    )


def _window_size_to_left(window_size) -> int:
    """Map an FA-style ``window_size`` to the kernel's ``window_size_left`` int.

    Accepts:
      * ``None`` or ``(-1, *)``     -> ``-1`` (sliding window disabled)
      * an ``int`` ``w`` (``w>=0``) -> ``w`` (treated as the left window)
      * a ``(left, right)`` pair    -> ``left``. Parallax decode is
        single-query causal (the one query sits at the end of the cache), so
        a positive ``right`` is meaningless and rejected.
    """
    if window_size is None:
        return -1
    if isinstance(window_size, int):
        return window_size
    try:
        left, right = window_size
    except (TypeError, ValueError):
        raise ValueError(
            "window_size must be None, an int, or a (left, right) pair; "
            f"got {window_size!r}"
        )
    if right is not None and right > 0:
        raise ValueError(
            "parallax decode is single-query causal; window_size right must be "
            f"<= 0 (got {right})."
        )
    return int(left)


def parallax_attn_with_kvcache(
    q: torch.Tensor,
    r: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_table: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    window_size=None,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Parallax decode against a KV cache — the canonical FA-style entry.

    Mirrors ``flash_attn_with_kvcache`` so a serving stack can swap Parallax in
    with a familiar signature. This is the single supported public entry;
    :func:`parallax_decode` is a deprecated alias kept for back-compat.

    Args:
        q, r: ``(B, 1, H, D)`` bf16/fp16. ``q`` is the base query, ``r`` the
            Parallax reweighting vector; the composite weight ``r·k_j`` is
            fused in-kernel (the caller never materializes it).
        k_cache, v_cache: ``(B, L, H, D)`` same dtype as ``q``. Dense
            (un-paged) for now; see ``page_table``.
        page_table: ``(B, max_num_blocks)`` int32 block table for paged KV.
            **Not implemented yet** — raises ``NotImplementedError`` if not
            ``None`` (lands with the paged-TMA prefill/extend kernel).
        seqused_k: ``(B,)`` int32 per-sequence valid KV length. When provided
            the kernel attends only to the first ``seqused_k[b]`` positions of
            ``k_cache[b]`` / ``v_cache[b]`` (which can therefore be allocated
            once at the max length and mutated in place across decode steps —
            the standard serving pattern). When ``None`` the kernel attends
            all ``k_cache.shape[1]`` positions. seqused_k unlocks
            "compile-once" across decode lengths: the kv_len-derived loop
            bounds are read at runtime per batch, so the compiled kernel is
            keyed on the bucket size, not the per-step length. Today
            seqused_k must contain values in [1, k_cache.shape[1]]; out-of-range
            values are clamped in-kernel (the graph-safe enforcement path) and
            raise ``ValueError`` in eager mode. Ragged / per-sequence padding
            (true varlen) lands with the paged-TMA prefill/extend kernel.
        window_size: FA-style causal sliding window. ``None`` or ``(-1, -1)``
            disables; ``(left, right)`` with ``right <= 0`` restricts the query
            to the most recent ``left`` keys. A bare int is treated as ``left``.
        scale: softmax scale; defaults to ``1 / sqrt(D)``.
        out: optional caller-owned output ``(B, 1, H, D)``, ``q.dtype``. If
            ``None`` a fresh tensor is allocated and returned (no silent reuse).

    Returns:
        ``(B, 1, H, D)`` tensor implementing the forward of Algorithm 1 in the
        Parallax paper (``== out`` when ``out`` is provided).

    Buffer-ownership / stability contracts:
        * **Explicit ownership.** ``out=None`` returns a freshly allocated
          tensor on every call. There is no hidden module-level output buffer,
          so successive ``out=None`` calls never alias each other's results.
        * **Stable-buffer contract (CUDA graphs / serving).** To capture this
          call in a CUDA graph, or to avoid a per-call allocation, pass stable
          ``q, r, k_cache, v_cache, out`` tensors and reuse the *same* tensor
          objects across calls. Provided inputs/outputs are memoized by data
          pointer and the internal split-K workspace is persistent and keyed by
          launch shape, so device addresses stay fixed across graph replays.
          ``seqused_k`` values are copied into an internal ``(B,)`` int32 buffer
          (one per ``(B, device)``), so callers may pass fresh tensors each step
          without leaking cache entries. For graph capture, use
          ``GraphedDecode.cache_seqlens`` directly.
          The workspace is **not** safe to share across concurrent calls of the
          same launch shape on different streams.
        * **Finite-padding contract.** The kernel reads KV in tiles of 64 and
          logically masks the partial last tile, but the PV matmul still forms
          ``0 * V_pad``. Any allocated-but-unused KV padding beyond ``L`` must
          be finite (e.g. zeros); NaN/Inf padding poisons the output because
          ``0 * NaN = NaN``.
    """
    if page_table is not None:
        raise NotImplementedError(
            "Paged KV (page_table) is not implemented yet — lands with the "
            "varlen paged prefill/extend kernel. Pass a dense "
            "(B, L, H, D) k_cache/v_cache for now."
        )
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            raise TypeError(f"seqused_k must be int32, got {seqused_k.dtype}")
        if seqused_k.ndim != 1 or seqused_k.shape[0] != q.shape[0]:
            raise ValueError(
                f"seqused_k must have shape (B={q.shape[0]},), "
                f"got {tuple(seqused_k.shape)}"
            )
        if not seqused_k.is_cuda or seqused_k.device != q.device:
            raise ValueError("seqused_k must live on the same CUDA device as q")
        # Eager-mode value-range check.  Skipped under graph capture
        # (the in-kernel clamp is the graph-safe enforcement).
        if not torch.cuda.is_current_stream_capturing():
            max_val = seqused_k.max().item()
            if max_val > k_cache.shape[1]:
                raise ValueError(
                    f"seqused_k max={max_val} exceeds k_cache.shape[1]={k_cache.shape[1]}; "
                    f"values must be in [1, {k_cache.shape[1]}]"
                )
            if seqused_k.min().item() < 1:
                raise ValueError(
                    f"seqused_k has values < 1; values must be in [1, {k_cache.shape[1]}]"
                )
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    window_size_left = _window_size_to_left(window_size)
    return _decode_core(
        q, r, k_cache, v_cache, float(scale),
        window_size_left=window_size_left, seqused_k=seqused_k, out=out,
    )


class GraphedDecode:
    """CUDA-graph-captured Parallax decode for a fixed launch shape.

    Small-batch decode is host-bound (launch + Python overhead dominates the
    few-microsecond kernel). This captures :func:`parallax_attn_with_kvcache`
    once and replays the graph each step, removing that overhead. It relies on
    the unified entry's explicit buffer ownership: inputs/output live in stable
    internal buffers, the split-K workspace is persistent, and there is no
    allocation or host sync inside the call — so the kernel launch captures
    cleanly.

    With seqused_k (runtime kv_len, the default for ``cache_seqlens != None``),
    one instance covers every decode step from ``kv_len = 1`` to
    ``max_kv_len``: K/V are pre-allocated at the cache ceiling, the captured
    graph reads the active length from a stable ``(B,)`` int32 buffer
    (``self.cache_seqlens``) at replay time, and the same compiled kernel
    handles all lengths.

    Usage (serving / iterative decode)::

        gd = GraphedDecode(B=1, H=8, max_kv_len=16384, head_dim=128)
        # decode step t: write k_t/v_t into gd.k[:, t-1, :] / gd.v[:, t-1, :],
        # update cache_seqlens, replay.
        gd.cache_seqlens.fill_(t)
        gd.q.copy_(q_t); gd.r.copy_(r_t)
        out = gd.replay()        # returns gd.out (a stable buffer)

        # Or the all-at-once helper for testing — copies full K/V and sets the
        # active length in one call:
        out = gd(q=q, r=r, k=k_full, v=v_full, cache_seqlens=lens)

    GQA: pass ``H_kv != H_q`` to construct one instance for the packed shape
    (``pack_n = H_q // H_kv`` derived in the dispatcher; supported values
    {1, 2, 4, 8}).

    Notes:
        * Shape-static over ``(B, H_q, H_kv, max_kv_len, head_dim, dtype,
          window_size, scale)``. One instance per *bucket*, not per kv_len.
        * ``cache_seqlens`` defaults to ``max_kv_len`` so a freshly constructed
          instance behaves like the legacy "attend everything" capture.
        * ``out`` is overwritten on every replay; clone it if you need to retain
          a step's result past the next replay.
    """

    def __init__(self, B: int, H: int, max_kv_len: int | None = None,
                 head_dim: int = 128, *,
                 H_kv: int | None = None,
                 dtype: torch.dtype = torch.bfloat16, window_size=None,
                 scale: float | None = None, device="cuda", warmup: int = 3,
                 # Back-compat: callers using the old `kv_len=` kwarg get a
                 # static-length capture (no seqused_k); same behavior as
                 # before. New code should pass `max_kv_len` instead.
                 kv_len: int | None = None):
        if max_kv_len is None and kv_len is None:
            raise ValueError("GraphedDecode requires max_kv_len (or kv_len for static-length back-compat)")
        if max_kv_len is not None and kv_len is not None:
            raise ValueError("pass exactly one of max_kv_len or kv_len")
        # Static-length mode: kv_len was passed — no seqused_k, capture attends
        # all `kv_len` positions exactly as the old GraphedDecode did.
        static_mode = max_kv_len is None
        cache_len = kv_len if static_mode else max_kv_len
        H_kv = H_kv if H_kv is not None else H
        if H % H_kv != 0:
            raise ValueError(f"H_q={H} must be a multiple of H_kv={H_kv}")
        self.q = torch.empty(B, 1, H, head_dim, device=device, dtype=dtype)
        self.r = torch.empty_like(self.q)
        self.k = torch.empty(B, cache_len, H_kv, head_dim, device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.out = torch.empty(B, 1, H, head_dim, device=device, dtype=dtype)
        self._scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self._window = window_size
        if static_mode:
            self.cache_seqlens = None
        else:
            # Stable (B,) int32 buffer captured into the graph; replay reads
            # its current value at launch time.
            self.cache_seqlens = torch.full((B,), cache_len, dtype=torch.int32, device=device)
        self.max_kv_len = cache_len

        def _call():
            parallax_attn_with_kvcache(
                self.q, self.r, self.k, self.v,
                seqused_k=self.cache_seqlens,
                window_size=self._window, scale=self._scale, out=self.out)

        # Warm up on a side stream: JIT-compile, allocate the split-K workspace,
        # and memoize the dlpack views — all of which must happen before capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(max(1, warmup)):
                _call()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            _call()

    def replay(self) -> torch.Tensor:
        """Replay the captured graph over the current buffer contents."""
        self.graph.replay()
        return self.out

    def __call__(self, *, q=None, r=None, k=None, v=None,
                 cache_seqlens=None) -> torch.Tensor:
        """Copy any provided inputs into the stable buffers, then replay.

        ``cache_seqlens`` is a ``(B,)`` int32 tensor (or Python int / list) of
        per-batch active KV lengths; ignored in static-length mode. Pass
        ``None`` to leave the previous value (e.g., when you mutate
        ``self.cache_seqlens`` in place yourself). Values must be in
        ``[1, max_kv_len]``; out-of-range values raise ``ValueError`` (the
        in-kernel clamp is the graph-safe fallback, but host-side validation
        catches contract violations early).
        """
        if q is not None:
            self.q.copy_(q)
        if r is not None:
            self.r.copy_(r)
        if k is not None:
            self.k.copy_(k)
        if v is not None:
            self.v.copy_(v)
        if cache_seqlens is not None and self.cache_seqlens is not None:
            if isinstance(cache_seqlens, int):
                if cache_seqlens < 1 or cache_seqlens > self.max_kv_len:
                    raise ValueError(
                        f"cache_seqlens={cache_seqlens} out of range "
                        f"[1, {self.max_kv_len}]"
                    )
                self.cache_seqlens.fill_(cache_seqlens)
            elif isinstance(cache_seqlens, (list, tuple)):
                t = torch.tensor(cache_seqlens, dtype=torch.int32,
                                 device=self.cache_seqlens.device)
                if min(cache_seqlens) < 1 or max(cache_seqlens) > self.max_kv_len:
                    raise ValueError(
                        f"cache_seqlens out of range [1, {self.max_kv_len}]: "
                        f"got min={min(cache_seqlens)}, max={max(cache_seqlens)}"
                    )
                self.cache_seqlens.copy_(t)
            else:
                # Tensor path: validate before copy (host-side, outside capture)
                _min = cache_seqlens.min().item()
                _max = cache_seqlens.max().item()
                if _min < 1 or _max > self.max_kv_len:
                    raise ValueError(
                        f"cache_seqlens out of range [1, {self.max_kv_len}]: "
                        f"got min={_min}, max={_max}"
                    )
                self.cache_seqlens.copy_(cache_seqlens)
        return self.replay()


def parallax_decode(q: torch.Tensor,
                    r: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    qk_scale: float,
                    *,
                    window_size_left: int = -1,
                    out: torch.Tensor | None = None) -> torch.Tensor:
    """Deprecated thin wrapper over the shared decode core.

    Kept for back-compat (existing benches/tests call it positionally). New
    code should prefer :func:`parallax_attn_with_kvcache`, which mirrors
    ``flash_attn_with_kvcache`` and documents the buffer-ownership,
    stable-buffer, and finite-padding contracts. This shim forwards directly to
    the shared core; the only differences are the positional ``qk_scale`` and
    the FA2 integer ``window_size_left`` convention.
    """
    return _decode_core(
        q, r, k, v, float(qk_scale),
        window_size_left=window_size_left, out=out,
    )
