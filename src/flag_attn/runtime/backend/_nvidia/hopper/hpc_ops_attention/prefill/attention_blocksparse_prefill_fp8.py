# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import os
from typing import Optional
import torch
import triton
import triton.language as tl

_BLOCK_M = 128
_BLOCK_N = 128
_HEAD_DIM = 128
_FP8_P_SCALE = 256.0
_LOG2E = 1.4426950408889634

import subprocess
from pathlib import Path
from flag_attn.utils import has_triton_tle, is_sm90_device
from flag_attn.hpc_ops_attention.prefill.attention_blocksparse_prefill_fp8 import (
    _LaunchContext,
    _check_inputs,
    _load_autotune_configs,
    _next_power_of_2,
    _normalize_sparsity_bucket,
    _prepare_launch_context,
    _quant_type_value,
    attention_with_kvcache_blocksparse_prefill_fp8_triton,
)

def _has_tle_raw_cuda_toolchain() -> bool:
    """Probe the same Clang/CUDA command used to materialize TLE raw sources."""
    if not torch.cuda.is_available():
        return False
    try:
        from triton.experimental.tle.raw.cuda.runtime import (
            _clang_flags,
            _get_cuda_gpu_arch,
            _resolve_clang,
        )

        result = subprocess.run(
            [
                _resolve_clang(),
                "-x",
                "cuda",
                "--cuda-device-only",
                _get_cuda_gpu_arch(),
                "-emit-llvm",
                "-S",
                "-",
                "-o",
                "-",
                *_clang_flags(),
            ],
            input=b'extern "C" __device__ void flag_attn_tle_probe() {}',
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (ImportError, OSError, RuntimeError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _hopper_autotune_configs_fallback():
    """Build Hopper/TLE configs when the optional YAML parser is unavailable."""

    def env_bool(name):
        value = os.environ.get(name, "0")
        if value not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        return bool(int(value))

    configs = [
        triton.Config(
            {
                "KV_PIPELINE_STAGES": 2,
                "PERSISTENT": persistent,
                "CONSUMER_NUM_REGS": consumer_num_regs,
                "CONSUMER_PINGPONG": pingpong,
                "SINGLE_VT_WRITER": env_bool("HPC_BSA_TLE_HOPPER_SINGLE_VT_WRITER"),
                "HOIST_QSCALE": env_bool("HPC_BSA_TLE_HOPPER_HOIST_QSCALE"),
                "EARLY_V_RELEASE": env_bool("HPC_BSA_TLE_HOPPER_EARLY_V_RELEASE"),
                "FUSE_SCORE_SCALE": env_bool("HPC_BSA_TLE_HOPPER_FUSE_SCORE_SCALE"),
            },
            num_warps=4,
            num_stages=2,
        )
        for persistent in (False, True)
        for consumer_num_regs in (192, 208, 224, 232)
        for pingpong in (False, True)
    ]
    forced = os.environ.get("HPC_BSA_TLE_HOPPER_FORCE_CONFIG")
    force_name = "HPC_BSA_TLE_HOPPER_FORCE_CONFIG"
    if forced is None:
        forced = os.environ.get("HPC_BSA_HOPPER_FORCE_CONFIG")
        force_name = "HPC_BSA_HOPPER_FORCE_CONFIG"
    if not forced:
        return configs
    try:
        values = tuple(map(int, forced.split(",")))
    except ValueError as exc:
        raise ValueError(f"{force_name} must contain comma-separated integers") from exc
    if len(values) == 3 and values[0] in (0, 1):
        persistent, consumer_num_regs, pingpong = values
        if consumer_num_regs not in (192, 208, 224, 232) or pingpong not in (0, 1):
            raise ValueError("Hopper force config must be PERSISTENT(0/1),CONSUMER_REGS(192/208/224/232),PINGPONG(0/1)")
        return [
            config
            for config in configs
            if int(config.kwargs["PERSISTENT"]) == persistent
            and config.kwargs["CONSUMER_NUM_REGS"] == consumer_num_regs
            and int(config.kwargs["CONSUMER_PINGPONG"]) == pingpong
        ]
    if len(values) == 3:
        pipeline_stages, num_warps, maxnreg = values
    elif len(values) == 4:
        pipeline_stages, num_warps, maxnreg, worker_regs = values
        if worker_regs <= 0:
            raise ValueError("legacy WORKER_REGS must be positive")
    else:
        raise ValueError(
            f"{force_name} must be PERSISTENT,CONSUMER_REGS,PINGPONG or the legacy STAGES,WARPS,MAXNREG[,WORKER_REGS]"
        )
    if pipeline_stages != 2 or num_warps != 4 or maxnreg not in (152, 160, 168):
        raise ValueError(f"unsupported legacy Hopper BSA force config: {forced}")
    return [
        config
        for config in configs
        if config.kwargs["PERSISTENT"]
        and config.kwargs["CONSUMER_NUM_REGS"] == 224
        and not config.kwargs["CONSUMER_PINGPONG"]
    ]


def _hopper_autotune_configs():
    return _load_autotune_configs("attention_blocksparse_prefill_fp8_hopper", _hopper_autotune_configs_fallback)


def _load_tle_hopper_dependencies():
    """Load optional TLE dependencies without using exceptions for control flow."""
    if not (has_triton_tle(3, 6, 0) and _has_tle_raw_cuda_toolchain()):
        return None, None

    try:
        import triton.experimental.tle.language as tle_module
        import triton.experimental.tle.language.raw as tle_raw_module
        from triton.experimental.tle.raw import dialect as tle_dialect
        from triton.language.core import (
            _unwrap_if_constexpr as unwrap_if_constexpr,
        )
        from triton.language.core import builtin as tle_builtin
        from triton.tools.tensor_descriptor import TensorDescriptor as descriptor_type
    except (ImportError, AttributeError, RuntimeError) as exc:
        return None, exc

    required_gpu_apis = (
        "alloc",
        "alloc_barrier",
        "alloc_barriers",
        "barrier_arrive",
        "barrier_wait",
        "buffered_tensor",
        "buffered_tensor_type",
        "copy",
        "local_ptr",
        "nv_mma_shared_layout",
        "smem",
        "warp_specialize",
        "wgmma",
        "wgmma_wait",
    )
    missing_gpu_apis = tuple(name for name in required_gpu_apis if not hasattr(tle_module.gpu, name))
    if missing_gpu_apis:
        error = AttributeError("TLE GPU API is missing " + ", ".join(sorted(missing_gpu_apis)))
        return None, error
    if not hasattr(tl, "float8e4nv"):
        return None, AttributeError("Triton language is missing float8e4nv")

    dependencies = (
        tle_module,
        tle_raw_module,
        tle_dialect,
        unwrap_if_constexpr,
        tle_builtin,
        descriptor_type,
    )
    return dependencies, None


_TLE_HOPPER_DEPENDENCIES, _TLE_HOPPER_IMPORT_ERROR = _load_tle_hopper_dependencies()


_HAS_TLE_HOPPER = _TLE_HOPPER_DEPENDENCIES is not None


if _HAS_TLE_HOPPER:
    (
        tle,
        tle_raw,
        dialect,
        triton_unwrap_if_constexpr,
        triton_builtin,
        TensorDescriptor,
    ) = _TLE_HOPPER_DEPENDENCIES
else:
    tle = None


if _HAS_TLE_HOPPER:

    @dialect(
        name="cuda",
        file=Path(__file__).with_name("attention_blocksparse_prefill_fp8_hopper_vtranspose.cu"),
        extern_func_name="bsa_compact_active_tiles",
        deferred=True,
    )
    def _compact_active_tiles_raw(*args, **kwargs):
        """TLE-Raw declaration for CUDA warp ballot/popcount compaction."""
        ...

    @dialect(
        name="cuda",
        file=Path(__file__).with_name("attention_blocksparse_prefill_fp8_hopper_vtranspose.cu"),
        extern_func_name="bsa_fp8_vtranspose_128x128",
        deferred=True,
    )
    def _vtranspose_128x128_raw(*args, **kwargs):
        """TLE-Raw declaration; the implementation lives in the CUDA file."""
        ...

    @triton.jit
    def _hopper_buf_phase(count, num_buffers: tl.constexpr):
        return count % num_buffers, count // num_buffers

    @triton.jit
    def _fma_rn_f32(a, b, c):
        """Force one IEEE FP32 FMA without crossing the TLE tensor layout."""
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [a, b, c],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton_builtin
    def _pin_qscale_to_wgmma_rows(zero_row_anchor, q_scale, _semantic=None):
        """Materialize QScale once in the WGMMA accumulator row layout.

        Elementwise inline assembly requires all tensor operands and its result
        to share one distributed layout.  ``zero_row_anchor`` is the consumer's
        64-row softmax accumulator, whose layout is fixed by the WGMMA result.
        Marking this exact ``0 + q_scale`` operation impure prevents loop-sink
        rematerialization while preserving the numerical value of QScale.
        """
        pinned = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [zero_row_anchor, q_scale],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
        # TLE's late WGMMA lowering sees the final blocked->MMA conversion and
        # hoists it out of the active-KV loop.  The impure anchor prevents
        # encoding rematerialization from cloning this value back into it.
        pinned.handle.set_attr(
            "tle.hoist_invariant_layout_conversion",
            _semantic.builder.get_bool_attr(True),
        )
        return pinned

    @triton_builtin
    def _hopper_smem_subview(buffer, offsets, shape, _semantic=None):
        """Return a static Shared memdesc subslice without allocating storage."""
        offsets = list(triton_unwrap_if_constexpr(offsets))
        shape = list(triton_unwrap_if_constexpr(shape))
        if not isinstance(buffer, tle.gpu.buffered_tensor):
            raise ValueError("Hopper Shared subview requires a buffered_tensor")
        if len(offsets) != len(buffer.shape) or len(shape) != len(buffer.shape):
            raise ValueError("Shared subview offsets/shape rank must match the source")
        offsets = [int(triton_unwrap_if_constexpr(value)) for value in offsets]
        shape = [int(triton_unwrap_if_constexpr(value)) for value in shape]
        for offset, extent, source_extent in zip(offsets, shape, buffer.shape):
            if offset < 0 or extent <= 0 or offset + extent > int(source_extent):
                raise ValueError("Shared subview is outside its source allocation")

        subview_ty = tle.gpu.buffered_tensor_type(
            buffer.dtype,
            shape,
            buffer.type.storage,
            buffer.type.layout,
            _semantic,
            alloc_shape=buffer.type.alloc_shape,
        )
        handle = _semantic.builder.create_memdesc_subslice(
            subview_ty.to_ir(_semantic.builder),
            buffer.handle,
            offsets,
        )
        return tle.gpu.buffered_tensor(
            handle,
            buffer.dtype,
            shape,
            buffer.type.storage,
            buffer.type.layout,
            _semantic,
            alloc_shape=buffer.type.alloc_shape,
        )

    @triton.jit
    def _hopper_decode_scheduled_tile(schedule_idx, total_tiles, num_q_tiles, num_head_q):
        """Decode the CUDA-compatible heavy-to-light persistent schedule."""
        head_batch_count = total_tiles // num_q_tiles
        reverse_idx = total_tiles - schedule_idx - 1
        q_tile = reverse_idx // head_batch_count
        head_batch = reverse_idx % head_batch_count
        q_head = head_batch % num_head_q
        batch = head_batch // num_head_q
        return q_tile, q_head, batch

    @triton.jit
    def _hopper_next_schedule_idx(schedule_idx, num_programs):
        """Mirror CUDA ``get_next_tile`` across successive CTA waves."""
        return schedule_idx + 2 * (num_programs - schedule_idx % num_programs) - 1

    @triton.jit
    def _bsa_fp8_hopper_producer(
        desc_q,
        desc_qs,
        desc_k,
        desc_v,
        desc_ks,
        cu_seqlens_q_ptr,
        block_ids_ptr,
        seqlens_kv_ptr,
        block_mask_ptr,
        q_smem,
        qs_smem,
        k_smem,
        ks_smem,
        v_smem,
        active_tiles_smem,
        active_count_smem,
        q_empties,
        q_fulls,
        qs_fulls,
        list_empty,
        list_full,
        k_empties,
        k_page_fulls,
        ks_page_fulls,
        v_empties,
        v_page_fulls,
        producer_sync,
        stride_block_ids_batch,
        stride_block_ids_page,
        stride_mask_batch,
        stride_mask_head,
        stride_mask_qtile,
        stride_mask_kvtile,
        num_head_q,
        num_head_kv,
        num_mask_kv_tiles,
        num_q_tiles,
        total_tiles,
        PAGE_SIZE: tl.constexpr,
        HAS_BLOCK_MASK: tl.constexpr,
        K_SCALE_PER_TOKEN: tl.constexpr,
        KV_LAYOUT: tl.constexpr,
        NUM_KV_STAGES: tl.constexpr,
        KV_TILE_BUCKET: tl.constexpr,
        BM_SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """TMA producer with stable mask compaction and a two-stage page ring."""
        program_idx = tl.program_id(0)
        num_programs = tl.num_programs(0)
        schedule_idx = program_idx
        tile_count = 0
        produced = 0
        pages_per_tile: tl.constexpr = BLOCK_N // PAGE_SIZE
        scale_groups_per_page: tl.constexpr = PAGE_SIZE // 32

        while schedule_idx < total_tiles:
            q_tile_padded, q_head, batch = _hopper_decode_scheduled_tile(
                schedule_idx, total_tiles, num_q_tiles, num_head_q
            )
            q_begin = tl.load(cu_seqlens_q_ptr + batch)
            q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
            q_len = q_end - q_begin
            kv_len = tl.load(seqlens_kv_ptr + batch)
            actual_q_tiles = (q_len + BLOCK_M - 1) // BLOCK_M
            q_tile = q_tile_padded - (num_q_tiles - actual_q_tiles)
            tile_valid = q_tile >= 0
            # Invalid padded tasks still traverse the uniform producer/consumer
            # protocol.  Point their guarded TMA traffic at tile zero so a
            # short sequence can never form a negative descriptor offset.
            q_tile = tl.maximum(q_tile, 0)
            q_local_start = q_tile * BLOCK_M
            program_valid = tile_valid & (q_local_start < q_len)
            kv_group_size = num_head_q // num_head_kv
            kv_head = q_head // kv_group_size
            q_start_in_kv = kv_len - q_len
            q_phase = tile_count

            # Q TMA starts before mask compaction, matching the native overlap.
            for consumer_idx in tl.static_range(0, 2):
                tle.gpu.barrier_wait(q_empties[consumer_idx], phaseIdx=q_phase)
                q_half_start = q_local_start + consumer_idx * BM_SPLIT
                tle.gpu.copy(
                    desc_q,
                    q_smem.slot(consumer_idx),
                    [BM_SPLIT, HEAD_DIM],
                    [q_begin + q_half_start, q_head * HEAD_DIM],
                    barrier=q_fulls[consumer_idx],
                )
                tle.gpu.copy(
                    desc_qs,
                    qs_smem.slot(consumer_idx),
                    [1, BM_SPLIT],
                    [batch * num_head_q + q_head, q_half_start],
                    barrier=qs_fulls[consumer_idx],
                )

            tle.gpu.barrier_wait(list_empty, phaseIdx=q_phase)
            q_tile_end = tl.minimum(q_local_start + BLOCK_M, q_len)
            causal_kv_end = q_start_in_kv + q_tile_end
            num_causal_kv_tiles = tl.maximum(0, (causal_kv_end + BLOCK_N - 1) // BLOCK_N)
            num_causal_kv_tiles = tl.minimum(num_causal_kv_tiles, KV_TILE_BUCKET)

            if HAS_BLOCK_MASK:
                mask_base_offset = batch * stride_mask_batch + q_head * stride_mask_head + q_tile * stride_mask_qtile
                num_tile_kv = tl.where(program_valid, num_causal_kv_tiles, 0).to(tl.int32)
                num_tile_with_mask = tl.minimum(num_tile_kv, num_mask_kv_tiles)
                compacted = tle_raw.call_smem(
                    _compact_active_tiles_raw,
                    [
                        block_mask_ptr + mask_base_offset,
                        num_tile_with_mask,
                        num_tile_kv,
                        active_tiles_smem,
                        active_count_smem,
                    ],
                    output_indices=[3, 4],
                    hint="bsa-active-list-ballot-popcount",
                )
                producer_active_tiles = compacted[0]
                producer_active_count = compacted[1]
            else:
                num_active = tl.where(program_valid, num_causal_kv_tiles, 0).to(tl.int32)
                producer_active_tiles = active_tiles_smem
                producer_active_count = active_count_smem
                tl.store(
                    tle.gpu.local_ptr(producer_active_count, (0,)),
                    num_active,
                )

            # Publish QScale, the compacted list, and its count together.
            tle.gpu.barrier_wait(producer_sync)
            tle.gpu.barrier_arrive(list_full, phaseIdx=q_phase)

            if HAS_BLOCK_MASK:
                num_active = tl.load(tle.gpu.local_ptr(producer_active_count, (0,)))

            for active_idx in range(0, KV_TILE_BUCKET):
                if active_idx < num_active:
                    if HAS_BLOCK_MASK:
                        kv_tile = tl.load(tle.gpu.local_ptr(producer_active_tiles, (active_idx,)))
                    else:
                        kv_tile = active_idx

                    # Debug-only invariant: a compacted list entry must remain
                    # inside the statically allocated KV-tile domain before it
                    # participates in page-table address arithmetic. Triton
                    # removes device assertions unless TRITON_DEBUG=1.
                    tl.device_assert(
                        (kv_tile >= 0) & (kv_tile < KV_TILE_BUCKET),
                        "producer loaded an invalid compacted kv_tile",
                    )

                    num_kv_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
                    first_logical_page = kv_tile * pages_per_tile

                    # Page-table loads are ordinary Global-memory operations,
                    # not TMA.  Resolve every physical page before waiting for
                    # the ring slot so an occupied stage can hide their latency.
                    # The public kernel supports PAGE_SIZE 32/64, hence at most
                    # four physical pages per 128-token WGMMA tile.
                    page_table_base = (
                        block_ids_ptr + batch * stride_block_ids_batch + first_logical_page * stride_block_ids_page
                    )
                    physical_page0 = tl.load(
                        page_table_base,
                        mask=first_logical_page < num_kv_pages,
                        other=-1,
                    )
                    physical_page1 = tl.load(
                        page_table_base + stride_block_ids_page,
                        mask=first_logical_page + 1 < num_kv_pages,
                        other=-1,
                    )
                    physical_page2 = physical_page0
                    physical_page3 = physical_page0
                    if PAGE_SIZE == 32:
                        physical_page2 = tl.load(
                            page_table_base + 2 * stride_block_ids_page,
                            mask=first_logical_page + 2 < num_kv_pages,
                            other=-1,
                        )
                        physical_page3 = tl.load(
                            page_table_base + 3 * stride_block_ids_page,
                            mask=first_logical_page + 3 < num_kv_pages,
                            other=-1,
                        )

                    stage, phase = _hopper_buf_phase(produced, NUM_KV_STAGES)
                    tle.gpu.barrier_wait(k_empties[stage], phaseIdx=phase)

                    # Prioritize data in first-use order.  The old per-page
                    # order was K0,V0,K1,V1 for PAGE_SIZE=64, which delayed K1
                    # (and therefore QK) behind a V transfer that is not needed
                    # until after QK and softmax.  Static transfer groups lower
                    # to K0,K1,[KS0,KS1],V0,V1 while retaining the existing
                    # per-page completion barriers.
                    num_transfer_groups: tl.constexpr = 2 + K_SCALE_PER_TOKEN
                    v_transfer_group: tl.constexpr = 1 + K_SCALE_PER_TOKEN
                    for transfer_group in tl.static_range(0, num_transfer_groups):
                        # K and KScale become reusable after both consumers
                        # finish QK/scale, while V remains live through softmax,
                        # transpose, and PV.  Waiting for v_empty before issuing
                        # K unnecessarily serialized those independent Shared
                        # allocations.  Delay only the V transfer group; K TMA
                        # can now run while the previous V stage is still live.
                        if transfer_group == v_transfer_group:
                            tle.gpu.barrier_wait(v_empties[stage], phaseIdx=phase)
                        for page_idx in tl.static_range(0, pages_per_tile):
                            if page_idx == 0:
                                physical_page = physical_page0
                            elif page_idx == 1:
                                physical_page = physical_page1
                            elif page_idx == 2:
                                physical_page = physical_page2
                            else:
                                physical_page = physical_page3

                            page_barrier_idx = stage * pages_per_tile + page_idx
                            if KV_LAYOUT == 0:
                                desc_offsets = [
                                    physical_page * PAGE_SIZE,
                                    kv_head * HEAD_DIM,
                                ]
                            else:
                                desc_offsets = [
                                    (physical_page * num_head_kv + kv_head) * PAGE_SIZE,
                                    0,
                                ]

                            if transfer_group == 0:
                                k_page = _hopper_smem_subview(
                                    k_smem.slot(stage),
                                    [page_idx * PAGE_SIZE, 0],
                                    [PAGE_SIZE, HEAD_DIM],
                                )
                                tle.gpu.copy(
                                    desc_k,
                                    k_page,
                                    [PAGE_SIZE, HEAD_DIM],
                                    desc_offsets,
                                    barrier=k_page_fulls[page_barrier_idx],
                                )
                            elif K_SCALE_PER_TOKEN and transfer_group == 1:
                                ks_page = _hopper_smem_subview(
                                    ks_smem.slot(stage),
                                    [
                                        page_idx * scale_groups_per_page,
                                        0,
                                    ],
                                    [scale_groups_per_page, 32],
                                )
                                tle.gpu.copy(
                                    desc_ks,
                                    ks_page,
                                    [scale_groups_per_page, 32],
                                    [
                                        physical_page * scale_groups_per_page,
                                        kv_head * 32,
                                    ],
                                    barrier=ks_page_fulls[page_barrier_idx],
                                )
                            else:
                                v_page = _hopper_smem_subview(
                                    v_smem.slot(stage),
                                    [page_idx * PAGE_SIZE, 0],
                                    [PAGE_SIZE, HEAD_DIM],
                                )
                                tle.gpu.copy(
                                    desc_v,
                                    v_page,
                                    [PAGE_SIZE, HEAD_DIM],
                                    desc_offsets,
                                    barrier=v_page_fulls[page_barrier_idx],
                                )
                    produced += 1

            schedule_idx = _hopper_next_schedule_idx(schedule_idx, num_programs)
            tile_count += 1

    @triton.jit
    def _bsa_fp8_hopper_consumer(
        desc_o,
        kscale_ptr,
        vscale_ptr,
        cu_seqlens_q_ptr,
        seqlens_kv_ptr,
        out_ptr,
        q_smem,
        qs_smem,
        k_smem,
        ks_smem,
        v_smem,
        vt_smem,
        active_tiles_smem,
        active_count_smem,
        q_empties,
        q_fulls,
        qs_fulls,
        list_empty,
        list_full,
        k_empties,
        k_page_fulls,
        ks_page_fulls,
        v_empties,
        v_page_fulls,
        vt_empties,
        vt_fulls,
        consumer_sync,
        ping_to_c0,
        ping_to_c1,
        stride_out_token,
        stride_out_head,
        stride_out_dim,
        num_head_q,
        num_head_kv,
        num_q_tiles,
        total_tiles,
        PAGE_SIZE: tl.constexpr,
        HAS_BLOCK_MASK: tl.constexpr,
        K_SCALE_PER_TOKEN: tl.constexpr,
        NUM_KV_STAGES: tl.constexpr,
        KV_TILE_BUCKET: tl.constexpr,
        CONSUMER_PINGPONG: tl.constexpr,
        SINGLE_VT_WRITER: tl.constexpr,
        HOIST_QSCALE: tl.constexpr,
        EARLY_V_RELEASE: tl.constexpr,
        FUSE_SCORE_SCALE: tl.constexpr,
        SCALE_LOG2E_OVER_SQRT_D: tl.constexpr,
        FP8_P_SCALE: tl.constexpr,
        BM_SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CID: tl.constexpr,
    ):
        """One WGMMA consumer computes one 64-row half across persistent tiles."""
        program_idx = tl.program_id(0)
        num_programs = tl.num_programs(0)
        consumer_idx: tl.constexpr = CID - 1
        pages_per_tile: tl.constexpr = BLOCK_N // PAGE_SIZE
        offs_half = tl.arange(0, BM_SPLIT)
        offs_128 = tl.arange(0, BLOCK_N)
        schedule_idx = program_idx
        tile_count = 0
        consumed = 0
        offs_n = offs_128
        offs_d = offs_128
        if CONSUMER_PINGPONG and consumer_idx == 1:
            tle.gpu.barrier_arrive(ping_to_c0)

        while schedule_idx < total_tiles:
            q_tile_padded, q_head, batch = _hopper_decode_scheduled_tile(
                schedule_idx, total_tiles, num_q_tiles, num_head_q
            )
            q_begin = tl.load(cu_seqlens_q_ptr + batch)
            q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
            q_len = q_end - q_begin
            kv_len = tl.load(seqlens_kv_ptr + batch)
            actual_q_tiles = (q_len + BLOCK_M - 1) // BLOCK_M
            q_tile = q_tile_padded - (num_q_tiles - actual_q_tiles)
            tile_valid = q_tile >= 0
            q_tile = tl.maximum(q_tile, 0)
            q_local_start = q_tile * BLOCK_M
            q_half_start = q_local_start + consumer_idx * BM_SPLIT
            kv_group_size = num_head_q // num_head_kv
            kv_head = q_head // kv_group_size
            q_start_in_kv = kv_len - q_len
            offs_m = q_half_start + offs_half
            q_valid = tile_valid & (offs_m < q_len)
            q_half_full = tile_valid & (q_half_start + BM_SPLIT <= q_len)
            min_q_position = q_start_in_kv + q_half_start
            q_phase = tile_count

            tle.gpu.barrier_wait(q_fulls[consumer_idx], phaseIdx=q_phase)
            tle.gpu.barrier_wait(qs_fulls[consumer_idx], phaseIdx=q_phase)
            tle.gpu.barrier_wait(list_full, phaseIdx=q_phase)
            q_scale = tl.load(
                tle.gpu.local_ptr(
                    qs_smem.slot(consumer_idx),
                    (offs_half * 0, offs_half),
                )
            ).to(tl.float32)
            num_active = tl.load(tle.gpu.local_ptr(active_count_smem, (0,)))

            m_i = tl.full((BM_SPLIT,), -float("inf"), tl.float32)
            l_i = tl.zeros((BM_SPLIT,), tl.float32)
            acc = tl.zeros((BM_SPLIT, HEAD_DIM), tl.float32)

            if HOIST_QSCALE:
                # The source QScale vector is loop invariant, but leaving its
                # layout implicit lets TLE materialize blocked->WGMMA scratch
                # traffic inside every active-KV iteration.  Tie it once to the
                # WGMMA row accumulator layout and keep the converted value in
                # registers for the complete online-softmax loop.
                if not K_SCALE_PER_TOKEN:
                    q_scale *= tl.load(kscale_ptr).to(tl.float32)
                    if FUSE_SCORE_SCALE:
                        q_scale *= SCALE_LOG2E_OVER_SQRT_D
                q_scale = _pin_qscale_to_wgmma_rows(l_i, q_scale)

            for active_idx in range(0, KV_TILE_BUCKET):
                if active_idx < num_active:
                    if HAS_BLOCK_MASK:
                        kv_tile = tl.load(tle.gpu.local_ptr(active_tiles_smem, (active_idx,)))
                    else:
                        kv_tile = active_idx
                    tl.device_assert(
                        (kv_tile >= 0) & (kv_tile < KV_TILE_BUCKET),
                        "consumer loaded an invalid compacted kv_tile",
                    )
                    stage, phase = _hopper_buf_phase(consumed, NUM_KV_STAGES)

                    for page_idx in tl.static_range(0, pages_per_tile):
                        page_barrier_idx = stage * pages_per_tile + page_idx
                        tle.gpu.barrier_wait(
                            k_page_fulls[page_barrier_idx],
                            phaseIdx=phase,
                        )

                    if CONSUMER_PINGPONG:
                        if consumer_idx == 0:
                            tle.gpu.barrier_wait(ping_to_c0)
                        else:
                            tle.gpu.barrier_wait(ping_to_c1)
                    scores = tle.gpu.wgmma(
                        q_smem.slot(consumer_idx),
                        k_smem.slot(stage),
                        out_dtype=tl.float32,
                        trans_b=True,
                    )
                    if CONSUMER_PINGPONG:
                        if consumer_idx == 0:
                            tle.gpu.barrier_arrive(ping_to_c1)
                        else:
                            tle.gpu.barrier_arrive(ping_to_c0)

                    # In the single-writer variant, overlap the only V
                    # transpose with the asynchronous QK completion window.
                    # Natural duplicate consumers would otherwise execute the
                    # same eight LDSM/eight STSM instructions concurrently and
                    # double the Shared-LSU conflict traffic.  vt_empty protects
                    # the shared destination from the preceding pair of PV
                    # readers; vt_full publishes it to consumer1 below.
                    if SINGLE_VT_WRITER:
                        consumer_vt = vt_smem.slot(stage)
                        if consumer_idx == 0:
                            for page_idx in tl.static_range(0, pages_per_tile):
                                page_barrier_idx = stage * pages_per_tile + page_idx
                                tle.gpu.barrier_wait(
                                    v_page_fulls[page_barrier_idx],
                                    phaseIdx=phase,
                                )
                            tle.gpu.barrier_wait(vt_empties[stage], phaseIdx=phase)
                            consumer_vt = tle_raw.call_smem(
                                _vtranspose_128x128_raw,
                                [v_smem.slot(stage), vt_smem.slot(stage)],
                                output_indices=[1],
                                hint=("bsa-vtranspose-single-writer-ldsm-prmt-stsm"),
                            )
                            if EARLY_V_RELEASE:
                                # Raw has consumed every source-V byte and its
                                # proxy fence has published the Vt writes.  Do
                                # not retain V through QK wait and softmax.
                                tle.gpu.barrier_wait(consumer_sync)
                                tle.gpu.barrier_arrive(v_empties[stage], phaseIdx=phase)
                                tle.gpu.barrier_arrive(vt_fulls[stage], phaseIdx=phase)
                    scores = tle.gpu.wgmma_wait(0, scores)

                    if K_SCALE_PER_TOKEN:
                        for page_idx in tl.static_range(0, pages_per_tile):
                            page_barrier_idx = stage * pages_per_tile + page_idx
                            tle.gpu.barrier_wait(
                                ks_page_fulls[page_barrier_idx],
                                phaseIdx=phase,
                            )
                        k_scale = tl.load(
                            tle.gpu.local_ptr(
                                ks_smem.slot(stage),
                                (offs_n // 32, offs_n % 32),
                            )
                        ).to(tl.float32)
                        score_scale = q_scale[:, None] * k_scale[None, :]
                    else:
                        if HOIST_QSCALE:
                            # QScale*KScale is already materialized once in the
                            # WGMMA row layout above.
                            score_scale_row = q_scale
                        else:
                            k_scale = tl.load(kscale_ptr).to(tl.float32)
                            score_scale_row = q_scale * k_scale
                            if FUSE_SCORE_SCALE:
                                score_scale_row *= SCALE_LOG2E_OVER_SQRT_D
                        score_scale = score_scale_row[:, None]
                    tle.gpu.barrier_arrive(k_empties[stage], phaseIdx=phase)

                    # The baseline mirrors the native CUDA dataflow and keeps
                    # Q/K quantization scale separate from attention scale.
                    # The guarded per-tensor variant below uses the equivalent
                    # positive row-scale identity to avoid materializing the
                    # first matrix-wide multiply.
                    if FUSE_SCORE_SCALE and not K_SCALE_PER_TOKEN:
                        # The quantization scale is positive and constant over
                        # each score row, so max(scores * s) == max(scores) * s.
                        # Keep scores unscaled until the existing shifted-score
                        # FMA and remove one full 64x128 FMUL from every tile.
                        scaled_scores = scores
                    else:
                        scaled_scores = scores * score_scale

                    # The producer only admits tiles up to the causal frontier.
                    # Except for the frontier/tail tile, all 64x128 scores are
                    # valid.  Keep that overwhelmingly common path free of the
                    # three broadcast masks: lowering their conjunction used to
                    # produce three ISETP/FSEL chains for every score register on
                    # every active KV tile.  Q-tail rows are intentionally not
                    # masked here, matching the native kernel: rows are
                    # independent and the epilogue suppresses their stores.
                    kv_tile_last = kv_tile * BLOCK_N + (BLOCK_N - 1)
                    score_tile_needs_mask = (kv_tile_last >= kv_len) | (kv_tile_last > min_q_position)
                    row_has_valid_score = tl.full((BM_SPLIT,), True, tl.int1)
                    if score_tile_needs_mask:
                        kv_tokens = kv_tile * BLOCK_N + offs_n
                        q_positions = q_start_in_kv + offs_m
                        score_invalid = (kv_tokens[None, :] >= kv_len) | (kv_tokens[None, :] > q_positions[:, None])
                        scaled_scores = tl.where(
                            score_invalid,
                            -float("inf"),
                            scaled_scores,
                        )
                        # A block-sparse frontier tile can begin after some
                        # rows in this Q half.  Those rows have no valid score
                        # in this tile even though the tile itself is active.
                        row_has_valid_score = (kv_tile * BLOCK_N < kv_len) & (kv_tile * BLOCK_N <= q_positions)

                    if FUSE_SCORE_SCALE and not K_SCALE_PER_TOKEN:
                        tile_max = tl.max(scaled_scores, axis=1) * score_scale_row
                    else:
                        tile_max = tl.max(scaled_scores, axis=1) * SCALE_LOG2E_OVER_SQRT_D
                    new_max = tl.maximum(m_i, tile_max)
                    # Preserve the online-softmax state for a row whose current
                    # sparse tile contains no causal key.  Evaluating the usual
                    # formulas directly would form -inf - -inf, poison p/acc
                    # with NaN, and send that NaN through the PV WGMMA.  Use a
                    # finite neutral max only for the arithmetic; m_i remains
                    # -inf until the row sees its first valid score.
                    row_had_mass = l_i > 0.0
                    row_has_mass = row_had_mass | row_has_valid_score
                    safe_new_max = tl.where(row_has_mass, new_max, 0.0)
                    safe_old_max = tl.where(row_had_mass, m_i, safe_new_max)
                    alpha = tl.exp2(safe_old_max - safe_new_max)
                    if FUSE_SCORE_SCALE and not K_SCALE_PER_TOKEN:
                        shifted_scores = _fma_rn_f32(
                            scaled_scores,
                            score_scale_row[:, None],
                            -safe_new_max[:, None],
                        )
                    else:
                        shifted_scores = _fma_rn_f32(
                            scaled_scores,
                            SCALE_LOG2E_OVER_SQRT_D,
                            -safe_new_max[:, None],
                        )
                    p = tl.exp2(shifted_scores)
                    l_i = _fma_rn_f32(l_i, alpha, tl.sum(p, axis=1))
                    acc *= alpha[:, None]
                    p_fp8 = (p * FP8_P_SCALE).to(tl.float8e4nv)

                    if SINGLE_VT_WRITER:
                        if consumer_idx == 0:
                            if not EARLY_V_RELEASE:
                                # The named barrier converges consumer0's four
                                # Raw warps.  Only then may it release source V
                                # and publish the proxy-fenced Vt writes.
                                tle.gpu.barrier_wait(consumer_sync)
                                tle.gpu.barrier_arrive(v_empties[stage], phaseIdx=phase)
                                tle.gpu.barrier_arrive(vt_fulls[stage], phaseIdx=phase)
                        else:
                            tle.gpu.barrier_wait(vt_fulls[stage], phaseIdx=phase)
                    else:
                        for page_idx in tl.static_range(0, pages_per_tile):
                            page_barrier_idx = stage * pages_per_tile + page_idx
                            tle.gpu.barrier_wait(
                                v_page_fulls[page_barrier_idx],
                                phaseIdx=phase,
                            )

                        consumer_vt = tle_raw.call_smem(
                            _vtranspose_128x128_raw,
                            [
                                v_smem.slot(stage),
                                vt_smem.slot(consumer_idx).slot(stage),
                            ],
                            output_indices=[1],
                            hint=("bsa-vtranspose-consumer-ldsm-prmt-stsm"),
                        )
                        tle.gpu.barrier_wait(consumer_sync)

                        # Both private consumers read source V, hence the
                        # baseline v_empty barrier has arrive_count=2.
                        tle.gpu.barrier_arrive(v_empties[stage], phaseIdx=phase)

                    acc = tle.gpu.wgmma(
                        p_fp8,
                        consumer_vt,
                        acc,
                        out_dtype=tl.float32,
                        trans_b=True,
                    )
                    acc = tle.gpu.wgmma_wait(0, acc)
                    if SINGLE_VT_WRITER:
                        # Consumer0 cannot overwrite this shared Vt stage until
                        # both asynchronous PV readers have completed.
                        tle.gpu.barrier_arrive(vt_empties[stage], phaseIdx=phase)
                    m_i = tl.where(row_has_mass, new_max, m_i)
                    consumed += 1

            # Q, QScale, and the active list are all dead for this consumer.
            tle.gpu.barrier_arrive(q_empties[consumer_idx], phaseIdx=q_phase)
            tle.gpu.barrier_arrive(list_empty, phaseIdx=q_phase)

            if K_SCALE_PER_TOKEN:
                v_scale = tl.load(vscale_ptr + kv_head).to(tl.float32)
            else:
                v_scale = tl.load(vscale_ptr).to(tl.float32)
            # A caller-provided mask may leave a query row without any causal
            # key.  The online update above keeps acc/l_i at zero for that row;
            # use a neutral denominator to return a finite zero instead of 0/0.
            # All non-empty rows retain the original normalization path.
            has_softmax_mass = l_i > 0.0
            safe_l_i = tl.where(has_softmax_mass, l_i, 1.0)
            output = acc * (v_scale / FP8_P_SCALE) / safe_l_i[:, None]
            output_bf16 = output.to(tl.bfloat16)

            if q_half_full:
                desc_o.store(
                    (q_begin + q_half_start, q_head * HEAD_DIM),
                    output_bf16,
                )
            else:
                out_offsets = (
                    (q_begin + offs_m[:, None]) * stride_out_token
                    + q_head * stride_out_head
                    + offs_d[None, :] * stride_out_dim
                )
                tl.store(
                    out_ptr + out_offsets,
                    output_bf16,
                    mask=q_valid[:, None],
                )

            schedule_idx = _hopper_next_schedule_idx(schedule_idx, num_programs)
            tile_count += 1

    # Keep the Hopper/TLE kernel on Triton's native autotuner.  TLE tensor
    # descriptors and warp-specialized launches are handled by Triton's kernel
    # interface directly; wrapping them in another tuner changes
    # the launch path and can corrupt partial output tiles.  The portable
    # Triton kernel above continues to use native Triton autotuning.
    @triton.autotune(
        configs=_hopper_autotune_configs(),
        key=[
            "Q_LEN_BUCKET",
            "KV_LEN_BUCKET",
            "SPARSITY_BUCKET",
            "OUTPUT_WAVES_BUCKET",
            "PAGE_SIZE",
            "HAS_BLOCK_MASK",
            "K_SCALE_PER_TOKEN",
            "KV_LAYOUT",
        ],
    )
    @triton.jit
    def _bsa_fp8_hopper_full_kernel(
        desc_q,
        desc_qs,
        desc_k,
        desc_v,
        desc_ks,
        desc_o,
        kscale_ptr,
        vscale_ptr,
        cu_seqlens_q_ptr,
        block_ids_ptr,
        seqlens_kv_ptr,
        block_mask_ptr,
        out_ptr,
        stride_block_ids_batch,
        stride_block_ids_page,
        stride_mask_batch,
        stride_mask_head,
        stride_mask_qtile,
        stride_mask_kvtile,
        stride_out_token,
        stride_out_head,
        stride_out_dim,
        num_head_q,
        num_head_kv,
        num_mask_kv_tiles,
        num_q_tiles,
        total_tiles,
        PAGE_SIZE: tl.constexpr,
        HAS_BLOCK_MASK: tl.constexpr,
        K_SCALE_PER_TOKEN: tl.constexpr,
        Q_LEN_BUCKET: tl.constexpr,
        KV_LEN_BUCKET: tl.constexpr,
        KV_TILE_BUCKET: tl.constexpr,
        SPARSITY_BUCKET: tl.constexpr,
        OUTPUT_WAVES_BUCKET: tl.constexpr,
        KV_LAYOUT: tl.constexpr,
        KV_PIPELINE_STAGES: tl.constexpr,
        PERSISTENT: tl.constexpr,
        CONSUMER_NUM_REGS: tl.constexpr,
        CONSUMER_PINGPONG: tl.constexpr,
        SINGLE_VT_WRITER: tl.constexpr,
        HOIST_QSCALE: tl.constexpr,
        EARLY_V_RELEASE: tl.constexpr,
        FUSE_SCORE_SCALE: tl.constexpr,
        SCALE_LOG2E_OVER_SQRT_D: tl.constexpr,
        FP8_P_SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Fixed Hopper microarchitecture with an autotuned launch policy."""
        tl.static_assert(BLOCK_M == 128)
        tl.static_assert(BLOCK_N == 128)
        tl.static_assert(HEAD_DIM == 128)
        tl.static_assert(BLOCK_N % PAGE_SIZE == 0)
        tl.static_assert(KV_PIPELINE_STAGES == 2)
        BM_SPLIT: tl.constexpr = BLOCK_M // 2
        PAGES_PER_TILE: tl.constexpr = BLOCK_N // PAGE_SIZE

        q_smem = tle.gpu.alloc(
            [2, BM_SPLIT, HEAD_DIM],
            dtype=tl.float8e4nv,
            layout=None,
            scope=tle.gpu.smem,
        )
        qs_smem = tle.gpu.alloc(
            [2, 1, BM_SPLIT],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
        )
        k_smem = tle.gpu.alloc(
            [KV_PIPELINE_STAGES, BLOCK_N, HEAD_DIM],
            dtype=tl.float8e4nv,
            layout=None,
            scope=tle.gpu.smem,
        )
        ks_smem = tle.gpu.alloc(
            [KV_PIPELINE_STAGES, BLOCK_N // 32, 32],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
        )
        v_smem = tle.gpu.alloc(
            [KV_PIPELINE_STAGES, BLOCK_N, HEAD_DIM],
            dtype=tl.float8e4nv,
            layout=None,
            scope=tle.gpu.smem,
        )
        if SINGLE_VT_WRITER:
            # One Vt ring is shared by both consumers.  Besides halving the
            # transpose instruction/conflict traffic, this saves 32 KiB of
            # dynamic Shared memory for the fixed two-stage pipeline.
            vt_smem = tle.gpu.alloc(
                [KV_PIPELINE_STAGES, HEAD_DIM, BLOCK_N],
                dtype=tl.float8e4nv,
                layout=None,
                scope=tle.gpu.smem,
            )
        else:
            # Baseline: both consumers transpose V into private rings.
            vt_smem = tle.gpu.alloc(
                [2, KV_PIPELINE_STAGES, HEAD_DIM, BLOCK_N],
                dtype=tl.float8e4nv,
                layout=None,
                scope=tle.gpu.smem,
            )
        active_tiles_smem = tle.gpu.alloc(
            [KV_TILE_BUCKET],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        active_count_smem = tle.gpu.alloc(
            [1],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )

        q_empties = tle.gpu.alloc_barriers(
            num_barriers=2,
            arrive_count=1,
            init=tle.gpu.READY,
        )
        q_fulls = tle.gpu.alloc_barriers(
            num_barriers=2,
            arrive_count=1,
            expect_bytes=BM_SPLIT * HEAD_DIM,
        )
        qs_fulls = tle.gpu.alloc_barriers(
            num_barriers=2,
            arrive_count=1,
            expect_bytes=BM_SPLIT * 4,
        )
        list_empty = tle.gpu.alloc_barrier(
            arrive_count=2,
            init=tle.gpu.READY,
        )
        list_full = tle.gpu.alloc_barrier(arrive_count=1)
        k_empties = tle.gpu.alloc_barriers(
            num_barriers=KV_PIPELINE_STAGES,
            arrive_count=2,
            init=tle.gpu.READY,
        )
        if SINGLE_VT_WRITER:
            v_empties = tle.gpu.alloc_barriers(
                num_barriers=KV_PIPELINE_STAGES,
                arrive_count=1,
                init=tle.gpu.READY,
            )
        else:
            v_empties = tle.gpu.alloc_barriers(
                num_barriers=KV_PIPELINE_STAGES,
                arrive_count=2,
                init=tle.gpu.READY,
            )
        k_page_fulls = tle.gpu.alloc_barriers(
            num_barriers=KV_PIPELINE_STAGES * PAGES_PER_TILE,
            arrive_count=1,
            expect_bytes=PAGE_SIZE * HEAD_DIM,
        )
        ks_page_fulls = tle.gpu.alloc_barriers(
            num_barriers=KV_PIPELINE_STAGES * PAGES_PER_TILE,
            arrive_count=1,
            expect_bytes=PAGE_SIZE * 4,
        )
        v_page_fulls = tle.gpu.alloc_barriers(
            num_barriers=KV_PIPELINE_STAGES * PAGES_PER_TILE,
            arrive_count=1,
            expect_bytes=PAGE_SIZE * HEAD_DIM,
        )

        if SINGLE_VT_WRITER:
            vt_empties = tle.gpu.alloc_barriers(
                num_barriers=KV_PIPELINE_STAGES,
                arrive_count=2,
                init=tle.gpu.READY,
            )
            vt_fulls = tle.gpu.alloc_barriers(
                num_barriers=KV_PIPELINE_STAGES,
                arrive_count=1,
            )
        else:
            # Compile-time-dead aliases keep the consumer signature uniform
            # without changing the baseline Shared allocation.
            vt_empties = v_empties
            vt_fulls = v_page_fulls

        producer_sync = tle.gpu.alloc_barrier(arrive_count=4 * 32)
        consumer_syncs = tle.gpu.alloc_barriers(num_barriers=2, arrive_count=4 * 32)
        pingpong = tle.gpu.alloc_barriers(num_barriers=2, arrive_count=4 * 32)
        ping_to_c0 = pingpong[0]
        ping_to_c1 = pingpong[1]

        tle.gpu.warp_specialize(
            [
                (
                    _bsa_fp8_hopper_producer,
                    (
                        desc_q,
                        desc_qs,
                        desc_k,
                        desc_v,
                        desc_ks,
                        cu_seqlens_q_ptr,
                        block_ids_ptr,
                        seqlens_kv_ptr,
                        block_mask_ptr,
                        q_smem,
                        qs_smem,
                        k_smem,
                        ks_smem,
                        v_smem,
                        active_tiles_smem,
                        active_count_smem,
                        q_empties,
                        q_fulls,
                        qs_fulls,
                        list_empty,
                        list_full,
                        k_empties,
                        k_page_fulls,
                        ks_page_fulls,
                        v_empties,
                        v_page_fulls,
                        producer_sync,
                        stride_block_ids_batch,
                        stride_block_ids_page,
                        stride_mask_batch,
                        stride_mask_head,
                        stride_mask_qtile,
                        stride_mask_kvtile,
                        num_head_q,
                        num_head_kv,
                        num_mask_kv_tiles,
                        num_q_tiles,
                        total_tiles,
                        PAGE_SIZE,
                        HAS_BLOCK_MASK,
                        K_SCALE_PER_TOKEN,
                        KV_LAYOUT,
                        KV_PIPELINE_STAGES,
                        KV_TILE_BUCKET,
                        BM_SPLIT,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM,
                    ),
                ),
                (
                    _bsa_fp8_hopper_consumer,
                    (
                        desc_o,
                        kscale_ptr,
                        vscale_ptr,
                        cu_seqlens_q_ptr,
                        seqlens_kv_ptr,
                        out_ptr,
                        q_smem,
                        qs_smem,
                        k_smem,
                        ks_smem,
                        v_smem,
                        vt_smem,
                        active_tiles_smem,
                        active_count_smem,
                        q_empties,
                        q_fulls,
                        qs_fulls,
                        list_empty,
                        list_full,
                        k_empties,
                        k_page_fulls,
                        ks_page_fulls,
                        v_empties,
                        v_page_fulls,
                        vt_empties,
                        vt_fulls,
                        consumer_syncs[0],
                        ping_to_c0,
                        ping_to_c1,
                        stride_out_token,
                        stride_out_head,
                        stride_out_dim,
                        num_head_q,
                        num_head_kv,
                        num_q_tiles,
                        total_tiles,
                        PAGE_SIZE,
                        HAS_BLOCK_MASK,
                        K_SCALE_PER_TOKEN,
                        KV_PIPELINE_STAGES,
                        KV_TILE_BUCKET,
                        CONSUMER_PINGPONG,
                        SINGLE_VT_WRITER,
                        HOIST_QSCALE,
                        EARLY_V_RELEASE,
                        FUSE_SCORE_SCALE,
                        SCALE_LOG2E_OVER_SQRT_D,
                        FP8_P_SCALE,
                        BM_SPLIT,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM,
                        1,
                    ),
                ),
                (
                    _bsa_fp8_hopper_consumer,
                    (
                        desc_o,
                        kscale_ptr,
                        vscale_ptr,
                        cu_seqlens_q_ptr,
                        seqlens_kv_ptr,
                        out_ptr,
                        q_smem,
                        qs_smem,
                        k_smem,
                        ks_smem,
                        v_smem,
                        vt_smem,
                        active_tiles_smem,
                        active_count_smem,
                        q_empties,
                        q_fulls,
                        qs_fulls,
                        list_empty,
                        list_full,
                        k_empties,
                        k_page_fulls,
                        ks_page_fulls,
                        v_empties,
                        v_page_fulls,
                        vt_empties,
                        vt_fulls,
                        consumer_syncs[1],
                        ping_to_c0,
                        ping_to_c1,
                        stride_out_token,
                        stride_out_head,
                        stride_out_dim,
                        num_head_q,
                        num_head_kv,
                        num_q_tiles,
                        total_tiles,
                        PAGE_SIZE,
                        HAS_BLOCK_MASK,
                        K_SCALE_PER_TOKEN,
                        KV_PIPELINE_STAGES,
                        KV_TILE_BUCKET,
                        CONSUMER_PINGPONG,
                        SINGLE_VT_WRITER,
                        HOIST_QSCALE,
                        EARLY_V_RELEASE,
                        FUSE_SCORE_SCALE,
                        SCALE_LOG2E_OVER_SQRT_D,
                        FP8_P_SCALE,
                        BM_SPLIT,
                        BLOCK_M,
                        BLOCK_N,
                        HEAD_DIM,
                        2,
                    ),
                ),
            ],
            [4, 4],
            [CONSUMER_NUM_REGS, CONSUMER_NUM_REGS],
        )


def _attention_with_kvcache_blocksparse_prefill_fp8_hopper_impl(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    quant_type=1,
    block_mask: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    sparsity_bucket: Optional[int] = None,
) -> torch.Tensor:
    """Run the selected SM90 TLE-Struct/TMA implementation.

    Capability checks are performed by the public selector before this
    implementation is entered.
    """
    quant_type_value = _quant_type_value(quant_type)
    _check_inputs(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type_value,
        block_mask,
        output,
    )

    launch = _prepare_launch_context(
        q,
        kcache,
        kscale,
        block_ids,
        max_seqlens_q,
        quant_type_value,
        block_mask,
        output,
        sparsity_bucket,
    )
    output = launch.output
    kscale_kernel = launch.kscale
    num_batch = launch.num_batch
    num_head_q = launch.num_head_q
    num_head_kv = launch.num_head_kv
    page_size = launch.page_size
    q_len_bucket = launch.q_len_bucket
    kv_len_bucket = launch.kv_len_bucket
    workload_sparsity_bucket = launch.sparsity_bucket
    kv_layout = launch.kv_layout
    has_mask = launch.has_mask
    num_mask_kv_tiles = launch.num_mask_kv_tiles
    mask_arg = launch.mask_arg
    mask_strides = launch.mask_strides

    def descriptor_allocator(size: int, align: int, stream: Optional[int]):
        del align, stream
        return torch.empty(size, dtype=torch.int8, device=q.device)

    triton.set_allocator(descriptor_allocator)
    desc_q = TensorDescriptor(
        q,
        shape=[q.shape[0], num_head_q * _HEAD_DIM],
        strides=[q.stride(0), 1],
        block_shape=[_BLOCK_M // 2, _HEAD_DIM],
    )
    desc_qs = TensorDescriptor(
        qscale,
        shape=[num_batch * num_head_q, qscale.shape[2]],
        strides=[qscale.stride(1), 1],
        block_shape=[1, _BLOCK_M // 2],
    )
    desc_o = TensorDescriptor(
        output,
        shape=[output.shape[0], num_head_q * _HEAD_DIM],
        strides=[output.stride(0), 1],
        block_shape=[_BLOCK_M // 2, _HEAD_DIM],
    )
    if kv_layout == 0:
        cache_desc_shape = [
            kcache.shape[0] * page_size,
            num_head_kv * _HEAD_DIM,
        ]
    else:
        cache_desc_shape = [
            kcache.shape[0] * num_head_kv * page_size,
            _HEAD_DIM,
        ]
    desc_k = TensorDescriptor(
        kcache,
        shape=cache_desc_shape,
        strides=[kcache.stride(1), 1],
        block_shape=[page_size, _HEAD_DIM],
    )
    desc_v = TensorDescriptor(
        vcache,
        shape=cache_desc_shape,
        strides=[vcache.stride(1), 1],
        block_shape=[page_size, _HEAD_DIM],
    )
    scale_groups_per_page = page_size // 32
    if quant_type_value == 0:
        desc_ks = TensorDescriptor(
            kscale_kernel,
            shape=[
                kcache.shape[0] * scale_groups_per_page,
                num_head_kv * 32,
            ],
            strides=[kscale_kernel.stride(1), 1],
            block_shape=[scale_groups_per_page, 32],
        )
    else:
        # The per-tensor specialization removes every use of this descriptor;
        # supply a valid descriptor type so the kernel signature stays uniform.
        desc_ks = TensorDescriptor(
            kcache,
            shape=cache_desc_shape,
            strides=[kcache.stride(1), 1],
            block_shape=[scale_groups_per_page, 32],
        )

    num_q_tiles = triton.cdiv(max_seqlens_q, _BLOCK_M)
    total_tiles = num_q_tiles * num_head_q * num_batch
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    output_waves_bucket = _next_power_of_2(max(1, triton.cdiv(total_tiles, num_sms)))

    def grid(meta):
        if meta["PERSISTENT"]:
            return (min(total_tiles, num_sms),)
        return (total_tiles,)

    _bsa_fp8_hopper_full_kernel[grid](
        desc_q,
        desc_qs,
        desc_k,
        desc_v,
        desc_ks,
        desc_o,
        kscale_kernel,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        mask_arg,
        output,
        *block_ids.stride(),
        *mask_strides,
        *output.stride(),
        num_head_q,
        num_head_kv,
        num_mask_kv_tiles,
        num_q_tiles,
        total_tiles,
        PAGE_SIZE=page_size,
        HAS_BLOCK_MASK=has_mask,
        K_SCALE_PER_TOKEN=quant_type_value == 0,
        Q_LEN_BUCKET=q_len_bucket,
        KV_LEN_BUCKET=kv_len_bucket,
        KV_TILE_BUCKET=triton.cdiv(kv_len_bucket, _BLOCK_N),
        SPARSITY_BUCKET=workload_sparsity_bucket,
        OUTPUT_WAVES_BUCKET=output_waves_bucket,
        KV_LAYOUT=kv_layout,
        SCALE_LOG2E_OVER_SQRT_D=_LOG2E / 11.313708498984761,
        FP8_P_SCALE=_FP8_P_SCALE,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=_HEAD_DIM,
    )
    return output


def _hopper_prefill_available(device: torch.device) -> bool:
    """Return whether the Hopper implementation may be entered."""
    return _HAS_TLE_HOPPER and device.type == "cuda" and is_sm90_device(device)


def _hopper_prefill_layout_compatible(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    output: Optional[torch.Tensor],
    max_seqlens_q: int,
    quant_type: object,
) -> bool:
    """Check TMA layout requirements before entering the Hopper implementation."""
    try:
        quant_type_value = _quant_type_value(quant_type)
        if q.ndim != 3 or q.stride(2) != 1 or q.stride(1) != _HEAD_DIM:
            return False
        if output is not None and (output.ndim != 3 or output.stride(2) != 1 or output.stride(1) != _HEAD_DIM):
            return False
        if qscale.ndim != 3:
            return False
        expected_qs_strides = (
            q.shape[1] * qscale.shape[2],
            qscale.shape[2],
            1,
        )
        if qscale.shape[2] < max_seqlens_q or qscale.shape[2] % 4 != 0 or tuple(qscale.stride()) != expected_qs_strides:
            return False
        if kcache.ndim != 4 or vcache.ndim != 4:
            return False
        page_size = kcache.shape[1]
        num_head_kv = kcache.shape[2]
        kv_layout = int(kcache.stride(1) < kcache.stride(2))
        expected_nhd = (
            page_size * num_head_kv * _HEAD_DIM,
            num_head_kv * _HEAD_DIM,
            _HEAD_DIM,
            1,
        )
        expected_hnd = (
            num_head_kv * page_size * _HEAD_DIM,
            _HEAD_DIM,
            page_size * _HEAD_DIM,
            1,
        )
        expected_cache_strides = expected_hnd if kv_layout else expected_nhd
        if tuple(kcache.stride()) != expected_cache_strides or tuple(vcache.stride()) != expected_cache_strides:
            return False
        if quant_type_value == 0:
            if kscale.element_size() == 1:
                if kscale.shape[-1] % 4 != 0:
                    return False
                kscale = kscale.view(torch.float32)
            scale_groups_per_page = page_size // 32
            expected_shape = (kcache.shape[0], scale_groups_per_page, num_head_kv, 32)
            expected_strides = (
                scale_groups_per_page * num_head_kv * 32,
                num_head_kv * 32,
                32,
                1,
            )
            if tuple(kscale.shape) != expected_shape or tuple(kscale.stride()) != expected_strides:
                return False
    except (IndexError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _hopper_prefill_compatible(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    output: Optional[torch.Tensor],
    max_seqlens_q: int,
    quant_type: object,
) -> bool:
    return _hopper_prefill_available(q.device) and _hopper_prefill_layout_compatible(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        output,
        max_seqlens_q,
        quant_type,
    )


def attention_with_kvcache_blocksparse_prefill_fp8_hopper(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    quant_type=1,
    block_mask: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    sparsity_bucket: Optional[int] = None,
) -> torch.Tensor:
    """Select Hopper BSA when available, otherwise use portable Triton."""
    if not _hopper_prefill_compatible(q, kcache, vcache, qscale, kscale, output, max_seqlens_q, quant_type):
        return attention_with_kvcache_blocksparse_prefill_fp8_triton(
            q,
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            quant_type,
            block_mask,
            output,
            sparsity_bucket=sparsity_bucket,
        )
    return _attention_with_kvcache_blocksparse_prefill_fp8_hopper_impl(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type,
        block_mask,
        output,
        sparsity_bucket=sparsity_bucket,
    )


attention_with_kvcache_blocksparse_prefill_fp8_tle = attention_with_kvcache_blocksparse_prefill_fp8_hopper


def attention_with_kvcache_blocksparse_prefill_fp8(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    quant_type=1,
    block_mask: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    sparsity_bucket: Optional[int] = None,
) -> torch.Tensor:
    """Select Hopper TLE when available, otherwise use portable Triton.

    Selection is capability based and happens before launching either kernel.
    Kernel compilation or execution failures are deliberately not swallowed.
    Use the suffixed entry points to pin one implementation in a benchmark.
    """
    implementation = attention_with_kvcache_blocksparse_prefill_fp8_triton
    if _hopper_prefill_compatible(q, kcache, vcache, qscale, kscale, output, max_seqlens_q, quant_type):
        implementation = _attention_with_kvcache_blocksparse_prefill_fp8_hopper_impl

    return implementation(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type,
        block_mask,
        output,
        sparsity_bucket=sparsity_bucket,
    )
