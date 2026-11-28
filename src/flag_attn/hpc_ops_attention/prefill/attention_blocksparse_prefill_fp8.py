# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_M = 128
_BLOCK_N = 128
_HEAD_DIM = 128
_FP8_P_SCALE = 256.0
_LOG2E = 1.4426950408889634

from flag_attn.runtime import torch_device_fn

from ..fp8 import decode_e4m3, round_e4m3


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _normalize_sparsity_bucket(block_mask: Optional[torch.Tensor], sparsity_bucket: Optional[int]) -> int:
    # Exact mask density is deliberately not read from the GPU here: doing so
    # would synchronize every production launch.  Workload generators that
    # already know the density (including our benchmark) can supply buckets
    # 0..3; an unspecified masked workload uses a separate generic bucket.
    if block_mask is None:
        return 0
    if sparsity_bucket is None:
        return 4
    value = int(sparsity_bucket)
    if value not in (0, 1, 2, 3):
        raise ValueError("sparsity_bucket must be one of 0, 1, 2, 3")
    return value


def _load_autotune_configs(name, fallback):
    """Use the shared backend config loader, with local defaults as a fallback."""
    try:
        from flag_attn.runtime import get_tuned_config

        configs = get_tuned_config(name)
    except (ImportError, OSError, TypeError, ValueError):
        return fallback()
    return configs or fallback()


def _portable_autotune_configs():
    """Return the portable launch configurations used by the migrated op."""
    configs = [
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_stages in (2, 3, 4)
        for num_warps in (4, 8)
    ]
    forced = os.environ.get("HPC_BSA_TRITON_FORCE_CONFIG")
    if forced:
        try:
            num_stages, num_warps = map(int, forced.split(","))
        except ValueError as exc:
            raise ValueError("HPC_BSA_TRITON_FORCE_CONFIG must be STAGES,WARPS") from exc
        configs = [config for config in configs if config.num_stages == num_stages and config.num_warps == num_warps]
        if not configs:
            raise ValueError(f"unsupported forced portable BSA config: {forced}")
    return configs


def _autotune_configs():
    return _load_autotune_configs("attention_blocksparse_prefill_fp8", _portable_autotune_configs)


@triton.autotune(
    configs=_autotune_configs(),
    key=[
        "Q_LEN_BUCKET",
        "KV_LEN_BUCKET",
        "SPARSITY_BUCKET",
        "PAGE_SIZE",
        "HAS_BLOCK_MASK",
        "K_SCALE_PER_TOKEN",
        "KV_LAYOUT",
    ],
)
@triton.jit
def _bsa_fp8_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qscale_ptr,
    kscale_ptr,
    vscale_ptr,
    cu_seqlens_q_ptr,
    block_ids_ptr,
    seqlens_kv_ptr,
    block_mask_ptr,
    out_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_qs_batch,
    stride_qs_head,
    stride_qs_token,
    stride_ks_page,
    stride_ks_group,
    stride_ks_head,
    stride_ks_token,
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
    PAGE_SIZE: tl.constexpr,
    MAX_KV_TOKENS: tl.constexpr,
    HAS_BLOCK_MASK: tl.constexpr,
    K_SCALE_PER_TOKEN: tl.constexpr,
    Q_LEN_BUCKET: tl.constexpr,
    KV_LEN_BUCKET: tl.constexpr,
    SPARSITY_BUCKET: tl.constexpr,
    KV_LAYOUT: tl.constexpr,
    SCALE_LOG2E_OVER_SQRT_D: tl.constexpr,
    FP8_P_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program computes one [BLOCK_M, HEAD_DIM] output tile."""
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_N == 128)
    tl.static_assert(BLOCK_N % PAGE_SIZE == 0)
    q_tile = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)

    q_begin = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(seqlens_kv_ptr + batch)

    q_local_start = q_tile * BLOCK_M
    program_valid = q_local_start < q_len
    kv_group_size = num_head_q // num_head_kv
    kv_head = q_head // kv_group_size
    q_start_in_kv = kv_len - q_len

    offs_m = q_local_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_valid = offs_m < q_len

    q_offsets = (q_begin + offs_m[:, None]) * stride_q_token + q_head * stride_q_head + offs_d[None, :] * stride_q_dim
    q = decode_e4m3(tl.load(q_ptr.to(tl.pointer_type(tl.uint8)) + q_offsets, mask=q_valid[:, None], other=0))

    qs_offsets = batch * stride_qs_batch + q_head * stride_qs_head + offs_m * stride_qs_token
    q_scale = tl.load(qscale_ptr + qs_offsets, mask=q_valid, other=0.0).to(tl.float32)

    # The CUDA implementation accumulates both softmax state and PV in FP32.
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    # Only tiles up to the causal end of this Q tile can contribute.  The loop
    # still has a compile-time maximum so Triton can pipeline it.
    q_tile_end = tl.minimum(q_local_start + BLOCK_M, q_len)
    causal_kv_end = q_start_in_kv + q_tile_end
    num_causal_kv_tiles = (causal_kv_end + BLOCK_N - 1) // BLOCK_N

    offs_n = tl.arange(0, BLOCK_N)
    max_kv_tiles = (MAX_KV_TOKENS + BLOCK_N - 1) // BLOCK_N
    # Keep the indirect page-table -> K/V load loop at the proven-safe depth.
    # Config.num_stages is still autotuned for dot-feeding loads.  Feeding the
    # same value into tl.range would additionally pipeline most pointer-chasing
    # loads and produced NaNs / intermittent illegal accesses at depth 4 on
    # the Triton 3.6 FlagTree backend.
    for kv_tile in tl.range(0, max_kv_tiles, 1, num_stages=2):
        tile_active = program_valid & (kv_tile < num_causal_kv_tiles)
        if HAS_BLOCK_MASK:
            mask_in_range = kv_tile < num_mask_kv_tiles
            mask_offset = (
                batch * stride_mask_batch
                + q_head * stride_mask_head
                + q_tile * stride_mask_qtile
                + kv_tile * stride_mask_kvtile
            )
            selected = (
                tl.load(
                    block_mask_ptr + mask_offset,
                    mask=mask_in_range,
                    other=0,
                )
                != 0
            )
            # Match the CUDA compatibility rule: if the supplied mask is
            # shorter than the causal range, retain exactly the first tile
            # after the represented range.
            selected = selected | (kv_tile == num_mask_kv_tiles)
            tile_active = tile_active & selected

        # This is a program-uniform branch.  A skipped BSA tile performs no
        # page-table loads, K/V loads, or Tensor-Core work.
        if tile_active:
            kv_tokens = kv_tile * BLOCK_N + offs_n
            kv_valid = kv_tokens < kv_len
            logical_pages = kv_tokens // PAGE_SIZE
            tokens_in_page = kv_tokens % PAGE_SIZE
            page_offsets = batch * stride_block_ids_batch + logical_pages * stride_block_ids_page
            physical_pages = tl.load(
                block_ids_ptr + page_offsets,
                mask=kv_valid,
                other=0,
            )

            k_offsets = (
                physical_pages[:, None] * stride_k_page
                + tokens_in_page[:, None] * stride_k_token
                + kv_head * stride_k_head
                + offs_d[None, :] * stride_k_dim
            )
            k = decode_e4m3(tl.load(k_ptr.to(tl.pointer_type(tl.uint8)) + k_offsets, mask=kv_valid[:, None], other=0))

            # E4M3 values are exact in FP16; accumulate the product in FP32.
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

            if K_SCALE_PER_TOKEN:
                # K scale storage is logically
                # [page, token//32, kv_head, token%32] in FP32.  The public API
                # also accepts its byte-identical FP8 view; the launcher views
                # that tensor back to FP32 without copying.
                ks_offsets = (
                    physical_pages * stride_ks_page
                    + (tokens_in_page // 32) * stride_ks_group
                    + kv_head * stride_ks_head
                    + (tokens_in_page % 32) * stride_ks_token
                )
                k_scale = tl.load(
                    kscale_ptr + ks_offsets,
                    mask=kv_valid,
                    other=0.0,
                ).to(tl.float32)
                score_scale = q_scale[:, None] * k_scale[None, :]
            else:
                k_scale = tl.load(kscale_ptr).to(tl.float32)
                score_scale = q_scale[:, None] * k_scale

            q_positions = q_start_in_kv + offs_m
            causal_valid = kv_tokens[None, :] <= q_positions[:, None]
            score_valid = q_valid[:, None] & kv_valid[None, :] & causal_valid
            scores = tl.where(
                score_valid,
                scores * score_scale * SCALE_LOG2E_OVER_SQRT_D,
                -float("inf"),
            )

            # Online softmax in base 2, equivalent to the CUDA implementation.
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m_i, tile_max)
            alpha = tl.exp2(m_i - new_max)
            p = tl.exp2(scores - new_max[:, None])
            tile_sum = tl.sum(p, axis=1)

            l_i = l_i * alpha + tile_sum
            acc *= alpha[:, None]

            # This quantization is part of the original BSA algorithm, not an
            # input Q/K/V quantization step.
            p_fp8 = round_e4m3(p * FP8_P_SCALE)

            v_offsets = (
                physical_pages[:, None] * stride_v_page
                + tokens_in_page[:, None] * stride_v_token
                + kv_head * stride_v_head
                + offs_d[None, :] * stride_v_dim
            )
            v = decode_e4m3(tl.load(v_ptr.to(tl.pointer_type(tl.uint8)) + v_offsets, mask=kv_valid[:, None], other=0))
            acc = tl.dot(p_fp8, v, acc, out_dtype=tl.float32)
            m_i = new_max

    if K_SCALE_PER_TOKEN:
        v_scale = tl.load(vscale_ptr + kv_head).to(tl.float32)
    else:
        v_scale = tl.load(vscale_ptr).to(tl.float32)

    # A completely empty sparse row intentionally follows the CUDA contract
    # and produces NaN (0 / 0).  Callers are expected to retain the diagonal
    # BSA tile for every valid Q tile.
    output = acc * (v_scale / FP8_P_SCALE) / l_i[:, None]
    out_offsets = (
        (q_begin + offs_m[:, None]) * stride_out_token + q_head * stride_out_head + offs_d[None, :] * stride_out_dim
    )
    tl.store(out_ptr + out_offsets, output.to(tl.bfloat16), mask=q_valid[:, None])


def _quant_type_value(quant_type) -> int:
    return int(getattr(quant_type, "value", quant_type))


def _check_inputs(
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
    quant_type: int,
    block_mask: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
) -> None:
    tensors = {
        "q": q,
        "kcache": kcache,
        "vcache": vcache,
        "qscale": qscale,
        "kscale": kscale,
        "vscale": vscale,
        "cu_seqlens_q": cu_seqlens_q,
        "block_ids": block_ids,
        "seqlens_kvcache": seqlens_kvcache,
    }
    for name, tensor in tensors.items():
        if tensor.device.type == "cpu":
            raise ValueError(f"{name} must be an accelerator tensor")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}")

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None or q.dtype != fp8_dtype or kcache.dtype != fp8_dtype or vcache.dtype != fp8_dtype:
        raise TypeError("q, kcache and vcache must use torch.float8_e4m3fn")
    if q.ndim != 3 or kcache.ndim != 4 or vcache.ndim != 4:
        raise ValueError("q must be rank 3 and kcache/vcache must be rank 4")
    if q.shape[2] != _HEAD_DIM or kcache.shape[3] != _HEAD_DIM or vcache.shape[3] != _HEAD_DIM:
        raise ValueError("only head_dim=128 is supported")
    if kcache.shape[:3] != vcache.shape[:3]:
        raise ValueError("kcache and vcache page/token/head dimensions must match")
    if q.shape[1] % kcache.shape[2] != 0:
        raise ValueError("num_head_q must be divisible by num_head_kv")
    if kcache.shape[1] not in (32, 64) or _BLOCK_N % kcache.shape[1] != 0:
        raise ValueError("paged BSA currently supports page sizes 32 and 64")
    if cu_seqlens_q.dtype != torch.int32 or block_ids.dtype != torch.int32 or seqlens_kvcache.dtype != torch.int32:
        raise TypeError("cu_seqlens_q, block_ids and seqlens_kvcache must be int32")
    if cu_seqlens_q.ndim != 1 or seqlens_kvcache.ndim != 1 or block_ids.ndim != 2:
        raise ValueError("invalid sequence metadata rank")
    num_batch = cu_seqlens_q.numel() - 1
    if num_batch <= 0 or seqlens_kvcache.numel() != num_batch or block_ids.shape[0] != num_batch:
        raise ValueError("batch dimensions of sequence metadata do not match")
    if block_ids.shape[1] <= 0:
        raise ValueError("block_ids must contain at least one logical KV page")
    if qscale.shape[:2] != (num_batch, q.shape[1]) or qscale.ndim != 3:
        raise ValueError("qscale must have shape [batch, num_head_q, max_seq_q_pad]")
    if qscale.dtype != torch.float32 or vscale.dtype != torch.float32:
        raise TypeError("qscale and vscale must be float32")
    if quant_type not in (0, 1):
        raise ValueError("quant_type must be 0 (per-token K) or 1 (per-tensor K/V)")
    if quant_type == 1:
        if kscale.dtype != torch.float32 or kscale.numel() < 1 or vscale.numel() < 1:
            raise TypeError("per-tensor K/V mode requires float32 scalar scales")
    else:
        if kscale.element_size() not in (1, 4):
            raise TypeError("per-token K scale must be an FP32 tensor or its FP8 byte view")
        if vscale.numel() != kcache.shape[2]:
            raise ValueError("per-head vscale must contain num_head_kv elements")
    if max_seqlens_q <= 0:
        raise ValueError("max_seqlens_q must be positive")

    expected_q_tiles = triton.cdiv(max_seqlens_q, _BLOCK_M)
    if block_mask is not None:
        if block_mask.device != q.device or block_mask.dtype != torch.uint8 or not block_mask.is_contiguous():
            raise ValueError("block_mask must be a contiguous uint8 tensor on the Q device")
        if block_mask.ndim != 4 or block_mask.shape[:3] != (
            num_batch,
            q.shape[1],
            expected_q_tiles,
        ):
            raise ValueError("block_mask must have shape [batch, num_head_q, ceil(max_seqlens_q/128), Kb]")
        if block_mask.shape[3] <= 0:
            raise ValueError("block_mask Kb dimension must be positive")

    if output is not None:
        if output.device != q.device or output.dtype != torch.bfloat16:
            raise ValueError("output must be a bfloat16 tensor on the Q device")
        if output.shape != q.shape or not output.is_contiguous():
            raise ValueError("output must be contiguous with shape [total_seq_q, num_head_q, 128]")


@dataclass(frozen=True)
class _LaunchContext:
    output: torch.Tensor
    kscale: torch.Tensor
    num_batch: int
    num_head_q: int
    num_head_kv: int
    page_size: int
    max_kv_tokens: int
    q_len_bucket: int
    kv_len_bucket: int
    sparsity_bucket: int
    kv_layout: int
    has_mask: bool
    num_mask_kv_tiles: int
    mask_arg: torch.Tensor
    mask_strides: tuple[int, int, int, int]


def _prepare_launch_context(
    q: torch.Tensor,
    kcache: torch.Tensor,
    kscale: torch.Tensor,
    block_ids: torch.Tensor,
    max_seqlens_q: int,
    quant_type: int,
    block_mask: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    sparsity_bucket: Optional[int],
) -> _LaunchContext:
    if output is None:
        output = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)

    kernel_kscale = kscale
    if quant_type == 0 and kscale.element_size() == 1:
        if kscale.shape[-1] % 4 != 0:
            raise ValueError("the FP8 K-scale byte view must have a last dimension divisible by 4")
        kernel_kscale = kscale.view(torch.float32)

    num_batch = block_ids.shape[0]
    num_head_q = q.shape[1]
    num_head_kv = kcache.shape[2]
    page_size = kcache.shape[1]
    max_kv_tokens = block_ids.shape[1] * page_size
    has_mask = block_mask is not None
    return _LaunchContext(
        output=output,
        kscale=kernel_kscale,
        num_batch=num_batch,
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        page_size=page_size,
        max_kv_tokens=max_kv_tokens,
        q_len_bucket=_next_power_of_2(max_seqlens_q),
        kv_len_bucket=_next_power_of_2(max_kv_tokens),
        sparsity_bucket=_normalize_sparsity_bucket(block_mask, sparsity_bucket),
        kv_layout=int(kcache.stride(1) < kcache.stride(2)),
        has_mask=has_mask,
        num_mask_kv_tiles=block_mask.shape[3] if has_mask else 0,
        mask_arg=block_mask if has_mask else q,
        mask_strides=block_mask.stride() if has_mask else (0, 0, 0, 0),
    )


def attention_with_kvcache_blocksparse_prefill_fp8_triton(
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
    """Run portable Triton FP8 paged block-sparse prefill attention."""
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
    max_kv_tokens = launch.max_kv_tokens
    q_len_bucket = launch.q_len_bucket
    kv_len_bucket = launch.kv_len_bucket
    workload_sparsity_bucket = launch.sparsity_bucket
    kv_layout = launch.kv_layout
    has_mask = launch.has_mask
    num_mask_kv_tiles = launch.num_mask_kv_tiles
    mask_arg = launch.mask_arg
    mask_strides = launch.mask_strides
    ks_strides = kscale_kernel.stride() if quant_type_value == 0 else (0, 0, 0, 0)

    grid = (triton.cdiv(max_seqlens_q, _BLOCK_M), num_head_q, num_batch)
    with torch_device_fn.device(q.device):
        _bsa_fp8_prefill_kernel[grid](
            q.view(torch.uint8),
            kcache.view(torch.uint8),
            vcache.view(torch.uint8),
            qscale,
            kscale_kernel,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            mask_arg,
            output,
            *q.stride(),
            *kcache.stride(),
            *vcache.stride(),
            *qscale.stride(),
            *ks_strides,
            *block_ids.stride(),
            *mask_strides,
            *output.stride(),
            num_head_q,
            num_head_kv,
            num_mask_kv_tiles,
            PAGE_SIZE=page_size,
            MAX_KV_TOKENS=max_kv_tokens,
            HAS_BLOCK_MASK=has_mask,
            K_SCALE_PER_TOKEN=quant_type_value == 0,
            Q_LEN_BUCKET=q_len_bucket,
            KV_LEN_BUCKET=kv_len_bucket,
            SPARSITY_BUCKET=workload_sparsity_bucket,
            KV_LAYOUT=kv_layout,
            SCALE_LOG2E_OVER_SQRT_D=_LOG2E / 11.313708498984761,
            FP8_P_SCALE=_FP8_P_SCALE,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            HEAD_DIM=_HEAD_DIM,
        )
    return output


# Historical entry points retain their signatures and use portable Triton.
attention_with_kvcache_blocksparse_prefill_fp8 = attention_with_kvcache_blocksparse_prefill_fp8_triton
attention_with_kvcache_blocksparse_prefill_fp8_hopper = attention_with_kvcache_blocksparse_prefill_fp8_triton
attention_with_kvcache_blocksparse_prefill_fp8_tle = attention_with_kvcache_blocksparse_prefill_fp8_triton
