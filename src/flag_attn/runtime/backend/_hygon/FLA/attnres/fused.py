# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_attn.runtime.backend._hygon.device_info import get_device_info
from flag_attn.runtime.backend._hygon.heuristics_config_utils import HEURISTICS_CONFIGS

# Hygon gfx936 uses the regular Triton/HIP backend.
HAS_TRITON_ATTNRES = True


@lru_cache(maxsize=1)
def _get_attnres_runtime_config():
    device_info = get_device_info()
    op_heuristics = HEURISTICS_CONFIGS.get("fused_attnres", {})
    device_heuristics = op_heuristics.get("by_sm_count", {}).get(
        device_info.sm_count, op_heuristics.get("default", {})
    )

    try:
        from flag_attn.runtime import get_tuned_config

        tuned_configs = get_tuned_config("_fused_attnres_fwd_kernel")
    except (ImportError, OSError, TypeError, ValueError):
        tuned_configs = []

    if tuned_configs:
        # This kernel uses the curated first entry to avoid runtime autotuning.
        launch_config = tuned_configs[0]
        num_warps = launch_config.num_warps or 4
        num_stages = launch_config.num_stages or 2
        kernel_meta = dict(launch_config.kwargs)
    else:
        num_warps, num_stages, kernel_meta = 4, 2, {}
    return device_heuristics, num_warps, num_stages, kernel_meta


@triton.jit
def _score_source(
    residual_ptr,
    row,
    D: tl.constexpr,
    offsets_d,
    mask_d,
    query_weight,
    eps: tl.constexpr,
    JOIN_REDUCTIONS: tl.constexpr,
):
    residual = tl.load(
        tl.multiple_of(residual_ptr + row * D + offsets_d, 16),
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    if JOIN_REDUCTIONS:
        sums = tl.sum(
            tl.join(residual * residual, residual * query_weight),
            axis=0,
        )
        sq_sum, dot = tl.split(sums)
    else:
        sq_sum = tl.sum(residual * residual, axis=0)
        dot = tl.sum(residual * query_weight, axis=0)
    source_logit = dot * tl.rsqrt(sq_sum / D + eps)
    return residual, source_logit


@triton.jit
def _fused_attnres_fwd_kernel(
    query,
    residuals,
    rms_weight,
    output_rms_weight,
    output_rms_bias,
    post_scale,
    output,
    logit,
    lse,
    N,
    L: tl.constexpr,
    D: tl.constexpr,
    JOIN_REDUCTIONS: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    HAS_ONORM_BIAS: tl.constexpr,
    HAS_POST_SCALE: tl.constexpr,
    RETURN_WEIGHTS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    mask_d = offsets_d < D

    if L == 1 and not RETURN_WEIGHTS:
        query_weight = tl.zeros([BD], dtype=tl.float32)
    else:
        query_values = tl.load(query + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        weight_values = tl.load(rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        query_weight = query_values * weight_values

    running_max = tl.full([], -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros([], dtype=tl.float32)
    output_values = tl.zeros([BD], dtype=tl.float32)

    if L == 1 and not RETURN_WEIGHTS:
        residual_ptr = tl.multiple_of(
            residuals[0] + row * D + offsets_d,
            16,
        )
        output_values = tl.load(
            residual_ptr,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        running_max = tl.zeros([], dtype=tl.float32)
        running_sum = tl.full([], 1.0, dtype=tl.float32)
    else:
        residual0, source_logit0 = _score_source(
            residuals[0],
            row,
            D,
            offsets_d,
            mask_d,
            query_weight,
            eps,
            L >= 3 and JOIN_REDUCTIONS,
        )
        source_score0 = source_logit0 * scale
        running_max = source_score0
        running_sum = tl.full([], 1.0, dtype=tl.float32)
        output_values = residual0

        if RETURN_WEIGHTS:
            tl.store(logit + row, source_logit0)

        if L > 1:
            # Keep the source body as a runtime loop.  The pointer mux remains
            # compile-time unrolled, while the large vector temporaries are
            # reused for each source instead of expanding the whole body L-1
            # times in the generated kernel.
            for source in range(1, L):
                source_ptr = residuals[1]
                for source_index in tl.static_range(2, L):
                    source_ptr = tl.where(
                        source == source_index,
                        residuals[source_index],
                        source_ptr,
                    )
                residual_norm, source_logit = _score_source(
                    source_ptr,
                    row,
                    D,
                    offsets_d,
                    mask_d,
                    query_weight,
                    eps,
                    L >= 3 and JOIN_REDUCTIONS,
                )
                source_score = source_logit * scale

                previous_max = running_max
                running_max = tl.maximum(running_max, source_score)
                rescale = tl.exp(previous_max - running_max)
                probability = tl.exp(source_score - running_max)
                running_sum = running_sum * rescale + probability
                output_values = (
                    output_values * rescale
                    + probability * residual_norm
                )

                if RETURN_WEIGHTS:
                    tl.store(logit + source * N + row, source_logit)
    output_values /= running_sum
    if HAS_ONORM:
        output_rstd = tl.rsqrt(tl.sum(output_values * output_values, axis=0) / D + eps)
        output_weight = tl.load(output_rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        output_values *= output_rstd * output_weight
    if HAS_POST_SCALE:
        output_scale = tl.load(post_scale + offsets_d, mask=mask_d, other=1.0).to(tl.float32)
        output_values *= output_scale
    if HAS_ONORM_BIAS:
        output_bias = tl.load(output_rms_bias + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        output_values += output_bias

    tl.store(output + row * D + offsets_d, output_values.to(output.dtype.element_ty), mask=mask_d)
    if RETURN_WEIGHTS:
        tl.store(lse + row, running_max + tl.log(running_sum))




def _validate_inputs(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None,
    output_rms_bias: torch.Tensor | None,
    post_scale: torch.Tensor | None,
) -> tuple[bool, torch.Size, int]:
    if not residuals:
        raise ValueError("residuals must contain at least one tensor")
    if len(residuals) > 16:
        raise ValueError(f"fused_attnres supports at most 16 residual sources, got {len(residuals)}")

    reference = residuals[0]
    reference_shape = reference.shape
    reference_device = reference.device
    reference_dtype = reference.dtype
    # PyTorch exposes Hygon HIP tensors through its is_cuda compatibility flag.
    if not reference.is_cuda:
        raise ValueError("fused_attnres requires Hygon accelerator tensors")
    if reference.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"fused_attnres supports float16 and bfloat16 residuals, got {reference.dtype}")
    if reference.ndim == 0:
        raise ValueError("residual tensors must have at least one dimension")
    hidden_size = reference_shape[-1]
    if hidden_size > 8192:
        raise ValueError(f"fused_attnres supports hidden_size <= 8192, got {hidden_size}")
    residuals_are_2d_contiguous = reference.ndim == 2 and reference.is_contiguous()

    for index, residual in enumerate(residuals):
        # The reference tensor was already checked above; avoid repeating its
        # metadata queries while retaining validation for unusual sequences.
        if index == 0 and residual is reference:
            continue
        if residual.shape != reference_shape:
            raise ValueError(
                f"residual {index} has shape {tuple(residual.shape)}, expected {tuple(reference.shape)}"
            )
        if residual.device != reference_device or residual.dtype != reference_dtype:
            raise ValueError("all residual tensors must have the same device and dtype")
        residuals_are_2d_contiguous = (
            residuals_are_2d_contiguous and residual.is_contiguous()
        )

    for name, tensor in (("query", query), ("rms_weight", rms_weight)):
        if tensor.device != reference_device or tensor.dtype != reference_dtype:
            raise ValueError(f"{name} must have the same device and dtype as residuals")
        if tensor.numel() != hidden_size:
            raise ValueError(f"{name} must contain {hidden_size} elements, got {tensor.numel()}")
    if output_rms_weight is not None:
        if output_rms_weight.device != reference.device or output_rms_weight.dtype != reference.dtype:
            raise ValueError("output_rms_weight must have the same device and dtype as residuals")
        if output_rms_weight.numel() != hidden_size:
            raise ValueError(
                f"output_rms_weight must contain {hidden_size} elements, got {output_rms_weight.numel()}"
            )
    if output_rms_bias is not None:
        if output_rms_bias.device != reference.device or output_rms_bias.dtype != reference.dtype:
            raise ValueError("output_rms_bias must have the same device and dtype as residuals")
        if output_rms_bias.numel() != hidden_size:
            raise ValueError(
                f"output_rms_bias must contain {hidden_size} elements, got {output_rms_bias.numel()}"
            )
    if post_scale is not None:
        if post_scale.device != reference.device or post_scale.dtype != reference.dtype:
            raise ValueError("post_scale must have the same device and dtype as residuals")
        if post_scale.numel() != hidden_size:
            raise ValueError(
                f"post_scale must contain {hidden_size} elements, got {post_scale.numel()}"
            )
    num_rows = (
        reference_shape[0]
        if residuals_are_2d_contiguous
        else (reference.numel() // hidden_size if hidden_size else 0)
    )
    return (
        residuals_are_2d_contiguous,
        reference_shape,
        num_rows,
    )


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
    output_rms_bias: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply forward-only Attention Residuals aggregation.

    Each residual source is RMS-normalized for scoring against ``query``. A
    softmax over the source/depth axis then mixes the original residuals. The
    implementation keeps an online softmax and the mixed hidden vector in
    registers, so every residual element is read once. ``output_rms_weight``
    optionally fuses the RMSNorm required by the following sublayer, and
    ``post_scale`` and ``output_rms_bias`` fuse following elementwise scale
    and bias operations into the same output tile.

    This API is intended for inference and does not provide an autograd formula.
    It is optimized for Kimi K3 shapes (``hidden_size=7168``, up to 9 sources),
    and supports up to 16 sources and hidden sizes up to 8192.
    """
    (
        residuals_are_2d_contiguous,
        output_shape,
        num_rows,
    ) = _validate_inputs(
        query, residuals, rms_weight, output_rms_weight, output_rms_bias, post_scale
    )
    hidden_size = output_shape[-1]
    if residuals_are_2d_contiguous:
        flat_residuals = tuple(residuals)
    else:
        flat_residuals = tuple(
            residual.reshape(-1, hidden_size).contiguous() for residual in residuals
        )
    if (
        len(flat_residuals) == 1
        and not return_weights
        and output_rms_weight is None
        and output_rms_bias is None
        and post_scale is None
    ):
        # With one source and no post operation, AttnRes is exactly a copy.
        output = torch.empty_like(flat_residuals[0])
        output.copy_(flat_residuals[0])
        return output if len(output_shape) == 2 else output.view(output_shape)

    query_flat = (
        query
        if query.ndim == 1 and query.is_contiguous()
        else query.reshape(-1).contiguous()
    )
    rms_weight_flat = (
        rms_weight
        if rms_weight.ndim == 1 and rms_weight.is_contiguous()
        else rms_weight.reshape(-1).contiguous()
    )
    output_rms_weight_flat = (
        None
        if output_rms_weight is None
        else (
            output_rms_weight
            if output_rms_weight.ndim == 1 and output_rms_weight.is_contiguous()
            else output_rms_weight.reshape(-1).contiguous()
        )
    )
    output_rms_bias_flat = (
        None
        if output_rms_bias is None
        else (
            output_rms_bias
            if output_rms_bias.ndim == 1 and output_rms_bias.is_contiguous()
            else output_rms_bias.reshape(-1).contiguous()
        )
    )
    post_scale_flat = (
        None
        if post_scale is None
        else (
            post_scale
            if post_scale.ndim == 1 and post_scale.is_contiguous()
            else post_scale.reshape(-1).contiguous()
        )
    )

    output = torch.empty_like(
        flat_residuals[0], memory_format=torch.contiguous_format
    )
    if return_weights:
        logit = torch.empty((len(residuals), num_rows), device=output.device, dtype=torch.float32)
        lse = torch.empty((num_rows,), device=output.device, dtype=torch.float32)
    else:
        logit = lse = None

    heuristic_config, num_warps, num_stages, kernel_meta = (
        _get_attnres_runtime_config()
    )
    join_reductions_max_rows = heuristic_config.get(
        "join_reductions_max_rows", 1024
    )

    _fused_attnres_fwd_kernel[(num_rows,)](
        query=query_flat,
        residuals=flat_residuals,
        rms_weight=rms_weight_flat,
        output_rms_weight=output_rms_weight_flat,
        output_rms_bias=output_rms_bias_flat,
        post_scale=post_scale_flat,
        output=output,
        logit=logit,
        lse=lse,
        N=num_rows,
        L=len(residuals),
        D=hidden_size,
        JOIN_REDUCTIONS=num_rows <= join_reductions_max_rows,
        eps=rms_eps,
        scale=scale,
        BD=triton.next_power_of_2(hidden_size),
        HAS_ONORM=output_rms_weight is not None,
        HAS_ONORM_BIAS=output_rms_bias is not None,
        HAS_POST_SCALE=post_scale is not None,
        RETURN_WEIGHTS=return_weights,
        **kernel_meta,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if len(output_shape) != 2:
        output = output.view(output_shape)
    if return_weights:
        probability = (logit * scale - lse[None, :]).exp()
        return output, probability.view(len(residuals), *output_shape[:-1])
    return output
