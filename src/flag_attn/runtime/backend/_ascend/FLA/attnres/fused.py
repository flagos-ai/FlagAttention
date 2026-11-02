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

"""Ascend Triton implementation of fused Attention Residuals.

The generic CUDA implementation in ``flag_attn.FLA.attnres.fused`` keeps the
sources in a tuple kernel argument. triton-ascend has no tuple-of-tensor
arguments, so the sources stay as separate tensors: the host passes a small
int64 table of their base addresses and the kernel casts each entry back to a
pointer, which avoids materialising a stacked copy of the residuals.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

_INT32_MAX = 2**31 - 1

# Small per-call address tables are reused when a caller keeps the same residual
# tensors across steps (the common decode/benchmark case); the key holds the
# exact addresses, so a rebuilt allocation can never alias a stale table.
_PTR_TABLE_CACHE: dict = {}
_MAX_PTR_TABLES = 64


def _residual_ptr_table(flat_residuals: Sequence[torch.Tensor]) -> torch.Tensor:
    ptrs = tuple(residual.data_ptr() for residual in flat_residuals)
    key = (flat_residuals[0].device, ptrs)
    table = _PTR_TABLE_CACHE.get(key)
    if table is None:
        table = torch.tensor(list(ptrs), dtype=torch.int64, device=flat_residuals[0].device)
        if len(_PTR_TABLE_CACHE) >= _MAX_PTR_TABLES:
            _PTR_TABLE_CACHE.clear()
        _PTR_TABLE_CACHE[key] = table
    return table


@triton.jit
def _fused_attnres_fwd_kernel(
    query,
    residual0,
    residual_ptrs,
    rms_weight,
    output_rms_weight,
    output,
    logit,
    lse,
    N,
    L: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    RETURN_WEIGHTS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets_d = tl.arange(0, BD)
    mask_d = offsets_d < D

    query_values = tl.load(query + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
    weight_values = tl.load(rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
    query_weight = query_values * weight_values

    running_max = tl.full([], -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros([], dtype=tl.float32)
    output_values = tl.zeros([BD], dtype=tl.float32)

    for source in tl.static_range(0, L):
        # ``residual0`` only supplies the element pointer type; every source is
        # a separate tensor indexed through the host-built address table.
        residual_ptr = tl.cast(tl.load(residual_ptrs + source), residual0.dtype)
        residual_values = tl.load(
            residual_ptr + row * D + offsets_d, mask=mask_d, other=0.0
        ).to(tl.float32)

        rstd = tl.rsqrt(tl.sum(residual_values * residual_values) / D + eps)
        source_logit = tl.sum(residual_values * query_weight) * rstd
        source_score = source_logit * scale

        previous_max = running_max
        running_max = tl.maximum(running_max, source_score)
        rescale = tl.exp(previous_max - running_max)
        probability = tl.exp(source_score - running_max)
        running_sum = running_sum * rescale + probability
        output_values = output_values * rescale + probability * residual_values

        if RETURN_WEIGHTS:
            tl.store(logit + source * N + row, source_logit)

    output_values /= running_sum
    if HAS_ONORM:
        output_rstd = tl.rsqrt(tl.sum(tl.where(mask_d, output_values * output_values, 0.0)) / D + eps)
        output_weight = tl.load(output_rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        output_values *= output_rstd * output_weight

    tl.store(output + row * D + offsets_d, output_values.to(output.dtype.element_ty), mask=mask_d)
    if RETURN_WEIGHTS:
        tl.store(lse + row, running_max + tl.log(running_sum))


def _validate_inputs(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None,
) -> None:
    if not residuals:
        raise ValueError("residuals must contain at least one tensor")
    if len(residuals) > 16:
        raise ValueError(f"fused_attnres supports at most 16 residual sources, got {len(residuals)}")

    reference = residuals[0]
    if reference.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"fused_attnres supports float16 and bfloat16 residuals, got {reference.dtype}")
    if reference.ndim == 0:
        raise ValueError("residual tensors must have at least one dimension")
    hidden_size = reference.shape[-1]
    if hidden_size > 8192:
        raise ValueError(f"fused_attnres supports hidden_size <= 8192, got {hidden_size}")

    for index, residual in enumerate(residuals):
        if residual.shape != reference.shape:
            raise ValueError(
                f"residual {index} has shape {tuple(residual.shape)}, expected {tuple(reference.shape)}"
            )
        if residual.device != reference.device or residual.dtype != reference.dtype:
            raise ValueError("all residual tensors must have the same device and dtype")

    for name, tensor in (("query", query), ("rms_weight", rms_weight)):
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
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


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply forward-only Attention Residuals aggregation on Ascend NPUs.

    Semantics follow ``flag_attn.FLA.attnres.fused.fused_attnres``: each residual
    source is RMS-normalized for scoring against ``query``, a softmax over the
    source axis mixes the original residuals with an online (streaming) softmax,
    and ``output_rms_weight`` optionally fuses the following RMSNorm.
    """
    _validate_inputs(query, residuals, rms_weight, output_rms_weight)

    output_shape = residuals[0].shape
    hidden_size = output_shape[-1]
    flat_residuals = [residual.reshape(-1, hidden_size).contiguous() for residual in residuals]
    num_rows = flat_residuals[0].shape[0]

    # triton-ascend has no tuple-of-tensor kernel args; pass the source base
    # addresses as a small int64 table and cast them back in the kernel, so no
    # stacked copy of the residuals is needed.
    if num_rows * hidden_size > _INT32_MAX:
        raise ValueError(
            f"fused_attnres on Ascend indexes residuals with int32 offsets, got "
            f"{num_rows * hidden_size} elements per source"
        )
    residual_ptrs = _residual_ptr_table(flat_residuals)

    output = torch.empty_like(flat_residuals[0])
    if return_weights:
        logit = torch.empty((len(residuals), num_rows), device=output.device, dtype=torch.float32)
        lse = torch.empty((num_rows,), device=output.device, dtype=torch.float32)
    else:
        logit = lse = None

    if num_rows > 0:
        _fused_attnres_fwd_kernel[(num_rows,)](
            query.reshape(-1).contiguous(),
            flat_residuals[0],
            residual_ptrs,
            rms_weight.reshape(-1).contiguous(),
            None if output_rms_weight is None else output_rms_weight.reshape(-1).contiguous(),
            output,
            logit,
            lse,
            num_rows,
            L=len(residuals),
            D=hidden_size,
            BD=triton.next_power_of_2(hidden_size),
            eps=rms_eps,
            scale=scale,
            HAS_ONORM=output_rms_weight is not None,
            RETURN_WEIGHTS=return_weights,
        )

    output = output.view(output_shape)
    if return_weights:
        probability = (logit * scale - lse[None, :]).exp()
        return output, probability.view(len(residuals), *output_shape[:-1])
    return output


__all__ = ["fused_attnres"]
