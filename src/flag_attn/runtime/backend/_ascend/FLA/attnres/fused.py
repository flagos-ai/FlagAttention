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
pointer, which avoids materialising a stacked copy of the residuals. A single
source is read directly, and the launch goes through a cached raw launcher
because the device time of this one-kernel operator is small enough that
Triton's per-call Python launch path dominates.

The single-row kernel is bound by fp32 vector throughput rather than bandwidth,
so with several sources it rotates two residual buffers to overlap the next
source's load with the current source's arithmetic, and it folds the trailing
RMSNorm into the softmax denominator to keep that epilogue to one full-width
pass.

One row per program caps that kernel near 380 GB/s of the ~1.08 TB/s this part
can reach, so wide row counts (>= ``_TWO_PASS_MIN_ROWS``) are handed to
``two_pass``: row-batched kernels that score every source once with [BR, BDB]
tiles and then mix them in a second pass, both at the HBM ceiling.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from . import fast_launch as _fast_launch
from . import two_pass as _two_pass

_INT32_MAX = 2**31 - 1

# The row-batched path reads the sources twice, so it only pays off once the row
# count keeps both passes at the HBM ceiling. Measured on D=7168 with the
# benchmark protocol, cost relative to the single-row kernel: 1.45x at N=128/256,
# 1.09x at N=512, 0.70x at N=1024, 0.39x at N=2048 and 0.87x for 9x8192.
_TWO_PASS_MIN_ROWS = 1024

_BD_CACHE: dict = {}
_LAUNCH_SITE: dict = {}

# Small per-call address tables are reused when a caller keeps the same residual
# tensors across steps (the common decode/benchmark case); the key holds the
# exact addresses, so a rebuilt allocation can never alias a stale table.
_PTR_TABLE_CACHE: dict = {}
_MAX_PTR_TABLES = 64


def _flat_rows(tensor: torch.Tensor, hidden: int) -> torch.Tensor:
    """Return a ``(rows, hidden)`` view without copying when already in layout."""
    if tensor.ndim == 2 and tensor.is_contiguous():
        return tensor
    return tensor.reshape(-1, hidden).contiguous()


def _flat_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Return a flat contiguous view without copying when already in layout."""
    if tensor.ndim == 1 and tensor.is_contiguous():
        return tensor
    return tensor.reshape(-1).contiguous()


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
    SINGLE: tl.constexpr,
    PREFETCH: tl.constexpr,
    NOMAX: tl.constexpr,
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

    if PREFETCH:
        # Rotating double buffer: the load of source s + 1 is issued before the
        # arithmetic of source s, so its DMA overlaps the fp32 vector pipeline
        # instead of serialising with it.
        next_ptr = tl.cast(tl.load(residual_ptrs), residual0.dtype)
        next_values = tl.load(next_ptr + row * D + offsets_d, mask=mask_d, other=0.0)

    for source in tl.range(0, L):
        if PREFETCH:
            residual_values = next_values.to(tl.float32)
            if source + 1 < L:
                # ``residual0`` only supplies the element pointer type; every
                # source is a separate tensor indexed through the host-built
                # address table.
                next_ptr = tl.cast(tl.load(residual_ptrs + source + 1), residual0.dtype)
                next_values = tl.load(
                    next_ptr + row * D + offsets_d, mask=mask_d, other=0.0
                )
        elif SINGLE:
            # One source: read it directly, so no address table is needed.
            residual_values = tl.load(
                residual0 + row * D + offsets_d, mask=mask_d, other=0.0
            ).to(tl.float32)
        else:
            residual_ptr = tl.cast(tl.load(residual_ptrs + source), residual0.dtype)
            residual_values = tl.load(
                residual_ptr + row * D + offsets_d, mask=mask_d, other=0.0
            ).to(tl.float32)

        rstd = tl.rsqrt(tl.sum(residual_values * residual_values) / D + eps)
        source_logit = tl.sum(residual_values * query_weight) * rstd
        source_score = source_logit * scale

        if NOMAX:
            # With the standard scale (1/sqrt(D)) the scores are O(1) after RMS
            # normalisation, so the max subtraction and its rescale tile are
            # redundant; dropping them removes one full-width vector op.
            probability = tl.exp(source_score)
            running_sum += probability
            output_values += probability * residual_values
        else:
            previous_max = running_max
            running_max = tl.maximum(running_max, source_score)
            rescale = tl.exp(previous_max - running_max)
            probability = tl.exp(source_score - running_max)
            running_sum = running_sum * rescale + probability
            output_values = output_values * rescale + probability * residual_values

        if RETURN_WEIGHTS:
            tl.store(logit + source * N + row, source_logit)

    if HAS_ONORM:
        # Fold the trailing RMSNorm into the softmax denominator:
        # (out / denom) * rsqrt(sum((out / denom)^2) / D + eps) * ow
        #   == out * ow * rsqrt(sum(out^2) / D + eps * denom^2),
        # which saves the full-width division by ``running_sum`` and its
        # temporary.
        squared_norm = tl.sum(tl.where(mask_d, output_values * output_values, 0.0)) / D
        output_weight = tl.load(output_rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        output_values = output_values * output_weight * tl.rsqrt(
            squared_norm + eps * running_sum * running_sum
        )
    else:
        output_values = output_values / running_sum

    tl.store(output + row * D + offsets_d, output_values.to(output.dtype.element_ty), mask=mask_d)
    if RETURN_WEIGHTS:
        if NOMAX:
            tl.store(lse + row, tl.log(running_sum))
        else:
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


def _prefetch_enabled(num_sources: int, has_output_norm: bool) -> bool:
    """Prefetch overlaps a source read with the current one.

    The double buffer does not fit in UB together with the output-norm weight
    tile on this device, so output-norm specializations keep the plain loop.
    """
    return num_sources > 1 and not has_output_norm


def _launch(grid, values, constexprs) -> None:
    """Launch through the cached raw launcher; fall back to Triton when needed."""
    key = (
        grid,
        values[0].dtype,
        values[0].device,
        constexprs["L"],
        constexprs["D"],
        constexprs["BD"],
        constexprs["eps"],
        constexprs["scale"],
        constexprs["HAS_ONORM"],
        constexprs["RETURN_WEIGHTS"],
        constexprs["SINGLE"],
        constexprs["PREFETCH"],
        constexprs["NOMAX"],
    )
    entry = _LAUNCH_SITE.get(key)
    if entry is None:
        entry = _fast_launch.build(_fused_attnres_fwd_kernel, grid, values, constexprs)
        _LAUNCH_SITE[key] = entry
    entry(values)


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
    flat_residuals = [_flat_rows(residual, hidden_size) for residual in residuals]
    num_rows = flat_residuals[0].shape[0]

    if num_rows * hidden_size > _INT32_MAX:
        raise ValueError(
            f"fused_attnres on Ascend indexes residuals with int32 offsets, got "
            f"{num_rows * hidden_size} elements per source"
        )

    # triton-ascend has no tuple-of-tensor kernel args; either pass the source
    # base addresses through a small int64 table (and cast them back in the
    # kernel) or, for a single source, let the kernel read ``residual0``.
    single = len(flat_residuals) == 1
    residual_ptrs = flat_residuals[0] if single else _residual_ptr_table(flat_residuals)

    plan = None
    if num_rows >= _TWO_PASS_MIN_ROWS and not single:
        plan = _two_pass.plan(
            hidden_size, num_rows, len(residuals), output_rms_weight is not None
        )

    output = torch.empty_like(flat_residuals[0])
    if return_weights or plan is not None:
        # The row-batched path keeps the weights out of the unified buffer, so
        # it always materialises the logits; they are also what
        # ``return_weights=True`` hands back.
        logit = torch.empty((len(residuals), num_rows), device=output.device, dtype=torch.float32)
        lse = torch.empty((num_rows,), device=output.device, dtype=torch.float32)
    else:
        logit = lse = None

    if num_rows > 0:
        bd = _BD_CACHE.get(hidden_size)
        if bd is None:
            bd = triton.next_power_of_2(hidden_size)
            _BD_CACHE[hidden_size] = bd
        values = [
            _flat_1d(query),
            flat_residuals[0],
            residual_ptrs,
            _flat_1d(rms_weight),
            None if output_rms_weight is None else _flat_1d(output_rms_weight),
            output,
            logit,
            lse,
            num_rows,
        ]
        # The rotating buffer costs one more residual tile of unified buffer,
        # which does not fit beside the output-norm tile (208KB > 192KB), so
        # prefetching stays off when the RMSNorm epilogue is fused in. ``single``
        # never prefetches: ``residual_ptrs`` is the residual itself there.
        constexprs = dict(
            L=len(residuals),
            D=hidden_size,
            BD=bd,
            eps=rms_eps,
            scale=scale,
            HAS_ONORM=output_rms_weight is not None,
            RETURN_WEIGHTS=return_weights,
            SINGLE=single,
            PREFETCH=_prefetch_enabled(len(residuals), output_rms_weight is not None),
            NOMAX=scale <= hidden_size ** -0.5,
        )
        if plan is not None:
            _two_pass.launch(
                values,
                plan,
                eps=rms_eps,
                scale=scale,
                has_onorm=output_rms_weight is not None,
                nomax=constexprs["NOMAX"],
            )
        else:
            _launch((num_rows,), values, constexprs)

    if output.shape != output_shape:
        output = output.view(output_shape)
    if return_weights:
        probability = (logit * scale - lse[None, :]).exp()
        return output, probability.view(len(residuals), *output_shape[:-1])
    return output


__all__ = ["fused_attnres"]
