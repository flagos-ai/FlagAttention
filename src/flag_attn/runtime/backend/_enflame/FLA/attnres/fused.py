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

"""Forward-only Attention Residuals for the Enflame GCU backend.

One program aggregates one flattened row using FP32 online softmax. Residual
sources use an import-time generated scalar-pointer launcher ABI.
"""

from __future__ import annotations

import inspect
import linecache
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

_MAX_RESIDUAL_SOURCES = 16
_MAX_HIDDEN_SIZE = 8192
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def _attnres_fwd_impl(
    query,
    residuals,
    rms_weight,
    output_rms_weight,
    output,
    logits,
    logsumexp,
    N,
    L: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    RETURN_WEIGHTS: tl.constexpr,
):
    """Fuse RMS scoring, online source mixing and optional output RMSNorm."""
    row = tl.program_id(0).to(tl.int64)
    offsets_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    mask_d = offsets_d < D

    query_values = tl.load(query + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
    weight_values = tl.load(rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
    query_weight = query_values * weight_values

    running_max = tl.full([], -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros([], dtype=tl.float32)
    output_values = tl.zeros([BD], dtype=tl.float32)

    # Each statically selected source has its own scalar base pointer.
    # Pointer-valued tl.where lowers to an unsupported arith.select on GCU.
    for source in tl.static_range(0, L):
        residual_values = tl.load(
            residuals[source] + row * D + offsets_d,
            mask=mask_d,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(residual_values * residual_values, axis=0) / D + eps)
        source_logit = tl.sum(residual_values * query_weight, axis=0) * rstd
        source_score = source_logit * scale

        previous_max = running_max
        running_max = tl.maximum(running_max, source_score)
        rescale = tl.exp(previous_max - running_max)
        source_weight = tl.exp(source_score - running_max)
        running_sum = running_sum * rescale + source_weight
        output_values = output_values * rescale + source_weight * residual_values

        if RETURN_WEIGHTS:
            # Store unscaled logits; logsumexp below includes scale.
            tl.store(logits + source * N + row, source_logit)

    output_values /= running_sum
    if HAS_ONORM:
        output_rstd = tl.rsqrt(tl.sum(tl.where(mask_d, output_values * output_values, 0.0), axis=0) / D + eps)
        output_weight = tl.load(output_rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        output_values *= output_rstd * output_weight

    tl.store(output + row * D + offsets_d, output_values.to(output.dtype.element_ty), mask=mask_d)
    if RETURN_WEIGHTS:
        tl.store(logsumexp + row, running_max + tl.log(running_sum))


def _make_attnres_fwd_kernel():
    """Generate the scalar-pointer launcher ABI required by the GCU backend."""
    residual_names = [f"residual_{index}" for index in range(_MAX_RESIDUAL_SOURCES)]
    declarations = []
    arguments = []
    for parameter in inspect.signature(_attnres_fwd_impl.fn).parameters.values():
        if parameter.name == "residuals":
            declarations.extend(residual_names)
            arguments.append("(" + ", ".join(residual_names) + ",)")
        else:
            declaration = parameter.name
            if parameter.annotation is not inspect.Parameter.empty:
                if parameter.annotation not in (tl.constexpr, "tl.constexpr"):
                    raise TypeError(f"Unsupported kernel annotation: {parameter.annotation}")
                declaration += ": tl.constexpr"
            declarations.append(declaration)
            arguments.append(parameter.name)

    source = (
        "@triton.jit\n"
        f"def _fused_attnres_fwd_kernel({', '.join(declarations)}):\n"
        f"    _attnres_fwd_impl({', '.join(arguments)})\n"
    )
    # Triton captures Python source with inspect.getsource during decoration.
    filename = f"{__file__}::<attnres-scalar-pointer-abi>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = {
        "__name__": __name__,
        "triton": triton,
        "tl": tl,
        "_attnres_fwd_impl": _attnres_fwd_impl,
    }
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    return namespace["_fused_attnres_fwd_kernel"]


_fused_attnres_fwd_kernel = _make_attnres_fwd_kernel()


def _validate_inputs(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None,
) -> None:
    """Check backend, shape and dtype requirements without reading device data."""
    if not residuals:
        raise ValueError("residuals must contain at least one tensor")
    if len(residuals) > _MAX_RESIDUAL_SOURCES:
        raise ValueError(
            f"fused_attnres supports at most {_MAX_RESIDUAL_SOURCES} residual sources, got {len(residuals)}"
        )

    reference = residuals[0]
    if reference.device.type != "gcu":
        raise ValueError("fused_attnres requires GCU tensors")
    if reference.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"fused_attnres supports float16 and bfloat16 residuals, got {reference.dtype}")
    if reference.ndim == 0:
        raise ValueError("residual tensors must have at least one dimension")
    hidden_size = reference.shape[-1]
    if hidden_size > _MAX_HIDDEN_SIZE:
        raise ValueError(f"fused_attnres supports hidden_size <= {_MAX_HIDDEN_SIZE}, got {hidden_size}")

    for index, residual in enumerate(residuals):
        if residual.shape != reference.shape:
            raise ValueError(f"residual {index} has shape {tuple(residual.shape)}, expected {tuple(reference.shape)}")
        if residual.device != reference.device or residual.dtype != reference.dtype:
            raise ValueError("all residual tensors must have the same device and dtype")

    vectors = (("query", query), ("rms_weight", rms_weight))
    if output_rms_weight is not None:
        vectors += (("output_rms_weight", output_rms_weight),)
    for name, tensor in vectors:
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            raise ValueError(f"{name} must have the same device and dtype as residuals")
        if tensor.numel() != hidden_size:
            raise ValueError(f"{name} must contain {hidden_size} elements, got {tensor.numel()}")


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply forward-only Attention Residuals on GCU.

    Score RMS-normalized sources against ``query``, apply ``scale`` and softmax
    over sources, then mix the original residuals in FP32. ``output_rms_weight``
    optionally applies output RMSNorm.

    Supports 1 to 16 equally shaped FP16/BF16 residuals of shape ``(..., D)``,
    with ``D <= 8192``. Query and normalization weights contain D elements and
    match the residual device and dtype. Output has the input shape and dtype;
    ``return_weights=True`` also returns FP32 probabilities of shape
    ``(len(residuals), *residuals[0].shape[:-1])``.

    No autograd formula is provided.
    """
    _validate_inputs(query, residuals, rms_weight, output_rms_weight)

    output_shape = residuals[0].shape
    hidden_size = output_shape[-1]
    num_sources = len(residuals)
    has_output_norm = output_rms_weight is not None
    flat_residuals = tuple(residual.reshape(-1, hidden_size).contiguous() for residual in residuals)
    num_rows = flat_residuals[0].shape[0]
    # Pad pointer arguments only; unused slots do not copy residual data.
    residual_ptrs = flat_residuals + (flat_residuals[0],) * (_MAX_RESIDUAL_SOURCES - num_sources)

    output = torch.empty_like(flat_residuals[0])
    if return_weights:
        logits = torch.empty((num_sources, num_rows), device=output.device, dtype=torch.float32)
        logsumexp = torch.empty((num_rows,), device=output.device, dtype=torch.float32)
    else:
        logits = logsumexp = None

    # Preserve the measured fixed GCU launch without autotuner dispatch.
    _fused_attnres_fwd_kernel[(num_rows,)](
        query.reshape(-1).contiguous(),
        *residual_ptrs,
        rms_weight.reshape(-1).contiguous(),
        None if output_rms_weight is None else output_rms_weight.reshape(-1).contiguous(),
        output,
        logits,
        logsumexp,
        N=num_rows,
        L=num_sources,
        D=hidden_size,
        eps=rms_eps,
        scale=scale,
        BD=triton.next_power_of_2(hidden_size),
        HAS_ONORM=has_output_norm,
        RETURN_WEIGHTS=return_weights,
        num_warps=1,
        num_stages=1,
    )

    output = output.view(output_shape)
    if return_weights:
        probabilities = (logits * scale - logsumexp[None, :]).exp()
        return output, probabilities.view(num_sources, *output_shape[:-1])
    return output


__all__ = ["fused_attnres"]
