"""Rotary positional embedding with a backend-neutral public interface."""

from __future__ import annotations

from typing import Optional

import torch


def _rotate(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool
) -> torch.Tensor:
    if interleaved:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat(
        (first * cos - second * sin, first * sin + second * cos), dim=-1
    )


def _native_rotary(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool,
) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    selected_cos = cos.index_select(0, positions.reshape(-1)).to(dtype=x.dtype)
    selected_sin = sin.index_select(0, positions.reshape(-1)).to(dtype=x.dtype)
    selected_cos = selected_cos.view(flat.shape[0], 1, -1)
    selected_sin = selected_sin.view(flat.shape[0], 1, -1)
    return _rotate(flat, selected_cos, selected_sin, interleaved).view_as(x)


def _mlu_rotary(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool,
) -> torch.Tensor:
    """Use FlagTree's Cambricon Triton kernel when it is installed."""
    try:
        from triton.ops.apply_rotary import apply_rotary
    except (ImportError, ModuleNotFoundError):
        return _native_rotary(x, positions, cos, sin, interleaved)

    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    flat_positions = positions.reshape(-1).to(device=x.device, dtype=torch.int32)
    output = torch.empty_like(flat)
    cu_seqlens = torch.tensor([0, flat.shape[0]], device=x.device, dtype=torch.int32)
    apply_rotary(
        output,
        flat,
        cos,
        sin,
        BLOCK_M=min(32, max(1, flat.shape[0])),
        token_offsets=flat_positions.view(1, -1),
        cu_seqlens=cu_seqlens,
        max_seqlen=flat.shape[0],
        interleaved=interleaved,
    )
    return output.view_as(x)


def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    *,
    interleaved: bool = False,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    ``q`` and ``k`` use ``[..., heads, head_dim]`` layout. ``cos`` and ``sin``
    have shape ``[max_position, head_dim // 2]`` and ``position_ids`` matches
    the leading token dimensions. On MLU, the FlagTree Cambricon Triton
    implementation is selected automatically when available.
    """
    if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have matching token and head dimensions")
    if cos.shape != sin.shape or cos.shape[-1] * 2 != q.shape[-1]:
        raise ValueError("cos and sin must be [max_position, head_dim // 2]")
    if position_ids is None:
        position_ids = torch.arange(q.shape[-3], device=q.device).expand(q.shape[:-2])
    if tuple(position_ids.shape) != tuple(q.shape[:-2]):
        raise ValueError("position_ids must match q and k leading dimensions")
    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)
    position_ids = position_ids.to(device=q.device, dtype=torch.long)

    kernel = _mlu_rotary if q.device.type == "mlu" else _native_rotary
    q_out = kernel(q, position_ids, cos, sin, interleaved)
    k_out = kernel(k, position_ids, cos, sin, interleaved)
    if inplace:
        q.copy_(q_out)
        k.copy_(k_out)
        return q, k
    return q_out, k_out


__all__ = ["rotary_embedding"]
