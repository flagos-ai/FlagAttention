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

"""Forward-only TLE implementation of dense Log-Linear Attention."""

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle

    HAS_TLE_LOG_LINEAR_ATTN = True
except ImportError:
    tle = None
    HAS_TLE_LOG_LINEAR_ATTN = False


CHUNK_SIZE = 64
BLOCK_KEY = 64
BLOCK_VALUE = 64
BLOCK_T = tl.constexpr(64)


@lru_cache(maxsize=None)
def _level_lut(device_index: int) -> torch.Tensor:
    device = torch.device("cuda", device_index)
    lut = torch.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.int32, device=device)
    rows = torch.arange(CHUNK_SIZE, device=device)[:, None]
    cols = torch.arange(CHUNK_SIZE, device=device)[None, :]
    for level in range(1, int(math.log2(CHUNK_SIZE)) + 1):
        half = 1 << (level - 1)
        anchor = rows - rows % half
        selected = (
            (rows % (2 * half) >= half)
            & (cols + half >= anchor)
            & (cols < anchor)
        )
        lut = torch.where(selected, level, lut)
    return lut


def _chunk_local_cumsum(g: torch.Tensor) -> torch.Tensor:
    batch, sequence, heads = g.shape
    return (
        g.view(batch, sequence // CHUNK_SIZE, CHUNK_SIZE, heads)
        .cumsum(dim=2)
        .view_as(g)
    )


@triton.jit
def _local_fwd_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    level_lut,
    local_output,
    states,
    chunk_gates,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    ASYNC_LOAD: tl.constexpr,
):
    chunk_index = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    tile_index = tl.program_id(2)
    key_tile = tile_index // NV
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H

    offsets_t = tl.arange(0, BLOCK_T)
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    token_offsets = chunk_index * BLOCK_T + offsets_t

    q_ptrs = q + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    k_ptrs = k + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    v_ptrs = (
        v
        + ((batch_index * T + token_offsets[:, None]) * H + head_index) * V
        + offsets_v[None, :]
    )
    b_q = tle.load(q_ptrs, is_async=ASYNC_LOAD).to(tl.float32)
    b_k = tle.load(k_ptrs, is_async=ASYNC_LOAD).to(tl.float32)
    b_v = tle.load(v_ptrs, is_async=ASYNC_LOAD).to(tl.float32)
    b_g = tl.load(g + (batch_index * T + token_offsets) * H + head_index)

    row = offsets_t[:, None]
    col = offsets_t[None, :]
    b_level = tl.load(level_lut + row * BLOCK_T + col)
    scale_ptrs = (
        level_scales
        + ((batch_index * T + token_offsets[:, None]) * H + head_index) * L
        + b_level
    )
    b_level_scale = tle.load(scale_ptrs, is_async=False).to(tl.float32)

    b_score = tl.dot(b_q.to(q.dtype.element_ty), tl.trans(b_k.to(k.dtype.element_ty)))
    b_score *= tl.exp(b_g[:, None] - b_g[None, :]) * b_level_scale
    b_score = tl.where(row >= col, b_score, 0.0)
    b_output = tl.dot(b_score.to(v.dtype.element_ty), b_v.to(v.dtype.element_ty))

    output_ptrs = (
        local_output
        + (
            (
                ((batch_index * T + token_offsets[:, None]) * H + head_index) * NK
                + key_tile
            )
            * V
        )
        + offsets_v[None, :]
    )
    tl.store(output_ptrs, b_output.to(local_output.dtype.element_ty))

    gate_last = tl.load(
        g + (batch_index * T + chunk_index * BLOCK_T + BLOCK_T - 1) * H + head_index
    )
    b_v_scaled = b_v * tl.exp(gate_last - b_g)[:, None]
    b_state = tl.dot(
        tl.trans(b_k.to(k.dtype.element_ty)),
        b_v_scaled.to(v.dtype.element_ty),
    )
    state_base = (
        (((batch_index * NT + chunk_index) * H + head_index) * K) * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    tl.store(states + state_base, b_state)
    if key_tile == 0 and value_tile == 0:
        tl.store(
            chunk_gates + (batch_index * NT + chunk_index) * H + head_index,
            gate_last,
        )


@triton.jit
def _merge_states_kernel(
    states,
    chunk_gates,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    LEVEL: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    ASYNC_LOAD: tl.constexpr,
):
    node_index = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    tile_index = tl.program_id(2)
    key_tile = tile_index // NV
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H

    right_end = (node_index + 1) * BLOCK - 1
    left_end = right_end - HALF
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)

    previous_level_base = tl.full((), LEVEL - 1, tl.int64) * B * NT * H * K * V
    left_base = (
        previous_level_base
        + (((batch_index * NT + left_end) * H + head_index) * K) * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    right_base = (
        previous_level_base
        + (((batch_index * NT + right_end) * H + head_index) * K) * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    left_state = tle.load(states + left_base, is_async=ASYNC_LOAD)
    right_state = tle.load(states + right_base, is_async=ASYNC_LOAD)

    offsets_chunk = tl.arange(0, HALF)
    gate_ptrs = (
        chunk_gates
        + (batch_index * NT + right_end - HALF + 1 + offsets_chunk) * H
        + head_index
    )
    right_decay = tl.exp(tl.sum(tl.load(gate_ptrs), axis=0))
    merged = left_state * right_decay + right_state

    output_level_base = tl.full((), LEVEL, tl.int64) * B * NT * H * K * V
    output_base = (
        output_level_base
        + (((batch_index * NT + right_end) * H + head_index) * K) * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    tl.store(states + output_base, merged)


@triton.jit
def _inter_fwd_kernel(
    q,
    g,
    level_scales,
    local_output,
    states,
    chunk_gates,
    output,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    MAX_BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    ASYNC_LOAD: tl.constexpr,
):
    chunk_index = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    tile_index = tl.program_id(2)
    key_tile = tile_index // NV
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H

    offsets_t = tl.arange(0, BLOCK_T)
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    token_offsets = chunk_index * BLOCK_T + offsets_t

    q_ptrs = q + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    local_ptrs = (
        local_output
        + (
            (
                ((batch_index * T + token_offsets[:, None]) * H + head_index) * NK
                + key_tile
            )
            * V
        )
        + offsets_v[None, :]
    )
    b_q = tle.load(q_ptrs, is_async=ASYNC_LOAD).to(tl.float32)
    b_output = tle.load(local_ptrs, is_async=ASYNC_LOAD).to(tl.float32)
    b_g = tl.load(g + (batch_index * T + token_offsets) * H + head_index)

    current_end = chunk_index - 1
    later_decay = tl.zeros([], dtype=tl.float32)
    for level in tl.static_range(NUM_LEVELS):
        block_size = 1 << level
        if (chunk_index & block_size) != 0:
            level_base = tl.full((), level, tl.int64) * B * NT * H * K * V
            state_base = (
                level_base
                + (((batch_index * NT + current_end) * H + head_index) * K) * V
                + offsets_k[:, None] * V
                + offsets_v[None, :]
            )
            b_state = tle.load(states + state_base, is_async=ASYNC_LOAD)
            scale_ptrs = (
                level_scales
                + ((batch_index * T + token_offsets) * H + head_index) * L
                + 7
                + level
            )
            b_scale = tl.load(scale_ptrs).to(tl.float32)
            b_q_scaled = b_q * (b_scale * tl.exp(b_g + later_decay))[:, None]
            b_output += tl.dot(
                b_q_scaled.to(q.dtype.element_ty),
                b_state.to(q.dtype.element_ty),
            )

            offsets_chunk = tl.arange(0, MAX_BLOCK)
            gate_ptrs = (
                chunk_gates
                + (batch_index * NT + current_end - block_size + 1 + offsets_chunk) * H
                + head_index
            )
            later_decay += tl.sum(
                tl.load(
                    gate_ptrs,
                    mask=offsets_chunk < block_size,
                    other=0.0,
                ),
                axis=0,
            )
            current_end -= block_size

    output_ptrs = (
        output
        + (
            (
                ((batch_index * T + token_offsets[:, None]) * H + head_index) * NK
                + key_tile
            )
            * V
        )
        + offsets_v[None, :]
    )
    tl.store(output_ptrs, b_output.to(output.dtype.element_ty))


@triton.jit
def _reduce_output_kernel(
    partial,
    output,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    NK: tl.constexpr,
    BV: tl.constexpr,
):
    token_index = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    value_tile = tl.program_id(2)
    batch_index = batch_head // H
    head_index = batch_head % H
    offsets_v = value_tile * BV + tl.arange(0, BV)

    accumulator = tl.zeros((BV,), dtype=tl.float32)
    for key_tile in tl.static_range(NK):
        partial_ptrs = (
            partial
            + ((((batch_index * T + token_index) * H + head_index) * NK + key_tile) * V)
            + offsets_v
        )
        accumulator += tl.load(partial_ptrs).to(tl.float32)

    output_ptrs = (
        output + ((batch_index * T + token_index) * H + head_index) * V + offsets_v
    )
    tl.store(output_ptrs, accumulator.to(output.dtype.element_ty))


def _validate_inputs(q, k, v, g, level_scales):
    tensors = (q, k, v, g, level_scales)
    if not HAS_TLE_LOG_LINEAR_ATTN:
        raise RuntimeError(
            "chunk_log_linear_attn requires a Triton build with triton.experimental.tle"
        )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("chunk_log_linear_attn requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all inputs must be on the same CUDA device")
    if any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("chunk_log_linear_attn is forward-only and does not support autograd")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("chunk_log_linear_attn supports BF16 q, k, and v")
    if g.dtype != torch.float32:
        raise ValueError("g must use float32")
    if level_scales.dtype != q.dtype:
        raise ValueError("level_scales must have the same dtype as q")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors with matching q/k shapes")
    if q.shape[2] != 1 or k.shape[2] != 1:
        raise ValueError("chunk_log_linear_attn requires one shared Q/K head")
    batch, sequence, _, key_dim = q.shape
    heads, value_dim = v.shape[2], v.shape[-1]
    if v.shape[:2] != (batch, sequence):
        raise ValueError("v must have shape [B, T, H, V]")
    if g.shape != (batch, sequence, heads):
        raise ValueError("g must have shape [B, T, H]")
    if sequence % CHUNK_SIZE != 0 or sequence & (sequence - 1):
        raise ValueError("sequence must be a power of two and divisible by 64")
    if key_dim not in (64, 128, 256) or value_dim not in (64, 128, 256):
        raise ValueError("the TLE path supports K/V in {64, 128, 256}")
    if level_scales.ndim != 4:
        raise ValueError("level_scales must be rank 4")
    levels = level_scales.shape[-1]
    if level_scales.shape[:3] != (batch, sequence, heads):
        raise ValueError("level_scales must have shape [B, T, H, L]")
    if levels != math.ceil(math.log2(sequence)) + 1:
        raise ValueError("invalid level_scales level count")
    return batch, sequence, heads, key_dim, value_dim, levels


def _prepare_tle_forward(
    q,
    k,
    v,
    g,
    level_scales,
    async_load=False,
    block_value=BLOCK_VALUE,
    local_warps=4,
    merge_warps=4,
    inter_warps=4,
    reduction_warps=4,
    num_stages=2,
):
    batch, sequence, heads, key_dim, value_dim, levels = _validate_inputs(
        q, k, v, g, level_scales
    )
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    level_scales = level_scales.contiguous()
    num_chunks = sequence // CHUNK_SIZE
    num_levels = max(1, math.ceil(math.log2(num_chunks)))
    num_key_tiles = key_dim // BLOCK_KEY
    num_value_tiles = value_dim // block_value
    num_tiles = num_key_tiles * num_value_tiles
    local_g = _chunk_local_cumsum(g.contiguous())
    level_lut = _level_lut(q.device.index or 0)
    partial_shape = (
        batch,
        sequence,
        heads,
        num_key_tiles,
        value_dim,
    )
    local_output = torch.empty(partial_shape, device=v.device, dtype=v.dtype)
    output_partial = torch.empty_like(local_output)
    output = output_partial.squeeze(3) if num_key_tiles == 1 else torch.empty_like(v)
    states = torch.empty(
        num_levels,
        batch,
        num_chunks,
        heads,
        key_dim,
        value_dim,
        device=v.device,
        dtype=torch.float32,
    )
    chunk_gates = torch.empty(
        batch, num_chunks, heads, device=v.device, dtype=torch.float32
    )

    def run():
        _local_fwd_kernel[(num_chunks, batch * heads, num_tiles)](
            q,
            k,
            v,
            local_g,
            level_scales,
            level_lut,
            local_output,
            states,
            chunk_gates,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            L=levels,
            NT=num_chunks,
            BK=BLOCK_KEY,
            BV=block_value,
            NK=num_key_tiles,
            NV=num_value_tiles,
            ASYNC_LOAD=async_load,
            num_warps=local_warps,
            num_stages=num_stages,
        )
        for level in range(1, num_levels):
            block = 1 << level
            half = block >> 1
            _merge_states_kernel[(num_chunks // block, batch * heads, num_tiles)](
                states,
                chunk_gates,
                B=batch,
                H=heads,
                K=key_dim,
                V=value_dim,
                NT=num_chunks,
                LEVEL=level,
                HALF=half,
                BLOCK=block,
                BK=BLOCK_KEY,
                BV=block_value,
                NV=num_value_tiles,
                ASYNC_LOAD=async_load,
                num_warps=merge_warps,
                num_stages=num_stages,
            )
        _inter_fwd_kernel[(num_chunks, batch * heads, num_tiles)](
            q,
            local_g,
            level_scales,
            local_output,
            states,
            chunk_gates,
            output_partial,
            B=batch,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            L=levels,
            NT=num_chunks,
            NUM_LEVELS=num_levels,
            MAX_BLOCK=1 << (num_levels - 1),
            BK=BLOCK_KEY,
            BV=block_value,
            NK=num_key_tiles,
            NV=num_value_tiles,
            ASYNC_LOAD=async_load,
            num_warps=inter_warps,
            num_stages=num_stages,
        )
        if num_key_tiles > 1:
            _reduce_output_kernel[(sequence, batch * heads, num_value_tiles)](
                output_partial,
                output,
                T=sequence,
                H=heads,
                V=value_dim,
                NK=num_key_tiles,
                BV=block_value,
                num_warps=reduction_warps,
            )

    return run, output


def chunk_log_linear_attn(q, k, v, g, level_scales):
    """Run dense, forward-only Log-Linear Attention with TLE kernels."""
    run, output = _prepare_tle_forward(q, k, v, g, level_scales)
    run()
    return output


__all__ = ["chunk_log_linear_attn"]
