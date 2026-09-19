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

"""TileLang forward baseline adapted from FLA Log-Linear Attention.

The source algorithm follows ``fla/ops/log_linear_attn/chunk.py``, which is
licensed under the MIT License.
"""

import math
from functools import lru_cache

import torch
import tilelang
import tilelang.language as T


CHUNK_SIZE = 64
BLOCK_KEY = 64
BLOCK_VALUE = 64


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
        g.float()
        .view(batch, sequence // CHUNK_SIZE, CHUNK_SIZE, heads)
        .cumsum(dim=2)
        .view_as(g)
    )


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def _log_linear_fwd_kernel(
    batch: int,
    sequence: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    levels: int,
    dtype: T.dtype = T.bfloat16,
):
    chunk = CHUNK_SIZE
    assert sequence % chunk == 0
    assert key_dim % BLOCK_KEY == 0
    assert value_dim % BLOCK_VALUE == 0

    num_chunks = sequence // chunk
    num_states = max(1, math.ceil(math.log2(num_chunks)))
    num_intra_levels = int(math.log2(chunk)) + 1
    num_key_tiles = key_dim // BLOCK_KEY
    num_value_tiles = value_dim // BLOCK_VALUE
    accum_dtype = T.float32

    @T.prim_func
    def main(
        q: T.Tensor((batch, sequence, 1, key_dim), dtype),
        k: T.Tensor((batch, sequence, 1, key_dim), dtype),
        v: T.Tensor((batch, sequence, heads, value_dim), dtype),
        g: T.Tensor((batch, sequence, heads), T.float32),
        level_scales: T.Tensor((batch, sequence, heads, levels), dtype),
        level_lut: T.Tensor((chunk, chunk), T.int32),
        output_partial: T.Tensor(
            (batch, sequence, heads, num_key_tiles, value_dim), dtype
        ),
    ):
        with T.Kernel(
            num_value_tiles, num_key_tiles, batch * heads, threads=256
        ) as (value_tile, key_tile, batch_head):
            batch_index = batch_head // heads
            head_index = batch_head % heads

            q_shared = T.alloc_shared((chunk, BLOCK_KEY), dtype)
            q_scaled_shared = T.alloc_shared((chunk, BLOCK_KEY), dtype)
            k_shared = T.alloc_shared((chunk, BLOCK_KEY), dtype)
            v_shared = T.alloc_shared((chunk, BLOCK_VALUE), dtype)
            v_scaled_shared = T.alloc_shared((chunk, BLOCK_VALUE), dtype)
            score_shared = T.alloc_shared((chunk, chunk), dtype)
            state_shared = T.alloc_shared((BLOCK_KEY, BLOCK_VALUE), dtype)
            output_shared = T.alloc_shared((chunk, BLOCK_VALUE), dtype)

            score = T.alloc_fragment((chunk, chunk), accum_dtype)
            output_acc = T.alloc_fragment((chunk, BLOCK_VALUE), accum_dtype)
            current_state = T.alloc_fragment(
                (BLOCK_KEY, BLOCK_VALUE), accum_dtype
            )
            carry_state = T.alloc_fragment(
                (BLOCK_KEY, BLOCK_VALUE), accum_dtype
            )
            states = T.alloc_shared(
                (num_states, BLOCK_KEY, BLOCK_VALUE), accum_dtype
            )
            T.clear(states)

            for chunk_index in T.serial(num_chunks):
                begin = chunk_index * chunk
                T.copy(
                    q[
                        batch_index,
                        begin : begin + chunk,
                        0,
                        key_tile * BLOCK_KEY : (key_tile + 1) * BLOCK_KEY,
                    ],
                    q_shared,
                )
                T.copy(
                    k[
                        batch_index,
                        begin : begin + chunk,
                        0,
                        key_tile * BLOCK_KEY : (key_tile + 1) * BLOCK_KEY,
                    ],
                    k_shared,
                )
                T.copy(
                    v[
                        batch_index,
                        begin : begin + chunk,
                        head_index,
                        value_tile * BLOCK_VALUE : (value_tile + 1) * BLOCK_VALUE,
                    ],
                    v_shared,
                )

                T.gemm(
                    q_shared,
                    k_shared,
                    score,
                    transpose_B=True,
                    clear_accum=True,
                )
                for row, col in T.Parallel(chunk, chunk):
                    level = level_lut[row, col]
                    decay = T.exp(
                        g[batch_index, begin + row, head_index]
                        - g[batch_index, begin + col, head_index]
                    )
                    score_shared[row, col] = T.if_then_else(
                        row >= col,
                        score[row, col]
                        * decay
                        * level_scales[
                            batch_index, begin + row, head_index, level
                        ],
                        0.0,
                    )

                T.gemm(
                    score_shared,
                    v_shared,
                    output_acc,
                    clear_accum=True,
                )

                for state_level in T.serial(num_states):
                    if (chunk_index & (1 << state_level)) != 0:
                        for row, col in T.Parallel(chunk, BLOCK_KEY):
                            q_scaled_shared[row, col] = (
                                q_shared[row, col]
                                * level_scales[
                                    batch_index,
                                    begin + row,
                                    head_index,
                                    num_intra_levels + state_level,
                                ]
                                * T.exp(g[batch_index, begin + row, head_index])
                            )
                        for row, col in T.Parallel(BLOCK_KEY, BLOCK_VALUE):
                            state_shared[row, col] = states[
                                state_level, row, col
                            ]
                        T.gemm(
                            q_scaled_shared,
                            state_shared,
                            output_acc,
                        )

                T.copy(output_acc, output_shared)
                T.copy(
                    output_shared,
                    output_partial[
                        batch_index,
                        begin : begin + chunk,
                        head_index,
                        key_tile,
                        value_tile * BLOCK_VALUE : (value_tile + 1) * BLOCK_VALUE,
                    ],
                )

                gate_last = g[
                    batch_index, begin + chunk - 1, head_index
                ]
                gate_scale = T.exp(gate_last)
                for state_level, row, col in T.Parallel(
                    num_states, BLOCK_KEY, BLOCK_VALUE
                ):
                    states[state_level, row, col] *= gate_scale

                for row, col in T.Parallel(chunk, BLOCK_VALUE):
                    v_scaled_shared[row, col] = v_shared[row, col] * T.exp(
                        gate_last - g[batch_index, begin + row, head_index]
                    )
                T.gemm(
                    k_shared,
                    v_scaled_shared,
                    current_state,
                    transpose_A=True,
                    clear_accum=True,
                )
                for row, col in T.Parallel(BLOCK_KEY, BLOCK_VALUE):
                    states[0, row, col] += current_state[row, col]

                carry_mask = ((~chunk_index) & (chunk_index + 1)) - 1
                for state_level in T.serial(num_states - 1):
                    if (carry_mask & (1 << state_level)) != 0:
                        for row, col in T.Parallel(BLOCK_KEY, BLOCK_VALUE):
                            carry_state[row, col] = states[
                                state_level, row, col
                            ]
                        for row, col in T.Parallel(BLOCK_KEY, BLOCK_VALUE):
                            states[state_level + 1, row, col] += carry_state[
                                row, col
                            ]
                        for row, col in T.Parallel(BLOCK_KEY, BLOCK_VALUE):
                            states[state_level, row, col] = 0.0

    return main


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def _reduce_output_kernel(
    batch: int,
    sequence: int,
    heads: int,
    value_dim: int,
    num_key_tiles: int,
    dtype: T.dtype = T.bfloat16,
):
    num_value_tiles = value_dim // BLOCK_VALUE

    @T.prim_func
    def main(
        partial: T.Tensor(
            (batch, sequence, heads, num_key_tiles, value_dim), dtype
        ),
        output: T.Tensor((batch, sequence, heads, value_dim), dtype),
    ):
        with T.Kernel(
            num_value_tiles, sequence, batch * heads, threads=128
        ) as (value_tile, token_index, batch_head):
            batch_index = batch_head // heads
            head_index = batch_head % heads
            accum = T.alloc_fragment((BLOCK_VALUE,), T.float32)
            T.clear(accum)
            for key_tile in T.serial(num_key_tiles):
                for col in T.Parallel(BLOCK_VALUE):
                    accum[col] += partial[
                        batch_index,
                        token_index,
                        head_index,
                        key_tile,
                        value_tile * BLOCK_VALUE + col,
                    ]
            for col in T.Parallel(BLOCK_VALUE):
                output[
                    batch_index,
                    token_index,
                    head_index,
                    value_tile * BLOCK_VALUE + col,
                ] = accum[col]

    return main


def _prepare_tilelang_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    level_scales: torch.Tensor,
) -> tuple[object, tuple[torch.Tensor, ...], torch.Tensor]:
    if not all(tensor.is_cuda for tensor in (q, k, v, g, level_scales)):
        raise ValueError("all inputs must be CUDA tensors")
    if q.dtype != torch.bfloat16:
        raise ValueError("the initial baseline currently supports BF16")
    if q.shape[2] != 1 or k.shape[2] != 1:
        raise ValueError("the initial baseline requires one shared Q/K head")

    batch, sequence, _, key_dim = q.shape
    value_dim = v.shape[-1]
    heads = v.shape[2]
    levels = level_scales.shape[-1]
    if sequence % CHUNK_SIZE != 0:
        raise ValueError("sequence length must be divisible by 64")
    if key_dim not in (64, 128, 256) or value_dim not in (64, 128, 256):
        raise ValueError("the baseline supports K/V in {64, 128, 256}")
    if key_dim % BLOCK_KEY or value_dim % BLOCK_VALUE:
        raise ValueError("K and V must be divisible by 64")
    expected_levels = math.ceil(math.log2(sequence)) + 1
    if levels != expected_levels:
        raise ValueError(
            f"level_scales must have {expected_levels} levels, got {levels}"
        )

    kernel = _log_linear_fwd_kernel(
        batch,
        sequence,
        heads,
        key_dim,
        value_dim,
        levels,
        T.bfloat16,
    )
    num_key_tiles = key_dim // BLOCK_KEY
    local_g = _chunk_local_cumsum(g.contiguous())
    partial = torch.empty(
        batch,
        sequence,
        heads,
        num_key_tiles,
        value_dim,
        device=v.device,
        dtype=v.dtype,
    )
    kernel_args = (
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        local_g.contiguous(),
        level_scales.contiguous(),
        _level_lut(q.device.index or 0),
        partial,
    )
    if num_key_tiles == 1:
        output = partial.squeeze(3)

        def run():
            kernel(*kernel_args)

        return run, (), output

    reduce_kernel = _reduce_output_kernel(
        batch,
        sequence,
        heads,
        value_dim,
        num_key_tiles,
        T.bfloat16,
    )
    output = torch.empty_like(v)

    def run():
        kernel(*kernel_args)
        reduce_kernel(partial, output)

    return run, (), output


def chunk_log_linear_attn_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    level_scales: torch.Tensor,
) -> torch.Tensor:
    """Forward-only dense TileLang baseline for Log-Linear Attention."""
    kernel, args, output = _prepare_tilelang_forward(
        q, k, v, g, level_scales
    )
    kernel(*args)
    return output
