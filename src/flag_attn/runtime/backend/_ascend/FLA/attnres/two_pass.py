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

"""Row-batched score and mixture kernels for Ascend AttnRes.

The score grid spans residual sources and row batches; each program streams one
source and reduces its row norms and query dots once, so independent sources
fill idle vector cores and no source-logit tile is carried through the loop.

The mixture kernel computes the source log-sum-exp for its row batch, then
streams column tiles to form the weighted mixture and its RMSNorm in one write.
"""

from __future__ import annotations

import triton
import triton.language as tl

_PLANS: dict = {}


@triton.jit
def _score_kernel(
    query,
    residual0,
    residual_ptrs,
    rms_weight,
    logit,
    N,
    D: tl.constexpr,
    BR: tl.constexpr,
    BDB: tl.constexpr,
    eps: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    groups = tl.cdiv(N, BR)
    source = pid // groups
    rows = (pid % groups) * BR + tl.arange(0, BR)
    blocks = tl.arange(0, BDB)
    residual_ptr = tl.cast(tl.load(residual_ptrs + source), residual0.dtype)
    # Two [BR, BDB] tiles accumulate the products over the whole row and are
    # reduced once per source; reducing per block instead costs one extra
    # full-width op per block.
    acc_square = tl.zeros([BR, BDB], tl.float32)
    acc_dot = tl.zeros([BR, BDB], tl.float32)
    for block in tl.range(0, D // BDB):
        cols = block * BDB + blocks
        query_values = tl.load(query + cols).to(tl.float32)
        weight_values = tl.load(rms_weight + cols).to(tl.float32)
        query_weight = query_values * weight_values
        offsets = rows[:, None] * D + cols[None, :]
        if EVEN:
            residual_values = tl.load(residual_ptr + offsets)
        else:
            residual_values = tl.load(
                residual_ptr + offsets, mask=rows[:, None] < N, other=0.0
            )
        residual_values = residual_values.to(tl.float32)
        acc_square = tl.fma(residual_values, residual_values, acc_square)
        acc_dot = tl.fma(residual_values, query_weight[None, :], acc_dot)
    source_logit = tl.sum(acc_dot, axis=1) * tl.rsqrt(tl.sum(acc_square, axis=1) / D + eps)
    if EVEN:
        tl.store(logit + source * N + rows, source_logit)
    else:
        tl.store(logit + source * N + rows, source_logit, mask=rows < N)


@triton.jit
def _mix_kernel(
    residual0,
    residual_ptrs,
    logit,
    lse,
    output_rms_weight,
    output,
    N,
    L: tl.constexpr,
    D: tl.constexpr,
    BR: tl.constexpr,
    BDB: tl.constexpr,
    TB: tl.constexpr,
    scale: tl.constexpr,
    eps: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    LP: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    blocks = tl.arange(0, BDB)
    sources = tl.arange(0, LP)
    logits = tl.load(
        logit + sources[None, :] * N + rows[:, None],
        mask=(sources[None, :] < L) & (rows[:, None] < N),
        other=-float("inf"),
    )
    # Mask after scaling: padded -inf logits become NaN at scale=0 or +inf
    # at negative scales, and must never participate in the source reduction.
    scores = tl.where(sources[None, :] < L, logits * scale, -float("inf"))
    row_max = tl.max(scores, axis=1)
    row_sum = tl.sum(tl.exp(scores - row_max[:, None]), axis=1)
    row_lse = row_max + tl.log(row_sum)
    tl.store(lse + rows, row_lse, mask=rows < N)

    # The weights are the normalised softmax, so the mixture needs no further
    # division and the fused RMSNorm reduces to rsqrt(mean(out^2) + eps).
    squared_norm = tl.zeros([BR], tl.float32)
    for block in tl.range(0, D // BDB):
        cols = block * BDB + blocks
        accumulator = tl.zeros([BR, BDB], tl.float32)
        for source in tl.range(0, L):
            source_logit = tl.load(logit + source * N + rows, mask=rows < N, other=0.0)
            # Normalize shifted exponentials directly: subtracting a rounded
            # large LSE would lose the small log-normalizer between close scores.
            probability = tl.exp(source_logit * scale - row_max) / row_sum
            residual_ptr = tl.cast(tl.load(residual_ptrs + source), residual0.dtype)
            offsets = rows[:, None] * D + cols[None, :]
            if EVEN:
                residual_values = tl.load(residual_ptr + offsets)
            else:
                residual_values = tl.load(
                    residual_ptr + offsets, mask=rows[:, None] < N, other=0.0
                )
            accumulator = tl.fma(probability[:, None], residual_values.to(tl.float32), accumulator)
        output_offsets = rows[:, None] * D + cols[None, :]
        if HAS_ONORM:
            squared_norm += tl.sum(accumulator * accumulator, axis=1)
        if EVEN:
            tl.store(output + output_offsets, accumulator.to(output.dtype.element_ty))
        else:
            tl.store(
                output + output_offsets,
                accumulator.to(output.dtype.element_ty),
                mask=rows[:, None] < N,
            )

    if HAS_ONORM:
        # Reuse the stored mixture rather than allocating a separate scratch.
        row_scale = tl.rsqrt(squared_norm / D + eps)
        tail = tl.arange(0, TB)
        for block in tl.range(0, D // TB):
            cols = block * TB + tail
            output_values = tl.load(output_rms_weight + cols).to(tl.float32)
            offsets = rows[:, None] * D + cols[None, :]
            if EVEN:
                accumulated = tl.load(output + offsets).to(tl.float32)
            else:
                accumulated = tl.load(
                    output + offsets, mask=rows[:, None] < N, other=0.0
                ).to(tl.float32)
            normalized = accumulated * row_scale[:, None] * output_values[None, :]
            if EVEN:
                tl.store(output + offsets, normalized.to(output.dtype.element_ty))
            else:
                tl.store(
                    output + offsets,
                    normalized.to(output.dtype.element_ty),
                    mask=rows[:, None] < N,
                )


def _power_of_two_divisor(hidden_size: int) -> int:
    """Largest power of two dividing ``hidden_size`` (block widths must divide it)."""
    return hidden_size & -hidden_size


def plan(hidden_size: int, num_rows: int, num_sources: int, has_onorm: bool) -> dict | None:
    """Pick the row batches and block widths for one shape."""
    key = (hidden_size, num_rows, num_sources, has_onorm)
    cached = _PLANS.get(key)
    if cached is not None:
        return cached

    lowbit = _power_of_two_divisor(hidden_size)
    if lowbit < 128 or not 2 <= num_sources <= 16:
        # A single source has no softmax to batch over, and hidden sizes without
        # a large power-of-two divisor leave the DMA too little to stream.
        return None

    # Pass 1 reduces once per source, after the column-block loop, and keeps
    # two [BR, BDB] fp32 accumulators plus one residual tile live.  A 128-wide
    # block keeps that footprint under the UB budget with 32 rows per program.
    row_batch = 32
    block_b = min(128, lowbit)
    config = dict(
        L=num_sources,
        D=hidden_size,
        BR1=row_batch,
        BDB1=block_b,
        EVEN1=num_rows % row_batch == 0,
        # Pass 2 needs an accumulator and a source tile live; BR=8/BDB=1024
        # leaves the tail room.
        BR2=8,
        BDB2=min(1024, lowbit),
        EVEN2=num_rows % 8 == 0,
        TB=min(1024, lowbit),
        LP=1 << (num_sources - 1).bit_length(),
    )
    if len(_PLANS) > 64:
        _PLANS.clear()
    _PLANS[key] = config
    return config


def launch(values, config, *, eps: float, scale: float, has_onorm: bool) -> None:
    """Launch both passes with the standard Triton launch interface.

    ``values`` contains query, residual0, residual_ptrs, rms_weight,
    output_rms_weight, output, logit, lse, and num_rows.
    """
    num_rows = values[8]
    score_values = [values[i] for i in (0, 1, 2, 3, 6, 8)]
    mix_values = [values[i] for i in (1, 2, 6, 7, 4, 5, 8)]
    _score_kernel[(config["L"] * triton.cdiv(num_rows, config["BR1"]),)](
        *score_values,
        D=config["D"], BR=config["BR1"], BDB=config["BDB1"],
        eps=eps, EVEN=config["EVEN1"],
    )
    _mix_kernel[(triton.cdiv(num_rows, config["BR2"]),)](
        *mix_values,
        L=config["L"], D=config["D"], BR=config["BR2"], BDB=config["BDB2"],
        TB=config["TB"], scale=scale, eps=eps, HAS_ONORM=has_onorm,
        LP=config["LP"], EVEN=config["EVEN2"],
    )


__all__ = ["plan", "launch"]
