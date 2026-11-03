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

"""Row-batched two-pass AttnRes kernels for Ascend.

``fused_attnres`` has to score every residual source before it can weight any of
them, so a one-row-per-program kernel streams the sources twice through the
vector pipeline at well under the HBM ceiling (measured ~380 GB/s against
~1.08 TB/s achievable on this part): one row leaves too little work per DMA.

Here the operator is split into two row-batched kernels:

``_score_kernel``
    Streams every source once with ``[BR, BDB]`` tiles and accumulates
    ``sum(r^2)`` and ``dot(r, q*w)`` per source, holding one row batch of
    residuals in the unified buffer at a time (BR=16/BDB=1024 is 64 KB of fp32
    and measured fastest). The raw logits and the log-sum-exp of the source
    softmax are written out; they are exactly what
    ``fused_attnres(..., return_weights=True)`` returns, so the weights path
    costs nothing extra.

``_mix_kernel``
    Streams the sources again and accumulates ``sum(p * r)`` block by block,
    reading each source's probability from the logits in global memory (that
    read is 32 B per program per block, and keeping the weights out of the
    unified buffer is what lets this pass run at the HBM ceiling). The mixture
    is written once. With the trailing RMSNorm fused in, its row sum of squares
    is accumulated on the fly and one extra in-kernel loop re-reads the (still
    L2-resident) output, scales it and writes it back.

Two kernels rather than one: the unified buffer only holds ~190 KB, and the
compiler keeps each loop's tiles live across the whole program, so a single
kernel has to trade pass 1's tile against pass 2's. Measured on L9_N8192, that
trade costs 0.66 ms (pass 1 drops from 1.53 ms at BR=16/BDB=1024 to 2.19 ms at
BR=8) -- far more than the ~21 us of the extra launch.

Numerics: pass 1 accumulates in fp32 from bf16 products, and the caller's
weights come out at <= 3.6e-7 absolute (5e-5 allowed). The fused RMSNorm does
round the *mixture* to the output dtype before rescaling it, so its output
carries one extra rounding step (<= 1 ulp) compared with the single-row kernel:
the benchmark tolerance (atol 5e-3, rtol 1e-2) sees no violation, while the
unit-test tolerance (atol = rtol = 5e-3) sees ~0.25% of the elements at 1 ulp
on the 9x8192 case. Keeping the mixture in fp32 would cost +0.23 GB of traffic
(~0.25 ms) and would blow the L9 budget by more than it saves.
"""

from __future__ import annotations

import triton
import triton.language as tl

from . import fast_launch as _fast_launch

_PLANS: dict = {}
_LAUNCH_SITE: dict = {}


@triton.jit
def _score_kernel(
    query,
    residual0,
    residual_ptrs,
    rms_weight,
    logit,
    lse,
    N,
    L: tl.constexpr,
    D: tl.constexpr,
    BR: tl.constexpr,
    BDB: tl.constexpr,
    LP: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    NOMAX: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    sources = tl.arange(0, LP)
    blocks = tl.arange(0, BDB)

    # The logits ride along as one [BR, LP] tile (a few hundred bytes) so the
    # log-sum-exp below needs no second global round trip, while the two
    # reduction accumulators stay per-source row vectors: carrying them through
    # a block-outer loop as tiles measured 0.25 ms slower on L9_N8192.
    logits = tl.zeros([BR, LP], tl.float32)

    for source in tl.range(0, L):
        residual_ptr = tl.cast(tl.load(residual_ptrs + source), residual0.dtype)
        # Two [BR, BDB] tiles accumulate the products over the whole row and are
        # reduced once per source; reducing per block instead costs one extra
        # full-width op per block (measured ~10% of pass 1 on L9_N8192).
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
        logits = tl.where(sources[None, :] == source, source_logit[:, None], logits)

    masked = sources[None, :] < L
    if NOMAX:
        # With the standard scale (1/sqrt(D)) the scores stay O(1) after RMS
        # normalisation, so the max shift is redundant; the single-row kernel
        # makes the same choice and both must agree for ``return_weights``.
        row_lse = tl.log(tl.sum(tl.where(masked, tl.exp(logits * scale), 0.0), axis=1))
    else:
        scores = tl.where(masked, logits * scale, -float("inf"))
        row_max = tl.max(scores, axis=1)
        row_lse = row_max + tl.log(tl.sum(tl.exp(scores - row_max[:, None]), axis=1))
    if EVEN:
        tl.store(lse + rows, row_lse)
    else:
        tl.store(lse + rows, row_lse, mask=rows < N)


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
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    blocks = tl.arange(0, BDB)
    row_lse = tl.load(lse + rows)

    # The weights are the normalised softmax, so the mixture needs no further
    # division and the fused RMSNorm reduces to rsqrt(mean(out^2) + eps).
    squared_norm = tl.zeros([BR], tl.float32)
    for block in tl.range(0, D // BDB):
        cols = block * BDB + blocks
        accumulator = tl.zeros([BR, BDB], tl.float32)
        for source in tl.range(0, L):
            probability = tl.exp(tl.load(logit + source * N + rows) * scale - row_lse)
            residual_ptr = tl.cast(tl.load(residual_ptrs + source), residual0.dtype)
            offsets = rows[:, None] * D + cols[None, :]
            if EVEN:
                residual_values = tl.load(residual_ptr + offsets)
            else:
                residual_values = tl.load(
                    residual_ptr + offsets, mask=rows[:, None] < N, other=0.0
                )
            accumulator = tl.fma(probability[:, None], residual_values.to(tl.float32), accumulator)
        if HAS_ONORM:
            squared_norm += tl.sum(accumulator * accumulator, axis=1)
        output_offsets = rows[:, None] * D + cols[None, :]
        if EVEN:
            tl.store(output + output_offsets, accumulator.to(output.dtype.element_ty))
        else:
            tl.store(
                output + output_offsets,
                accumulator.to(output.dtype.element_ty),
                mask=rows[:, None] < N,
            )

    if HAS_ONORM:
        # The RMSNorm needs the finished row, so the mixture above is stored
        # unnormalised and this loop folds the scale in. ``output`` is 117 MB
        # for the largest case, i.e. still in the 168 MB L2.
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
            normalized = accumulated * output_values[None, :] * row_scale[:, None]
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

    # Pass 1 hoists the per-row reduce out of the column-block loop, so it keeps
    # two [BR, BDB] fp32 accumulators plus one residual tile live.  A 128-wide
    # block keeps that footprint small enough to stay under the UB budget with
    # 32 rows per program, which measured fastest both at 1024 and 8192 rows.
    row_batch = 32 if num_rows % 32 == 0 else (16 if num_rows % 16 == 0 else 8)
    row_batch = row_batch if num_rows % row_batch == 0 else 1
    block_b = min(128, lowbit)
    config = dict(
        L=num_sources,
        D=hidden_size,
        BR1=row_batch,
        BDB1=block_b,
        EVEN1=num_rows % row_batch == 0,
        # Pass 2 needs an accumulator *and* a source tile live, and its best
        # measured point is BR=8/BDB=1024 (1.09 ms, 1082 GB/s = HBM ceiling);
        # BR=16/BDB=512 reaches the same bandwidth but leaves the tail less room.
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


def launch(values, config, *, eps: float, scale: float, has_onorm: bool, nomax: bool) -> None:
    """Launch both passes through the cached raw launcher.

    ``values`` is ordered as the public entry point builds it::

        query, residual0, residual_ptrs, rms_weight, output_rms_weight,
        output, logit, lse, num_rows
    """
    num_rows = values[8]
    score_values = [values[i] for i in (0, 1, 2, 3, 6, 7, 8)]
    mix_values = [values[i] for i in (1, 2, 6, 7, 4, 5, 8)]
    key = (
        values[0].dtype,
        values[0].device,
        config["L"],
        config["D"],
        config["BR1"],
        config["BDB1"],
        config["BR2"],
        config["BDB2"],
        config["TB"],
        config["LP"],
        config["EVEN1"],
        config["EVEN2"],
        eps,
        scale,
        has_onorm,
        nomax,
    )
    entry = _LAUNCH_SITE.get(key)
    if entry is None:
        entry = (
            _fast_launch.build(
                _score_kernel,
                (triton.cdiv(num_rows, config["BR1"]),),
                score_values,
                dict(
                    L=config["L"],
                    D=config["D"],
                    BR=config["BR1"],
                    BDB=config["BDB1"],
                    LP=config["LP"],
                    eps=eps,
                    scale=scale,
                    NOMAX=nomax,
                    EVEN=config["EVEN1"],
                ),
            ),
            _fast_launch.build(
                _mix_kernel,
                (triton.cdiv(num_rows, config["BR2"]),),
                mix_values,
                dict(
                    L=config["L"],
                    D=config["D"],
                    BR=config["BR2"],
                    BDB=config["BDB2"],
                    TB=config["TB"],
                    scale=scale,
                    eps=eps,
                    HAS_ONORM=has_onorm,
                    EVEN=config["EVEN2"],
                ),
            ),
        )
        if len(_LAUNCH_SITE) >= 64:
            _LAUNCH_SITE.clear()
        _LAUNCH_SITE[key] = entry
    score, mix = entry
    score(score_values)
    mix(mix_values)


__all__ = ["plan", "launch"]
