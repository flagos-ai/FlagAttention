# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Optional Ascend Vector sort of exact BF16 score and block ID keys."""

from pathlib import Path
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as raw
from triton.experimental.tle.raw import dialect


@dialect(
    name="cann",
    file=str(Path(__file__).with_name("infllmv2_top64_raw.cpp")),
    extern_func_name="infllm_sort64",
    pipeline={"core": "vector", "pipe": "PIPE_V"},
)
def sort64(*args, **kwargs): ...


@triton.jit
def select_blocks_raw_kernel(
    score_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k_ptr,
    total_q,
    max_blocks,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    block_b: tl.constexpr,
    max_q: tl.constexpr,
):
    tl.static_assert(topk == 64)
    tl.static_assert(block_b >= 128)
    tl.static_assert(block_b <= 4096)
    tl.static_assert(block_b % 32 == 0)
    tl.static_assert(block_b & (block_b - 1) == 0)
    for qi in tl.range(tl.program_id(0), max_q, tl.num_programs(0)):
        hk, batch = tl.program_id(1), tl.program_id(2)
        qs, qe = tl.load(cu_q_ptr + batch), tl.load(cu_q_ptr + batch + 1)
        if qs + qi < qe:
            klen = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
            qpos = klen - (qe - qs) + qi
            valid_blocks = tl.minimum(
                max_blocks, tl.minimum(tl.cdiv(klen, block_size), (qpos + block_size) // block_size)
            )
            rank = tl.arange(0, 64)
            base = out_ptr + (hk.to(tl.int64) * total_q + qs + qi) * 64
            if valid_blocks <= 64:
                tl.store(base + rank, tl.where(rank < valid_blocks, rank, -1))
            else:
                lane = tl.arange(0, block_b)
                score = tl.load(
                    score_ptr + (hk.to(tl.int64) * total_q + qs + qi) * max_blocks + lane, lane < valid_blocks, other=0.0
                ).to(tl.float32)
                score = tl.where(score == score, score, float("-inf"))
                bits = score.to(tl.int32, bitcast=True)
                magnitude = (bits & 2147483647) >> 16
                ordered = tl.where(bits < 0, 32767 - magnitude, 32768 + magnitude)
                packed = (ordered << 12) | (4095 - lane)
                # Monotonic normal FP32 bit patterns preserve all 28 key bits.
                # Pad with 0x00800000 (FLT_MIN), the smallest positive normal:
                # every vbitsort/vmrgsort4 lane is then a normal finite binary32
                # and padding sorts strictly below every real key.
                key = tl.where(lane < valid_blocks, packed + 0x00800000, 0x00800000)
                gate = tl.reshape(key.to(tl.float32, bitcast=True), (1, block_b))
                ids = tl.reshape(lane.to(tl.uint32), (1, block_b))
                pairs = tl.zeros((1, 2 * block_b), tl.float32)
                scratch = tl.zeros((1, 2 * block_b), tl.float32)
                best = tl.zeros((1, 64), tl.float32)
                pairs, scratch, best = raw.call(sort64, [ids, gate, pairs, scratch, best], output_indices=[2, 3, 4])
                values = tl.reshape(best, (64,))
                chosen = 4095 - ((values.to(tl.int32, bitcast=True) - 0x00800000) & 4095)
                position = tl.sum((chosen[:, None] > chosen[None, :]).to(tl.int32), 1)
                tl.store(base + position, chosen)
