# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""TLE-Raw chunk top-k for Ascend half-precision inputs."""

import os

import torch
import triton
import triton.language as tl

# These dimensions are part of the AscendC UB layout contract.
BLOCK_TOKENS = 64
BLOCK_CHUNKS = 128
MAX_TOPK = 8
_GROUP_TOKENS = tl.constexpr(16)
_MAX_TOPK = tl.constexpr(MAX_TOPK)

try:
    import triton.experimental.tle.language.raw as tle_raw
    from triton.experimental.tle.raw import dialect

    _HAS_TLE_RAW = True
except ImportError:  # Non-FlagTree builds use the Triton selection kernel.
    tle_raw = None
    dialect = None
    _HAS_TLE_RAW = False

if _HAS_TLE_RAW:

    @dialect(
        name="cann",
        file=os.path.join(os.path.dirname(__file__), "_moba_topk_raw.cpp"),
        extern_func_name="moba_sort_topk",
        pipeline={"core": "vector", "pipe": "PIPE_V"},
    )
    def _sort_topk_op(*args, **kwargs):
        ...


def is_supported(q: torch.Tensor, routed_topk: int) -> bool:
    return (
        _HAS_TLE_RAW
        and q.dtype in (torch.float16, torch.bfloat16)
        and 32 <= q.shape[-1] <= 128
        and 1 <= routed_topk <= MAX_TOPK
    )


@triton.jit(
    do_not_specialize=["num_chunks", "seqlen", "num_tiles", "i_tok_lo", "topk"]
)
def chunk_topk_raw_kernel(
    q_ptr,
    chunk_means_ptr,
    chunk_end_ptr,
    batch_end_ptr,
    selected_ptr,
    num_chunks,
    seqlen,
    num_tiles,
    i_tok_lo,
    topk,
    limits_ptr,
    limit_off,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_cm_c,
    stride_cm_h,
    stride_cm_d,
    stride_sel_r,
    stride_sel_h,
    stride_sel_t,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_LIMITS: tl.constexpr,
):
    """Select chunk ids with a Cube dot and stable Raw top-eight merge.

    The launch signature matches the Triton fallback. Each limits pair stores
    (exclusive chunk end, first possible chunk index), before tile rounding.
    """
    tl.static_assert(BLOCK_TOK == 64 and BLOCK_CHUNK == 128)
    tl.static_assert(1 <= TOPK and TOPK <= _MAX_TOPK)
    pid = tl.program_id(0)
    ph = tl.program_id(1)
    ot = i_tok_lo + pid * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tm = ot < seqlen
    od = tl.arange(0, BLOCK_D)
    md = od < HEAD_DIM
    qt = tl.load(
        q_ptr + ot[:, None] * stride_q_t + od[None, :] * stride_q_d + ph * stride_q_h,
        mask=tm[:, None] & md[None, :],
        other=0.0,
    )
    cc = tl.arange(0, BLOCK_CHUNK)
    fields = tl.arange(0, 2 * _MAX_TOPK)
    best = tl.broadcast_to(
        tl.where(fields[None, :] % 2 == 0, float("-inf"), 0.0), (BLOCK_TOK, 2 * _MAX_TOPK)
    )
    neg_inf = float("-inf")
    if USE_LIMITS:
        tile_limit = tl.load(limits_ptr + (limit_off + pid) * 2)
        tile_start = tl.load(limits_ptr + (limit_off + pid) * 2 + 1) // BLOCK_CHUNK
        num_tiles_eff = (tile_limit + BLOCK_CHUNK - 1) // BLOCK_CHUNK
    else:
        num_tiles_eff = num_tiles
        tile_start = 0
    for tile in range(tile_start, num_tiles_eff):
        oc = tile * BLOCK_CHUNK + cc
        cml = oc < num_chunks
        means_t = tl.load(
            chunk_means_ptr + od[:, None] * stride_cm_d
            + oc[None, :] * stride_cm_c + (ph // GROUP_SIZE) * stride_cm_h,
            mask=md[:, None] & cml[None, :],
            other=0.0,
        )
        gate = tl.dot(qt, means_t, out_dtype=tl.float32)
        ce = tl.load(chunk_end_ptr + oc, mask=cml, other=0)
        be = tl.load(batch_end_ptr + oc, mask=cml, other=0)
        valid = (
            tm[:, None] & cml[None, :]
            & (ot[:, None] >= ce[None, :]) & (ot[:, None] < be[None, :])
        )
        work = tl.where(valid, gate, neg_inf)
        ids = tl.broadcast_to(oc[None, :].to(tl.uint32), (BLOCK_TOK, BLOCK_CHUNK))
        # Scratch is reused for each group of 16 tokens inside the Raw op.
        pairs = tl.zeros((_GROUP_TOKENS, 2 * BLOCK_CHUNK), tl.float32)
        local = tl.zeros((_GROUP_TOKENS, 8 * _MAX_TOPK), tl.float32)
        # Gate/scratch are mutable; best is both input and output state.
        work, pairs, local, best = tle_raw.call(
            _sort_topk_op, [ids, work, pairs, local, best], output_indices=[1, 2, 3, 4]
        )
    vals, ids = tl.split(tl.reshape(best, (BLOCK_TOK, _MAX_TOPK, 2)))
    ranks = tl.arange(0, _MAX_TOPK)
    tl.store(
        selected_ptr + ranks[None, :] * stride_sel_r
        + ph * stride_sel_h + ot[:, None] * stride_sel_t,
        tl.where(vals > neg_inf, ids.to(tl.int32, bitcast=True), -1),
        mask=tm[:, None] & (ranks[None, :] < TOPK),
    )
