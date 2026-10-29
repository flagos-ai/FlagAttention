# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
import os

import torch
import triton
import triton.language as tl

from ..chunk_h import chunk_bwd_dh, chunk_fwd_h, chunk_fwd_h_fused_cumsum
from ..cumsum_gla import chunk_local_cumsum
from ..index import prepare_chunk_indices
from ..utils import check_shared_mem, input_guard

RCP_LN2 = 1.4426950216
OP_NAME = "chunk_gla"
_SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Conservative aggregate state-tile work limit for the gfx936 streaming path.
# Keep this paired with the separate sequential-chunk bound below.
_STREAM_FUSED_MAX_STATE_WORK = 128 * 128 * 128


BK_LIST = [32, 64] if check_shared_mem() else [16, 32]
BV_LIST = [64, 128] if check_shared_mem("ampere") else [16, 32]


def _select_chunk_size(sequence_length: int) -> int:
    return min(64, max(16, triton.next_power_of_2(sequence_length)))


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    # Hygon gfx936: winners observed across all supported forward shape families.
    configs=[
        triton.Config({"BK": 64}, num_warps=1, num_stages=2),
        triton.Config({"BK": 64}, num_warps=2, num_stages=2),
    ],
    key=["BC", "K"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    TRIANGULAR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if TRIANGULAR:
        i_i = 0
        i_j = 0
        for i_row in range(NC - 1):
            row_start = i_row * (i_row + 1) // 2
            row_end = (i_row + 1) * (i_row + 2) // 2
            in_row = (i_c >= row_start) & (i_c < row_end)
            i_i = tl.where(in_row, i_row + 1, i_i)
            i_j = tl.where(in_row, i_c - row_start, i_j)
    else:
        i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, i_k * BK),
            (BC, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, i_k * BK),
            (BC, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (K, T),
            (1, H * K),
            (i_k * BK, i_t * BT + i_j * BC),
            (BK, BC),
            (0, 1),
        )
        p_gk = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (K, T),
            (1, H * K),
            (i_k * BK, i_t * BT + i_j * BC),
            (BK, BC),
            (0, 1),
        )
        p_gn = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * exp2(b_g - b_gn[None, :]) * scale
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * exp2(b_gn[:, None] - b_gk)
        # [BC, BC] using tf32 to improve precision here.
        b_A += tl.dot(b_qg, b_kg)

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + i_i * BC, i_j * BC),
        (BC, BC),
        (1, 0),
    )
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    # Keep both occupancy regimes selected by D/H families (1 and 2 warps).
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
    ],
    key=["BK", "BT", "K", "BC"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    m_s = tl.arange(0, BC)[:, None] >= tl.arange(0, BC)[None, :]

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT

    p_q = tl.make_block_ptr(
        q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0)
    )
    p_g = tl.make_block_ptr(
        g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, 0), (BC, BK), (1, 0)
    )
    p_gk = tl.make_block_ptr(
        g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, 0), (BC, BK), (1, 0)
    )
    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)

    # Use tile dot to avoid a large [BC, BC, BK] temporary that can hurt occupancy.
    g_max = tl.max(b_g, axis=0)
    g_min = tl.min(b_g, axis=0)
    g_mid = 0.5 * (g_max + g_min)
    b_qg = b_q * exp2(b_g - g_mid[None, :])
    b_kg = b_k * exp2(-b_gk + g_mid[None, :])
    b_A = tl.dot(b_qg, tl.trans(b_kg)) * scale
    b_A = tl.where(m_s, b_A, 0.0)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=1)],
    key=["BC", "BK"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_i * BC >= T:
        return

    m_s = tl.arange(0, BC)[:, None] >= tl.arange(0, BC)[None, :]

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += ((i_k * all + bos) * H + i_h) * BC

    p_q = tl.make_block_ptr(
        q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
    )
    p_g = tl.make_block_ptr(
        g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0)
    )
    p_gk = tl.make_block_ptr(
        g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0)
    )
    p_A = tl.make_block_ptr(
        A, (T, BC), (H * BC, 1), (i_t * BT + i_i * BC, 0), (BC, BC), (1, 0)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)

    g_max = tl.max(b_g, axis=0)
    g_min = tl.min(b_g, axis=0)
    g_mid = 0.5 * (g_max + g_min)
    b_qg = b_q * exp2(b_g - g_mid[None, :])
    b_kg = b_k * exp2(-b_gk + g_mid[None, :])
    b_A = tl.dot(b_qg, tl.trans(b_kg)) * scale
    b_A = tl.where(m_s, b_A, 0.0)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=1)],
    key=["BC", "NK"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A,
    A2,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_c * BC >= T:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(
            A + (i_k * all + bos) * H * BC + i_h * BC,
            (T, BC),
            (H * BC, 1),
            (i_t * BT + i_c * BC, 0),
            (BC, BC),
            (1, 0),
        )
        b_A += tl.load(p_A, boundary_check=(0, 1))
    p_A2 = tl.make_block_ptr(
        A2 + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT + i_c * BC, i_c * BC),
        (BC, BC),
        (1, 0),
    )
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"BK": BK, "BV": BV},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "H", "K", "HV", "STATE_V_FIRST", "V"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(
                h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
            )
        else:
            p_h = tl.make_block_ptr(
                h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
            )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    b_o *= scale
    p_v = tl.make_block_ptr(
        v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_A = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o_fused_intra(
    q,
    k,
    v,
    g,
    h,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute q @ h and the intra-chunk A @ v without materializing A.

    This path is used only when no backward graph is needed.  It trades the
    global A read/write for a local reconstruction of each A tile while the
    corresponding v tile is already resident, removing a large intermediate
    tensor and one full round trip through device memory.
    """
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V

    # Keep the same BC-row numerical decomposition as the reference A
    # kernels.  This avoids changing exponent range/rounding while still
    # eliminating the global A materialization.
    q_rows = tl.arange(0, BC)
    k_cols = tl.arange(0, BC)
    for i_i in range(NC):
        b_o = tl.zeros([BC, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
            )
            p_g = tl.make_block_ptr(
                g, (T, K), (HV * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
            )
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(
                    h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
                )
            else:
                p_h = tl.make_block_ptr(
                    h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
                )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            b_h = tl.load(p_h, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

        b_o *= scale

        for i_c in range(NC):
            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_q = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
                )
                p_g = tl.make_block_ptr(
                    g, (T, K), (HV * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
                )
                p_k = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0)
                )
                p_gk = tl.make_block_ptr(
                    g, (T, K), (HV * K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0)
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_g = tl.load(p_g, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
                b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
                if i_c < i_i:
                    o_k = i_k * BK + tl.arange(0, BK)
                    b_gn = tl.load(
                        g + (i_t * BT + i_i * BC) * HV * K + o_k,
                        mask=o_k < K,
                        other=0.0,
                    )
                    p_ki = tl.make_block_ptr(
                        k,
                        (K, T),
                        (1, H * K),
                        (i_k * BK, i_t * BT + i_c * BC),
                        (BK, BC),
                        (0, 1),
                    )
                    p_gki = tl.make_block_ptr(
                        g,
                        (K, T),
                        (1, HV * K),
                        (i_k * BK, i_t * BT + i_c * BC),
                        (BK, BC),
                        (0, 1),
                    )
                    b_ki = tl.load(p_ki, boundary_check=(0, 1)).to(tl.float32)
                    b_gki = tl.load(p_gki, boundary_check=(0, 1)).to(tl.float32)
                    b_qg = b_q * exp2(b_g - b_gn[None, :]) * scale
                    b_A += tl.dot(
                        b_qg,
                        b_ki * exp2(b_gn[:, None] - b_gki),
                    )
                else:
                    g_max = tl.max(b_g, axis=0)
                    g_min = tl.min(b_g, axis=0)
                    g_mid = 0.5 * (g_max + g_min)
                    b_qg = b_q * exp2(b_g - g_mid[None, :])
                    b_A += tl.dot(
                        b_qg,
                        tl.trans(b_k * exp2(-b_gk + g_mid[None, :]))
                    ) * scale
            m_s = (i_i * BC + q_rows[:, None]) >= (i_c * BC + k_cols[None, :])
            b_A = tl.where(m_s, b_A, 0.0)
            p_v = tl.make_block_ptr(
                v, (T, V), (HV * V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0)
            )
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_o += tl.dot(b_A.to(b_v.dtype), b_v)
        p_o = tl.make_block_ptr(
            o, (T, V), (HV * V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))



@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    # Hygon gfx936: two measured occupancy regimes for small/large D families.
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
    ],
    key=["BK", "NC", "BT", "H", "K"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_bwd_kernel_intra(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, i_k * BK),
        (BC, BK),
        (1, 0),
    )
    # [BC, BK]
    b_g = tl.load(p_g, boundary_check=(0, 1))

    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(
                k + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_j * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_gk = tl.make_block_ptr(
                g + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_j * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_dA = tl.make_block_ptr(
                dA + (bos * H + i_h) * BT,
                (T, BT),
                (H * BT, 1),
                (i_t * BT + i_i * BC, i_j * BC),
                (BC, BC),
                (1, 0),
            )
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp2(b_gn[None, :] - b_gk)
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))

            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= exp2(b_g - b_gn[None, :])
    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    o_dA = (
        bos * H * BT
        + (i_t * BT + i_i * BC + tl.arange(0, BC)) * H * BT
        + i_h * BT
        + i_i * BC
    )
    p_kj = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gkj = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_dq = tl.make_block_ptr(
        dq + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, i_k * BK),
        (BC, BK),
        (1, 0),
    )

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        # (SY 09/17) important to not use bf16 here to have a good precision.
        b_dq += tl.where(
            m_i, b_dA[:, None] * b_kj[None, :] * exp2(b_g - b_gkj[None, :]), 0.0
        )
        p_kj += H * K
        p_gkj += H * K
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # [BC, BK]
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H * K + i_h * K + o_k

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_j * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_gq = tl.make_block_ptr(
                g + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_j * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_dA = tl.make_block_ptr(
                dA + (bos * H + i_h) * BT,
                (BT, T),
                (1, H * BT),
                (i_i * BC, i_t * BT + i_j * BC),
                (BC, BC),
                (0, 1),
            )

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.load(p_gq, boundary_check=(0, 1))
            b_qg = b_q * tl.where(m_j[:, None], exp2(b_gq - b_gn[None, :]), 0)
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= exp2(b_gn[None, :] - b_g)
    o_dA = (
        bos * H * BT
        + (i_t * BT + i_i * BC) * H * BT
        + i_h * BT
        + i_i * BC
        + tl.arange(0, BC)
    )
    p_qj = q + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gqj = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_dk = tl.make_block_ptr(
        dk + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, i_k * BK),
        (BC, BK),
        (1, 0),
    )
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * H * BT)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dk += tl.where(
            m_i, b_dA[:, None] * b_qj[None, :] * exp2(b_gqj[None, :] - b_g), 0.0
        )
        p_qj += H * K
        p_gqj += H * K
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    # Small-BV and large-BV families select different occupancy settings.
    configs=[
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=1, num_stages=2),
    ],
    key=["BV", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_bwd_kernel_dA(
    v,
    do,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(
            do + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (V, T),
            (1, H * V),
            (i_v * BV, i_t * BT),
            (BV, BT),
            (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        b_dA += tl.dot(b_do, b_v)

    p_dA = tl.make_block_ptr(
        dA + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s, b_dA * scale, 0.0)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    # Hygon gfx936: retain measured small/large state-tile winners.
    configs=[
        triton.Config({"BK": 32, "BV": 32}, num_warps=4, num_stages=4),
        triton.Config({"BK": 32, "BV": 32}, num_warps=8, num_stages=2),
    ],
    key=["BT", "H", "K", "V", "STATE_V_FIRST"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_bwd_kernel_dv(
    k,
    g,
    A,
    do,
    dh,
    dv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1)
    )
    p_do = tl.make_block_ptr(
        do + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))

    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0.0)
    # (SY 09/17) important to disallow tf32 here to maintain a good precision.
    b_dv = tl.dot(b_A, b_do.to(b_A.dtype), allow_tf32=False)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_gk = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_gn = g + (bos + min(i_t * BT + BT, T) - 1) * H * K + i_h * K + o_k
        if STATE_V_FIRST:
            # dh stored as [V, K]; read a logical [BK, BV] tile via on-the-fly transpose
            p_dh = tl.make_block_ptr(
                dh + (i_tg * H + i_h) * K * V,
                (K, V),
                (1, K),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (0, 1),
            )
        else:
            p_dh = tl.make_block_ptr(
                dh + (i_tg * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_gn = exp2(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
        b_k = (b_k * b_gn).to(b_k.dtype)
        # [BT, BV]
        # (SY 09/17) it is ok to have bf16 interchunk gradient contribution here
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    # Hygon gfx936: alternate K/V state tiles cover long and small D families.
    configs=[
        triton.Config({"BK": 16, "BV": 32}, num_warps=4, num_stages=4),
        triton.Config({"BK": 32, "BV": 16}, num_warps=4, num_stages=3),
    ],
    key=["BT", "H", "K", "V", "STATE_V_FIRST"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_bwd_kernel_inter(
    q,
    k,
    v,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dq2,
    dk2,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    h += (i_tg * H + i_h) * K * V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K

    p_gk = tl.make_block_ptr(
        g, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    p_gn = g + (min(T, i_t * BT + BT) - 1) * H * K + o_k
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_do = tl.make_block_ptr(
            do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        if STATE_V_FIRST:
            # h / dh stored as [V, K] -- the [BV, BK] tile is now a contiguous read
            p_h = tl.make_block_ptr(
                h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
            )
            p_dh = tl.make_block_ptr(
                dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
            )
        else:
            p_h = tl.make_block_ptr(
                h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1)
            )
            p_dh = tl.make_block_ptr(
                dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1)
            )
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # [BK]
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

    b_dgk *= exp2(b_gn)
    b_dq *= scale
    b_dq = b_dq * exp2(b_gk)
    b_dk = b_dk * exp2(b_gn[None, :] - b_gk)
    p_q = tl.make_block_ptr(
        q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    p_dq = tl.make_block_ptr(
        dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    p_dk = tl.make_block_ptr(
        dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    b_dg = (
        b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]
    )
    p_dq = tl.make_block_ptr(
        dq2, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    p_dk = tl.make_block_ptr(
        dk2, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    p_dg = tl.make_block_ptr(
        dg, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_intra_gk(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = k.shape
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    # For BF16 and 64-token chunks, one 64-row tile avoids triangular sub-tiles at K <= 128.
    # Other dtypes, wider K, and explicit 128-token chunks keep the existing tiles.
    use_wide_bf16_tile = q.dtype == torch.bfloat16 and K <= 128 and BT == 64
    BC = min(
        64 if use_wide_bf16_tile else (32 if H <= 32 else 16),
        BT,
    )
    NC = triton.cdiv(BT, BC)

    A = q.new_empty(B, T, H, BT, dtype=torch.float)
    triangular_pairs = NC * (NC - 1) // 2
    if triangular_pairs:
        grid = (NT, triangular_pairs, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
            q=q,
            k=k,
            g=g,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            NC=NC,
            TRIANGULAR=True,
        )

    grid = (NT, NC, B * H)
    # load the entire [BC, K] blocks into SRAM at once
    if K <= 256:
        BK = max(triton.next_power_of_2(K), 16)
        chunk_gla_fwd_A_kernel_intra_sub_intra[grid](
            q=q,
            k=k,
            g=g,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
        )
    # split then merge
    else:
        BK = min(128, triton.next_power_of_2(K))
        NK = triton.cdiv(K, BK)
        A_intra = q.new_empty(NK, B, T, H, BC, dtype=torch.float)

        grid = (NK, NT * NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_split[grid](
            q=q,
            k=k,
            g=g,
            A=A_intra,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            B=B,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            NC=NC,
        )

        grid = (NT, NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid](
            A=A_intra,
            A2=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            B=B,
            H=H,
            BT=BT,
            BC=BC,
            NK=NK,
        )
    return A


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * HV)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return o
def chunk_gla_fwd_o_fused_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size
    BC = min(64, BT)
    NC = triton.cdiv(BT, BC)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Match the wide tiles used by the separate A and output kernels where
    # possible, avoiding repeated small-K and small-V work in the fused path.
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))
    o = torch.empty_like(v)
    grid = (triton.cdiv(V, BV), NT, B * HV)
    chunk_gla_fwd_kernel_o_fused_intra[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        h=h,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NC=NC,
        STATE_V_FIRST=state_v_first,
    )
    return o
def chunk_gla_bwd_dA(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, V = v.shape
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BV = min(64, triton.next_power_of_2(V))

    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    grid = (NT, B * H)
    chunk_gla_bwd_kernel_dA[grid](
        v=v,
        do=do,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dA


def chunk_gla_bwd_dv(
    k: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_gla_bwd_kernel_dv[grid](
        k=k,
        g=g,
        A=A,
        do=do,
        dh=dh,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return dv


def chunk_gla_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = q.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    grid = (NK * NC, NT, B * H)
    chunk_gla_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        dA=dA,
        dq=dq,
        dk=dk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    return dq, dk


def chunk_gla_bwd_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dg = torch.empty_like(g)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)

    def grid(meta):
        return (triton.cdiv(K, meta["BK"]), NT, B * H)

    chunk_gla_bwd_kernel_inter[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dq2=dq2,
        dk2=dk2,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return dq2, dk2, dg


def chunk_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    fuse_intra: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_fused_h_cumsum = (
        g_cumsum is None
        and os.environ.get("FLAG_ATTN_GLA_FUSED_CUMSUM_H", "1") != "0"
        and initial_state is None
        and not output_final_state
        and not state_v_first
        and cu_seqlens is None
        # The synchronized fused H path is beneficial for V=128 as well;
        # keep V=64 on the general kernel because its tiny state is launch
        # bound and the fused variant regresses there.
        and v.shape[-1] >= 128
    )
    if use_fused_h_cumsum:
        g_cumsum, h = chunk_fwd_h_fused_cumsum(
            k=k,
            v=v,
            g=g,
            chunk_size=chunk_size,
        )
        ht = None
    else:
        if g_cumsum is None:
            g_cumsum = chunk_local_cumsum(
                g,
                chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
            )

        h, ht = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum,
            gv=None,
            h0=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            states_in_fp32=False,
            state_v_first=state_v_first,
        )

    if fuse_intra and v.shape[-1] <= 128:
        A = None
        o = chunk_gla_fwd_o_fused_intra(
            q=q,
            k=k,
            v=v,
            g=g_cumsum,
            h=h,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )
    else:
        # The intra A is kept in fp32 for the backward path and for wide V.
        A = chunk_gla_fwd_intra_gk(
            q=q,
            k=k,
            g=g_cumsum,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        o = chunk_gla_fwd_o_gk(
            q=q,
            v=v,
            g=g_cumsum,
            A=A,
            h=h,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )
    return g_cumsum, A, h, ht, o


def chunk_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(
            g,
            chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
        )

    if h is None:
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum,
            gv=None,
            h0=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            states_in_fp32=True,
            state_v_first=state_v_first,
        )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=g_cumsum,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=True,
        state_v_first=state_v_first,
    )

    dv = chunk_gla_bwd_dv(
        k=k,
        g=g_cumsum,
        A=A,
        do=do,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )

    # dq dk in fp32
    dA = chunk_gla_bwd_dA(
        v=v,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    dq, dk = chunk_gla_bwd_dqk_intra(
        q=q,
        k=k,
        g=g_cumsum,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    dq, dk, dg = chunk_gla_bwd_dqkg(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g_cumsum,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )
    return dq, dk, dv, dg, dh0


class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
    ):
        chunk_size = _select_chunk_size(q.shape[1])
        stream_fused_mode = os.environ.get("FLAG_ATTN_GLA_STREAM_FUSED", "auto").lower()
        stream_fused_chunks = triton.cdiv(q.shape[1], chunk_size)
        stream_fused_bk = triton.next_power_of_2(q.shape[-1])
        stream_fused_bv = triton.cdiv(v.shape[-1], 32) * 32
        stream_fused_work = (
            q.shape[0]
            * v.shape[2]
            * stream_fused_chunks
            * stream_fused_bk
            * stream_fused_bv
        )
        use_fused_state = (
            stream_fused_mode not in ("0", "false", "off")
            and q.dtype == torch.bfloat16
            and g.dtype == torch.bfloat16
            and chunk_size == 64
            and (
                (
                    q.shape[-1] <= 64
                    and v.shape[-1] <= 64
                    and stream_fused_chunks <= 16
                )
                or (
                    64 < q.shape[-1] <= 128
                    and 64 < v.shape[-1] <= 128
                    and stream_fused_chunks <= 8
                    and stream_fused_work <= _STREAM_FUSED_MAX_STATE_WORK
                )
            )
            and initial_state is None
            and not output_final_state
            and not state_v_first
            and cu_seqlens is None
            and not any(x is not None and x.requires_grad for x in (q, k, v, g))
        )
        if use_fused_state:
            o = chunk_gla_fwd_o_fused_state(
                q=q,
                k=k,
                v=v,
                g=g,
                scale=scale,
                chunk_size=chunk_size,
            )
            return o, None

        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(
                cu_seqlens,
                chunk_size,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
        else:
            chunk_indices = None

        # For BF16 K=V=64 inference without an initial/final state, use one
        # 64-row tile to fuse the intra-chunk A@V work into output generation.
        # This removes the FP32 A intermediate without splitting/recomputing
        # the tile. Keep an env opt-out for reproducible A/B measurements.
        use_fused_intra = (
            os.environ.get("FLAG_ATTN_GLA_FUSED_INTRA", "auto").lower()
            not in ("0", "false", "off")
            and q.dtype == torch.bfloat16
            and g.dtype == torch.bfloat16
            and q.shape[-1] == 64
            and v.shape[-1] == 64
            and initial_state is None
            and not output_final_state
            and not state_v_first
            and cu_seqlens is None
            and not any(x is not None and x.requires_grad for x in (q, k, v, g))
        )
        g_cumsum, A, _, ht, o = chunk_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=None,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
            fuse_intra=use_fused_intra,
        )
        # recompute g_cumsum in bwd pass
        if g.dtype != torch.float:
            g_cumsum = None
        else:
            g = None
        if use_fused_intra:
            # No input can require a backward graph on this path, so the
            # materialized A is intentionally absent.
            ctx.fused_intra = True
            return o, ht
        ctx.fused_intra = False
        ctx.save_for_backward(q, k, v, g, g_cumsum, initial_state, A, chunk_indices)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.state_v_first = state_v_first
        return o, ht

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        q, k, v, g, g_cumsum, initial_state, A, chunk_indices = ctx.saved_tensors
        chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
        dq, dk, dv, dg, dh0 = chunk_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=g_cumsum,
            scale=scale,
            h=None,
            A=A,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=ctx.state_v_first,
        )
        return dq.to(q), dk.to(k), dv.to(v), dg, None, dh0, None, None, None, None


def _validate_chunk_gla_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    cu_seqlens: torch.Tensor | None,
) -> None:
    tensors = {"q": q, "k": k, "v": v, "g": g}
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        bad = next(name for name, value in tensors.items() if not isinstance(value, torch.Tensor))
        raise TypeError(f"{bad} must be a torch.Tensor")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 4:
        raise ValueError("q, k, v and g must all have shape [B, T, H, D]")
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError("q, k and g must have identical shapes")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must match q in batch, sequence and head dimensions")
    if q.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise TypeError(f"unsupported q dtype {q.dtype}; expected float16, bfloat16 or float32")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k and v must use the same dtype")
    if g.dtype not in (q.dtype, torch.float32):
        raise TypeError("g must use q.dtype or torch.float32")
    if len({q.device, k.device, v.device, g.device}) != 1:
        raise ValueError("q, k, v and g must be on the same device")
    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("initial_state must be a torch.Tensor or None")
        if initial_state.ndim != 4:
            raise ValueError("initial_state must have shape [N, H, K, V] or [N, H, V, K]")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        if initial_state.dtype != torch.float32:
            raise TypeError("initial_state must be torch.float32")
        expected_n = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        expected_tail = (v.shape[-1], q.shape[-1]) if state_v_first else (q.shape[-1], v.shape[-1])
        if initial_state.shape != (expected_n, q.shape[2], *expected_tail):
            raise ValueError(
                "initial_state shape must be "
                f"({expected_n}, {q.shape[2]}, {expected_tail[0]}, {expected_tail[1]})"
            )
    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor or None")
        if q.shape[0] != 1:
            raise ValueError("q batch size must be 1 when cu_seqlens is provided")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a one-dimensional tensor with N+1 entries")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must use int32 or int64")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same device as q")


@torch.compiler.disable
def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: int | None = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, T, H, K]`.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` (or `[N, H, V, K]` if `state_v_first=True`)
            for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`
            (or `[N, H, V, K]` if `state_v_first=True`). Default: `False`.
        state_v_first (Optional[bool]):
            Store the recurrent state in V-first `[V, K]` layout instead of the default `[K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` (or `[N, H, V, K]` if `state_v_first=True`)
            if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from flag_attn import chunk_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = chunk_gla(
            q, k, v, g,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gla(
            q, k, v, g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    _validate_chunk_gla_inputs(q, k, v, g, initial_state, state_v_first, cu_seqlens)
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert v.shape == (
        *q.shape[:3],
        v.shape[-1],
    ), "v must be of shape (batch size, seq len, num of head, head dim)."
    o, final_state = ChunkGLAFunction.apply(
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
    )
    return o, final_state


@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o_fused_state(
    q,
    k,
    v,
    g,
    o,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    LOG2E: tl.constexpr,
    scale,
):
    """Stream chunk-state updates into output for small zero-state inference."""
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    q0 = q + (i_b * T * H + i_h) * K
    k0 = k + (i_b * T * H + i_h) * K
    g0 = g + (i_b * T * H + i_h) * K
    v0 = v + (i_b * T * HV + i_hv) * V
    o0 = o + (i_b * T * HV + i_hv) * V

    state = tl.zeros((BK, BV), dtype=tl.float32)
    rows = tl.arange(0, BT)

    # Keep the boundary state in this CTA and consume it immediately for the
    # corresponding output chunk. This avoids materializing all chunk states.
    for i_t in range(tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(
            q0, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g0, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k0, (K, T), (1, H * K), (0, i_t * BT), (BK, BT), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v0, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_o = tl.make_block_ptr(
            o0, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_gk = tl.cumsum(b_g, axis=0) * LOG2E
        b_qg = (b_q * exp2(b_gk)).to(b_q.dtype)
        # The ordinary path stores chunk-start states in the input dtype.
        b_o = tl.dot(b_qg, state.to(b_q.dtype))

        b_k_col = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_k_row = tl.trans(b_k_col)
        b_v = tl.load(p_v, boundary_check=(0, 1))

        last_col = tl.minimum((i_t + 1) * BT, T) - 1 - i_t * BT
        b_gk_last = tl.sum(
            tl.where(rows[:, None] == last_col, b_gk, 0.0), axis=0
        )

        g_mid = 0.5 * (tl.max(b_gk, axis=0) + tl.min(b_gk, axis=0))
        b_q_local = b_q * exp2(b_gk - g_mid[None, :])
        b_k_local = b_k_row * exp2(-b_gk + g_mid[None, :])
        b_a = tl.dot(b_q_local, tl.trans(b_k_local))
        causal = rows[:, None] >= rows[None, :]
        b_a = tl.where(causal, b_a, 0.0)
        b_o = b_o * scale + tl.dot((b_a * scale).to(b_v.dtype), b_v)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        b_k_weighted = (
            b_k_col * exp2(b_gk_last[:, None] - tl.trans(b_gk))
        ).to(k.dtype.element_ty)
        state = state * exp2(b_gk_last)[:, None] + tl.dot(b_k_weighted, b_v)


def chunk_gla_fwd_o_fused_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    BK, BV = triton.next_power_of_2(K), 32
    split_mode = os.environ.get("FLAG_ATTN_GLA_STREAM_K_SPLIT", "auto").lower()
    if split_mode not in ("0", "false", "off", "no") and K > 64 and V > 64:
        try:
            num_cus = torch.cuda.get_device_properties(q.device).multi_processor_count
        except (AttributeError, RuntimeError):
            num_cus = 0
        base_ctas = triton.cdiv(V, BV) * B * HV
        # A 2-way split did not amortize the extra reduction at this CTA
        # scale. Try 4-way splitting only when the unsplit grid is below half
        # the CU count and four partitions can supply at least one wave.
        if (
            num_cus > 0
            and base_ctas * 2 < num_cus
            and base_ctas * 4 >= num_cus
            and BK // 16 >= 4
        ):
            return chunk_gla_fwd_o_fused_state_k_split(
                q=q,
                k=k,
                v=v,
                g=g,
                scale=scale,
                chunk_size=chunk_size,
                split_count=4,
            )

    o = torch.empty_like(v)
    grid = (triton.cdiv(V, BV), B * HV)
    chunk_gla_fwd_kernel_o_fused_state[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        o=o,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BK=BK,
        BV=BV,
        LOG2E=RCP_LN2,
        scale=scale,
    )
    return o

@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o_fused_state_k_split(
    q,
    k,
    v,
    g,
    partial,
    T,
    N: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BK_FULL: tl.constexpr,
    BV: tl.constexpr,
    LOG2E: tl.constexpr,
    scale,
):
    """Split recurrent-state rows across CTAs; compute chunk-local attention once."""
    i_v, i_bh, i_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    q0 = q + (i_b * T * H + i_h) * K
    k0 = k + (i_b * T * H + i_h) * K
    g0 = g + (i_b * T * H + i_h) * K
    v0 = v + (i_b * T * HV + i_hv) * V
    partial0 = partial + i_k * N + (i_b * T * HV + i_hv) * V

    state = tl.zeros((BK, BV), dtype=tl.float32)
    rows = tl.arange(0, BT)
    k_offset = i_k * BK

    for i_t in range(tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(
            q0, (T, K), (H * K, 1), (i_t * BT, k_offset), (BT, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g0, (T, K), (H * K, 1), (i_t * BT, k_offset), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k0, (K, T), (1, H * K), (k_offset, i_t * BT), (BK, BT), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v0, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_o = tl.make_block_ptr(
            partial0,
            (T, V),
            (HV * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_gk = tl.cumsum(b_g, axis=0) * LOG2E
        b_qg = (b_q * exp2(b_gk)).to(b_q.dtype)
        b_o = tl.dot(b_qg, state.to(b_q.dtype)) * scale

        b_k_col = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if i_k == 0:
            p_q_full = tl.make_block_ptr(
                q0, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK_FULL), (1, 0)
            )
            p_g_full = tl.make_block_ptr(
                g0, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK_FULL), (1, 0)
            )
            p_k_full = tl.make_block_ptr(
                k0, (K, T), (1, H * K), (0, i_t * BT), (BK_FULL, BT), (0, 1)
            )
            b_q_full = tl.load(p_q_full, boundary_check=(0, 1))
            b_g_full = tl.load(p_g_full, boundary_check=(0, 1)).to(tl.float32)
            b_gk_full = tl.cumsum(b_g_full, axis=0) * LOG2E
            g_mid = 0.5 * (tl.max(b_gk_full, axis=0) + tl.min(b_gk_full, axis=0))
            b_q_local = b_q_full * exp2(b_gk_full - g_mid[None, :])
            b_k_col_full = tl.load(p_k_full, boundary_check=(0, 1)).to(tl.float32)
            b_k_local = tl.trans(b_k_col_full) * exp2(
                -b_gk_full + g_mid[None, :]
            )
            b_a = tl.dot(b_q_local, tl.trans(b_k_local))
            causal = rows[:, None] >= rows[None, :]
            b_a = tl.where(causal, b_a, 0.0)
            b_o += tl.dot((b_a * scale).to(b_v.dtype), b_v)

        tl.store(p_o, b_o, boundary_check=(0, 1))

        last_col = tl.minimum((i_t + 1) * BT, T) - 1 - i_t * BT
        b_gk_last = tl.sum(tl.where(rows[:, None] == last_col, b_gk, 0.0), axis=0)
        b_k_weighted = (
            b_k_col * exp2(b_gk_last[:, None] - tl.trans(b_gk))
        ).to(k.dtype.element_ty)
        state = state * exp2(b_gk_last)[:, None] + tl.dot(b_k_weighted, b_v)


@triton.jit
def chunk_gla_fwd_o_split_reduce_kernel(
    partial, o, N: tl.constexpr, P: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(P):
        acc += tl.load(partial + i * N + offsets, offsets < N, other=0.0)
    tl.store(o + offsets, acc.to(o.dtype.element_ty), offsets < N)


def chunk_gla_fwd_o_fused_state_k_split(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int,
    split_count: int,
) -> torch.Tensor:
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    BK_FULL, BV = triton.next_power_of_2(K), 32
    BK = BK_FULL // split_count
    partial = torch.empty((split_count, *v.shape), dtype=torch.float32, device=v.device)
    o = torch.empty_like(v)
    grid = (triton.cdiv(V, BV), B * HV, split_count)
    chunk_gla_fwd_kernel_o_fused_state_k_split[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        partial=partial,
        T=T,
        N=o.numel(),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BK=BK,
        BK_FULL=BK_FULL,
        BV=BV,
        LOG2E=RCP_LN2,
        scale=scale,
    )
    n = o.numel()
    chunk_gla_fwd_o_split_reduce_kernel[(triton.cdiv(n, 1024),)](
        partial, o, n, split_count, 1024
    )
    return o
