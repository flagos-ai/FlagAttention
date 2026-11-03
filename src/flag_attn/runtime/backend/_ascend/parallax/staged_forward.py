"""Fixed-length, full-causal Parallax forward staged through GM intermediates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _parallax_fused_short_kernel(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale_log2,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    DIRECT_OUTPUT: tl.constexpr = False,
):
    """Fuse full-length T=128 causal Parallax forward for tiny head counts."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // HQ
    hq = bh % HQ
    hk = hq // G
    rows = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, 128)
    ds = tl.arange(0, BD)
    row_mask = rows < 128
    feature_mask = ds < D

    q_offsets = ((batch * 128 + rows[:, None]) * HQ + hq) * D + ds[None, :]
    k_offsets = ((batch * 128 + cols[:, None]) * H + hk) * D + ds[None, :]
    qv = tl.load(q + q_offsets, row_mask[:, None] & feature_mask[None, :], other=0.0)
    rv = tl.load(r + q_offsets, row_mask[:, None] & feature_mask[None, :], other=0.0)
    kv = tl.load(k + k_offsets, feature_mask[None, :], other=0.0)
    vv = tl.load(v + k_offsets, feature_mask[None, :], other=0.0)

    scores = tl.dot(qv, tl.trans(kv), out_dtype=tl.float32) * scale_log2
    rk = tl.dot(rv, tl.trans(kv), out_dtype=tl.float32)
    causal = (rows[:, None] >= cols[None, :]) & row_mask[:, None]
    scores = tl.where(causal, scores, -float("inf"))
    rk = tl.where(causal, rk, 0.0)
    row_max = tl.max(scores, axis=1, keep_dims=True)
    weights = tl.exp2(scores - row_max)
    denom = tl.sum(weights, axis=1, keep_dims=True)
    normalized = weights / denom
    correction_mean = tl.sum(weights * rk, axis=1, keep_dims=True) / denom

    mean_value = tl.dot(normalized.to(vv.dtype), vv, out_dtype=tl.float32)
    if DIRECT_OUTPUT:
        corrected_weights = normalized * (1.0 + correction_mean - rk)
        output = tl.dot(corrected_weights.to(vv.dtype), vv, out_dtype=tl.float32)
    else:
        correction_value = tl.dot((normalized * rk).to(vv.dtype), vv, out_dtype=tl.float32)
        output = mean_value * (1.0 + correction_mean) - correction_value

    out_offsets = ((batch * 128 + rows[:, None]) * HQ + hq) * D + ds[None, :]
    scalar_offsets = (batch * 128 + rows) * HQ + hq
    tl.store(o + out_offsets, output, row_mask[:, None] & feature_mask[None, :])
    tl.store(barv + out_offsets, mean_value, row_mask[:, None] & feature_mask[None, :])
    tl.store(d1 + scalar_offsets, tl.sum(weights, axis=1), row_mask)
    tl.store(bart + scalar_offsets, tl.sum(correction_mean, axis=1), row_mask)
    tl.store(m + scalar_offsets, tl.sum(row_max, axis=1), row_mask)


@triton.jit
def _score_tiles(
    q,
    r,
    k,
    score_ptr,
    rk_ptr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    COMPACT_CAUSAL: tl.constexpr = False,
):
    if COMPACT_CAUSAL:
        triangular_pid = tl.program_id(0).to(tl.int32)
        triangle_root = tl.sqrt((8 * triangular_pid + 1).to(tl.float32))
        pid_m = ((triangle_root - 1.0) * 0.5).to(tl.int32)
        pid_n = triangular_pid - pid_m * (pid_m + 1) // 2
        bh = tl.program_id(1).to(tl.int32)
    else:
        pid_m = tl.program_id(0).to(tl.int32)
        pid_n = tl.program_id(1).to(tl.int32)
        bh = tl.program_id(2).to(tl.int32)
    batch = bh // HQ
    hq = bh % HQ
    hk = hq // G
    row_start = pid_m * BM
    col_start = pid_n * BN
    if col_start > row_start + BM - 1:
        return

    rows = row_start + tl.arange(0, BM)
    cols = col_start + tl.arange(0, BN)
    d = tl.arange(0, BD)
    row_mask = rows < T
    col_mask = cols < T
    feature_mask = d < D
    q_offsets = ((batch * T + rows[:, None]) * HQ + hq) * D + d[None, :]
    r_offsets = q_offsets
    kv_offsets = ((batch * T + cols[:, None]) * H + hk) * D + d[None, :]
    bq = tl.load(q + q_offsets, row_mask[:, None] & feature_mask[None, :], other=0.0)
    br = tl.load(r + r_offsets, row_mask[:, None] & feature_mask[None, :], other=0.0)
    bk = tl.load(k + kv_offsets, col_mask[:, None] & feature_mask[None, :], other=0.0)

    qk = tl.dot(bq, tl.trans(bk), out_dtype=tl.float32) * SCALE_LOG2
    rk = tl.dot(br, tl.trans(bk), out_dtype=tl.float32)
    if col_start >= row_start:
        causal = rows[:, None] >= cols[None, :]
        qk = tl.where(causal, qk, -float("inf"))

    score_offsets = ((bh * T + rows[:, None]) * T + cols[None, :])
    tile_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(score_ptr + score_offsets, qk, tile_mask)
    tl.store(rk_ptr + score_offsets, rk, tile_mask)


@triton.jit
def _score_tiles_d128(
    q,
    r,
    k,
    score_ptr,
    rk_ptr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    COMPACT_CAUSAL: tl.constexpr,
):
    if COMPACT_CAUSAL:
        rows_per_key_tile: tl.constexpr = BN // BM
        triangular_pid = tl.program_id(0).to(tl.int32) // rows_per_key_tile
        triangle_root = tl.sqrt((8 * triangular_pid + 1).to(tl.float32))
        coarse_row = ((triangle_root - 1.0) * 0.5).to(tl.int32)
        pid_n = triangular_pid - coarse_row * (coarse_row + 1) // 2
        pid_m = coarse_row * rows_per_key_tile + tl.program_id(0) % rows_per_key_tile
    else:
        pid_m = tl.program_id(0).to(tl.int32)
        pid_n = tl.program_id(1).to(tl.int32)
    bh = tl.program_id(1 if COMPACT_CAUSAL else 2).to(tl.int32)
    batch = bh // HQ
    hq = bh % HQ
    hk = hq // G
    row_start = pid_m * BM
    if COMPACT_CAUSAL:
        col_start = pid_n * BN
    else:
        first_row = row_start
        first_key = tl.maximum(first_row - WINDOW_SIZE_LEFT + 1, 0)
        first_key_tile = first_key // BN
        col_start = (first_key_tile + pid_n) * BN
    if col_start > row_start + BM - 1:
        return
    if (not COMPACT_CAUSAL) and (col_start + BN <= tl.maximum(row_start - WINDOW_SIZE_LEFT + 1, 0)):
        return

    rows = row_start + tl.arange(0, BM)
    cols = col_start + tl.arange(0, BN)
    d = tl.arange(0, BD)
    row_mask = rows < T
    col_mask = cols < T
    feature_mask = d < D
    q_offsets = ((batch * T + rows[:, None]) * HQ + hq) * D + d[None, :]
    kv_offsets = ((batch * T + cols[:, None]) * H + hk) * D + d[None, :]
    bq = tl.load(q + q_offsets, row_mask[:, None] & feature_mask[None, :], other=0.0)
    br = tl.load(r + q_offsets, row_mask[:, None] & feature_mask[None, :], other=0.0)
    bk = tl.load(k + kv_offsets, col_mask[:, None] & feature_mask[None, :], other=0.0)
    qk = tl.dot(bq, tl.trans(bk), out_dtype=tl.float32) * SCALE_LOG2
    rk = tl.dot(br, tl.trans(bk), out_dtype=tl.float32)
    valid = (rows[:, None] >= cols[None, :]) & row_mask[:, None] & col_mask[None, :]
    if WINDOW_SIZE_LEFT >= 0:
        valid = valid & (cols[None, :] >= rows[:, None] - WINDOW_SIZE_LEFT + 1)
    qk = tl.where(valid, qk, -float("inf"))
    rk = tl.where(valid, rk, 0.0)
    score_offsets = ((bh * T + rows[:, None]) * T + cols[None, :])
    tile_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(score_ptr + score_offsets, qk, tile_mask)
    tl.store(rk_ptr + score_offsets, rk, tile_mask)


@triton.jit
def _softmax_correction_d128(
    score_ptr,
    rk_ptr,
    prob_ptr,
    weighted_rk_ptr,
    d1_ptr,
    bart_ptr,
    m_ptr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    BR: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
):
    row_block = tl.program_id(0).to(tl.int32)
    bh = tl.program_id(1).to(tl.int32)
    batch = bh // HQ
    hq = bh % HQ
    rows = row_block * BR + tl.arange(0, BR)
    cols = tl.arange(0, triton.next_power_of_2(T))
    score_offsets = ((bh * T + rows[:, None]) * T + cols[None, :])
    row_valid = rows < T
    valid = row_valid[:, None] & (cols[None, :] < T) & (cols[None, :] <= rows[:, None])
    if WINDOW_SIZE_LEFT >= 0:
        valid = valid & (cols[None, :] >= rows[:, None] - WINDOW_SIZE_LEFT + 1)
    scores = tl.load(score_ptr + score_offsets, valid, other=-float("inf")).to(tl.float32)
    rk = tl.load(rk_ptr + score_offsets, valid, other=0.0).to(tl.float32)
    row_max = tl.max(scores, axis=1, keep_dims=True)
    p = tl.exp2(scores - row_max)
    denom = tl.sum(p, axis=1, keep_dims=True)
    weighted_rk = p * rk
    bart = tl.sum(weighted_rk, axis=1, keep_dims=True) / denom
    scalar_offsets = (batch * T + rows) * HQ + hq
    output_valid = row_valid[:, None] & (cols[None, :] < T)
    normalized_p = p / denom
    corrected_weight = (p * (1.0 + bart) - weighted_rk) / denom
    tl.store(prob_ptr + score_offsets, normalized_p.to(tl.bfloat16), output_valid)
    tl.store(weighted_rk_ptr + score_offsets, corrected_weight.to(tl.bfloat16), output_valid)
    tl.store(d1_ptr + scalar_offsets, tl.sum(p, axis=1), row_valid)
    tl.store(bart_ptr + scalar_offsets, tl.sum(bart, axis=1), row_valid)
    tl.store(m_ptr + scalar_offsets, tl.sum(row_max, axis=1), row_valid)


@triton.jit
def _pv_tiles_d128(
    prob_ptr,
    corrected_ptr,
    v,
    o,
    barv,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int32)
    bh = tl.program_id(1).to(tl.int32)
    batch = bh // HQ
    hq = bh % HQ
    hk = hq // G
    rows = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    d = tl.arange(0, BD)
    row_mask = rows < T
    accum = tl.zeros((BM, BD), dtype=tl.float32)
    correction_accum = tl.zeros((BM, BD), dtype=tl.float32)
    if WINDOW_SIZE_LEFT < 0:
        col_start = 0
    else:
        col_start = tl.maximum(pid_m * BM - WINDOW_SIZE_LEFT + 1, 0)
        col_start = (col_start // BN) * BN
    col_end = tl.minimum(T, tl.cdiv((pid_m + 1) * BM, BN) * BN)
    for key_start in range(col_start, col_end, BN):
        key_rows = key_start + cols
        key_mask = key_rows < T
        weight_offsets = ((bh * T + rows[:, None]) * T + key_rows[None, :])
        value_offsets = ((batch * T + key_rows[:, None]) * H + hk) * D + d[None, :]
        weight_mask = row_mask[:, None] & key_mask[None, :]
        value_mask = key_mask[:, None] & (d[None, :] < D)
        prob = tl.load(prob_ptr + weight_offsets, weight_mask, other=0.0)
        corrected = tl.load(corrected_ptr + weight_offsets, weight_mask, other=0.0)
        values = tl.load(v + value_offsets, value_mask, other=0.0)
        accum = tl.dot(prob, values, out_dtype=tl.float32, acc=accum)
        correction_accum = tl.dot(corrected, values, out_dtype=tl.float32, acc=correction_accum)
    output_offsets = ((batch * T + rows[:, None]) * HQ + hq) * D + d[None, :]
    output_mask = row_mask[:, None] & (d[None, :] < D)
    tl.store(barv + output_offsets, accum, output_mask)
    tl.store(o + output_offsets, correction_accum, output_mask)


@triton.jit
def _softmax_correction(
    score_ptr,
    rk_ptr,
    prob_ptr,
    weighted_rk_ptr,
    d1_ptr,
    bart_ptr,
    m_ptr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    BR: tl.constexpr,
):
    row_block = tl.program_id(0).to(tl.int32)
    bh = tl.program_id(1).to(tl.int32)
    batch = bh // HQ
    hq = bh % HQ
    rows = row_block * BR + tl.arange(0, BR)
    cols = tl.arange(0, triton.next_power_of_2(T))
    score_offsets = ((bh * T + rows[:, None]) * T + cols[None, :])
    row_valid = rows < T
    causal = cols[None, :] <= rows[:, None]
    valid = row_valid[:, None] & (cols[None, :] < T) & causal
    scores = tl.load(score_ptr + score_offsets, valid, other=-float("inf")).to(tl.float32)
    rk = tl.load(rk_ptr + score_offsets, valid, other=0.0).to(tl.float32)
    row_max = tl.max(scores, axis=1, keep_dims=True)
    p = tl.exp2(scores - row_max)
    denom = tl.sum(p, axis=1, keep_dims=True)
    weighted_rk = p * rk
    bart = tl.sum(weighted_rk, axis=1, keep_dims=True) / denom
    scalar_offsets = (batch * T + rows) * HQ + hq
    dense_output = row_valid[:, None] & (cols[None, :] < T)

    normalized_p = p / denom
    corrected_weight = (p * (1.0 + bart) - weighted_rk) / denom
    tl.store(prob_ptr + score_offsets, normalized_p.to(tl.bfloat16), dense_output)
    tl.store(weighted_rk_ptr + score_offsets, corrected_weight.to(tl.bfloat16), dense_output)
    tl.store(d1_ptr + scalar_offsets, tl.sum(p, axis=1), row_valid)
    tl.store(bart_ptr + scalar_offsets, tl.sum(bart, axis=1), row_valid)
    tl.store(m_ptr + scalar_offsets, tl.sum(row_max, axis=1), row_valid)


@triton.jit
def _pv_tiles(
    prob_ptr,
    corrected_ptr,
    v,
    o,
    barv,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    SKIP_FUTURE: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int32)
    bh = tl.program_id(1).to(tl.int32)
    batch = bh // HQ
    hq = bh % HQ
    hk = hq // G
    rows = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    d = tl.arange(0, BD)
    row_mask = rows < T
    accum = tl.zeros((BM, BD), dtype=tl.float32)
    correction_accum = tl.zeros((BM, BD), dtype=tl.float32)

    if SKIP_FUTURE:
        # A row block can only attend to keys before its exclusive end. Keep
        # the bound aligned to PV tiles; the causal weights still mask the
        # partially valid final tile on a non-aligned sequence length.
        col_end = tl.minimum(T, tl.cdiv((pid_m + 1) * BM, BN) * BN)
    else:
        col_end = T

    for col_start in range(0, col_end, BN):
        key_rows = col_start + cols
        key_mask = key_rows < T
        weight_offsets = ((bh * T + rows[:, None]) * T + key_rows[None, :])
        value_offsets = ((batch * T + key_rows[:, None]) * H + hk) * D + d[None, :]
        weight_mask = row_mask[:, None] & key_mask[None, :]
        value_mask = key_mask[:, None] & (d[None, :] < D)
        prob = tl.load(prob_ptr + weight_offsets, weight_mask, other=0.0)
        corrected = tl.load(corrected_ptr + weight_offsets, weight_mask, other=0.0)
        values = tl.load(v + value_offsets, value_mask, other=0.0)
        accum = tl.dot(prob, values, out_dtype=tl.float32, acc=accum)
        correction_accum = tl.dot(corrected, values, out_dtype=tl.float32, acc=correction_accum)

    output_offsets = ((batch * T + rows[:, None]) * HQ + hq) * D + d[None, :]
    output_mask = row_mask[:, None] & (d[None, :] < D)
    tl.store(barv + output_offsets, accum, output_mask)
    tl.store(o + output_offsets, correction_accum, output_mask)


def staged_parallax_forward(q, r, k, v, scale):
    """Staged fixed-length D64 full-causal path, returning the backward state."""
    batch, time, hq, dim = q.shape
    h = k.shape[2]
    group_size = hq // h
    score_shape = (batch, hq, time, time)
    score = torch.empty(score_shape, device=q.device, dtype=q.dtype)
    rk = torch.empty_like(score)
    prob = torch.empty(score_shape, device=q.device, dtype=torch.bfloat16)
    weighted_rk = torch.empty_like(prob)
    output = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1 = torch.empty((batch, time, hq), device=q.device, dtype=torch.float32)
    bart = torch.empty_like(d1)
    m = torch.empty_like(d1)

    bd = triton.next_power_of_2(dim)
    bm_score = 128
    bn_score = 128
    score_tiles_m = triton.cdiv(time, bm_score)
    compact_causal = True
    score_grid = (
        score_tiles_m * (score_tiles_m + 1) // 2,
        batch * hq,
        1,
    )
    _score_tiles[score_grid](
        q,
        r,
        k,
        score,
        rk,
        time,
        HQ=hq,
        H=h,
        G=group_size,
        D=dim,
        BM=bm_score,
        BN=bn_score,
        BD=bd,
        SCALE_LOG2=scale * 1.4426950216,
        COMPACT_CAUSAL=compact_causal,
        num_warps=8,
        num_stages=1,
    )
    # Keep the softmax tile within the UB capacity as sequence length grows.
    br = min(8, 8192 // triton.next_power_of_2(time))
    _softmax_correction[(triton.cdiv(time, br), batch * hq)](
        score,
        rk,
        prob,
        weighted_rk,
        d1,
        bart,
        m,
        time,
        HQ=hq,
        BR=br,
        num_warps=4,
        num_stages=1,
    )
    bm_pv = 128
    bn_pv = 128
    bd_pv = triton.next_power_of_2(dim)
    _pv_tiles[(triton.cdiv(time, bm_pv), batch * hq)](
        prob,
        weighted_rk,
        v,
        output,
        barv,
        time,
        HQ=hq,
        H=h,
        G=group_size,
        D=dim,
        BM=bm_pv,
        BN=bn_pv,
        BD=bd_pv,
        SKIP_FUTURE=True,
        num_warps=8,
        num_stages=1,
    )
    return output, barv, d1, bart, m


def staged_parallax_forward_short(q, r, k, v, scale):
    """Single-kernel forward for the official short T128 training shapes."""
    batch, time, hq, dim = q.shape
    h = k.shape[2]
    group_size = hq // h
    output = torch.empty_like(q)
    barv = torch.empty_like(q)
    scalar_shape = (batch, time, hq)
    d1 = torch.empty(scalar_shape, device=q.device, dtype=torch.float32)
    bart = torch.empty_like(d1)
    m = torch.empty_like(d1)
    bm = 16 if dim == 64 else 32
    _parallax_fused_short_kernel[(triton.cdiv(time, bm), batch * hq)](
        q,
        r,
        k,
        v,
        output,
        barv,
        d1,
        bart,
        m,
        float(scale) * 1.4426950216,
        HQ=hq,
        H=h,
        G=group_size,
        D=dim,
        BM=bm,
        BD=triton.next_power_of_2(dim),
        DIRECT_OUTPUT=(dim == 128),
        num_warps=4,
        num_stages=1,
    )
    return output, barv, d1, bart, m


def staged_parallax_forward_d128(q, r, k, v, scale, window_size_left=-1, block_m=64):
    """Tiled D128 forward path with compact causal scheduling and saved state."""
    batch, time, hq, dim = q.shape
    h = k.shape[2]
    group_size = hq // h
    score_shape = (batch, hq, time, time)
    score = torch.empty(score_shape, device=q.device, dtype=q.dtype)
    rk = torch.empty_like(score)
    prob = torch.empty_like(score)
    corrected = torch.empty_like(score)
    output = torch.empty_like(q)
    barv = torch.empty_like(q)
    scalar_shape = (batch, time, hq)
    d1 = torch.empty(scalar_shape, device=q.device, dtype=torch.float32)
    bart = torch.empty_like(d1)
    m = torch.empty_like(d1)

    bm = block_m
    bn = 128
    bd = triton.next_power_of_2(dim)
    compact_causal = window_size_left < 0
    if compact_causal:
        key_tiles = triton.cdiv(time, bn)
        score_grid = (key_tiles * (key_tiles + 1) // 2 * (bn // bm), batch * hq)
    else:
        query_tiles = triton.cdiv(time, bm)
        key_tiles = triton.cdiv(window_size_left + bm + bn - 2, bn)
        score_grid = (query_tiles, key_tiles, batch * hq)
    _score_tiles_d128[score_grid](
        q,
        r,
        k,
        score,
        rk,
        time,
        HQ=hq,
        H=h,
        G=group_size,
        D=dim,
        BM=bm,
        BN=bn,
        BD=bd,
        SCALE_LOG2=scale * 1.4426950216,
        WINDOW_SIZE_LEFT=window_size_left,
        COMPACT_CAUSAL=compact_causal,
        num_warps=8,
        num_stages=1,
    )
    # Keep the softmax tile within the UB capacity as sequence length grows.
    br = min(8, 8192 // triton.next_power_of_2(time))
    _softmax_correction_d128[(triton.cdiv(time, br), batch * hq)](
        score,
        rk,
        prob,
        corrected,
        d1,
        bart,
        m,
        time,
        HQ=hq,
        BR=br,
        WINDOW_SIZE_LEFT=window_size_left,
        num_warps=4,
        num_stages=1,
    )
    _pv_tiles_d128[(triton.cdiv(time, bm), batch * hq)](
        prob,
        corrected,
        v,
        output,
        barv,
        time,
        HQ=hq,
        H=h,
        G=group_size,
        D=dim,
        BM=bm,
        BN=bn,
        BD=bd,
        WINDOW_SIZE_LEFT=window_size_left,
        num_warps=8,
        num_stages=1,
    )
    return output, barv, d1, bart, m
