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

"""T-Head Triton kernels for paged relative attention and Split-KV merging."""

import triton
import triton.language as tl


@triton.jit
def _attention(
    Q, K, V, R, O, PO, LSE, BT, KL, CU,
    sq0: tl.constexpr, sq1: tl.constexpr, sq2: tl.constexpr,
    sk0: tl.constexpr, sk1: tl.constexpr, sk2: tl.constexpr, sk3: tl.constexpr,
    sv0: tl.constexpr, sv1: tl.constexpr, sv2: tl.constexpr, sv3: tl.constexpr,
    sr0: tl.constexpr, sr1: tl.constexpr, sr2: tl.constexpr,
    so0: tl.constexpr, so1: tl.constexpr, so2: tl.constexpr,
    sb0: tl.constexpr, sb1: tl.constexpr,
    scale, TOTAL_Q: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
    DQ: tl.constexpr, DV: tl.constexpr, BDQ: tl.constexpr, BDV: tl.constexpr,
    PAGE: tl.constexpr, EXTENT: tl.constexpr, G: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, SPLITS: tl.constexpr, STAGES: tl.constexpr,
    CAUSAL: tl.constexpr, LOCAL: tl.constexpr,
    HAS_LEFT: tl.constexpr, HAS_RIGHT: tl.constexpr,
    LEFT: tl.constexpr, RIGHT: tl.constexpr,
):
    mblock = tl.program_id(0)
    batch = tl.program_id(1) // HK
    kh = tl.program_id(1) % HK
    split = tl.program_id(2)
    qstart = tl.load(CU + batch)
    qend = tl.load(CU + batch + 1)
    qlen = qend - qstart
    klen = tl.load(KL + batch)
    if mblock * BM < qlen * G:
        # PackGQA: packed row = logical_query * group_size + head_within_group.
        rows = mblock * BM + tl.arange(0, BM)
        qi = rows // G
        hi = kh * G + rows % G
        validq = qi < qlen
        d = tl.arange(0, BDQ)
        dv = tl.arange(0, BDV)
        q = tl.load(Q + (qstart + qi[:, None]) * sq0 + hi[:, None] * sq1 + d[None, :] * sq2,
                    validq[:, None] & (d[None, :] < DQ), 0)
        # BlockInfo.get_n_block_min_max: bound the visible KV interval BEFORE splitting.
        nmax = tl.cdiv(klen, BN)
        if CAUSAL or (LOCAL and HAS_RIGHT):
            qmax = tl.cdiv((mblock + 1) * BM, G)
            right_edge = qmax + klen - qlen
            if not CAUSAL:
                right_edge += RIGHT
            nmax = tl.minimum(nmax, tl.cdiv(tl.maximum(right_edge, 0), BN))
        nmin = 0
        if LOCAL and HAS_LEFT:
            qmin = mblock * BM // G
            nmin = tl.maximum((qmin + klen - qlen - LEFT) // BN, 0)
        per_split = tl.cdiv(tl.maximum(nmax - nmin, 0), SPLITS)
        nmin = nmin + split * per_split
        nmax = tl.minimum(nmin + per_split, nmax)

        rowmax = tl.full((BM,), -float("inf"), tl.float32)
        rowsum = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, BDV), tl.float32)
        lane = tl.arange(0, BN)
        qpos = qi + klen - qlen
        # CuTe's consumer visits the last visible key block first.
        # The PPU compiler controls legal buffering and waits.
        for step in tl.range(0, tl.maximum(nmax - nmin, 0), num_stages=STAGES):
            nblock = nmax - 1 - step
            ki = nblock * BN + lane
            page_id = tl.load(BT + batch * sb0 + (ki // PAGE) * sb1, ki < klen, 0)
            kinpage = ki % PAGE
            kp = K + page_id[:, None] * sk0 + kinpage[:, None] * sk1 + kh * sk2 + d[None, :] * sk3
            vp = V + page_id[:, None] * sv0 + kinpage[:, None] * sv1 + kh * sv2 + dv[None, :] * sv3
            k = tl.load(kp, (ki[:, None] < klen) & (d[None, :] < DQ), 0)
            v = tl.load(vp, (ki[:, None] < klen) & (dv[None, :] < DV), 0)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            dist = qpos[:, None] - ki[None, :]
            ridx = tl.minimum(tl.maximum(dist, 0), EXTENT - 1)
            rel_ptr = R + (qstart + qi[:, None]) * sr0 + hi[:, None] * sr1 + ridx * sr2
            bias_valid = (validq[:, None] & (dist >= 0) & (dist < EXTENT)).to(tl.int32)
            # Predicated native gather keeps the score add in its MMA layout.
            if R.dtype.element_ty == tl.bfloat16:
                scores = tl.inline_asm_elementwise(
                    "{ .reg .pred p; .reg .b32 raw; .reg .f32 bias; "
                    "ppu.mov.b32 raw, 0; ppu.setp.ne.u32 p, $3, 0; "
                    "@p ppu.ld.global.u16 raw, [$2]; "
                    "ppu.shl.b32 raw, raw, 16; ppu.mov.b32 bias, raw; "
                    "ppu.add.f32 $0, $1, bias; }",
                    "=f,f,l,r", [scores, rel_ptr.to(tl.uint64), bias_valid],
                    dtype=tl.float32, is_pure=False, pack=1,
                )
            else:
                scores = tl.inline_asm_elementwise(
                    "{ .reg .pred p; .reg .b32 raw; .reg .f32 bias; "
                    "ppu.mov.b32 raw, 0; ppu.setp.ne.u32 p, $3, 0; "
                    "@p ppu.ld.global.u16 raw, [$2]; "
                    "ppu.cvt.f32.f16 bias, raw; ppu.add.f32 $0, $1, bias; }",
                    "=f,f,l,r", [scores, rel_ptr.to(tl.uint64), bias_valid],
                    dtype=tl.float32, is_pure=False, pack=1,
                )
            mask = validq[:, None] & (ki[None, :] < klen)
            if CAUSAL:
                mask = mask & (ki[None, :] <= qpos[:, None])
            if LOCAL and HAS_LEFT:
                mask = mask & (ki[None, :] >= qpos[:, None] - LEFT)
            if LOCAL and HAS_RIGHT:
                mask = mask & (ki[None, :] <= qpos[:, None] + RIGHT)
            scores = tl.where(mask, scores, -float("inf"))
            newmax = tl.maximum(rowmax, tl.max(scores, axis=1))
            safe_max = tl.where(newmax == -float("inf"), 0., newmax)
            alpha = tl.exp2((rowmax - safe_max) * 1.4426950408889634)
            p = tl.exp2((scores - safe_max[:, None]) * 1.4426950408889634)
            rowsum = rowsum * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            # Same probability precision as CuTe BF16/FP16 PV MMA; FP32 accumulation.
            acc = tl.dot(p.to(q.dtype), v, acc, out_dtype=tl.float32)
            rowmax = newmax

        invsum = 1. / tl.where(rowsum > 0, rowsum, 1.)
        normalized = acc * invsum[:, None]
        if SPLITS > 1:
            # CuTe partial outputs are already normalized; combine is weighted by LSE.
            poff = ((split * TOTAL_Q + qstart + qi[:, None]) * H + hi[:, None]) * DV + dv[None, :]
            tl.store(PO + poff, normalized, validq[:, None] & (dv[None, :] < DV))
            lse = tl.where(
                rowsum > 0,
                rowmax + tl.log2(tl.where(rowsum > 0, rowsum, 1.)) * 0.6931471805599453,
                -float("inf"),
            )
            tl.store(LSE + (split * TOTAL_Q + qstart + qi) * H + hi, lse, validq)
        else:
            tl.store(O + (qstart + qi[:, None]) * so0 + hi[:, None] * so1 + dv[None, :] * so2,
                     normalized, validq[:, None] & (dv[None, :] < DV))


@triton.jit
def _combine(PO, LSE, O, TOTAL_Q: tl.constexpr, H: tl.constexpr, DV: tl.constexpr,
             so0: tl.constexpr, so1: tl.constexpr, so2: tl.constexpr,
             SPLITS: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    splits = tl.arange(0, BS)
    d = tl.arange(0, BD)
    lse = tl.load(LSE + splits * TOTAL_Q * H + row, splits < SPLITS, -float("inf"))
    m = tl.max(lse, 0)
    safe_m = tl.where(m == -float("inf"), 0., m)
    weights = tl.exp2((lse - safe_m) * 1.4426950408889634)
    total = tl.sum(weights, 0)
    weights = weights / tl.where(total > 0, total, 1.)
    partial = tl.load(PO + (splits[:, None] * TOTAL_Q * H + row) * DV + d[None, :],
                      (splits[:, None] < SPLITS) & (d[None, :] < DV), 0)
    result = tl.sum(partial * weights[:, None], 0)
    tl.store(O + (row // H) * so0 + (row % H) * so1 + d * so2, result, d < DV)
