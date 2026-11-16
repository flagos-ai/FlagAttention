"""Forgetting Attention / ACP with optional V7.6 TLE acceleration on SM90.

Q is contiguous BF16/FP16 [B, M, Hq, D], K/V [B, N, Hkv, D], and
log_fgate contiguous FP32 [B, N, Hq]. Inputs must be finite; gates and the
scalar pruning threshold must be nonpositive. Require 0 < M <= N,
Hq % Hkv == 0 and D in {16, 32, 60, 64, 100, 128}.

Use torch.inference_mode(), head_first=False and seq_start=None.
The TLE path requires a compatible Triton 3.6 / FlagTree compiler. M > 1 uses
explicit TMA + WGMMA; M = 1 uses TMA + vector reductions to preserve
reference rounding. No backward, CUDA Graph or cached attention output.

The exact prefix specializations follow the PyTorch accumulation tree for
Torch 2.10 and eligible Torch 2.9 H20 decode inputs.
Compiled-launch reuse depends on Triton runtime interfaces; revalidate
after compiler upgrades. Without compatible TLE, use the ordinary Triton path.
"""
import importlib
import math
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_attn.utils import has_triton_tle, is_sm90_device

TLE_AVAILABLE = has_triton_tle(
    3, 6, 0,
    required_api=(
        "gpu.types.BlockEncoding", "gpu.alloc", "gpu.alloc_barrier",
        "gpu.alloc_barriers", "gpu.copy", "gpu.barrier_wait", "gpu.wgmma",
        "gpu.wgmma_wait", "gpu.local_ptr", "gpu.set_layout",
    ),
)

TORCH_2_9 = torch.__version__.startswith("2.9.")


if TLE_AVAILABLE:
    try:
        import triton.experimental.tle.language as tle
        from triton import knobs
        from triton.runtime import jit as runtime_jit
        from triton.tools.tensor_descriptor import TensorDescriptor
        from triton.experimental.tle.language.gpu.types import BlockEncoding as BlockedLayout
    except ImportError:
        tle = None
        TLE_AVAILABLE = False
else:
    tle = None

# Compiled launch reuse (code and launch metadata, never caller tensors)

_trace=ContextVar('v76_launch_trace',default=None)
@contextmanager
def capture():
    records=[]
    token=_trace.set(records)
    try:
        yield records
    finally:
        _trace.reset(token)
def cdiv(x,y):return (x+y-1)//y
def pow2(x):return 1<<(x-1).bit_length()
class LaunchCache:
    def __init__(self,fn,limit=256):
        self.fn=fn
        self.limit=limit
        self.plans=OrderedDict()
        self.hits=0
        self.misses=0
        self.dist_context=getattr(runtime_jit,'DistributedRtContext',None)
    def __call__(self,key,grid,args,kwargs,fast=True):
        fn=self.fn
        # Preserve ordinary JIT behavior for debug/instrumented/hooked/distributed use.
        allowed=(fast and not fn.pre_run_hooks and not knobs.runtime.debug
                 and not knobs.compilation.instrumentation_mode)
        if self.dist_context is not None and self.dist_context().is_lite_mode:
            allowed=False
        plan=self.plans.get(key) if allowed else None
        if plan is None:
            compiled=fn[grid](*args,**kwargs)
            self.misses+=1
            if allowed:
                tail=tuple(kwargs[n] for n in fn.arg_names[len(args):])
                assert all(isinstance(x,(int,float,bool,str)) for x in tail)
                canonical=tuple(grid)+(1,)*(3-len(grid))
                # Runner retains code/grid metadata only, not this call's arguments.
                self.plans[key]=(compiled,compiled[canonical],tail)
                if len(self.plans)>self.limit:
                    self.plans.popitem(last=False)
        else:
            compiled,runner,tail=plan
            runner(*args,*tail)
            self.hits+=1
        records=_trace.get()
        if records is not None:
            records.append((fn,compiled,grid,dict(kwargs),plan is not None))
        return compiled

# Scalar threshold allocation cache

@lru_cache(maxsize=128)
def _get_cached_scalar_adaptive_threshold(
    device_index: int,
    B: int,
    H: int,
    value: float,
):
    """Create one reusable broadcast threshold view.

    v5.7a: cached scalar adaptive threshold.

    The returned tensor has shape (B, H), dtype FP32 and zero
    strides, matching torch.as_tensor(value).broadcast_to((B, H)).
    """
    scalar = torch.tensor(
        value,
        dtype=torch.float32,
        device=torch.device("cuda", device_index),
    )

    return torch.broadcast_to(
        scalar,
        (B, H),
    )

# Exact prefix scan and ACP boundary kernels

@triton.jit
def _add(a,b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;",constraints="=f,f,f",
        args=[a,b],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def _exact_prepare(X,L,S,THR, XB:tl.constexpr,XH:tl.constexpr,XT:tl.constexpr,
                   TB:tl.constexpr,TH:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
                   C:tl.constexpr,LOGC:tl.constexpr,NQ:tl.constexpr,NK:tl.constexpr):
    h=tl.program_id(0)
    b=tl.program_id(1)
    i=tl.arange(0,C)
    qi=tl.arange(0,NQ)
    ki=tl.arange(0,NK)
    anchor=tl.full((NQ,),0,tl.float32)
    ends=tl.full((NK,),0,tl.float32)
    carry=tl.full((),0,tl.float32)
    for chunk in range(tl.cdiv(N,C)):
        t=chunk*C+i
        x=tl.load(X+b*XB+h*XH+t*XT,t<N,other=0)
        x=tl.where(i==0,_add(x,carry),x)
        for m in tl.static_range(LOGC):
            src=(i//(2<<m))*(2<<m)+(1<<m)-1
            left=tl.gather(x,src,axis=0)
            x=tl.where((i&(1<<m))!=0,_add(x,left),x)
        tl.store(L+(b*H+h)*N+t,x,t<N)
        carry=tl.sum(tl.where(i==C-1,x,0),0)
        av=tl.gather(x,(qi*128)%C,axis=0)
        ev=tl.gather(x,(ki*64+63)%C,axis=0)
        anchor=tl.where(qi*128//C==chunk,av,anchor)
        ends=tl.where((ki*64+63)//C==chunk,ev,ends)
    threshold=tl.load(THR+b*TB+h*TH)
    old=(ki[None,:]<N//64)&((anchor[:,None]-ends[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*64
    tl.store(S+(b*H+h)*(N//128)+qi,starts,qi<N//128)

@triton.jit
def _h100_exact_prepare(X,L,S,THR, XB:tl.constexpr,XH:tl.constexpr,XT:tl.constexpr,
                   TB:tl.constexpr,TH:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
                   C:tl.constexpr,LOGC:tl.constexpr,NQ:tl.constexpr,NK:tl.constexpr,M:tl.constexpr,
                   BN:tl.constexpr,QM:tl.constexpr, COUNTER_OFFSET:tl.constexpr):
    h=tl.program_id(0)
    b=tl.program_id(1)
    i=tl.arange(0,C)
    qi=tl.arange(0,NQ)
    ki=tl.arange(0,NK)
    anchor=tl.full((NQ,),0,tl.float32)
    ends=tl.full((NK,),0,tl.float32)
    carry=tl.full((),0,tl.float32)
    pos_q=N-M+qi*QM
    pos_k=tl.minimum(ki*BN+BN-1,N-1)
    for chunk in range(tl.cdiv(N,C)):
        t=chunk*C+i
        x=tl.load(X+b*XB+h*XH+t*XT,t<N,other=0)
        x=tl.where(i==0,_add(x,carry),x)
        for m in tl.static_range(LOGC):
            src=(i//(2<<m))*(2<<m)+(1<<m)-1
            left=tl.gather(x,src,axis=0)
            x=tl.where((i&(1<<m))!=0,_add(x,left),x)
        tl.store(L+(b*H+h)*N+t,x,t<N)
        carry=tl.sum(tl.where(i==C-1,x,0),0)
        if not COUNTER_OFFSET:
            av=tl.gather(x,pos_q%C,axis=0)
            ev=tl.gather(x,pos_k%C,axis=0)
            anchor=tl.where(pos_q//C==chunk,av,anchor)
            ends=tl.where(pos_k//C==chunk,ev,ends)
    if COUNTER_OFFSET:
        # Decode needs just the last query anchor. Read the tile endpoints
        # once after publishing the prefix instead of gathering in every chunk.
        tl.debug_barrier()
        row_prefix = L + (b * H + h) * N
        anchor = tl.load(row_prefix + pos_q)
        ends = tl.load(row_prefix + pos_k)
    threshold=tl.load(THR+b*TB+h*TH)
    old=(ki[None,:]<tl.cdiv(N,BN))&((anchor[:,None]-ends[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+qi,starts,qi<tl.cdiv(M,QM))
    if COUNTER_OFFSET:
        tl.store(S + COUNTER_OFFSET + b * H + h, 0)

@triton.jit
def _general_starts(P, T, S, M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
                    BN:tl.constexpr,QM:tl.constexpr,NK:tl.constexpr, COUNTER_OFFSET:tl.constexpr):
    h,b,group=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    r=group*16+tl.arange(0,16)
    j=tl.arange(0,NK)
    pp=P+(b*H+h)*N
    a=tl.load(pp+N-M+r*QM,r<tl.cdiv(M,QM),other=0)
    end=tl.minimum(j*BN+BN-1,N-1)
    e=tl.load(pp+end,j<tl.cdiv(N,BN),other=0)
    threshold=tl.load(T)
    # Nonpositive log gates make the prefix nonincreasing. The first query
    # and last key give the largest decay bias in each tile.
    old=(j[None,:]<tl.cdiv(N,BN))&((a[:,None]-e[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+r,starts,r<tl.cdiv(M,QM))
    if COUNTER_OFFSET:
        tl.store(S + COUNTER_OFFSET + b * H + h, 0)


@triton.jit
def _decode_split_count(kv_tiles, max_splits: tl.constexpr):
    # Leave enough work per CTA to amortize the partial stores and merge.
    return tl.minimum(max_splits, tl.maximum(1, kv_tiles // 8))


# Compile the reducer's layout independently of the TLE attention layout.
@triton.jit(noinline=True)
def _reduce_decode_partials(
    partial, output, row, split_count,
    D: tl.constexpr, MAX_SPLITS: tl.constexpr,
):
    splits = tl.arange(0, MAX_SPLITS)
    features = tl.arange(0, D)
    active = splits < split_count
    state = partial + (row * MAX_SPLITS + splits) * (2 * D)
    local_max = tl.load(state + D, mask=active, other=-float("inf"), cache_modifier=".cg")
    local_sum = tl.load(state + D + 1, mask=active, other=0.0, cache_modifier=".cg")
    local_acc = tl.load(
        state[:, None] + features[None, :], mask=active[:, None], other=0.0, cache_modifier=".cg",
    )
    # Each partial is unnormalized and relative to its own base-2 maximum.
    weights = tl.exp2(local_max - tl.max(local_max, axis=0))
    normalizer = tl.sum(weights * local_sum, axis=0)
    acc = tl.sum(weights[:, None] * local_acc, axis=0)
    tl.store(output + row * D + features, acc * (1.0 / normalizer))

# Multi-query TMA + WGMMA attention

if TLE_AVAILABLE:
    @triton.jit
    def _tle_h100_kernel(QDESC,KDESC,VDESC,LOG_LAMBDA,START_INDEX,O,sm_scale,SLOTS:tl.constexpr,
                            M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,BN:tl.constexpr,
                            BM:tl.constexpr,STATIC:tl.constexpr,V_SLOTS:tl.constexpr):
        BLOCK_M:tl.constexpr=BM
        BLOCK_N:tl.constexpr=BN
        BLOCK_DMODEL:tl.constexpr=D
        stride_log_lambda_z:tl.constexpr=H*N
        stride_log_lambda_h:tl.constexpr=N
        stride_start_index_z:tl.constexpr=H*tl.cdiv(M,128)
        stride_start_index_h:tl.constexpr=tl.cdiv(M,128)
        stride_start_index_mb:tl.constexpr=1
        stride_oz:tl.constexpr=M*H*D
        stride_oh:tl.constexpr=D
        stride_om:tl.constexpr=H*D
        stride_ok:tl.constexpr=1
        # General packed causal inference. ACP remains grouped by 128 Q.
        dtype = O.dtype.element_ty
        h = tl.program_id(0)
        hk = h // (H // HK)
        b = tl.program_id(1)
        mblock = tl.program_id(2)
        log2e: tl.constexpr = 1.4426950408889634
        qk_scale = sm_scale * log2e
        qbuf = tle.gpu.alloc([BLOCK_M, BLOCK_DMODEL], dtype)
        kbuf = tle.gpu.alloc([SLOTS, BLOCK_N, BLOCK_DMODEL], dtype)
        vbuf = tle.gpu.alloc([V_SLOTS, BLOCK_N, BLOCK_DMODEL], dtype)
        qbar = tle.gpu.alloc_barrier(expect_bytes=BLOCK_M * BLOCK_DMODEL * 2)
        kbars = tle.gpu.alloc_barriers(SLOTS, expect_bytes=BLOCK_N * BLOCK_DMODEL * 2)
        vbars = tle.gpu.alloc_barriers(V_SLOTS, expect_bytes=BLOCK_N * BLOCK_DMODEL * 2)

        tle.gpu.copy(QDESC, qbuf, [BLOCK_M, BLOCK_DMODEL],
                     [b * M + mblock * BLOCK_M, h * BLOCK_DMODEL], barrier=qbar)
        lo = tl.load(START_INDEX + b * stride_start_index_z + h * stride_start_index_h
                     + (mblock * BLOCK_M // 128) * stride_start_index_mb)
        lo = (lo // BLOCK_N) * BLOCK_N
        hi = tl.minimum(N, N - M + (mblock + 1) * BLOCK_M)
        if lo < hi:
            tle.gpu.copy(KDESC, kbuf.slot(0), [BLOCK_N, BLOCK_DMODEL],
                         [b * N + lo, hk * BLOCK_DMODEL], barrier=kbars[0])
            tle.gpu.copy(VDESC, vbuf.slot(0), [BLOCK_N, BLOCK_DMODEL],
                         [b * N + lo, hk * BLOCK_DMODEL], barrier=vbars[0])

        rm = mblock * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = tl.arange(0, BLOCK_N)
        rd = tl.arange(0, BLOCK_DMODEL)
        prefix = LOG_LAMBDA + b * stride_log_lambda_z + h * stride_log_lambda_h
        if STATIC and M % BM == 0:
            gq = tl.load(prefix + N - M + rm, cache_modifier=".cg")
        else:
            gq = tl.load(prefix + N - M + rm, rm < M, other=0, cache_modifier=".cg")
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)
        diagonal_start = ((N - M + mblock * BLOCK_M + 1) // BLOCK_N) * BLOCK_N
        tle.gpu.barrier_wait(qbar, phaseIdx=0)

        for start_n in range(lo, hi, BLOCK_N):
            it = (start_n - lo) // BLOCK_N
            slot = it % SLOTS
            phase = it // SLOTS
            next_n = start_n + BLOCK_N
            tle.gpu.barrier_wait(kbars[slot], phaseIdx=phase)
            qk_async = tle.gpu.wgmma(qbuf, kbuf.slot(slot), trans_b=True, input_precision="ieee")
            # The next slot was released by the previous iteration's PV wait.
            if SLOTS == 2:
                if next_n < hi:
                    nxt = (it + 1) % SLOTS
                    tle.gpu.copy(KDESC, kbuf.slot(nxt), [BLOCK_N, BLOCK_DMODEL],
                                 [b * N + next_n, hk * BLOCK_DMODEL], barrier=kbars[nxt])
            if V_SLOTS == 2:
                if next_n < hi:
                    nxt = (it + 1) % V_SLOTS
                    tle.gpu.copy(VDESC, vbuf.slot(nxt), [BLOCK_N, BLOCK_DMODEL],
                                 [b * N + next_n, hk * BLOCK_DMODEL], barrier=vbars[nxt])
            if STATIC and N % BN == 0:
                gk = tl.load(prefix + start_n + rn, cache_modifier=".cg")
            else:
                gk = tl.load(prefix + start_n + rn, start_n + rn < N, other=0, cache_modifier=".cg")
            s = tle.gpu.wgmma_wait(0, qk_async) * qk_scale
            decay_bias = gq[:, None] - gk[None, :]
            # Upstream PTX fuses the decay product, not the QK scaling product.
            s = tl.fma(decay_bias, log2e, s)
            if not STATIC or N % BN != 0:
                s = tl.where((start_n + rn)[None, :] < N, s, -float("inf"))
            if start_n >= diagonal_start:
                s = tl.where((N - M + rm[:, None]) >= (start_n + rn)[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(s - m_new[:, None])
            p_sum = tl.sum(p, 1)
            v_slot = it % V_SLOTS
            tle.gpu.barrier_wait(vbars[v_slot], phaseIdx=it // V_SLOTS)
            # Match upstream TTGIR: rescaled old acc is the WGMMA accumulator.
            acc *= alpha[:, None]
            pv_async = tle.gpu.wgmma(p.to(dtype), vbuf.slot(v_slot), acc=acc, input_precision="ieee")
            l_i = l_i * alpha + p_sum
            m_i = m_new
            acc = tle.gpu.wgmma_wait(0, pv_async)
            if SLOTS == 1:
                if next_n < hi:
                    tle.gpu.copy(KDESC, kbuf.slot(0), [BLOCK_N, BLOCK_DMODEL],
                                 [b * N + next_n, hk * BLOCK_DMODEL], barrier=kbars[0])
            if V_SLOTS == 1:
                if next_n < hi:
                    # PV has released the V slot. The new copy can overlap QK.
                    tle.gpu.copy(VDESC, vbuf.slot(0), [BLOCK_N, BLOCK_DMODEL],
                                 [b * N + next_n, hk * BLOCK_DMODEL], barrier=vbars[0])

        acc = acc * (1.0 / l_i[:, None])
        op = O + b * stride_oz + h * stride_oh + rm[:, None] * stride_om + rd[None, :] * stride_ok
        if STATIC and M % BM == 0:
            tl.store(op, acc.to(dtype), cache_modifier=".cg")
        else:
            tl.store(op, acc.to(dtype), rm[:, None] < M, cache_modifier=".cg")

    # Single-query TMA + vector attention

    PV_LAYOUT=tl.constexpr(BlockedLayout([1,8],[4,8],[4,1],[1,0]))

    @triton.jit
    def _tle_decode_kernel(
        Q, KDESC, VDESC, PREFIX, STARTS, O, PARTIAL, sm_scale,
        M: tl.constexpr, N: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
        D: tl.constexpr, BN: tl.constexpr, SLOTS: tl.constexpr, MAX_SPLITS: tl.constexpr,
        COUNTER_OFFSET: tl.constexpr, PARTIAL_OFFSET: tl.constexpr,
    ):
        head = tl.program_id(0)
        batch = tl.program_id(1)
        kv_head = head // (H // HK)
        row = batch * H + head
        lo = tl.load(STARTS + row)
        hi = N
        if MAX_SPLITS > 1:
            PARTIAL += PARTIAL_OFFSET
            split = tl.program_id(2)
            kv_tiles = tl.cdiv(N - lo, BN)
            split_count = _decode_split_count(kv_tiles, MAX_SPLITS)
            if split < split_count:
                # Partition only the retained tiles. The last interval ends at N
                # when the cache has a partial tail tile.
                hi = tl.minimum(N, lo + ((split + 1) * kv_tiles // split_count) * BN)
                lo += (split * kv_tiles // split_count) * BN
                acc, mi, li = _tle_decode_range(
                    Q, KDESC, VDESC, PREFIX, sm_scale, batch, kv_head, row, lo, hi,
                    N, D, BN, SLOTS,
                )
                features = tl.arange(0, D)
                if split_count == 1:
                    tl.store(O + row * D + features, (acc * (1.0 / li)).to(Q.dtype.element_ty))
                else:
                    state = PARTIAL + (row * MAX_SPLITS + split) * (2 * D)
                    tl.store(state + features, acc)
                    # Broadcast the singleton reductions into the vector layout.
                    stats = tl.where(features == 0, mi, li)
                    tl.store(state + D + features, stats, mask=features < 2)
                    # Publish all lanes' stores before signaling completion. Only
                    # the last split reduces, so no CTA waits for unscheduled work.
                    tl.debug_barrier()
                    old_count = tl.atomic_add(
                        STARTS + COUNTER_OFFSET + row, 1, sem="acq_rel", scope="gpu",
                    )
                    if old_count == split_count - 1:
                        _reduce_decode_partials(PARTIAL, O, row, split_count, D, MAX_SPLITS)
        else:
            acc, _, li = _tle_decode_range(
                Q, KDESC, VDESC, PREFIX, sm_scale, batch, kv_head, row, lo, hi,
                N, D, BN, SLOTS,
            )
            tl.store(O + row * D + tl.arange(0, D), (acc * (1.0 / li)).to(Q.dtype.element_ty))

    @triton.jit
    def _tle_decode_range(
        Q, KDESC, VDESC, PREFIX, sm_scale, batch, kv_head, row, lo, hi,
        N: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, SLOTS: tl.constexpr,
    ):
        dtype = Q.dtype.element_ty
        ks = tle.gpu.alloc([SLOTS, BN, D], dtype)
        vs = tle.gpu.alloc([SLOTS, BN, D], dtype)
        kb = tle.gpu.alloc_barriers(SLOTS, expect_bytes=BN * D * 2)
        vb = tle.gpu.alloc_barriers(SLOTS, expect_bytes=BN * D * 2)
        tle.gpu.copy(KDESC, ks.slot(0), [BN, D], [batch * N + lo, kv_head * D], barrier=kb[0])
        tle.gpu.copy(VDESC, vs.slot(0), [BN, D], [batch * N + lo, kv_head * D], barrier=vb[0])
        # Distinct range expression prevents TLE hint conflicts when BN == D.
        rn_pair = tl.reshape(tl.arange(0, 2 * BN), (BN, 2))
        rn = tl.sum(rn_pair, 1) // 4
        rd = tl.arange(0, D)
        pp = PREFIX + row * N
        gq = tl.load(pp + N - 1)
        # Keep a rank-2 parent layout for the decode reductions.
        mi = tl.full((1,), -float("inf"), tl.float32)
        li = tl.zeros((1,), tl.float32)
        acc = tl.zeros([D], tl.float32)
        log2e: tl.constexpr = 1.4426950408889634
        scale = sm_scale * log2e
        q = tl.load(Q + row * D + rd)
        for start in range(lo, hi, BN):
            it = (start - lo) // BN
            slot = it % SLOTS
            phase = it // SLOTS
            tle.gpu.barrier_wait(kb[slot], phaseIdx=phase)
            k = tl.load(tl.max_contiguous(tl.multiple_of(tle.gpu.local_ptr(ks.slot(slot)), [1, 8]), [1, 8]))
            if start + BN < hi:
                nxt = (it + 1) % SLOTS
                tle.gpu.copy(KDESC, ks.slot(nxt), [BN, D],
                             [batch * N + start + BN, kv_head * D], barrier=kb[nxt])
                tle.gpu.copy(VDESC, vs.slot(nxt), [BN, D],
                             [batch * N + start + BN, kv_head * D], barrier=vb[nxt])
            product = tle.gpu.set_layout((q[None, :] * k).to(dtype).to(tl.float32), PV_LAYOUT)
            score = tl.sum(product, 1)[:, None] * scale
            # A nonzero masked filler avoids CSE with the D-wide zero accumulator.
            # Invalid logits are replaced by -inf below, so this filler is unobservable.
            gk = tl.load(pp + start + rn, start + rn < N, other=-10000)[:, None]
            score = tl.fma(gq - gk, log2e, score)
            score = tl.where((start + rn < N)[:, None], score, -float("inf"))
            mnew = tl.maximum(mi, tl.max(score, 0))
            alpha = tl.exp2(mi - mnew)
            p = tl.exp2(score - mnew[None, :])
            psum = tl.sum(p, 0)
            tle.gpu.barrier_wait(vb[slot], phaseIdx=phase)
            v = tl.load(tl.max_contiguous(tl.multiple_of(tle.gpu.local_ptr(vs.slot(slot)), [1, 8]), [1, 8]))
            product_v = tle.gpu.set_layout(p * v, PV_LAYOUT)
            acc = acc * alpha + tl.sum(product_v, 0)
            li = li * alpha + psum
            mi = mnew
        return acc, mi, li

# Prefix and boundary launchers

if TLE_AVAILABLE:
    general=LaunchCache(_h100_exact_prepare)
    anchor=LaunchCache(_exact_prepare)
    starts_launch=LaunchCache(_general_starts)


def _allocate_starts(batch, heads, query_groups, device, counter_offset):
    shape = (batch, heads, query_groups)
    # Split decode has one start and one counter per batch/head. Keep the
    # counter plane in the same allocation so preparation can initialize it.
    if counter_offset:
        shape = (2, *shape)
    return torch.empty(shape, device=device, dtype=torch.int32)


def exact(gate, threshold, m, bn, qm, anchor_mode=False, fast=True, *,
          counter_offset=0, workspace_elements=0):
    if workspace_elements:
        b, n, h = gate.shape
        gate_strides = (gate.stride(0), gate.stride(2), gate.stride(1))
    else:
        b, h, n = gate.shape
        gate_strides = gate.stride()
    ln=(n-1).bit_length()
    lr=(b*h-1).bit_length()
    lx=min(9,max(4,(9+ln-lr)//2))
    chunk=2*(1<<lx)
    prefix_shape = (workspace_elements,) if workspace_elements else (b, h, n)
    prefix = torch.empty(prefix_shape, device=gate.device, dtype=torch.float32)
    starts = _allocate_starts(b, h, cdiv(m, qm), gate.device, counter_offset)
    args=(gate,prefix,starts,threshold,*gate_strides,*threshold.stride(),
          h,n,chunk,chunk.bit_length()-1,pow2(cdiv(m,qm)),pow2(cdiv(n,bn)))
    if not anchor_mode:
        args+= (m,bn,qm,counter_offset)
    # Every pointer except gate is an aligned fresh allocation or validated scalar buffer.
    key=(gate.device.index,b,gate.data_ptr()%16,args[4:])
    num_warps = 1 if workspace_elements else 4
    (anchor if anchor_mode else general)(key,(h,b),args,dict(num_warps=num_warps),fast)
    return prefix,starts
def starts(prefix,threshold,M,N,H,BN,QM,fast=True,*,counter_offset=0):
    B=prefix.shape[0]
    NQ=cdiv(M,QM)
    out = _allocate_starts(B, H, NQ, prefix.device, counter_offset)
    args=(prefix,threshold,out,M,N,H,BN,QM,pow2(cdiv(N,BN)),counter_offset)
    starts_launch((prefix.device.index,B,args[3:]),(H,B,cdiv(NQ,16)),args,dict(num_warps=4),fast)
    return out

prepare_starts = starts

# Input contract, resource selection and public entry

def _decode_block_n(n):
    return min(128, max(16, pow2(n)))


def prepare(q, k, v, gate, head_first, seq_start, sm_scale, adaptive_threshold, prep_mode, fast, *,
            counter_offset=0, workspace_elements=0):
    if head_first or seq_start is not None or not torch.is_inference_mode_enabled():
        raise NotImplementedError("general profile supports BTHD inference without seq_start")
    B,M,H,D=q.shape
    if k.ndim!=4 or v.shape!=k.shape:
        raise ValueError("K/V shape mismatch")
    N,HK=k.shape[1:3]
    if D not in (16,32,60,64,100,128) or k.shape!=(B,N,HK,D) or HK<=0 or H%HK:
        raise NotImplementedError("require supported D and Q heads divisible by KV heads")
    if not (0<M<=N and B>0 and H>0):
        raise ValueError("require 0 < M <= N and positive batch/heads")
    if not (q.dtype in (torch.bfloat16,torch.float16) and q.is_cuda and
            all(t.dtype==q.dtype and t.device==q.device and t.is_contiguous()
                and t.data_ptr()%16==0 for t in (q,k,v))):
        raise NotImplementedError("require contiguous, 16-byte-aligned CUDA BF16/FP16 inputs")
    if gate.shape!=(B,N,H) or gate.dtype!=torch.float32 or gate.device!=q.device or not gate.is_contiguous():
        raise NotImplementedError("gate must be contiguous FP32 BNH")
    if not isinstance(adaptive_threshold,(int,float)) or not math.isfinite(adaptive_threshold) or adaptive_threshold>0:
        raise NotImplementedError("require a finite nonpositive scalar ACP threshold")
    if torch.cuda.get_device_capability(q.device)!=(9,0):
        raise NotImplementedError("Hopper SM90 required")
    scale=1/math.sqrt(D) if sm_scale is None else sm_scale
    if not isinstance(scale,(int,float)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite scalar")
    BN=_decode_block_n(N) if M==1 else (64 if D<=64 else 128)
    QM=1 if M==1 else 128
    threshold=_get_cached_scalar_adaptive_threshold(q.device.index,B,H,float(adaptive_threshold))
    if workspace_elements:
        # Torch 2.9 uses the same per-row Sklansky scan for these decode inputs.
        prefix, starts = exact(gate, threshold, M, BN, QM, fast=fast,
                               counter_offset=counter_offset, workspace_elements=workspace_elements)
    elif prep_mode and B*H>1 and torch.__version__.startswith('2.10.'):
        prefix,starts=exact(gate.transpose(1,2),threshold,M,BN,QM,fast=fast,counter_offset=counter_offset)
    elif (B,M,N,H,D)==(4,4096,4096,32,64) and torch.__version__.startswith('2.10.'):
        prefix,starts=exact(gate.transpose(1,2),threshold,M,BN,QM,anchor_mode=True,fast=fast)
    else:
        # The original Torch accumulation tree is part of the bitwise contract.
        prefix=torch.cumsum(gate.transpose(1,2),dim=-1,dtype=torch.float32)
        starts=prepare_starts(prefix,threshold,M,N,H,BN,QM,fast,counter_offset=counter_offset)
    out=torch.empty((*q.shape[:-1],pow2(D)),device=q.device,dtype=q.dtype)
    return prefix,starts,out,scale,BN

@lru_cache(maxsize=16)
def shared_layout(layout_type,D):
    return () if layout_type is None else (layout_type(
        swizzle_byte_width=min(128,D*2),element_bitwidth=16,rank=2,transposed=False),)

@lru_cache(maxsize=512)
def h100_config(B,M,N,H,HK,D):
    # ACP grouping / reduction BN are fixed; only independent resource choices vary.
    prep=0 if (B,M,N,H,D)==(4,4096,4096,32,64) else 1
    return (64,1 if D>64 else 2,prep,True,0)


@lru_cache(maxsize=16)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@lru_cache(maxsize=16)
def _is_h20(device_index: int) -> bool:
    return "H20" in torch.cuda.get_device_name(device_index)


def _use_h20_tiles(q, k) -> bool:
    if q.ndim != 4 or k.ndim != 4 or q.shape[-1] != 128:
        return False
    batch, queries, heads, _ = q.shape
    rows = batch * heads
    # Small prefill grids are sensitive to launch overhead. Split decode
    # needs the fused prefix preparation to amortize the extra CTAs.
    prefill = queries >= 2048 and rows * queries >= 131072
    decode = TORCH_2_9 and queries == 1 and 8 <= rows <= 16 and 8192 <= k.shape[1] <= 32768
    return (prefill or decode) and _is_h20(q.device.index)


def _max_decode_splits(batch: int, heads: int, kv_tiles: int, device_index: int) -> int:
    if kv_tiles < 16:
        return 1
    # A decode CTA is one query head. Target one full wave on the current GPU.
    groups = triton.cdiv(_sm_count(device_index), batch * heads)
    return min(16, triton.next_power_of_2(groups), triton.next_power_of_2(kv_tiles // 8))


def make_entry(
    kind, kernel, decode, Descriptor, layout_type=None, config_override=None,
    fast=True, *, optimize_hopper=False,
):
    main_launch=LaunchCache(kernel)
    decode_launch=LaunchCache(decode)
    def forgetting_attention(q,k,v,log_fgate,*,head_first=False,seq_start=None,
                             sm_scale=None,adaptive_threshold=None):
        with torch.cuda.device(q.device):
            B,M,H,realD=q.shape
            N,HK=k.shape[1:3]
            cfg=h100_config(B,M,N,H,HK,realD) if config_override is None else config_override
            BM,SLOTS,PREP,STATIC,MMA_N=cfg
            D=pow2(realD)
            max_splits = 1
            if M==1:
                BM=1
                SLOTS=2
                if optimize_hopper and B > 0 and H > 0 and N > 0:
                    max_splits = _max_decode_splits(B, H, cdiv(N, _decode_block_n(N)), q.device.index)
            counter_offset = B * H if max_splits > 1 else 0
            partial_offset = workspace_elements = 0
            if counter_offset and PREP and TORCH_2_9 and B * H > 1:
                # Share one FP32 allocation. Align the first partial to 16 bytes.
                partial_offset = cdiv(B * H * N, 4) * 4
                workspace_elements = partial_offset + B * H * max_splits * 2 * D
            prefix,starts,out,scale,BN=prepare(q,k,v,log_fgate,head_first,seq_start,
                                              sm_scale,adaptive_threshold,PREP,fast,
                                              counter_offset=counter_offset,
                                              workspace_elements=workspace_elements)
            v_slots = SLOTS
            if optimize_hopper and M >= 64 and realD == 128:
                # Keep the original ACP boundaries from prepare(). With the
                # smaller compute tile, prefetch K and reuse one V buffer.
                BN, SLOTS, v_slots = 64, 2, 1
            if D!=realD:
                q,k,v=(F.pad(x,(0,D-realD)) for x in (q,k,v))
            layout=shared_layout(layout_type,D)
            qd=q if M==1 else Descriptor(q,[B*M,H*D],[H*D,1],[BM,D],*layout)
            kd=Descriptor(k,[B*N,HK*D],[HK*D,1],[BN,D],*layout)
            vd=Descriptor(v,[B*N,HK*D],[HK*D,1],[BN,D],*layout)
            kwargs = {} if M == 1 else dict(BM=BM, STATIC=STATIC, V_SLOTS=v_slots)
            key = (q.device.index, q.dtype, B, M, N, H, HK, D, BM, BN, SLOTS,
                   v_slots, STATIC, MMA_N, max_splits, type(scale), float(scale))
            kwargs.update(SLOTS=SLOTS,M=M,N=N,H=H,HK=HK,D=D,BN=BN,num_warps=4,num_stages=1)
            if M == 1:
                partial = out
                if partial_offset:
                    partial = prefix
                elif max_splits > 1:
                    # [acc, max, sum, padding] keeps the vector start aligned.
                    partial = torch.empty((B, H, max_splits, 2 * D),
                                          device=q.device, dtype=torch.float32)
                kwargs["MAX_SPLITS"] = max_splits
                kwargs["COUNTER_OFFSET"] = counter_offset
                kwargs["PARTIAL_OFFSET"] = partial_offset
                decode_launch(
                    key, (H, B, max_splits),
                    (qd, kd, vd, prefix, starts, out, partial, scale), kwargs, fast,
                )
            else:
                main_launch(
                    key, (H, B, cdiv(M, BM)),
                    (qd, kd, vd, prefix, starts, out, scale), kwargs, fast,
                )
            return out if D==realD else out[...,:realD].contiguous()
    return forgetting_attention

def entry_for(config=None, fast=True, *, optimize_hopper=False):
    return make_entry(
        "tle", _tle_h100_kernel, _tle_decode_kernel,
        TensorDescriptor, None, config, fast, optimize_hopper=optimize_hopper,
    )


_tle_forgetting_attention = entry_for() if TLE_AVAILABLE else None
_hopper_forgetting_attention = entry_for(optimize_hopper=True) if TLE_AVAILABLE else None


def parallel_forgetting_attn(
    q, k, v, log_fgate, *, head_first=False, seq_start=None,
    sm_scale=None, adaptive_threshold=None,
):
    """Run causal Forgetting Attention on query positions aligned to the KV tail.

    On the SM90 TLE path, use contiguous BF16/FP16 Q [B, M, Hq, D] and
    K/V [B, N, Hkv, D] with Hq divisible by Hkv. Pass contiguous FP32
    log_fgate [B, N, Hq] with finite nonpositive values, a finite
    nonpositive adaptive_threshold, and run inside torch.inference_mode().
    head_first must be False and seq_start must be None on this path.

    Eligible H20 inputs use smaller prefill tiles or per-head KV splits.
    These specializations preserve the ACP mask and can change floating-point
    reduction order. The result has the same shape and dtype as Q.
    """
    if (
        q.is_cuda
        and is_sm90_device(q.device)
        and TLE_AVAILABLE
    ):
        entry = _hopper_forgetting_attention if _use_h20_tiles(q, k) else _tle_forgetting_attention
        return entry(
            q, k, v, log_fgate, head_first=head_first, seq_start=seq_start,
            sm_scale=sm_scale, adaptive_threshold=adaptive_threshold,
        )

    dimension = q.shape[-1]
    if dimension not in (16, 32, 60, 64, 100, 128):
        raise ValueError("head dimension must be one of 16, 32, 60, 64, 100, 128")
    if k.shape[-1] != dimension or v.shape[-1] != dimension:
        raise ValueError("q, k and v must have the same head dimension")
    head_axis = 1 if head_first else 2
    heads, kv_heads = q.shape[head_axis], k.shape[head_axis]
    if kv_heads <= 0 or heads <= 0 or heads % kv_heads or v.shape[head_axis] != kv_heads:
        raise ValueError("query heads must be a positive multiple of key/value heads")
    # Keep the scale of the original dimension when zero-padding q/k/v.
    scale = 1.0 / math.sqrt(dimension) if sm_scale is None else sm_scale
    padded = 1 << (dimension - 1).bit_length()
    if dimension != padded:
        q, k, v = (F.pad(t, (0, padded - dimension)) for t in (q, k, v))
    groups = heads // kv_heads
    if groups != 1:
        k, v = (t.repeat_interleave(groups, dim=head_axis) for t in (k, v))
    entry = importlib.import_module(".naive", __package__).forgetting_attention
    with torch.cuda.device(q.device):
        output = entry(
            q, k, v, log_fgate, head_first=head_first, seq_start=seq_start,
            sm_scale=scale, adaptive_threshold=adaptive_threshold,
        )
    return output if dimension == padded else output[..., :dimension].contiguous()


forgetting_attention = parallel_forgetting_attn
