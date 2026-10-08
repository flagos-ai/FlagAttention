"""V7.6 TLE Forgetting Attention / ACP, Hopper SM90 inference.

Q is contiguous BF16/FP16 [B, M, Hq, D], K/V [B, N, Hkv, D], and
log_fgate contiguous FP32 [B, N, Hq]. Inputs must be finite; gates and the
scalar pruning threshold must be nonpositive. Require 0 < M <= N,
Hq % Hkv == 0 and D in {16, 32, 60, 64, 100, 128}.

Use torch.inference_mode(), head_first=False and seq_start=None.
A compatible Triton 3.6 / FlagTree TLE compiler is required. M > 1 uses
explicit TMA + WGMMA; M = 1 uses TMA + vector reductions to preserve
reference rounding. No backward, CUDA Graph or cached attention output.

The Torch 2.10 prefix specialization matches its accumulation tree.
Other Torch versions retain torch.cumsum and require fresh validation.
Compiled-launch reuse depends on Triton runtime interfaces; revalidate
after compiler upgrades.
"""
import math
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import knobs
from triton.runtime import jit as runtime_jit
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.experimental.tle.language.gpu.types import BlockEncoding as BlockedLayout

# Compiled launch reuse (code and launch metadata, never caller tensors)

_trace=ContextVar('v76_launch_trace',default=None)
@contextmanager
def capture():
    records=[];token=_trace.set(records)
    try:yield records
    finally:_trace.reset(token)
def cdiv(x,y):return (x+y-1)//y
def pow2(x):return 1<<(x-1).bit_length()
class LaunchCache:
    def __init__(self,fn,limit=256):
        self.fn=fn;self.limit=limit;self.plans=OrderedDict()
        self.hits=0;self.misses=0
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
                if len(self.plans)>self.limit:self.plans.popitem(last=False)
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
    h=tl.program_id(0);b=tl.program_id(1)
    i=tl.arange(0,C);qi=tl.arange(0,NQ);ki=tl.arange(0,NK)
    anchor=tl.full((NQ,),0,tl.float32);ends=tl.full((NK,),0,tl.float32)
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
                   BN:tl.constexpr,QM:tl.constexpr):
    h=tl.program_id(0);b=tl.program_id(1)
    i=tl.arange(0,C);qi=tl.arange(0,NQ);ki=tl.arange(0,NK)
    anchor=tl.full((NQ,),0,tl.float32);ends=tl.full((NK,),0,tl.float32)
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
        av=tl.gather(x,pos_q%C,axis=0)
        ev=tl.gather(x,pos_k%C,axis=0)
        anchor=tl.where(pos_q//C==chunk,av,anchor)
        ends=tl.where(pos_k//C==chunk,ev,ends)
    threshold=tl.load(THR+b*TB+h*TH)
    old=(ki[None,:]<tl.cdiv(N,BN))&((anchor[:,None]-ends[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+qi,starts,qi<tl.cdiv(M,QM))

@triton.jit
def _general_starts(P, T, S, M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
                    BN:tl.constexpr,QM:tl.constexpr,NK:tl.constexpr):
    h,b,group=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    r=group*16+tl.arange(0,16)
    j=tl.arange(0,NK)
    pp=P+(b*H+h)*N
    a=tl.load(pp+N-M+r*QM,r<tl.cdiv(M,QM),other=0)
    end=tl.minimum(j*BN+BN-1,N-1)
    e=tl.load(pp+end,j<tl.cdiv(N,BN),other=0)
    threshold=tl.load(T)
    old=(j[None,:]<tl.cdiv(N,BN))&((a[:,None]-e[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+r,starts,r<tl.cdiv(M,QM))

# Multi-query TMA + WGMMA attention

@triton.jit
def _tle_h100_kernel(QDESC,KDESC,VDESC,LOG_LAMBDA,START_INDEX,O,sm_scale,SLOTS:tl.constexpr,
                        M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,BN:tl.constexpr,
                        BM:tl.constexpr,STATIC:tl.constexpr):
    BLOCK_M:tl.constexpr=BM
    BLOCK_N:tl.constexpr=BN
    BLOCK_DMODEL:tl.constexpr=D
    STORE_L:tl.constexpr=False
    stride_log_lambda_z:tl.constexpr=H*N
    stride_log_lambda_h:tl.constexpr=N
    stride_log_lambda_n:tl.constexpr=1
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
    vbuf = tle.gpu.alloc([SLOTS, BLOCK_N, BLOCK_DMODEL], dtype)
    qbar = tle.gpu.alloc_barrier(expect_bytes=BLOCK_M * BLOCK_DMODEL * 2)
    kbars = tle.gpu.alloc_barriers(SLOTS, expect_bytes=BLOCK_N * BLOCK_DMODEL * 2)
    vbars = tle.gpu.alloc_barriers(SLOTS, expect_bytes=BLOCK_N * BLOCK_DMODEL * 2)

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
        tle.gpu.barrier_wait(vbars[slot], phaseIdx=phase)
        # Match upstream TTGIR: rescaled old acc is the WGMMA accumulator.
        acc *= alpha[:, None]
        pv_async = tle.gpu.wgmma(p.to(dtype), vbuf.slot(slot), acc=acc, input_precision="ieee")
        l_i = l_i * alpha + p_sum
        m_i = m_new
        acc = tle.gpu.wgmma_wait(0, pv_async)
        if SLOTS == 1:
            if next_n < hi:
                tle.gpu.copy(KDESC, kbuf.slot(0), [BLOCK_N, BLOCK_DMODEL],
                             [b * N + next_n, hk * BLOCK_DMODEL], barrier=kbars[0])
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
def _tle_decode_kernel(QDESC,KDESC,VDESC,PREFIX,STARTS,O,sm_scale,
                       M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
                       BN:tl.constexpr,SLOTS:tl.constexpr):
    dtype=O.dtype.element_ty
    h,b=tl.program_id(0),tl.program_id(1)
    hk=h//(H//HK)
    ks=tle.gpu.alloc([SLOTS,BN,D],dtype)
    vs=tle.gpu.alloc([SLOTS,BN,D],dtype)
    kb=tle.gpu.alloc_barriers(SLOTS,expect_bytes=BN*D*2)
    vb=tle.gpu.alloc_barriers(SLOTS,expect_bytes=BN*D*2)
    lo=tl.load(STARTS+b*H+h)
    tle.gpu.copy(KDESC,ks.slot(0),[BN,D],[b*N+lo,hk*D],barrier=kb[0])
    tle.gpu.copy(VDESC,vs.slot(0),[BN,D],[b*N+lo,hk*D],barrier=vb[0])
    # Distinct range expression prevents TLE hint conflicts when BN == D.
    rn_pair=tl.reshape(tl.arange(0,2*BN),(BN,2))
    rn=tl.sum(rn_pair,1)//4
    rd=tl.arange(0,D)
    pp=PREFIX+(b*H+h)*N
    gq=tl.load(pp+N-1)
    mi=tl.full((),-float("inf"),tl.float32)
    li=tl.zeros((),tl.float32)
    acc=tl.zeros([D],tl.float32)
    log2e:tl.constexpr=1.4426950408889634
    scale=sm_scale*log2e
    q=tl.load(QDESC+(b*H+h)*D+rd)
    for start in range(lo,N,BN):
        it=(start-lo)//BN
        slot=it%SLOTS;phase=it//SLOTS
        tle.gpu.barrier_wait(kb[slot],phaseIdx=phase)
        k=tl.load(tl.max_contiguous(tl.multiple_of(tle.gpu.local_ptr(ks.slot(slot)),[1,8]),[1,8]))
        if start+BN<N:
            nxt=(it+1)%SLOTS
            tle.gpu.copy(KDESC,ks.slot(nxt),[BN,D],[b*N+start+BN,hk*D],barrier=kb[nxt])
            tle.gpu.copy(VDESC,vs.slot(nxt),[BN,D],[b*N+start+BN,hk*D],barrier=vb[nxt])
        product=tle.gpu.set_layout((q[None,:]*k).to(dtype).to(tl.float32),PV_LAYOUT)
        score=tl.sum(product,1)*scale
        # A nonzero masked filler avoids CSE with the D-wide zero accumulator.
        # Invalid logits are replaced by -inf below, so this filler is unobservable.
        gk=tl.load(pp+start+rn,start+rn<N,other=-10000)
        score=tl.fma(gq-gk,log2e,score)
        score=tl.where(start+rn<N,score,-float("inf"))
        mnew=tl.maximum(mi,tl.max(score,0))
        alpha=tl.exp2(mi-mnew)
        p=tl.exp2(score-mnew)
        psum=tl.sum(p,0)
        tle.gpu.barrier_wait(vb[slot],phaseIdx=phase)
        v=tl.load(tl.max_contiguous(tl.multiple_of(tle.gpu.local_ptr(vs.slot(slot)),[1,8]),[1,8]))
        product_v=tle.gpu.set_layout(p[:,None]*v,PV_LAYOUT)
        acc=acc*alpha+tl.sum(product_v,0)
        li=li*alpha+psum
        mi=mnew
    tl.store(O+(b*H+h)*D+rd,(acc*(1.0/li)).to(dtype))

# Prefix and boundary launchers

general=LaunchCache(_h100_exact_prepare)
anchor=LaunchCache(_exact_prepare)
starts_launch=LaunchCache(_general_starts)
def exact(gate,threshold,m,bn,qm,anchor_mode=False,fast=True):
    b,h,n=gate.shape
    ln=(n-1).bit_length();lr=(b*h-1).bit_length()
    lx=min(9,max(4,(9+ln-lr)//2));chunk=2*(1<<lx)
    prefix=torch.empty((b,h,n),device=gate.device,dtype=torch.float32)
    starts=torch.empty((b,h,cdiv(m,qm)),device=gate.device,dtype=torch.int32)
    args=(gate,prefix,starts,threshold,*gate.stride(),*threshold.stride(),
          h,n,chunk,chunk.bit_length()-1,pow2(cdiv(m,qm)),pow2(cdiv(n,bn)))
    if not anchor_mode:args+= (m,bn,qm)
    # Every pointer except gate is an aligned fresh allocation or validated scalar buffer.
    key=(gate.device.index,b,gate.data_ptr()%16,args[4:])
    (anchor if anchor_mode else general)(key,(h,b),args,dict(num_warps=4),fast)
    return prefix,starts
def starts(prefix,threshold,M,N,H,BN,QM,fast=True):
    B=prefix.shape[0];NQ=cdiv(M,QM)
    out=torch.empty((B,H,NQ),device=prefix.device,dtype=torch.int32)
    args=(prefix,threshold,out,M,N,H,BN,QM,pow2(cdiv(N,BN)))
    starts_launch((prefix.device.index,B,args[3:]),(H,B,cdiv(NQ,16)),args,dict(num_warps=4),fast)
    return out

prepare_starts = starts

# Input contract, resource selection and public entry

def prepare(q,k,v,gate,head_first,seq_start,sm_scale,adaptive_threshold,prep_mode,fast):
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
    BN=min(128,max(16,pow2(N))) if M==1 else (64 if D<=64 else 128)
    QM=1 if M==1 else 128
    threshold=_get_cached_scalar_adaptive_threshold(q.device.index,B,H,float(adaptive_threshold))
    if prep_mode and B*H>1 and torch.__version__.startswith('2.10.'):
        prefix,starts=exact(gate.transpose(1,2),threshold,M,BN,QM,fast=fast)
    elif (B,M,N,H,D)==(4,4096,4096,32,64) and torch.__version__.startswith('2.10.'):
        prefix,starts=exact(gate.transpose(1,2),threshold,M,BN,QM,anchor_mode=True,fast=fast)
    else:
        # The original Torch accumulation tree is part of the bitwise contract.
        prefix=torch.cumsum(gate.transpose(1,2),dim=-1,dtype=torch.float32)
        starts=prepare_starts(prefix,threshold,M,N,H,BN,QM,fast)
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

def make_entry(kind,kernel,decode,Descriptor,layout_type=None,config_override=None,fast=True):
    main_launch=LaunchCache(kernel);decode_launch=LaunchCache(decode)
    def forgetting_attention(q,k,v,log_fgate,*,head_first=False,seq_start=None,
                             sm_scale=None,adaptive_threshold=None):
        with torch.cuda.device(q.device):
            B,M,H,realD=q.shape;N,HK=k.shape[1:3]
            cfg=h100_config(B,M,N,H,HK,realD) if config_override is None else config_override
            BM,SLOTS,PREP,STATIC,MMA_N=cfg
            if M==1:BM=1;SLOTS=2
            prefix,starts,out,scale,BN=prepare(q,k,v,log_fgate,head_first,seq_start,
                                              sm_scale,adaptive_threshold,PREP,fast)
            D=pow2(realD)
            if D!=realD:
                q,k,v=(F.pad(x,(0,D-realD)) for x in (q,k,v))
            layout=shared_layout(layout_type,D)
            qd=q if M==1 else Descriptor(q,[B*M,H*D],[H*D,1],[BM,D],*layout)
            kd=Descriptor(k,[B*N,HK*D],[HK*D,1],[BN,D],*layout)
            vd=Descriptor(v,[B*N,HK*D],[HK*D,1],[BN,D],*layout)
            kwargs={} if M==1 else dict(BM=BM,STATIC=STATIC)
            key=(q.device.index,q.dtype,B,M,N,H,HK,D,BM,SLOTS,STATIC,MMA_N,type(scale),float(scale))
            args=(qd,kd,vd,prefix,starts,out,scale)
            kwargs.update(SLOTS=SLOTS,M=M,N=N,H=H,HK=HK,D=D,BN=BN,num_warps=4,num_stages=1)
            (decode_launch if M==1 else main_launch)(key,(H,B,cdiv(M,BM)),args,kwargs,fast)
            return out if D==realD else out[...,:realD].contiguous()
    return forgetting_attention

def entry_for(config=None, fast=True):
    return make_entry(
        "tle", _tle_h100_kernel, _tle_decode_kernel,
        TensorDescriptor, None, config, fast,
    )


forgetting_attention = entry_for()
