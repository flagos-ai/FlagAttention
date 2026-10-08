"""
Implementation of Forgetting Attention.

Our code is adapted from https://github.com/FlagOpen/FlagAttention/blob/ee91638dec6da8c00c4113d179f469e0ffcd5852/src/flag_attn/flash.py. The code is modified to implement Forgetting Attention.

The original license info from FlagAttention:

Copyright 2023 BAAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Unoptimized Triton ACP reference, not a dense PyTorch implementation.
# Forward code is copied from zhixuan-lin/forgetting-transformer:
# commit 883f260c636e87971339749b0d8310004794a5cc
# src/forgetting_transformer/ops/forgetting_attention.py
# Original complete source SHA256:
# 6a71b9ebef4dbfaf489ca32fb4ad32ed51a3dc474b6d681c21ddf026eef1b5d2
# This forward-only extraction retains the original forward operations and
# launch configuration. Backward and standalone upstream tests are excluded.
import math
from collections import defaultdict
from typing import Literal, Optional, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

OFFICIAL_COMMIT = "883f260c636e87971339749b0d8310004794a5cc"
__all__ = ["forgetting_attention"]

def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

class ForgettingAttention(torch.autograd.Function):
    events = defaultdict(lambda: {
        "fwd_start_event": torch.cuda.Event(enable_timing=True),
        "fwd_end_event": torch.cuda.Event(enable_timing=True),
        "bwd_start_event": torch.cuda.Event(enable_timing=True),
        "bwd_end_event": torch.cuda.Event(enable_timing=True),

        "fwd_find_index_start_event": torch.cuda.Event(enable_timing=True),
        "fwd_find_index_end_event": torch.cuda.Event(enable_timing=True),
        "bwd_find_index_kv_start_event": torch.cuda.Event(enable_timing=True),
        "bwd_find_index_kv_end_event": torch.cuda.Event(enable_timing=True),
        "bwd_find_index_q_start_event": torch.cuda.Event(enable_timing=True),
        "bwd_find_index_q_end_event": torch.cuda.Event(enable_timing=True),
    })
    info = defaultdict(lambda: {
        "fwd_time": 0.0,
        "fwd_count": 0,
        "bwd_time": 0.0,
        "bwd_count": 0,

        "fwd_find_index_time": 0.0,
        "fwd_find_index_count": 0,
        "bwd_find_index_kv_time": 0.0,
        "bwd_find_index_kv_count": 0,
        "bwd_find_index_q_time": 0.0,
        "bwd_find_index_q_count": 0,
    })

    @staticmethod
    def forward(ctx, q, k, v, log_fgate, seq_start, causal, sm_scale, adaptive_threshold, return_log_normalizer, return_start_index, record_time_key, record_attention_time, record_find_index_time):
        if record_attention_time:
            ForgettingAttention.events[record_time_key]["fwd_start_event"].record()


        assert causal, "Only causal attention is supported"
        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
        assert Dk in {16, 32, 64, 128}, "We only support head dims in {16, 32, 64, 128}"


        B, H, M, D = q.shape
        N = k.shape[2]
        assert log_fgate.shape == (B, H, N)
        assert log_fgate.dtype == torch.float32, "log_fgate must be in torch.float32. It is best to do this cast before logsigmoid, e.g., torch.nn.functional.logsigmoid(fgate_logit.float())."
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        if adaptive_threshold is not None:
            if isinstance(adaptive_threshold, str):
                assert adaptive_threshold == "auto", f'adaptive_threshold must be either the string "auto", a float, or a Tensor, but got {adaptive_threshold}.'
                # Otherwise we could calculate the max L2 norms manually 
                max_q_norm = torch.linalg.vector_norm(q, dim=-1).max(dim=-1).values
                max_k_norm = torch.linalg.vector_norm(k, dim=-1).max(dim=-1).values
                assert max_q_norm.size() == max_k_norm.size() == (B, H)

                logit_upper_bound = max_q_norm * max_k_norm * sm_scale
                tolerance = -10
                # Note we should use N instead of M here
                adaptive_threshold = -(2 * logit_upper_bound + math.log(N)) + tolerance
                
            adaptive_threshold = torch.as_tensor(adaptive_threshold, dtype=torch.float, device=q.device)
            try:
                adaptive_threshold = torch.broadcast_to(adaptive_threshold, (B, H))
            except RuntimeError:
                raise RuntimeError(f'adaptive_threshold must be either the string "auto" or broadcastable to (batch_size, num_heads) = ({B}, {H}), but got {adaptive_threshold.size()}.')
            assert adaptive_threshold.size() == (B, H)

        if seq_start is not None:
            has_seq_start = True
            assert seq_start.shape == (B,)
        else:
            has_seq_start = False
            seq_start = torch.zeros((B,), device=q.device, dtype=torch.long)
        log_fgate = log_fgate.float()
        if has_seq_start:
            log_fgate = log_fgate.clone()
            # We absolutely don't want masked value to affect result. If we
            # don't do this then it could via affecting numerical precision of
            # cumsum
            mask_index = (torch.arange(N, device=q.device)[None, None, :] < seq_start[:, None, None])
            mask_index = torch.broadcast_to(mask_index, log_fgate.size())
            log_fgate[mask_index] = 0.0

        log_lambda = torch.cumsum(log_fgate, dim=-1, dtype=log_fgate.dtype).float()

        Hk, Hv = k.shape[1], v.shape[1]
        assert Hk == Hv, "num of heads in k and v should be equal"
        assert H == Hk, "groupped query attention has not been tested. You can uncomment this if you know what you are doing."
        assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
        num_groups = H // Hk

        P_SEQ = N - M
        larger_m = M > N
        assert (not larger_m), "The key/value tensors must be longer than the query tensor"


        # contiguity
        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q)

        with torch.cuda.device(device):

            if M > 1:
                config = get_fwd_config(B, H, M, N, D, causal)
                BLOCK_M, BLOCK_N, num_stages, num_warps = config
            else:
                BLOCK_N, num_stages, num_warps = min(128, max(16, triton.next_power_of_2(N))), 1, 4
                BLOCK_M = 1

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0

            
            start_index = torch.empty((B, H, triton.cdiv(M, BLOCK_M)), dtype=torch.long, device=q.device)
            if adaptive_threshold is not None:
                grid = (H, B)
                if record_find_index_time:
                    ForgettingAttention.events[record_time_key]["fwd_find_index_start_event"].record()
                _find_start_index_kernel[grid](
                    log_lambda,
                    start_index,
                    adaptive_threshold,
                    log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                    start_index.stride(0), start_index.stride(1), start_index.stride(2),
                    adaptive_threshold.stride(0), adaptive_threshold.stride(1),
                    B, H, M, N, P_SEQ,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                    num_warps=1
                )
                if record_find_index_time:
                    ForgettingAttention.events[record_time_key]["fwd_find_index_end_event"].record()
                    torch.cuda.synchronize()
                    elapsed = ForgettingAttention.events[record_time_key]["fwd_find_index_start_event"].elapsed_time(ForgettingAttention.events[record_time_key]["fwd_find_index_end_event"])
                    ForgettingAttention.info[record_time_key]["fwd_find_index_time"] += elapsed
                    ForgettingAttention.info[record_time_key]["fwd_find_index_count"] += 1

            # Actual forward
            # consider using 3d grid to avoid div & rem
            # grid = (triton.cdiv(M, BLOCK_M), H, B)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), H, B)
            o = torch.empty_like(q)
            L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
            _fwd_kernel[grid](
                q, k, v, log_lambda, seq_start, start_index, sm_scale,
                L, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                start_index.stride(0), start_index.stride(1), start_index.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                B, H, M, N, P_SEQ, num_groups,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
                IS_CAUSAL=causal, LARGER_M=larger_m, HAS_SEQ_START=has_seq_start,
                IS_ADAPTIVE=adaptive_threshold is not None,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_warps=num_warps, num_stages=num_stages,
            )

        # autograd context maintenance
        ctx.save_for_backward(q, k, v, o, L, log_lambda, seq_start, adaptive_threshold)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.has_seq_start = has_seq_start
        ctx.record_time_key = record_time_key
        ctx.record_attention_time = record_attention_time
        ctx.record_find_index_time = record_find_index_time

        has_extra_return = return_log_normalizer or return_start_index

        if record_attention_time:
            ForgettingAttention.events[record_time_key]["fwd_end_event"].record()
            torch.cuda.synchronize()
            elapsed = ForgettingAttention.events[record_time_key]["fwd_start_event"].elapsed_time(ForgettingAttention.events[record_time_key]["fwd_end_event"])
            ForgettingAttention.info[record_time_key]["fwd_time"] += elapsed
            ForgettingAttention.info[record_time_key]["fwd_count"] += 1
        if has_extra_return:
            outs = (
                o,
                L if return_log_normalizer else None,
                start_index if return_start_index else None
            )
            return outs
        return o

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError("This bundled ACP reference is forward-only")

def forgetting_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_fgate: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    adaptive_threshold: Optional[Union[Literal["auto"], float, torch.Tensor]] = None,
):
    """
    A FlashAttention-based implementation of Forgetting Attention. 

    Note:
    - q, k, v should be in bfloat16/float16 and log_fgate should be in float32.
    - We only support seqlen_q <= seqlen_k
    - We only support causal attention
    - Head dimension must be in one of {16, 32, 64, 128}

    Arguments:
        - q: (batch_size, seqlen_q, num_heads, head_dim) unless head_first=True.
        - k: (batch_size, seqlen_k, num_heads, head_dim) unless head_first=True.
        - v: (batch_size, seqlen_k, num_heads, head_dim) unless head_first=True.
        - log_fgate: (batch_size, seqlen_k, num_heads) unless head_first=True. 
              This should be the **log** of the forget gates. This is typically the 
              output of torch.nn.functional.logsigmoid.
        - head_first: if True, the order the num_heads and seqlen_* axis of the all 
              FloatTensor inputs and outputs should be (num_heads, seq_len_*) instead of
              (seq_len_*, num_heads)
        - seq_start: If not None, should be LongTensor with shape (batch_size,) 
              and range in [0, seq_len_k). For each batch index batch_id, no attention 
              will be allocated to tokens before the token index seq_start[batch_id]. 
              This is useful for left-padded inputs.
        - sm_scale: The scaling of attention scores before applying softmax. If
              None, it defaults to (1.0 / math.sqrt(head_dim))
        - adaptive_threshold: The threshold for adaptive computation pruning. This
              should be either the string "auto", a float, or a Tensor that is 
              broadcastable to (batch_size, num_heads). If "auto", the threshold would 
              be computed automatically based on the L2 norms of queries and keys.

    Returns:
        out (torch.Tensor): (batch_size, seqlen_q, num_heads, head_dim) unless head_first=True.
    """
    for name, entry in dict(q=q, k=k, v=v).items():
        assert entry.dtype in [torch.float16, torch.bfloat16], f"Only torch.float16 or torch.bfloat16 are supported for q/k/v, but got {entry.dtype} for {name}."
    if not head_first:
        q, k, v = [rearrange(item, "b t h d -> b h t d") for item in (q, k, v)]
        log_fgate = rearrange(log_fgate, "b t h -> b h t")
    out = ForgettingAttention.apply(q, k, v, log_fgate, seq_start, True, sm_scale, adaptive_threshold, False, False, None, False, False)
    if not head_first:
        out = rearrange(out, "b h t d -> b t h d")
    return out

def get_fwd_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 4, 4
    elif torch.cuda.get_device_capability() == (9, 0):
        # H100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 2, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else: # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9):
        # L40S
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        raise ValueError(f"Unsupported device capability {torch.cuda.get_device_capability()}. Please open an issue.")
        # BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

@triton.jit
def _fwd_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, START_INDEX, sm_scale,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_start_index_z, stride_start_index_h, stride_start_index_mb,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr, HAS_SEQ_START: tl.constexpr,
    IS_ADAPTIVE: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    loge2: tl.constexpr = 0.6931471805599453
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)


    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m, cache_modifier=".cg")

    #Dot I trick: to place q in registers, it saves shared memory
    # if BLOCK_DMODEL < 128:
    #     I = tl.where(offs_k[:, None] == offs_k,
    #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
    #     q = tl.dot(q, I, input_precision="ieee").to(input_dtype)
    # else:
    #     I = tl.where(offs_m_base[:, None] == offs_m_base,
    #                  tl.full((BLOCK_M, BLOCK_M), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_M, BLOCK_M), 0.0, dtype=input_dtype))
    #     q = tl.dot(I, q, input_precision="ieee").to(input_dtype)

    # NOTE: Loop-Bound-For-N
    # The indices in m-dimension that this block may access is in `[start_m * BLOCK_M, (start_m + 1) * BLOCK_M)`.
    # According to the rule of causal masking, then max index in n-dimension that this block may access
    # is `P_SEQ + (start_m + 1) * BLOCK_M`.
    # However, the upper bound of index in n-dimension should never exceed the sequence length of k/v(`P_SEQ + N_CTX`).
    # `P_SEQ + (start_m + 1) * BLOCK_M` may be larger than `N`.
    # At this case, there would be illegal memory access when loading k & v tiles
    # if mask_n is not applied for loading(only when `DIVISIBLE_N`` is true).
    # See also https://github.com/FlagOpen/FlagAttention/pull/8
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    offs_n_init = offs_n_base

    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        lo = tl.minimum(seq_start, hi)
    else:
        lo = 0
        seq_start = 0

    if IS_ADAPTIVE:
        # No need to multiple start_m by BLOCK_M here
        START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h + start_m * stride_start_index_mb
        start_index = tl.load(START_INDEX)
        lo = tl.maximum(start_index, lo)
    lo = (lo // BLOCK_N) * BLOCK_N
    offs_n_init += lo

    # loop over k, v and update accumulators
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    log_lambda_in_ptrs = LOG_LAMBDA + (offs_n_init * stride_log_lambda_n) # (BLOCK_N, BLOCK_DMODEL)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
            log_lambda_in = tl.load(log_lambda_in_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n, cache_modifier=".cg")

        # -- compute qk ---
        # s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if BLOCK_M > 1:
            s = tl.dot(q, k, input_precision="ieee") * qk_scale
        else:
            # (1, D), (D, T)
            s = tl.sum((q.T * k).to(tl.float32), axis=0, keep_dims=True) * qk_scale
        decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
        s += decay_bias * log2e

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))
        if HAS_SEQ_START:
            s = tl.where(offs_n[None, :] >= seq_start, s, float("-inf"))


        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new))
        p = tl.math.exp2(s - m_i_new[:, None])

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)


        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        if BLOCK_M > 1:
            acc += tl.dot(p.to(input_dtype), v, input_precision="ieee")
        else:
            acc += tl.sum(p.T * v, axis=0, keep_dims=True)

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n

    # write back l & o
    if IS_CAUSAL and (LARGER_M or HAS_SEQ_START):
        is_empty_line = (offs_m + P_SEQ) < seq_start
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * loge2 + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * loge2 + tl.log(l_i) # log(normalizer)


    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")

@triton.jit
def _find_start_index_kernel(
    LOG_LAMBDA, 
    START_INDEX,
    THRESHOLD,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_start_index_z, stride_start_index_h, stride_start_index_mb,
    stride_threshold_z, stride_threshold_h,
    Z, H, M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    # -- grid id --
    off_h = tl.program_id(0)
    off_z = tl.program_id(1)

    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h
    THRESHOLD += off_z * stride_threshold_z + off_h * stride_threshold_h
    start_index = 0

    log_lambda_out_ptr = LOG_LAMBDA + P_SEQ * stride_log_lambda_n
    start_index_ptr = START_INDEX



    threshold = tl.load(THRESHOLD)
    for start_m in range(0, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)

        log_lambda_out = tl.load(log_lambda_out_ptr)

        offset_n = start_index + BLOCK_N - 1
        if not DIVISIBLE_N:
            offset_n = tl.minimum(N - 1, offset_n)
        log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
        decay = log_lambda_out - log_lambda_in

        while decay < threshold:
            start_index += BLOCK_N

            offset_n = start_index + BLOCK_N - 1
            if not DIVISIBLE_N:
                offset_n = tl.minimum(N - 1, offset_n)
            log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
            decay = log_lambda_out - log_lambda_in


        tl.store(start_index_ptr, start_index.to(START_INDEX.dtype.element_ty))
        start_index_ptr += stride_start_index_mb
        log_lambda_out_ptr += stride_log_lambda_n * BLOCK_M
