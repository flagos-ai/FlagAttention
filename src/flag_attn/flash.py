import math

import torch
import triton
import triton.language as tl

from flag_attn.split_kv import _fwd_combine_kv_splits, _fwd_split_kv_kernel
from flag_attn.split_kv import get_fwd_config as get_fwd_config_kv_split
from flag_attn.split_kv import num_splits_herustic
from flag_attn.total import _total_attention_kernel

from .dropout import philox_cuda_seed_offset

__all__ = ["attention"]


def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x


def rounded_multiple(a, b):
    return (a + b - 1) // b * b


# --------------------------- public API ---------------------------
class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        causal,
        sm_scale,
        dropout_p,
        return_log_normalizer,
        return_total_attention,
        return_seed_offset,
    ):
        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
        assert Dk in {16, 32, 64, 128}

        B, H, M, D = q.shape
        N = k.shape[2]
        Hk, Hv = k.shape[1], v.shape[1]
        assert Hk == Hv, "num of heads in k and v should be equal"
        assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
        num_groups = H // Hk

        P_SEQ = N - M
        larger_m = M > N

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # contiguity
        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q)
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count

        with torch.cuda.device(device):
            # Dropout preparation.
            is_dropout = dropout_p > 0
            if is_dropout:
                offset_increment = B * H * M * N
                seed, offset = philox_cuda_seed_offset(offset_increment)
            else:
                seed, offset = 0, 0

            config_for_split_kv = get_fwd_config_kv_split(B, H, M, N, D, causal)
            S = num_splits_herustic(
                B, H, M, N, config_for_split_kv[0], config_for_split_kv[1], num_sms, 128
            )
            split_kv: bool = S > 1
            # print(f"flag_attn choose {S} splits")

            if not split_kv:
                config = get_fwd_config(B, H, M, N, D, causal)
                BLOCK_M, BLOCK_N, num_stages, num_warps = config

                divisible_m = M % BLOCK_M == 0
                divisible_n = N % BLOCK_N == 0
                # consider using 3d grid to avoid div & rem
                grid = (triton.cdiv(M, BLOCK_M), H, B)
                o = torch.empty_like(q)
                L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
                _fwd_kernel[grid](
                    q,
                    k,
                    v,
                    sm_scale,
                    dropout_p,
                    seed,
                    offset,
                    L,
                    o,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    o.stride(0),
                    o.stride(1),
                    o.stride(2),
                    o.stride(3),
                    B,
                    H,
                    M,
                    N,
                    P_SEQ,
                    num_groups,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=D,
                    IS_CAUSAL=causal,
                    IS_DROPOUT=is_dropout,
                    LARGER_M=larger_m,
                    DIVISIBLE_M=divisible_m,
                    DIVISIBLE_N=divisible_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            else:  # split kv
                assert not is_dropout, "Cannot apply dropout with splitkv."
                BLOCK_M, BLOCK_N, num_stages, num_warps = config_for_split_kv

                divisible_m = M % BLOCK_M == 0
                divisible_n = N % BLOCK_N == 0
                # consider using 3d grid to avoid div & rem
                multiple_l = torch.empty(
                    (B, H, S, M), dtype=torch.float32, device="cuda"
                )
                multiple_o = torch.empty(
                    (B, H, S, M, D), dtype=torch.float16, device="cuda"
                )
                grid = (triton.cdiv(M, BLOCK_M), S, H * B)
                N_SPLIT_SIZE = triton.cdiv(triton.cdiv(N, BLOCK_N), S) * BLOCK_N
                _fwd_split_kv_kernel[grid](
                    q,
                    k,
                    v,
                    sm_scale,
                    multiple_l,
                    multiple_o,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    multiple_o.stride(0),
                    multiple_o.stride(1),
                    multiple_o.stride(2),
                    multiple_o.stride(3),
                    multiple_o.stride(4),
                    B,
                    H,
                    M,
                    N,
                    P_SEQ,
                    N_SPLIT_SIZE,
                    S,
                    num_groups,
                    BLOCK_M=BLOCK_M,
                    BLOCK_DMODEL=D,
                    BLOCK_N=BLOCK_N,
                    IS_CAUSAL=causal,
                    LARGER_M=larger_m,
                    DIVISIBLE_M=divisible_m,
                    DIVISIBLE_N=divisible_n,
                    num_stages=num_stages,
                    num_warps=num_warps,
                )

                L = torch.empty((B, H, M), dtype=torch.float32, device="cuda")
                o = torch.empty_like(q)
                grid = (triton.cdiv(M, BLOCK_M), H, B)
                _fwd_combine_kv_splits[grid](
                    multiple_o,
                    multiple_l,
                    o,
                    L,
                    multiple_o.stride(0),
                    multiple_o.stride(1),
                    multiple_o.stride(2),
                    multiple_o.stride(3),
                    multiple_o.stride(4),
                    o.stride(0),
                    o.stride(1),
                    o.stride(2),
                    o.stride(3),
                    B,
                    H,
                    M,
                    S,
                    BLOCK_M=BLOCK_M,
                    BLOCK_DMODEL=D,
                    DIVISIBLE_M=divisible_m,
                    num_stages=num_stages,
                    num_warps=num_warps,
                )

            # total attention
            if return_total_attention:
                tot_attn = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
                grid = (triton.cdiv(N, BLOCK_N), H, B)
                _total_attention_kernel[grid](
                    q,
                    k,
                    L,
                    tot_attn,
                    sm_scale,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    B,
                    H,
                    M,
                    N,
                    P_SEQ,
                    num_groups,
                    BLOCK_M=BLOCK_M,
                    BLOCK_DMODEL=D,
                    BLOCK_N=BLOCK_N,
                    CAUSAL=causal,
                    DIVISIBLE_M=divisible_m,
                    DIVISIBLE_N=divisible_n,
                    num_stages=num_stages,
                    num_warps=num_warps,
                )

        # autograd context maintenance
        ctx.save_for_backward(q, k, v, o, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.seed = seed
        ctx.offset = offset

        has_extra_return = True in (
            return_log_normalizer,
            return_total_attention,
            return_seed_offset,
        )
        if has_extra_return:
            outs = (
                o,
                L if return_log_normalizer else None,
                tot_attn if return_total_attention else None,
                seed if is_dropout and return_seed_offset else None,
                offset if is_dropout and return_seed_offset else None,
            )
            return outs
        return o

    @staticmethod
    def backward(ctx, do, *ignored):
        q, k, v, o, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        dropout_p = ctx.dropout_p
        is_dropout = ctx.dropout_p > 0
        seed = ctx.seed
        offset = ctx.offset

        B, H, M, D = q.shape
        N = k.shape[2]
        Hk = k.shape[1]
        num_groups = H // Hk
        P_SEQ = N - M
        larger_m = M > N

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q)
        with torch.cuda.device(device):
            config = get_bwd_config(B, H, M, N, D, causal)
            BLOCK_M, BLOCK_N, num_stages, num_warps = config

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0

            delta = torch.empty_like(L)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_preprocess[grid](
                o,
                do,
                delta,
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                delta.stride(0),
                delta.stride(1),
                delta.stride(2),
                M,
                BLOCK_M=BLOCK_M,
                D_HEAD=D,
                DIVISIBLE_M=divisible_m,
            )

            # NOTE that dk & dv always have the same number of heads as q, instead of q.
            dk = torch.empty((B, H, N, D), dtype=k.dtype, device=q.device)
            dv = torch.empty((B, H, N, D), dtype=v.dtype, device=q.device)
            grid = (triton.cdiv(N, BLOCK_N), H, B)
            _bwd_kv_kernel[grid](
                q,
                k,
                v,
                sm_scale,
                do,
                dk,
                dv,
                L,
                delta,
                dropout_p,
                seed,
                offset,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                dk.stride(0),
                dk.stride(1),
                dk.stride(2),
                dk.stride(3),
                dv.stride(0),
                dv.stride(1),
                dv.stride(2),
                dv.stride(3),
                B,
                H,
                M,
                N,
                P_SEQ,
                num_groups,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=D,
                BLOCK_N=BLOCK_N,
                CAUSAL=causal,
                IS_DROPOUT=is_dropout,
                DIVISIBLE_M=divisible_m,
                DIVISIBLE_N=divisible_n,
                num_stages=num_stages,
                num_warps=num_warps,
            )

            dq = torch.zeros_like(q)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_q_kernel[grid](
                q,
                k,
                v,
                sm_scale,
                do,
                dq,
                L,
                delta,
                dropout_p,
                seed,
                offset,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                dq.stride(0),
                dq.stride(1),
                dq.stride(2),
                dq.stride(3),
                B,
                H,
                M,
                N,
                P_SEQ,
                num_groups,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=D,
                BLOCK_N=BLOCK_N,
                CAUSAL=causal,
                IS_DROPOUT=is_dropout,
                LARGER_M=larger_m,
                DIVISIBLE_M=divisible_m,
                DIVISIBLE_N=divisible_n,
                num_stages=num_stages,
                num_warps=num_warps,
            )
            dk = dk.reshape((B, Hk, num_groups, N, D)).sum(2)
            dv = dv.reshape((B, Hk, num_groups, N, D)).sum(2)
        return dq, dk, dv, None, None, None, None, None, None


def attention(
    q,
    k,
    v,
    causal=False,
    sm_scale=None,
    dropout_p=0.0,
    return_log_normalizer=False,
    return_total_attention=False,
    return_seed_offset=False,
):
    """
    An implementation of FlashAttention v2(https://arxiv.org/abs/2307.08691).

    Arguments:
        q(torch.Tensor): The first queries. The shape is (batch_size, num_heads_q, seqlen_q, headdim).
        k(torch.Tensor): The first keys. The shape is (batch_size, num_heads_k, seqlen_k, headdim).
        v(torch.Tensor): The values. The shape is (batch_size, num_heads_k, seqlen_k, headdim).
        causal(bool): Whether causal masking is applied to attention scores before applying softmax.
        sm_scale(float): The scaling of attention scores before applying softmax.
        dropout_p(float): Dropout probability.
        return_log_normalizer(bool): Whether to return the log normalizer of softmax inside attention.
        return_total_attention(bool): Whether to return the sum of attention along q's sequence dimendion.
        return_seed_offset(bool): Whether to return dropout seed and offset

    Returns:
        out(torch.Tensor): The output. The shape is (batch_size, num_heads_q, seqlen_q, headdim).

        If `return_log_normalizer` or `return_total_attention` or `return_seed_offset` is True,
            return the following results in addition.

        log_normalizer(torch.Tensor): The log normalizer. The shape is (batch_size, num_heads_q, seqlen_q).
        total_attention(torch.Tensor): The total attention. The shape is (batch_size, num_heads_q, seqlen_k).
        seed(int): The Philox seed used in dropout.
        offset(int): The starting Philox offset used in dropout.

    Notes:
        `num_heads_q` must be a multiple of `num_heads_k`.
    """
    return FlashAttention.apply(
        q,
        k,
        v,
        causal,
        sm_scale,
        dropout_p,
        return_log_normalizer,
        return_total_attention,
        return_seed_offset,
    )


# --------------------------- Forward ---------------------------
# NOTE: this function can be overwritten at runtime to use your custom config
def get_fwd_config(B, H, M, N, D, causal):
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else:  # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    dropout_p,
    seed,
    offset,
    L,
    O,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    M,
    N,
    P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
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
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M  # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]

    # initialize pointers to value-like data
    q_ptrs = Q + (
        offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    )  # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (
        offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    )  # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    # Dot I trick: to place q in registers, it saves shared memory
    if BLOCK_DMODEL < 128:
        identity = tl.where(
            offs_k[:, None] == offs_k,
            tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
            tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype),
        )
        q = tl.dot(q, identity).to(input_dtype)
    # else:
    #     I = tl.where(offs_m_base[:, None] == offs_m_base,
    #                  tl.full((BLOCK_M, BLOCK_M), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_M, BLOCK_M), 0.0, dtype=input_dtype))
    #     q = tl.dot(I, q).to(input_dtype)

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

    # loop over k, v and update accumulators
    offs_n_init = offs_n_base
    k_ptrs = K + (
        offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn
    )  # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (
        offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk
    )  # (BLOCK_N, BLOCK_DMODEL)
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)

        # -- apply dropout --
        if IS_DROPOUT:
            offs_rng = start_n + offs_rng_base
            pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
            p *= pmask.to(tl.float32)

        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # write back l & o
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        log_sum = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        log_sum = m_i * sm_scale + tl.log(l_i)  # log(normalizer)

    # -- scale o due to dropout
    if IS_DROPOUT:
        scale = 1.0 / (1.0 - dropout_p)
        acc *= scale

    if DIVISIBLE_M:
        tl.store(l_ptrs, log_sum, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        tl.store(l_ptrs, log_sum, mask=mask_m, cache_modifier=".cg")
        tl.store(
            o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg"
        )


# --------------------------- Backward ---------------------------
# NOTE: this function can be overwritten at runtime to use your custom config
def get_bwd_config(B, H, M, N, D, causal):
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            BLOCK_M = 128 if D <= 64 else 64
            BLOCK_N = 64
            num_stages = 2
            num_warps = 4
        else:
            BLOCK_M = 64
            BLOCK_N = 64
            num_stages = 3 if D <= 64 else 2
            num_warps = 4
    elif torch.cuda.get_device_capability() == (
        8,
        6,
    ):  # tune for RTX-3090, device_capability(8, 6)
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dz,
    stride_dh,
    stride_dm,
    M,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh

    # compute (Out * Dout).sum() for vector interpretation
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)

    # load
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok

    if DIVISIBLE_M:
        o = tl.load(o_ptrs).to(tl.float32)
        do = tl.load(do_ptrs).to(tl.float32)
    else:
        mask_m = off_m < M
        o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)

    # compute
    delta = tl.sum(o * do, axis=1)

    # (NOTE) dropout scaling doesn't affect delta's value
    # when dropout is applied, o and do are actually scaled.
    # original_o equals o times reverse scale while original_do is do times scale,
    # and thus delta remains unchanged.

    # write-back
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kv_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DK,
    DV,
    L,
    D,
    dropout_p,
    seed,
    offset,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvh,
    stride_dvn,
    stride_dvk,
    Z,
    H,
    M,
    N,
    P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh

    # offset pointers for batch/head
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh

    # offset pointers for batch/head
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (
        offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk
    )  # (BLOCK_M, BLOCK_DMODEL)
    k_ptrs = K + (
        offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    )  # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (
        offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    )  # (BLOCK_N, BLOCK_DMODEL)
    do_ptrs = DO + (
        offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok
    )  # (BLOCK_M, BLOCK_DMODEL)

    dv_ptrs = DV + (
        offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk
    )  # (BLOCK_N, BLOCK_DMODEL)
    dk_ptrs = DK + (
        offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
    )  # (BLOCK_N, BLOCK_DMODEL)

    # k and v stay in SRAM throughout
    if DIVISIBLE_N:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
    else:
        mask_n = offs_n < N
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])

    # dropout
    if IS_DROPOUT:
        colblock_base = off_z * H * M * N + off_h * M * N + start_n * BLOCK_N
        offs_rng_base = offset + colblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]
        rp = 1.0 / (1.0 - dropout_p)

    # initialize dk amd dv
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # loop over a col
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = (P_SEQ + offs_m[:, None]) >= (
            offs_n[None, :]
        )  # (BLOCK_M, BLOCK_N)

        # load q1, k1, q2, k2, v, do on-chip
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
        else:
            mask_m = offs_m < M
            valid_mask = mask_m[:, None]  # & mask_n
            q = tl.load(q_ptrs, mask=mask_m[:, None])
        # recompute p = softmax(qk * sm_scale, dim=-1)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # s = tl.where(valid_mask, s , float("-inf"))
        # if CAUSAL:
        #     s = tl.where(causal_mask, s, float("-inf"))

        # -- recompute p ---
        if DIVISIBLE_M:
            log_sum = tl.load(L + offs_m)
        else:
            log_sum = tl.load(L + offs_m, mask=mask_m)
        p = tl.math.exp2(s * qk_scale - log_sum[:, None] * log2e)  # (BLOCK_M, BLOCK_N)

        if not DIVISIBLE_M:
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        # compute dv = dot(p, do)
        if DIVISIBLE_M:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=mask_m[:, None])  # (BLOCK_M, BLOCK_DMODEL)

        if IS_DROPOUT:
            # do *= rp
            offs_rng = offs_rng_base + start_m * N
            pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
            p_masked = p * pmask
            p_masked = p_masked.to(input_dtype)

        # -- apply dropout --
        if IS_DROPOUT:
            dv += (
                tl.dot(tl.trans(p_masked), do) * rp
            )  # (BLOCK_N, BLOCK_DMODEL)  # still correct
        else:
            dv += tl.dot(
                tl.trans(p).to(input_dtype), do
            )  # (BLOCK_N, BLOCK_DMODEL)  # still correct

        # compute dp = dot(v, do)
        if DIVISIBLE_M:
            delta = tl.load(D + offs_m)
        else:
            delta = tl.load(D + offs_m, mask=mask_m)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        # -- apply dropout --
        if IS_DROPOUT:
            dp *= rp
            dp *= pmask

        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - delta[:, None])  # (BLOCK_M, BLOCK_N)

        if not DIVISIBLE_M:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds = ds.to(input_dtype)

        # compute dk = dot(ds.T, q) masking
        dk += tl.dot(tl.trans(ds), q)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom

    dk *= sm_scale
    if DIVISIBLE_N:
        tl.store(dk_ptrs, dk.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL,)
    else:
        tl.store(
            dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None]
        )  # (BLOCK_N, BLOCK_DMODEL)
        tl.store(
            dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None]
        )  # (BLOCK_N, BLOCK_DMODEL,)


@triton.jit
def _bwd_q_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    L,
    D,
    dropout_p,
    seed,
    offset,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    Z,
    H,
    M,
    N,
    P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
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
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    # offset pointers for batch/head
    DQ += off_z * stride_dqz + off_h * stride_dqh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (
        offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    )  # (BLOCK_M, BLOCK_DMODEL)
    k_ptrs = K + (
        offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk
    )  # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (
        offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk
    )  # (BLOCK_N, BLOCK_DMODEL)

    dq_ptrs = DQ + (
        offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
    )  # (BLOCK_M, BLOCK_DMODEL)
    do_ptrs = DO + (
        offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    )  # (BLOCK_M, BLOCK_DMODEL)

    # pointer to row-wise quantities in value-like data
    d_ptrs = D + offs_m
    l_ptrs = L + offs_m

    # load q: it will stay in SRAM throughout
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        log_sum = tl.load(l_ptrs)
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        log_sum = tl.load(l_ptrs, mask=mask_m)

    # initialize dq
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # loop over k, v and update accumulator
    # see note "Loop-Bound-For-N"
    if CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    # dropout
    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]
        rp = 1.0 / (1.0 - dropout_p)
        do *= rp.to(do.dtype)

    # loop over a row
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_base

        # load k1, k2, v on chip
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k = tl.load(k_ptrs)
        else:
            mask_n = offs_n < N
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k = tl.load(k_ptrs, mask=mask_n[:, None])

        # recompute p = softmax(qk * sm_scale, dim=-1)
        if not DIVISIBLE_N:
            valid_mask = mask_n  # & mask_m[:, None]
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (
                offs_n[None, :]
            )  # (BLOCK_M, BLOCK_N)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # if CAUSAL:
        #     s = tl.where(causal_mask & valid_mask, s, float("-inf"))
        # else:
        #     s = tl.where(valid_mask, s, float("-inf"))
        p = tl.math.exp2(s * qk_scale - log_sum[:, None] * log2e)  # (BLOCK_M, BLOCK_N)

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))

        if IS_DROPOUT:
            offs_rng = start_n + offs_rng_base
            pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
            dp *= pmask
            # p_dropout = p * pmask.to(tl.float32)

        # no need to mask dp
        # if CAUSAL:
        #     dp = tl.where(causal_mask & valid_mask, dp, 0.0)
        # else:
        #     dp = tl.where(valid_mask, dp, 0.0)

        # compute ds = p * (dp - delta[:, None])
        # move scale out to dq at last
        ds = p * (dp - delta[:, None])  # (BLOCK_M, BLOCK_N)

        # mask ds to ensure no small values
        if not DIVISIBLE_N:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        dq += tl.dot(ds.to(input_dtype), k)

        # increment pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    dq *= sm_scale
    if DIVISIBLE_M:
        tl.store(dq_ptrs, dq.to(input_dtype))
    else:
        tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])
