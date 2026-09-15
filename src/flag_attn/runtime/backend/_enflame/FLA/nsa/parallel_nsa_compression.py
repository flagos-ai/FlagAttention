# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Enflame S60 backend for parallel NSA compression."""

import importlib
import torch
import triton
import triton.language as tl
from ..index import (
    prepare_chunk_offsets_enflame as prepare_chunk_offsets,
)
from flag_attn.parallel_nsa.triton_ops_helper import (
    autotune_cache_kwargs,
    exp,
    log,
)
from flag_attn.parallel_nsa.utils import (
    check_shared_mem,
)


# Compression attention forward

MAX_GRID_X = 65535


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BC": bc}, num_warps=num_warps)
        for bc in [64, 32, 16]
        for num_warps in [1, 2, 4]
    ],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["token_offset"])
def _parallel_nsa_compression_fwd_kernel_enflame(
    q,
    k,
    v,
    o,
    lse,
    scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    token_offset,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0) + token_offset
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS

    p_o = tl.make_block_ptr(
        o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)
    )
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)
    # max scores for the current block
    b_m = tl.full([G], float("-inf"), dtype=tl.float32)
    # lse = log(acc) + m
    b_acc = tl.zeros([G], dtype=tl.float32)

    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)

        p_k = tl.make_block_ptr(
            k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (boc * H + i_h) * V,
            (TC, V),
            (H * V, 1),
            (i_c, i_v * BV),
            (BC, BV),
            (1, 0),
        )
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BC, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((o_c < NC)[None, :], b_s, float("-inf"))

        # [G]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp(b_mp - b_m)
        # [G, BC]
        b_p = exp(b_s - b_m[:, None])
        # [G]
        b_acc = tl.math.fma(b_acc, b_r, tl.sum(b_p, 1))

        # [G, BV]
        b_o = tl.math.fma(b_o, b_r[:, None], tl.dot(b_p.to(b_q.dtype), b_v))

        b_mp = b_m
    if NC == 0:
        b_lse = tl.zeros([G], dtype=tl.float32)
    else:
        b_o = b_o / b_acc[:, None]
        b_lse = b_m + log(b_acc)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(
            lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G),
            b_lse.to(lse.dtype.element_ty),
        )


# ===========================================================================
# Backward kernel: dQ
# ===========================================================================


def parallel_nsa_compression_fwd_enflame(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
):
    B, T, HQ, K, V = *q.shape, v.shape[-1]
    H = k.shape[2]
    G = HQ // H
    BS = block_size
    if check_shared_mem("hopper", q.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    chunk_offsets = (
        prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None
    )

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    for token_offset in range(0, T, MAX_GRID_X):
        grid = (
            min(
                MAX_GRID_X,
                T - token_offset,
            ),
            NV,
            B * H,
        )
        _parallel_nsa_compression_fwd_kernel_enflame[grid](
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            chunk_offsets=chunk_offsets,
            token_offset=token_offset,
            T=T,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            BS=BS,
            BK=BK,
            BV=BV,
        )
    return o, lse


# Compression attention backward

MAX_GRID_X = 65535
MAX_GRID_Y = 255
MAX_GRID_Z = 255


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(
    do_not_specialize=[
        "T",
        "token_offset",
        "value_offset",
        "bh_offset",
    ]
)
def _parallel_nsa_compression_bwd_kernel_dq_enflame(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    token_offset,
    value_offset,
    bh_offset,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0) + token_offset
    i_v = tl.program_id(1) + value_offset
    i_bh = tl.program_id(2) + bh_offset
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ * K

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)
        p_k = tl.make_block_ptr(
            k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (boc * H + i_h) * V,
            (V, TC),
            (1, H * V),
            (i_v * BV, i_c),
            (BV, BC),
            (0, 1),
        )
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BC]
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_p = exp(b_s - b_lse[:, None])
        b_p = tl.where((o_c < NC)[None, :], b_p, 0)

        # [G, BV] @ [BV, BC] -> [G, BC]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [G, BC] @ [BC, BK] -> [G, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=1)],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(
    do_not_specialize=[
        "T",
        "TC",
        "value_offset",
        "chunk_offset",
        "bh_offset",
    ]
)
def _parallel_nsa_compression_bwd_kernel_dkv_enflame(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    value_offset,
    chunk_offset,
    bh_offset,
    T,
    TC,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v = tl.program_id(0) + value_offset
    i_c = tl.program_id(1) + chunk_offset
    i_bh = tl.program_id(2) + bh_offset
    i_b, i_h = i_bh // H, i_bh % H

    all = B * TC

    if IS_VARLEN:
        i_n, i_c = tl.load(chunk_indices + i_c * 2).to(tl.int32), tl.load(
            chunk_indices + i_c * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        # the number of compression representations in total
        TC = tl.cdiv(T, BS)
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_k = tl.make_block_ptr(
        k + (boc * H + i_h) * K, (TC, K), (H * K, 1), (i_c * BC, 0), (BC, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + (boc * H + i_h) * V,
        (TC, V),
        (H * V, 1),
        (i_c * BC, i_v * BV),
        (BC, BV),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * all * H + boc * H + i_h) * K,
        (TC, K),
        (H * K, 1),
        (i_c * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (boc * H + i_h) * V,
        (TC, V),
        (H * V, 1),
        (i_c * BC, i_v * BV),
        (BC, BV),
        (1, 0),
    )

    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    # [BC, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)

    for i in range(i_c * BC * BS, T):
        o_c = i_c * BC + tl.arange(0, BC)

        p_q = tl.make_block_ptr(
            q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
        )
        # [G, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        p_do = tl.make_block_ptr(
            do + (bos + i) * HQ * V,
            (HQ, V),
            (V, 1),
            (i_h * G, i_v * BV),
            (G, BV),
            (1, 0),
        )
        p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [G]
        b_lse = tl.load(p_lse)
        b_delta = tl.load(p_delta)
        # [BC, G]
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_p = exp(b_s - b_lse[None, :])
        b_p = tl.where((i >= max(0, (o_c + 1) * BS - 1))[:, None], b_p, 0)
        # [BC, G] @ [G, BV] -> [BC, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BC, BV] @ [BV, G] -> [BC, G]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BC, G]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BC, G] @ [G, BK] -> [BC, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class _ChunkedKernel3D:
    def __init__(
        self,
        kernel,
        offset_names,
    ):
        self.kernel = kernel
        self.offset_names = offset_names

    def __getitem__(self, grid):
        grid_x, grid_y, grid_z = map(
            int,
            grid,
        )

        def launch(*args, **kwargs):
            for offset_x in range(
                0,
                grid_x,
                MAX_GRID_X,
            ):
                count_x = min(
                    MAX_GRID_X,
                    grid_x - offset_x,
                )

                for offset_y in range(
                    0,
                    grid_y,
                    MAX_GRID_Y,
                ):
                    count_y = min(
                        MAX_GRID_Y,
                        grid_y - offset_y,
                    )

                    for offset_z in range(
                        0,
                        grid_z,
                        MAX_GRID_Z,
                    ):
                        count_z = min(
                            MAX_GRID_Z,
                            grid_z - offset_z,
                        )

                        call_kwargs = dict(kwargs)
                        call_kwargs[
                            self.offset_names[0]
                        ] = offset_x
                        call_kwargs[
                            self.offset_names[1]
                        ] = offset_y
                        call_kwargs[
                            self.offset_names[2]
                        ] = offset_z

                        self.kernel[
                            (
                                count_x,
                                count_y,
                                count_z,
                            )
                        ](
                            *args,
                            **call_kwargs,
                        )

        return launch


parallel_nsa_compression_bwd_kernel_dq_enflame = (
    _ChunkedKernel3D(
        _parallel_nsa_compression_bwd_kernel_dq_enflame,
        (
            "token_offset",
            "value_offset",
            "bh_offset",
        ),
    )
)

parallel_nsa_compression_bwd_kernel_dkv_enflame = (
    _ChunkedKernel3D(
        _parallel_nsa_compression_bwd_kernel_dkv_enflame,
        (
            "value_offset",
            "chunk_offset",
            "bh_offset",
        ),
    )
)


# S60 compression dispatch

def _find_autotuner(kernel):
    current = kernel
    chain = []
    seen = set()

    while current is not None:
        if id(current) in seen:
            raise RuntimeError(
                "compression decorator chain cycle"
            )

        seen.add(id(current))
        chain.append(type(current).__name__)

        configs = getattr(current, "configs", None)

        if configs is not None:
            return current, chain

        current = getattr(current, "fn", None)

    raise RuntimeError(
        "compression Autotuner not found: "
        f"{chain}"
    )


def _to_int32(value):
    if (
        isinstance(value, torch.Tensor)
        and value.dtype != torch.int32
    ):
        return value.to(dtype=torch.int32)

    return value


def install_enflame_compression_compat():
    """Install S60 compression compatibility on the active path."""
    from .bwd_preprocess import (
        parallel_attn_bwd_preprocess_enflame,
    )
    from ..index import (
        prepare_chunk_indices_enflame,
        prepare_chunk_offsets_enflame,
        prepare_lens_enflame,
        prepare_token_indices_enflame,
    )

    module = importlib.import_module(
        "flag_attn.parallel_nsa."
        "parallel_nsa_compression"
    )










    module.parallel_nsa_compression_fwd = (
        parallel_nsa_compression_fwd_enflame
    )
    module.parallel_nsa_compression_bwd_kernel_dq = (
        parallel_nsa_compression_bwd_kernel_dq_enflame
    )
    module.parallel_nsa_compression_bwd_kernel_dkv = (
        parallel_nsa_compression_bwd_kernel_dkv_enflame
    )

    module.parallel_attn_bwd_preprocess = (
        parallel_attn_bwd_preprocess_enflame
    )

    index_bindings = {
        "prepare_lens": prepare_lens_enflame,
        "prepare_token_indices": (
            prepare_token_indices_enflame
        ),
        "prepare_chunk_offsets": (
            prepare_chunk_offsets_enflame
        ),
        "prepare_chunk_indices": (
            prepare_chunk_indices_enflame
        ),
    }

    for name, function in index_bindings.items():
        if hasattr(module, name):
            setattr(module, name, function)


    return module


def parallel_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    module = install_enflame_compression_compat()

    cu_seqlens = _to_int32(cu_seqlens)

    return module.parallel_nsa_compression(
        q=q,
        k=k,
        v=v,
        block_size=block_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )


__all__ = [
    "parallel_nsa_compression",
    "install_enflame_compression_compat",
]
