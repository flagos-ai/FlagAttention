# Copyright 2026 FlagOS Contributors

# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
import yaml

from flag_attn.FLA.compat import libentry, libtuner
from flag_attn.FLA.index import prepare_chunk_indices, prepare_chunk_offsets
from flag_attn.FLA.utils import exp
from flag_attn.FLA.utils import use_cuda_graph

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "tune_configs.yaml"


@lru_cache(maxsize=1)
def _load_tuned_config() -> dict[str, list[dict[str, Any]]]:
    if not _CONFIG_PATH.is_file():
        return {}
    data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Autotune config must be a mapping: {_CONFIG_PATH}")
    return data


def _autotune_configs(kernel_name: str, defaults: list[triton.Config]) -> list[triton.Config]:
    """Use tune_configs.yaml unless FLAG_ATTN_THEAD_GDR_TUNED_CONFIGS=0."""
    if os.environ.get("FLAG_ATTN_THEAD_GDR_TUNED_CONFIGS", "1") == "0":
        return defaults
    entries = _load_tuned_config().get(kernel_name)
    if not entries:
        return defaults
    configs: list[triton.Config] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{kernel_name}[{index}] must be a mapping")
        meta = entry.get("META", {})
        if not isinstance(meta, dict):
            raise ValueError(f"{kernel_name}[{index}].META must be a mapping")
        kwargs = {
            name: entry[name]
            for name in ("num_warps", "num_stages", "num_ctas", "maxnreg")
            if name in entry
        }
        configs.append(triton.Config(meta, **kwargs))
    return configs


@libentry()
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=_autotune_configs(
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        [
            triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [2, 4]
            for num_stages in [2, 3, 4]
            for BV in [32, 64]
        ],
    ),
    cache_results=True,
    key=[
        "H",
        "K",
        "V",
        "BT",
        "IS_VARLEN",
        "USE_INITIAL_STATE",
        "STORE_FINAL_STATE",
    ],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t * stride_h,
                (K, V),
                (V, 1),
                (128, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * stride_h,
                (K, V),
                (V, 1),
                (192, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            ).to(tl.float32)
            b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                ).to(tl.float32)
                b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                ).to(tl.float32)
                b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                ).to(tl.float32)
                b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state

@libentry()
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fused_wu_h_kernel(
    k, v, beta, g, A, h, v_new, ht, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr,
    V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_hg = i_h // (H // Hg)
    k += (i_b * T * Hg + i_hg).to(tl.int64) * K
    v += (i_b * T * H + i_h).to(tl.int64) * V
    v_new += (i_b * T * H + i_h).to(tl.int64) * V
    beta += i_b * T * H + i_h
    g += i_b * T * H + i_h
    A += (i_b * T * H + i_h).to(tl.int64) * BT
    h += (i_b * tl.cdiv(T, BT) * H + i_h).to(tl.int64) * K * V
    ht += i_bh.to(tl.int64) * K * V
    rows = tl.arange(0, BT)
    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    if TRANSPOSED:
        # Keep each operand-B tile at K64 and reuse the FP32 accumulator.
        state1 = tl.zeros((BV, 64), tl.float32)
        state2 = tl.zeros((BV, 64), tl.float32)
        for chunk in range(tl.cdiv(T, BT)):
            p_h1 = tl.make_block_ptr(h + chunk * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            p_h2 = tl.make_block_ptr(h + chunk * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h1, tl.trans(state1).to(p_h1.dtype.element_ty))
            tl.store(p_h2, tl.trans(state2).to(p_h2.dtype.element_ty))
            t = chunk * BT + rows
            p_k1 = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (chunk * BT, 0), (BT, 64), (1, 0))
            p_k2 = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (chunk * BT, 64), (BT, 64), (1, 0))
            keys1 = tl.load(p_k1)
            prediction = tl.dot(state1.to(keys1.dtype), tl.trans(keys1), allow_tf32=False)
            keys2 = tl.load(p_k2)
            prediction = tl.dot(state2.to(keys2.dtype), tl.trans(keys2), acc=prediction, allow_tf32=False)
            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (chunk * BT, i_v * BV), (BT, BV), (1, 0))
            values = tl.load(p_v)
            gate = tl.load(g + t * H).to(tl.float32)
            strength = tl.load(beta + t * H).to(tl.float32)
            residual = (tl.trans(values).to(tl.float32) - tl.exp(gate)[None, :] * prediction) * strength[None, :]
            p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (chunk * BT, 0), (BT, BT), (1, 0))
            inverse = tl.load(p_A)
            updated = tl.dot(residual.to(values.dtype), tl.trans(inverse), allow_tf32=False)
            p_new = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (chunk * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_new, tl.trans(updated).to(p_new.dtype.element_ty))
            last_gate = tl.load(g + (chunk * BT + BT - 1) * H).to(tl.float32)
            decay = tl.exp(last_gate)
            weighted = (updated * tl.exp(last_gate - gate)[None, :]).to(values.dtype)
            # Reload after the public intermediate store; do not retain both B fragments.
            update_keys1 = tl.load(p_k1)
            state1 = tl.dot(weighted, update_keys1, acc=state1 * decay, allow_tf32=False)
            update_keys2 = tl.load(p_k2)
            state2 = tl.dot(weighted, update_keys2, acc=state2 * decay, allow_tf32=False)
        cols = tl.arange(0, 64)
        tl.store(ht + cols[None, :] * V + offs_v[:, None], state1)
        tl.store(ht + (cols[None, :] + 64) * V + offs_v[:, None], state2)
    else:
        state = tl.zeros((K, BV), tl.float32)
        for chunk in range(tl.cdiv(T, BT)):
            p_h = tl.make_block_ptr(h + chunk * H * K * V, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0))
            tl.store(p_h, state.to(p_h.dtype.element_ty))
            t = chunk * BT + rows
            p_k = tl.make_block_ptr(k, (T, K), (Hg * K, 1), (chunk * BT, 0), (BT, K), (1, 0))
            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (chunk * BT, i_v * BV), (BT, BV), (1, 0))
            p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (chunk * BT, 0), (BT, BT), (1, 0))
            keys = tl.load(p_k)
            values = tl.load(p_v)
            gate = tl.load(g + t * H).to(tl.float32)
            strength = tl.load(beta + t * H).to(tl.float32)
            prediction = tl.dot(keys, state.to(keys.dtype), allow_tf32=False)
            residual = (values.to(tl.float32) - tl.exp(gate)[:, None] * prediction) * strength[:, None]
            inverse = tl.load(p_A)
            updated = tl.dot(inverse, residual.to(values.dtype), allow_tf32=False)
            p_new = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (chunk * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_new, updated.to(p_new.dtype.element_ty))
            last_gate = tl.load(g + (chunk * BT + BT - 1) * H).to(tl.float32)
            weighted = updated * tl.exp(last_gate - gate)[:, None]
            state = state * tl.exp(last_gate) + tl.dot(tl.trans(keys), weighted.to(keys.dtype), allow_tf32=False)
        tl.store(ht + offs_k[:, None] * V + offs_v[None, :], state)

def chunk_gated_delta_rule_fused_wu_h(
    k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
    g: torch.Tensor, A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, Hg, K = k.shape
    H, V = v.shape[-2:]
    h = k.new_empty(B, triton.cdiv(T, 64), H, K, V)
    v_new = torch.empty_like(v)
    final_state = k.new_empty(B, H, K, V, dtype=torch.float32)
    # Four warps each own 16 value rows; retain more tasks for small grids.
    transposed = B * H >= 32
    block_v = 64 if transposed else 32
    chunk_gated_delta_rule_fused_wu_h_kernel[(triton.cdiv(V, block_v), B * H)](
        k, v, beta, g, A, h, v_new, final_state, T,
        H=H, Hg=Hg, K=K, V=V, BT=64, BV=block_v, TRANSPOSED=transposed,
        num_warps=4, num_stages=2,
    )
    return h, v_new, final_state
