# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Enflame S60 attention backend."""

from __future__ import annotations

import importlib

_OPERATOR_EXPORTS = {
    "SPARSE_BLOCK_SIZE": (".msa.index_topk", "SPARSE_BLOCK_SIZE"),
    "chunk_gdn2": (".FLA.gdn2", "chunk_gdn2"),
    "chunk_gdn2_native": (".FLA.gdn2.native.chunk_fwd", "chunk_gdn2_fwd"),
    "chunk_gla": (".FLA.gla", "chunk_gla"),
    "chunk_kda": (".FLA.kda", "chunk_kda"),
    "chunk_kda_fwd_infer": (".FLA.kda.chunk_kda", "chunk_kda_fwd_infer"),
    "fused_attnres": (".FLA.attnres", "fused_attnres"),
    "install_msa_prefill": (".msa", "install_msa_prefill"),
    "minimax_m3_index_decode": (".msa", "minimax_m3_index_decode"),
    "minimax_m3_index_score": (".msa", "minimax_m3_index_score"),
    "minimax_m3_index_score_topk": (".msa", "minimax_m3_index_score_topk"),
    "minimax_m3_index_topk": (".msa", "minimax_m3_index_topk"),
    "minimax_m3_sparse_attn": (".msa", "minimax_m3_sparse_attn"),
    "minimax_m3_sparse_attn_decode": (".msa", "minimax_m3_sparse_attn_decode"),
    "parallel_parallax": (".FLA.parallax", "parallel_parallax"),
    "parallax_decode": (".FLA.parallax", "parallax_decode"),
    "parallax_attn_with_kvcache": (".FLA.parallax", "parallax_attn_with_kvcache"),
    "parallel_nsa": (".FLA.nsa.parallel_nsa", "parallel_nsa"),
    "parallel_nsa_compression": (".FLA.nsa.parallel_nsa_compression", "parallel_nsa_compression"),
    "sage_attention": (".sage_attention.attn_qk_int8_per_block", "forward"),
    "sage_attention_forward": (".sage_attention.attn_qk_int8_per_block", "forward"),
    "sage_attention_per_block_int8": (".sage_attention.quant_per_block", "per_block_int8"),
}



def __getattr__(name: str):
    try:
        module_name, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(importlib.import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_OPERATOR_EXPORTS)
