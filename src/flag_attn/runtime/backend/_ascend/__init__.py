# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
import importlib

_OPERATOR_EXPORTS = {
    "chunk_gdn2": (".gdn2.api", "chunk_gdn2"),
    "chunk_gla": (".gla", "chunk_gla"),
    "chunk_kda_fwd_infer": (".kda.api", "chunk_kda_fwd_infer"),
    "parallel_nsa_compression": (".nsa.api", "parallel_nsa_compression"),
    "minimax_m3_index_score": (".minimax_sparse_attention.index_topk", "minimax_m3_index_score"),
    "minimax_m3_index_topk": (".minimax_sparse_attention.index_topk", "minimax_m3_index_topk"),
    "minimax_m3_index_decode_score": (".minimax_sparse_attention.index_topk", "minimax_m3_index_decode_score"),
    "minimax_m3_index_decode": (".minimax_sparse_attention.index_topk", "minimax_m3_index_decode"),
    "minimax_m3_sparse_attn": (".minimax_sparse_attention.sparse_attn", "minimax_m3_sparse_attn"),
    "forward": (".attn_qk_int8_per_block", "forward"),
    "per_block_int8": (".ops", "quant_per_block_int8"),
    "quant_per_block_int8": (".ops", "quant_per_block_int8"),
    "sage_attention": (".attn_qk_int8_per_block", "forward"),
    "sage_attention_forward": (".attn_qk_int8_per_block", "forward"),
    "sage_attention_per_block_int8": (".ops", "quant_per_block_int8"),
}

__all__ = [
    "chunk_gdn2", "chunk_gla", "chunk_kda_fwd_infer", "forward",
    "minimax_m3_index_decode", "minimax_m3_index_decode_score",
    "minimax_m3_index_score", "minimax_m3_index_topk", "minimax_m3_sparse_attn",
    "parallel_nsa_compression", "per_block_int8", "quant_per_block_int8",
    "sage_attention", "sage_attention_forward", "sage_attention_per_block_int8",
]


def __getattr__(name):
    try:
        module_name, attribute_name = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
