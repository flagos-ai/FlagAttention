# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Enflame S60 attention backend."""

from __future__ import annotations

import importlib


_OPERATOR_EXPORTS = {
    "chunk_gla": (".FLA.gla", "chunk_gla"),
    "chunk_gdn2": (".FLA.gdn2", "chunk_gdn2"),
    "chunk_gdn2_native": (
        ".FLA.gdn2.native.chunk_fwd",
        "chunk_gdn2_fwd",
    ),
    "chunk_kda": (".FLA.kda", "chunk_kda"),
    "install_msa_prefill": (".msa", "install_msa_prefill"),
    "minimax_m3_index_decode": (".msa", "minimax_m3_index_decode"),
    "minimax_m3_index_score": (".msa", "minimax_m3_index_score"),
    "minimax_m3_index_score_topk": (
        ".msa",
        "minimax_m3_index_score_topk",
    ),
    "minimax_m3_index_topk": (".msa", "minimax_m3_index_topk"),
    "minimax_m3_sparse_attn": (".msa", "minimax_m3_sparse_attn"),
    "minimax_m3_sparse_attn_decode": (
        ".msa",
        "minimax_m3_sparse_attn_decode",
    ),
    "parallel_nsa": (".FLA.nsa", "parallel_nsa"),
    "parallel_nsa_compression": (
        ".FLA.nsa",
        "parallel_nsa_compression",
    ),
    "sage_attention_forward": (
        ".sage_attention.attn_qk_int8_per_block",
        "forward",
    ),
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
