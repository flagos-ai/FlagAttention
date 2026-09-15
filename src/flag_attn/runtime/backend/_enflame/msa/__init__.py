# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""S60 MSA backend bindings."""

from __future__ import annotations

import importlib

from .index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_decode_score,
    minimax_m3_index_score,
    minimax_m3_index_score_topk,
    minimax_m3_index_topk,
)
from .sparse_attn import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)


def install_msa_prefill(use_tle: bool = False) -> None:
    """Install the Enflame prefill and decode bindings."""
    if use_tle:
        raise RuntimeError(
            "S60 GCU300 compiler does not lower the Triton TLE dialect"
        )

    public = importlib.import_module(
        "flag_attn.minimax_sparse_attention"
    )
    root = importlib.import_module("flag_attn")
    index_module = importlib.import_module(
        "flag_attn.minimax_sparse_attention.index_topk"
    )
    sparse_module = importlib.import_module(
        "flag_attn.minimax_sparse_attention.sparse_attn"
    )

    bindings = {
        "minimax_m3_index_score": minimax_m3_index_score,
        "minimax_m3_index_score_topk": minimax_m3_index_score_topk,
        "minimax_m3_index_topk": minimax_m3_index_topk,
        "minimax_m3_index_decode": minimax_m3_index_decode,
        "minimax_m3_sparse_attn": (
            minimax_m3_sparse_attn
        ),
        "minimax_m3_sparse_attn_decode": (
            minimax_m3_sparse_attn_decode
        ),
    }

    for name, function in bindings.items():
        setattr(public, name, function)
        setattr(root, name, function)

        if "index" in name:
            setattr(index_module, name, function)
        else:
            setattr(sparse_module, name, function)


__all__ = [
    "SPARSE_BLOCK_SIZE",
    "install_msa_prefill",
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_score_topk",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
]
