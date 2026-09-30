"""Ascend MiniMax sparse attention and block indexing."""
import importlib

_OPERATOR_EXPORTS = {
    "minimax_m3_index_score": (".index_topk", "minimax_m3_index_score"),
    "minimax_m3_index_topk": (".index_topk", "minimax_m3_index_topk"),
    "minimax_m3_index_decode_score": (".index_topk", "minimax_m3_index_decode_score"),
    "minimax_m3_index_decode": (".index_topk", "minimax_m3_index_decode"),
    "minimax_m3_sparse_attn": (".sparse_attn", "minimax_m3_sparse_attn"),
    "minimax_m3_sparse_attn_decode": (".sparse_attn", "minimax_m3_sparse_attn_decode"),
}

__all__ = sorted(_OPERATOR_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
