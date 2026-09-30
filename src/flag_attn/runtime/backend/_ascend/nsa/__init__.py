"""Ascend selected and compressed sparse attention."""

import importlib

_OPERATOR_EXPORTS = {
    "parallel_nsa_fwd": (".forward", "parallel_nsa_fwd"),
    "parallel_nsa_compression": (".compression", "parallel_nsa_compression"),
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
