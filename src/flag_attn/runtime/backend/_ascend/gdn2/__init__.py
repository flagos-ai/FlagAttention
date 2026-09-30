"""Ascend GDN2 implementation, separate from backend-wide common ops."""

import importlib

_OPERATOR_EXPORTS = {
    "chunk_gdn2": (".chunk_fwd_infer", "chunk_gdn2_fwd_infer"),
    "chunk_gdn2_fwd_infer": (".chunk_fwd_infer", "chunk_gdn2_fwd_infer"),
}
__all__ = sorted(_OPERATOR_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    if name == "chunk_gdn2":
        def value(*args, **kwargs):
            implementation = getattr(importlib.import_module(module_name, __name__), attribute_name)
            return implementation(*args, **kwargs)
    else:
        value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
