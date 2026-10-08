# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
import importlib


_OPERATOR_EXPORTS = {
    "forward": (".attn_qk_int8_per_block", "forward"),
    "quant_per_block_int8": (".ops", "quant_per_block_int8"),
    "per_block_int8": (".ops", "quant_per_block_int8"),
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
