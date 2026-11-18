# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Parallel Native Sparse Attention."""

from typing import Literal

from flag_attn.runtime.backend import resolve_operator

_OPERATOR_EXPORTS = {
    "parallel_nsa_compression": (
        "parallel_nsa_compression",
        "flag_attn.parallel_nsa.parallel_nsa_compression",
        "parallel_nsa_compression",
    )
}


def __getattr__(name: str):
    try:
        operator, module, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = resolve_operator(operator, module, symbol)
    globals()[name] = value
    return value


def parallel_nsa(*args, mode: Literal["full", "compression"] = "full", **kwargs):
    """Run full/selected NSA or compression using the selected backend.

    Each component falls back independently to its portable implementation
    when the backend does not export it. Arguments and returns are unchanged.
    """
    if mode == "full":
        implementation = resolve_operator("parallel_nsa", "flag_attn.parallel_nsa.parallel_nsa")
    elif mode == "compression":
        implementation = __getattr__("parallel_nsa_compression")
    else:
        raise ValueError(f"Unsupported NSA mode: {mode!r}; expected 'full' or 'compression'")
    return implementation(*args, **kwargs)


__all__ = ["parallel_nsa", "parallel_nsa_compression"]
