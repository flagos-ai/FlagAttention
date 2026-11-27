# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Lazy vendor overrides with an explicit generic fallback.

Backend packages declare compatible public symbols in _OPERATOR_EXPORTS.
Missing packages or symbols retain the generic implementation. A declared
implementation that fails to import or execute is an error, not a fallback.
Selection is cached for the process's detected vendor; callers can supply
vendor explicitly for inspection. No kernel is called by the registry.
"""

import os
from functools import lru_cache
from importlib import import_module
from importlib.util import resolve_name


@lru_cache(maxsize=None)
def _backend_exports(vendor: str) -> dict[str, tuple[str, str]]:
    if not vendor.isidentifier():
        raise ValueError(f"Invalid backend name: {vendor!r}")
    module_name = f"flag_attn.runtime.backend._{vendor}"
    try:
        backend = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        return {}
    exports = vars(backend).get("_OPERATOR_EXPORTS", {})
    generic_only = vars(backend).get("_GENERIC_ONLY_OPS", ())
    return {
        name: (resolve_name(module, module_name) if module.startswith(".") else module, symbol)
        for name, (module, symbol) in exports.items()
        if name not in generic_only
    }


@lru_cache(maxsize=None)
def _resolve_operator(name: str, generic_module: str, generic_name: str, vendor: str):
    module_name, symbol = _backend_exports(vendor).get(name, (generic_module, generic_name))
    return getattr(import_module(module_name), symbol)


def resolve_operator(
    name: str,
    generic_module: str,
    generic_name: str | None = None,
    *,
    vendor: str | None = None,
):
    """Return the selected vendor's exported implementation or the generic one.

    generic_module must identify a concrete implementation, rather than a
    public facade that would recursively resolve the same operator. Stage
    facades resolve each primitive separately, preserving their controls.
    Unknown symbols and dependency errors in declared overrides propagate.
    """
    # Set before importing public operators to validate the portable path on
    # hardware that also has a registered specialization.
    if os.environ.get("FLAG_ATTN_FORCE_GENERIC", "0").lower() in {"1", "true", "yes", "on"}:
        return getattr(import_module(generic_module), generic_name or name)
    if vendor is None:
        vendor = import_module("flag_attn.runtime").device.vendor_name
    return _resolve_operator(name, generic_module, generic_name or name, vendor)


__all__ = ["resolve_operator"]
