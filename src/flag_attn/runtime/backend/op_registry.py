# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Lazy vendor overrides with an explicit generic fallback.

Backend packages declare compatible public symbols in _OPERATOR_EXPORTS.
Missing packages or symbols retain the generic implementation. A declared
implementation that fails to import or execute is an error, not a fallback.
Selection is cached for the process's detected vendor; callers can supply
vendor explicitly for inspection. No kernel is called by the registry.

A vendor whose GPU generations need separate kernels may ship a subpackage per
generation, declaring the same table. Resolution then prefers the running
generation's export over the vendor's own, and the vendor's over the generic
implementation, so a kernel written for one generation never runs on another.
"""

from functools import lru_cache
from importlib import import_module
from importlib.util import resolve_name

from .arch_finder import detect_arch


def _exports_of(module_name: str) -> dict[str, tuple[str, str]]:
    """Return one backend package's declared exports, resolved to absolute names."""
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
def _backend_exports(vendor: str, arch: str | None = None) -> dict[str, tuple[str, str]]:
    if not vendor.isidentifier():
        raise ValueError(f"Invalid backend name: {vendor!r}")
    vendor_module = f"flag_attn.runtime.backend._{vendor}"
    exports = _exports_of(vendor_module)
    if arch:
        if not arch.isidentifier():
            raise ValueError(f"Invalid arch name: {arch!r}")
        # The generation's table wins over the vendor's for the names it declares,
        # and leaves every other name resolving exactly as it did before.
        exports = {**exports, **_exports_of(f"{vendor_module}.{arch}")}
    return exports


@lru_cache(maxsize=None)
def _resolve_operator(name: str, generic_module: str, generic_name: str, vendor: str, arch: str | None):
    module_name, symbol = _backend_exports(vendor, arch).get(name, (generic_module, generic_name))
    return getattr(import_module(module_name), symbol)


def resolve_operator(
    name: str,
    generic_module: str,
    generic_name: str | None = None,
    *,
    vendor: str | None = None,
    arch: str | None = None,
):
    """Return the selected implementation: generation, then vendor, then generic.

    generic_module must identify a concrete implementation, rather than a
    public facade that would recursively resolve the same operator. Stage
    facades resolve each primitive separately, preserving their controls.
    Unknown symbols and dependency errors in declared overrides propagate.
    Pass vendor or arch explicitly to inspect a selection this host would not
    make; omitting both selects what this host runs.
    """
    if vendor is None:
        vendor = import_module("flag_attn.runtime").device.vendor_name
    if arch is None:
        arch = detect_arch(vendor)
    return _resolve_operator(name, generic_module, generic_name or name, vendor, arch)


__all__ = ["resolve_operator"]
