# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Architecture selection inside the NVIDIA backend.

``op_registry`` resolves one ``(module, symbol)`` pair per vendor and caches
it, so it cannot itself distinguish Hopper from Blackwell. This module is
that resolved symbol: each public name here is a thin facade that picks the
architecture-specialized implementation on first call and caches it.

Selection order for an operator, by the running device's compute capability:

* ``10.x`` / ``12.x`` (Blackwell) -> ``blackwell``, then ``hopper``, then generic
* ``9.x``             (Hopper)    -> ``hopper``, then generic
* anything else                   -> generic

A specialized route that exists but cannot run here (for instance a module
that needs a Triton extension this toolchain does not ship) is skipped rather
than raised, so the operator still resolves to a working implementation.
"""

from functools import lru_cache
from importlib import import_module

# Operator -> per-architecture (module suffix, symbol). A missing entry for an
# architecture means "fall through to the next candidate".
_ARCH_EXPORTS = {
    "chunk_gated_delta_rule": {
        "blackwell": (".blackwell.gated_delta_rule", "chunk_gated_delta_rule"),
        "hopper": (".hopper.gated_delta_rule", "chunk_gated_delta_rule"),
    },
    "chunk_gated_delta_rule_fwd": {
        "blackwell": (".blackwell.gated_delta_rule", "chunk_gated_delta_rule_fwd"),
        "hopper": (".hopper.gated_delta_rule", "chunk_gated_delta_rule_fwd"),
    },
    "fused_recurrent_gated_delta_rule_fwd": {
        "hopper": (".hopper.gated_delta_rule", "fused_recurrent_gated_delta_rule_fwd"),
    },
    "fused_recurrent_gated_delta_rule_packed_decode": {
        "hopper": (
            ".hopper.gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
        ),
    },
}

# Generic fallbacks, used when no specialized route is importable.
_GENERIC_EXPORTS = {
    "chunk_gated_delta_rule": ("flag_attn.FLA.gated_delta_rule.api", "chunk_gated_delta_rule"),
    "chunk_gated_delta_rule_fwd": (
        "flag_attn.FLA.gated_delta_rule.chunk",
        "chunk_gated_delta_rule_fwd",
    ),
}

_OPERATOR_EXPORTS = {name: (".dispatch", name) for name in _ARCH_EXPORTS}

__all__ = sorted(_ARCH_EXPORTS)


@lru_cache(maxsize=1)
def architecture() -> str:
    """Return ``"blackwell"``, ``"hopper"`` or ``"other"`` for this device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "other"
        major = torch.cuda.get_device_capability()[0]
    except Exception:
        return "other"
    if major in (10, 12):
        return "blackwell"
    if major == 9:
        return "hopper"
    return "other"


def _candidates(name: str) -> list[tuple[str, str]]:
    arch = architecture()
    order = {
        "blackwell": ("blackwell", "hopper"),
        "hopper": ("hopper",),
        "other": (),
    }[arch]
    exports = _ARCH_EXPORTS.get(name, {})
    found = [exports[key] for key in order if key in exports]
    generic = _GENERIC_EXPORTS.get(name)
    return [*found, *([generic] if generic else [])]


@lru_cache(maxsize=None)
def _resolve(name: str):
    errors = []
    for module_name, symbol in _candidates(name):
        # Relative candidates are anchored at this module's package, not at
        # the module itself -- ``__name__`` here is ``..._nvidia.dispatch``,
        # which is not a package, so every relative import would fail.
        package = __package__ if module_name.startswith(".") else None
        try:
            return getattr(import_module(module_name, package), symbol)
        except (AttributeError, ImportError) as exc:
            # A specialized route whose toolchain requirements are unmet is a
            # reason to try the next candidate, not a hard failure.
            errors.append(f"{module_name}.{symbol}: {type(exc).__name__}: {exc}")
    raise ImportError(
        f"no usable implementation of {name!r} for architecture {architecture()!r}; "
        + "; ".join(errors)
    )


def __getattr__(name):
    if name not in _ARCH_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def dispatch(*args, **kwargs):
        return _resolve(name)(*args, **kwargs)

    dispatch.__name__ = name
    dispatch.__qualname__ = name
    dispatch.__doc__ = f"Architecture-dispatched {name} for NVIDIA GPUs."
    globals()[name] = dispatch
    return dispatch
