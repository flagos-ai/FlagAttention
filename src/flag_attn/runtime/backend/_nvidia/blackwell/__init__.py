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

"""NVIDIA Blackwell (SM100/SM120) attention backend.

Kernels that need Blackwell instructions, such as tcgen05 MMA or its second
TMA generation, belong here. Declare each one in ``_OPERATOR_EXPORTS`` under
the operator name it replaces, mapping to ``(".module", "symbol")``;
``op_registry`` then selects it ahead of the vendor-wide and generic
implementations on a Blackwell device, and ignores it everywhere else.

The table is empty today, so every operator resolves to the generic
pure-Triton implementation, which runs correctly on Blackwell. A kernel added
here should be guarded the way ``hopper`` guards its TLE imports, so a
Blackwell machine whose Triton lacks the required extension keeps the generic
implementation rather than failing to import.
"""

from __future__ import annotations

import importlib

_OPERATOR_EXPORTS: dict[str, tuple[str, str]] = {}


def __getattr__(name: str):
    try:
        module_name, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_OPERATOR_EXPORTS)
