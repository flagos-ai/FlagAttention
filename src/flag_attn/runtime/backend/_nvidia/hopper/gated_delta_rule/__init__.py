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

"""NVIDIA Hopper gated delta rule.

This is the Hopper-specialized implementation, moved here from the former
generic location (``flag_attn.FLA.gated_delta_rule``). It keeps the routes
that depend on Hopper-class hardware and on Triton's TLE extension: warp
specialization over ``tle.pipe``, TMA descriptor loads, and the fused
cumsum/KKT/triangular-solve prologue. Modules that need TLE import it at
module scope, so they are reached lazily through the attribute hooks below
and never imported on a toolchain without it.

Portable fallbacks live in ``flag_attn.FLA.gated_delta_rule``; the Blackwell
route lives in ``flag_attn.runtime.backend._nvidia.blackwell``.
"""

import importlib

_OPERATOR_EXPORTS = {
    "chunk_gated_delta_rule": (".api", "chunk_gated_delta_rule"),
    "chunk_gated_delta_rule_fwd": (".chunk", "chunk_gated_delta_rule_fwd"),
    "fused_recurrent_gated_delta_rule_fwd": (
        ".fused_recurrent",
        "fused_recurrent_gated_delta_rule_fwd",
    ),
    "fused_recurrent_gated_delta_rule_packed_decode": (
        ".packed_decode",
        "fused_recurrent_gated_delta_rule_packed_decode",
    ),
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
