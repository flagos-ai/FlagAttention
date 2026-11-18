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

"""NVIDIA Hopper (SM90) attention backend.

These kernels use Triton Language Extensions: warp specialization, wgmma, and
TMA descriptors. They run only on Hopper, and only on a Triton that ships
``triton.experimental.tle``. ``_OPERATOR_EXPORTS`` is therefore built after
checking for TLE, so a Hopper machine without it keeps the generic pure-Triton
implementations instead of failing to import.
"""

from __future__ import annotations

import importlib

from flag_attn.utils import has_triton_tle

_TLE_OPERATOR_EXPORTS = {
    "chunk_kda": (".chunk_kda", "chunk_kda_fwd_infer"),
    "chunk_kda_fwd_infer": (".chunk_kda", "chunk_kda_fwd_infer"),
}

_OPERATOR_EXPORTS = _TLE_OPERATOR_EXPORTS if has_triton_tle() else {}


def __getattr__(name: str):
    try:
        module_name, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_OPERATOR_EXPORTS)
