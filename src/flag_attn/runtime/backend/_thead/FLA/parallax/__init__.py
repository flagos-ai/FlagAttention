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

import importlib


_EXPORTS = {
    "ParallaxFunction": (".parallel", "ParallaxFunction"),
    "parallax_attn_with_kvcache": (".decode", "parallax_attn_with_kvcache"),
    "parallax_decode": (".decode", "parallax_decode"),
    "parallel_parallax": (".parallel", "parallel_parallax"),
    "parallel_parallax_bwd": (".parallel", "parallel_parallax_bwd"),
    "parallel_parallax_fwd": (".parallel", "parallel_parallax_fwd"),
}


def __getattr__(name: str):
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
