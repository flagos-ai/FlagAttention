# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""DiffKV attention with standard Triton and optional TLE backends."""

from .api import (
    DEFAULT_LAYOUT,
    DiffKVLayout,
    OP_NAME,
    diffkv_attention,
    unified_attention_diffkv,
    unified_attention_diffkv_fallback,
    unified_attention_diffkv_tle,
)

__all__ = [
    "OP_NAME",
    "diffkv_attention",
    "unified_attention_diffkv",
    "unified_attention_diffkv_tle",
    "unified_attention_diffkv_fallback",
    "DiffKVLayout",
    "DEFAULT_LAYOUT",
]
