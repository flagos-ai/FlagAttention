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

from flag_attn.runtime.backend import resolve_operator

_OPERATOR_EXPORTS = {
    "sage_attention": ("sage_attention", "flag_attn.sageattention.attn_qk_int8_per_block", "forward"),
    "forward": ("sage_attention", "flag_attn.sageattention.attn_qk_int8_per_block", "forward"),
    "per_block_int8": ("sage_attention_per_block_int8", "flag_attn.sageattention.quant_per_block", "per_block_int8"),
}


def __getattr__(name: str):
    try:
        operator, module, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = resolve_operator(operator, module, symbol)
    globals()[name] = value
    return value


__all__ = ["sage_attention", "forward", "per_block_int8"]
