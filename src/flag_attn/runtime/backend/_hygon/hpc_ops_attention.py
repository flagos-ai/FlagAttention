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

"""Hygon gfx936 entry points for Hy3 dynamic decode.

The generic Hopper modules route to the gfx936 paged split-K workspace in
``flag_attn.hpc_ops_attention.decode.hygon`` whenever they run on a Hygon
device, so the concrete implementations are the generic dynamic-decode
functions themselves.  Re-exporting them here registers the vendor
specialization with the operator registry, which makes the upstream Hy3
dynamic-decode tests and benchmarks run on Hygon instead of skipping.
"""

from flag_attn.hpc_ops_attention.decode.dynamic.bf16_dynamic import (
    attention_decode_bf16_dynamic,
)
from flag_attn.hpc_ops_attention.decode.dynamic.fp8_qkpertoken_perhead_vperhead_dynamic import (
    attention_decode_fp8 as attention_decode_fp8_qk_dynamic,
)
from flag_attn.hpc_ops_attention.decode.dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic import (
    attention_decode_fp8 as attention_decode_fp8_kv_dynamic,
)

__all__ = [
    "attention_decode_bf16_dynamic",
    "attention_decode_fp8_qk_dynamic",
    "attention_decode_fp8_kv_dynamic",
]
