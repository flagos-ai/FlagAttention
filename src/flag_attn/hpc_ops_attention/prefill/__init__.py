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

"""HPC attention operators for the prefill phase."""

from .attention_blocksparse_prefill_fp8 import (
    attention_with_kvcache_blocksparse_prefill_fp8,
    attention_with_kvcache_blocksparse_prefill_fp8_hopper,
    attention_with_kvcache_blocksparse_prefill_fp8_tle,
    attention_with_kvcache_blocksparse_prefill_fp8_triton,
)

__all__ = [
    "attention_with_kvcache_blocksparse_prefill_fp8",
    "attention_with_kvcache_blocksparse_prefill_fp8_hopper",
    "attention_with_kvcache_blocksparse_prefill_fp8_tle",
    "attention_with_kvcache_blocksparse_prefill_fp8_triton",
]
