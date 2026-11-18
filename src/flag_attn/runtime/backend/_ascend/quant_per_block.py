# Copyright 2026 FlagOS Contributors
# Copyright 2024 SageAttention Team
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

"""Re-export the generic per-block INT8 quantization.

This file differed from the generic one only in import paths and blank lines,
so it reuses the generic kernel rather than keeping a copy that has to be kept
in sync. Replace this module with a real implementation if Ascend needs to
diverge; a difference that is only tile or launch tuning belongs in this
backend's tune_configs.yaml instead.
"""

from flag_attn.sage_attention.quant_per_block import (
    per_block_int8,
    quant_per_block_int8_kernel,
)

__all__ = ["per_block_int8", "quant_per_block_int8_kernel"]
