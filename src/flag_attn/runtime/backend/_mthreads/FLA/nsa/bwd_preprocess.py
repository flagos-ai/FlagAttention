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

"""Re-export the generic NSA backward preprocessing.

This vendor's kernel was byte-identical to the generic one, so it reuses it
instead of keeping a copy. Replace this module with a real implementation if
the vendor ever needs to diverge.
"""

from flag_attn.FLA.nsa.bwd_preprocess import (
    parallel_attn_bwd_kernel_preprocess,
    parallel_attn_bwd_preprocess,
)

__all__ = ["parallel_attn_bwd_kernel_preprocess", "parallel_attn_bwd_preprocess"]
