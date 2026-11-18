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

"""Re-export the generic GDN2 gate kernels.

This file differed from the generic one only in import paths, so it reuses the
generic kernels rather than keeping a copy that has to be kept in sync.
Replace this module with a real implementation if Hygon needs to diverge; a
difference that is only tile or launch tuning belongs in tune_configs.yaml.
"""

from flag_attn.FLA.gdn2.native.gate import (
    KDAGateFunction,
    autocast_custom_bwd,
    autocast_custom_fwd,
    fused_kda_gate,
    input_guard,
    kda_gate_bwd,
    kda_gate_bwd_kernel,
    kda_gate_chunk_cumsum,
    kda_gate_chunk_cumsum_vector_kernel,
    kda_gate_fwd,
    kda_gate_fwd_kernel,
    naive_kda_gate,
    naive_kda_lowerbound_gate,
    softplus,
)

__all__ = [
    "KDAGateFunction",
    "autocast_custom_bwd",
    "autocast_custom_fwd",
    "fused_kda_gate",
    "input_guard",
    "kda_gate_bwd",
    "kda_gate_bwd_kernel",
    "kda_gate_chunk_cumsum",
    "kda_gate_chunk_cumsum_vector_kernel",
    "kda_gate_fwd",
    "kda_gate_fwd_kernel",
    "naive_kda_gate",
    "naive_kda_lowerbound_gate",
    "softplus",
]
