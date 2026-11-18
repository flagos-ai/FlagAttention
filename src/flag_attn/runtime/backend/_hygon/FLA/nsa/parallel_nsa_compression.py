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

"""Re-export the generic NSA compression kernels.

This file differed from the generic one only in import paths, so it reuses the
generic kernels rather than keeping a copy that has to be kept in sync.
Replace this module with a real implementation if Hygon needs to diverge; a
difference that is only tile or launch tuning belongs in tune_configs.yaml.
"""

from flag_attn.FLA.nsa.parallel_nsa_compression import (
    ParallelNSACompressionFunction,
    parallel_nsa_compression,
    parallel_nsa_compression_bwd,
    parallel_nsa_compression_bwd_kernel_dkv,
    parallel_nsa_compression_bwd_kernel_dq,
    parallel_nsa_compression_fwd,
    parallel_nsa_compression_fwd_kernel,
)

__all__ = [
    "ParallelNSACompressionFunction",
    "parallel_nsa_compression",
    "parallel_nsa_compression_bwd",
    "parallel_nsa_compression_bwd_kernel_dkv",
    "parallel_nsa_compression_bwd_kernel_dq",
    "parallel_nsa_compression_fwd",
    "parallel_nsa_compression_fwd_kernel",
]
