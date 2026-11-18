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

"""Re-export the generic NSA kernels.

This file was a copy of the generic one whose only behavioral difference was
capping the output tile to BV=64, because MetaX registers cannot hold the
default D=128 tile. That cap now lives in this backend's tune_configs.yaml,
under ``tile_limits.parallel_nsa_fwd``, so the generic kernel applies it here
and nowhere else, and there is no second copy to keep in sync.
"""

from flag_attn.FLA.nsa.parallel_nsa import (
    ParallelNSAFunction,
    parallel_nsa,
    parallel_nsa_block_mask,
    parallel_nsa_bwd,
    parallel_nsa_bwd_kernel_dkv,
    parallel_nsa_bwd_kernel_dq,
    parallel_nsa_fwd,
    parallel_nsa_fwd_kernel,
    parallel_nsa_kernel_mask,
    parallel_nsa_kernel_topk,
    parallel_nsa_topk,
)

__all__ = [
    "ParallelNSAFunction",
    "parallel_nsa",
    "parallel_nsa_block_mask",
    "parallel_nsa_bwd",
    "parallel_nsa_bwd_kernel_dkv",
    "parallel_nsa_bwd_kernel_dq",
    "parallel_nsa_fwd",
    "parallel_nsa_fwd_kernel",
    "parallel_nsa_kernel_mask",
    "parallel_nsa_kernel_topk",
    "parallel_nsa_topk",
]
