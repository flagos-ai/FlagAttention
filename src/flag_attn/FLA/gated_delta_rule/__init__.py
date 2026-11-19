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

"""Portable gated delta rule forward.

Plain Triton with no vendor or architecture assumptions: explicit pointer
arithmetic with masks, ``tl.dot``, ``tl.cumsum`` and elementwise builtins
only -- no block pointers, tensor descriptors, TMA, warp specialization or
Triton extension modules. This is the fallback every backend resolves to
when it declares no override of its own, and the reference semantics the
specialized backends must match.

Architecture-specialized routes live under ``flag_attn.runtime.backend``.
"""

from flag_attn.FLA.gated_delta_rule.api import (
    _l2_normalize_last_dim,
    chunk_gated_delta_rule,
)
from flag_attn.FLA.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
from flag_attn.FLA.gated_delta_rule.prepare import gdn_fwd_prepare
from flag_attn.FLA.gated_delta_rule.wy_fast import recompute_w_u_fwd

__all__ = [
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
    "gdn_fwd_prepare",
    "recompute_w_u_fwd",
]
