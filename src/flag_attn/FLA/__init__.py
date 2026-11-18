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

"""Operators developed against the flash-linear-attention repository.

FLA is flash-linear-attention (https://github.com/fla-org/flash-linear-attention).
A subpackage here corresponds to one upstream ``fla.ops`` operator and keeps
that operator's upstream name, so a reader can find the counterpart of either
repository in the other. Operators with no upstream counterpart, such as
MiniMax M3 or SageAttention, live beside this package rather than inside it.

Vendor packages mirror these names under
``runtime/backend/_<vendor>/FLA/<operator>``.
"""

__all__ = [
    "attnres",
    "forgetting_attn",
    "gated_delta_rule",
    "gdn2",
    "gla",
    "kda",
    "log_linear_attn",
    "moba",
    "nsa",
    "parallax",
    "wall_attn",
]
