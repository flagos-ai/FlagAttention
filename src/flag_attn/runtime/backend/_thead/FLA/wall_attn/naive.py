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

"""Re-export the generic naive Wall-Attention reference.

This vendor's copy was byte-identical to the generic one. A reference
implementation describes the operator's semantics rather than any device, so
there is nothing for a vendor to specialize here.
"""

from flag_attn.FLA.wall_attn.naive import naive_wall_attn

__all__ = ["naive_wall_attn"]
