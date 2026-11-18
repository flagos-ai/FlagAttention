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

"""T-Head Inkling FA4 paged relative attention."""

from .relative_attention import (
    get_backend,
    inkling_fa4_rel_attention,
    inkling_fa4_rel_attention_triton,
)

__all__ = [
    "get_backend",
    "inkling_fa4_rel_attention",
    "inkling_fa4_rel_attention_triton",
]
