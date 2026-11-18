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

"""Measured common PPU preset for all sequence lengths and attention modes."""


def get_config(head_dim: int, head_dim_v: int, causal: bool, local: bool) -> dict[str, int]:
    """Return the common preset while keeping the launch-policy signature stable."""
    return dict(BLOCK_M=16, BLOCK_N=128, num_warps=8, num_stages=1)
