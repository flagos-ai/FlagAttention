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

"""Parameterized local linear attention implemented with Triton kernels."""

from .decode import HAS_TLE, parallax_attn_with_kvcache, parallax_decode
from .parallel import (
    ParallaxFunction,
    parallel_parallax,
    parallel_parallax_bwd,
    parallel_parallax_fwd,
)

__all__ = [
    "HAS_TLE",
    "ParallaxFunction",
    "parallax_attn_with_kvcache",
    "parallax_decode",
    "parallel_parallax",
    "parallel_parallax_bwd",
    "parallel_parallax_fwd",
]
