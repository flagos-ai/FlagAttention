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

"""Compatibility helpers for running the GLA kernels without FlagGems."""

from collections.abc import Sequence
from typing import Any

import triton


def libtuner(
    *,
    configs: Sequence[triton.Config],
    key: Sequence[str],
    use_cuda_graph: bool = False,
    **kwargs: Any,
):
    """Map FlagGems autotuning metadata to Triton's native autotuner."""
    del use_cuda_graph
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)
