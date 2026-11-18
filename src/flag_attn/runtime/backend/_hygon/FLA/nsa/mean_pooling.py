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

"""Re-export the generic NSA mean-pooling kernels.

This file differed from the generic one only in import paths, so it reuses the
generic kernels rather than keeping a copy that has to be kept in sync.
Replace this module with a real implementation if Hygon needs to diverge; a
difference that is only tile or launch tuning belongs in this backend's
tune_configs.yaml instead.
"""

from flag_attn.FLA.nsa.mean_pooling import (
    MeanPoolingFunction,
    mean_pooling,
    mean_pooling_bwd,
    mean_pooling_bwd_kernel,
    mean_pooling_fwd,
    mean_pooling_fwd_kernel,
)

__all__ = [
    "MeanPoolingFunction",
    "mean_pooling",
    "mean_pooling_bwd",
    "mean_pooling_bwd_kernel",
    "mean_pooling_fwd",
    "mean_pooling_fwd_kernel",
]
