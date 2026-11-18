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

import inspect
import os

import triton

from .backend import DeviceDetector
from .configs_loader import TunedConfigLoader, tile_limit, tuned_config_or

device = DeviceDetector()
torch_device_fn = device.torch_device_fn
config_loader = TunedConfigLoader(device.vendor_name)


def get_tuned_config(op_name: str):
    return config_loader.get_tuned_config(op_name)


def tuned_config(op_name: str, fallback):
    """Return this backend's tuned configurations for op_name, else fallback.

    Kernels use this so a backend that only needs different tiles or launch
    bounds can say so in its tune_configs.yaml instead of forking the kernel.
    """
    return tuned_config_or(op_name, fallback)


def autotune_cache_kwargs() -> dict:
    if "cache_results" not in inspect.signature(triton.autotune).parameters:
        return {}
    return {"cache_results": os.environ.get("FLA_CACHE_RESULTS", "1") == "1"}


def clamp_tile(op_name: str, tile: str, value: int) -> int:
    """Clamp a kernel's computed tile to this backend's cap, when it sets one."""
    return tile_limit(op_name, tile, value)
