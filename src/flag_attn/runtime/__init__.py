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

from .configs_loader import TunedConfigLoader

config_loader = TunedConfigLoader()


def get_tuned_config(op_name: str):
    return config_loader.get_tuned_config(op_name)


def autotune_cache_kwargs() -> dict:
    if "cache_results" not in inspect.signature(triton.autotune).parameters:
        return {}
    return {"cache_results": os.environ.get("FLA_CACHE_RESULTS", "1") == "1"}
