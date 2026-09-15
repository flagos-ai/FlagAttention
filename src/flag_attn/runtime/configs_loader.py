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

from functools import lru_cache
from inspect import signature
from pathlib import Path

import triton
import yaml

from .backend import get_backend_name


class TunedConfigLoader:
    def __init__(self):
        backend = get_backend_name()
        config_path = Path(__file__).parent / "backend" / f"_{backend}" / "tune_configs.yaml"
        try:
            with config_path.open() as config_file:
                self._configs = yaml.safe_load(config_file) or {}
        except FileNotFoundError:
            self._configs = {}
        self._config_parameters = signature(triton.Config).parameters

    @lru_cache(maxsize=None)
    def get_tuned_config(self, op_name: str) -> list[triton.Config]:
        result = []
        for entry in self._configs.get(op_name, []):
            kwargs = {}
            for name in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
                if name in entry and name in self._config_parameters:
                    kwargs[name] = entry[name]
            result.append(triton.Config(entry.get("META", {}), **kwargs))
        return result
