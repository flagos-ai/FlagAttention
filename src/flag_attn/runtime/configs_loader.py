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
    def __init__(
        self,
        backend: str | None = None,
        config_filename: str | Path = "tune_configs.yaml",
    ):
        backend = backend or get_backend_name()
        config_path = (
            Path(__file__).parent / "backend" / f"_{backend}" / config_filename
        )
        self._config_path = config_path
        self._allow_empty_tuning = Path(config_filename).is_absolute()
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

    def get_workload_config(self, op_name: str, workload: tuple) -> triton.Config:
        """Select one measured configuration by the complete native tuning key."""
        entries = self._configs.get(op_name, [])
        keys = []
        for entry in entries:
            if "WORKLOAD" not in entry:
                raise ValueError(f"{op_name}: an exact WORKLOAD is required")
            key = tuple(entry["WORKLOAD"])
            if key in keys:
                raise ValueError(f"Duplicate workload for {op_name}: {key}")
            keys.append(key)
        try:
            index = keys.index(workload)
        except ValueError as error:
            raise KeyError(
                f"No measured configuration for {op_name} workload={workload}; "
                "run offline tuning and load its YAML file"
            ) from error
        return self.get_tuned_config(op_name)[index]

    def get_workload_kernel(self, op_name: str, kernel, arguments: dict):
        """Use native single-config Autotune, retaining the original JIT/heuristics.

        An explicitly empty file enables the original offline search. Missing
        files or unmatched workloads never silently trigger that search.
        """
        if not self._configs:
            if not self._config_path.is_file():
                raise FileNotFoundError(self._config_path)
            if not self._allow_empty_tuning:
                raise ValueError(
                    "Empty runtime workload YAML; offline tuning requires an explicit external file"
                )
            return kernel
        tuner = kernel.fn
        key = tuple(arguments[name] for name in tuner.keys)
        key += tuple(
            str(arguments[name].dtype)
            for name in tuner.arg_names
            if name in arguments and hasattr(arguments[name], "dtype")
        )
        return self._workload_kernel(op_name, kernel, key)

    @lru_cache(maxsize=128)
    def _workload_kernel(self, op_name: str, kernel, workload: tuple):
        # Cache only immutable keys and native launchers, never input tensors.
        config = self.get_workload_config(op_name, workload)
        tuner = kernel.fn
        options = {}
        if "cache_results" in signature(triton.autotune).parameters:
            options["cache_results"] = tuner.cache_results
        selected = triton.autotune(configs=[config], key=tuner.keys, **options)(
            tuner.fn
        )
        return triton.heuristics(dict(kernel.values))(selected)
