# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Private exact-workload selection for the GDN2 native single-config path."""

from functools import lru_cache
from importlib.resources import files
from inspect import signature

import triton
import yaml

from flag_attn.runtime.configs_loader import TunedConfigLoader


@lru_cache(maxsize=1)
def _tuning_source():
    # Resolve installed package resources, never paths relative to the caller.
    resource = files("flag_attn.runtime.backend._nvidia").joinpath("tune_configs.yaml")
    try:
        with resource.open() as stream:
            data = yaml.safe_load(stream) or {}
    except FileNotFoundError:
        data = {}
    return data, TunedConfigLoader(backend="nvidia")


@lru_cache(maxsize=None)
def _workload_configs(op_name: str) -> dict[tuple, triton.Config]:
    data, loader = _tuning_source()
    entries = data.get(op_name, [])
    configs = loader.get_tuned_config(op_name)
    if len(entries) != len(configs):
        raise ValueError(f"{op_name}: WORKLOAD/config count mismatch")
    result = {}
    for entry, config in zip(entries, configs):
        if "WORKLOAD" not in entry:
            raise ValueError(f"{op_name}: an exact WORKLOAD is required")
        workload = tuple(entry["WORKLOAD"])
        if workload in result:
            raise ValueError(f"Duplicate workload for {op_name}: {workload}")
        result[workload] = config
    return result


@lru_cache(maxsize=128)
def _workload_kernel(op_name: str, kernel, workload: tuple):
    # Cache native launchers and immutable metadata, never input tensors.
    try:
        config = _workload_configs(op_name)[workload]
    except KeyError as error:
        raise KeyError(
            f"No measured configuration for {op_name} workload={workload}; "
            "add the offline-tuned workload to tune_configs.yaml"
        ) from error
    tuner = kernel.fn
    options = {}
    if "cache_results" in signature(triton.autotune).parameters:
        options["cache_results"] = tuner.cache_results
    selected = triton.autotune(configs=[config], key=tuner.keys, **options)(tuner.fn)
    return triton.heuristics(dict(kernel.values))(selected)


def get_gdn2_workload_kernel(op_name: str, kernel, launch_arguments: dict):
    """Return the cached single-config launcher for the complete native key."""
    tuner = kernel.fn
    workload = tuple(launch_arguments[name] for name in tuner.keys)
    workload += tuple(
        str(launch_arguments[name].dtype)
        for name in tuner.arg_names
        if name in launch_arguments and hasattr(launch_arguments[name], "dtype")
    )
    return _workload_kernel(op_name, kernel, workload)
