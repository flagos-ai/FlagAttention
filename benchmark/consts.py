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

"""NVIDIA baseline lookup and scaling factor helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_DTYPE_TO_SF_KEY = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.float64": "fp64",
    "torch.int8": "int8",
    "torch.float8_e4m3fn": "fp8",
    "torch.float8_e5m2": "fp8",
}
_BOTTLENECK_TO_SF = {"mem": "bandwidth_gbps", "tensor": "tensor", "cuda": "vector"}
_SIG_INDEX_CACHE: dict[tuple[int, str], dict[str, dict[tuple, list[str]]]] = {}


def _tensor_shapes(value: Any) -> tuple[tuple[int, ...], ...]:
    """Collect the tensor shapes written by FlagAttention's record_shapes()."""
    shapes: list[tuple[int, ...]] = []
    stack = [value]
    while stack:
        item = stack.pop(0)
        if isinstance(item, Mapping):
            shape = item.get("shape")
            if isinstance(shape, (list, tuple)) and all(
                isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
            ):
                shapes.append(tuple(shape))
            else:
                stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
    return tuple(sorted(shapes))


def _first_tensor_shape(value: Any) -> list[int] | None:
    """Return the first recorded tensor shape for a direct baseline key lookup."""
    stack = [value]
    while stack:
        item = stack.pop(0)
        if isinstance(item, Mapping):
            shape = item.get("shape")
            if isinstance(shape, (list, tuple)) and all(
                isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
            ):
                return list(shape)
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
    return None


def _base_shapes(config: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(config, Mapping) or not isinstance(config.get("inputs"), Mapping):
        return ()
    return _tensor_shapes(config["inputs"])


def _scalar_values(value: Any) -> tuple[str, ...]:
    scalars: list[str] = []
    stack = [value]
    while stack:
        item = stack.pop(0)
        if isinstance(item, Mapping):
            if "shape" in item:
                continue
            if "scalar" in item:
                scalars.append(repr(item["scalar"]))
            else:
                stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
        elif isinstance(item, (bool, int, float, str)):
            scalars.append(repr(item))
    return tuple(sorted(scalars))


def _signature_index(
    base_data: Mapping[str, Any], op_name: str, shapes: Mapping[str, Any]
) -> dict[str, dict[tuple, list[str]]]:
    cache_key = (id(base_data), op_name)
    if cache_key not in _SIG_INDEX_CACHE:
        index: dict[str, dict[tuple, list[str]]] = {}
        for shape_key, per_dtype in shapes.items():
            if not isinstance(per_dtype, Mapping):
                continue
            for dtype, record in per_dtype.items():
                if not isinstance(record, Mapping):
                    continue
                signature = _base_shapes(record.get("config"))
                if signature:
                    index.setdefault(dtype, {}).setdefault(signature, []).append(shape_key)
        _SIG_INDEX_CACHE[cache_key] = index
    return _SIG_INDEX_CACHE[cache_key]


def lookup_base_record_ex(
    base_data: Mapping[str, Any], op_name: str, dtype: str, shape_detail: Any
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Look up an NV record by op, shape and dtype; return a miss reason."""
    op_entry = base_data.get(op_name)
    if not isinstance(op_entry, Mapping):
        return None, f"operator '{op_name}' not found in baseline"
    shapes = op_entry.get("shapes")
    if not isinstance(shapes, Mapping):
        return None, f"operator '{op_name}' has no shape records in baseline"

    if isinstance(shape_detail, Mapping):
        # The AttnRes benchmark uses semantic dimensions instead of tensor inputs.
        semantic = [
            (key, per_dtype)
            for key, per_dtype in shapes.items()
            if isinstance(per_dtype, Mapping)
            and any(
                isinstance(record, Mapping)
                and isinstance(record.get("config"), Mapping)
                and record["config"].get("dims") == {
                    key: str(value) if str(value).startswith("torch.") else value
                    for key, value in shape_detail.items()
                }
                for record in per_dtype.values()
            )
        ]
        if len(semantic) == 1:
            key, per_dtype = semantic[0]
            record = per_dtype.get(str(dtype))
            if isinstance(record, Mapping):
                return record, None
            return None, (
                f"operator '{op_name}' shape {key} has no matching "
                f"dtype {dtype} in baseline"
            )

    missing_dtype_shape = None
    signature = _tensor_shapes(shape_detail)
    if signature:
        first_shape = _first_tensor_shape(shape_detail)
        per_dtype = shapes.get(str(first_shape))
        if isinstance(per_dtype, Mapping):
            record = per_dtype.get(str(dtype))
            if isinstance(record, Mapping):
                return record, None
            missing_dtype_shape = first_shape

        keys = _signature_index(base_data, op_name, shapes).get(str(dtype), {}).get(signature, [])
        if len(keys) == 1:
            record = shapes[keys[0]][str(dtype)]
            return record, None
        if len(keys) > 1:
            scalars = set(_scalar_values(shape_detail))
            for key in keys:
                record = shapes[key][str(dtype)]
                config = record.get("config")
                inputs = config.get("inputs", {}) if isinstance(config, Mapping) else {}
                baseline_scalars = set(_scalar_values(inputs))
                if baseline_scalars and baseline_scalars <= scalars:
                    return record, None
    elif isinstance(shape_detail, (list, tuple)):
        per_dtype = shapes.get(str(list(shape_detail)))
        if isinstance(per_dtype, Mapping):
            record = per_dtype.get(str(dtype))
            if isinstance(record, Mapping):
                return record, None
            missing_dtype_shape = list(shape_detail)

    if missing_dtype_shape is not None:
        return None, (
            f"operator '{op_name}' shape {missing_dtype_shape} has no matching "
            f"dtype {dtype} in baseline"
        )

    return None, (
        f"operator '{op_name}' has no matching shape/dtype in baseline "
        f"(shape_detail={shape_detail}, dtype={dtype})"
    )


def lookup_scaling_factor(
    base_data: Mapping[str, Any],
    vendor: str,
    bottle_neck_unit: str,
    dtype: str,
    self_chip: str,
) -> float | None:
    """Return this chip's measured or nominal factor for the NV bottleneck."""
    scaling = base_data.get("_scaling_factors")
    resource = _BOTTLENECK_TO_SF.get(bottle_neck_unit)
    if not isinstance(scaling, Mapping) or resource is None:
        return None
    chips = scaling.get("chips")
    if not isinstance(chips, list):
        return None
    candidates = [
        chip for chip in chips
        if isinstance(chip, Mapping) and chip.get("vendor") == vendor.lower()
    ]
    chip = next(
        (
            item for item in candidates
            if item.get("chip") and str(item["chip"]).lower() in self_chip.lower()
        ),
        None,
    )
    if chip is None:
        chip = candidates[0] if len(candidates) == 1 else None
    if chip is None:
        return None
    for source in ("measured", "nominal"):
        block = chip.get(source)
        if not isinstance(block, Mapping):
            continue
        value = block.get(resource)
        if resource != "bandwidth_gbps":
            value = (
                value.get(_DTYPE_TO_SF_KEY.get(str(dtype)))
                if isinstance(value, Mapping)
                else None
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    return None
