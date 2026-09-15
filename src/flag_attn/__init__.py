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

import importlib

from flag_attn import runtime

# Match FlagGems and FlagGems-vllm: the package exposes strings, while
# runtime.device retains the structured vendor/device metadata.
device = runtime.device.name
vendor_name = runtime.device.vendor_name
vendor = vendor_name
backend_info = runtime.device

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "0.0.0"
    version_tuple = (0, 0, 0)


from flag_attn.piecewise import attention as piecewise_attention  # noqa: F401
from flag_attn.flash import attention as flash_attention  # noqa: F401
from flag_attn.split_kv import attention as flash_attention_split_kv  # noqa: F401
from flag_attn.paged import attention as paged_attention  # noqa: F401
from flag_attn import testing  # noqa: F401

_OPERATOR_EXPORTS = {
    "chunk_gated_delta_rule": (
        "flag_attn.FLA.gated_delta_rule",
        "chunk_gated_delta_rule",
    ),
    "chunk_gla": (
        "flag_attn.FLA.gated_linear_attention",
        "chunk_gla",
    ),
    "chunk_gdn2": ("flag_attn.gdn2", "chunk_gdn2"),
    "chunk_kda": ("flag_attn.FLA.chunk_kda", "chunk_kda_fwd_infer"),
}

for _name in (
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
):
    _OPERATOR_EXPORTS[_name] = ("flag_attn.minimax_sparse_attention", _name)

if vendor_name in {"enflame", "metax", "mthreads"}:
    for _name in ("chunk_gdn2", "chunk_kda"):
        _OPERATOR_EXPORTS[_name] = (
            f"flag_attn.runtime.backend._{vendor_name}.FLA.{_name.removeprefix('chunk_')}",
            _name,
        )


def __getattr__(name: str):
    """Load optional attention kernels only when their public API is used."""
    try:
        module_name, attribute_name = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "device",
    "vendor",
    "vendor_name",
    "backend_info",
    "piecewise_attention",
    "flash_attention",
    "flash_attention_split_kv",
    "paged_attention",
    "chunk_gated_delta_rule",
    "chunk_gla",
    "chunk_gdn2",
    "chunk_kda",
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
]
