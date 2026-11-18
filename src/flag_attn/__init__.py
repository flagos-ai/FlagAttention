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
from flag_attn import testing  # noqa: F401

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

_OPERATOR_EXPORTS = {
    "piecewise_attention": ("flag_attn.piecewise", "attention"),
    "flash_attention": ("flag_attn.flash", "attention"),
    "flash_attention_split_kv": ("flag_attn.split_kv", "attention"),
    "paged_attention": ("flag_attn.paged", "attention"),
    "sage_attention": ("flag_attn.sage_attention.attn_qk_int8_per_block", "forward"),
    "parallel_nsa": ("flag_attn.parallel_nsa", "parallel_nsa"),
    "parallel_nsa_compression": ("flag_attn.parallel_nsa.parallel_nsa_compression", "parallel_nsa_compression"),
    "parallel_moba": ("flag_attn.FLA.moba.parallel", "parallel_moba"),
    "parallel_forgetting_attn": ("flag_attn.forgetting_attention", "parallel_forgetting_attn"),
    "log_linear_attn": ("flag_attn.FLA.log_linear_attn.chunk_tle", "log_linear_attn"),
    "diffkv_attention": ("flag_attn.diffkv_attention.api", "diffkv_attention"),
    "hy3_attention": ("flag_attn.hpc_ops_attention", "hy3_attention"),
    "parallel_parallax": ("flag_attn.FLA.parallax", "parallel_parallax"),
    "fused_attnres": ("flag_attn.FLA.attnres", "fused_attnres"),
    "inkling_fa4_rel_attention": (
        "flag_attn.inkling_fa4",
        "inkling_fa4_rel_attention",
    ),
    "chunk_log_linear_attn": (
        "flag_attn.FLA.log_linear_attn",
        "chunk_log_linear_attn",
    ),
    "chunk_gated_delta_rule": (
        "flag_attn.FLA.gated_delta_rule",
        "chunk_gated_delta_rule",
    ),
    "chunk_gla": (
        "flag_attn.FLA.gated_linear_attention.chunk_gla",
        "chunk_gla",
    ),
    "chunk_gdn2": ("flag_attn.gdn2.chunk", "chunk_gdn2"),
    "chunk_kda": ("flag_attn.FLA.chunk_kda", "chunk_kda_fwd_infer"),
    "parallel_wall_attn": ("flag_attn.FLA.wall_attn", "parallel_wall_attn"),
    "InfLLMV2Config": (
        "flag_attn.infllmv2",
        "InfLLMV2Config",
    ),
    "infllmv2_attention": (
        "flag_attn.infllmv2",
        "infllmv2_attention",
    ),
    "infllmv2_decode": (
        "flag_attn.infllmv2",
        "infllmv2_decode",
    ),
}

for _name in (
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
):
    _OPERATOR_EXPORTS[_name] = (
        "flag_attn.minimax_sparse_attention.index_topk"
        if "index" in _name
        else "flag_attn.minimax_sparse_attention"
        if _name == "minimax_m3_sparse_attn"
        else "flag_attn.minimax_sparse_attention.sparse_attn",
        _name,
    )


# Initialize only lightweight namespaces before publishing same-named APIs.
# Later submodule imports then cannot replace these callable exports with
# package objects. Implementations remain lazy inside each namespace.
for _package_name in ("sage_attention", "parallel_nsa", "diffkv_attention"):
    importlib.import_module(f"{__name__}.{_package_name}")
    globals().pop(_package_name, None)


_STAGE_FACADES = {"minimax_m3_sparse_attn", "infllmv2_attention", "parallel_parallax", "hy3_attention", "parallel_nsa"}


def __getattr__(name: str):
    """Load optional attention kernels only when their public API is used."""
    try:
        module_name, attribute_name = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    if name in _STAGE_FACADES:
        value = getattr(importlib.import_module(module_name), attribute_name)
    else:
        operator = "chunk_kda_fwd_infer" if name == "chunk_kda" else name
        value = runtime.backend.resolve_operator(operator, module_name, attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "sage_attention",
    "parallel_nsa",
    "parallel_nsa_compression",
    "parallel_moba",
    "parallel_forgetting_attn",
    "log_linear_attn",
    "diffkv_attention",
    "hy3_attention",
    "parallel_parallax",
    "fused_attnres",
    "inkling_fa4_rel_attention",
    "device",
    "vendor",
    "vendor_name",
    "backend_info",
    "piecewise_attention",
    "flash_attention",
    "flash_attention_split_kv",
    "paged_attention",
    "chunk_log_linear_attn",
    "chunk_gated_delta_rule",
    "chunk_gla",
    "chunk_gdn2",
    "chunk_kda",
    "InfLLMV2Config",
    "infllmv2_attention",
    "infllmv2_decode",
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "parallel_wall_attn",
]
