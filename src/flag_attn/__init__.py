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

from importlib import import_module

from flag_attn import runtime
from flag_attn import testing  # noqa: F401

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

# Entries are (module, attribute, backend operator). Omitted fields default
# to the public name. A None backend operator lets the entry handle dispatch.
_OPERATOR_EXPORTS = {
    "chunk_gated_delta_rule_fwd": ("FLA.gated_delta_rule.chunk",),
    "fused_recurrent_gated_delta_rule_packed_decode": ("FLA.gated_delta_rule.packed_decode",),
    "fused_recurrent_gated_delta_rule_fwd": ("FLA.gated_delta_rule.fused_recurrent",),
    "sage_attention": ("sage_attention.attn_qk_int8_per_block", "forward"),
    "parallel_nsa": ("parallel_nsa", "parallel_nsa", None),
    "parallel_nsa_compression": ("parallel_nsa.parallel_nsa_compression",),
    "parallel_moba": ("FLA.moba.parallel",),
    "parallel_forgetting_attn": ("forgetting_attention.parallel",),
    "log_linear_attn": ("FLA.log_linear_attn.chunk_tle",),
    "diffkv_attention": ("diffkv_attention.api",),
    "hy3_attention": ("hpc_ops_attention", "hy3_attention", None),
    "hy3_attention_prepare": ("hpc_ops_attention", "hy3_attention_prepare", None),
    "parallel_parallax": ("FLA.parallax", "parallel_parallax", None),
    "fused_attnres": ("FLA.attnres",),
    "inkling_fa4_rel_attention": ("inkling_fa4",),
    "piecewise_attention": ("piecewise", "attention"),
    "flash_attention": ("flash", "attention"),
    "flash_attention_split_kv": ("split_kv", "attention"),
    "paged_attention": ("paged", "attention"),
    "chunk_log_linear_attn": ("FLA.log_linear_attn",),
    "chunk_gated_delta_rule": ("FLA.gated_delta_rule",),
    "chunk_gla": ("FLA.gated_linear_attention.chunk_gla",),
    "chunk_gdn2": ("gdn2.chunk",),
    "chunk_kda": ("FLA.chunk_kda", "chunk_kda_fwd_infer", "chunk_kda_fwd_infer"),
    "InfLLMV2Config": ("infllmv2",),
    "infllmv2_attention": ("infllmv2", "infllmv2_attention", None),
    "infllmv2_decode": ("infllmv2",),
    "minimax_m3_index_decode": ("minimax_sparse_attention.index_topk",),
    "minimax_m3_index_decode_score": ("minimax_sparse_attention.index_topk",),
    "minimax_m3_index_score": ("minimax_sparse_attention.index_topk",),
    "minimax_m3_index_topk": ("minimax_sparse_attention.index_topk",),
    "minimax_m3_sparse_attn": ("minimax_sparse_attention", "minimax_m3_sparse_attn", None),
    "minimax_m3_sparse_attn_decode": ("minimax_sparse_attention.sparse_attn",),
    "parallel_wall_attn": ("FLA.wall_attn",),
    "attention_with_kvcache_blocksparse_prefill_fp8": (
        "hpc_ops_attention.prefill.attention_blocksparse_prefill_fp8",
        "attention_with_kvcache_blocksparse_prefill_fp8",
        "hy3_attention",
    ),
    "attention_decode_bf16_static": (
        "hpc_ops_attention.decode.static.bf16_static",
        "attention_decode_bf16_static",
        "hy3_attention_decode_bf16_static",
    ),
    "attention_decode_bf16_dynamic": (
        "hpc_ops_attention.decode.dynamic.bf16_dynamic",
        "attention_decode_bf16_dynamic",
        "hy3_attention_decode_bf16_dynamic",
    ),
    "attention_decode_fp8": (
        "hpc_ops_attention.decode.static.fp8_qkpertoken_perhead_vperhead_static",
        "attention_decode_fp8",
        "hy3_attention_decode_fp8_qk_static",
    ),
    "chunk_kda_fwd_infer": ("FLA.chunk_kda",),
    "forgetting_attention": ("forgetting_attention.parallel", "forgetting_attention", "parallel_forgetting_attn"),
    "parallax_decode": ("FLA.parallax.decode",),
    "parallax_attn_with_kvcache": ("FLA.parallax.decode",),
}


def __getattr__(name: str):
    """Load optional attention kernels only when their public API is used."""
    try:
        module, *options = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = f"{__name__}.{module}"
    attribute = options[0] if options else name
    operator = options[1] if len(options) > 1 else name
    if operator is None:
        value = getattr(import_module(module), attribute)
    else:
        value = runtime.backend.resolve_operator(operator, module, attribute)
    globals()[name] = value
    return value


# Reserve public names before Python binds their same-named subpackages.
for _name, (_module, *_) in _OPERATOR_EXPORTS.items():
    if _module.partition(".")[0] == _name:
        import_module(f".{_name}", __name__)
        globals().pop(_name, None)
        # Cache the package-level wrapper before a nested import can shadow it.
        if _module == _name:
            __getattr__(_name)


__all__ = [*_OPERATOR_EXPORTS, "device", "vendor", "vendor_name", "backend_info"]
