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

"""Hardware-specific attention operators grouped by execution phase."""

from typing import Literal

from flag_attn.runtime.backend import resolve_operator

from . import decode, prefill

_HY3_DECODE_IMPLEMENTATIONS = {
    "bf16_static": ("static.bf16_static", "attention_decode_bf16_static"),
    "bf16_dynamic": ("dynamic.bf16_dynamic", "attention_decode_bf16_dynamic"),
    "fp8_qk_static": ("static.fp8_qkpertoken_perhead_vperhead_static", "attention_decode_fp8"),
    "fp8_qk_dynamic": ("dynamic.fp8_qkpertoken_perhead_vperhead_dynamic", "attention_decode_fp8"),
    "fp8_kv_static": ("static.fp8_qpertoken_perhead_kvpertensor_static", "attention_decode_fp8"),
    "fp8_kv_dynamic": ("dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic", "attention_decode_fp8"),
}


def hy3_attention(
    *args,
    stage: Literal["prefill", "decode"] = "prefill",
    variant: Literal[
        "bf16_static", "bf16_dynamic", "fp8_qk_static", "fp8_qk_dynamic", "fp8_kv_static", "fp8_kv_dynamic"
    ]
    | None = None,
    **kwargs,
):
    """Run Hy3 prefill or an explicitly selected decode implementation.

    Prefill forwards the tensor/cache/scale arguments to
    prefill.attention_with_kvcache_blocksparse_prefill_fp8; variant must
    be omitted. Decode requires variant and forwards (inputs, workspace)
    unchanged. qk denotes per-token Q/K with per-head V quantization;
    kv denotes per-token Q with per-tensor K/V quantization. Prepare the
    workspace using the matching implementation module before calling.
    """
    if stage == "prefill":
        if variant is not None:
            raise ValueError("Hy3 prefill does not accept a decode variant")
        return resolve_operator(
            "hy3_attention",
            "flag_attn.hpc_ops_attention.prefill.attention_blocksparse_prefill_fp8",
            "attention_with_kvcache_blocksparse_prefill_fp8",
        )(*args, **kwargs)
    if stage != "decode":
        raise ValueError(f"Unsupported Hy3 stage: {stage!r}; expected 'prefill' or 'decode'")
    try:
        module_name, function_name = _HY3_DECODE_IMPLEMENTATIONS[variant]
    except KeyError as exc:
        raise ValueError(
            f"Hy3 decode requires variant in {tuple(_HY3_DECODE_IMPLEMENTATIONS)}; got {variant!r}"
        ) from exc
    implementation = resolve_operator(
        f"hy3_attention_decode_{variant}", f"{__name__}.decode.{module_name}", function_name
    )
    return implementation(*args, **kwargs)


__all__ = ["hy3_attention", "decode", "prefill"]
