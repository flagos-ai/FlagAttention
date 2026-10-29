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

"""Ascend HY3 decode, split by the leaf format the KV cache holds.

``bf16`` is the BF16 implementation, ``fp8`` the FP8 one, ``mtp_reduce`` the
split reduction chain both of them launch at MTP>=2, and ``launch`` the host-side
launch fast path the FP8 entry uses.  This package re-exports the BF16 entries so
that the ``static`` and ``dynamic`` wrappers keep importing them from here.
"""

import importlib

# name -> (module relative to this package, attribute; None binds the module)
_OPERATOR_EXPORTS = {
    "HAS_TLE": (".bf16", "HAS_TLE"),
    "USE_TLE": (".bf16", "USE_TLE"),
    "AscendBF16MTP1Workspace": (".bf16", "AscendBF16MTP1Workspace"),
    "attention_decode_ascend_bf16_mtp1": (".bf16", "attention_decode_ascend_bf16_mtp1"),
    "attention_decode_ascend_bf16_mtp2": (".bf16", "attention_decode_ascend_bf16_mtp2"),
    "attention_decode_ascend_bf16_mtp3": (".bf16", "attention_decode_ascend_bf16_mtp3"),
    "attention_decode_ascend_bf16_mtp4": (".bf16", "attention_decode_ascend_bf16_mtp4"),
    "prepare_ascend_bf16_mtp1_workspace": (".bf16", "prepare_ascend_bf16_mtp1_workspace"),
    "prepare_ascend_bf16_mtp2_workspace": (".bf16", "prepare_ascend_bf16_mtp2_workspace"),
    "prepare_ascend_bf16_mtp3_workspace": (".bf16", "prepare_ascend_bf16_mtp3_workspace"),
    "prepare_ascend_bf16_mtp4_workspace": (".bf16", "prepare_ascend_bf16_mtp4_workspace"),
    "refresh_ascend_bf16_mtp1_task_map": (".bf16", "refresh_ascend_bf16_mtp1_task_map"),
}

__all__ = sorted(_OPERATOR_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name, __name__)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value
