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

"""Ascend HY3 decode with a task map that can be refreshed between steps."""

import importlib

# name -> (module relative to this package, attribute; None binds the module)
_OPERATOR_EXPORTS = {
    "BLOCK_SIZE": (".bf16_dynamic", "BLOCK_SIZE"),
    "HEAD_DIM": (".bf16_dynamic", "HEAD_DIM"),
    "DynamicBF16Inputs": (".bf16_dynamic", "DynamicBF16Inputs"),
    "DynamicBF16Workspace": (".bf16_dynamic", "DynamicBF16Workspace"),
    "attention_decode_bf16_dynamic": (".bf16_dynamic", "attention_decode_bf16_dynamic"),
    "bf16_dynamic_workspace_is_reset": (".bf16_dynamic", "bf16_dynamic_workspace_is_reset"),
    "prepare_dynamic_bf16_workspace": (".bf16_dynamic", "prepare_dynamic_bf16_workspace"),
    "refresh_dynamic_bf16_task_map": (".bf16_dynamic", "refresh_dynamic_bf16_task_map"),
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
