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

"""NVIDIA attention backend.

The generic implementations under ``src/flag_attn`` are written against NVIDIA
and use plain Triton only, so this package declares no vendor-wide overrides.
What is specific to one GPU generation lives in a subpackage instead:

- ``hopper``: SM90 kernels using Triton Language Extensions (warp
  specialization, wgmma, TMA descriptors).
- ``blackwell``: SM100/SM120 kernels.

``op_registry`` prefers the running generation's exports over this package's,
and this package's over the generic implementation, so a kernel written for one
generation never runs on another. ``tune_configs.yaml`` holds the autotuning
configurations shared by every NVIDIA generation.
"""

