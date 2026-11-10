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

"""Compile-time tile sizes and workspace limits for Ascend HY3 BF16 prefill."""

import triton.language as tl


_LOGICAL_BLOCK_M = 128
_BLOCK_N = 128
_HEAD_DIM = 128
_PREFILL_TOKENS_PER_SPLIT_MAX = 8192
# Pad score rows with 4-KiB-aligned byte strides to avoid aligned-stride
# access penalties. Probability rows stay tightly packed for PV loads.
_SCORES_ROW_PAD = 128
# Number of split slots held in the reusable scores/probabilities stage.
_PREFILL_STAGE_CAP = 8      # how many splits the scores buffer holds in loop mode
# Allow the mask-free softmax path when preparation establishes its preconditions.
_PREFILL_UNMASK_SM = True
# tile must be a multiple of the qk/pv inner block (_BLOCK_N)
_TILE_STEP = 128
_PREFILL_BLOCK_M = 128
# QK tiles can span multiple logical query blocks when no block mask is supplied;
# their row indices remain independent of the softmax/PV tile size.
_PREFILL_QK_BLOCK_M = 256
# Match masked QK tiles to one 128-row logical mask block.
_PREFILL_QK_BLOCK_M_MASKED = 128
# Minimum KV-block count for constructing an active-block list.
_PREFILL_LIST_MIN_KV_TILES = 12
_PREFILL_VEC_ROWS = 1
# Maximum heads processed serially by one softmax program.
_PREFILL_HEADS_PER_PROG_MAX = 32
_BLOCK_N_JIT = tl.constexpr(_BLOCK_N)
_HEAD_DIM_JIT = tl.constexpr(_HEAD_DIM)
_LOGICAL_BLOCK_M_JIT = tl.constexpr(_LOGICAL_BLOCK_M)
_PREFILL_BLOCK_M_JIT = tl.constexpr(_PREFILL_BLOCK_M)
_PREFILL_VEC_ROWS_JIT = tl.constexpr(_PREFILL_VEC_ROWS)
