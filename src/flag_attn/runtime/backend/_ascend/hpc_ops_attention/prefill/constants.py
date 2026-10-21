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

"""Tunables shared by the Ascend HY3 FP8 block-sparse prefill kernels.

Every value here is a compile-time constant of at least one kernel, so it is
imported into the module that defines the kernel rather than passed around at
run time.  The values and their provenance comments are unchanged.
"""

import triton.language as tl


_LOGICAL_BLOCK_M = 128
_BLOCK_M = 8
_BLOCK_N = 128
_HEAD_DIM = 128
_FP8_P_SCALE = 256.0
_PREFILL_Q_TOKENS = 4
_PREFILL_TOKENS_PER_SPLIT = 1024  # legacy: disabled kernel variants only
_PREFILL_TOKENS_PER_SPLIT_MAX = 8192
# bf16 scores row stride = tokens_per_split * 2 bytes. A 256B-row store at a
# 4KB-multiple row stride degrades on this backend (measured 8192B->546 GB/s,
# 16384B->282, 32768B->169; non-4KB-multiple 16640B/17408B in the same family
# still reach 520+). So for a 4KB-multiple stride pad each scores row by 128
# elements (256B); probabilities **must not** be padded (pv p load degrades).
_SCORES_ROW_PAD = 128
# One softmax program covers (row, HEADS_PER_PROG heads) over **all** splits of
# a stage: programs drop from rows*stage_slots*heads to rows*heads/hpp, since
# softmax time is mostly program count x fixed overhead (ablation). A switch
# here once allowed the old per-split kernel, but it kept q-major addressing
# after head-major, crashing out of bounds; deleted, split-loop is only path.
_PREFILL_STAGE_CAP = 8      # how many splits the scores buffer holds in loop mode
# benchmark only: False forces the masked path for A/B
# bisect only: independently toggle unmasked qk / pv
_PREFILL_UNMASK_SM = True
# tile must be a multiple of the qk/pv inner block (_BLOCK_N)
_TILE_STEP = 128
_PREFILL_BLOCK_M = 128
# qk's M tile is decoupled from softmax/pv: SCORES indexes absolute q rows, so
# qk can use a larger M tile while softmax/pv blocks by 128 rows. Measured
# q=512/kv=32768/32q4kv: qk 128 rows 6.17ms -> 256 rows 5.67ms (-8%, programs
# 1024 -> 512, fixed cost amortised); 512 rows overflows UB (q + scores >192KB).
_PREFILL_QK_BLOCK_M = 256
# With a mask use a 128-row tile: skipping is decided per 128 rows, raising the
# rate from 54~56% to 74~75% (q=512/kv=32768 whole op +5.5 points, bit-exact).
_PREFILL_QK_BLOCK_M_MASKED = 128
# Min kv tiles for the list path (launcher crossover table); below it use dense.
_PREFILL_LIST_MIN_KV_TILES = 12
_PREFILL_VEC_ROWS = 1
# Q heads per softmax program. Short-case softmax is mostly program slots x fixed
# overhead (measured ~42ns/empty slot, 76ns/slot at 512/4096), so serialising HPP
# heads per program amortises it. An elements-per-program gate once limited it to
# small cases, but the head loop was fully unrolled then; round 11 made it runtime
# and opening to num_head_q (cap 32) gained on **all** shapes (long -0.9%,
# 512/4096 -4.5%, (127,127) -19%, (9,129) -40%); gate and constant deleted.
_PREFILL_HEADS_PER_PROG_MAX = 32
_PREFILL_STAGE_SLOTS = 2
# Decode path: arithmetic E4M3 decode (integer bit ops, no table) beats LUT:
# q=512/kv=4096 (64 pages) KV 1038us -> 56us (18x), Q 267us -> 23us (11x),
# output **bit-identical**, 22 FP8 prefill tests green; both default to arithmetic.
# The override switches below are benchmark/control only (None = default above).
# Extra pages per program for long KV just cap the grid (ppp=1/4 decode kernels
# are equally fast, 55.4~58.2us vs 56.4~56.9us); short KV keeps ppp=1.
_ARITHMETIC_KV_PAGES_MAX = 8
_ARITHMETIC_KV_PAGES_PER_PROGRAM = 1
# decode_kv head-dim blocking is decoupled from decode_q: 64 beats 32 by 12~23%
# (tile becomes 64x64, store row width 128B instead of 64B; d_block=128 overflows
# UB), and pages_per_program=1 beats 4. The loop below caps the grid at 65535.
_ARITHMETIC_KV_D_BLOCK = 64

_BLOCK_M_JIT = tl.constexpr(_BLOCK_M)
_BLOCK_N_JIT = tl.constexpr(_BLOCK_N)
_HEAD_DIM_JIT = tl.constexpr(_HEAD_DIM)
_LOGICAL_BLOCK_M_JIT = tl.constexpr(_LOGICAL_BLOCK_M)
_FP8_P_SCALE_JIT = tl.constexpr(_FP8_P_SCALE)
_PREFILL_TOKENS_PER_SPLIT_JIT = tl.constexpr(_PREFILL_TOKENS_PER_SPLIT)
_PREFILL_BLOCK_M_JIT = tl.constexpr(_PREFILL_BLOCK_M)
_PREFILL_VEC_ROWS_JIT = tl.constexpr(_PREFILL_VEC_ROWS)
