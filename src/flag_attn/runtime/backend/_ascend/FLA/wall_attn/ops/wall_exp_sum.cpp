// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Read-only scores; distinct bf16 probabilities, fp32 sums and scratch outputs.
// Contract: BS=128, contiguous columns, aligned row bases. No input/output alias.
#include "Utils.h"

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_wall_exp_sum(
    memref_t<__ubuf__ float, 2> *score,
    memref_t<__ubuf__ bfloat16_t, 2> *prob,
    memref_t<__ubuf__ float, 1> *rowsum,
    memref_t<__ubuf__ float, 1> *scratch) {
  auto s = score->aligned + score->offset;
  auto p = prob->aligned + prob->offset;
  auto z = rowsum->aligned + rowsum->offset;
  auto w = scratch->aligned + scratch->offset; // >=128 fp32, 32-byte aligned
  pipe_barrier(PIPE_ALL);
  set_mask_norm();
  set_vector_mask((uint64_t)0, ~(uint64_t)0); // 64 active fp32 lanes
  for (int64_t r = 0; r < score->sizes[0]; ++r) {
    // Unary strides are 32-byte blocks; fp32 repeat = 8 blocks.
    vexp(w, s + r * score->strides[0], 2, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    // 64 fp32 -> 64 bf16 per repeat; dst advances 4 blocks, src 8.
    vconv_f322bf16r(p + r * prob->strides[0], w, 2, 1, 1, 4, 8);
    pipe_barrier(PIPE_V); // finish conversion before overwriting fp32 scratch
    // 128 -> 64 -> 1, entirely vector reduction of fp32 exp values.
    vadd(w, w, w + 64, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcadd(w + 64, w, 1, 1, 1, 8, 0);
    // V->scalar visibility before one scalar read/store PER ROW (not element).
    pipe_barrier(PIPE_ALL);
    z[r * rowsum->strides[0]] = w[64];
    pipe_barrier(PIPE_ALL); // protect scratch reuse and publish scalar store
  }
  set_mask_norm();
  set_vector_mask(~(uint64_t)0, ~(uint64_t)0);
  pipe_barrier(PIPE_ALL);
}
