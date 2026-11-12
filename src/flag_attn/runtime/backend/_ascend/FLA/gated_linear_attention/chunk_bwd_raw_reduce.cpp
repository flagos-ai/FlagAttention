// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "Utils.h"

extern "C" {
// Contiguous [128,64] FP32 state tiles. vcadd writes 128 scalar sums.
__aiv__ void _mlir_ciface_gla_bwd_state_product(
    memref_t<__ubuf__ float, 2>* dh, memref_t<__ubuf__ float, 2>* h,
    memref_t<__ubuf__ float, 2>* out) {
  __ubuf__ float* d = dh->aligned + dh->offset;
  __ubuf__ float* x = h->aligned + h->offset;
  __ubuf__ float* o = out->aligned + out->offset;
  set_mask_count();
  set_vector_mask(0, 8192);
  vmul(x, x, d, 128, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  set_mask_norm();
  set_vector_mask(0, static_cast<uint64_t>(-1));
  vcadd(o, x, 128, 1, 1, 8, 0);
  pipe_barrier(PIPE_V);
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

// In-place inclusive suffix sum of [64,128], with a [1,128] state carry.
__aiv__ void _mlir_ciface_gla_bwd_reverse_sum(
    memref_t<__ubuf__ float, 2>* carry, memref_t<__ubuf__ float, 2>* tile) {
  __ubuf__ float* x = tile->aligned + tile->offset;
  __ubuf__ float* c = carry->aligned + carry->offset;
  set_mask_count();
  set_vector_mask(0, 128);
  vadd(x + 63 * 128, x + 63 * 128, c, 2, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  for (int row = 62; row >= 0; --row) {
    vadd(x + row * 128, x + row * 128, x + (row + 1) * 128, 2, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
  }
  set_mask_norm();
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
  pipe_barrier(PIPE_V);
}
}
