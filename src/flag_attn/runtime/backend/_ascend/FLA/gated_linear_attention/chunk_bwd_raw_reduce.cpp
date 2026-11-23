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

// Independent row groups share one barrier per tree level: 6 up, 5 down,
// then one broadcast carry. No extra UB tile or GM traffic is required.
__aiv__ void _mlir_ciface_gla_bwd_reverse_tree(
    memref_t<__ubuf__ float, 2>* carry, memref_t<__ubuf__ float, 2>* tile) {
  __ubuf__ float* x = tile->aligned + tile->offset;
  __ubuf__ float* c = carry->aligned + carry->offset;
  for (int d = 1; d <= 32; d *= 2) {
    int groups = 64 / (2 * d);
    if (d <= 4) {
      set_mask_norm();
      set_vector_mask(0, static_cast<uint64_t>(-1));
      vadd(x, x, x + d * 128, groups, 1, 1, 1, 32 * d, 32 * d, 32 * d);
      vadd(x + 64, x + 64, x + d * 128 + 64, groups, 1, 1, 1, 32 * d, 32 * d, 32 * d);
    } else {
      // Larger row strides exceed the repeat-stride field of a vector op.
      set_mask_count();
      set_vector_mask(0, 128);
      for (int j = 0; j < groups; ++j) {
        int row = j * 2 * d;
        vadd(x + row * 128, x + row * 128, x + (row + d) * 128, 2, 1, 1, 1, 8, 8, 8);
      }
    }
    pipe_barrier(PIPE_V);
  }
  for (int d = 16; d >= 1; d /= 2) {
    int groups = 64 / (2 * d) - 1;
    if (d <= 4) {
      set_mask_norm();
      set_vector_mask(0, static_cast<uint64_t>(-1));
      vadd(x + d * 128, x + d * 128, x + 2 * d * 128, groups, 1, 1, 1, 32 * d, 32 * d, 32 * d);
      vadd(x + d * 128 + 64, x + d * 128 + 64, x + 2 * d * 128 + 64,
           groups, 1, 1, 1, 32 * d, 32 * d, 32 * d);
    } else {
      set_mask_count();
      set_vector_mask(0, 128);
      for (int j = 0; j < groups; ++j) {
        int row = j * 2 * d + d;
        vadd(x + row * 128, x + row * 128, x + (row + d) * 128, 2, 1, 1, 1, 8, 8, 8);
      }
    }
    pipe_barrier(PIPE_V);
  }
  set_mask_norm();
  set_vector_mask(0, static_cast<uint64_t>(-1));
  vadd(x, x, c, 64, 1, 1, 1, 16, 16, 0);
  vadd(x + 64, x + 64, c + 64, 64, 1, 1, 1, 16, 16, 0);
  pipe_barrier(PIPE_V);
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
}
