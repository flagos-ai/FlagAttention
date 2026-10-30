// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "Utils.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)
#define INTRINSIC_NO_ARGS(NAME) NAME()

// Experimental single-AIV triangular inverse.  Input and output are explicit
// UB buffers owned by the Triton wrapper.  The loop is over matrix rows and
// recurrence terms; each term updates all 32 columns with one vector op.
extern "C" {
__aiv__ void _mlir_ciface_raw_inverse32_diag(
    memref_t<__ubuf__ float, 2> *lower,
    memref_t<__ubuf__ float, 2> *identity,
    memref_t<__ubuf__ float, 2> *zero_offsets,
    memref_t<__ubuf__ float, 2> *inverse) {
  __ubuf__ float *lp = lower->aligned + lower->offset;
  __ubuf__ float *ip = identity->aligned + identity->offset;
  __ubuf__ uint32_t *offsetp =
      (__ubuf__ uint32_t *)(zero_offsets->aligned + zero_offsets->offset);
  __ubuf__ float *op = inverse->aligned + inverse->offset;
  __ubuf__ float *scratch = op + 32 * 32;
  if (lower->sizes[0] != 32 || lower->sizes[1] != 32 ||
      identity->sizes[0] != 32 || identity->sizes[1] != 32 ||
      zero_offsets->sizes[0] != 1 || zero_offsets->sizes[1] != 32 ||
      inverse->sizes[0] != 33 || inverse->sizes[1] != 32 ||
      lower->offset != 0 || identity->offset != 1024 ||
      zero_offsets->offset != 2048 || inverse->offset != 2080 ||
      lower->strides[1] != 1 || identity->strides[1] != 1 ||
      inverse->strides[1] != 1 || lower->strides[0] != 32 ||
      identity->strides[0] != 32 || inverse->strides[0] != 32 ||
      zero_offsets->strides[1] != 1 ||
      lower->offset + 31 * lower->strides[0] + 31 >= identity->offset ||
      identity->offset + 31 * identity->strides[0] + 31 >= zero_offsets->offset ||
      zero_offsets->offset + 31 >= inverse->offset ||
      inverse->offset + 32 * inverse->strides[0] + 31 >= 99 * 32) {
    return;
  }

  // Full write, including upper triangle and diagonal. Keep each row as one
  // explicit 32-element count-mode operation for the tail/mask contract.
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, 32);
  for (int32_t row = 0; row < 32; ++row) {
    INTRINSIC(vadds, op + row * 32, ip + row * 32, 0.0f,
              1, 1, 1, 8, 8);
    INTRINSIC(pipe_barrier, PIPE_V);
  }

  // M=I+L, with unit diagonal. For each row i, inv[i,:] starts at e_i and
  // subtracts L[i,k] * inv[k,:] using 32-lane CCEC vector operations.
  for (int32_t row = 1; row < 32; ++row) {
    __ubuf__ float *dst = op + row * 32;
    for (int32_t k = 0; k < row; ++k) {
      const uint32_t base = static_cast<uint32_t>(
          reinterpret_cast<uint64_t>(lp + row * 32 + k));
      // Gather the scalar coefficient into 32 lanes using repeated zero byte
      // offsets; this is a real CCEC vgather, not scalar coefficient math.
      INTRINSIC(vgather, (__ubuf__ uint32_t *)scratch, offsetp, base, 8, 1);
      INTRINSIC(pipe_barrier, PIPE_V);
      INTRINSIC(vmul, scratch, op + k * 32, scratch,
                1, 1, 1, 1, 8, 8, 8);
      INTRINSIC(pipe_barrier, PIPE_V);
      INTRINSIC(vsub, dst, dst, scratch, 1, 1, 1, 1, 8, 8, 8);
      INTRINSIC(pipe_barrier, PIPE_V);
    }
  }
  INTRINSIC_NO_ARGS(set_mask_norm);
  INTRINSIC(set_vector_mask, static_cast<uint64_t>(-1),
            static_cast<uint64_t>(-1));
  INTRINSIC(pipe_barrier, PIPE_V);
}
}
