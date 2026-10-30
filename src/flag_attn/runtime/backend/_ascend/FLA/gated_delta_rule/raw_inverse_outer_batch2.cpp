// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "Utils.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)
#define INTRINSIC_NO_ARGS(NAME) NAME()

// CANN-style cumulative outer-product inversion, expressed only with CCEC
// vector intrinsics. All UB operands are views into one caller-owned backing
// allocation; offsets below are checked against the memref descriptors.
extern "C" {
__aiv__ void _mlir_ciface_raw_inverse32_outer_batch2(
    memref_t<__ubuf__ uint32_t, 2> *lower_u,
    memref_t<__ubuf__ uint32_t, 2> *identity_u,
    memref_t<__ubuf__ uint32_t, 2> *gather_offsets,
    memref_t<__ubuf__ uint32_t, 2> *inverse_u) {
  __ubuf__ float *lower = (__ubuf__ float *)(lower_u->aligned + lower_u->offset);
  __ubuf__ float *identity =
      (__ubuf__ float *)(identity_u->aligned + identity_u->offset);
  __ubuf__ uint32_t *offsets = gather_offsets->aligned + gather_offsets->offset;
  __ubuf__ float *inverse =
      (__ubuf__ float *)(inverse_u->aligned + inverse_u->offset);
  // The output is the [33,32] inverse view at element offset 2080. The parent
  // [203,32] UB allocation has an explicitly reserved work tail directly after
  // this output view; it is partitioned here instead of exposed as another
  // Raw output (which would trigger multi-output alias planning).
  __ubuf__ float *work = inverse + 33 * 32;

  if (lower_u->sizes[0] != 32 || lower_u->sizes[1] != 32 ||
      identity_u->sizes[0] != 32 || identity_u->sizes[1] != 32 ||
      gather_offsets->sizes[0] != 1 || gather_offsets->sizes[1] != 32 ||
      inverse_u->sizes[0] != 33 || inverse_u->sizes[1] != 32 ||
      lower_u->offset != 0 || identity_u->offset != 1024 ||
      gather_offsets->offset != 2048 || inverse_u->offset != 2080 ||
      lower_u->strides[0] != 32 || lower_u->strides[1] != 1 ||
      identity_u->strides[0] != 32 || identity_u->strides[1] != 1 ||
      gather_offsets->strides[1] != 1 ||
      inverse_u->strides[0] != 32 || inverse_u->strides[1] != 1 ||
      lower_u->offset + 1024 > identity_u->offset ||
      identity_u->offset + 1024 > gather_offsets->offset ||
      gather_offsets->offset + 32 > inverse_u->offset ||
      inverse_u->offset + 1056 > 3136 ||
      3136 + 105 * 32 > 203 * 32) {
    return;
  }

  // One backing region work layout: accum[32,32], column[32],
  // column_brcb[32,8], row_broadcast[32,32], column_broadcast[32,32].
  __ubuf__ float *accum = work;
  __ubuf__ float *column = work + 32 * 32;
  __ubuf__ float *column_brcb = work + 33 * 32;
  __ubuf__ float *row_broadcast = work + 41 * 32;
  __ubuf__ float *column_broadcast = work + 73 * 32;

  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, 32);
  for (int32_t row = 0; row < 32; ++row) {
    INTRINSIC(vector_dup, accum + row * 32, 0.0f, 1, 1, 0, 8, 0);
  }

  // Initialize the full 32x32 result, including upper triangle and diagonal.
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, 32);
  for (int32_t row = 0; row < 32; ++row) {
    INTRINSIC(vadds, inverse + row * 32, identity + row * 32, 0.0f,
              1, 1, 1, 8, 8);
  }
  // Initialization writes independent rows; one barrier makes both complete
  // before the recurrence starts reading accumulator and inverse rows.
  INTRINSIC(pipe_barrier, PIPE_V);

  // The CANN recurrence accumulates each solved inverse row into all future
  // rows: accum[i:, :] += L[i:, i-1] * inverse[i-1, :].
  for (int32_t i = 1; i < 32; ++i) {
    const int32_t cur_i = i - 1;
    const int32_t valid_rows = 32 - i;
    const uint32_t base = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(lower + i * 32 + cur_i));

    // CANN Gather: 32-byte byte offsets select the strided lower-triangle
    // column into a contiguous vector. The count mask writes only valid rows.
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, valid_rows);
    INTRINSIC(vgather, (__ubuf__ uint32_t *)column, offsets, base, 8, 1);
    INTRINSIC(pipe_barrier, PIPE_V);

    // Broadcast col[:,None] to [valid_rows,32] via vbrcb (8 lanes) then four
    // strided vcopy block passes. Broadcast inverse[i-1,:] over rows with the
    // same real vector-copy primitive and zero source repeat stride.
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(vbrcb, (__ubuf__ uint32_t *)column_brcb,
              (__ubuf__ uint32_t *)column, 1, 8, CEIL_DIV(valid_rows, 8));
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC(set_vector_mask, 0, 8);
    for (int32_t group = 0; group < 4; ++group) {
      INTRINSIC(vcopy, (__ubuf__ uint32_t *)(column_broadcast + i * 32 + group * 8),
                (__ubuf__ uint32_t *)column_brcb,
                valid_rows, 1, 0, 4, 1);
      INTRINSIC(vcopy, (__ubuf__ uint32_t *)(row_broadcast + i * 32 + group * 8),
                (__ubuf__ uint32_t *)(inverse + cur_i * 32 + group * 8),
                valid_rows, 1, 1, 4, 0);
    }
    INTRINSIC(pipe_barrier, PIPE_V);

    // The CANN C-API vmadd contract is dst = src0 * dst + src1, so it cannot
    // directly express accum += column * row. Reuse the no-longer-needed
    // broadcast-column slot for the product and add it to accum explicitly.
    const int32_t count = valid_rows * 32;
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, count);
    INTRINSIC(vmul, column_broadcast + i * 32, column_broadcast + i * 32,
              row_broadcast + i * 32, CEIL_DIV(count, 64),
              1, 1, 1, 8, 8, 8);
    INTRINSIC(pipe_barrier, PIPE_V);
    INTRINSIC(vadd, accum + i * 32, accum + i * 32,
              column_broadcast + i * 32, CEIL_DIV(count, 64),
              1, 1, 1, 8, 8, 8);
    INTRINSIC(pipe_barrier, PIPE_V);

    // Solve the newly accumulated row: inv[i,:] = e_i - accum[i,:].
    INTRINSIC(set_vector_mask, 0, 32);
    INTRINSIC(vsub, inverse + i * 32, identity + i * 32,
              accum + i * 32, 1, 1, 1, 1, 8, 8, 8);
    INTRINSIC(pipe_barrier, PIPE_V);
  }

  // Fully write the packed padding row as well, so the returned ABI operand
  // never exposes poisoned or stale UB contents.
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, 32);
  INTRINSIC(set_vector_mask, 0, 32);
  INTRINSIC(vadds, inverse + 32 * 32, accum + 32,
            0.0f, 1, 1, 1, 8, 8);
  INTRINSIC_NO_ARGS(set_mask_norm);
  INTRINSIC(set_vector_mask, static_cast<uint64_t>(-1),
            static_cast<uint64_t>(-1));
  INTRINSIC(pipe_barrier, PIPE_V);
}
}
