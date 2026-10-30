// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "Utils.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)
#define INTRINSIC_NO_ARGS(NAME) NAME()

// Advance two independent 32x32 lower-triangular Raw inverses at the same
// recurrence index. All six operands and the two work sets are disjoint views
// into one explicit caller-owned UB arena.
extern "C" {
__aiv__ void _mlir_ciface_raw_inverse32_outer_dual(
    memref_t<__ubuf__ uint32_t, 2> *lower_a_u,
    memref_t<__ubuf__ uint32_t, 2> *lower_d_u,
    memref_t<__ubuf__ uint32_t, 2> *identity_a_u,
    memref_t<__ubuf__ uint32_t, 2> *identity_d_u,
    memref_t<__ubuf__ uint32_t, 2> *gather_offsets_u,
    memref_t<__ubuf__ uint32_t, 2> *inverse_pair_u) {
  __ubuf__ float *lower_a =
      (__ubuf__ float *)(lower_a_u->aligned + lower_a_u->offset);
  __ubuf__ float *lower_d =
      (__ubuf__ float *)(lower_d_u->aligned + lower_d_u->offset);
  __ubuf__ float *identity_a =
      (__ubuf__ float *)(identity_a_u->aligned + identity_a_u->offset);
  __ubuf__ float *identity_d =
      (__ubuf__ float *)(identity_d_u->aligned + identity_d_u->offset);
  __ubuf__ uint32_t *offsets =
      gather_offsets_u->aligned + gather_offsets_u->offset;
  __ubuf__ float *inverse_pair =
      (__ubuf__ float *)(inverse_pair_u->aligned + inverse_pair_u->offset);
  __ubuf__ float *inverse_a = inverse_pair;
  __ubuf__ float *inverse_d = inverse_pair + 33 * 32;

  if (lower_a_u->sizes[0] != 32 || lower_a_u->sizes[1] != 32 ||
      lower_d_u->sizes[0] != 32 || lower_d_u->sizes[1] != 32 ||
      identity_a_u->sizes[0] != 32 || identity_a_u->sizes[1] != 32 ||
      identity_d_u->sizes[0] != 32 || identity_d_u->sizes[1] != 32 ||
      gather_offsets_u->sizes[0] != 1 || gather_offsets_u->sizes[1] != 32 ||
      inverse_pair_u->sizes[0] != 66 || inverse_pair_u->sizes[1] != 32 ||
      lower_a_u->offset != 0 || lower_d_u->offset != 1024 ||
      identity_a_u->offset != 2048 || identity_d_u->offset != 3072 ||
      gather_offsets_u->offset != 4096 || inverse_pair_u->offset != 4128 ||
      lower_a_u->strides[0] != 32 || lower_a_u->strides[1] != 1 ||
      lower_d_u->strides[0] != 32 || lower_d_u->strides[1] != 1 ||
      identity_a_u->strides[0] != 32 || identity_a_u->strides[1] != 1 ||
      identity_d_u->strides[0] != 32 || identity_d_u->strides[1] != 1 ||
      gather_offsets_u->strides[1] != 1 ||
      inverse_pair_u->strides[0] != 32 || inverse_pair_u->strides[1] != 1 ||
      lower_a_u->offset + 1024 > lower_d_u->offset ||
      lower_d_u->offset + 1024 > identity_a_u->offset ||
      identity_a_u->offset + 1024 > identity_d_u->offset ||
      identity_d_u->offset + 1024 > gather_offsets_u->offset ||
      gather_offsets_u->offset + 32 > inverse_pair_u->offset ||
      inverse_pair_u->offset + 66 * 32 > 6240 ||
      6240 + 2 * 105 * 32 > 405 * 32) {
    return;
  }

  __ubuf__ float *work_a = inverse_d + 33 * 32;
  __ubuf__ float *work_d = work_a + 105 * 32;
  __ubuf__ float *accum_a = work_a;
  __ubuf__ float *column_a = work_a + 32 * 32;
  __ubuf__ float *column_brcb_a = work_a + 33 * 32;
  __ubuf__ float *row_broadcast_a = work_a + 41 * 32;
  __ubuf__ float *column_broadcast_a = work_a + 73 * 32;
  __ubuf__ float *accum_d = work_d;
  __ubuf__ float *column_d = work_d + 32 * 32;
  __ubuf__ float *column_brcb_d = work_d + 33 * 32;
  __ubuf__ float *row_broadcast_d = work_d + 41 * 32;
  __ubuf__ float *column_broadcast_d = work_d + 73 * 32;

  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, 32);
  for (int32_t row = 0; row < 32; ++row) {
    INTRINSIC(vector_dup, accum_a + row * 32, 0.0f, 1, 1, 0, 8, 0);
    INTRINSIC(vector_dup, accum_d + row * 32, 0.0f, 1, 1, 0, 8, 0);
  }
  for (int32_t row = 0; row < 32; ++row) {
    INTRINSIC(vadds, inverse_a + row * 32, identity_a + row * 32, 0.0f,
              1, 1, 1, 8, 8);
    INTRINSIC(vadds, inverse_d + row * 32, identity_d + row * 32, 0.0f,
              1, 1, 1, 8, 8);
  }
  INTRINSIC(pipe_barrier, PIPE_V);

  // Same-block recurrence dependencies remain serialized. The A and D
  // producer/consumer operations are independent and share only barriers.
  for (int32_t i = 1; i < 32; ++i) {
    const int32_t cur_i = i - 1;
    const int32_t valid_rows = 32 - i;
    const uint32_t base_a = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(lower_a + i * 32 + cur_i));
    const uint32_t base_d = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(lower_d + i * 32 + cur_i));

    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, valid_rows);
    INTRINSIC(vgather, (__ubuf__ uint32_t *)column_a, offsets, base_a, 8, 1);
    INTRINSIC(vgather, (__ubuf__ uint32_t *)column_d, offsets, base_d, 8, 1);
    INTRINSIC(pipe_barrier, PIPE_V);

    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(vbrcb, (__ubuf__ uint32_t *)column_brcb_a,
              (__ubuf__ uint32_t *)column_a, 1, 8, CEIL_DIV(valid_rows, 8));
    INTRINSIC(vbrcb, (__ubuf__ uint32_t *)column_brcb_d,
              (__ubuf__ uint32_t *)column_d, 1, 8, CEIL_DIV(valid_rows, 8));
    INTRINSIC(pipe_barrier, PIPE_V);

    INTRINSIC(set_vector_mask, 0, 8);
    for (int32_t group = 0; group < 4; ++group) {
      INTRINSIC(vcopy,
                (__ubuf__ uint32_t *)(column_broadcast_a + i * 32 + group * 8),
                (__ubuf__ uint32_t *)column_brcb_a,
                valid_rows, 1, 0, 4, 1);
      INTRINSIC(vcopy,
                (__ubuf__ uint32_t *)(row_broadcast_a + i * 32 + group * 8),
                (__ubuf__ uint32_t *)(inverse_a + cur_i * 32 + group * 8),
                valid_rows, 1, 1, 4, 0);
      INTRINSIC(vcopy,
                (__ubuf__ uint32_t *)(column_broadcast_d + i * 32 + group * 8),
                (__ubuf__ uint32_t *)column_brcb_d,
                valid_rows, 1, 0, 4, 1);
      INTRINSIC(vcopy,
                (__ubuf__ uint32_t *)(row_broadcast_d + i * 32 + group * 8),
                (__ubuf__ uint32_t *)(inverse_d + cur_i * 32 + group * 8),
                valid_rows, 1, 1, 4, 0);
    }
    INTRINSIC(pipe_barrier, PIPE_V);

    const int32_t count = valid_rows * 32;
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, count);
    INTRINSIC(vmul, column_broadcast_a + i * 32,
              column_broadcast_a + i * 32, row_broadcast_a + i * 32,
              CEIL_DIV(count, 64), 1, 1, 1, 8, 8, 8);
    INTRINSIC(vmul, column_broadcast_d + i * 32,
              column_broadcast_d + i * 32, row_broadcast_d + i * 32,
              CEIL_DIV(count, 64), 1, 1, 1, 8, 8, 8);
    INTRINSIC(pipe_barrier, PIPE_V);

    INTRINSIC(vadd, accum_a + i * 32, accum_a + i * 32,
              column_broadcast_a + i * 32, CEIL_DIV(count, 64),
              1, 1, 1, 8, 8, 8);
    INTRINSIC(vadd, accum_d + i * 32, accum_d + i * 32,
              column_broadcast_d + i * 32, CEIL_DIV(count, 64),
              1, 1, 1, 8, 8, 8);
    INTRINSIC(pipe_barrier, PIPE_V);

    INTRINSIC(set_vector_mask, 0, 32);
    INTRINSIC(vsub, inverse_a + i * 32, identity_a + i * 32,
              accum_a + i * 32, 1, 1, 1, 1, 8, 8, 8);
    INTRINSIC(vsub, inverse_d + i * 32, identity_d + i * 32,
              accum_d + i * 32, 1, 1, 1, 1, 8, 8, 8);
    INTRINSIC(pipe_barrier, PIPE_V);
  }

  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, 32);
  INTRINSIC(vector_dup, inverse_a + 32 * 32, 0.0f, 1, 1, 0, 8, 0);
  INTRINSIC(vector_dup, inverse_d + 32 * 32, 0.0f, 1, 1, 0, 8, 0);
  INTRINSIC_NO_ARGS(set_mask_norm);
  INTRINSIC(set_vector_mask, static_cast<uint64_t>(-1),
            static_cast<uint64_t>(-1));
  INTRINSIC(pipe_barrier, PIPE_V);
}
}
