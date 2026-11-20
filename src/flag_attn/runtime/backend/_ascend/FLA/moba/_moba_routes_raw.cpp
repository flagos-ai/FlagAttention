// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Route histogram and bucket kernels for one (token tile, head) group.
// Up to eight rank segments use separate UB buffers; -1 denotes no route.
#include "Utils.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)
#define INTRINSIC_NO_ARGS(NAME) NAME()

static constexpr int32_t VDUP_CHUNK_I32 = 255 * (256 / 4);

__aiv__ __attribute__((always_inline)) void
vdup_i32(__ubuf__ int32_t *dst, int32_t value, int32_t n) {
  constexpr int32_t ELEMS_PER_REPEAT = 256 / 4;
  for (int32_t done = 0; done < n; done += VDUP_CHUNK_I32) {
    const int32_t len = MIN(n - done, VDUP_CHUNK_I32);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, len);
    INTRINSIC(vector_dup, dst + done, value, CEIL_DIV(len, ELEMS_PER_REPEAT), 1,
              1, 8, 0);
    INTRINSIC_NO_ARGS(set_mask_norm);
  }
}

// Rank-unrolled histogram body: cnt[chunk] += 1 over one rank segment.
#define HIST_RANK(P)                                                           \
  for (int32_t t = 0; t < bt; ++t) {                                           \
    const int32_t x = (P)[t];                                                  \
    if (x >= 0 && x < nc) {                                                    \
      c[x] = c[x] + 1;                                                         \
    }                                                                          \
  }

// st and bs are in/out cursors for tile-local destinations and compact positions.
// Each call writes positions in rank-major order: rank * bt + token.
#define BUCKET_RANK(P, RANK)                                                   \
  for (int32_t t = 0; t < bt; ++t) {                                           \
    const int32_t x = (P)[t];                                                  \
    if (x >= 0 && x < nc) {                                                    \
      const int32_t dest = st[x];                                              \
      st[x] = dest + 1;                                                        \
      const int32_t q = bs[x];                                                 \
      bs[x] = q + 1;                                                           \
      o[dest] = row_base + t * num_heads + h;                                  \
      p[(RANK) * bt + t] = q;                                                  \
    }                                                                          \
  }

extern "C" {

// Per-(tile, head) chunk histogram.
__aiv__ void _mlir_ciface_moba_routes_hist(
    memref_t<__ubuf__ int32_t, 1> *i0, memref_t<__ubuf__ int32_t, 1> *i1,
    memref_t<__ubuf__ int32_t, 1> *i2, memref_t<__ubuf__ int32_t, 1> *i3,
    memref_t<__ubuf__ int32_t, 1> *i4, memref_t<__ubuf__ int32_t, 1> *i5,
    memref_t<__ubuf__ int32_t, 1> *i6, memref_t<__ubuf__ int32_t, 1> *i7,
    int32_t topk, int32_t bt, int32_t nc,
    memref_t<__ubuf__ int32_t, 1> *cnt) {
  __ubuf__ int32_t *c = cnt->aligned + cnt->offset;
  pipe_barrier(PIPE_ALL);
  vdup_i32(c, 0, nc);
  if (topk > 0) HIST_RANK(i0->aligned + i0->offset)
  if (topk > 1) HIST_RANK(i1->aligned + i1->offset)
  if (topk > 2) HIST_RANK(i2->aligned + i2->offset)
  if (topk > 3) HIST_RANK(i3->aligned + i3->offset)
  if (topk > 4) HIST_RANK(i4->aligned + i4->offset)
  if (topk > 5) HIST_RANK(i5->aligned + i5->offset)
  if (topk > 6) HIST_RANK(i6->aligned + i6->offset)
  if (topk > 7) HIST_RANK(i7->aligned + i7->offset)
}

// Counting-sort bucket of one (tile, head): runs in chunk order + positions.
__aiv__ void _mlir_ciface_moba_routes_bucket(
    memref_t<__ubuf__ int32_t, 1> *i0, memref_t<__ubuf__ int32_t, 1> *i1,
    memref_t<__ubuf__ int32_t, 1> *i2, memref_t<__ubuf__ int32_t, 1> *i3,
    memref_t<__ubuf__ int32_t, 1> *i4, memref_t<__ubuf__ int32_t, 1> *i5,
    memref_t<__ubuf__ int32_t, 1> *i6, memref_t<__ubuf__ int32_t, 1> *i7,
    memref_t<__ubuf__ int32_t, 1> *start, memref_t<__ubuf__ int32_t, 1> *base,
    int32_t topk, int32_t bt, int32_t h, int32_t num_heads, int32_t row_base,
    memref_t<__ubuf__ int32_t, 1> *out,
    memref_t<__ubuf__ int32_t, 1> *pos) {
  __ubuf__ int32_t *st = start->aligned + start->offset;
  __ubuf__ int32_t *bs = base->aligned + base->offset;
  __ubuf__ int32_t *o = out->aligned + out->offset;
  __ubuf__ int32_t *p = pos->aligned + pos->offset;
  const int32_t nc = (int32_t)(start->sizes[0]);

  // No -1 fill: Triton pre-fills the run buffer once, because a tile may be
  // streamed through several calls and a per-call fill would wipe earlier blocks.
  pipe_barrier(PIPE_ALL);

  if (topk > 0) BUCKET_RANK(i0->aligned + i0->offset, 0)
  if (topk > 1) BUCKET_RANK(i1->aligned + i1->offset, 1)
  if (topk > 2) BUCKET_RANK(i2->aligned + i2->offset, 2)
  if (topk > 3) BUCKET_RANK(i3->aligned + i3->offset, 3)
  if (topk > 4) BUCKET_RANK(i4->aligned + i4->offset, 4)
  if (topk > 5) BUCKET_RANK(i5->aligned + i5->offset, 5)
  if (topk > 6) BUCKET_RANK(i6->aligned + i6->offset, 6)
  if (topk > 7) BUCKET_RANK(i7->aligned + i7->offset, 7)
}
}
