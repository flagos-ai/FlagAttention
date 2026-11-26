// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "Utils.h"

// Stable top-eight selection from a contiguous [64, 128] gate tile.
// UB scratch: pairs [16, 256], sorted [16, 64]; best [64, 16] is in/out state.
// Each proposal stores an FP32 score followed by uint32 id bits.
extern "C" __aiv__ void _mlir_ciface_moba_sort_topk(
    memref_t<__ubuf__ uint32_t, 2> *ids,
    memref_t<__ubuf__ float, 2> *gate,
    memref_t<__ubuf__ float, 2> *pairs,
    memref_t<__ubuf__ float, 2> *sorted,
    memref_t<__ubuf__ float, 2> *best) {
  const int tokens = gate->sizes[0];
  const int chunks = gate->sizes[1];
  const int group_tokens = 16;
  auto scores = gate->aligned + gate->offset;
  auto indices = ids->aligned + ids->offset;
  auto proposals = pairs->aligned + pairs->offset;
  auto local = sorted->aligned + sorted->offset;
  auto running = best->aligned + best->offset;

  // Reuse sorting scratch across groups to stay within the UB budget.
  for (int group = 0; group < tokens; group += group_tokens) {
    auto tile = scores + group * chunks;
    auto tile_ids = indices + group * chunks;
    auto current = running + group * 16;
    set_mask_norm();
    set_vector_mask(~0ULL, ~0ULL);
    vbitsort(proposals, tile, tile_ids, group_tokens * chunks / 32);
    pipe_barrier(PIPE_V);

    // Only the first eight proposals of each sorted group of 32 can win.
    // Integer copies preserve the id field, without FP32 denormal handling.
    set_vector_mask(0, 65535ULL);
    vadds((__ubuf__ int32_t *)tile, (__ubuf__ int32_t *)proposals,
          0, group_tokens * 4, 1, 1, 2, 8);
    pipe_barrier(PIPE_V);
    set_vector_mask(~0ULL, ~0ULL);
    __ubuf__ float *lists[4] = {tile, tile + 16, tile + 32, tile + 48};
    vmrgsort4(local, lists, group_tokens, 8, 8, 8, 8, false, 15);
    pipe_barrier(PIPE_V);

    // Existing candidates precede the new tile so ties keep smaller chunk ids.
    for (int token = 0; token < group_tokens; ++token) {
      __ubuf__ float *merge[4] = {
          current + token * 16, local + token * 64,
          local + token * 64, local + token * 64};
      vmrgsort4(proposals + token * 32, merge, 1, 8, 8, 0, 0, false, 3);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask(0, 65535ULL);
    vadds((__ubuf__ int32_t *)current, (__ubuf__ int32_t *)proposals,
          0, group_tokens, 1, 1, 2, 4);
    pipe_barrier(PIPE_ALL);
    set_vector_mask(~0ULL, ~0ULL);
  }
}
