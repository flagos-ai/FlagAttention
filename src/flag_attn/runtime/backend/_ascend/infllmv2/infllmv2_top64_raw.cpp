// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Memref ABI excerpt from Huawei AscendNPU-IR Utils.h (Apache-2.0).
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// https://github.com/Ascend/ascendnpu-ir/blob/master/bishengir/lib/Template/include/Utils.h
// Upstream Git blob: c5c52db2ef577d4e8319be3454b87028af76378a
#include <cstddef>
#include <cstdint>
#ifndef __aiv__
#define __aiv__ [aicore]
#endif

template <typename T, std::size_t Dim> struct memref_t {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

// Exact descending sort of normal finite encoded keys, one query per call.
// The Python wrapper pads tail lanes with FLT_MIN (0x00800000), so every value
// entering vbitsort/vmrgsort4 is a positive normal binary32; the smallest real
// key is >= 0x00800001 in any padded call. All index tie bits are in the score,
// so the auxiliary payload is irrelevant.
extern "C" __aiv__ void _mlir_ciface_infllm_sort64(
    memref_t<__ubuf__ uint32_t, 2> *ids,
    memref_t<__ubuf__ float, 2> *gate,
    memref_t<__ubuf__ float, 2> *pairs,
    memref_t<__ubuf__ float, 2> *scratch,
    memref_t<__ubuf__ float, 2> *best) {
  const int n = gate->sizes[1];
  auto input = gate->aligned + gate->offset;
  auto index = ids->aligned + ids->offset;
  auto a = pairs->aligned + pairs->offset;
  auto b = scratch->aligned + scratch->offset;
  auto out = best->aligned + best->offset;
  set_mask_norm();
  set_vector_mask(~0ULL, ~0ULL);
  vbitsort(a, input, index, n / 32);
  pipe_barrier(PIPE_V);
  int length = 32;
  int groups = n / 32;
  while (groups > 1) {
    for (int first = 0; first < groups; first += 4) {
      const int available = groups - first;
      __ubuf__ float *lists[4] = {
          a + first * length * 2,
          a + (first + 1) * length * 2,
          a + (first + (available > 2 ? 2 : 0)) * length * 2,
          a + (first + (available > 3 ? 3 : 0)) * length * 2};
      vmrgsort4(b + first * length * 2, lists, 1,
                length, length, available > 2 ? length : 0,
                available > 3 ? length : 0, false, available > 3 ? 15 : 3);
    }
    pipe_barrier(PIPE_V);
    auto swap = a;
    a = b;
    b = swap;
    length *= 4;
    groups = (groups + 3) / 4;
  }
  pipe_barrier(PIPE_ALL);
  for (int i = 0; i < 64; ++i) out[i] = a[i * 2];
  pipe_barrier(PIPE_ALL);
  set_vector_mask(~0ULL, ~0ULL);
}
