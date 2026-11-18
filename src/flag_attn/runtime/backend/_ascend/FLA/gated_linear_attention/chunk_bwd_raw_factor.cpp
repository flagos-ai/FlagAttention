// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "Utils.h"

// Fixed contiguous FP32 UB tiles: reference[1,128], remaining tensors[64,128].
// gate and operand are initialized writable outputs of the Raw call.
template<bool QUERY>
__aiv__ inline void factor(memref_t<__ubuf__ float, 2>* ref,
                          memref_t<__ubuf__ float, 2>* gate,
                          memref_t<__ubuf__ float, 2>* operand,
                          memref_t<__ubuf__ float, 2>* scratch) {
  __ubuf__ float* r = ref->aligned + ref->offset;
  __ubuf__ float* g = gate->aligned + gate->offset;
  __ubuf__ float* x = operand->aligned + operand->offset;
  __ubuf__ float* w = scratch->aligned + scratch->offset;
  set_mask_norm();
  set_vector_mask(0, static_cast<uint64_t>(-1));
  vsub(g, g, r, 64, 1, 1, 1, 16, 16, 0);
  vsub(g + 64, g + 64, r + 64, 64, 1, 1, 1, 16, 16, 0);
  pipe_barrier(PIPE_V);
  set_mask_count();
  set_vector_mask(0, 8192);
  vmuls(g, g, QUERY ? -0.6931471805599453f : 0.6931471805599453f, 128, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vmuls(w, g, -1.0f, 128, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vexp(g, g, 128, 1, 1, 8, 8);
  vexp(w, w, 128, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vmul(x, x, w, 128, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  set_mask_norm();
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

extern "C" {
__aiv__ void _mlir_ciface_gla_bwd_key_factor(
    memref_t<__ubuf__ float, 2>* ref, memref_t<__ubuf__ float, 2>* gate,
    memref_t<__ubuf__ float, 2>* operand, memref_t<__ubuf__ float, 2>* scratch) {
  factor<false>(ref, gate, operand, scratch);
}

__aiv__ void _mlir_ciface_gla_bwd_query_factor(
    memref_t<__ubuf__ float, 2>* ref, memref_t<__ubuf__ float, 2>* gate,
    memref_t<__ubuf__ float, 2>* operand, memref_t<__ubuf__ float, 2>* scratch) {
  factor<true>(ref, gate, operand, scratch);
}
}
