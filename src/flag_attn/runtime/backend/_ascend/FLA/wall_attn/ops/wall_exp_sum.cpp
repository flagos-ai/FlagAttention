// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// In-place UB exp2 for FP16 [BT,256]; FP32 sums [1,BT], scratch [16,256].
#include "Utils.h"
extern "C" __aiv__ void _mlir_ciface_wall_exp_sum_inplace(
    memref_t<__ubuf__ half,2>* input,
    memref_t<__ubuf__ float,2>* sums,
    memref_t<__ubuf__ float,2>* scratch) {
  auto s=input->aligned+input->offset;

  auto z=sums->aligned+sums->offset;
  auto w=scratch->aligned+scratch->offset;
  const int rows=input->sizes[0];
  set_mask_norm();set_vector_mask((uint64_t)0,~(uint64_t)0);
  for(int r=0;r<rows;r+=16) {
    vconv_f162f32(w,s+r*256,64,1,1,8,4);
    pipe_barrier(PIPE_V);
    vmuls(w,w,0.6931471805599453f,64,1,1,8,8);
    pipe_barrier(PIPE_V);
    vexp(w,w,64,1,1,8,8);
    pipe_barrier(PIPE_V);
    vconv_f322f16(s+r*256,w,64,1,1,4,8);
    pipe_barrier(PIPE_V);
    vadd(w,w,w+64,16,1,1,1,32,32,32);
    vadd(w+128,w+128,w+192,16,1,1,1,32,32,32);
    pipe_barrier(PIPE_V);
    vadd(w,w,w+128,16,1,1,1,32,32,32);
    pipe_barrier(PIPE_V);
    vcadd(z+r,w,16,1,1,32,0);
    pipe_barrier(PIPE_V);
  }
  set_vector_mask(~(uint64_t)0,~(uint64_t)0);
}
