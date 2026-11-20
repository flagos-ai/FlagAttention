// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"
extern "C" __aiv__ void _mlir_ciface_stats_bounds(
    memref_t<__ubuf__ float,2>* g,
    memref_t<__ubuf__ float,2>* q,
    memref_t<__ubuf__ float,2>* k,
    memref_t<__ubuf__ float,2>* v,
    memref_t<__ubuf__ float,2>* state,
    memref_t<__ubuf__ float,2>* work,
    memref_t<__ubuf__ float,1>* aux) {
  auto gg=g->aligned+g->offset;
  auto qq=q->aligned+q->offset;
  auto kk=k->aligned+k->offset;
  auto vv=v->aligned+v->offset;
  auto ss=state->aligned+state->offset;
  auto w=work->aligned+work->offset;
  auto a=aux->aligned+aux->offset;
  // This operation clobbers the g, q, k, and v tiles.
  // State row 0 stores channel sums. Rows 3 through 6 store g, Q, K, and V bounds.
  const int rows=g->sizes[0];
  set_mask_count(); set_vector_mask(0,rows*16);
  vmul(qq,qq,qq,1,1,1,1,8,8,8);
  vmul(kk,kk,kk,1,1,1,1,8,8,8);
  vabs(vv,vv,1,1,1,8,8);
  vadds(w,gg,0.f,1,1,1,8,8);
  pipe_barrier(PIPE_V);

  set_mask_norm();set_vector_mask(0,0xffff);
  vcadd(a,qq,rows,1,1,2,0);
  vcadd(a+64,kk,rows,1,1,2,0);
  set_vector_mask(0,~(uint64_t)0);
  vcmax(a+128,vv,rows/4,1,1,8,2);
  vcmax(a+144,gg,rows/4,1,1,8,2);
  set_mask_count();set_vector_mask(0,(rows/2)*16);
  vadd(w,w,w+(rows/2)*16,1,1,1,1,8,8,8);
  pipe_barrier(PIPE_V);

  set_mask_norm();set_vector_mask(0,~(uint64_t)0);
  vcmax(qq,a,1,1,1,8,2);
  vcmax(kk,a+64,1,1,1,8,2);
  set_vector_mask(0,0xffff);
  vcmax(vv,a+128,1,1,1,8,2);
  vcmax(gg,a+144,1,1,1,8,2);
  set_mask_count();set_vector_mask(0,(rows/4)*16);
  vadd(w,w,w+(rows/4)*16,1,1,1,1,8,8,8);
  pipe_barrier(PIPE_V);

  set_mask_norm();set_vector_mask(0,1);
  vmax(ss+48,ss+48,gg,1,1,1,1,2,2,2);
  vmax(ss+64,ss+64,qq,1,1,1,1,2,2,2);
  vmax(ss+80,ss+80,kk,1,1,1,1,2,2,2);
  vmax(ss+96,ss+96,vv,1,1,1,1,2,2,2);
  set_mask_count();set_vector_mask(0,(rows/8)*16);
  vadd(w,w,w+(rows/8)*16,1,1,1,1,8,8,8);
  pipe_barrier(PIPE_V);
  for(int half=rows/16;half>=1;half/=2) {
    set_vector_mask(0,half*16);
    vadd(w,w,w+half*16,1,1,1,1,8,8,8);
    pipe_barrier(PIPE_V);
  }
  set_mask_norm();set_vector_mask(0,0xffff);
  vadd(ss,ss,w,1,1,1,1,2,2,2);
  pipe_barrier(PIPE_V);
  set_vector_mask(~(uint64_t)0,~(uint64_t)0);
}
