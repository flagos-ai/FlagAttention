// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"
extern "C" __aiv__ void _mlir_ciface_prefix_owned(
 memref_t<__ubuf__ float,1>* center,memref_t<__ubuf__ float,2>* input,memref_t<__ubuf__ float,1>* carry,memref_t<__ubuf__ float,2>* work) {
 auto a=input->aligned+input->offset;auto b=work->aligned+work->offset;auto z=carry->aligned+carry->offset;auto c=center->aligned+center->offset;
 int rows=input->sizes[0],width=input->sizes[1];auto src=a;auto dst=b;
 set_mask_count();
 for(int shift=1;shift<rows;shift*=2) {
  set_vector_mask(0,shift*width);vadds(dst,src,0.f,1,1,1,8,8);
  set_vector_mask(0,(rows-shift)*width);vadd(dst+shift*width,src+shift*width,src,1,1,1,1,8,8,8);
  pipe_barrier(PIPE_V);auto t=src;src=dst;dst=t;
 }
 if(src!=a){set_vector_mask(0,rows*width);vadds(a,src,0.f,1,1,1,8,8);pipe_barrier(PIPE_V);}
 set_mask_norm();set_vector_mask(0,0xffff);
 vadd(a,a,z,rows,1,1,1,2,2,0);pipe_barrier(PIPE_V);
 vadds(z,a+(rows-1)*width,0.f,1,1,1,2,2);pipe_barrier(PIPE_V);
 vsub(a,a,c,rows,1,1,1,2,2,0);pipe_barrier(PIPE_V);
 set_vector_mask(~(uint64_t)0,~(uint64_t)0);
}
