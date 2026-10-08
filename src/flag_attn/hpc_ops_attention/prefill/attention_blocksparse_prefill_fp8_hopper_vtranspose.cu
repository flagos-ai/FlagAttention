// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// CUDA-critical sparse compaction and Hopper FP8 V-tile transpose used by the
// TLE-Struct attention kernel.
//
// TLE-Raw expands each rank-2 tensor/memdesc operand into the standard memref
// tuple below: allocated pointer, aligned pointer, offset, two sizes, and two
// strides. Both operands are existing 128-byte-swizzled NVMMA buffers: the
// gathered [N,D] V stage and the [D,N] destination consumed by PV WGMMA.

#include <stdint.h>

using shared_u8_ptr =
    unsigned char __attribute__((address_space(3))) *;
using shared_i32_ptr = int32_t __attribute__((address_space(3))) *;
using global_const_u8_ptr =
    const unsigned char __attribute__((address_space(1))) *;

static __device__ __forceinline__ uint32_t to_smem_addr(shared_u8_ptr ptr) {
  const void *generic_ptr = ptr;
  return static_cast<uint32_t>(__cvta_generic_to_shared(generic_ptr));
}

static __device__ __forceinline__ int swizzle_128b_offset(
    int row, int col, int row_stride) {
  // NVMMA SW128 XORs the three 16-byte-column bits with the low three
  // row bits.  STSM addresses always start on a 16-byte chunk.
  const int chunk = (col >> 4) ^ (row & 7);
  return row * row_stride + (chunk << 4);
}

// Keep the LDSM itself separate from the PRMT/STSM sequence so the caller can
// issue the next tile's LDSM before consuming the current tile's registers.
// The helpers are force-inlined: they are scheduling scaffolding, not runtime
// calls, and the two four-register tuples stay in the consumer registers.
static __device__ __forceinline__ void bsa_vtranspose_ldsm_tile(
    shared_u8_ptr src_aligned, int src_row_stride, int d_tile, int lane,
    int inverse_0189_src_row, int n_tile, uint32_t &v0, uint32_t &v1,
    uint32_t &v2, uint32_t &v3) {
  // CUTE SM75_U16x8_LDSM_T source partition for a 16x32 byte tile:
  // lanes 0..15 address D[0..15], lanes 16..31 address D[16..31].
  const int src_row = n_tile * 16 + inverse_0189_src_row;
  const int src_col = d_tile * 32 + (lane >> 4) * 16;
  const int src_physical =
      swizzle_128b_offset(src_row, src_col, src_row_stride);
  const uint32_t src_addr =
      to_smem_addr(src_aligned + src_physical);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
      : "r"(src_addr));
}

static __device__ __forceinline__ void bsa_vtranspose_prmt_stsm_tile(
    shared_u8_ptr dst_aligned, int dst_row_stride, int d_tile, int lane,
    int n_tile, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
  uint32_t vt0, vt1, vt2, vt3;
  asm volatile("prmt.b32 %0, %1, %2, 0x6420;"
               : "=r"(vt0)
               : "r"(v0), "r"(v1));
  asm volatile("prmt.b32 %0, %1, %2, 0x7531;"
               : "=r"(vt1)
               : "r"(v0), "r"(v1));
  asm volatile("prmt.b32 %0, %1, %2, 0x6420;"
               : "=r"(vt2)
               : "r"(v2), "r"(v3));
  asm volatile("prmt.b32 %0, %1, %2, 0x7531;"
               : "=r"(vt3)
               : "r"(v2), "r"(v3));

  // CUTE SM90_U32x4_STSM_N destination partition.  In physical [D,N]
  // coordinates, lane rows are ordered
  //   0,2,...,14, 1,3,...,15, 16,18,...,30, 17,19,...,31.
  const int dst_row_in_tile =
      (lane >> 4) * 16 + (lane & 7) * 2 + ((lane >> 3) & 1);
  const int dst_row = d_tile * 32 + dst_row_in_tile;
  const int dst_col = n_tile * 16;
  const int dst_physical =
      swizzle_128b_offset(dst_row, dst_col, dst_row_stride);
  const uint32_t dst_addr =
      to_smem_addr(dst_aligned + dst_physical);

  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
      "[%0], {%1, %2, %3, %4};"
      :
      : "r"(dst_addr), "r"(vt0), "r"(vt1), "r"(vt2), "r"(vt3)
      : "memory");
}

// Build the stable sparse KV-tile list with the same warp ballot/popcount
// algorithm as the native CUDA kernel. Only the first warp of the four-warp
// producer participates; all lanes retain the same running output count.
extern "C" __device__ __attribute__((used)) void bsa_compact_active_tiles(
    global_const_u8_ptr mask_ptr, int32_t num_tile_with_mask,
    int32_t num_tile_kv, shared_i32_ptr active_alloc,
    shared_i32_ptr active_aligned, int64_t active_offset,
    int64_t active_size0, int64_t active_stride0,
    shared_i32_ptr count_alloc, shared_i32_ptr count_aligned,
    int64_t count_offset, int64_t count_size0, int64_t count_stride0) {
  (void)active_alloc;
  (void)active_size0;
  (void)count_alloc;
  (void)count_size0;

  constexpr int kProducerWarps = 4;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int producer_warp =
      (static_cast<int>(threadIdx.x) >> 5) & (kProducerWarps - 1);
  if (producer_warp != 0) {
    return;
  }

  active_aligned += active_offset;
  count_aligned += count_offset;
  int32_t num_active = 0;

  for (int32_t base = 0; base < num_tile_with_mask; base += 32) {
    const int32_t tile = base + lane;
    const bool active = tile < num_tile_with_mask && mask_ptr[tile] != 0;
    const uint32_t ballot = __ballot_sync(0xffffffffu, active);
    if (active) {
      const uint32_t earlier_lanes = (1u << lane) - 1u;
      const int32_t rank = __popc(ballot & earlier_lanes);
      active_aligned[(num_active + rank) * active_stride0] = tile;
    }
    num_active += __popc(ballot);
  }

  // CUDA compatibility rule: when the supplied mask is shorter than the
  // causal range, retain exactly the first tile after the represented range.
  if (num_tile_with_mask < num_tile_kv) {
    if (lane == 0) {
      active_aligned[num_active * active_stride0] = num_tile_with_mask;
    }
    ++num_active;
  }

  __syncwarp();
  if (lane == 0) {
    count_aligned[0 * count_stride0] = num_active;
  }
}

// TLE-Raw looks this symbol up in the standalone LLVM module.  Do not mark
// the exported entry force-inline: with no caller in this translation unit,
// Clang may otherwise omit the definition before TLE imports it.
extern "C" __device__ __attribute__((used)) void bsa_fp8_vtranspose_128x128(
    shared_u8_ptr src_alloc, shared_u8_ptr src_aligned, int64_t src_offset,
    int64_t src_size0, int64_t src_size1, int64_t src_stride0,
    int64_t src_stride1, shared_u8_ptr dst_alloc,
    shared_u8_ptr dst_aligned, int64_t dst_offset, int64_t dst_size0,
    int64_t dst_size1, int64_t dst_stride0, int64_t dst_stride1) {
  (void)src_alloc;
  (void)src_size0;
  (void)src_size1;
  (void)dst_alloc;
  (void)dst_size0;
  (void)dst_size1;
  (void)src_stride1;
  (void)dst_stride1;

  const int lane = static_cast<int>(threadIdx.x) & 31;
  // The raw region runs independently in each four-warp consumer partition.
  // Masking the physical warp id gives its local warp inside either consumer.
  constexpr int kConsumerWarps = 4;
  const int warp = (static_cast<int>(threadIdx.x) >> 5) & (kConsumerWarps - 1);
  const int src_row_stride = static_cast<int>(src_stride0);
  const int dst_row_stride = static_cast<int>(dst_stride0);

  src_aligned += src_offset;
  dst_aligned += dst_offset;

  // Keep the native LDSM_T/PRMT/STSM_N pipeline, but adapt its result to
  // TLE's RS-WGMMA register contract.  Native CUDA applies matching 0189
  // permutations to both P registers and Vt shared memory.  TLE already
  // supplies P in its own RS layout, for which a semantic V^T is required.
  // Pre-permuting the 16 source rows by inverse(0189) makes the unchanged
  // native PRMT/STSM sequence publish exactly that semantic V^T.
  const int d_tile = warp;
  const int logical_src_row = lane & 15;
  const int inverse_0189_src_row =
      (logical_src_row & 1) | ((logical_src_row & 2) << 1) |
      ((logical_src_row & 4) << 1) | ((logical_src_row & 8) >> 2);

  // Prime tile 0, then run two-tile software pipelining.  The next LDSM is
  // issued before the current tile's first PRMT(0x6420), which is the
  // dependency that dominates the Raw V-transpose short-scoreboard samples.
  // The pair loop is fully unrolled, so the two register tuples are not local
  // memory arrays and the generated order remains LDSM(next) -> PRMT/STSM.
  uint32_t v0a, v1a, v2a, v3a;
  bsa_vtranspose_ldsm_tile(
      src_aligned, src_row_stride, d_tile, lane, inverse_0189_src_row, 0,
      v0a, v1a, v2a, v3a);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const int n_tile = pair * 2;
    uint32_t v0b, v1b, v2b, v3b;
    bsa_vtranspose_ldsm_tile(
        src_aligned, src_row_stride, d_tile, lane, inverse_0189_src_row,
        n_tile + 1, v0b, v1b, v2b, v3b);

    bsa_vtranspose_prmt_stsm_tile(
        dst_aligned, dst_row_stride, d_tile, lane, n_tile,
        v0a, v1a, v2a, v3a);

    if (pair < 3) {
      // Refill the first tuple before consuming v0b..v3b.  At the next
      // unrolled iteration this becomes the current tile, while the second
      // tuple is refilled again before its PRMT chain.
      bsa_vtranspose_ldsm_tile(
          src_aligned, src_row_stride, d_tile, lane,
          inverse_0189_src_row, n_tile + 2, v0a, v1a, v2a, v3a);
    }

    bsa_vtranspose_prmt_stsm_tile(
        dst_aligned, dst_row_stride, d_tile, lane, n_tile + 1,
        v0b, v1b, v2b, v3b);
  }

  // Publish generic-proxy STSM writes to the async proxy used by WGMMA.
  // The CTA barrier in the TLE caller then makes every warp's tile visible.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
