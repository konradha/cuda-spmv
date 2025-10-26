#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef NNZ_PER_TILE
#define NNZ_PER_TILE 4096
#endif
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

__device__ unsigned int g_tile_counter = 0;

__device__ __forceinline__ int upper_bound_nnz(const int *__restrict__ rp,
                                               int n, int j) {
  if (j < 0)
    return 0;
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (rp[mid + 1] <= j)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  unsigned m = 0xffffffffu;
#pragma unroll
  for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
    v += __shfl_down_sync(m, v, off, WARP_SIZE);
  }
  return v;
}

__global__ void persistent_queue_kernel(int n, const int *__restrict__ rp,
                                        const int *__restrict__ ci,
                                        const float *__restrict__ a,
                                        const float *__restrict__ x,
                                        float *__restrict__ y) {
  const int warps_per_block = blockDim.x / WARP_SIZE;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & (WARP_SIZE - 1);

  const int nnz_total = rp[n];
  if (nnz_total <= 0)
    return;
  const int tiles = (nnz_total + NNZ_PER_TILE - 1) / NNZ_PER_TILE;

  for (;;) {
    unsigned int tile = atomicAdd(&g_tile_counter, 1u);
    if (tile >= (unsigned)tiles)
      break;

    const int j0 = tile * NNZ_PER_TILE;
    const int j1 = min(j0 + NNZ_PER_TILE, nnz_total);

    const int row_begin = upper_bound_nnz(rp, n, j0 - 1);
    int row_end_excl = upper_bound_nnz(rp, n, j1 - 1) + 1;
    if (row_end_excl > n)
      row_end_excl = n;

    for (int row = row_begin + warp_id; row < row_end_excl;
         row += warps_per_block) {
      const int rlo = rp[row];
      const int rhi = rp[row + 1];
      const int lo = max(rlo, j0);
      const int hi = min(rhi, j1);
      if (lo >= hi)
        continue;

      float acc = 0.0f;
      for (int j = lo + lane; j < hi; j += WARP_SIZE) {
        const int col = ci[j];
        acc += a[j] * __ldg(&x[col]);
      }
      acc = warp_reduce_sum(acc);

      if (lane == 0) {
        const bool full_row = (rlo >= j0) && (rhi <= j1);
        if (full_row)
          y[row] = acc;
        else
          atomicAdd(&y[row], acc);
      }
    }
    __syncthreads();
  }
}

extern "C" void pqueue(int n, const int *rp, const int *ci, const float *a,
                       const float *x, float *y) {
  void *dctr = nullptr;
  cudaGetSymbolAddress(&dctr, g_tile_counter);
  cudaMemset(dctr, 0, sizeof(unsigned int));

  const int threads = THREADS_PER_BLOCK;
  int blocks = 128;
  persistent_queue_kernel<<<blocks, threads>>>(n, rp, ci, a, x, y);
}
