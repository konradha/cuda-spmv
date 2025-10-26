#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef WARPS_PER_ROW
#define WARPS_PER_ROW 4
#endif

__global__ void multiwarp_kernel(int n, const int *__restrict__ rp,
                                 const int *__restrict__ ci,
                                 const float *__restrict__ a,
                                 const float *__restrict__ x,
                                 float *__restrict__ y) {
  static_assert(WARPS_PER_ROW >= 1, "WARPS_PER_ROW must be >= 1");
  static_assert((WARPS_PER_ROW & (WARPS_PER_ROW - 1)) == 0,
                "WARPS_PER_ROW power of 2");

  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp_in_block = threadIdx.x >> 5;

  if (blockDim.x != WARPS_PER_ROW * WARP_SIZE)
    return;

  extern __shared__ float smem[];
  float *warp_partials = smem;

  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    const int start = rp[row];
    const int end = rp[row + 1];

    float sum = 0.0f;
    for (int j = start + warp_in_block * WARP_SIZE + lane; j < end;
         j += WARPS_PER_ROW * WARP_SIZE) {
      const int col = ci[j];
      sum += a[j] * x[col];
    }

    unsigned mask = 0xffffffffu;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(mask, sum, offset, WARP_SIZE);
    }
    if (lane == 0) {
      warp_partials[warp_in_block] = sum;
    }
    __syncthreads();
    if (warp_in_block == 0) {
      float total = 0.0f;
      for (int w = 0; w < WARPS_PER_ROW; ++w)
        total += warp_partials[w];
      if (lane == 0) {
        y[row] = total;
      }
    }
    __syncthreads();
  }
}

extern "C" void spmv_multiwarp(int n, const int *rp, const int *ci,
                               const float *a, const float *x, float *y) {
  // 1 row/ block, WARPS_PER_ROW warps / block
  const int threads = WARPS_PER_ROW * WARP_SIZE;
  int blocks = (n + 1) / 1;
  if (blocks < 1)
    blocks = 1;

  const size_t shmem_bytes = WARPS_PER_ROW * sizeof(float);
  multiwarp_kernel<<<blocks, threads, shmem_bytes>>>(n, rp, ci, a, x, y);
}
