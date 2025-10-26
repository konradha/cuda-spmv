#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void spmv_warp_per_row(int n, const int *__restrict__ rp,
                                  const int *__restrict__ ci,
                                  const float *__restrict__ a,
                                  const float *__restrict__ x,
                                  float *__restrict__ y) {

  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warps_per_block = blockDim.x >> 5;
  const int warp_in_block = threadIdx.x >> 5;
  const int global_warp = blockIdx.x * warps_per_block + warp_in_block;
  const int total_warps = gridDim.x * warps_per_block;

  for (int row = global_warp; row < n; row += total_warps) {
    const int start = rp[row];
    const int end = rp[row + 1];
    float sum = 0.0f;
    for (int j = start + lane; j < end; j += WARP_SIZE) {
      const int col = ci[j];
      sum += a[j] * x[col];
    }

    unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(mask, sum, offset, WARP_SIZE);
    }

    if (lane == 0) {
      y[row] = sum;
    }
  }
}

extern "C" void warp_per_row(int n, const int *rp, const int *ci,
                             const float *a, const float *x, float *y) {
  int threads = 128; // that's 4 warps
  // 1 warp/row
  int warps_needed = (n + 1 + (WARP_SIZE - 1)) / WARP_SIZE;
  int blocks =
      (warps_needed + (threads / WARP_SIZE) - 1) / (threads / WARP_SIZE);
  if (blocks < 1)
    blocks = 1;

  spmv_warp_per_row<<<blocks, threads>>>(n, rp, ci, a, x, y);
}
