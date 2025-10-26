#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__device__ inline float warp_reduce_sum(float v) {
  unsigned mask = 0xffffffffu;
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset, WARP_SIZE);
  }
  return v;
}

__global__ void spmv_warp_per_row_vec4_kernel(int n, const int *__restrict__ rp,
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
    int j = start;
    int prefix = (4 - (j & 3)) & 3;
    int j_aligned = j + min(prefix, max(0, end - j));

    for (int jj = j + lane; jj < j_aligned; jj += WARP_SIZE) {
      sum += a[jj] * x[ci[jj]];
    }
    int remaining = end - j_aligned;
    int vec_quads_total = remaining >> 2;
    int quad_start = (j_aligned >> 2);
    for (int q = lane; q < vec_quads_total; q += WARP_SIZE) {
      int base_quad = quad_start + q;
      const float4 *a4 = reinterpret_cast<const float4 *>(a);
      const int4 *c4 = reinterpret_cast<const int4 *>(ci);
      float4 av = a4[base_quad];
      int4 cv = c4[base_quad];
      sum += av.x * x[cv.x];
      sum += av.y * x[cv.y];
      sum += av.z * x[cv.z];
      sum += av.w * x[cv.w];
    }

    int j_vec_end = j_aligned + (vec_quads_total << 2);
    for (int jj = j_vec_end + lane; jj < end; jj += WARP_SIZE) {
      sum += a[jj] * x[ci[jj]];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
      y[row] = sum;
  }
}

extern "C" void warp_vectorized(int n, const int *rp, const int *ci,
                                const float *a, const float *x, float *y) {
  // 4 warps / block ?
  const int threads = 128;
  const int warps_per_block = threads / WARP_SIZE;
  int warps_needed = (n + (1) - 1) / 1; // is that correct??
  int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;
  if (blocks < 1)
    blocks = 1;

  spmv_warp_per_row_vec4_kernel<<<blocks, threads>>>(n, rp, ci, a, x, y);
}
