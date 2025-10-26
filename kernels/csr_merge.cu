#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__device__ __forceinline__ int upper_bound_row(const int *__restrict__ rp,
                                               int n, int j) {
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

__global__ void merge_kernel(int n, const int *__restrict__ rp,
                             const int *__restrict__ ci,
                             const float *__restrict__ a,
                             const float *__restrict__ x,
                             float *__restrict__ y) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warps_per_block = blockDim.x >> 5;
  const int warp_in_block = threadIdx.x >> 5;
  const int global_warp = blockIdx.x * warps_per_block + warp_in_block;
  const int total_warps = gridDim.x * warps_per_block;

  const int nnz_total = rp[n];
  if (nnz_total == 0)
    return;

  const int chunk = (nnz_total + total_warps - 1) / total_warps;
  const int j0 = min(global_warp * chunk, nnz_total);
  const int j1 = min(j0 + chunk, nnz_total);
  if (j0 >= j1)
    return;

  int j = j0 + lane;
  if (j >= j1)
    return;

  int row = upper_bound_row(rp, n, j);
  if (row >= n)
    return;
  int row_end = rp[row + 1];

  float acc = 0.0f;

  for (;; j += WARP_SIZE) {
    if (j >= j1)
      break;

    while (j >= row_end) {
      if (acc != 0.0f)
        atomicAdd(&y[row], acc);
      acc = 0.0f;
      ++row;
      if (row >= n)
        goto done;
      row_end = rp[row + 1];
    }

    if (j >= j1)
      break;

    const int col = ci[j];
    acc += a[j] * x[col];
  }

done:
  if (row < n && acc != 0.0f) {
    atomicAdd(&y[row], acc);
  }
}

extern "C" void merge(int n, const int *rp, const int *ci, const float *a,
                      const float *x, float *y) {
  const int threads = 128;
  const int warps_per_block = threads / WARP_SIZE;

  int nnz_total = 0;
  cudaMemcpy(&nnz_total, rp + n, sizeof(int), cudaMemcpyDeviceToHost);
  if (nnz_total <= 0) {
    return;
  }

  const int nnz_per_warp = 4096;
  int total_warps = (nnz_total + nnz_per_warp - 1) / nnz_per_warp;
  if (total_warps < 1)
    total_warps = 1;

  int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
  if (blocks < 1)
    blocks = 1;

  merge_kernel<<<blocks, threads>>>(n, rp, ci, a, x, y);
}
