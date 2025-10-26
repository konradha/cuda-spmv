/*

LightSpMV: Faster CSR-based Sparse Matrix-Vector
Multiplication on CUDA-enabled GPUs

Liu, Schmidt

   */

#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__device__ inline float warp_reduce_sum(float val, int width) {
  for (int offset = width >> 1; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffu, val, offset, width);
  }
  return val;
}

template <int V> __device__ int getRowIndexWarp(int *row_counter) {
  const int warpLaneId = threadIdx.x & (WARP_SIZE - 1);
  const int warpVectorId = warpLaneId / V;

  int row;
  if (warpLaneId == 0) {
    row = atomicAdd(row_counter, WARP_SIZE / V);
  }
  row = __shfl_sync(0xffffffffu, row, 0, WARP_SIZE);
  return row + warpVectorId;
}

template <int V>
__global__ void
lightspmv_kernel(int n, const int *__restrict__ rp, const int *__restrict__ ci,
                 const float *__restrict__ a, const float *__restrict__ x,
                 float *__restrict__ y, int *row_counter, float alpha,
                 float beta) {

  const int laneId = threadIdx.x % V;

  int row = getRowIndexWarp<V>(row_counter);

  while (row < n) {
    const int row_start = rp[row];
    const int row_end = rp[row + 1];

    float sum = 0.0f;

    if (V == WARP_SIZE) {
      int i = row_start - (row_start & (V - 1)) + laneId;
      if (i >= row_start && i < row_end) {
        sum += a[i] * x[ci[i]];
      }
      for (i += V; i < row_end; i += V) {
        sum += a[i] * x[ci[i]];
      }
    } else {
      for (int i = row_start + laneId; i < row_end; i += V) {
        sum += a[i] * x[ci[i]];
      }
    }

    sum *= alpha;
    sum = warp_reduce_sum(sum, V);

    if (laneId == 0) {
      y[row] = sum + beta * y[row];
    }

    row = getRowIndexWarp<V>(row_counter);
  }
}

extern "C" void lightspmv(int n, const int *rp, const int *ci, const float *a,
                          const float *x, float *y) {

  double alpha = 1.;
  double beta = 0.;

  int *row_counter;
  cudaMalloc(&row_counter, sizeof(int));
  cudaMemset(row_counter, 0, sizeof(int));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  const int T = prop.maxThreadsPerBlock;
  const int B = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor / T;

  int64_t total_nnz = 0;
  cudaMemcpy(&total_nnz, &rp[n], sizeof(int), cudaMemcpyDeviceToHost);

  int mean = (n > 0) ? ((total_nnz + n - 1) / n) : 0;

  if (mean <= 2) {
    lightspmv_kernel<2><<<B, T>>>(n, rp, ci, a, x, y, row_counter, alpha, beta);
  } else if (mean <= 4) {
    lightspmv_kernel<4><<<B, T>>>(n, rp, ci, a, x, y, row_counter, alpha, beta);
  } else if (mean <= 64) {
    lightspmv_kernel<8><<<B, T>>>(n, rp, ci, a, x, y, row_counter, alpha, beta);
  } else {
    lightspmv_kernel<32>
        <<<B, T>>>(n, rp, ci, a, x, y, row_counter, alpha, beta);
  }

  cudaFree(row_counter);
}
