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
                 float *__restrict__ y, int *row_counter) {

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

    sum = warp_reduce_sum(sum, V);

    if (laneId == 0) {
      y[row] = sum;
    }

    row = getRowIndexWarp<V>(row_counter);
  }
}

struct LightSpMVContext {
  int *row_counter;
  int T;
  int B;
  bool initialized;

  LightSpMVContext() : row_counter(nullptr), T(0), B(0), initialized(false) {}

  void init() {
    if (initialized)
      return;

    cudaMalloc(&row_counter, sizeof(int));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    T = prop.maxThreadsPerBlock;
    B = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor / T;

    initialized = true;
  }

  ~LightSpMVContext() {
    if (initialized && row_counter) {
      cudaFree(row_counter);
    }
  }
};

static LightSpMVContext g_lightspmv_ctx;

template <int V>
static void lightspmv_fixed_impl(int n, const int *rp, const int *ci,
                                 const float *a, const float *x, float *y) {

  g_lightspmv_ctx.init();

  cudaMemsetAsync(g_lightspmv_ctx.row_counter, 0, sizeof(int));
  lightspmv_kernel<V><<<g_lightspmv_ctx.B, g_lightspmv_ctx.T>>>(
      n, rp, ci, a, x, y, g_lightspmv_ctx.row_counter);
}

extern "C" void lightspmv_v2(int n, const int *rp, const int *ci,
                             const float *a, const float *x, float *y) {
  lightspmv_fixed_impl<2>(n, rp, ci, a, x, y);
}

extern "C" void lightspmv_v4(int n, const int *rp, const int *ci,
                             const float *a, const float *x, float *y) {
  lightspmv_fixed_impl<4>(n, rp, ci, a, x, y);
}

extern "C" void lightspmv_v8(int n, const int *rp, const int *ci,
                             const float *a, const float *x, float *y) {
  lightspmv_fixed_impl<8>(n, rp, ci, a, x, y);
}

extern "C" void lightspmv_v32(int n, const int *rp, const int *ci,
                              const float *a, const float *x, float *y) {
  lightspmv_fixed_impl<32>(n, rp, ci, a, x, y);
}

extern "C" void lightspmv(int n, const int *rp, const int *ci, const float *a,
                          const float *x, float *y) {

  g_lightspmv_ctx.init();

  static int cached_n = -1;
  static int cached_mean = 0;

  if (cached_n != n) {
    int h_rp_end;
    cudaMemcpy(&h_rp_end, &rp[n], sizeof(int), cudaMemcpyDeviceToHost);
    cached_mean = (n > 0) ? ((h_rp_end + n - 1) / n) : 0;
    cached_n = n;
  }

  if (cached_mean <= 2) {
    lightspmv_fixed_impl<2>(n, rp, ci, a, x, y);
  } else if (cached_mean <= 4) {
    lightspmv_fixed_impl<4>(n, rp, ci, a, x, y);
  } else if (cached_mean <= 64) {
    lightspmv_fixed_impl<8>(n, rp, ci, a, x, y);
  } else {
    lightspmv_fixed_impl<32>(n, rp, ci, a, x, y);
  }
}
