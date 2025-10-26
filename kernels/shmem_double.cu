#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef CHUNK_NNZ
#define CHUNK_NNZ 1024
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define CPASYNC_CAPABLE 1
#else
#define CPASYNC_CAPABLE 0
#endif

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

#if CPASYNC_CAPABLE
__device__ __forceinline__ void cp_async_16(void *smem, const void *gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem),
               "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

__global__ void cpasync_db_kernel(int n, const int *__restrict__ rp,
                                  const int *__restrict__ ci,
                                  const float *__restrict__ a,
                                  const float *__restrict__ x,
                                  float *__restrict__ y) {
  const int tid = threadIdx.x;
  const int warps_per_block = blockDim.x / WARP_SIZE;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & (WARP_SIZE - 1);

  const int nnz_total = rp[n];
  if (nnz_total <= 0)
    return;

  const int tiles = (nnz_total + CHUNK_NNZ - 1) / CHUNK_NNZ;

  extern __shared__ unsigned char smem_raw[];
  float *sm_a0 = (float *)smem_raw;
  int *sm_ci0 = (int *)(sm_a0 + CHUNK_NNZ);
  float *sm_a1 = (float *)(sm_ci0 + CHUNK_NNZ);

  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    int j_chunk_base = tile * CHUNK_NNZ;
    int j_chunk_end = min(j_chunk_base + CHUNK_NNZ, nnz_total);
    int chunk_len = j_chunk_end - j_chunk_base;
    if (chunk_len <= 0)
      continue;

#if CPASYNC_CAPABLE
    int vec16 = (chunk_len + 3) >> 2;
    for (int v = tid; v < vec16; v += blockDim.x) {
      int j = j_chunk_base + (v << 2);
      cp_async_16(&sm_a0[v << 2], &a[j]);
      cp_async_16(&sm_ci0[v << 2], &ci[j]);
    }
    cp_async_commit();
    cp_async_wait();
    __syncthreads();
#else
    for (int j = tid; j < chunk_len; j += blockDim.x) {
      sm_a0[j] = a[j_chunk_base + j];
      sm_ci0[j] = ci[j_chunk_base + j];
    }
    __syncthreads();
#endif

    int row_begin = upper_bound_nnz(rp, n, j_chunk_base - 1);
    int row_end_excl = upper_bound_nnz(rp, n, j_chunk_end - 1);
    if (row_begin >= n)
      continue;
    row_end_excl = min(row_end_excl + 1, n);

    for (int row = row_begin + warp_id; row < row_end_excl;
         row += warps_per_block) {
      int row_lo = rp[row];
      int row_hi = rp[row + 1];

      int local_lo = max(row_lo, j_chunk_base) - j_chunk_base;
      int local_hi = min(row_hi, j_chunk_end) - j_chunk_base;
      if (local_lo >= local_hi)
        continue;

      float acc = 0.0f;
      for (int t = local_lo + lane; t < local_hi; t += WARP_SIZE) {
        int col = sm_ci0[t];
        acc += sm_a0[t] * __ldg(&x[col]);
      }
      acc = warp_reduce_sum(acc);

      if (lane == 0) {
        bool full_row = (row_lo >= j_chunk_base) && (row_hi <= j_chunk_end);
        if (full_row) {
          y[row] = acc;
        } else {
          atomicAdd(&y[row], acc);
        }
      }
    }
    __syncthreads();
  }
}

extern "C" void cpasync_double(int n, const int *rp, const int *ci,
                               const float *a, const float *x, float *y) {
  const int threads = THREADS_PER_BLOCK;
  int blocks = 128;
  size_t smem_bytes = (sizeof(float) + sizeof(int)) * (size_t)CHUNK_NNZ * 2;
  cpasync_db_kernel<<<blocks, threads, smem_bytes>>>(n, rp, ci, a, x, y);
}
