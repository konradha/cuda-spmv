#include <cuda_runtime.h>

__global__ void spmv_naive_kernel(int n,
                                  const int* __restrict__ rp,
                                  const int* __restrict__ ci,
                                  const float* __restrict__ a,
                                  const float* __restrict__ x,
                                  float* __restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int s = rp[row];
    int e = rp[row + 1];
    float sum = 0.0f;
    for (int j = s; j < e; ++j) sum += a[j] * x[ci[j]];
    y[row] = sum;
}

extern "C" void kernel_spmv_fp32(int n,
                                 const int* rp,
                                 const int* ci,
                                 const float* a,
                                 const float* x,
                                 float* y) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    spmv_naive_kernel<<<blocks, threads>>>(n, rp, ci, a, x, y);
}

