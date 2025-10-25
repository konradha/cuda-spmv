#include "spmv.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <string>
#include <vector>

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  return ms;
}
static double gflops_i64(long long nnz, double ms) {
  return (2.0 * double(nnz)) / (ms * 1e6);
}
static double gbytes_i64(int nrows, long long nnz) {
  return double((nrows + 1) * 4 + nnz * (4 + 4 + 4) + nrows * 4) / 1e9;
}

static float median(std::vector<float> v) {
  if (v.empty())
    return 0.f;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n & 1)
    return v[n / 2];
  return 0.5f * (v[n / 2 - 1] + v[n / 2]);
}
static float mean(const std::vector<float> &v) {
  if (v.empty())
    return 0.f;
  double s = 0;
  for (float x : v)
    s += x;
  return float(s / v.size());
}
static float stdev(const std::vector<float> &v, float m) {
  if (v.empty())
    return 0.f;
  double s2 = 0;
  for (float x : v) {
    double d = x - m;
    s2 += d * d;
  }
  return float(std::sqrt(s2 / v.size()));
}

static cusparseSpMVAlg_t to_alg(CuSparseAlg a) {
  switch (a) {
  case CuSparseAlg::CsrAdaptive:
    return CUSPARSE_SPMV_CSR_ALG2;
  case CuSparseAlg::Default:
  default:
    return CUSPARSE_SPMV_ALG_DEFAULT;
  }
}

static void run_cusparse_collect(const CSR32 &A, float *d_x, float *d_y,
                                 int warmup, int repeat, CuSparseAlg alg,
                                 std::vector<float> &samples) {
  cusparseHandle_t h;
  cusparseCreate(&h);
  cusparseSpMatDescr_t mat;
  cusparseDnVecDescr_t x, y;
  int64_t m = A.rows, n = A.cols, nnz = A.nnz;
  int *d_indptr = nullptr, *d_indices = nullptr;
  float *d_data = nullptr;
  cudaMalloc((void **)&d_indptr, (A.rows + 1) * sizeof(int));
  cudaMalloc((void **)&d_indices, A.nnz * sizeof(int));
  cudaMalloc((void **)&d_data, A.nnz * sizeof(float));
  cudaMemcpy(d_indptr, A.indptr.data(), (A.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, A.indices.data(), A.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, A.data.data(), A.nnz * sizeof(float),
             cudaMemcpyHostToDevice);
  cusparseCreateCsr(&mat, m, n, nnz, d_indptr, d_indices, d_data,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnVec(&x, n, d_x, CUDA_R_32F);
  cusparseCreateDnVec(&y, m, d_y, CUDA_R_32F);
  float alpha = 1.0f, beta = 0.0f;
  size_t buff = 0;
  void *dbuf = nullptr;
  cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, x,
                          &beta, y, CUDA_R_32F, to_alg(alg), &buff);
  cudaMalloc(&dbuf, buff);
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  for (int i = 0; i < warmup; i++) {
    cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, x, &beta, y,
                 CUDA_R_32F, to_alg(alg), dbuf);
    cudaDeviceSynchronize();
  }
  samples.clear();
  samples.reserve(repeat);
  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(s);
    cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, x, &beta, y,
                 CUDA_R_32F, to_alg(alg), dbuf);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    samples.push_back(elapsed_ms(s, e));
  }
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  cudaFree(dbuf);
  cusparseDestroyDnVec(x);
  cusparseDestroyDnVec(y);
  cusparseDestroySpMat(mat);
  cusparseDestroy(h);
  cudaFree(d_indptr);
  cudaFree(d_indices);
  cudaFree(d_data);
}

static void run_kernel_collect(const CSR32 &A, float *d_x, float *d_y,
                               int warmup, int repeat,
                               std::vector<float> &samples) {
  int *d_indptr = nullptr, *d_indices = nullptr;
  float *d_data = nullptr;
  cudaMalloc((void **)&d_indptr, (A.rows + 1) * sizeof(int));
  cudaMalloc((void **)&d_indices, A.nnz * sizeof(int));
  cudaMalloc((void **)&d_data, A.nnz * sizeof(float));
  cudaMemcpy(d_indptr, A.indptr.data(), (A.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, A.indices.data(), A.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, A.data.data(), A.nnz * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  for (int i = 0; i < warmup; i++) {
    kernel_spmv_fp32(A.rows, d_indptr, d_indices, d_data, d_x, d_y);
    cudaDeviceSynchronize();
  }
  samples.clear();
  samples.reserve(repeat);
  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(s);
    kernel_spmv_fp32(A.rows, d_indptr, d_indices, d_data, d_x, d_y);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    samples.push_back(elapsed_ms(s, e));
  }
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  cudaFree(d_indptr);
  cudaFree(d_indices);
  cudaFree(d_data);
}

std::vector<RunRow> benchmark_all(const std::string &mtx_path, int warmup,
                                  int repeat) {
  CSR32 A = load_mtx_to_csr(mtx_path);
  std::string name =
      mtx_path.substr(mtx_path.find_last_of("/\\") == std::string::npos
                          ? 0
                          : mtx_path.find_last_of("/\\") + 1);
  float *dx = nullptr, *dy = nullptr, *dy_ref = nullptr;
  cudaMalloc((void **)&dx, A.cols * sizeof(float));
  cudaMalloc((void **)&dy, A.rows * sizeof(float));
  cudaMalloc((void **)&dy_ref, A.rows * sizeof(float));
  std::vector<float> hx(A.cols);
  for (size_t i = 0; i < hx.size(); ++i)
    hx[i] = float((i * 2654435761u) % 1000) / 1000.0f;
  cudaMemcpy(dx, hx.data(), A.cols * sizeof(float), cudaMemcpyHostToDevice);

  std::vector<float> s_def, s_adp, s_ker;
  run_cusparse_collect(A, dx, dy_ref, warmup, repeat, CuSparseAlg::Default,
                       s_def);
  run_cusparse_collect(A, dx, dy, warmup, repeat, CuSparseAlg::CsrAdaptive,
                       s_adp);
  run_kernel_collect(A, dx, dy, warmup, repeat, s_ker);

  std::vector<float> y0(A.rows), y1(A.rows);
  cudaMemcpy(y0.data(), dy_ref, A.rows * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y1.data(), dy, A.rows * sizeof(float), cudaMemcpyDeviceToHost);
  double diff = 0;
  for (int i = 0; i < A.rows; i++) {
    double d = std::fabs(double(y0[i]) - double(y1[i]));
    if (d > diff)
      diff = d;
  }

  float m_def = mean(s_def), med_def = median(s_def),
        sd_def = stdev(s_def, m_def);
  float m_adp = mean(s_adp), med_adp = median(s_adp),
        sd_adp = stdev(s_adp, m_adp);
  float best_cusp = std::min(m_def, m_adp);
  float m_ker = mean(s_ker), med_ker = median(s_ker),
        sd_ker = stdev(s_ker, m_ker);

  double gb = gbytes_i64(A.rows, A.nnz);

  RunRow r1;
  r1.matrix_name = name;
  r1.rows = A.rows;
  r1.cols = A.cols;
  r1.nnz = A.nnz;
  r1.method = "cusparse_default";
  r1.stats.ms_mean = m_def;
  r1.stats.ms_median = med_def;
  r1.stats.ms_std = sd_def;
  r1.stats.gflops = gflops_i64(A.nnz, m_def);
  r1.stats.gbps = gb / (m_def / 1000.0);
  r1.stats.check_max_abs = -1.0;
  r1.stats.pct_of_best_cusparse = 100.0;
  r1.stats.samples_ms = s_def;

  RunRow r2;
  r2.matrix_name = name;
  r2.rows = A.rows;
  r2.cols = A.cols;
  r2.nnz = A.nnz;
  r2.method = "cusparse_csr_adaptive";
  r2.stats.ms_mean = m_adp;
  r2.stats.ms_median = med_adp;
  r2.stats.ms_std = sd_adp;
  r2.stats.gflops = gflops_i64(A.nnz, m_adp);
  r2.stats.gbps = gb / (m_adp / 1000.0);
  r2.stats.check_max_abs = -1.0;
  r2.stats.pct_of_best_cusparse = 100.0 * (best_cusp / std::min(m_def, m_adp));
  r2.stats.samples_ms = s_adp;

  RunRow r3;
  r3.matrix_name = name;
  r3.rows = A.rows;
  r3.cols = A.cols;
  r3.nnz = A.nnz;
  r3.method = "kernel_baseline";
  r3.stats.ms_mean = m_ker;
  r3.stats.ms_median = med_ker;
  r3.stats.ms_std = sd_ker;
  r3.stats.gflops = gflops_i64(A.nnz, m_ker);
  r3.stats.gbps = gb / (m_ker / 1000.0);
  r3.stats.check_max_abs = diff;
  r3.stats.pct_of_best_cusparse = 100.0 * (best_cusp / m_ker);
  r3.stats.samples_ms = s_ker;

  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dy_ref);
  return {r1, r2, r3};
}
