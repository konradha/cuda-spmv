#include "spmv.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  cudaEventElapsedTime(&ms, a, b);
  return ms;
}

static double gflops_i64(long long nnz, double ms) {

  return (2.0 * double(nnz)) / (ms * 1e6);
}

static double gbytes_i64(int nrows, long long nnz) {
  // values + col_indices + row_ptr + x_reads + y_writes
  // = (nnz*4) + (nnz*4) + ((rows+1)*4) + (nnz*4) + (rows*4)
  long long bytes = (long long)(nrows + 1) * 4LL + nnz * 4LL + nnz * 4LL +
                    nnz * 4LL + (long long)(nrows) * 4LL;
  return double(bytes) / 1e9;
}
static float median(std::vector<float> v) {
  if (v.empty())
    return 0.f;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  if (n & 1)
    return v[n / 2];
  return 0.5f * (v[n / 2 - 1] + v[n / 2]);
}
static float mean(const std::vector<float> &v) {
  if (v.empty())
    return 0.f;
  double s = 0.0;
  for (float x : v)
    s += x;
  return float(s / v.size());
}
static float stdev(const std::vector<float> &v, float m) {
  if (v.empty())
    return 0.f;
  double s2 = 0.0;
  for (float x : v) {
    const double d = double(x) - double(m);
    s2 += d * d;
  }
  return float(std::sqrt(s2 / v.size()));
}

static cusparseSpMVAlg_t to_alg(CuSparseAlg a) {
  switch (a) {
  case CuSparseAlg::CsrAdaptive: // TODO rename
    return CUSPARSE_SPMV_CSR_ALG2;
  case CuSparseAlg::Default:
  default:
    return CUSPARSE_SPMV_ALG_DEFAULT;
  }
}

struct DeviceCSR {
  int rows = 0, cols = 0;
  long long nnz = 0;
  int *d_indptr = nullptr;
  int *d_indices = nullptr;
  float *d_values = nullptr;
  float *d_x = nullptr;
  float *d_y = nullptr;
  float *d_y_ref = nullptr;

  ~DeviceCSR() {
    cudaFree(d_indptr);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y_ref);
  }
};

// baseline
extern "C" void kernel_spmv_fp32(int rows, const int *indptr,
                                 const int *indices, const float *values,
                                 const float *x, float *y);

extern "C" void warp_per_row(int rows, const int *indptr, const int *indices,
                             const float *values, const float *x, float *y);

extern "C" void spmv_multiwarp(int rows, const int *indptr, const int *indices,
                               const float *values, const float *x, float *y);

extern "C" void warp_vectorized(int rows, const int *indptr, const int *indices,
                                const float *values, const float *x, float *y);

extern "C" void merge(int rows, const int *indptr, const int *indices,
                      const float *values, const float *x, float *y);

extern "C" void cpasync_double(int rows, const int *indptr, const int *indices,
                               const float *values, const float *x, float *y);

extern "C" void pqueue(int rows, const int *indptr, const int *indices,
                       const float *values, const float *x, float *y);

extern "C" void lightspmv(int rows, const int *indptr, const int *indices,
                          const float *values, const float *x, float *y);

using SpmvLauncher = void (*)(int, const int *, const int *, const float *,
                              const float *, float *);

struct KernelSpec {
  const char *name;
  SpmvLauncher fn;
};

static const KernelSpec kKernels[] = {
    {"baseline", &kernel_spmv_fp32},
    {"warp_per_row", &warp_per_row},
    {"multiwarp_per_row", &spmv_multiwarp},
    {"warp_vectorized", &warp_vectorized},
    {"merge", &merge},
    {"cpasync_double", &cpasync_double},
    {"lightspmv", &lightspmv},
    // {"persistent_queue", &pqueue}
};

static void run_cusparse_collect(const DeviceCSR &D, int warmup, int repeat,
                                 CuSparseAlg alg, std::vector<float> &samples) {
  cusparseHandle_t h = nullptr;
  cusparseCreate(&h);

  cusparseSpMatDescr_t mat = nullptr;
  cusparseDnVecDescr_t x = nullptr, y = nullptr;

  const int64_t m = D.rows, n = D.cols, nnz = D.nnz;

  cusparseCreateCsr(&mat, m, n, nnz, D.d_indptr, D.d_indices, D.d_values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnVec(&x, n, (void *)D.d_x, CUDA_R_32F);
  cusparseCreateDnVec(&y, m, (void *)D.d_y_ref, CUDA_R_32F);

  float alpha = 1., beta = 0.;
  size_t buff = 0;
  cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, x,
                          &beta, y, CUDA_R_32F, to_alg(alg), &buff);

  void *dbuf = nullptr;
  cudaMalloc(&dbuf, buff);

  samples.clear();
  samples.reserve(repeat);
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  for (int i = -warmup; i < repeat; ++i) {
    cudaMemset(D.d_y_ref, 0, D.rows * sizeof(float));
    cudaEventRecord(s);
    cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, x, &beta, y,
                 CUDA_R_32F, to_alg(alg), dbuf);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    if (i >= 0)
      samples.push_back(elapsed_ms(s, e));
  }
  cudaEventDestroy(s);
  cudaEventDestroy(e);

  cudaFree(dbuf);
  cusparseDestroyDnVec(x);
  cusparseDestroyDnVec(y);
  cusparseDestroySpMat(mat);
  cusparseDestroy(h);
}

static void run_launcher_collect(const DeviceCSR &D, int warmup, int repeat,
                                 SpmvLauncher launch,
                                 std::vector<float> &samples) {
  for (int i = 0; i < warmup; ++i) {
    cudaMemset(D.d_y, 0, D.rows * sizeof(float));
    launch(D.rows, D.d_indptr, D.d_indices, D.d_values, D.d_x, D.d_y);
    cudaDeviceSynchronize();
  }

  samples.clear();
  samples.reserve(repeat);
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  for (int i = 0; i < repeat; ++i) {
    cudaMemset(D.d_y, 0, D.rows * sizeof(float));
    cudaEventRecord(s);
    launch(D.rows, D.d_indptr, D.d_indices, D.d_values, D.d_x, D.d_y);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    samples.push_back(elapsed_ms(s, e));
  }
  cudaEventDestroy(s);
  cudaEventDestroy(e);
}

std::vector<RunRow> benchmark_all(const std::string &mtx_path, int warmup,
                                  int repeat) {
  CSR32 A = load_mtx_to_csr(mtx_path);
  const std::string name =
      mtx_path.substr(mtx_path.find_last_of("/\\") == std::string::npos
                          ? 0
                          : mtx_path.find_last_of("/\\") + 1);

  std::vector<float> hx(A.cols);
  for (size_t i = 0; i < hx.size(); ++i)
    hx[i] = float((i) % 1000) / 1000.0f;

  DeviceCSR D;
  D.rows = A.rows;
  D.cols = A.cols;
  D.nnz = A.nnz;

  cudaMalloc((void **)&D.d_indptr, (A.rows + 1) * sizeof(int));
  cudaMalloc((void **)&D.d_indices, A.nnz * sizeof(int));
  cudaMalloc((void **)&D.d_values, A.nnz * sizeof(float));
  cudaMalloc((void **)&D.d_x, A.cols * sizeof(float));
  cudaMalloc((void **)&D.d_y, A.rows * sizeof(float));
  cudaMalloc((void **)&D.d_y_ref, A.rows * sizeof(float));

  cudaMemcpy(D.d_indptr, A.indptr.data(), (A.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(D.d_indices, A.indices.data(), A.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(D.d_values, A.data.data(), A.nnz * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(D.d_x, hx.data(), A.cols * sizeof(float), cudaMemcpyHostToDevice);

  std::vector<float> s_def, s_adp;
  run_cusparse_collect(D, warmup, repeat, CuSparseAlg::Default, s_def);
  run_cusparse_collect(D, warmup, repeat, CuSparseAlg::CsrAdaptive, s_adp);

  const float m_def = mean(s_def), med_def = median(s_def),
              sd_def = stdev(s_def, m_def);
  const float m_adp = mean(s_adp), med_adp = median(s_adp),
              sd_adp = stdev(s_adp, m_adp);

  const float best_cusp_ms = std::min(m_def, m_adp);
  const double gb = gbytes_i64(A.rows, A.nnz);

  std::vector<RunRow> out;

  {
    RunRow r{};
    r.matrix_name = name;
    r.rows = A.rows;
    r.cols = A.cols;
    r.nnz = A.nnz;
    r.method = "cusparse_default";
    r.stats.ms_mean = m_def;
    r.stats.ms_median = med_def;
    r.stats.ms_std = sd_def;
    r.stats.gflops = gflops_i64(A.nnz, m_def);
    r.stats.gbps = gb / (double(m_def) / 1000.0);
    r.stats.check_max_abs = -1.0;
    r.stats.pct_of_best_cusparse = 100.0 * (best_cusp_ms / m_def);
    r.stats.samples_ms = s_def;
    out.emplace_back(std::move(r));
  }
  {
    RunRow r{};
    r.matrix_name = name;
    r.rows = A.rows;
    r.cols = A.cols;
    r.nnz = A.nnz;
    r.method = "cusparse_csr_alg2";
    r.stats.ms_mean = m_adp;
    r.stats.ms_median = med_adp;
    r.stats.ms_std = sd_adp;
    r.stats.gflops = gflops_i64(A.nnz, m_adp);
    r.stats.gbps = gb / (double(m_adp) / 1000.0);
    r.stats.check_max_abs = -1.0;
    r.stats.pct_of_best_cusparse = 100.0 * (best_cusp_ms / m_adp);
    r.stats.samples_ms = s_adp;
    out.emplace_back(std::move(r));
  }

  std::vector<float> y_ref(A.rows);
  cudaMemcpy(y_ref.data(), D.d_y_ref, A.rows * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (const auto &spec : kKernels) {
    std::vector<float> s;
    run_launcher_collect(D, warmup, repeat, spec.fn, s);

    const float m = mean(s), med = median(s), sd = stdev(s, m);

    std::vector<float> y(A.rows);
    cudaMemcpy(y.data(), D.d_y, A.rows * sizeof(float), cudaMemcpyDeviceToHost);
    double diff = 0.0;
    for (int i = 0; i < A.rows; ++i) {
      const double d = std::fabs(double(y[i]) - double(y_ref[i]));
      if (d > diff)
        diff = d;
    }

    RunRow r{};
    r.matrix_name = name;
    r.rows = A.rows;
    r.cols = A.cols;
    r.nnz = A.nnz;
    r.method = spec.name;
    r.stats.ms_mean = m;
    r.stats.ms_median = med;
    r.stats.ms_std = sd;
    r.stats.gflops = gflops_i64(A.nnz, m);
    r.stats.gbps = gb / (double(m) / 1000.0);
    r.stats.check_max_abs = diff;
    r.stats.pct_of_best_cusparse = 100.0 * (best_cusp_ms / m);
    r.stats.samples_ms = std::move(s);

    out.emplace_back(std::move(r));
  }

  return out;
}
