#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct CSR32 {
  int rows;
  int cols;
  int64_t nnz;
  std::vector<int> indptr;
  std::vector<int> indices;
  std::vector<float> data;
};

CSR32 load_mtx_to_csr(const std::string &path);

enum class CuSparseAlg { Default, CsrAdaptive };

struct RunStats {
  float ms_mean;
  float ms_median;
  float ms_std;
  double gflops;
  double gbps;
  double check_max_abs;
  double pct_of_best_cusparse;
  std::vector<float> samples_ms;
};

struct RunRow {
  std::string matrix_name;
  int rows;
  int cols;
  int64_t nnz;
  std::string method;
  RunStats stats;
};

#ifdef __cplusplus
extern "C" {
#endif
void kernel_spmv_fp32(int num_rows, const int *row_ptr, const int *col_ind,
                      const float *vals, const float *x, float *y);
#ifdef __cplusplus
}
#endif

std::vector<RunRow> benchmark_all(const std::string &mtx_path, int warmup,
                                  int repeat);
