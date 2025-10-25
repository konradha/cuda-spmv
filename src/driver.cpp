#include "spmv.h"
#include <cstdlib>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " dataset.mtx num_warmup num_iter\n";
    return 1;
  }
  std::string path = argv[1];
  int warmup = std::atoi(argv[2]);
  int repeat = std::atoi(argv[3]);

  std::cout << "matrix,rows,cols,nnz,method,ms_mean,ms_median,ms_std,gflops,"
               "gbps,check_max_abs,pct_of_best_cusparse,samples_ms\n";

  try {
    auto rows = benchmark_all(path, warmup, repeat);
    for (auto &r : rows) {
      std::ostringstream s;
      for (size_t i = 0; i < r.stats.samples_ms.size(); ++i) {
        if (i)
          s << ";";
        s << r.stats.samples_ms[i];
      }
      std::cout << r.matrix_name << "," << r.rows << "," << r.cols << ","
                << r.nnz << "," << r.method << "," << r.stats.ms_mean << ","
                << r.stats.ms_median << "," << r.stats.ms_std << ","
                << r.stats.gflops << "," << r.stats.gbps << ","
                << r.stats.check_max_abs << "," << r.stats.pct_of_best_cusparse
                << "," << s.str() << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
