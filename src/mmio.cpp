#include "spmv.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct Triplet {
  int r;
  int c;
  float v;
};

static bool is_comment_line(const char *s) { return s[0] == '%'; }
static bool is_blank_line(const char *s) {
  for (const char *p = s; *p; ++p) {
    if (!std::isspace((unsigned char)*p))
      return false;
  }
  return true;
}

CSR32 load_mtx_to_csr(const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error("open fail");
  char buf[1024];
  if (!std::fgets(buf, sizeof(buf), f)) {
    std::fclose(f);
    throw std::runtime_error("read fail: empty");
  }
  if (std::strncmp(buf, "%%MatrixMarket", 14) != 0) {
    std::fclose(f);
    throw std::runtime_error("bad header");
  }
  bool symmetric = std::strstr(buf, "symmetric");
  bool is_pattern = std::strstr(buf, "pattern");
  bool is_integer = std::strstr(buf, "integer");
  do {
    if (!std::fgets(buf, sizeof(buf), f)) {
      std::fclose(f);
      throw std::runtime_error("header fail");
    }
  } while (is_comment_line(buf) || is_blank_line(buf));
  int M = 0, N = 0;
  long long L = 0;
  if (std::sscanf(buf, "%d %d %lld", &M, &N, &L) != 3) {
    std::fclose(f);
    throw std::runtime_error("size fail");
  }
  std::vector<Triplet> T;
  T.reserve(L * (symmetric ? 2 : 1));
  long long read = 0;
  while (read < L) {
    if (!std::fgets(buf, sizeof(buf), f)) {
      std::fclose(f);
      throw std::runtime_error("entry fail");
    }
    if (is_comment_line(buf) || is_blank_line(buf))
      continue;
    int r = 0, c = 0;
    if (is_pattern) {
      if (std::sscanf(buf, "%d %d", &r, &c) != 2)
        continue;
      float v = 1.0f;
      r--;
      c--;
      T.push_back({r, c, v});
      if (symmetric && r != c)
        T.push_back({c, r, v});
      read++;
    } else if (is_integer) {
      long long iv = 0;
      if (std::sscanf(buf, "%d %d %lld", &r, &c, &iv) != 3)
        continue;
      float v = float(iv);
      r--;
      c--;
      T.push_back({r, c, v});
      if (symmetric && r != c)
        T.push_back({c, r, v});
      read++;
    } else {
      double dv = 0.0;
      if (std::sscanf(buf, "%d %d %lf", &r, &c, &dv) != 3)
        continue;
      float v = float(dv);
      r--;
      c--;
      T.push_back({r, c, v});
      if (symmetric && r != c)
        T.push_back({c, r, v});
      read++;
    }
  }
  std::fclose(f);
  std::sort(T.begin(), T.end(), [](const Triplet &a, const Triplet &b) {
    if (a.r != b.r)
      return a.r < b.r;
    return a.c < b.c;
  });
  std::vector<Triplet> U;
  U.reserve(T.size());
  for (size_t i = 0; i < T.size();) {
    size_t j = i + 1;
    float sv = T[i].v;
    while (j < T.size() && T[j].r == T[i].r && T[j].c == T[i].c) {
      sv += T[j].v;
      j++;
    }
    U.push_back({T[i].r, T[i].c, sv});
    i = j;
  }
  CSR32 C;
  C.rows = M;
  C.cols = N;
  C.nnz = (int64_t)U.size();
  C.indptr.assign(M + 1, 0);
  for (auto &t : U)
    C.indptr[t.r + 1]++;
  for (int i = 0; i < M; i++)
    C.indptr[i + 1] += C.indptr[i];
  C.indices.assign(C.nnz, 0);
  C.data.assign(C.nnz, 0);
  std::vector<int> ctr = C.indptr;
  for (auto &t : U) {
    int p = ctr[t.r]++;
    C.indices[p] = t.c;
    C.data[p] = t.v;
  }
  return C;
}
