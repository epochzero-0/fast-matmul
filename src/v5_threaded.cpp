// std::thread over the same AVX2 kernel as v4, partitioning output rows (the
// i dimension) across threads. Each thread owns a disjoint, contiguous band
// of rows [row_begin, row_end) of C, so there is no shared-write race and no
// locking is needed.
//
// Thread count defaults to std::thread::hardware_concurrency(), overridable
// by the MATMUL_THREADS environment variable (used for the required
// thread-scaling sweep).
#include "matmul.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <thread>
#include <vector>

namespace mm {
namespace {

// Same ikj + AVX2 FMA kernel as v4_simd.cpp, duplicated here (not #included)
// so this file stays self-contained. Operates on a row band [row_begin, row_end).
void simd_band(const float* A, const float* B, float* C, int N, int K,
               int row_begin, int row_end) {
  const int N8 = N - (N % 8);

  for (int i = row_begin; i < row_end; ++i) {
    float* crow = C + static_cast<std::size_t>(i) * N;
    const float* arow = A + static_cast<std::size_t>(i) * K;

    for (int k = 0; k < K; ++k) {
      const __m256 a = _mm256_set1_ps(arow[k]);
      const float* brow = B + static_cast<std::size_t>(k) * N;

      int j = 0;
      for (; j < N8; j += 8) {
        __m256 c = _mm256_loadu_ps(crow + j);
        __m256 b = _mm256_loadu_ps(brow + j);
        c = _mm256_fmadd_ps(a, b, c);
        _mm256_storeu_ps(crow + j, c);
      }
      for (; j < N; ++j) {
        crow[j] += arow[k] * brow[j];
      }
    }
  }
}

unsigned thread_count() {
  if (const char* env = std::getenv("MATMUL_THREADS")) {
    int n = std::atoi(env);
    if (n > 0) return static_cast<unsigned>(n);
  }
  unsigned hc = std::thread::hardware_concurrency();
  return hc ? hc : 1u;
}

void run(const float* A, const float* B, float* C, int M, int N, int K) {
  std::memset(C, 0, static_cast<std::size_t>(M) * N * sizeof(float));

  unsigned nthreads = thread_count();
  if (nthreads > static_cast<unsigned>(M)) nthreads = static_cast<unsigned>(M);
  if (nthreads <= 1) {
    simd_band(A, B, C, N, K, 0, M);
    return;
  }

  std::vector<std::thread> pool;
  pool.reserve(nthreads);

  int rows_per = M / static_cast<int>(nthreads);
  int remainder = M % static_cast<int>(nthreads);
  int row = 0;
  for (unsigned t = 0; t < nthreads; ++t) {
    int band = rows_per + (static_cast<int>(t) < remainder ? 1 : 0);
    int row_begin = row;
    int row_end = row + band;
    row = row_end;
    pool.emplace_back(simd_band, A, B, C, N, K, row_begin, row_end);
  }
  for (auto& th : pool) th.join();
}

}  // namespace

REGISTER_MATMUL(Variant{
    "5 threads", Kind::Cpu,
    /*prepare=*/nullptr,
    /*run=*/run,
    /*fetch=*/nullptr,
    /*teardown=*/nullptr});

}  // namespace mm
