// Explicit AVX2 intrinsics, ikj loop order.
//
// For each (i,k): broadcast A[i*K+k] into a 256-bit lane (8x float), then walk
// 8 elements of B's row k and C's row i at a time:
//   C[i][j..j+7] += A[i][k] * B[k][j..j+7]
// via _mm256_fmadd_ps. B and C are stride-1 along j, so these are contiguous
// unaligned loads/stores. N not divisible by 8 is handled by a scalar tail
// loop over the remaining j after the last full 8-wide block.
#include "matmul.h"

#include <cstddef>
#include <cstring>
#include <immintrin.h>

namespace mm {
namespace {

// run() is called repeatedly (warmup + every timed rep) without the harness
// re-zeroing C in between, and this kernel accumulates into C via +=, so it
// must zero its own output first.
void run(const float* A, const float* B, float* C, int M, int N, int K) {
  std::memset(C, 0, static_cast<std::size_t>(M) * N * sizeof(float));

  const int N8 = N - (N % 8);

  for (int i = 0; i < M; ++i) {
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
      // Scalar tail for N % 8 != 0.
      for (; j < N; ++j) {
        crow[j] += arow[k] * brow[j];
      }
    }
  }
}

}  // namespace

REGISTER_MATMUL(Variant{
    "4 simd avx2", Kind::Cpu,
    /*prepare=*/nullptr,
    /*run=*/run,
    /*fetch=*/nullptr,
    /*teardown=*/nullptr});

}  // namespace mm
