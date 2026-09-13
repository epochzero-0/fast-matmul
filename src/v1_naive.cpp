#include "matmul.h"

namespace mm {

// Correctness oracle and the baseline row. Plain i,j,k triple loop, no tricks.
void matmul_naive_reference(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
}

namespace {
void run(const float* A, const float* B, float* C, int M, int N, int K) {
  matmul_naive_reference(A, B, C, M, N, K);
}
}  // namespace

REGISTER_MATMUL(Variant{
    "1 naive ijk", Kind::Cpu,
    /*prepare=*/nullptr,
    /*run=*/run,
    /*fetch=*/nullptr,
    /*teardown=*/nullptr});

}  // namespace mm
