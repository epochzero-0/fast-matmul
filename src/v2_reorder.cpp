#include "matmul.h"

namespace mm {
namespace {

void run(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * N + j] = 0.0f;
    }
  }
  // i-k-j order: the inner loop walks B and C with stride 1 (row k of B, row i of C), unlike naive ijk which walks B with stride N.
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float a = A[i * K + k];
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
}

}  // namespace

REGISTER_MATMUL(Variant{
    "2 reorder ikj", Kind::Cpu,
    /*prepare=*/nullptr,
    /*run=*/run,
    /*fetch=*/nullptr,
    /*teardown=*/nullptr});

}  // namespace mm
