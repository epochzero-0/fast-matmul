// v8: cuBLAS SGEMM.
//
// cuBLAS is column-major; this project is row-major throughout. The fix is
// not to transpose any data, only to reinterpret it and swap operands:
//
//   A is M x K row-major, ld=K   -> that same buffer, read as column-major,
//                                    is the K x M matrix A^T (ld=K still
//                                    matches, since a row-major M x K matrix
//                                    and a column-major K x M matrix with the
//                                    same ld are the identical byte layout).
//   B is K x N row-major, ld=N   -> read as column-major, is N x K, i.e. B^T.
//   C is M x N row-major, ld=N   -> read as column-major, is N x M, i.e. C^T.
//
// We want C = A * B. Transposing both sides of that identity gives
//   C^T = B^T * A^T
// which is exactly a plain (no-transpose) column-major GEMM between the
// column-major reinterpretations of B and A, producing the column-major
// reinterpretation of C -- i.e. exactly the row-major C we need, with zero
// actual data transposition. So we call cublasSgemm with A and B swapped,
// op = CUBLAS_OP_N on both, dimensions (N, M, K), and leading dimensions
// N, K, N (the row-major strides of B, A, C respectively).

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matmul.h"

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err_ = (expr);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err_));                                \
      std::exit(1);                                                         \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(expr)                                                   \
  do {                                                                       \
    cublasStatus_t st_ = (expr);                                             \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__,   \
                   __LINE__, static_cast<int>(st_));                        \
      std::exit(1);                                                         \
    }                                                                        \
  } while (0)

namespace mm {
namespace {

float* d_A = nullptr;
float* d_B = nullptr;
float* d_C = nullptr;
int g_M = 0, g_N = 0, g_K = 0;
cublasHandle_t g_handle = nullptr;

void prepare(const float* A, const float* B, int M, int N, int K) {
  g_M = M;
  g_N = N;
  g_K = K;
  size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
  size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
  size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
  CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
  CUDA_CHECK(cudaMalloc(&d_C, c_bytes));

  CUDA_CHECK(cudaMemcpy(d_A, A, a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, b_bytes, cudaMemcpyHostToDevice));

  CUBLAS_CHECK(cublasCreate(&g_handle));
}

// Timed region: launch + synchronize, nothing else. cublasSgemm on the
// default stream is synchronous with respect to the host from cuBLAS's
// perspective, but cudaDeviceSynchronize() makes the timed boundary explicit
// and matches the contract used by the other GPU variants.
void run(const float* /*A*/, const float* /*B*/, float* /*C*/, int M, int N,
         int K) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // C^T (N x M, col-major) = B^T (N x K, col-major) * A^T (K x M, col-major)
  // i.e. cublasSgemm(op_B=N, op_A=N, m=N, n=M, k=K, B, ldb=N, A, lda=K, C, ldc=N)
  CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           d_B, N, d_A, K, &beta, d_C, N));
  CUDA_CHECK(cudaDeviceSynchronize());
}

void fetch(float* C, int M, int N) {
  size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);
  CUDA_CHECK(cudaMemcpy(C, d_C, c_bytes, cudaMemcpyDeviceToHost));
}

void teardown() {
  CUBLAS_CHECK(cublasDestroy(g_handle));
  g_handle = nullptr;
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  d_A = d_B = d_C = nullptr;
}

}  // namespace

REGISTER_MATMUL(Variant{
    "8 cublas", Kind::Gpu,
    /*prepare=*/prepare,
    /*run=*/run,
    /*fetch=*/fetch,
    /*teardown=*/teardown});

}  // namespace mm
