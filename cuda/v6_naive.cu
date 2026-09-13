// v6: one CUDA thread per output element. No shared memory, no tiling.
// This is the CUDA analogue of v1_naive: a direct triple-loop translation
// where the outer two loops (i, j) become the 2D thread grid and only the
// k-reduction stays a device-side loop.

#include <cstdio>
#include <cstdlib>

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

namespace mm {
namespace {

// Device-side state, set up in prepare(), consumed by run(), released by
// teardown(). File-local because the ABI has no per-variant context pointer.
float* d_A = nullptr;
float* d_B = nullptr;
float* d_C = nullptr;
int g_M = 0, g_N = 0, g_K = 0;

__global__ void naive_kernel(const float* A, const float* B, float* C, int M,
                              int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}

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
}

// Timed region: kernel launch + synchronize, nothing else.
void run(const float* /*A*/, const float* /*B*/, float* /*C*/, int M, int N,
         int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void fetch(float* C, int M, int N) {
  size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);
  CUDA_CHECK(cudaMemcpy(C, d_C, c_bytes, cudaMemcpyDeviceToHost));
}

void teardown() {
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  d_A = d_B = d_C = nullptr;
}

}  // namespace

REGISTER_MATMUL(Variant{
    "6 cuda naive", Kind::Gpu,
    /*prepare=*/prepare,
    /*run=*/run,
    /*fetch=*/fetch,
    /*teardown=*/teardown});

}  // namespace mm
