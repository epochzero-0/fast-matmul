// v7: shared-memory tiled CUDA matmul. Each thread block cooperatively loads
// a TILE x TILE tile of A and a TILE x TILE tile of B into __shared__ memory,
// syncs, then each thread accumulates its partial dot product from the tile
// before moving to the next tile along K. Classic tiled-GEMM pattern.
//
// Sizes not a multiple of TILE (e.g. 1023) are handled by bounds-checking
// every global load: out-of-range elements are loaded as 0 into shared
// memory instead of being skipped, which keeps the inner product correct
// without needing a separate ragged-edge code path.

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

constexpr int kTile = 32;  // tile edge length, in elements

float* d_A = nullptr;
float* d_B = nullptr;
float* d_C = nullptr;

__global__ void tiled_kernel(const float* A, const float* B, float* C, int M,
                              int N, int K) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];

  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * kTile + ty;
  int col = blockIdx.x * kTile + tx;

  float acc = 0.0f;
  int num_tiles = (K + kTile - 1) / kTile;

  for (int t = 0; t < num_tiles; ++t) {
    int a_col = t * kTile + tx;
    int b_row = t * kTile + ty;

    As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    for (int k = 0; k < kTile; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}

void prepare(const float* A, const float* B, int M, int N, int K) {
  size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
  size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
  size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
  CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
  CUDA_CHECK(cudaMalloc(&d_C, c_bytes));

  CUDA_CHECK(cudaMemcpy(d_A, A, a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, b_bytes, cudaMemcpyHostToDevice));
}

void run(const float* /*A*/, const float* /*B*/, float* /*C*/, int M, int N,
         int K) {
  dim3 block(kTile, kTile);
  dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    "7 cuda tiled shared", Kind::Gpu,
    /*prepare=*/prepare,
    /*run=*/run,
    /*fetch=*/fetch,
    /*teardown=*/teardown});

}  // namespace mm
