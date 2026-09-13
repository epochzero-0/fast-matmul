#include "matmul.h"

#include <algorithm>

namespace mm {
namespace {

#ifndef MATMUL_TILE_SIZE
#define MATMUL_TILE_SIZE 256
#endif

// Sized for L2, not L1, and that is a measured result rather than a derivation.
// Three tiles of kTile*kTile floats (the A tile, the B tile and the C accumulator
// tile touched per block) make up the working set. Swept at N=1024, g++-12:
//
//   kTile=32  ->  12 KiB   fits L1d (32 KiB)      15.50 GFLOP/s
//   kTile=64  ->  48 KiB                          29.93
//   kTile=128 ->  192 KiB  fits L2  (1 MiB/core)  35.97
//   kTile=256 ->  768 KiB  fits L2                39.96  <- chosen
//   kTile=384 ->  1.69 MiB exceeds L2             38.73
//   kTile=512 ->  3 MiB    exceeds L2             39.91
//
// The L1-sized tile is the slowest one measured, by a factor of 2.6: a 32-wide
// inner j loop is too short to amortize loop overhead and gives up the
// vectorization the full-row ikj loop in v2 gets for free.
//
// Above 256 this is a plateau, not a peak, so 256 is chosen for being the most
// repeatable rather than the fastest. Medians of three runs: 256 gave 39.96 /
// 40.77 / 39.09, a 4% range, while 384 gave 41.22 / 38.73 / 35.30, a 17% range
// that beat 256 once and lost to it twice. A single sweep had 384 on top at
// 41.86 and that reading did not survive repetition. Full data in TRACKER.md.
constexpr int kTile = MATMUL_TILE_SIZE;

void run(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * N + j] = 0.0f;
    }
  }

  for (int ii = 0; ii < M; ii += kTile) {
    int i_max = std::min(ii + kTile, M);
    for (int jj = 0; jj < N; jj += kTile) {
      int j_max = std::min(jj + kTile, N);
      for (int kk = 0; kk < K; kk += kTile) {
        int k_max = std::min(kk + kTile, K);
        // Same ikj order as v2, applied within each (ii,kk,jj) block.
        for (int i = ii; i < i_max; ++i) {
          for (int k = kk; k < k_max; ++k) {
            float a = A[i * K + k];
            for (int j = jj; j < j_max; ++j) {
              C[i * N + j] += a * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

}  // namespace

REGISTER_MATMUL(Variant{
    "3 tiled", Kind::Cpu,
    /*prepare=*/nullptr,
    /*run=*/run,
    /*fetch=*/nullptr,
    /*teardown=*/nullptr});

}  // namespace mm
