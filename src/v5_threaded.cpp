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

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#endif

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

// Logical-processor placement.
//
// Verified on the target machine (AMD Ryzen 7 7840HS, Zen 4, 8 physical
// cores / 16 logical) via GetLogicalProcessorInformationEx(RelationProcessorCore,
// ...): each RelationProcessorCore entry's GROUP_AFFINITY mask covered exactly
// two adjacent bits, core c -> {2c, 2c+1} for c in [0,8) -- e.g. core 3's mask
// was 0xc0 (bits 6,7). So the "adjacent SMT siblings" assumption holds here;
// this code assumes it generally (logical 2c and 2c+1 are siblings of
// physical core c) since that is what the OS actually reported, but if that
// ever isn't true elsewhere, pinning still succeeds -- it just no longer
// guarantees distinct physical cores.
//
// For nthreads <= physical_cores, thread t gets the FIRST logical of
// physical core t (2t), so no two threads share a core. Once threads exceed
// the physical core count, threads t >= physical_cores wrap back and take
// the second (sibling) logical of core (t % physical_cores), filling siblings
// only after every core has one thread.
int logical_for_thread(unsigned t, unsigned physical_cores, unsigned logicals_per_core,
                        unsigned hc) {
  if (physical_cores == 0) return -1;
  unsigned sibling_slot = t / physical_cores;
  if (sibling_slot >= logicals_per_core) return -1;  // more threads than logical procs to give
  unsigned core = t % physical_cores;
  unsigned logical = core * logicals_per_core + sibling_slot;
  if (logical >= hc) return -1;
  return static_cast<int>(logical);
}

// Pin one worker thread to a specific logical processor. Failure is not
// fatal -- an unpinned thread still produces correct results, just with
// nondeterministic placement.
void pin_thread(std::thread& th, int logical) {
  if (logical < 0) return;
#ifdef _WIN32
  DWORD_PTR mask = static_cast<DWORD_PTR>(1) << static_cast<unsigned>(logical);
  SetThreadAffinityMask(reinterpret_cast<HANDLE>(th.native_handle()), mask);
#elif defined(__linux__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(logical, &cpuset);
  pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
#else
  (void)th;
#endif
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

  unsigned hc = std::thread::hardware_concurrency();
  if (!hc) hc = nthreads;
  // Assume 2-way SMT (verified on the target 7840HS, see logical_for_thread's
  // comment); halve hc for the physical core count used to pick placements.
  unsigned physical_cores = hc >= 2 ? hc / 2 : hc;
  const unsigned logicals_per_core = 2;

  int rows_per = M / static_cast<int>(nthreads);
  int remainder = M % static_cast<int>(nthreads);
  int row = 0;
  for (unsigned t = 0; t < nthreads; ++t) {
    int band = rows_per + (static_cast<int>(t) < remainder ? 1 : 0);
    int row_begin = row;
    int row_end = row + band;
    row = row_end;
    pool.emplace_back(simd_band, A, B, C, N, K, row_begin, row_end);
    int logical = logical_for_thread(t, physical_cores, logicals_per_core, hc);
    pin_thread(pool.back(), logical);
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
