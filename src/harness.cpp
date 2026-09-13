// Benchmark harness. Times every registered variant against every requested
// size, checks correctness against mm::matmul_naive_reference, and prints a
// markdown table per size.
//
// See HANDOVER.md for the variant ABI and how to add a new variant.

#include "matmul.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

#ifndef MATMUL_BUILD_FLAGS
#define MATMUL_BUILD_FLAGS "(unknown)"
#endif

namespace {

// ---- aligned buffers -------------------------------------------------

float* aligned_alloc_floats(size_t count) {
  size_t bytes = count * sizeof(float);
  size_t rem = bytes % 64;
  if (rem != 0) bytes += (64 - rem);
#if defined(_WIN32)
  void* p = _aligned_malloc(bytes, 64);
#else
  void* p = std::aligned_alloc(64, bytes);
#endif
  return static_cast<float*>(p);
}

void aligned_free_floats(float* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

// ---- CLI -------------------------------------------------------------

struct Args {
  std::vector<int> sizes{256, 512, 1024, 2048};
  int reps = 7;
  double budget = 2.0;
  std::string only;
  std::string csv;
};

std::vector<int> parse_sizes(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
  }
  return out;
}

Args parse_args(int argc, char** argv) {
  Args a;
  bool sizes_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--sizes") {
      a.sizes = parse_sizes(next());
      sizes_set = true;
    } else if (arg == "--reps") {
      a.reps = std::atoi(next().c_str());
    } else if (arg == "--budget") {
      a.budget = std::atof(next().c_str());
    } else if (arg == "--only") {
      a.only = next();
    } else if (arg == "--csv") {
      a.csv = next();
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      std::exit(2);
    }
  }
  (void)sizes_set;
  return a;
}

// ---- misc --------------------------------------------------------------

std::string compiler_id_version() {
#if defined(__clang__)
  return "Clang " __clang_version__;
#elif defined(__GNUC__)
  return std::string("GNU G++ ") + __VERSION__;
#else
  return "unknown compiler";
#endif
}

std::string fmt_speedup(double naive_median, double this_median) {
  double s = naive_median / this_median;
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2fx", s);
  return buf;
}

struct TimedResult {
  std::vector<double> reps_seconds;  // every recorded rep, seconds
  double median_seconds = 0.0;
};

double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n % 2 == 1) return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

TimedResult time_variant(const mm::Variant& v, const float* A, const float* B,
                          float* C, int M, int N, int K, int max_reps,
                          double budget_seconds) {
  TimedResult result;

  // One untimed warmup run.
  v.run(A, B, C, M, N, K);

  auto budget_start = std::chrono::steady_clock::now();
  for (int rep = 0; rep < max_reps; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    v.run(A, B, C, M, N, K);
    auto t1 = std::chrono::steady_clock::now();
    result.reps_seconds.push_back(std::chrono::duration<double>(t1 - t0).count());

    double elapsed = std::chrono::duration<double>(t1 - budget_start).count();
    if (rep + 1 >= 3 && elapsed >= budget_seconds) break;
  }

  result.median_seconds = median_of(result.reps_seconds);
  return result;
}

struct RowResult {
  std::string name;
  double median_ms;
  double gflops;
  std::string speedup;
  bool pass;
};

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  std::printf("# matmul benchmark\n\n");
  std::printf("- compiler: %s\n", compiler_id_version().c_str());
  std::printf("- build flags: %s\n", MATMUL_BUILD_FLAGS);
  std::printf("- seed: 42\n");
  std::printf("- reps (max): %d\n", args.reps);
  std::printf("- budget: %.2fs\n", args.budget);
  std::printf("- only filter: %s\n", args.only.empty() ? "(none)" : args.only.c_str());
  std::printf("\n");

  std::ofstream csv;
  if (!args.csv.empty()) {
    csv.open(args.csv, std::ios::out | std::ios::trunc);
    csv << "size,variant,median_ms,gflops,speedup,verify\n";
  }

  bool all_pass = true;

  for (int S : args.sizes) {
    int M = S, N = S, K = S;
    size_t a_n = static_cast<size_t>(M) * K;
    size_t b_n = static_cast<size_t>(K) * N;
    size_t c_n = static_cast<size_t>(M) * N;

    float* A = aligned_alloc_floats(a_n);
    float* B = aligned_alloc_floats(b_n);
    float* C = aligned_alloc_floats(c_n);
    float* refC = aligned_alloc_floats(c_n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < a_n; ++i) A[i] = dist(rng);
    for (size_t i = 0; i < b_n; ++i) B[i] = dist(rng);

    mm::matmul_naive_reference(A, B, refC, M, N, K);

    double max_abs_ref = 0.0;
    for (size_t i = 0; i < c_n; ++i) {
      max_abs_ref = std::max(max_abs_ref, std::fabs(static_cast<double>(refC[i])));
    }

    double naive_median = -1.0;
    std::vector<RowResult> rows;

    int count = mm::variant_count();
    for (int vi = 0; vi < count; ++vi) {
      const mm::Variant& v = mm::variant_at(vi);
      std::string name = v.name;
      bool is_naive = (name == "1 naive ijk");
      bool matches_filter = args.only.empty() || name.find(args.only) != std::string::npos;
      if (!is_naive && !matches_filter) continue;

      std::memset(C, 0, c_n * sizeof(float));

      if (v.prepare) v.prepare(A, B, M, N, K);

      TimedResult timed = time_variant(v, A, B, C, M, N, K, args.reps, args.budget);

      if (v.fetch) v.fetch(C, M, N);

      double max_abs_diff = 0.0;
      int diff_count = 0;
      int diff_idx[3] = {-1, -1, -1};
      for (size_t i = 0; i < c_n; ++i) {
        double d = std::fabs(static_cast<double>(C[i]) - static_cast<double>(refC[i]));
        if (d > max_abs_diff) max_abs_diff = d;
        if (d / std::max(max_abs_ref, 1e-30) > 1e-4 && diff_count < 3) {
          diff_idx[diff_count] = static_cast<int>(i);
          ++diff_count;
        }
      }

      bool pass = (max_abs_diff / std::max(max_abs_ref, 1e-30)) <= 1e-4;
      if (!pass) all_pass = false;

      if (v.teardown) v.teardown();

      double median_s = timed.median_seconds;
      double median_ms = median_s * 1000.0;
      double gflops = (2.0 * M * N * K) / (median_s * 1e9);

      if (is_naive) naive_median = median_s;

      RowResult row;
      row.name = name;
      row.median_ms = median_ms;
      row.gflops = gflops;
      row.pass = pass;
      rows.push_back(row);

      if (!pass) {
        std::printf("FAIL %s: max_abs_diff=%.6g, first differing indices: ", name.c_str(),
                    max_abs_diff);
        for (int k = 0; k < diff_count; ++k) std::printf("%d ", diff_idx[k]);
        std::printf("\n");
      }
    }

    // Static-init order across translation units does not follow the CMake source
    // list, so registration order is not ladder order. Variant names are digit
    // prefixed ("1 naive ijk", "2 reorder ikj", ...), so sorting by name restores it.
    std::sort(rows.begin(), rows.end(),
              [](const RowResult& a, const RowResult& b) { return a.name < b.name; });

    std::printf("## N=%d\n\n", S);
    std::printf("| Variant | median ms | GFLOP/s | speedup vs naive | verify |\n");
    std::printf("|---|---|---|---|---|\n");
    for (auto& row : rows) {
      std::string speedup = (naive_median > 0.0)
                                 ? fmt_speedup(naive_median, row.median_ms / 1000.0)
                                 : "n/a";
      std::printf("| %s | %.4f | %.2f | %s | %s |\n", row.name.c_str(), row.median_ms,
                  row.gflops, speedup.c_str(), row.pass ? "PASS" : "FAIL");
      if (csv.is_open()) {
        csv << S << "," << row.name << "," << row.median_ms << "," << row.gflops << ","
            << speedup << "," << (row.pass ? "PASS" : "FAIL") << "\n";
      }
    }
    std::printf("\n");

    aligned_free_floats(A);
    aligned_free_floats(B);
    aligned_free_floats(C);
    aligned_free_floats(refC);
  }

  return all_pass ? 0 : 1;
}
