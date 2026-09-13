// Contract for every matmul variant. Do not change without updating HANDOVER.md.
//
// C = A * B, all row-major, all float32.
//   A is M x K   (lda = K)
//   B is K x N   (ldb = N)
//   C is M x N   (ldc = N)
//
// A variant registers itself at static-init time with REGISTER_MATMUL. The harness
// never calls a variant directly and never includes a variant header.
#pragma once

#include <cstddef>

namespace mm {

// Optional. Called once per (M,N,K) before any timed run. GPU variants allocate
// device buffers and upload A and B here. May be null.
using PrepareFn = void (*)(const float* A, const float* B, int M, int N, int K);

// The timed region. Exactly this call is what the table measures.
// CPU variants write into C. GPU variants launch the kernel and must synchronize
// before returning, and may ignore C.
using RunFn = void (*)(const float* A, const float* B, float* C, int M, int N, int K);

// Optional. Copies the result into host C for the correctness check, outside the
// timed region. GPU variants implement this; CPU variants leave it null.
using FetchFn = void (*)(float* C, int M, int N);

// Optional. Frees whatever prepare allocated. May be null.
using TeardownFn = void (*)();

enum class Kind { Cpu, Gpu };

struct Variant {
  const char* name;   // shown in the table, keep it short
  Kind kind;
  PrepareFn prepare;  // nullable
  RunFn run;          // never null
  FetchFn fetch;      // nullable; null means run() already wrote host C
  TeardownFn teardown; // nullable
};

// Registration order in the table is registration order; keep ladder order by
// naming files v1_..., v2_..., and listing them in that order in CMakeLists.txt.
void register_variant(const Variant& v);
int variant_count();
const Variant& variant_at(int i);

// The naive triple loop, also used as the correctness reference. Declared here so
// the harness can call it without going through the registry.
void matmul_naive_reference(const float* A, const float* B, float* C,
                            int M, int N, int K);

struct Registrar {
  explicit Registrar(const Variant& v) { register_variant(v); }
};

}  // namespace mm

// One registration per translation unit.
#define REGISTER_MATMUL(...)                              \
  namespace {                                             \
  const ::mm::Registrar mm_registrar_{ __VA_ARGS__ };     \
  }
