# matmul

One operation — `C = A × B`, row-major float32 — implemented eight times and measured
by the same harness. Every row below is printed by that harness, and every variant is
checked against the naive reference on every run, so a fast wrong answer fails the
build rather than topping the table.

## The table

`./build/matmul --sizes 1024,2048`, all rows PASS, exit 0.

### N = 1024

| Variant | median ms | GFLOP/s | speedup vs naive | reps | spread | verify |
|---|---|---|---|---|---|---|
| 1 naive ijk | 5707.9191 | 0.38 | 1.00x | 3 | 2.0% | PASS |
| 2 reorder ikj | 43.8247 | 49.00 | 130.24x | 46 | 21.2% | PASS |
| 3 tiled | 46.8590 | 45.83 | 121.81x | 43 | 11.6% | PASS |
| 4 simd avx2 | 59.1334 | 36.32 | 96.53x | 34 | 3.4% | PASS |
| 5 threads | 9.6634 | 222.23 | 590.68x | 204 | 41.8% | PASS |
| 6 cuda naive | 2.6091 | 823.08 | 2187.71x | 742 | 177.7% | PASS |
| 7 cuda tiled shared | 2.0863 | 1029.35 | 2735.96x | 950 | 117.2% | PASS |
| 8 cublas | 0.3042 | 7058.33 | 18760.74x | 4096 | 1389.7% | PASS |

### N = 2048

| Variant | median ms | GFLOP/s | speedup vs naive | reps | spread | verify |
|---|---|---|---|---|---|---|
| 1 naive ijk | 51752.1399 | 0.33 | 1.00x | 3 | 0.8% | PASS |
| 2 reorder ikj | 901.4924 | 19.06 | 57.41x | 3 | 7.4% | PASS |
| 3 tiled | 495.6084 | 34.66 | 104.42x | 5 | 1.4% | PASS |
| 4 simd avx2 | 514.7822 | 33.37 | 100.53x | 4 | 2.1% | PASS |
| 5 threads | 84.6314 | 203.00 | 611.50x | 24 | 6.6% | PASS |
| 6 cuda naive | 20.2330 | 849.10 | 2557.81x | 92 | 554.4% | PASS |
| 7 cuda tiled shared | 16.1379 | 1064.57 | 3206.87x | 124 | 20.6% | PASS |
| 8 cublas | 2.1736 | 7903.78 | 23809.11x | 917 | 121.7% | PASS |

The ladder is not monotonic. Variant 4 loses to variant 2, and variant 3 loses to
variant 2 at 1024 and wins at 2048. Both are explained below and neither is hidden.

## Hardware, build, method

| | |
|---|---|
| CPU | AMD Ryzen 7 7840HS, 8 cores / 16 threads, Zen 4 |
| L1d / L2 / L3 | 32 KiB per core / 1 MiB per core / 16 MiB shared |
| Memory | 32 GiB DDR5-5600, 2 channels x 64-bit = 89.6 GB/s theoretical |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB, sm_89, driver 581.86 |
| OS | WSL2 Ubuntu 24.04 on Windows 11 |
| Compiler | g++-12 (12.4.0), `-O3 -march=native` |
| CUDA | nvcc 12.0.140, `-arch=sm_89` |
| Sizes | 1024 and 2048, square |
| Data | float32, seed 42, `uniform_real_distribution(-1, 1)`, 64-byte aligned |
| Statistic | **median**, not mean |
| Reps | as many as fit a 2.0 s budget per variant per size, min 3, cap 4096 |
| Warmup | one untimed run before timing begins |
| Correctness | every variant vs naive, every run, `max_abs_diff / max_abs_ref <= 1e-4` |

GPU rows time the kernel launch plus `cudaDeviceSynchronize` and nothing else.
**Device allocation and the host-to-device copy happen in `prepare`, the copy back in
`fetch`, both outside the timed region, so GPU numbers exclude PCIe transfer.** That
is why the variant interface has four function pointers instead of one.

The machine was in a fixed state for every number: AC power, Lenovo Vantage thermal
mode Performance, Windows power mode Best performance, discrete-GPU-direct (hybrid
off), factory GPU overclock on, no other load. Measurements are taken serially, never
two at once.

### The `spread` column

`(max - min) / median` across the reps behind that row. It is reported because it is
large for some rows and that is a fact about the machine, not something to average
away. Variant 5 shows 41.8% because `run()` spawns and joins its threads on every
call; the GPU rows show more because a 0.3 ms kernel sits close to launch overhead.
The medians are stable even where the spread is not — see "what went wrong" below.

## Roofline

**CPU peak, from a measured clock.** Sustained all-core clock under a 16-thread run
was sampled at 4753–4799 MHz (median 4780). Single-threaded it was 4862–4904 MHz
(median 4887) — only 2% higher.

```
CPU peak (all cores)   = 8 cores x 32 flop/cycle x 4.780 GHz = 1224 GFLOP/s
CPU peak (one core)    = 1 core  x 32 flop/cycle x 4.887 GHz =  156 GFLOP/s
```

32 flop/cycle/core is two FMA units x 8 floats x 2 flop. Zen 4 runs 512-bit
operations on 256-bit units across two cycles, so AVX-512 does not raise this ceiling.

```
CPU machine balance    = 1224 / 89.6  = 13.7 flop/byte
GPU peak (upper bound) = 2560 cores x 2 flop/cycle x 3.105 GHz = 15898 GFLOP/s
GPU machine balance    = 15898 / 192  = 82.8 flop/byte
```

**The GPU peak is an upper bound, not a measured sustained figure.** NVIDIA clock
telemetry is not live under WSL2's virtualised GPU: `nvidia-smi` reports 210 MHz SM,
405 MHz memory and 1 W identically at idle and during a cuBLAS run measured at 7904
GFLOP/s, from inside WSL and from Windows, via both `--query-gpu` and `dmon`. A
reading that does not move between idle and full load is not a reading, so the
number above uses `clocks.max.sm` and is labelled accordingly. As a sanity check,
cuBLAS reaching 50% of it is an ordinary FP32 non-tensor GEMM efficiency.

### Fraction of peak, and which limit each variant hits

| Variant | 1024 | 2048 | % of peak (1024) | Bound |
|---|---|---|---|---|
| 1 naive ijk | 0.38 | 0.33 | 0.2% | memory **latency** |
| 2 reorder ikj | 49.00 | 19.06 | 31.3% | cache capacity at 2048 |
| 3 tiled | 45.83 | 34.66 | 29.3% | compute, once blocked |
| 4 simd avx2 | 36.32 | 33.37 | 23.2% | compute, code generation |
| 5 threads | 222.23 | 203.00 | 18.2% | compute, parallel efficiency |
| 6 cuda naive | 823.08 | 849.10 | 5.2% | memory bandwidth |
| 7 cuda tiled shared | 1029.35 | 1064.57 | 6.5% | memory bandwidth, reduced |
| 8 cublas | 7058.33 | 7903.78 | 44.4% | compute |

CPU rows 1–4 are against the one-core peak, row 5 against the all-core peak, GPU rows
against the GPU upper bound.

**Variant 1 is latency-bound, not bandwidth-bound.** Its inner loop walks a column of
B with stride N, so every iteration touches a new cache line. At 0.38 GFLOP/s it is
issuing 0.19 G-FMA/s, and at 64 bytes per line that is 12.2 GB/s against 89.6 GB/s
available. It is not saturating memory; it is waiting on it, one miss at a time.

**Variant 2 is a cache-capacity story.** It falls from 49.00 to 19.06 GFLOP/s between
1024 and 2048. At 1024 the three matrices total 12 MiB and fit inside 16 MiB of L3.
At 2048 they total 48 MiB and do not, so B is re-streamed from DRAM for every row of
A — about 32 GiB of traffic in 0.90 s, or 38 GB/s, 43% of theoretical bandwidth. That
drop is the L3 capacity wall, measured.

**The GPU rows are bandwidth-bound and that is why variant 7 wins.** Variant 6 reads
2N floats from global memory per output element, an arithmetic intensity of about
0.25 flop/byte against a machine balance of 82.8. Variant 7's 32x32 shared-memory
tile raises reuse by roughly the tile width, to about 8 flop/byte — still far below
82.8, so still bandwidth-bound, which is why it gains 1.25x rather than 32x.

## The gap to cuBLAS

**Variant 7 reaches 14.6% of cuBLAS at 1024** (1029.35 vs 7058.33 GFLOP/s) and
**13.5% at 2048** (1064.57 vs 7903.78). cuBLAS is 6.9x faster. Beating it was never
the goal; the gap is the honest measure of what a straightforward shared-memory tile
buys against a library that uses register-level blocking, vectorised loads, deeper
tiling and hand-tuned scheduling per architecture.

## What went wrong, and what it changed

Four things were measured, found wrong, and corrected. They are listed because the
corrections are the content.

**The harness was under-sampling.** `--reps` was a hard cap of 7 and it, not the 2 s
budget, ended the timing loop, so a variant running at 8 ms/rep was sampled across
56 ms — one thermal instant. Five identical runs of variant 5 returned 253, 262, 262,
258 and 339 GFLOP/s. Reps now run until the budget is spent, which put ~250 reps
behind that row and narrowed run-to-run range from 34% to 20%.

**Threads were not pinned.** With 8 physical cores exposed as 16 logical, the OS was
free to place N threads on N distinct cores or to double them onto SMT siblings, and
those differ by about 2x. Three identical sweeps at four threads gave 89.79, 177.96
and 88.96 GFLOP/s. After pinning each thread to a distinct physical core: 89.53,
89.04, 89.24. The core-to-logical mapping was read from
`GetLogicalProcessorInformationEx` rather than assumed.

**A conclusion reversed.** From the unpinned numbers, 8 to 16 threads gained 8%, and
the write-up concluded the FP units were saturated. Pinned, the same step gains 1.50x,
which says the opposite — the kernel leaves issue slots idle and a second thread on
the same core finds the pipes free. The original claim was not imprecise, it was
backwards, and it was backwards because of a measurement bug.

**A tidy explanation was refuted.** Variant 4 is 26% slower than variant 2 at 1024,
and the obvious cause was vector width: gcc autovectorising the plain loop to AVX-512
(16 floats) against hand-written `_mm256_fmadd_ps` (8 floats). Disassembly supported
it — `objdump` shows variant 2 emitting `vfmadd` in both `zmm` and `ymm` forms without
being asked. Rebuilding with `-mprefer-vector-width=256`, which forces gcc down to the
same width, moved variant 2 from 50.58 to 49.16 GFLOP/s — a 3% change, nowhere near
36.32. **Width is not the cause.** At equal width the compiler's code generation beats
the hand-written loop, and the AVX-512 story, however neat, is measured to be false.

## Reproducing

The CPU variants build anywhere with a C++17 compiler. The CUDA variants need a host
compiler nvcc accepts, which is the one real constraint:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_HOST_COMPILER=g++-12
cmake --build build -j
./build/matmul --sizes 1024,2048
```

`USE_CUDA` defaults to whether CMake finds a CUDA compiler, so a machine without one
configures and builds variants 1–5 with no flags. `-DUSE_CUDA=ON` forces it and fails
loudly if the toolkit is absent.

**Under WSL2, export `LD_LIBRARY_PATH=/usr/lib/wsl/lib` before running.** Ubuntu's
`nvidia-cuda-toolkit` installs its own `libcuda.so.1` into `/lib/x86_64-linux-gnu`
alongside the working WSL one, and if the loader picks Ubuntu's the run aborts at the
first `cudaMalloc` with "no CUDA-capable device is detected" even though `nvidia-smi`
works. The variants abort there rather than reporting a fast empty result, which is
the error checking behaving correctly.

Useful flags: `--sizes`, `--reps`, `--budget`, `--only <substring>`, `--csv <path>`.
`MATMUL_THREADS` overrides the thread count for variant 5.

**On Windows this builds CPU-only.** nvcc accepts only MSVC as a host compiler there
(`crt/host_config.h` gates on `_MSC_VER`, which MinGW g++ never defines). The
alternative was moving the whole project to `cl.exe` and `/arch:AVX2`, which would
have replaced the vectoriser that the variant-4-versus-2 result is a claim about, so
the build moved to WSL2 instead and kept gcc. CUDA 12.0 refuses gcc 13+, so g++-12
compiles both halves rather than mixing two compilers into one binary.

## Limits of these numbers

- **`compute-sanitizer` could not run.** CUDA 12.0's sanitizer does not support the
  WSL2 driver shim; the target dies before its first instrumented API call regardless
  of filesystem or library path. In its place the kernels were verified at 17 sizes
  around every tile and warp boundary — 1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
  127, 129, 255, 257, 513 — all eight variants PASS at every one. That catches an
  out-of-bounds access that changes a result; it does not catch a benign one. It is
  weaker than memcheck and is not presented as equivalent.
- **The thread-scaling study below was measured on native Windows, not WSL.** Pinning
  does not work inside WSL2, whose cpu0–15 are virtual processors the hypervisor
  places as it chooses: the same binary that is repeatable to ~1% natively varies by
  22% under WSL, and shows only 1.07x from 4 to 8 threads. Guest topology is not the
  cause — `/sys/devices/system/cpu/*/topology` reports the same adjacent-pair layout
  as Windows.
- **The CPU rows are measured under WSL2**, which costs roughly 10% against the same
  binary run natively. They are reported this way so that the CPU and GPU halves of
  the table come from one environment rather than two.

### Thread scaling (native Windows, pinned, N=1024)

| threads | GFLOP/s | vs 1 thread | step |
|---|---|---|---|
| 1 | 41.87 | 1.00x | |
| 2 | 55.85 | 1.33x | 1.33x |
| 4 | 89.24 | 2.13x | 1.60x |
| 8 | 168.74 | 4.03x | 1.89x |
| 16 | 253.27 | 6.05x | 1.50x |

Median of three sweeps, each repeatable to within 1%. The 1.50x from 8 to 16 threads
across 8 physical cores is the SMT result quoted above. The weak first step, 1.33x,
is not explained by clock: single-threaded and all-core sustained clocks were measured
at 4887 and 4780 MHz, 2% apart.
