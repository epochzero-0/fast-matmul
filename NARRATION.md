# NARRATION

Five questions, two sentences each. Every number here is in the README table or was
printed by the harness.

### 1. Why does the loop reorder help?

The naive `i-j-k` loop walks a column of B with stride N, so each multiply pulls in a
fresh 64-byte cache line and uses 4 bytes of it before moving on, and the line is
evicted long before the loop comes back to it. Reordering to `i-k-j` makes the inner
loop walk B along a row so consecutive iterations land in the same cache line, which
is worth 0.38 to 49.00 GFLOP/s at N=1024 — 130x, with the same arithmetic in a
different order.

### 2. What is the tile size chosen for?

kTile = 256 targets L2, not L1: the three 256x256 float tiles live in 768 KiB, which
fits this Zen 4 core's 1 MiB private L2, and the L1-sized 32 tile is the slowest one
measured at 15.50 against 39.96 GFLOP/s. It was chosen by sweeping rather than
derived, and above 256 the curve is a plateau rather than a peak, so 256 is the most
repeatable point on it (4% range over three runs) rather than the fastest single
reading.

### 3. What does shared memory do in variant 7 that global memory doesn't?

Each block stages a 32x32 tile of A and B into shared memory once and all 1024
threads in the block read their operands from there, instead of every thread issuing
its own global loads for values its neighbours are also loading. That lifts
arithmetic intensity from roughly 0.25 to roughly 8 flop/byte and is worth 1.25x
(823.08 to 1029.35 GFLOP/s) — only 1.25x, because 8 is still far under the GPU's 82.8
flop/byte balance, so the kernel is still bandwidth-bound afterwards.

### 4. Is this memory-bound or compute-bound, and how do you know from the numbers?

Memory-bound nearly all the way up, and the numbers distinguish which kind: variant 1
is latency-bound rather than bandwidth-bound because it moves only 12.2 GB/s of the
89.6 GB/s available, so it is stalling on one miss at a time rather than saturating
the bus. Variant 2 is capacity-bound — it drops from 49.00 to 19.06 GFLOP/s between
N=1024 and N=2048, exactly where the working set grows from 12 MiB to 48 MiB and
stops fitting in 16 MiB of L3 — and the GPU rows sit at 0.25 to 8 flop/byte against a
machine balance of 82.8, which is why cuBLAS at 44% of peak is the only row that
looks compute-bound.

### 5. Why is it slower than cuBLAS, and roughly by how much?

Variant 7 reaches 14.6% of cuBLAS at N=1024 (1029.35 against 7058.33 GFLOP/s) and
13.5% at N=2048, so cuBLAS is about 6.9x faster. Variant 7 stops after one level of
blocking — global into shared — while cuBLAS blocks a second time from shared into
registers so each thread computes many outputs instead of one, issues wider vector
loads, and ships schedules tuned per architecture.
