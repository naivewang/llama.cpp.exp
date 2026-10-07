# splitK microbench — root-cause findings

## TL;DR
The splitK tensor-parallel GEMM microbench (`--tp-gemm ... -p splitK`) reports
**3.84 µs**, which is **invalid**. The split-state and GEMM dispatch are *correct*,
but the measured perf does **not** include the actual GEMM + AllReduce execution —
the meta-backend graph replay is not draining the work for the splitK/AllReduce path.

Mirror (38 µs) and splitN (30 µs) are valid; splitK is not.

## What is CORRECT (verified)
1. **Split-state**:
   - `a`, `b` → axis-0 split: 2560×5120 per device (13.9 MB Q8_0). ✓
   - `out` → PARTIAL (full shape 5120×1, each device holds a partial sum). ✓
   - `out_ar` → MIRRORED (the AllReduce result). ✓
2. **GEMM dispatch** (CUDA backend diagnostic): the GEMM is launched with the
   correct dims — `src0=2560×5120` Q8_0 (13.9 MB), `src1=2560×1` F32, `dst=5120×1`.
   So each device computes its k-slice correctly.

## The anomaly (measured)
| policy  | graph_compute (perf) | GEMM kernel (hip events) | data/device |
|---------|----------------------|--------------------------|-------------|
| mirror  | **38.5 µs**          | ~31 µs (DRAM, 27.8 MB)   | 27.8 MB     |
| splitN  | **30.1 µs**          | (n/a)                    | 27.8 MB     |
| splitK  | **3.84 µs**          | ~6.8 µs (L2-warm, 13.9 MB)| 13.9 MB    |

- `ggml_backend_synchronize` takes **0–1 µs for ALL policies** — so the perf number
  *is* the `graph_compute` call time (the sync adds nothing).
- mirror: graph_compute (38 µs) ≈ GEMM kernel (31 µs) → the replay **includes** the GEMM.
- splitK: graph_compute (3.84 µs) < GEMM kernel (6.8 µs) → the replay **excludes** the GEMM.

So the *same* meta-backend `graph_compute` is synchronous for mirror/splitN but
**not** for splitK. The splitK's GEMM + AllReduce land on stream(s) that the
replay/sync path does not drain, so only the dispatch overhead (~3.84 µs) is timed.

Note: splitK does *more* work than mirror (GEMM k-slice + AllReduce + out_ar add_id),
so a correct number should be **≥ ~25–30 µs** (GEMM ~7–20 µs + AR ~20 µs), not 3.84 µs.

## Why splitK differs from mirror
- mirror: 1 subgraph, no AllReduce.
- splitK: 2 subgraphs — subgraph 0 = GEMM (`out`, PARTIAL), then AllReduce, then
  subgraph 1 = `out_ar` (add_id, MIRRORED). The `new_subgraph` boundary fires on
  `split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL` (ggml-backend-meta.cpp:2203).

The AllReduce path (`ggml_backend_meta_graph_compute`, ~line 2440+) dispatches the
subgraph compute and the AllReduce as async ops on the device/comm streams. For the
splitK case the replay returns without those having completed, and
`ggml_backend_synchronize` (0 µs) does not catch up — i.e. the GEMM/AR are enqueued
on a stream the meta device's sync does not join.

## Separate issue: TEST-mode crash
`ggml_backend_meta_buffer_get_tensor` has a `switch` on `split_state.axis` with cases
for 0/1/2 and MIRRORED, but **PARTIAL hits `default` → `GGML_ABORT`**. This fires
because the microbench's *whole-ctx* allocation puts `out` (a compute result) in a
**static** buffer, so the harness comparison reads it as an init tensor. Real models
use a two-buffer allocation (static weights + compute activations), which avoids this.
Fix = add a PARTIAL case to `get_tensor` (and/or a two-buffer alloc in the harness).

## Status
- **C0** (committed `911c1ead3`): `initialize_tensors` only inits leaf tensors
  (`op == GGML_OP_NONE`) — avoids the init-tensor PARTIAL abort path.
- **C1** (this doc): production per-device dims + GEMM dispatch confirmed correct;
  the 3.84 µs is a replay-sync gap, not a split-state bug.
- **Remaining**: (a) fix the splitK replay to drain the GEMM/AR streams (meta-backend
  change), and (b) add a PARTIAL readback case to `get_tensor` for TEST-mode.

## Reproduce
```
cd /home/elvinw/llama/llama.cpp.exp
./build-hip/bin/test-backend-ops perf --meta-devices ROCm0,ROCm1 \
  --tp-gemm Q8_0:5120:5120:1 -o MUL_MAT -p splitK   # 3.84 us (INVALID)
./build-hip/bin/test-backend-ops perf --meta-devices ROCm0,ROCm1 \
  --tp-gemm Q8_0:5120:5120:1 -o MUL_MAT -p mirror   # 38 us (valid)
```
