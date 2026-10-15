# llama.cpp multi-GPU (tensor split) — decode improvement plan

Base: exp repo `master` @ `415b19ec7` (T1 committed). Each item → its own branch off `master`.
HW: 2× RX 9070 (gfx1201, RDNA4, 28 CU, wave32, ~570 GB/s achievable DRAM BW), no P2P,
GPU0 = PCIe5.0 x16, GPU1 = PCIe4.0 x4. Internal host-staged AllReduce (`GGML_HIP_RCCL=OFF`).

## Roofline (per decode token, per GPU, 1.7B Q6_K, tensor split)

| kernel   | us/tok | regime     | notes |
|----------|-------:|------------|-------|
| AllReduce| ~2200 (41%) | handshake  | 48 AR/tok (2/layer×24). HW-limited: no P2P, x4 slot, 2× threadfence_system + PCIe RT. Sleep is a wash, T1 (grid) done. |
| GEMV     | ~2100 (39%) | memory, 48% BW | 168 half-size GEMVs/tok. 63% BW in layer split → tensor-split fragmentation + launch overhead is the gap. |
| FlashAttn| ~210 (4%)  | memory     | KV-cache read, seq-bound. |
| quant_q8_1| ~205 (4%) | launch     | on-the-fly activation quant, 1 per GEMV. |
| small    | ~490 (9%) | launch     | RMSNorm + rope + residual-add. |

Baseline measured: **1.7B Q6_K tensor, 185 tok/s (5.4 ms/token)**. Gains below are vs this.

---

## Items (ranked by gain/effort)

### 1. `imp/01-qkv-gateup-concat` — Concat Q·K·V and gate·up weights  [CODE]
- At load, fuse (Q,K,V)→one weight and (gate,up)→one weight (per-GPU, along n_out).
- Graph: 3 attn GEMVs→1, 2 FFN GEMVs→1. Cuts 72 GEMV launches/token. No new kernel (reuse GEMV on fused weight; split output).
- Target: GEMV launch overhead. Est **~4–5%** (−220 µs/tok). Effort: low–med.
- Validate: A/B llama-bench 1.7B tensor (same-commit, n=10) + trace GEMV launch count.

### 2. `imp/02-quant-fuse-gemv` — Fuse quant_q8_1 into GEMV  [CODE]
- Make the Q6_K GEMV quantize its f32 input inline (or have the AR emit q8_1) so the separate `quantize_q8_1` kernel disappears.
- Target: quant (205 µs). Est **~3%** (−140 µs). Effort: med (kernel / cross-kernel).
- Validate: A/B + trace (quant kernel count → ~0).

### 3. `imp/03-layer-split-config` — Layer split for small models  [CONFIG, no code]
- `-sm layer` avoids AR entirely; wins only when AR/X > 0.5 (≈ <3B).
- Est: ~0 for 1.7B, large for <1B. Effort: trivial (bench protocol). Document threshold.

### 4. `imp/04-small-kernel-batch` — Batch/fuse RMSNorm+rope+residual  [CODE]
- Fewer launches for the ~490 µs of tiny launch-bound kernels (fuse rope into norm, or residual-add into neighbor).
- Target: small (9%). Est **~2%** (−100 µs). Effort: med.
- Validate: A/B + trace small-kernel time.

### 5. `imp/05-gemv-tiling-m1` — GEMV tiling for m=1  [CODE, hard]
- More in-flight memory requests / larger vectors in the m=1 Q6_K GEMV to push 48%→~58% BW.
- Target: GEMV efficiency. Est **~4–5%** (−250 µs). Effort: high (kernel rewrite).
- Validate: GEMV microbench (BW) + A/B.

### 6. `imp/06-ar-d2h-overlap` — Overlap AR D2H with producing-GEMV tail  [CODE]
- Dedicated AR stream so the pinned D2H overlaps the tail of the GEMV that produced the partial sum.
- Target: AR launch gap. Est **~1–2%**. Effort: low–med.
- Validate: A/B + AR gap in trace.

### 7. `imp/07-x16-slot-hw` — x16 slot for GPU1  [HARDWARE, no code]
- Halves the x4 cross-GPU AR latency. Est **~2–4%**. Track only; no code.

### 8. `imp/08-p2p-board-hw` — P2P-capable board  [HARDWARE, no code]
- Device-side AR (no pinned handshake) → AR ~10× faster. Est **~15%+**. Track only; no code.

### 9. `imp/09-rccl-2rank` — RCCL 2-rank after AMD fix  [BLOCKED, no code]
- Butterfly AR via RCCL. Est **~10%+**. Blocked on AMD 2-rank SHM-transport crash (filed). Track only.

### 10. `imp/10-ar-graph-capture-check` — Does host-staged AR break HIP-graph capture?  [DIAGNOSTIC]
- If the host-sync AR can't be captured, the 48 AR/tok force eager launches (hidden host overhead). Quantify; may reveal free win or motivate a device-side AR.
- Est **~1–2%** (if confirmed). Effort: low.
- Validate: compare graph-capture on/off around AR, or count eager launches.

---

## Execution order (autonomous, one-by-one)
#10 (quick diag) → #3 (config) → #1 → #2 → #4 → #6 → #5.
Then #7/#8 are hardware (document, no build). #9 dropped (RCCL deprecated).
Front-load the quick, low-risk items (#10 diag, #3 config) to validate the A/B harness, then the value code items.

## Method (every code item)
- Branch off `master` (has T1). Build `ggml-hip`. A/B vs `master` same-commit, `llama-bench` 1.7B Q6_K tensor, `-p 512 -n 128`, n≥10, report tg128 mean±sd.
- Keep VRAM < 16 GB/GPU (1.7B fits ~0.65 GB/GPU).
- Commit on the item branch only when the A/B is a clean, non-overlapping win.

---

## STATUS (autonomous pass)

### DONE (committed)

| # | Item | Result |
|---|------|--------|
| **3** | layer-split config | **`e0dc2d564`**. 1.7B: layer tg128 201.08 vs tensor 185.00 (**+8.7%**), pp512 **2.06×**. Rule: use `-sm layer` for small models (<~3B, AR/X > 0.5). See imp03_findings.md |
| **10** | AR graph-capture diag | **`546588420`**. Graph capture IS active (4410 hipGraphLaunch/196 graphs); AR runs EAGERLY between replays (4504 hipEventSync + 8860 kernel launches). No standalone quick win; real fix = device-side async AR. See imp10_findings.md |

### DEFERRED (code items) — with code-analysis root causes

The roofline's "~14% software headroom" is **real but NOT low-hanging**. Every path below
requires substantial, risky changes (verified against the source):

- **#1 QKV concat — DONE: `fuse-qkv` GGUF tool (bit-exact; prefill +3.7%, decode neutral).**
  Built `tools/fuse-qkv` (on `imp/01-qkv-gateup-concat`): rewrites each layer's separate
  `attn_q/k/v` into one fused `attn_qkv.weight`, so the **existing** fused-QKV graph path fires
  (1 GEMV instead of 3). The earlier "contiguous-split mis-assigns heads" fear did NOT materialize:
  the split is on the *output* dim (heads), and a fused `[Q|K|V]` is split contiguously into
  `[Q,K½]→GPU0 | V,K½→GPU1` — valid (a few K-heads straddle, which is fine for correctness). The
  tool is **bit-exact** (identical deterministic prime output). Perf (1.7B Q6_K, n=10): **pp512
  +3.69%** (prefill), **tg128 +0.37%** (decode, flat) — decode-neutral because the memory-bound
  GEMV reads the same total bytes and graph capture already amortizes launches. FFN gate/up not
  fused (dense `build_ffn` has no fused path). So a legitimate **prefill** win + reusable tool,
  **not a decode win**. See `imp01_results.md`.
- **#2 quant-in-GEMV — NOT VIABLE as framed.** `ggml-cuda.cu:1807`: `use_mul_mat_vec_q` requires
  `src1->type == F32`; pre-quantizing the input to q8_1 routes to the **slower cuBLAS path**
  (`:1872`). The `quantize_q8_1` is backend-inserted per matmul (not a dedupe-able graph node).
  Dedup needs a backend input-quant **fusion** (complex).
- **#5 GEMV tiling — LOW realistic payoff.** The tensor-split 48% (vs layer 63%) is mostly the
  **half-size structural penalty**, not tile efficiency; the layer-split 63% is already near the
  Q6_K-dequant ceiling. Grids are well-occupied (524288 blocks ≫ 224 concurrent). Kernel rewrite,
  uncertain.
- **#4 small-kernel batch — architecturally hard.** RMSNorm/rope/add have different structures; clean
  fusion requires new kernels + graph rewrites for ~2%.
- **#6 AR D2H overlap — BLOCKED by data dependency.** The AR's partial-sum input isn't ready until the
  producing sub-graph finishes; the next sub-graph needs the AR output. No overlap window.
- **#7 / #8 — hardware** (x16 slot, P2P board). See plan.
- **#9 RCCL — dropped** (deprecated).

### Bottom line
**Delivered (measured, same-commit A/B, bit-exact where applicable):**
- **T1** — AR grid scaling. **done, tg128 +5.27%** (decode), committed `415b19ec7`.
- **#3** — layer-split config. **done, +8.7% tg128** for small models (AR/X>0.5), committed `e0dc2d564`.
- **#1** — `fuse-qkv` GGUF tool. **done, bit-exact, pp512 +3.69%** (prefill), decode neutral,
  committed `3d1a58cb3` (on `imp/01`).
- **#10** — AR graph-capture diagnostic. done (no quick win); #2/#4/#5/#6 analyzed & deferred/blocked
  (root causes above).

**The decode story for this workload:** the two real decode levers are **T1** (done) and, for small
models, **layer-split** (#3). Beyond that it is **hardware** (x16 slot / P2P board, ~2× for AR). The
roofline's "~14%" included launch-overhead items that, once graph capture is accounted for, are not
decode wins (fusion is prefill-only). QKV+gateup prefill fusion (with a dense `build_ffn` fused path)
would add more prefill, but is a separate, higher-risk change.
