#pragma once
// llama-gpu-worker: Phase 2 PoC of the "GPU child process" plan.
//
// The service (parent) process never initializes a GPU. All GPU state (model
// weights, context, KV, graphs) lives in a disposable child process reached
// over a UNIX domain socket. Sleep = kill the child; wake = respawn +
// restore_kv.
//
// Wire protocol: little-endian frames on a single connection (one in flight):
//   header: u32 total_len (incl. 8-byte header) | u32 opcode     (8 bytes)
//   payload: total_len - 8 bytes
//
// Opcodes (request/response):
//   OP_INIT       req: (none)
//                 rsp: i32 n_vocab, i32 n_ctx
//   OP_DECODE     req: i32 n_tokens, i32 seq, i32 token[n_tokens]
//                 rsp: f32 logits[n_vocab]  (logits of the last token; client
//                 samples greedily and feeds the next token back)
//   OP_DUMP_KV    req: i32 seq
//                 rsp: u32 size, u8 kv[size]      (llama_state_seq_get_data_ext)
//   OP_RESTORE_KV req: i32 seq, u32 size, i32 pad, i32 pad, u8 kv[size]
//                 rsp: u32 ok
//   OP_SHUTDOWN   req: (none); rsp: u32 ok        (worker exits cleanly)
//   OP_PING       req: (none); rsp: u32 ok

#include <cstdint>

namespace gwp {

constexpr uint32_t OP_INIT       = 1;
constexpr uint32_t OP_DECODE     = 2;
constexpr uint32_t OP_DUMP_KV    = 3;
constexpr uint32_t OP_RESTORE_KV = 4;
constexpr uint32_t OP_SHUTDOWN   = 5;
constexpr uint32_t OP_PING       = 6;

// frame header is total_len (u32) + opcode (u32)
constexpr uint32_t HDR_LEN = 8;

} // namespace gwp
