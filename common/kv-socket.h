#pragma once
// kv-socket: framing protocol for the KV state handoff channel between a
// GPU-free parent (the gpu-worker carrier thread, a.k.a. "watchdog") and a
// llama-server child, over an inherited connected socketpair fd (--kv-fd N).
//
// NO FILES: the KV blob never touches disk. The carrier holds it in its own
// RAM: dump -> carrier RAM -> (server dies) -> new server restores it.
// The carrier treats the blob as opaque bytes - it only needs this header
// (framing + opcodes); the server additionally (de)serializes the states in
// tools/server/server-kv-socket.h.
//
// Setup (carrier side):
//   int sv[2]; socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
//   pid = fork();
//   if (pid == 0) { close(sv[0]); dup2(sv[1], N); close(sv[1]); execv("llama-server", ... "--kv-fd", "N" ...); }
//   // carrier keeps sv[0]
//
// Framing (little-endian): u32 total_len (incl. 8-byte header) | u32 opcode

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/socket.h>
#include <sys/time.h>
#include <vector>

namespace kvsock {

constexpr uint32_t KV_OP_READY     = 1; // server -> carrier: boot, ctx + prompt cache ready;
                                        // awaiting KV_OP_RESTORE or KV_OP_SKIP
constexpr uint32_t KV_OP_RESTORE   = 2; // carrier -> server: boot; payload = serialized prompt-cache
                                        // states (server-side layout), inserted before serving
constexpr uint32_t KV_OP_SKIP      = 3; // carrier -> server: boot; nothing to restore
constexpr uint32_t KV_OP_DUMP      = 4; // carrier -> server: runtime; request all prompt-cache states
constexpr uint32_t KV_OP_DUMP_RSP  = 5; // server -> carrier: u32 ok (1=ok, 0=failed), then serialized
                                        // states (empty on failure)
constexpr uint32_t KV_OP_PING      = 6; // either direction
constexpr uint32_t KV_OP_PONG      = 7; // reply: u32 ok
constexpr uint32_t KV_OP_SHUTDOWN    = 8; // carrier -> server: exit gracefully now (the KV was
                                          // already dumped to the carrier); the server replies
                                          // first, then runs its normal SIGINT teardown (releases
                                          // the GPU itself) and exits 0
constexpr uint32_t KV_OP_SHUTDOWN_RSP = 9; // server -> carrier: u32 ok

constexpr uint32_t HDR_LEN = 8;
constexpr size_t   MAX_FRAME = 64ull << 30; // 64 GiB sanity cap

// ---------------- socket framing --------------------------------------------

inline bool send_all(int fd, const uint8_t * data, size_t len) {
    while (len > 0) {
        const ssize_t n = ::send(fd, data, len, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        data += n;
        len -= (size_t) n;
    }
    return true;
}

inline bool recv_all(int fd, uint8_t * data, size_t len) {
    while (len > 0) {
        const ssize_t n = ::recv(fd, data, len, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        data += n;
        len -= (size_t) n;
    }
    return true;
}

inline bool send_frame(int fd, uint32_t op, const uint8_t * payload, size_t len) {
    uint8_t hdr[HDR_LEN];
    uint32_t total = (uint32_t) (HDR_LEN + len);
    std::memcpy(hdr,     &total, 4);
    std::memcpy(hdr + 4, &op,    4);
    if (!send_all(fd, hdr, HDR_LEN)) return false;
    return len == 0 || send_all(fd, payload, len);
}

// timeout_ms = 0: block indefinitely (until EOF/error)
inline bool recv_frame(int fd, uint32_t & op, std::vector<uint8_t> & payload, int timeout_ms) {
    if (timeout_ms > 0) {
        struct timeval tv { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }
    uint8_t hdr[HDR_LEN];
    if (!recv_all(fd, hdr, HDR_LEN)) return false;
    uint32_t total = 0, op_in = 0;
    std::memcpy(&total, hdr, 4);
    std::memcpy(&op_in, hdr + 4, 4);
    if (total < HDR_LEN || total > MAX_FRAME) return false;
    payload.resize(total - HDR_LEN);
    if (payload.size() > 0 && !recv_all(fd, payload.data(), payload.size())) return false;
    op = op_in;
    return true;
}

} // namespace kvsock
