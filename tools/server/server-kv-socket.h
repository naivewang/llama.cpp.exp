#pragma once
// server-kv-socket: server-side extension of the KV handoff channel
// (framing protocol: common/kv-socket.h).
//
// This header adds the (de)serialization of the llama-server prompt-cache
// states into the opaque blob that travels over the channel. The carrier
// (gpu-worker) never parses this - it only needs common/kv-socket.h.
//
// NO FILES: the KV blob never touches disk. The carrier holds it in its own
// RAM: dump -> carrier RAM -> (server dies) -> new server restores it.

#include "kv-socket.h"
#include "server-task.h"

#include <cstdint>
#include <cstring>
#include <list>
#include <vector>

namespace kvsock {

// ---------------- little-endian (de)serialization of prompt-cache states ----
//
// layout:
//   u32 n_states
//   per state:
//     i32 n_tokens, i32 tokens[n_tokens]
//     u32 n_ckpts
//     per ckpt: i64 n_tokens, i32 id_task, i32 pos_min, i32 pos_max,
//               u32 n, u8 data_tgt[n], u32 n, u8 data_dft[n], u32 n, u8 data_spec[n]
//     u32 n_main, u8 main[n_main]
//     u32 n_drft, u8 drft[n_drft]

inline void w32(std::vector<uint8_t> & v, uint32_t x) {
    const uint8_t * b = reinterpret_cast<const uint8_t *>(&x);
    v.insert(v.end(), b, b + 4);
}
inline void w64(std::vector<uint8_t> & v, uint64_t x) {
    const uint8_t * b = reinterpret_cast<const uint8_t *>(&x);
    v.insert(v.end(), b, b + 8);
}
inline void wblob(std::vector<uint8_t> & v, const uint8_t * p, size_t n) {
    v.insert(v.end(), p, p + n);
}
inline bool r32(const uint8_t *& p, const uint8_t * end, uint32_t & x) {
    if (end - p < 4) return false;
    std::memcpy(&x, p, 4);
    p += 4;
    return true;
}
inline bool r64(const uint8_t *& p, const uint8_t * end, uint64_t & x) {
    if (end - p < 8) return false;
    std::memcpy(&x, p, 8);
    p += 8;
    return true;
}
inline bool rblob(const uint8_t *& p, const uint8_t * end, std::vector<uint8_t> & out, size_t n) {
    if (end - p < (ptrdiff_t) n) return false;
    out.assign(p, p + n);
    p += n;
    return true;
}

inline void serialize_states(const std::list<server_prompt_cache_state> & states, std::vector<uint8_t> & out) {
    out.clear();
    w32(out, (uint32_t) states.size());
    for (const auto & st : states) {
        // v1: text-only prompt cache states (no media chunks)
        if (st.prompt.tokens.has_mtmd) {
            GGML_ABORT("kv: media chunks in the prompt cache are not supported yet");
        }
        const llama_tokens & toks = st.prompt.tokens.get_tokens();
        w32(out, (uint32_t) toks.size());
        for (auto t : toks) {
            w32(out, (uint32_t) t);
        }
        w32(out, (uint32_t) st.prompt.checkpoints.size());
        for (const auto & ck : st.prompt.checkpoints) {
            w64(out, (uint64_t) ck.n_tokens);
            w32(out, (uint32_t) ck.id_task);
            w32(out, (uint32_t) ck.pos_min);
            w32(out, (uint32_t) ck.pos_max);
            w32(out, (uint32_t) ck.data_tgt.size());  wblob(out, ck.data_tgt.data(),  ck.data_tgt.size());
            w32(out, (uint32_t) ck.data_dft.size());  wblob(out, ck.data_dft.data(),  ck.data_dft.size());
            w32(out, (uint32_t) ck.data_spec.size()); wblob(out, ck.data_spec.data(), ck.data_spec.size());
        }
        w32(out, (uint32_t) st.data.main.size()); wblob(out, st.data.main.data(), st.data.main.size());
        w32(out, (uint32_t) st.data.drft.size()); wblob(out, st.data.drft.data(), st.data.drft.size());
    }
}

inline bool deserialize_states(const uint8_t * data, size_t len, std::list<server_prompt_cache_state> & out) {
    const uint8_t * p = data;
    const uint8_t * end = data + len;
    uint32_t n_states = 0;
    if (!r32(p, end, n_states)) return false;
    if (n_states > 1024) return false;
    out.clear();
    for (uint32_t i = 0; i < n_states; i++) {
        server_prompt_cache_state st;
        uint32_t n_tokens = 0;
        if (!r32(p, end, n_tokens)) return false;
        if (n_tokens > (1u << 24)) return false;
        llama_tokens toks(n_tokens);
        for (uint32_t t = 0; t < n_tokens; t++) {
            uint32_t tok = 0;
            if (!r32(p, end, tok)) return false;
            toks[t] = (llama_token) tok;
        }
        st.prompt.tokens = server_tokens(toks, false /* has_mtmd */);
        uint32_t n_ckpts = 0;
        if (!r32(p, end, n_ckpts)) return false;
        if (n_ckpts > 1024) return false;
        for (uint32_t c = 0; c < n_ckpts; c++) {
            common_prompt_checkpoint ck;
            uint64_t x64 = 0;
            uint32_t x = 0;
            if (!r64(p, end, x64)) return false; ck.n_tokens = (int64_t) x64;
            if (!r32(p, end, x)) return false;   ck.id_task  = (int32_t) x;
            if (!r32(p, end, x)) return false;   ck.pos_min  = (llama_pos) x;
            if (!r32(p, end, x)) return false;   ck.pos_max  = (llama_pos) x;
            if (!r32(p, end, x)) return false;
            if (!rblob(p, end, ck.data_tgt, x)) return false;
            if (!r32(p, end, x)) return false;
            if (!rblob(p, end, ck.data_dft, x)) return false;
            if (!r32(p, end, x)) return false;
            if (!rblob(p, end, ck.data_spec, x)) return false;
            st.prompt.checkpoints.push_back(std::move(ck));
        }
        uint32_t n = 0;
        if (!r32(p, end, n)) return false;
        if (!rblob(p, end, st.data.main, n)) return false;
        if (!r32(p, end, n)) return false;
        if (!rblob(p, end, st.data.drft, n)) return false;
        out.push_back(std::move(st));
    }
    return p == end;
}

} // namespace kvsock
