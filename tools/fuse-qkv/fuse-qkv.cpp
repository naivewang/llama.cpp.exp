// fuse-qkv: rewrite a GGUF so each layer's separate attn_q/attn_k/attn_v weights
// are replaced by a single fused attn_qkv.weight [ne0, neq1+nk1+nv1] (rows concatenated
// Q|K|V). This lets llama.cpp use the existing fused-QKV graph path (one GEMV instead of
// three per layer). Block-safe: quantization is along ne0; rows (ne1) are concatenated.
//
// Only QKV is fused. FFN gate/up stay separate (dense build_ffn has no fused path).
// Non-QKV tensors are copied verbatim from the input file; metadata is copied.
//
// usage: fuse-qkv input.gguf output.gguf
#include "gguf.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

// returns part (0=q,1=k,2=v) and layer via *il if name is "blk.<N>.attn_{q,k,v}.weight", else -1
static int parse_qkv(const char * name, int * il) {
    if (strncmp(name, "blk.", 4) != 0) return -1;
    const char * p = name + 4;
    int n = 0; bool dig = false;
    while (*p >= '0' && *p <= '9') { n = n*10 + (*p - '0'); p++; dig = true; }
    if (!dig) return -1;
    *il = n;
    if (strcmp(p, ".attn_q.weight") == 0) return 0;
    if (strcmp(p, ".attn_k.weight") == 0) return 1;
    if (strcmp(p, ".attn_v.weight") == 0) return 2;
    return -1;
}

int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s in.gguf out.gguf\n", argv[0]); return 1; }
    std::string in_path = argv[1], out_path = argv[2];

    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params p = { /*.no_alloc*/ true, /*.ctx*/ &ctx_meta };
    auto * ctx_in = gguf_init_from_file(in_path.c_str(), p);
    if (!ctx_in) { fprintf(stderr, "failed to load %s\n", in_path.c_str()); return 1; }
    const int n_in = gguf_get_n_tensors(ctx_in);
    printf("input: %d tensors\n", n_in);

    auto * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);

    // data context holding the fused tensors (6 GB is ample for <=27B dense QKV)
    const size_t MEM = 8ull*1024*1024*1024;
    struct ggml_init_params gp = { .mem_size = MEM, .no_alloc = false };
    struct ggml_context * ctx_data = ggml_init(gp);
    if (!ctx_data) { fprintf(stderr, "ggml_init failed (%zu MB)\n", MEM/1048576); return 1; }

    // read a tensor's raw bytes from the input file
    std::ifstream fin(in_path, std::ios::binary);
    if (!fin.is_open()) { fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }
    const size_t data_off_in = gguf_get_data_offset(ctx_in);
    auto read_tensor = [&](const char * tname) -> std::vector<char> {
        const int idx = gguf_find_tensor(ctx_in, tname);
        if (idx < 0) { fprintf(stderr, "not found in input: %s\n", tname); exit(1); }
        const size_t n = gguf_get_tensor_size(ctx_in, idx);
        fin.seekg(data_off_in + gguf_get_tensor_offset(ctx_in, idx));
        std::vector<char> b(n);
        fin.read(b.data(), n);
        return b;
    };

    std::unordered_map<std::string, struct ggml_tensor *> fused;

    for (int i = 0; i < n_in; ++i) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        int il = -1;
        const int part = parse_qkv(name, &il);

        if (part == 0) {
            // attn_q.weight -> emit fused qkv here
            char nq[128], nk[128], nv[128], nf[128];
            snprintf(nq, sizeof nq, "blk.%d.attn_q.weight", il);
            snprintf(nk, sizeof nk, "blk.%d.attn_k.weight", il);
            snprintf(nv, sizeof nv, "blk.%d.attn_v.weight", il);
            snprintf(nf, sizeof nf, "blk.%d.attn_qkv.weight", il);

            auto * tq = ggml_get_tensor(ctx_meta, nq);
            auto * tk = ggml_get_tensor(ctx_meta, nk);
            auto * tv = ggml_get_tensor(ctx_meta, nv);
            if (!tq || !tk || !tv) { fprintf(stderr, "missing q/k/v for blk.%d\n", il); return 1; }
            if (tq->type != tk->type || tq->type != tv->type) { fprintf(stderr, "type mismatch blk.%d\n", il); return 1; }
            if (tq->ne[0] != tk->ne[0] || tq->ne[0] != tv->ne[0]) { fprintf(stderr, "ne0 mismatch blk.%d\n", il); return 1; }

            const int64_t ne0 = tq->ne[0];
            const int64_t nq1 = tq->ne[1], nk1 = tk->ne[1], nv1 = tv->ne[1];
            const int64_t ne1 = nq1 + nk1 + nv1;
            int64_t ne[4] = { ne0, ne1, 1, 1 };
            auto * tf = ggml_new_tensor(ctx_data, tq->type, 2, ne);
            if (!tf) { fprintf(stderr, "ggml_new_tensor failed blk.%d\n", il); return 1; }
            ggml_set_name(tf, nf);

            std::vector<char> bq = read_tensor(nq);
            std::vector<char> bk = read_tensor(nk);
            std::vector<char> bv = read_tensor(nv);
            const size_t row = ggml_row_size(tq->type, ne0);
            char * dst = (char *) tf->data;
            for (int64_t r = 0; r < nq1; ++r) memcpy(dst + (r)*row,         bq.data() + r*row, row);
            for (int64_t r = 0; r < nk1; ++r) memcpy(dst + (nq1+r)*row,     bk.data() + r*row, row);
            for (int64_t r = 0; r < nv1; ++r) memcpy(dst + (nq1+nk1+r)*row, bv.data() + r*row, row);

            gguf_add_tensor(ctx_out, tf);
            fused[nf] = tf;
            continue;
        }
        if (part == 1 || part == 2) continue; // folded into attn_qkv

        // copy everything else verbatim (metadata from ggml view; data from file at write)
        auto * t = ggml_get_tensor(ctx_meta, name);
        if (!t) { fprintf(stderr, "no ggml tensor for %s\n", name); return 1; }
        gguf_add_tensor(ctx_out, t);
    }

    printf("output: %d tensors (%zu fused qkv)\n", gguf_get_n_tensors(ctx_out), fused.size());

    // ---- write the output file: metadata, then per-tensor data ----
    std::ofstream fout(out_path, std::ios::binary);
    if (!fout.is_open()) { fprintf(stderr, "cannot open %s for write\n", out_path.c_str()); return 1; }

    std::vector<uint8_t> meta(gguf_get_meta_size(ctx_out));
    gguf_get_meta_data(ctx_out, meta.data());
    fout.write((const char *) meta.data(), meta.size());

    for (int i = 0; i < gguf_get_n_tensors(ctx_out); ++i) {
        const char * name = gguf_get_tensor_name(ctx_out, i);
        size_t n = 0;
        if (fused.count(name)) {
            auto * t = fused[name];
            n = ggml_nbytes(t);
            fout.write((const char *) t->data, n);
        } else {
            const int idx = gguf_find_tensor(ctx_in, name);
            if (idx < 0) { fprintf(stderr, "tensor %s not in input\n", name); return 1; }
            n = gguf_get_tensor_size(ctx_in, idx);
            fin.seekg(data_off_in + gguf_get_tensor_offset(ctx_in, idx));
            std::vector<char> buf(n);
            fin.read(buf.data(), n);
            fout.write(buf.data(), n);
        }
        const size_t aligned = (n + GGUF_DEFAULT_ALIGNMENT - 1) & ~((size_t)GGUF_DEFAULT_ALIGNMENT - 1);
        const size_t pad = aligned - n;
        if (pad) { std::vector<char> z(pad, 0); fout.write(z.data(), pad); }
    }
    fout.close();
    printf("wrote %s\n", out_path.c_str());
    return 0;
}
