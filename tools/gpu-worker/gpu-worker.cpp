// llama-gpu-worker: "sleep = drop the GPU process", llama.cpp edition.
//
// build (repo root; standalone, no CMake target):
//   g++ -std=c++17 \
//     -DCPPHTTPLIB_OPENSSL_SUPPORT -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_HIP -DLLAMA_SHARED \
//     -Icommon/. -Ivendor/nlohmann/.. -Ivendor/nlohmann -Ivendor/sheredom/.. \
//     -Isrc/../include -Iggml/src/../include \
//     -O2 -fPIC tools/gpu-worker/gpu-worker.cpp -o <build>/bin/llama-gpu-worker \
//     -L<build>/bin -Wl,-rpath,$PWD/<build>/bin \
//     -lllama -lllama-common -lggml-hip -lggml-cpu -lggml-base -lpthread -ldl
//
// One binary, three modes:
//
//   worker (child, owns the GPU):
//     llama-gpu-worker --worker --uds /tmp/x.sock -m MODEL -c 4096 -ngl 99 [...]
//     loads model+context (GPU per usual llama.cpp args), listens on a UNIX
//     domain socket and serves: INIT, DECODE, DUMP_KV, RESTORE_KV, SHUTDOWN, PING
//
//   server (the carrier: this binary's MAIN THREAD drives a real llama-server child):
//     llama-gpu-worker --server -m MODEL [-ngl 99 -c 4096 ...server args...] \
//                      [--prompt "..."] [--n-gen 16] [--n-resume 8] \
//                      [--port N] [--server-bin PATH]
//     The carrier (main thread) never loads a model and never touches a GPU.
//     It fork+execs llama-server with an inherited socketpair fd (--kv-fd, no
//     files anywhere) and is the KV-state "watchdog": the dumped KV blob lives
//     in the carrier's own RAM as opaque bytes.
//       1. control run: spawn(SKIP) -> prompt + (n_gen + n_resume) tokens
//       2. sleep run:   spawn(SKIP) -> prompt + n_gen tokens -> DUMP -> blob
//          in carrier RAM -> SHUTDOWN over the kv channel ("inform and exit":
//          the server releases the GPU itself and exits 0; SIGKILL is only a
//          fallback) -> respawn(RESTORE blob) -> prompt + generated-so-far +
//          n_resume more tokens
//       3. greedy decoding is deterministic, so slept output must equal the
//          control-run output exactly
//     Inference over HTTP (/v1/completions); everything else passed through
//     to the server verbatim. rocm-smi / nvidia-smi snapshots per phase.
//
//   client (parent, GPU-free service side) - the earlier raw-UDS PoC:
//     llama-gpu-worker -m MODEL [--prompt "..."] [--n-gen 16] [--n-resume 8]
//     fork+execs a --worker child over a UNIX domain socket (a file, unlike
//     --server mode) with raw DECODE / DUMP_KV / RESTORE_KV. Kept as a direct
//     KV-operation reference; prefer --server mode.

#include "gpu-worker.h"

#include "common.h"
#include "arg.h"
#include "log.h"
#include "kv-socket.h"
#include "llama.h"

#include "json.hpp" // vendored nlohmann (carrier mode's HTTP responses only)

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <signal.h>
#include <arpa/inet.h>
#include <netinet/in.h>

// ------------------------------------------------------------------ wire

static bool read_all(int fd, void * buf, size_t n) {
    uint8_t * p = static_cast<uint8_t *>(buf);
    while (n > 0) {
        const ssize_t r = read(fd, p, n);
        if (r <= 0) return false;
        p += r; n -= static_cast<size_t>(r);
    }
    return true;
}

static bool write_all(int fd, const void * buf, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(buf);
    while (n > 0) {
        const ssize_t w = write(fd, p, n);
        if (w <= 0) return false;
        p += w; n -= static_cast<size_t>(w);
    }
    return true;
}

static void put_u32(std::vector<uint8_t> & v, uint32_t x) {
    const uint8_t b[4] = { uint8_t(x), uint8_t(x >> 8), uint8_t(x >> 16), uint8_t(x >> 24) };
    v.insert(v.end(), b, b + 4);
}
static void put_i32(std::vector<uint8_t> & v, int32_t x) { put_u32(v, uint32_t(x)); }
static uint32_t get_u32(const uint8_t * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
static int32_t get_i32(const uint8_t * p) { return int32_t(get_u32(p)); }
static float  get_f32(const uint8_t * p) {
    uint32_t u = get_u32(p); float f; std::memcpy(&f, &u, 4); return f;
}

static bool gwp_debug = [](){ const char * e = getenv("GWP_DEBUG"); return e && e[0]; }();
#define DBG(fmt, ...) do { if (gwp_debug) { fprintf(stderr, "[gwp pid=%d] ", (int) getpid()); fprintf(stderr, fmt "\n", ##__VA_ARGS__); fflush(stderr);} } while(0)
static std::string hex8(const uint8_t * p) {
    static const char * h = "0123456789abcdef";
    std::string s; s.reserve(16);
    for (int i = 0; i < 8; i++) { s += h[p[i] >> 4]; s += h[p[i] & 15]; }
    return s;
}

static bool send_frame(int fd, uint32_t op, const std::vector<uint8_t> & payload) {
    std::vector<uint8_t> hdr;   // put_u32 appends; must start empty so bytes land at 0..7
    put_u32(hdr, uint32_t(gwp::HDR_LEN + payload.size()));
    put_u32(hdr, op);
    DBG("send op=%u payload=%zu raw=%s", op, payload.size(), hex8(hdr.data()).c_str());
    return write_all(fd, hdr.data(), hdr.size()) &&
           (payload.empty() || write_all(fd, payload.data(), payload.size()));
}

static bool send_frame_raw(int fd, uint32_t op, const uint8_t * payload, size_t n) {
    std::vector<uint8_t> hdr;   // put_u32 appends; must start empty so bytes land at 0..7
    put_u32(hdr, uint32_t(gwp::HDR_LEN + n));
    put_u32(hdr, op);
    DBG("send op=%u payload=%zu raw=%s", op, n, hex8(hdr.data()).c_str());
    return write_all(fd, hdr.data(), hdr.size()) &&
           (n == 0 || write_all(fd, payload, n));
}

static bool recv_frame(int fd, uint32_t * op, std::vector<uint8_t> * payload) {
    uint8_t hdr[8];
    if (!read_all(fd, hdr, sizeof(hdr))) return false;
    const uint32_t total = get_u32(hdr);
    *op = get_u32(hdr + 4);
    DBG("recv op=%u total=%u raw=%s", *op, total, hex8(hdr).c_str());
    if (total < gwp::HDR_LEN || total > 4ull * 1024 * 1024 * 1024) {
        fprintf(stderr, "[gwp pid=%d] bad frame length %u (op=%u)\n", (int) getpid(), total, *op);
        return false;
    }
    payload->resize(total - gwp::HDR_LEN);
    return payload->empty() || read_all(fd, payload->data(), payload->size());
}

// ------------------------------------------------------------------ worker

static int run_worker(int argc, char ** argv) {
    std::string uds_path;
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--worker") continue;
        if (a == "--uds") { uds_path = argv[++i]; continue; }
        args.push_back(a);
    }
    if (uds_path.empty()) {
        fprintf(stderr, "worker: --uds PATH is required\n");
        return 1;
    }

    // build a proper argv (argv[0] = program name) for the common parser
    std::vector<std::string> storage;
    storage.push_back("llama-gpu-worker");
    for (auto & a : args) storage.push_back(a);
    std::vector<char *> cargv(storage.size());
    for (size_t i = 0; i < storage.size(); i++) cargv[i] = storage[i].data();

    common_params params;
    common_init();
    if (!common_params_parse(static_cast<int>(storage.size()), cargv.data(),
            params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    // the worker serves a single sequence; cap n_seq_max so context init can't fail
    params.n_parallel = 1;

    common_init_result_ptr init_res = common_init_from_params(params);
    if (!init_res) {
        fprintf(stderr, "worker: failed to load model\n");
        return 1;
    }
    llama_model * model = init_res->model();
    llama_context * lctx = init_res->context();
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    unlink(uds_path.c_str());
    int lfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (lfd < 0) { perror("socket"); return 1; }
    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    if (snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", uds_path.c_str()) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "worker: uds path too long\n");
        return 1;
    }
    if (bind(lfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(lfd, 1) < 0) { perror("listen"); return 1; }

    printf("[worker pid=%d] ready on %s (vocab=%d, ctx=%d)\n",
           (int) getpid(), uds_path.c_str(), n_vocab, params.n_ctx);
    fflush(stdout);

    for (;;) {
        const int cfd = accept(lfd, nullptr, nullptr);
        if (cfd < 0) { perror("accept"); return 1; }
        printf("[worker pid=%d] client connected\n", (int) getpid());
        fflush(stdout);

        bool client_alive = true;
        while (client_alive) {
            uint32_t op = 0;
            std::vector<uint8_t> req;
            if (!recv_frame(cfd, &op, &req)) { client_alive = false; break; }

            switch (op) {
                case gwp::OP_INIT: {
                    std::vector<uint8_t> rsp;
                    put_i32(rsp, n_vocab);
                    put_i32(rsp, params.n_ctx);
                    client_alive = send_frame(cfd, gwp::OP_INIT, rsp);
                    break;
                }
                case gwp::OP_DECODE: {
                    if (req.size() < 8) { client_alive = false; break; }
                    const int n_tokens = get_i32(req.data());
                    if (n_tokens <= 0 || size_t(n_tokens) * 4 + 8 > req.size()) { client_alive = false; break; }

                    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
                    batch.n_tokens = n_tokens; // llama_batch_init leaves n_tokens unset
                    const int32_t pos = llama_memory_seq_pos_max(llama_get_memory(lctx), 0) + 1; // -1 + 1 == 0 when empty
                    for (int i = 0; i < n_tokens; i++) {
                        batch.token[i]     = get_i32(req.data() + 8 + i * 4);
                        batch.pos[i]       = pos + i;
                        batch.n_seq_id[i]  = 1;
                        batch.seq_id[i][0] = 0;
                        batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
                    }
                    const int rc = llama_decode(lctx, batch);
                    llama_batch_free(batch);
                    if (rc != 0) {
                        fprintf(stderr, "[worker] llama_decode failed (%d)\n", rc);
                        client_alive = false;
                        break;
                    }
                    client_alive = send_frame_raw(cfd, gwp::OP_DECODE,
                            reinterpret_cast<const uint8_t *>(llama_get_logits(lctx)),
                            size_t(n_vocab) * 4);
                    break;
                }
                case gwp::OP_DUMP_KV: {
                    if (req.size() < 4) { client_alive = false; break; }
                    const auto seq = get_i32(req.data());
                    const size_t size = llama_state_seq_get_size_ext(lctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                    std::vector<uint8_t> blob(size);
                    if (size) llama_state_seq_get_data_ext(lctx, blob.data(), size, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                    std::vector<uint8_t> rsp;
                    put_u32(rsp, uint32_t(size));
                    rsp.insert(rsp.end(), blob.begin(), blob.end());
                    client_alive = send_frame(cfd, gwp::OP_DUMP_KV, rsp);
                    printf("[worker pid=%d] dumped KV seq %d: %.2f MiB\n",
                           (int) getpid(), int(seq), size / 1048576.0);
                    fflush(stdout);
                    break;
                }
                case gwp::OP_RESTORE_KV: {
                    if (req.size() < 12) { client_alive = false; break; }
                    const auto seq  = get_i32(req.data());
                    const size_t size = get_u32(req.data() + 4);
                    // header = seq(4) + size(4) + 2 pads(8) = 16 bytes; kv blob starts at offset 16
                    if (16 + size > req.size()) { client_alive = false; break; }
                    const size_t written = llama_state_seq_set_data_ext(
                        lctx, req.data() + 16, size, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                    std::vector<uint8_t> rsp;
                    put_u32(rsp, written == size ? 1 : 0);
                    client_alive = send_frame(cfd, gwp::OP_RESTORE_KV, rsp);
                    printf("[worker pid=%d] restored KV seq %d: %.2f MiB (written %zu)\n",
                           (int) getpid(), int(seq), size / 1048576.0, written);
                    fflush(stdout);
                    break;
                }
                case gwp::OP_PING: {
                    std::vector<uint8_t> rsp; put_u32(rsp, 1);
                    client_alive = send_frame(cfd, gwp::OP_PING, rsp);
                    break;
                }
                case gwp::OP_SHUTDOWN: {
                    std::vector<uint8_t> rsp; put_u32(rsp, 1);
                    send_frame(cfd, gwp::OP_SHUTDOWN, rsp);
                    printf("[worker pid=%d] shutdown requested\n", (int) getpid());
                    fflush(stdout);
                    close(cfd);
                    close(lfd);
                    return 0;
                }
                default:
                    fprintf(stderr, "[worker] unknown op %u\n", op);
                    client_alive = false;
            }
        }
        close(cfd);
        printf("[worker pid=%d] client disconnected, waiting for next\n", (int) getpid());
        fflush(stdout);
    }
}

// ------------------------------------------------------------------ client

namespace {

// a forked worker process + its ready pipe
struct worker_proc {
    pid_t pid = -1;
    int out_pipe = -1;
    std::string uds;

    bool start(const std::string & exe, const std::string & model, int n_ctx, int ngl) {
        char tmp[] = "/tmp/llama-gpu-worker-XXXXXX";
        if (!mkdtemp(tmp)) return false;
        uds = std::string(tmp) + "/worker.sock";

        std::vector<std::string> storage = {
            exe, "--worker", "--uds", uds,
            "-m", model, "-c", std::to_string(n_ctx), "-ngl", std::to_string(ngl),
        };
        std::vector<char *> cargs(storage.size() + 1, nullptr);
        for (size_t i = 0; i < storage.size(); i++) cargs[i] = const_cast<char *>(storage[i].c_str());

        int pfd[2];
        if (pipe(pfd) < 0) return false;
        const pid_t child = fork();
        if (child < 0) { perror("fork"); return false; }
        if (child == 0) {
            dup2(pfd[1], STDOUT_FILENO);
            close(pfd[0]);
            close(pfd[1]);
            execv(exe.c_str(), cargs.data());
            _exit(127);
        }
        close(pfd[1]);
        out_pipe = pfd[0];
        pid = child;

        // wait for the ready line (init output goes to stderr)
        char c;
        for (int lines = 0; lines < 300; lines++) {
            std::string line;
            ssize_t r;
            while ((r = read(out_pipe, &c, 1)) == 1) {
                if (c == '\n') break; // complete line
                line += c;
            }
            if (line.find("ready on") != std::string::npos) {
                printf("client: %s\n", line.c_str());
                return true;
            }
            if (r <= 0) return false; // EOF - no more data will arrive
        }
        kill_hard();
        return false;
    }

    void kill_hard() {
        if (pid > 0) {
            kill(pid, SIGKILL);
            int st;
            while (waitpid(pid, &st, 0) < 0 && errno == EINTR);
            printf("client: worker pid %d SIGKILLed (signal %d), reaped\n", (int) pid, st & 0x7f);
            pid = -1;
        }
        if (out_pipe >= 0) { close(out_pipe); out_pipe = -1; }
    }

    void shutdown_clean() {
        while (pid > 0 && waitpid(pid, nullptr, 0) < 0 && errno == EINTR);
        pid = -1;
        if (out_pipe >= 0) { close(out_pipe); out_pipe = -1; }
    }
};

int connect_uds(const std::string & uds) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", uds.c_str());
    if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) { close(fd); return -1; }
    return fd;
}

void snapshot(const char * label) {
    printf("\n========== %s ==========\n", label);
    fflush(stdout);
    (void) system(
        "command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showmemuse --showpids 2>/dev/null | "
        "awk '/Current Memory Use|GPU Memory Allocated|KFD process information/{p=1} /End of ROCm SMI/{p=0} p && NF' || "
        "(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,power.draw --format=csv || "
        "echo '(no smi tool found)')");
    fflush(stdout);
}

// returns false on transport error
bool op_decode(int fd, const std::vector<llama_token> & tokens, std::vector<float> * logits) {
    std::vector<uint8_t> req;
    put_i32(req, int32_t(tokens.size()));
    put_i32(req, 0);
    for (auto t : tokens) put_i32(req, int32_t(t));
    if (!send_frame(fd, gwp::OP_DECODE, req)) return false;
    uint32_t op;
    std::vector<uint8_t> rsp;
    if (!recv_frame(fd, &op, &rsp) || op != gwp::OP_DECODE) return false;
    logits->resize(rsp.size() / 4);
    for (size_t i = 0; i < logits->size(); i++) (*logits)[i] = get_f32(rsp.data() + i * 4);
    return true;
}

bool op_init(int fd, int * n_vocab, int * n_ctx) {
    std::vector<uint8_t> req;
    if (!send_frame(fd, gwp::OP_INIT, req)) return false;
    uint32_t op;
    std::vector<uint8_t> rsp;
    if (!recv_frame(fd, &op, &rsp) || op != gwp::OP_INIT || rsp.size() < 8) return false;
    *n_vocab = get_i32(rsp.data());
    *n_ctx = get_i32(rsp.data() + 4);
    return true;
}

bool op_shutdown(int fd) {
    std::vector<uint8_t> req;
    if (!send_frame(fd, gwp::OP_SHUTDOWN, req)) return false;
    uint32_t op;
    std::vector<uint8_t> rsp;
    return recv_frame(fd, &op, &rsp) && op == gwp::OP_SHUTDOWN;
}

} // namespace

static int run_client(int argc, char ** argv) {
    std::string prompt = "Explain in one sentence what a GPU is. Then list three typical uses.";
    int n_gen = 16, n_resume = 8, worker_ngl = 99, worker_ctx = 4096;
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--prompt")     { prompt = argv[++i]; continue; }
        if (a == "--n-gen")      { n_gen = atoi(argv[++i]); continue; }
        if (a == "--n-resume")   { n_resume = atoi(argv[++i]); continue; }
        if (a == "--worker-ngl") { worker_ngl = atoi(argv[++i]); continue; }
        if (a == "--worker-ctx") { worker_ctx = atoi(argv[++i]); continue; }
        args.push_back(a);
    }

    // build a proper argv (argv[0] = program name) for the common parser
    std::vector<std::string> storage;
    storage.push_back("llama-gpu-worker");
    for (auto & a : args) storage.push_back(a);
    std::vector<char *> cargv(storage.size());
    for (size_t i = 0; i < storage.size(); i++) cargv[i] = storage[i].data();

    common_params params;
    common_init();
    if (!common_params_parse(static_cast<int>(storage.size()), cargv.data(),
            params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "client: -m MODEL is required\n");
        return 1;
    }
    // The service side never uses the GPU: CPU-only model, tokenization only.
    // We deliberately do NOT call llama_backend_init() here: the client must stay
    // out of the GPU process table entirely.
    params.n_gpu_layers = 0;

    char exe[4096];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) { fprintf(stderr, "client: cannot resolve /proc/self/exe\n"); return 1; }
    exe[n] = 0;

    common_init_result_ptr init_res = common_init_from_params(params, /*model_only=*/true);
    if (!init_res) { fprintf(stderr, "client: failed to load model (CPU, tokenization only)\n"); return 1; }
    llama_model * model = init_res->model();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, prompt, /*add_special=*/true);
    printf("client pid=%d (GPU-free service side; %zu prompt tokens)\n",
           (int) getpid(), prompt_tokens.size());

    auto greedy = [](const std::vector<float> & logits) {
        return int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
    };

    bool ok = true;
    std::vector<llama_token> control, slept;

    snapshot("PHASE 0: baseline (no worker)");

    // ---- run 1: control (never sleeps)
    {
        worker_proc w;
        if (!w.start(exe, params.model.path, worker_ctx, worker_ngl)) return 1;
        const int fd = connect_uds(w.uds);
        if (fd < 0) { fprintf(stderr, "client: connect failed\n"); w.kill_hard(); return 1; }

        int n_vocab = 0, n_ctx = 0;
        if (!op_init(fd, &n_vocab, &n_ctx)) ok = false;
        else printf("client: INIT ok (vocab=%d, ctx=%d)\n", n_vocab, n_ctx);

        std::vector<float> logits;
        if (ok && !op_decode(fd, prompt_tokens, &logits)) ok = false;
        for (int i = 0; ok && i < n_gen + n_resume; i++) {
            const int32_t t = greedy(logits);
            control.push_back(llama_token(t));
            std::vector<llama_token> one = { llama_token(t) };
            if (!op_decode(fd, one, &logits)) ok = false;
        }
        if (ok && !op_shutdown(fd)) ok = false;
        close(fd);
        w.shutdown_clean();
        printf("client: control run done (%zu tokens)\n", control.size());
    }

    snapshot("PHASE 1: control worker exited (GPU should be idle)");

    // ---- run 2: sleep/wake
    {
        worker_proc w;
        if (!w.start(exe, params.model.path, worker_ctx, worker_ngl)) return 1;
        int fd = connect_uds(w.uds);
        if (fd < 0) { fprintf(stderr, "client: connect failed\n"); w.kill_hard(); return 1; }

        int n_vocab = 0, n_ctx = 0;
        if (!op_init(fd, &n_vocab, &n_ctx)) ok = false;

        std::vector<float> logits;
        int32_t pending = -1; // sampled but not yet decoded; crosses the sleep boundary
        if (ok && !op_decode(fd, prompt_tokens, &logits)) ok = false;
        for (int i = 0; ok && i < n_gen; i++) {
            pending = greedy(logits);
            slept.push_back(llama_token(pending));
            if (i < n_gen - 1) {
                std::vector<llama_token> one = { llama_token(pending) };
                if (!op_decode(fd, one, &logits)) ok = false;
            }
        }
        // worker state: prompt + slept[0..n_gen-2]; client holds slept[n_gen-1] undecoded
        snapshot("PHASE 1b: worker alive (expect: worker listed, VRAM up)");

        // dump KV to RAM
        std::vector<uint8_t> kv;
        if (ok) {
            std::vector<uint8_t> dreq;
            put_i32(dreq, 0);
            if (!send_frame(fd, gwp::OP_DUMP_KV, dreq)) ok = false;
            else {
                uint32_t op;
                std::vector<uint8_t> rsp;
                if (!recv_frame(fd, &op, &rsp) || op != gwp::OP_DUMP_KV || rsp.size() < 4) ok = false;
                else {
                    const size_t size = get_u32(rsp.data());
                    if (4 + size > rsp.size()) ok = false;
                    else kv.assign(rsp.begin() + 4, rsp.begin() + 4 + size);
                }
            }
        }
        if (ok) printf("client: KV blob in RAM: %.2f MiB\n", kv.size() / 1048576.0);
        close(fd);
        fd = -1;

        w.kill_hard();
        sleep(2);
        snapshot("PHASE 2: worker dropped (expect: no worker pid, VRAM ~0)");

        // respawn + restore
        if (ok && !w.start(exe, params.model.path, worker_ctx, worker_ngl)) ok = false;
        if (ok) {
            fd = connect_uds(w.uds);
            if (fd < 0) ok = false;
        }
        if (ok) {
            int nv = 0, nc = 0;
            if (!op_init(fd, &nv, &nc)) ok = false;
            std::vector<uint8_t> rreq;
            put_i32(rreq, 0);
            put_u32(rreq, uint32_t(kv.size()));
            put_u32(rreq, 0);
            put_u32(rreq, 0);
            rreq.insert(rreq.end(), kv.begin(), kv.end());
            if (!send_frame(fd, gwp::OP_RESTORE_KV, rreq)) ok = false;
            else {
                uint32_t op;
                std::vector<uint8_t> rsp;
                if (!recv_frame(fd, &op, &rsp) || op != gwp::OP_RESTORE_KV) ok = false;
                else printf("client: RESTORE_KV ok=%u\n", get_u32(rsp.data()));
            }
        }
        if (ok) {
            // decode the token that was pending at sleep time, then continue
            std::vector<llama_token> one = { llama_token(pending) };
            if (!op_decode(fd, one, &logits)) ok = false;
            for (int i = 0; ok && i < n_resume; i++) {
                const int32_t t = greedy(logits);
                slept.push_back(llama_token(t));
                std::vector<llama_token> next = { llama_token(t) };
                if (!op_decode(fd, next, &logits)) ok = false;
            }
        }
        snapshot("PHASE 3: worker respawned + KV restored (expect: worker listed, VRAM up)");

        if (ok && fd >= 0 && !op_shutdown(fd)) ok = false;
        if (fd >= 0) close(fd);
        w.shutdown_clean();
    }

    snapshot("PHASE 4: all workers exited (GPU should be idle)");

    // ---- verify: slept must equal control token-for-token (greedy = deterministic)
    const size_t n_check = std::min(slept.size(), control.size());
    bool same = (slept.size() == control.size());
    for (size_t i = 0; same && i < n_check; i++) {
        if (slept[i] != control[i]) same = false;
    }
    printf("\n================ SUMMARY ================\n");
    printf("control tokens : %zu\n", control.size());
    printf("slept tokens   : %zu (pre-sleep %d, post-restore %d)\n",
           slept.size(), n_gen, int(std::max<size_t>(0, slept.size() - n_gen)));
    printf("continuity (slept == control): %s\n", (same && ok) ? "PASS" : "FAIL");
    printf("control text : %s\n", common_detokenize(vocab, control).c_str());
    printf("slept text   : %s\n", common_detokenize(vocab, slept).c_str());
    return (ok && same) ? 0 : 1;
}

// ------------------------------------------------------------------ server mode
//
// The carrier: this binary's main thread drives a real llama-server child.
// No files anywhere - the KV channel is a socketpair whose child end is
// inherited across fork+exec as fd GWP_KV_FD (--kv-fd). The KV blob is held
// as opaque bytes in the carrier's RAM: dump -> RAM -> SIGKILL -> respawn ->
// restore. The carrier never loads a model or touches a GPU.

namespace {

constexpr int GWP_KV_FD = 31;

struct http_resp {
    int status = 0;
    std::string body;
};

static http_resp http_request(const std::string & host, int port, const std::string & path, const std::string & body) {
    http_resp resp;
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return resp;
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    addr.sin_addr.s_addr = inet_addr(host.c_str());
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) { ::close(fd); return resp; }
    const std::string req = "POST " + path + " HTTP/1.1\r\n"
                            "Host: " + host + ":" + std::to_string(port) + "\r\n"
                            "Content-Type: application/json\r\n"
                            "Content-Length: " + std::to_string(body.size()) + "\r\n"
                            "Connection: close\r\n\r\n" + body;
    const uint8_t * p = reinterpret_cast<const uint8_t *>(req.data());
    size_t left = req.size();
    while (left > 0) {
        const ssize_t n = ::send(fd, p, left, MSG_NOSIGNAL);
        if (n <= 0) break;
        p += n; left -= size_t(n);
    }
    char buf[8192];
    std::string raw;
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        raw.append(buf, size_t(n));
    }
    ::close(fd);
    const size_t sep = raw.find("\r\n\r\n");
    if (sep == std::string::npos) return resp;
    const size_t sp = raw.find(' ', 5);
    resp.status = sp == std::string::npos ? 0 : std::atoi(raw.c_str() + sp + 1);
    resp.body = raw.substr(sep + 4);
    return resp;
}

// GET /v1/models until 200 (model load + boot handshake happen before listen)
static bool wait_http_ready(int port, int timeout_s) {
    const int64_t t0 = ggml_time_ms();
    for (;;) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            sockaddr_in addr {};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(uint16_t(port));
            addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
                const char * get = "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
                (void) ::send(fd, get, strlen(get), MSG_NOSIGNAL);
                char buf[512];
                std::string raw;
                for (;;) {
                    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
                    if (n <= 0) break;
                    raw.append(buf, size_t(n));
                    if (raw.find("\r\n\r\n") != std::string::npos && raw.size() > 100) break;
                }
                ::close(fd);
                // "HTTP/1.1 200 ..." - status starts at offset 9
                if (raw.size() > 12 && raw.compare(9, 3, "200") == 0) return true;
            } else {
                ::close(fd);
            }
        }
        if (ggml_time_ms() - t0 > timeout_s * 1000) return false;
        usleep(500 * 1000);
    }
}

static int pick_free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in a {};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) < 0) { ::close(fd); return -1; }
    socklen_t len = sizeof(a);
    if (::getsockname(fd, reinterpret_cast<sockaddr *>(&a), &len) < 0) { ::close(fd); return -1; }
    const int port = ntohs(a.sin_port);
    ::close(fd);
    return port;
}

// a forked llama-server child + the carrier end of its kv channel
struct server_child {
    pid_t pid = -1;
    int kv_fd = -1; // carrier end of the socketpair (child end = GWP_KV_FD)
    int port = 0;

    // restore_blob = nullptr: fresh start (SKIP); otherwise sent as RESTORE
    bool start(const std::string & bin, const std::vector<std::string> & args, int port_,
               const std::vector<uint8_t> * restore_blob) {
        port = port_;
        int sv[2] = { -1, -1 };
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { perror("socketpair"); return false; }
        pid = fork();
        if (pid < 0) { perror("fork"); close(sv[0]); close(sv[1]); return false; }
        if (pid == 0) {
            // child: hand the socketpair end to the server as --kv-fd
            close(sv[0]);
            if (dup2(sv[1], GWP_KV_FD) != GWP_KV_FD) _exit(127);
            if (sv[1] != GWP_KV_FD) close(sv[1]);
            std::vector<std::string> storage;
            storage.reserve(args.size() + 12); // argv holds .data() pointers - no reallocation
            std::vector<char *> cargs;
            auto push = [&](const std::string & s) { storage.push_back(s); cargs.push_back(storage.back().data()); };
            push(bin);
            for (const auto & a : args) push(a);
            push("--host"); push("127.0.0.1");
            push("--port"); push(std::to_string(port_));
            push("--kv-fd"); push(std::to_string(GWP_KV_FD));
            cargs.push_back(nullptr);
            execv(bin.c_str(), cargs.data());
            fprintf(stderr, "carrier: execv(%s) failed: %s\n", bin.c_str(), strerror(errno));
            _exit(127);
        }
        close(sv[1]);
        kv_fd = sv[0];

        // boot handshake: server sends READY after model load, before listening
        uint32_t op = 0;
        std::vector<uint8_t> pl;
        if (!kvsock::recv_frame(kv_fd, op, pl, 300 * 1000)) {
            fprintf(stderr, "carrier: no READY from server (timeout)\n");
            kill_hard();
            return false;
        }
        if (op != kvsock::KV_OP_READY) {
            fprintf(stderr, "carrier: expected READY, got op=%u\n", op);
            kill_hard();
            return false;
        }
        const bool ok = restore_blob && !restore_blob->empty()
            ? kvsock::send_frame(kv_fd, kvsock::KV_OP_RESTORE, restore_blob->data(), restore_blob->size())
            : kvsock::send_frame(kv_fd, kvsock::KV_OP_SKIP, nullptr, 0);
        printf("carrier: %s (%.2f MiB) -> server pid %d\n",
               restore_blob && !restore_blob->empty() ? "sent RESTORE" : "sent SKIP (fresh)",
               restore_blob ? restore_blob->size() / 1048576.0 : 0.0, (int) pid);
        if (!ok) { kill_hard(); return false; }
        return true;
    }

    // POST /v1/completions (greedy, ignore_eos: exactly max_tokens tokens)
    bool completion(const std::string & prompt, int max_tokens, std::string & out_text) {
        nlohmann::json req = {
            { "model", "test" },
            { "prompt", prompt },
            { "max_tokens", max_tokens },
            { "temperature", 0 },
            { "stream", false },
            { "ignore_eos", true },
        };
        const http_resp r = http_request("127.0.0.1", port, "/v1/completions", req.dump());
        if (r.status != 200) {
            fprintf(stderr, "carrier: completion HTTP %d: %.300s\n", r.status, r.body.c_str());
            return false;
        }
        try {
            auto j = nlohmann::json::parse(r.body);
            out_text = j["choices"][0]["text"].get<std::string>();
            return true;
        } catch (const std::exception & e) {
            fprintf(stderr, "carrier: completion JSON error: %s\n", e.what());
            return false;
        }
    }

    // request the full prompt-cache dump; the blob lands in the carrier's RAM
    bool kv_dump(std::vector<uint8_t> & blob) {
        if (!kvsock::send_frame(kv_fd, kvsock::KV_OP_DUMP, nullptr, 0)) return false;
        uint32_t op = 0;
        std::vector<uint8_t> rsp;
        if (!kvsock::recv_frame(kv_fd, op, rsp, 0) || op != kvsock::KV_OP_DUMP_RSP || rsp.size() < 4) return false;
        uint32_t ok = 0;
        std::memcpy(&ok, rsp.data(), 4);
        if (!ok) {
            fprintf(stderr, "carrier: server rejected DUMP\n");
            return false;
        }
        blob.assign(rsp.begin() + 4, rsp.end());
        return true;
    }

    void kill_hard() {
        if (pid > 0) {
            ::kill(pid, SIGKILL);
            int st;
            while (waitpid(pid, &st, 0) < 0 && errno == EINTR);
            printf("carrier: server pid %d SIGKILLed, reaped\n", (int) pid);
            pid = -1;
        }
        if (kv_fd >= 0) { ::shutdown(kv_fd, SHUT_RDWR); ::close(kv_fd); kv_fd = -1; }
    }

    // graceful "inform and exit" over the kv channel: the server acks, then
    // runs its normal SIGINT teardown (releases the GPU itself) and exits 0.
    // Returns true if the ack was received.
    bool request_exit() {
        if (kv_fd < 0) {
            return false;
        }
        if (!kvsock::send_frame(kv_fd, kvsock::KV_OP_SHUTDOWN, nullptr, 0)) {
            return false;
        }
        uint32_t op = 0;
        std::vector<uint8_t> rsp;
        if (kvsock::recv_frame(kv_fd, op, rsp, 10 * 1000) && op == kvsock::KV_OP_SHUTDOWN_RSP) {
            printf("carrier: server acked SHUTDOWN (it now exits gracefully on its own)\n");
            return true;
        }
        fprintf(stderr, "carrier: no SHUTDOWN ack from server\n");
        return false;
    }

    // wait for the (informed) server to exit; SIGKILL fallback if it hangs
    void stop_graceful(int timeout_s = 120) {
        request_exit();
        if (pid > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            for (;;) {
                int st = 0;
                const pid_t r = waitpid(pid, &st, WNOHANG);
                if (r == pid) {
                    printf("carrier: server pid %d exited (status %d)\n", (int) pid, WIFEXITED(st) ? WEXITSTATUS(st) : -1);
                    pid = -1;
                    break;
                }
                if (r < 0) {
                    break;
                }
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(timeout_s)) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (pid > 0) {
                fprintf(stderr, "carrier: server did not exit in %d s, SIGKILL fallback\n", timeout_s);
                kill_hard();
            }
        }
        if (kv_fd >= 0) { ::shutdown(kv_fd, SHUT_RDWR); ::close(kv_fd); kv_fd = -1; }
    }

    // graceful: SIGTERM, 10s, SIGKILL fallback
    void stop() {
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            for (int i = 0; i < 100; i++) {
                int st;
                const pid_t r = waitpid(pid, &st, WNOHANG);
                if (r == pid) { pid = -1; break; }
                if (r < 0) break;
                usleep(100 * 1000);
            }
            if (pid > 0) kill_hard();
        }
        if (kv_fd >= 0) { ::shutdown(kv_fd, SHUT_RDWR); ::close(kv_fd); kv_fd = -1; }
    }
};

} // namespace

static int run_server_mode(int argc, char ** argv) {
    std::string prompt = "Explain in one sentence what a GPU is. Then list three typical uses.";
    int n_gen = 16, n_resume = 8, port = 0;
    std::string server_bin;
    std::vector<std::string> srv_args; // everything the carrier doesn't consume
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--server")     { continue; }
        if (a == "--prompt")     { prompt = argv[++i]; continue; }
        if (a == "--n-gen")      { n_gen = atoi(argv[++i]); continue; }
        if (a == "--n-resume")   { n_resume = atoi(argv[++i]); continue; }
        if (a == "--port")       { port = atoi(argv[++i]); continue; }
        if (a == "--server-bin") { server_bin = argv[++i]; continue; }
        srv_args.push_back(a);
    }
    if (server_bin.empty()) {
        // default: llama-server next to this binary
        char exe[4096];
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        if (n > 0 && strstr(exe, "llama-gpu-worker")) {
            exe[n] = 0;
            server_bin = std::string(exe, strlen(exe) - strlen("llama-gpu-worker")) + "llama-server";
        }
    }
    if (server_bin.empty()) { fprintf(stderr, "carrier: no server binary (use --server-bin PATH)\n"); return 1; }
    if (srv_args.empty())   { fprintf(stderr, "carrier: no server args (e.g. -m MODEL -ngl 99 -c 4096)\n"); return 1; }
    if (port == 0) { port = pick_free_port(); if (port < 0) { perror("pick_free_port"); return 1; } }

    printf("carrier pid=%d: %s (args: %zu passed through) port=%d n_gen=%d n_resume=%d\n",
           (int) getpid(), server_bin.c_str(), srv_args.size(), port, n_gen, n_resume);

    bool ok = true;
    std::string control, s1, s2;
    std::vector<uint8_t> blob; // the KV blob lives in the carrier's RAM, nowhere else

    snapshot("PHASE 0: baseline (no server)");

    // ---- run 1: control (never sleeps)
    {
        server_child sc;
        ok = sc.start(server_bin, srv_args, port, nullptr);
        if (ok && !wait_http_ready(port, 300)) { fprintf(stderr, "carrier: control server not ready\n"); ok = false; }
        if (ok) ok = sc.completion(prompt, n_gen + n_resume, control);
        if (ok) printf("carrier: control run done (%d tokens)\n", n_gen + n_resume);
        sc.stop();
    }

    snapshot("PHASE 1: control server exited (expect: VRAM ~0)");

    // ---- run 2: sleep/wake
    {
        server_child sc;
        ok = ok && sc.start(server_bin, srv_args, port, nullptr);
        if (ok && !wait_http_ready(port, 300)) { fprintf(stderr, "carrier: sleep-run server not ready\n"); ok = false; }
        if (ok) ok = sc.completion(prompt, n_gen, s1);

        snapshot("PHASE 1b: server alive (expect: server pid listed, VRAM up)");

        if (ok) {
            // let the slot go idle and publish its state to the prompt cache
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            if (sc.kv_dump(blob)) printf("carrier: KV blob in RAM: %.2f MiB\n", blob.size() / 1048576.0);
            else ok = false;
        }

        // "inform and exit": the KV is already in our RAM, so ask the server
        // to shut down over the kv channel - it releases the GPU itself and
        // exits 0 (SIGKILL is only a fallback for a hung server)
        sc.stop_graceful();
        // VRAM release can still lag the process exit a moment;
        // give it time so the snapshot below shows the real state
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        snapshot("PHASE 2: server exited (expect: no server pid, VRAM ~0)");

        // wake: respawn + RESTORE, then continue from where the sleep left off
        server_child wc;
        ok = ok && wc.start(server_bin, srv_args, port, blob.empty() ? nullptr : &blob);
        if (ok && !wait_http_ready(port, 300)) { fprintf(stderr, "carrier: wake server not ready\n"); ok = false; }
        if (ok) ok = wc.completion(prompt + s1, n_resume, s2);

        snapshot("PHASE 3: server respawned + KV restored (expect: VRAM up)");

        wc.stop();
    }

    snapshot("PHASE 4: all servers exited (expect: VRAM ~0)");

    const std::string slept = s1 + s2;
    const bool same = slept == control;
    printf("\n================ SUMMARY ================\n");
    printf("control (%d tok) : %s\n", n_gen + n_resume, control.c_str());
    printf("slept   (%zu tok) : %s\n", slept.size(), slept.c_str());
    printf("KV blob         : %.2f MiB (carrier RAM only, no files)\n", blob.size() / 1048576.0);
    printf("continuity (slept == control): %s\n", (same && ok) ? "PASS" : "FAIL");
    return (ok && same) ? 0 : 1;
}

int main(int argc, char ** argv) {
    bool worker_mode = false, server_mode = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--worker") { worker_mode = true; break; }
        if (a == "--server") { server_mode = true; break; }
    }
    if (worker_mode) return run_worker(argc, argv);
    if (server_mode) return run_server_mode(argc, argv);
    return run_client(argc, argv);
}
