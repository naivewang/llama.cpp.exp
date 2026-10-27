// llama-gpu-worker: Phase 2 PoC — "sleep = drop the GPU process", llama.cpp edition.
//
// One binary, two modes:
//
//   worker (child, owns the GPU):
//     llama-gpu-worker --worker --uds /tmp/x.sock -m MODEL -c 4096 -ngl 99 [...]
//     loads model+context (GPU per usual llama.cpp args), listens on a UNIX
//     domain socket and serves: INIT, DECODE, DUMP_KV, RESTORE_KV, SHUTDOWN, PING
//
//   client (parent, GPU-free service side):
//     llama-gpu-worker -m MODEL [--prompt "..."] [--n-gen 16] [--n-resume 8]
//     never initializes a GPU (CPU model load, tokenization only);
//     fork+execs the worker and validates the full Phase 2 lifecycle:
//       1. control run: prompt + (n_gen + n_resume) tokens, never sleeps
//       2. sleep run:   prompt + n_gen tokens -> DUMP_KV -> SIGKILL worker
//          -> respawn worker -> RESTORE_KV -> n_resume more tokens
//       3. greedy decoding is deterministic, so sleep-run output must equal
//          the control-run output token-for-token
//     Plus phased rocm-smi / nvidia-smi snapshots around each transition.

#include "gpu-worker.h"

#include "common.h"
#include "arg.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <signal.h>

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

int main(int argc, char ** argv) {
    bool worker_mode = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--worker") { worker_mode = true; break; }
    }
    return worker_mode ? run_worker(argc, argv) : run_client(argc, argv);
}
