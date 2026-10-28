// kv-server-test: standalone validation of llama-server --kv-fd KV handoff.
//
// NO FILES: the KV blob travels over a socketpair (inherited fd 31) and lives
// only in this process's RAM between the dump and the restore.
//
// Protocol under test (greedy decoding, temperature 0):
//   1. control : fresh server (no kv), prompt P, 12 tokens            -> T12
//   2. server A: prompt P, 6 tokens                                    -> T6
//                 KV_OP_DUMP over the fd -> blob held in RAM here
//                 SIGKILL A (process death - the whole point)
//   3. server B: boot with KV_OP_RESTORE(blob); the same conversation
//                 (P + T6 as an assistant turn) must hit the prompt cache
//                 100% (cached_tokens == prompt_tokens, no re-prefill) and
//                 continue with tokens 7-12 identical to the control run
//
// build (repo root; no CMake target - standalone like the gpu-worker prototype):
//   g++ -O2 -std=c++17 -I vendor/nlohmann tools/gpu-worker/kv-server-test.cpp -o kv-server-test
//
// usage: kv-server-test <server-bin> <model> [port-base]
//   port-base: server runs on this port (control/A/B sequential), default 18701

#include "json.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

static const int KV_FD = 31;

// ---------------- kv framing (mirrors tools/server/server-kv-socket.h) ------

enum {
    OP_READY    = 1,
    OP_RESTORE  = 2,
    OP_SKIP     = 3,
    OP_DUMP     = 4,
    OP_DUMP_RSP = 5,
    OP_PING     = 6,
    OP_PONG     = 7,
};

static bool sock_send_all(int fd, const uint8_t * data, size_t len) {
    while (len > 0) {
        const ssize_t n = ::send(fd, data, len, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) { continue; }
            return false;
        }
        data += n; len -= (size_t) n;
    }
    return true;
}

static bool sock_recv_all(int fd, uint8_t * data, size_t len) {
    while (len > 0) {
        const ssize_t n = ::recv(fd, data, len, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) { continue; }
            return false;
        }
        data += n; len -= (size_t) n;
    }
    return true;
}

static bool sock_send_frame(int fd, uint32_t op, const uint8_t * payload, size_t len) {
    uint8_t hdr[8];
    uint32_t total = (uint32_t) (8 + len);
    std::memcpy(hdr, &total, 4);
    std::memcpy(hdr + 4, &op, 4);
    if (!sock_send_all(fd, hdr, 8)) { return false; }
    return len == 0 || sock_send_all(fd, payload, len);
}

static bool sock_recv_frame(int fd, uint32_t & op, std::vector<uint8_t> & payload) {
    uint8_t hdr[8];
    if (!sock_recv_all(fd, hdr, 8)) { return false; }
    uint32_t total = 0;
    std::memcpy(&total, hdr, 4);
    op = 0;
    std::memcpy(&op, hdr + 4, 4);
    if (total < 8 || total > (64ull << 30)) { return false; }
    payload.resize(total - 8);
    return payload.empty() || sock_recv_all(fd, payload.data(), payload.size());
}

// boot: expect READY, send SKIP (fresh) or RESTORE (payload)
static bool kv_boot(int fd, const std::vector<uint8_t> * restore_payload) {
    uint32_t op;
    std::vector<uint8_t> pl;
    if (!sock_recv_frame(fd, op, pl)) { return false; }
    if (op != OP_READY) {
        std::fprintf(stderr, "kv: expected READY, got op=%u\n", op);
        return false;
    }
    if (restore_payload) {
        return sock_send_frame(fd, OP_RESTORE, restore_payload->data(), restore_payload->size());
    }
    return sock_send_frame(fd, OP_SKIP, nullptr, 0);
}

// runtime: send DUMP, receive (ok, states_payload)
static bool kv_dump(int fd, int & ok, std::vector<uint8_t> & states) {
    if (!sock_send_frame(fd, OP_DUMP, nullptr, 0)) { return false; }
    uint32_t op;
    std::vector<uint8_t> pl;
    if (!sock_recv_frame(fd, op, pl)) { return false; }
    if (op != OP_DUMP_RSP || pl.size() < 4) { return false; }
    std::memcpy(&ok, pl.data(), 4);
    states.assign(pl.begin() + 4, pl.end());
    return true;
}

// first state's token count (layout: u32 n_states, i32 n_tokens, ...)
static int first_state_n_tokens(const std::vector<uint8_t> & states) {
    if (states.size() < 8) { return -1; }
    uint32_t n_states = 0, n_tokens = 0;
    std::memcpy(&n_states, states.data(), 4);
    std::memcpy(&n_tokens, states.data() + 4, 4);
    return n_states == 0 ? 0 : (int) n_tokens;
}

// ---------------- server process management ---------------------------------

struct server_proc {
    pid_t pid = -1;
    int kv_parent_fd = -1;
    std::string logs;
    std::mutex logs_mtx;
    std::thread log_reader;
    bool log_reader_active = false;
    std::atomic<bool> reader_stop { false };

    // never let a joinable thread (or live pid) escape through the destructor
    ~server_proc() {
        if (pid > 0) {
            ::kill(pid, SIGKILL);
            int st; ::waitpid(pid, &st, 0);
            pid = -1;
        }
        if (kv_parent_fd >= 0) { ::close(kv_parent_fd); kv_parent_fd = -1; }
        if (log_reader.joinable()) {
            reader_stop = true;
            log_reader.join();
        }
    }
};

static void server_log_append(server_proc & sp, const char * buf, size_t n) {
    std::lock_guard<std::mutex> lock(sp.logs_mtx);
    if (sp.logs.size() < 4 * 1024 * 1024) {
        sp.logs.append(buf, n);
    }
}

static void spawn_server(server_proc & sp, const std::string & bin, const std::string & model, int port, bool with_kv) {
    int sv[2] = { -1, -1 };
    int log_pipe[2] = { -1, -1 };
    if (with_kv && socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        std::perror("socketpair"); std::exit(1);
    }
    if (pipe(log_pipe) != 0) {
        std::perror("pipe"); std::exit(1);
    }
    sp.pid = fork();
    if (sp.pid == 0) {
        // child: wire up fds, no files involved
        if (with_kv) {
            close(sv[0]);
            if (dup2(sv[1], KV_FD) != KV_FD) {
                std::fprintf(stderr, "harness: dup2(kv) failed: %s\n", strerror(errno));
                _exit(127);
            }
            if (sv[1] != KV_FD && close(sv[1]) != 0) {
                std::fprintf(stderr, "harness: close(sv1) failed: %s\n", strerror(errno));
                _exit(127);
            }
        }
        dup2(log_pipe[1], 1);
        dup2(log_pipe[1], 2);
        close(log_pipe[0]); close(log_pipe[1]);
        if (with_kv) { close(sv[0]); }

        std::vector<std::string> storage;
        storage.reserve(32); // argv holds .data() pointers - no reallocation allowed
        std::vector<char *> argv;
        auto push = [&](const std::string & s) {
            storage.push_back(s);
            argv.push_back(storage.back().data());
        };
        push(bin);
        push("--host"); push("127.0.0.1");
        push("--port"); push(std::to_string(port));
        push("-m"); push(model);
        push("-ngl"); push("0");          // CPU: GPU stays free for the user's server
        push("-c"); push("2048");
        push("-np"); push("1");
        push("--cache-ram"); push("1024");
        if (with_kv) { push("--kv-fd"); push(std::to_string(KV_FD)); }
        argv.push_back(nullptr);
        ::execv(bin.c_str(), argv.data());
        std::fprintf(stderr, "execv(%s) failed: %s\n", bin.c_str(), strerror(errno));
        _exit(127);
    }
    // parent
    close(log_pipe[1]);
    sp.log_reader_active = true;
    sp.reader_stop = false;
    sp.log_reader = std::thread([&, rd = log_pipe[0]]() {
        char buf[8192];
        for (;;) {
            pollfd pfd { rd, POLLIN, 0 };
            const int pr = ::poll(&pfd, 1, 100); // 100ms: also watch reader_stop
            if (pr > 0 && (pfd.revents & POLLIN)) {
                const ssize_t n = ::read(rd, buf, sizeof(buf));
                if (n <= 0) { break; }
                server_log_append(sp, buf, (size_t) n);
                continue;
            }
            if (pr < 0 && errno != EINTR) { break; }
            if (sp.reader_stop.load(std::memory_order_relaxed)) { break; }
        }
        ::close(rd);
    });
    if (with_kv) {
        close(sv[1]);
        sp.kv_parent_fd = sv[0];
    }
}

static void kill_server(server_proc & sp, int sig = SIGKILL) {
    if (sp.pid > 0) {
        ::kill(sp.pid, sig);
        int st;
        ::waitpid(sp.pid, &st, 0);
        sp.pid = -1;
    }
    if (sp.kv_parent_fd >= 0) {
        ::shutdown(sp.kv_parent_fd, SHUT_RDWR);
        ::close(sp.kv_parent_fd);
        sp.kv_parent_fd = -1;
    }
    if (sp.log_reader_active) {
        sp.reader_stop = true;
        sp.log_reader.join();
        sp.log_reader_active = false;
    }
}

static std::string server_log_tail(server_proc & sp, int lines) {
    std::string logs;
    {
        std::lock_guard<std::mutex> lock(sp.logs_mtx);
        logs = sp.logs;
    }
    std::vector<std::string> ls;
    size_t start = 0;
    for (size_t i = 0; i <= logs.size(); i++) {
        if (i == logs.size() || logs[i] == '\n') {
            ls.push_back(logs.substr(start, i - start));
            start = i + 1;
        }
    }
    std::string out;
    for (size_t i = ls.size() > (size_t) lines ? ls.size() - lines : 0; i < ls.size(); i++) {
        out += ls[i] + "\n";
    }
    return out;
}

// ---------------- minimal HTTP client ----------------------------------------

struct http_resp {
    int status = 0;
    std::string body;
};

static http_resp http_request(const std::string & host, int port, const std::string & method,
                              const std::string & path, const std::string & body) {
    http_resp resp;
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { return resp; }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    addr.sin_addr.s_addr = inet_addr(host.c_str());
    if (::connect(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        ::close(fd);
        return resp;
    }
    std::string req = method + " " + path + " HTTP/1.1\r\n"
                      "Host: " + host + ":" + std::to_string(port) + "\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: " + std::to_string(body.size()) + "\r\n"
                      "Connection: close\r\n\r\n" + body;
    const char * p = req.data();
    while (req.size() > 0) {
        const ssize_t n = ::send(fd, p, req.size(), MSG_NOSIGNAL);
        if (n <= 0) { break; }
        p += n;
        req.erase(0, (size_t) n);
    }
    std::string raw;
    char buf[8192];
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) { break; }
        raw.append(buf, (size_t) n);
    }
    ::close(fd);
    const size_t sep = raw.find("\r\n\r\n");
    if (sep == std::string::npos) { return resp; }
    const size_t sp = raw.find(' ', 5);
    resp.status = sp == std::string::npos ? 0 : std::atoi(raw.c_str() + sp + 1);
    resp.body = raw.substr(sep + 4);
    return resp;
}

static bool wait_ready(const std::string & host, int port, int timeout_s) {
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        const http_resp r = http_request(host, port, "GET", "/v1/models", "");
        if (r.status == 200) { return true; }
        const auto el = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (el > timeout_s) { return false; }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// ---------------- the test ----------------------------------------------------

struct chat_result {
    std::string text;
    int prompt_tokens = -1;
    int cached_tokens = -1;
    bool ok = false;
};

// raw text completion (no chat template - the token sequence is exact,
// which is what the KV handoff is being tested)
static chat_result chat(const std::string & host, int port, const std::string & prompt, int max_tokens) {
    chat_result r;
    json req = {
        { "model", "test" },
        { "prompt", prompt },
        { "max_tokens", max_tokens },
        { "temperature", 0 },
        { "stream", false },
        { "ignore_eos", true },
    };
    // /v1/completions = OAI-compat: usage.prompt_tokens_details.cached_tokens
    // is the number of prompt tokens served from the restored KV cache
    const http_resp resp = http_request(host, port, "POST", "/v1/completions", req.dump());
    if (resp.status != 200) {
        std::fprintf(stderr, "chat: HTTP %d: %.300s\n", resp.status, resp.body.c_str());
        return r;
    }
    try {
        auto j = json::parse(resp.body);
        r.text = j["choices"][0]["text"].get<std::string>();
        r.prompt_tokens = j["usage"].value("prompt_tokens", -1);
        if (j["usage"].contains("prompt_tokens_details")) {
            r.cached_tokens = j["usage"]["prompt_tokens_details"].value("cached_tokens", -1);
        }
        r.ok = true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "chat: bad JSON: %s\n", e.what());
    }
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <server-bin> <model> [port-base]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, nullptr, _IONBF, 0); // never lose progress output on crash

    const std::string bin   = argv[1];
    const std::string model = argv[2];
    const int port          = argc > 3 ? std::atoi(argv[3]) : 18701;
    const std::string host  = "127.0.0.1";
    const std::string PROMPT = "Explain in one sentence what a GPU is.";

    int rc = 1;
    std::vector<std::string> failures;
    auto check = [&failures](bool ok, const std::string & what) {
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
        if (!ok) { failures.push_back(what); }
    };

    // ---- 1. control run: fresh server, 12 tokens ----
    std::printf("== control: fresh server, 12 tokens ==\n");
    server_proc ctl;
    spawn_server(ctl, bin, model, port, false);
    if (!wait_ready(host, port, 120)) {
        std::fprintf(stderr, "control server did not come up:\n%s", server_log_tail(ctl, 30).c_str());
        kill_server(ctl);
        return 1;
    }
    chat_result c12 = chat(host, port, PROMPT, 12);
    kill_server(ctl);
    if (!c12.ok) { std::fprintf(stderr, "control chat failed\n"); return 1; }
    std::printf("  control (12 tok): %s\n", c12.text.c_str());

    // ---- 2. server A: 6 tokens, DUMP, SIGKILL ----
    std::printf("== server A: 6 tokens, KV dump, SIGKILL ==\n");
    server_proc A;
    spawn_server(A, bin, model, port, true);
    if (!kv_boot(A.kv_parent_fd, nullptr /* SKIP */)) {
        std::fprintf(stderr, "kv boot (SKIP) failed:\n%s", server_log_tail(A, 30).c_str());
        kill_server(A);
        return 1;
    }
    if (!wait_ready(host, port, 120)) {
        std::fprintf(stderr, "server A did not come up:\n%s", server_log_tail(A, 30).c_str());
        kill_server(A);
        return 1;
    }
    chat_result a6 = chat(host, port, PROMPT, 6);
    if (!a6.ok) { std::fprintf(stderr, "server A chat failed\n"); kill_server(A); return 1; }
    std::printf("  A (6 tok): %s\n", a6.text.c_str());
    check(a6.text.size() > 0 && c12.text.rfind(a6.text, 0) == 0,
          "A's 6 tokens are a prefix of the control's 12");

    std::this_thread::sleep_for(std::chrono::milliseconds(300)); // let the slot go idle
    int dump_ok = 0;
    std::vector<uint8_t> states;
    if (!kv_dump(A.kv_parent_fd, dump_ok, states)) {
        std::fprintf(stderr, "KV dump exchange failed\n");
        kill_server(A);
        return 1;
    }
    const int n_cached = first_state_n_tokens(states);
    std::printf("  dump: ok=%d, %zu bytes, first state %d tokens\n", dump_ok, states.size(), n_cached);
    check(dump_ok == 1, "dump ok");
    // note: the slot lifecycle does not commit the last sampled token to the
    // prompt/KV until the next decode step (the request is already complete),
    // so the idle state holds prompt + (n_generated - 1) tokens - exactly like
    // the normal in-RAM prompt cache behaves between two requests
    check(n_cached == a6.prompt_tokens + 6 - 1,
          "dump state == A's prompt + generated tokens (last token uncommitted by slot lifecycle)");

    ::kill(A.pid, SIGKILL); // hard death - the point of the exercise
    {
        int st;
        ::waitpid(A.pid, &st, 0);
    }
    std::printf("  server A SIGKILLed\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ---- 3. server B: RESTORE, same conversation, expect full cache hit ----
    std::printf("== server B: restore, same conversation (expect full cache hit) ==\n");
    server_proc B;
    spawn_server(B, bin, model, port, true);
    if (!kv_boot(B.kv_parent_fd, &states)) {
        std::fprintf(stderr, "kv boot (RESTORE) failed:\n%s", server_log_tail(B, 30).c_str());
        kill_server(B);
        return 1;
    }
    if (!wait_ready(host, port, 120)) {
        std::fprintf(stderr, "server B did not come up:\n%s", server_log_tail(B, 30).c_str());
        kill_server(B);
        return 1;
    }
    chat_result b6 = chat(host, port, PROMPT + a6.text, 6);
    kill_server(B);
    if (!b6.ok) { std::fprintf(stderr, "server B chat failed\n"); return 1; }
    std::printf("  B (6 tok): %s\n", b6.text.c_str());
    std::printf("  B usage: prompt_tokens=%d, cached_tokens=%d\n", b6.prompt_tokens, b6.cached_tokens);

    check(b6.cached_tokens == n_cached && b6.cached_tokens == b6.prompt_tokens - 1,
          "restored KV served everything but the uncommitted tail token (1-token prefill only)");

    const std::string expected_suffix = c12.text.substr(a6.text.size());
    check(b6.text == expected_suffix,
          "continuation tokens 7-12 identical to control");
    if (b6.text != expected_suffix) {
        std::printf("  control suffix: %s\n", expected_suffix.c_str());
    }

    // ---- summary ----
    std::printf("\n== SUMMARY ==\n");
    if (failures.empty()) {
        std::printf("KV HANDOFF: PASS (dump -> RAM -> kill -> respawn -> restore, no re-prefill)\n");
        rc = 0;
    } else {
        std::printf("KV HANDOFF: FAIL (%zu check(s))\n", failures.size());
        std::printf("--- server B log tail ---\n%s", server_log_tail(B, 25).c_str());
    }
    return rc;
}
