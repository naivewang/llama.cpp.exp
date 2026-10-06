// ---------------------------------------------------------------------------
// ar_bench.cpp — standalone AllReduce latency microbenchmark.
//
// Measures the *isolated* per-AR wall-clock latency of the internal
// host-staged AllReduce (ggml/src/ggml-cuda/allreduce.cu) on 2 ROCm GPUs,
// across a range of payload sizes.  No model, no VRAM pressure: this is the
// cleanest way to see the effect of changes to allreduce.cu.
//
// It inits the 2 device backends + the internal AR pipeline, then drives
// ggml_cuda_ar_allreduce() back-to-back and times it.
//
// Build (from llama.cpp.exp):  see build command at the bottom of the README.
// ---------------------------------------------------------------------------
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ \
    printf("  HIP FAIL %s: %s (line %d)\n", #x, hipGetErrorString(e_), __LINE__);} } while(0)

// Internal AR pipeline API (defined in allreduce.cu, no public header).
// Global namespace, C++ linkage — declarations match the definitions.
struct ggml_cuda_ar_pipeline;
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices);
void                    ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p);
bool                    ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline * p,
                                               ggml_backend_t * backends,
                                               ggml_tensor ** tensors);

int main(int argc, char ** argv) {
    int base_iters = 200;
    if (argc > 1) base_iters = atoi(argv[1]);

    ggml_backend_reg_t reg = ggml_backend_cuda_reg();
    size_t n_dev = ggml_backend_reg_dev_count(reg);
    printf("ROCm devices: %zu\n", n_dev);
    if (n_dev < 2) { printf("need >= 2 devices\n"); return 1; }

    const int devs[2] = {0, 1};
    ggml_backend_t backends[2] = {};
    for (int i = 0; i < 2; ++i) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, devs[i]);
        backends[i] = ggml_backend_dev_init(dev, nullptr);
        if (!backends[i]) { printf("dev %d backend init failed\n", i); return 1; }
        size_t free_mib, total_mib;
        ggml_backend_dev_memory(ggml_backend_get_device(backends[i]), &free_mib, &total_mib);
        printf("  backend[%d] = %s  (%zu MiB free / %zu MiB total)\n",
               i, ggml_backend_name(backends[i]), free_mib/1024, total_mib/1024);
    }

    ggml_cuda_ar_pipeline * p = ggml_cuda_ar_pipeline_init(devs, 2);
    if (!p) { printf("AR pipeline init FAILED\n"); return 1; }
    printf("AR pipeline: OK (internal host-staged)\n\n");

    // Payload sizes (f32 elements).  Decode AR is ~2048 elems (8 KB f32,
    // 4 KB bf16 on the wire).
    const std::vector<int64_t> sizes = {
        512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 262144, 1048576, 4194304
    };

    printf("%-12s %-9s %-11s %-12s %-11s %s\n",
           "elems", "bytes(f32)", "wire(bf16)", "us/AR", "GB/s(wire)", "chk");
    for (int64_t ne : sizes) {
        int iters = ne > 65536 ? std::max(20, base_iters/4) : base_iters;

        struct ggml_init_params gp = { .mem_size = 64*1024*1024, .mem_buffer = nullptr, .no_alloc = true };
        struct ggml_context * ctx = ggml_init(gp);

        ggml_tensor * tensors[2] = {};
        ggml_backend_buffer_t bufs[2] = {};
        for (int i = 0; i < 2; ++i) {
            tensors[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
            bufs[i] = ggml_backend_alloc_buffer(backends[i], ne*sizeof(float));
            ggml_backend_tensor_alloc(bufs[i], tensors[i], ggml_backend_buffer_get_base(bufs[i]));
            tensors[i]->flags |= GGML_TENSOR_FLAG_COMPUTE;   // both shards contribute
            float * host = new float[ne];
            for (int64_t j = 0; j < ne; ++j) host[j] = (float)(i+1);  // dev0=1.0, dev1=2.0
            hipSetDevice(devs[i]);
            CK(hipMemcpy(tensors[i]->data, host, ne*sizeof(float), hipMemcpyHostToDevice));
            delete[] host;
        }
        hipDeviceSynchronize();

        // warmup
        for (int w = 0; w < 16; ++w) ggml_cuda_ar_allreduce(p, backends, tensors);
        for (int i = 0; i < 2; ++i) { hipSetDevice(devs[i]); CK(hipDeviceSynchronize()); }

        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) ggml_cuda_ar_allreduce(p, backends, tensors);
        for (int i = 0; i < 2; ++i) { hipSetDevice(devs[i]); CK(hipDeviceSynchronize()); }
        auto t1 = std::chrono::steady_clock::now();
        double us_ar = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()
                       / 1000.0 / iters;

        // correctness (one-shot): the AR is IN-PLACE, so the timed loop above
        // accumulated (doubled) the values.  Reset to 1.0/2.0, do ONE AR, check 3.0.
        for (int i = 0; i < 2; ++i) {
            float * host = new float[ne];
            for (int64_t j = 0; j < ne; ++j) host[j] = (float)(i+1);
            hipSetDevice(devs[i]);
            CK(hipMemcpy(tensors[i]->data, host, ne*sizeof(float), hipMemcpyHostToDevice));
            delete[] host;
        }
        hipDeviceSynchronize();
        ggml_cuda_ar_allreduce(p, backends, tensors);
        for (int i = 0; i < 2; ++i) { hipSetDevice(devs[i]); CK(hipDeviceSynchronize()); }
        float chk[4] = {0,0,0,0};
        hipSetDevice(0);
        CK(hipMemcpy(chk, tensors[0]->data, 4*sizeof(float), hipMemcpyDeviceToHost));
        bool ok = std::fabs(chk[0]-3.0) < 1e-3 && std::fabs(chk[3]-3.0) < 1e-3;

        double wire = (double)ne*2;                       // bf16 on the wire
        printf("%-12lld %-9.0f %-11.0f %-12.2f %-11.3f %s\n",
               (long long)ne, (double)ne*4, wire, us_ar, wire*2/(us_ar*1e-6)/1e9, ok?"ok":"BAD");

        for (int i = 0; i < 2; ++i) ggml_backend_buffer_free(bufs[i]);
        ggml_free(ctx);
    }

    ggml_cuda_ar_pipeline_free(p);
    printf("\ndone\n");
    return 0;
}
