// tp-copy-ggml.cpp
//
// Direct cross-die copy head-to-head through the REAL ggml-metal copy path
// (Phase C host-bounce vs Phase D Infinity-Fabric peer blit).
//
// Unlike the scheduler-based prototypes (tp-*-ggml.cpp), which host-stage their
// cross-die copies and never reach the Phase-D code, this bench calls
// ggml_backend_tensor_copy(src_on_die_b, dst_on_die_0) directly. For two metal
// private buffers that lands in ggml_metal_buffer_cpy_tensor, which:
//   - GGML_METAL_PEER_ENABLE unset  -> returns false -> ggml_backend_tensor_copy
//     falls back to malloc + get + set (host bounce, Phase C);
//   - GGML_METAL_PEER_ENABLE set + same peer group -> newRemoteBufferViewForDevice
//     + blit over Infinity Fabric (Phase D).
// So the two env settings exercise exactly the two paths, end to end.
//
// Build:
//   LL=~/Desktop/ollama-metal/work/llama.cpp
//   clang++ -std=c++17 -O2 tp-copy-ggml.cpp -o tp-copy-ggml \
//       -I$LL/ggml/include -L$LL/build-test/bin \
//       -lggml -lggml-base -lggml-metal -Wl,-rpath,$LL/build-test/bin -framework Foundation
//
// Run:  ./tp-copy-ggml         # all src dies -> die 0

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using clk = std::chrono::high_resolution_clock;
static double us_since(clk::time_point t0){ return std::chrono::duration<double,std::micro>(clk::now()-t0).count(); }
static double median(std::vector<double> v){ if(v.empty())return 0; std::sort(v.begin(),v.end()); return v[v.size()/2]; }

int main(int argc, char ** argv) {
    ggml_backend_reg_t reg = ggml_backend_metal_reg();
    int ndev = (int) ggml_backend_reg_dev_count(reg);
    if (ndev < 2) { fprintf(stderr,"need >=2 Metal devices\n"); return 1; }

    std::vector<ggml_backend_t> backend(ndev);
    std::vector<ggml_backend_buffer_type_t> buft(ndev);
    std::vector<ggml_backend_dev_t> dev(ndev);
    for (int d=0; d<ndev; d++){ dev[d]=ggml_backend_reg_dev_get(reg,d); backend[d]=ggml_backend_dev_init(dev[d],NULL); buft[d]=ggml_backend_dev_buffer_type(dev[d]); }

    printf("# cross-die copy: ggml_backend_tensor_copy through real ggml_metal_buffer_cpy_tensor\n");
    printf("# dies=%d  (host = Phase C bounce, peer = Phase D Infinity Fabric)\n", ndev);
    for (int d=0; d<ndev; d++) printf("#   die %d: %s\n", d, ggml_backend_dev_description(dev[d]));

    // sizes in bytes
    size_t sizes[] = { 16*1024, 64*1024, 256*1024, 1u<<20, 4u<<20, 16u<<20, 64u<<20, 256u<<20 };
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-1,1);

    auto bench_pair = [&](int s, int t){  // copy die s -> die t
        printf("\n## die %d -> die %d\n", s, t);
        printf("%10s | %10s | %10s | %8s | %7s\n","bytes","host GB/s","peer GB/s","speedup","correct");
        printf("-----------+------------+------------+----------+--------\n");
        for (size_t bytes : sizes) {
            int64_t n = (int64_t)(bytes/sizeof(float));
            // src on die s, dst on die t
            ggml_context * cs = ggml_init({ ggml_tensor_overhead()+128, NULL, true });
            ggml_context * ct = ggml_init({ ggml_tensor_overhead()+128, NULL, true });
            ggml_tensor * src = ggml_new_tensor_1d(cs, GGML_TYPE_F32, n);
            ggml_tensor * dst = ggml_new_tensor_1d(ct, GGML_TYPE_F32, n);
            ggml_backend_buffer_t bs = ggml_backend_alloc_ctx_tensors_from_buft(cs, buft[s]);
            ggml_backend_buffer_t bt = ggml_backend_alloc_ctx_tensors_from_buft(ct, buft[t]);
            std::vector<float> h((size_t)n); for(auto&z:h) z=dist(rng);
            ggml_backend_tensor_set(src, h.data(), 0, bytes);

            int reps = bytes >= (16u<<20) ? 20 : 200;
            auto timed = [&]()->double{
                ggml_backend_tensor_copy(src, dst);  // warmup
                ggml_backend_synchronize(backend[t]);
                std::vector<double> tt;
                for (int r=0;r<reps;r++){ auto t0=clk::now(); ggml_backend_tensor_copy(src,dst); ggml_backend_synchronize(backend[t]); tt.push_back(us_since(t0)); }
                return median(tt); // microseconds
            };

            unsetenv("GGML_METAL_PEER_ENABLE");
            double us_host = timed();
            // correctness of host path
            std::vector<float> oh((size_t)n); ggml_backend_tensor_get(dst, oh.data(), 0, bytes);
            bool ok_host = memcmp(oh.data(), h.data(), bytes)==0;

            // wipe dst so the peer result must actually transfer
            { std::vector<float> zero((size_t)n,0.0f); ggml_backend_tensor_set(dst, zero.data(),0,bytes); }
            setenv("GGML_METAL_PEER_ENABLE","1",1);
            double us_peer = timed();
            std::vector<float> op((size_t)n); ggml_backend_tensor_get(dst, op.data(), 0, bytes);
            bool ok_peer = memcmp(op.data(), h.data(), bytes)==0;

            double gbh = bytes / (us_host*1e3);  // bytes / (us*1e-6) / 1e9
            double gbp = bytes / (us_peer*1e3);
            printf("%10zu | %10.2f | %10.2f | %8.2f | %s/%s\n", bytes, gbh, gbp, gbh>0?gbp/gbh:0,
                   ok_host?"ok":"BAD", ok_peer?"ok":"BAD");
            fflush(stdout);

            ggml_backend_buffer_free(bs); ggml_backend_buffer_free(bt);
            ggml_free(cs); ggml_free(ct);
        }
    };

    for (int s=1; s<ndev; s++) bench_pair(s, 0);

    printf("\n# peer GB/s >> host GB/s confirms the Phase-D fabric blit is engaged.\n");
    printf("# (intra-card vs cross-card pairs may differ if the fabric topology does.)\n");

    for (int d=0; d<ndev; d++) ggml_backend_free(backend[d]);
    return 0;
}
