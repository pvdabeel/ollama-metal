// tp-ffn-ggml.cpp
//
// Step 2 of the TP exploration (see docs/tensor-parallel-exploration.md):
// a graph-level (Approach C) tensor-parallel FFN prototype built on the REAL
// ggml-metal backend + ggml_backend_sched, NOT a hand-rolled MPS mock.
//
// The point is to prove that Approach C is expressible in the existing
// framework: we build ONE ggml graph that contains N parallel sub-matmuls
// whose weights live on different dies, plus an add (= all-reduce). The
// scheduler then:
//   * places each sub-matmul on the die that holds its weight slice,
//   * inserts the cross-die copies for the broadcast (x -> all dies) and the
//     gather (partials -> the add), which go through our Phase C (host) /
//     Phase D (Infinity Fabric peer) ggml_metal_buffer_cpy_tensor path.
//
// FFN modelled (no activation; irrelevant to the TP mechanism + perf):
//   single: o = Wdown_full @ (Wup_full @ x)
//   TP    : o = sum_d  Wdown_s[d] @ (Wup_s[d] @ x)      (column/row split)
// The two are mathematically identical, so we also get a correctness check.
//
// We measure single-die vs N-die TP, with the cross-die copies forced through
// the host path (GGML_METAL_PEER_ENABLE unset) and the fabric path (set),
// swept over batch sizes (decode -> prefill).
//
// Build (links the patched dylibs in build-test/bin):
//   LL=~/Desktop/ollama-metal/work/llama.cpp
//   clang++ -std=c++17 -O2 tp-ffn-ggml.cpp -o tp-ffn-ggml \
//       -I$LL/ggml/include -L$LL/build-test/bin \
//       -lggml -lggml-base -lggml-metal \
//       -Wl,-rpath,$LL/build-test/bin -framework Foundation
//
// Run:
//   ./tp-ffn-ggml            # 2 dies, H=5120 I=13824
//   ./tp-ffn-ggml 5120 13824 4 40

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-metal.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}
static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char ** argv) {
    const int64_t H = argc > 1 ? atoll(argv[1]) : 5120;
    const int64_t I = argc > 2 ? atoll(argv[2]) : 13824;
    int           N = argc > 3 ? atoi(argv[3])  : 2;
    const int     reps = argc > 4 ? atoi(argv[4]) : 40;

    ggml_backend_reg_t reg = ggml_backend_metal_reg();
    int ndev = (int) ggml_backend_reg_dev_count(reg);
    if (ndev < 2) { fprintf(stderr, "need >=2 Metal devices, have %d\n", ndev); return 1; }
    if (N > ndev) N = ndev;
    const int64_t Is = I / N;

    std::vector<ggml_backend_t>             backend(N);
    std::vector<ggml_backend_buffer_type_t> buft(N);
    std::vector<ggml_backend_dev_t>         dev(N);
    for (int d = 0; d < N; d++) {
        dev[d]     = ggml_backend_reg_dev_get(reg, d);
        backend[d] = ggml_backend_dev_init(dev[d], NULL);
        buft[d]    = ggml_backend_dev_buffer_type(dev[d]);
    }

    // ggml_backend_sched requires a CPU backend as the last fallback entry.
    ggml_backend_t             cpu_backend = ggml_backend_cpu_init();
    std::vector<ggml_backend_t>             sched_backends = backend; sched_backends.push_back(cpu_backend);
    std::vector<ggml_backend_buffer_type_t> sched_bufts    = buft;    sched_bufts.push_back(ggml_backend_cpu_buffer_type());
    const int n_sched = N + 1;

    printf("# TP FFN ggml prototype (Approach C) — real ggml-metal + sched\n");
    printf("# dies=%d  H=%lld  I=%lld  I/die=%lld  reps=%d  dtype=f16\n",
           N, (long long)H, (long long)I, (long long)Is, reps);
    for (int d = 0; d < N; d++)
        printf("#   die %d: %s\n", d, ggml_backend_dev_description(dev[d]));

    // ---- host weights (f16) ----
    // Wup  {H, I}: data[out*H + in]   (in=H rows, out=I cols)
    // Wdown{I, H}: data[out*I + in]   (in=I rows, out=H cols)
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.04f, 0.04f);
    std::vector<ggml_fp16_t> hUp((size_t)H * I), hDown((size_t)I * H);
    for (auto & v : hUp)   v = ggml_fp32_to_fp16(dist(rng));
    for (auto & v : hDown) v = ggml_fp32_to_fp16(dist(rng));

    auto tensor_overhead_ctx = [](int n) {
        return ggml_tensor_overhead() * n + 4096;
    };

    // ---- weights on backends ----
    // single-die full weights on die0
    ggml_init_params wp0 { tensor_overhead_ctx(4), NULL, true };
    ggml_context * ctx_w0 = ggml_init(wp0);
    ggml_tensor * Wup0  = ggml_new_tensor_2d(ctx_w0, GGML_TYPE_F16, H, I);
    ggml_tensor * Wdn0  = ggml_new_tensor_2d(ctx_w0, GGML_TYPE_F16, I, H);
    ggml_set_name(Wup0, "Wup_full"); ggml_set_name(Wdn0, "Wdn_full");
    ggml_backend_buffer_t bw0 = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w0, buft[0]);
    ggml_backend_tensor_set(Wup0, hUp.data(),   0, ggml_nbytes(Wup0));
    ggml_backend_tensor_set(Wdn0, hDown.data(), 0, ggml_nbytes(Wdn0));

    // TP slices: Wup_s[d] {H, Is}, Wdn_s[d] {Is, H}, each on die d
    std::vector<ggml_context*>       ctx_wd(N);
    std::vector<ggml_backend_buffer_t> bwd(N);
    std::vector<ggml_tensor*>        Wup_s(N), Wdn_s(N);
    std::vector<ggml_fp16_t>         dnslice((size_t)Is * H);
    for (int d = 0; d < N; d++) {
        ggml_init_params wp { tensor_overhead_ctx(4), NULL, true };
        ctx_wd[d] = ggml_init(wp);
        Wup_s[d] = ggml_new_tensor_2d(ctx_wd[d], GGML_TYPE_F16, H, Is);
        Wdn_s[d] = ggml_new_tensor_2d(ctx_wd[d], GGML_TYPE_F16, Is, H);
        bwd[d]   = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wd[d], buft[d]);
        // Wup_s[d] = contiguous block of hUp (outer index = out): [d*Is*H, (d+1)*Is*H)
        ggml_backend_tensor_set(Wup_s[d], hUp.data() + (size_t)d * Is * H, 0, ggml_nbytes(Wup_s[d]));
        // Wdn_s[d] {Is,H}: slice[out*Is + in_l] = hDown[out*I + d*Is + in_l]  (strided)
        for (int64_t out = 0; out < H; out++)
            for (int64_t inl = 0; inl < Is; inl++)
                dnslice[out * Is + inl] = hDown[out * I + d * Is + inl];
        ggml_backend_tensor_set(Wdn_s[d], dnslice.data(), 0, ggml_nbytes(Wdn_s[d]));
    }

    std::vector<ggml_fp16_t> hx;             // input, regenerated per M

    // run one scenario for a given M; returns median ms, and (optionally) reads output
    auto run_scenario = [&](int64_t M, bool tp, std::vector<float> * out_f32) -> double {
        // NOTE: parallel=true (ggml's concurrent multi-backend execution) corrupts
        // an operand type in the metal bin-op path with >2 metal backends and
        // aborts (ggml-metal-ops.cpp op_bin F32 assert). parallel=false is robust
        // and still overlaps the dies (async Metal command buffers). Opt back in
        // with TP_PARALLEL for 2-die experiments.
        bool par = getenv("TP_PARALLEL") != NULL;
        ggml_backend_sched_t sched =
            ggml_backend_sched_new(sched_backends.data(), sched_bufts.data(), n_sched,
                                   GGML_DEFAULT_GRAPH_SIZE, /*parallel*/ par, /*op_offload*/ false);

        ggml_init_params gp { tensor_overhead_ctx(64) + ggml_graph_overhead(), NULL, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H, M);
        ggml_set_name(x, "x"); ggml_set_input(x);

        ggml_tensor * o;
        if (!tp) {
            ggml_tensor * a = ggml_mul_mat(ctx, Wup0, x);   // {I, M}
            o = ggml_mul_mat(ctx, Wdn0, a);                 // {H, M}
        } else {
            o = nullptr;
            for (int d = 0; d < N; d++) {
                ggml_tensor * a_d = ggml_mul_mat(ctx, Wup_s[d], x);   // {Is, M} on die d
                ggml_tensor * o_d = ggml_mul_mat(ctx, Wdn_s[d], a_d); // {H, M}  on die d
                o = o ? ggml_add(ctx, o, o_d) : o_d;                  // all-reduce
            }
        }
        ggml_set_name(o, "o");
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);

        if (getenv("TP_DEBUG") && tp) {
            int nn = ggml_graph_n_nodes(gf);
            fprintf(stderr, "## graph dump (tp, M=%lld): %d nodes\n", (long long)M, nn);
            for (int i = 0; i < nn; i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                const char * s0 = n->src[0] ? ggml_type_name(n->src[0]->type) : "-";
                const char * s1 = n->src[1] ? ggml_type_name(n->src[1]->type) : "-";
                fprintf(stderr, "  [%2d] %-10s %-8s  src0=%s(%s)  src1=%s(%s)\n",
                        i, ggml_op_name(n->op), ggml_type_name(n->type),
                        n->src[0] ? n->src[0]->name : "-", s0,
                        n->src[1] ? n->src[1]->name : "-", s1);
            }
        }

        if (!ggml_backend_sched_reserve(sched, gf)) {
            fprintf(stderr, "sched reserve failed\n");
        }
        ggml_backend_sched_alloc_graph(sched, gf);

        // set input x
        hx.resize((size_t)H * M);
        for (auto & v : hx) v = ggml_fp32_to_fp16(dist(rng));
        ggml_backend_tensor_set(x, hx.data(), 0, ggml_nbytes(x));

        auto sync_all = [&]() { for (int d = 0; d < N; d++) ggml_backend_synchronize(backend[d]); };

        // warmup
        for (int w = 0; w < 5; w++) { ggml_backend_sched_graph_compute(sched, gf); }
        sync_all();

        std::vector<double> t;
        for (int r = 0; r < reps; r++) {
            auto t0 = clk::now();
            ggml_backend_sched_graph_compute(sched, gf);
            sync_all();
            t.push_back(ms_since(t0));
        }

        if (out_f32) {
            out_f32->resize((size_t)H * M);
            std::vector<ggml_fp16_t> tmp((size_t)H * M);
            ggml_backend_tensor_get(o, tmp.data(), 0, ggml_nbytes(o));
            for (size_t i = 0; i < tmp.size(); i++) (*out_f32)[i] = ggml_fp16_to_fp32(tmp[i]);
        }

        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        return median(t);
    };

    // ---- correctness at M=256: single vs TP(peer) ----
    setenv("GGML_METAL_PEER_ENABLE", "1", 1);
    {
        std::vector<float> os, ot;
        run_scenario(256, false, &os);
        run_scenario(256, true,  &ot);
        double maxabs = 0, maxrel = 0;
        for (size_t i = 0; i < os.size(); i++) {
            double ad = std::fabs(os[i] - ot[i]);
            maxabs = std::max(maxabs, ad);
            double den = std::fabs(os[i]) + 1e-4;
            maxrel = std::max(maxrel, ad / den);
        }
        printf("\n# correctness (M=256): max|abs|=%.4g  max rel=%.4g  -> %s\n",
               maxabs, maxrel, (maxrel < 0.05 ? "OK" : "CHECK"));
    }

    int64_t Ms[] = {1, 8, 64, 256, 512, 1024, 2048};
    printf("\n%6s | %10s | %10s | %10s | %9s | %9s\n",
           "tokens", "single ms", "tp_host ms", "tp_peer ms", "spd_host", "spd_peer");
    printf("-------+------------+------------+------------+-----------+----------\n");
    for (int64_t M : Ms) {
        double s    = run_scenario(M, false, nullptr);
        unsetenv("GGML_METAL_PEER_ENABLE");
        double t_h  = run_scenario(M, true,  nullptr);
        setenv("GGML_METAL_PEER_ENABLE", "1", 1);
        double t_p  = run_scenario(M, true,  nullptr);
        printf("%6lld | %10.3f | %10.3f | %10.3f | %9.2f | %9.2f\n",
               (long long)M, s, t_h, t_p, s / t_h, s / t_p);
        fflush(stdout);
    }
    printf("\n# spd_* > 1.0 => %d-die TP beats a single die end-to-end (incl. sched copies).\n", N);

    for (int d = 0; d < N; d++) { ggml_backend_buffer_free(bwd[d]); ggml_free(ctx_wd[d]); ggml_backend_free(backend[d]); }
    ggml_backend_buffer_free(bw0); ggml_free(ctx_w0);
    ggml_backend_free(cpu_backend);
    return 0;
}
