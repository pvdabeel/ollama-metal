// tp-block-ggml.cpp
//
// Step 3 of the TP exploration (see docs/tensor-parallel-exploration.md):
// a FULL transformer block (RMSNorm + GQA attention + residual + RMSNorm +
// FFN + residual, with the two all-reduces a real TP layer needs) built on the
// real ggml-metal backend + ggml_backend_sched, at the actual qwen3.5moe dims.
//
// Step 2 measured an isolated FFN matmul pair. A real layer also has attention
// (head-parallel core + a 2nd all-reduce on the O projection), norms, residual
// adds and an activation -- parts that do NOT parallelise -- so the end-to-end
// per-layer speedup is necessarily lower than the per-FFN microbench. This
// prototype measures that realistic per-layer number before committing to the
// (large) llama.cpp loader+graph surgery.
//
// Tensor-parallel layout (Megatron style):
//   Wq  column-split by head        (out = n_head*head_dim)
//   Wk,Wv replicated on every die   (tiny: n_head_kv*head_dim)
//   attention core: each die runs its head-group (no cross-die comm)
//   Wo  row-split by head           -> ALL-REDUCE #1
//   Wgate,Wup column-split          (FFN intermediate; MoE active-compute proxy)
//   Wdown row-split                 -> ALL-REDUCE #2
// Single-die baseline computes the whole block on die0. The two are
// mathematically identical -> doubles as a correctness check.
//
// Build:
//   LL=~/Desktop/ollama-metal/work/llama.cpp
//   clang++ -std=c++17 -O2 tp-block-ggml.cpp -o tp-block-ggml \
//       -I$LL/ggml/include -L$LL/build-test/bin \
//       -lggml -lggml-base -lggml-metal -lggml-cpu \
//       -Wl,-rpath,$LL/build-test/bin -framework Foundation
//
// Run:
//   ./tp-block-ggml          # 2 dies, qwen3.5moe dims
//   ./tp-block-ggml 4 40     # 4 dies, 40 reps

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

// hparams: default qwen3.5moe; override via env (TP_E/TP_NH/TP_NKV/TP_HD/TP_F)
// e.g. gemma4:31b -> TP_E=5376 TP_NH=32 TP_NKV=16 TP_HD=512 TP_F=21504
static int64_t E   = 3072;   // n_embd
static int64_t NH  = 32;     // n_head
static int64_t NKV = 2;      // n_head_kv (GQA)
static int64_t HD  = 256;    // head_dim (key/value length)
static int64_t HQ  = NH * HD;
static int64_t HKV = NKV * HD;
static int64_t F   = 8192;   // FFN intermediate (dense FFN for a dense model)

// weight storage type: F16 by default; TP_QUANT=q4k uses Q4_K (the deployed quant).
// Q4_K blocks are 256 elems along ne0 (the contraction dim); all splits used here
// keep ne0 (and IN/N for row-splits) divisible by 256, so per-split quantization is
// block-aligned and bit-identical to quantizing the full matrix.
static ggml_type WT = GGML_TYPE_F16;

static std::mt19937 rng(7);
static std::uniform_real_distribution<float> dist(-0.03f, 0.03f);

struct Weights {
    // host F32 weight masters (row-major, ggml layout data[i1*ne0 + i0], tensor {ne0=in, ne1=out});
    // converted to F16 or quantized to WT on upload.
    std::vector<float> Wq, Wk, Wv, Wo, Wg, Wu, Wd;
    // norm weights are F32 (activations/residual stream are F32, like real llama.cpp)
    std::vector<float> nrm1, nrm2;
};

static std::vector<float> randf32(size_t n) {
    std::vector<float> v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}

// allocate a tensor in ctx and set from host slice (column-split: contiguous out-block;
// row-split: strided over in). full is {IN, OUT}; we pull a sub-block.
enum SplitKind { FULL, COL, ROW };

int main(int argc, char ** argv) {
    int   N    = argc > 1 ? atoi(argv[1]) : 2;
    int   reps = argc > 2 ? atoi(argv[2]) : 40;

    auto geti = [](const char * k, int64_t d){ const char * v = getenv(k); return v ? (int64_t)atoll(v) : d; };
    E  = geti("TP_E", 3072);  NH = geti("TP_NH", 32); NKV = geti("TP_NKV", 2);
    HD = geti("TP_HD", 256);  F  = geti("TP_F", 8192);
    HQ = NH * HD;  HKV = NKV * HD;
    { const char * q = getenv("TP_QUANT"); if (q && (!strcmp(q,"q4k")||!strcmp(q,"1"))) WT = GGML_TYPE_Q4_K; }

    ggml_backend_reg_t reg = ggml_backend_metal_reg();
    int ndev = (int) ggml_backend_reg_dev_count(reg);
    if (ndev < 2) { fprintf(stderr, "need >=2 Metal devices\n"); return 1; }
    if (N > ndev) N = ndev;

    std::vector<ggml_backend_t>             backend(N);
    std::vector<ggml_backend_buffer_type_t> buft(N);
    std::vector<ggml_backend_dev_t>         dev(N);
    for (int d = 0; d < N; d++) {
        dev[d]     = ggml_backend_reg_dev_get(reg, d);
        backend[d] = ggml_backend_dev_init(dev[d], NULL);
        buft[d]    = ggml_backend_dev_buffer_type(dev[d]);
    }
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    std::vector<ggml_backend_t>             sb = backend; sb.push_back(cpu_backend);
    std::vector<ggml_backend_buffer_type_t> sbt = buft;   sbt.push_back(ggml_backend_cpu_buffer_type());
    const int n_sched = N + 1;
    const bool par = getenv("TP_PARALLEL") != NULL;  // parallel=true is buggy with >2 metal backends

    const int64_t nh_local = NH / N;   // query heads per die
    // GQA KV placement: split KV heads across dies when divisible (Megatron-correct),
    // otherwise replicate them on every die. Either way nh_local/nkv_local must be a
    // positive integer (the GQA broadcast factor in ggml_mul_mat).
    const bool split_kv = (NKV % N == 0);
    const int64_t nkv_local = split_kv ? (NKV / N) : NKV;
    const SplitKind KV_KIND = split_kv ? COL : FULL;
    const int64_t HKV_l = nkv_local * HD;
    (void)HKV_l;

    printf("# TP full-block prototype — real ggml-metal + sched\n");
    printf("# dies=%d  E=%lld nH=%lld nKV=%lld hd=%lld HQ=%lld F=%lld  kv=%s(%lld/die)  wt=%s  reps=%d\n",
           N, (long long)E, (long long)NH, (long long)NKV, (long long)HD, (long long)HQ, (long long)F,
           split_kv?"split":"replicated", (long long)nkv_local, ggml_type_name(WT), reps);
    for (int d = 0; d < N; d++) printf("#   die %d: %s\n", d, ggml_backend_dev_description(dev[d]));

    // ---- host master weights (F32) ----
    Weights W;
    W.Wq = randf32((size_t)E*HQ);  W.Wk = randf32((size_t)E*HKV); W.Wv = randf32((size_t)E*HKV);
    W.Wo = randf32((size_t)HQ*E);  W.Wg = randf32((size_t)E*F);   W.Wu = randf32((size_t)E*F);
    W.Wd = randf32((size_t)F*E);   W.nrm1 = randf32(E);           W.nrm2 = randf32(E);

    auto ovh = [](int n){ return ggml_tensor_overhead()*n + 8192; };

    // helper: create a weight tensor on a backend and fill from a master with a split.
    // master is {IN, OUT}; data[out*IN + in]. We produce a tensor {ne0, ne1}:
    //   FULL: {IN, OUT}
    //   COL : {IN, OUT/N}  slice over OUT (contiguous block d*OUT/N..)
    //   ROW : {IN/N, OUT}  slice over IN  (strided gather)
    auto mk = [&](ggml_context * ctx, ggml_backend_buffer_type_t bt_unused,
                  const std::vector<float> & master, int64_t IN, int64_t OUT,
                  SplitKind k, int d, const char * name) -> ggml_tensor* {
        int64_t ne0, ne1;
        if (k == FULL) { ne0 = IN;   ne1 = OUT; }
        else if (k == COL) { ne0 = IN; ne1 = OUT / N; }
        else { ne0 = IN / N; ne1 = OUT; }
        ggml_tensor * t = ggml_new_tensor_2d(ctx, WT, ne0, ne1);
        ggml_set_name(t, name);
        return t; // data set later (after buffer alloc)
    };
    // fill helper: assemble the F32 split segment {ne0,ne1}, then convert to the
    // tensor's storage type (F16 cast or Q4_K quantize) and upload.
    auto upload = [&](ggml_tensor * t, const std::vector<float> & seg) {
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(seg.size());
            for (size_t i = 0; i < seg.size(); i++) h[i] = ggml_fp32_to_fp16(seg[i]);
            ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<char> q(ggml_nbytes(t));
            ggml_quantize_chunk(t->type, seg.data(), q.data(), 0, t->ne[1], t->ne[0], nullptr);
            ggml_backend_tensor_set(t, q.data(), 0, ggml_nbytes(t));
        }
    };
    auto fill = [&](ggml_tensor * t, const std::vector<float> & master,
                    int64_t IN, int64_t OUT, SplitKind k, int d) {
        std::vector<float> seg;
        if (k == FULL) {
            seg.assign(master.begin(), master.begin() + (size_t)IN*OUT);
        } else if (k == COL) {
            int64_t outl = OUT / N;            // contiguous out-block [d*outl, (d+1)*outl)
            seg.assign(master.begin() + (size_t)d*outl*IN, master.begin() + (size_t)(d+1)*outl*IN);
        } else {                               // ROW: ne0=IN/N, strided gather over IN
            int64_t INl = IN / N;
            seg.resize((size_t)INl*OUT);
            for (int64_t out = 0; out < OUT; out++)
                for (int64_t inl = 0; inl < INl; inl++)
                    seg[out*INl + inl] = master[out*IN + d*INl + inl];
        }
        upload(t, seg);
    };

    // ---- single-die weights (all FULL on die0) ----
    ggml_context * cw0 = ggml_init({ovh(9), NULL, true});
    ggml_tensor *Sq=mk(cw0,buft[0],W.Wq,E,HQ,FULL,0,"Sq"), *Sk=mk(cw0,buft[0],W.Wk,E,HKV,FULL,0,"Sk"),
                *Sv=mk(cw0,buft[0],W.Wv,E,HKV,FULL,0,"Sv"), *So=mk(cw0,buft[0],W.Wo,HQ,E,FULL,0,"So"),
                *Sg=mk(cw0,buft[0],W.Wg,E,F,FULL,0,"Sg"),  *Su=mk(cw0,buft[0],W.Wu,E,F,FULL,0,"Su"),
                *Sd=mk(cw0,buft[0],W.Wd,F,E,FULL,0,"Sd"),
                *Sn1=ggml_new_tensor_1d(cw0,GGML_TYPE_F32,E), *Sn2=ggml_new_tensor_1d(cw0,GGML_TYPE_F32,E);
    ggml_set_name(Sn1,"Sn1"); ggml_set_name(Sn2,"Sn2");
    ggml_backend_buffer_t bw0 = ggml_backend_alloc_ctx_tensors_from_buft(cw0, buft[0]);
    fill(Sq,W.Wq,E,HQ,FULL,0); fill(Sk,W.Wk,E,HKV,FULL,0); fill(Sv,W.Wv,E,HKV,FULL,0);
    fill(So,W.Wo,HQ,E,FULL,0); fill(Sg,W.Wg,E,F,FULL,0); fill(Su,W.Wu,E,F,FULL,0); fill(Sd,W.Wd,F,E,FULL,0);
    ggml_backend_tensor_set(Sn1,W.nrm1.data(),0,ggml_nbytes(Sn1));
    ggml_backend_tensor_set(Sn2,W.nrm2.data(),0,ggml_nbytes(Sn2));

    // ---- TP weights (per die) ----
    std::vector<ggml_context*> cwd(N);
    std::vector<ggml_backend_buffer_t> bwd(N);
    std::vector<ggml_tensor*> Tq(N),Tk(N),Tv(N),To(N),Tg(N),Tu(N),Td(N),Tn1(N),Tn2(N);
    for (int d=0; d<N; d++) {
        cwd[d] = ggml_init({ovh(9), NULL, true});
        Tq[d]=mk(cwd[d],buft[d],W.Wq,E,HQ,COL,d,"Tq");      // column-split Q heads
        Tk[d]=mk(cwd[d],buft[d],W.Wk,E,HKV,KV_KIND,d,"Tk"); // split or replicated KV
        Tv[d]=mk(cwd[d],buft[d],W.Wv,E,HKV,KV_KIND,d,"Tv");
        To[d]=mk(cwd[d],buft[d],W.Wo,HQ,E,ROW,d,"To");      // row-split heads
        Tg[d]=mk(cwd[d],buft[d],W.Wg,E,F,COL,d,"Tg");
        Tu[d]=mk(cwd[d],buft[d],W.Wu,E,F,COL,d,"Tu");
        Td[d]=mk(cwd[d],buft[d],W.Wd,F,E,ROW,d,"Td");
        Tn1[d]=ggml_new_tensor_1d(cwd[d],GGML_TYPE_F32,E); Tn2[d]=ggml_new_tensor_1d(cwd[d],GGML_TYPE_F32,E);
        ggml_set_name(Tn1[d],"Tn1"); ggml_set_name(Tn2[d],"Tn2");
        bwd[d] = ggml_backend_alloc_ctx_tensors_from_buft(cwd[d], buft[d]);
        fill(Tq[d],W.Wq,E,HQ,COL,d); fill(Tk[d],W.Wk,E,HKV,KV_KIND,d); fill(Tv[d],W.Wv,E,HKV,KV_KIND,d);
        fill(To[d],W.Wo,HQ,E,ROW,d); fill(Tg[d],W.Wg,E,F,COL,d); fill(Tu[d],W.Wu,E,F,COL,d);
        fill(Td[d],W.Wd,F,E,ROW,d);
        ggml_backend_tensor_set(Tn1[d],W.nrm1.data(),0,ggml_nbytes(Tn1[d]));
        ggml_backend_tensor_set(Tn2[d],W.nrm2.data(),0,ggml_nbytes(Tn2[d]));
    }

    const float eps = 1e-6f, scale = 1.0f/std::sqrt((float)HD);

    // attention core for one die: q_proj {nhl*HD, M}, k/v {HKV, M} -> attn out {nhl*HD, M}
    auto attn_core = [&](ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                         int64_t nhl, int64_t nkvl, int64_t M) -> ggml_tensor* {
        ggml_tensor * Q = ggml_reshape_3d(c, q, HD, nhl, M);      // {hd, nhl, M}
        Q = ggml_permute(c, Q, 0, 2, 1, 3);                       // {hd, M, nhl}
        ggml_tensor * K = ggml_reshape_3d(c, k, HD, nkvl, M);     // {hd, nkvl, M}
        K = ggml_permute(c, K, 0, 2, 1, 3);                       // {hd, M, nkvl}
        ggml_tensor * KQ = ggml_mul_mat(c, K, Q);                 // {M, M, nhl} (GQA broadcast)
        KQ = ggml_soft_max_ext(c, KQ, nullptr, scale, 0.0f);
        ggml_tensor * V = ggml_reshape_3d(c, v, HD, nkvl, M);     // {hd, nkvl, M}
        V = ggml_cont(c, ggml_permute(c, V, 1, 2, 0, 3));         // {M, hd, nkvl}
        ggml_tensor * KQV = ggml_mul_mat(c, V, KQ);               // {hd, M, nhl}
        KQV = ggml_permute(c, KQV, 0, 2, 1, 3);                   // {hd, nhl, M}
        return ggml_cont_2d(c, KQV, nhl*HD, M);                   // {nhl*HD, M}
    };

    std::vector<float> hx;

    auto run = [&](int64_t M, bool tp, std::vector<float> * out) -> double {
        ggml_backend_sched_t sched = ggml_backend_sched_new(sb.data(), sbt.data(), n_sched,
                                        GGML_DEFAULT_GRAPH_SIZE, par, false);
        ggml_context * c = ggml_init({ovh(4096) + ggml_graph_overhead_custom(4096, false), NULL, true});
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, E, M);
        ggml_set_name(x,"x"); ggml_set_input(x);

        ggml_tensor * o;
        if (!tp) {
            ggml_tensor * h  = ggml_mul(c, ggml_rms_norm(c, x, eps), Sn1);
            ggml_tensor * q  = ggml_mul_mat(c, Sq, h);
            ggml_tensor * k  = ggml_mul_mat(c, Sk, h);
            ggml_tensor * v  = ggml_mul_mat(c, Sv, h);
            ggml_tensor * a  = attn_core(c, q, k, v, NH, NKV, M);
            ggml_tensor * ao = ggml_mul_mat(c, So, a);            // {E, M}
            ggml_tensor * x2 = ggml_add(c, x, ao);
            ggml_tensor * h2 = ggml_mul(c, ggml_rms_norm(c, x2, eps), Sn2);
            ggml_tensor * g  = ggml_silu(c, ggml_mul_mat(c, Sg, h2));
            ggml_tensor * u  = ggml_mul_mat(c, Su, h2);
            ggml_tensor * dn = ggml_mul_mat(c, Sd, ggml_mul(c, g, u)); // {E, M}
            o = ggml_add(c, x2, dn);
        } else {
            ggml_tensor * h  = ggml_mul(c, ggml_rms_norm(c, x, eps), Tn1[0]);
            // attention: per die, all-reduce on O
            ggml_tensor * ao = nullptr;
            for (int d=0; d<N; d++) {
                ggml_tensor * q = ggml_mul_mat(c, Tq[d], h);
                ggml_tensor * k = ggml_mul_mat(c, Tk[d], h);
                ggml_tensor * v = ggml_mul_mat(c, Tv[d], h);
                ggml_tensor * a = attn_core(c, q, k, v, nh_local, nkv_local, M);
                ggml_tensor * od = ggml_mul_mat(c, To[d], a);     // {E, M} partial
                ao = ao ? ggml_add(c, ao, od) : od;
            }
            ggml_tensor * x2 = ggml_add(c, x, ao);
            ggml_tensor * h2 = ggml_mul(c, ggml_rms_norm(c, x2, eps), Tn2[0]);
            ggml_tensor * dnsum = nullptr;
            for (int d=0; d<N; d++) {
                ggml_tensor * g = ggml_silu(c, ggml_mul_mat(c, Tg[d], h2));
                ggml_tensor * u = ggml_mul_mat(c, Tu[d], h2);
                ggml_tensor * dd = ggml_mul_mat(c, Td[d], ggml_mul(c, g, u)); // {E, M} partial
                dnsum = dnsum ? ggml_add(c, dnsum, dd) : dd;
            }
            o = ggml_add(c, x2, dnsum);
        }
        ggml_set_name(o, "o");
        ggml_cgraph * gf = ggml_new_graph_custom(c, 4096, false);
        ggml_build_forward_expand(gf, o);

        ggml_backend_sched_reserve(sched, gf);
        ggml_backend_sched_alloc_graph(sched, gf);
        hx.resize((size_t)E*M); for (auto & z : hx) z = dist(rng);
        ggml_backend_tensor_set(x, hx.data(), 0, ggml_nbytes(x));

        auto sync = [&]{ for (int d=0; d<N; d++) ggml_backend_synchronize(backend[d]); };
        for (int w=0; w<5; w++) ggml_backend_sched_graph_compute(sched, gf);
        sync();
        std::vector<double> t;
        for (int r=0; r<reps; r++) { auto t0=clk::now(); ggml_backend_sched_graph_compute(sched, gf); sync(); t.push_back(ms_since(t0)); }
        if (out) { out->resize((size_t)E*M);
                   ggml_backend_tensor_get(o, out->data(), 0, ggml_nbytes(o)); }
        ggml_free(c); ggml_backend_sched_free(sched);
        return median(t);
    };

    setenv("GGML_METAL_PEER_ENABLE","1",1);
    { std::vector<float> os, ot; run(128,false,&os); run(128,true,&ot);
      double mab=0,mr=0; for (size_t i=0;i<os.size();i++){double a=std::fabs(os[i]-ot[i]); mab=std::max(mab,a); mr=std::max(mr,a/(std::fabs(os[i])+1e-3));}
      double thr = (WT==GGML_TYPE_F16)?0.08:0.12;  // Q4_K: only F32 all-reduce order differs (weights bit-identical)
      printf("\n# correctness (M=128): max|abs|=%.4g max rel=%.4g -> %s\n", mab, mr, mr<thr?"OK":"CHECK"); }

    int64_t Ms[] = {1, 8, 64, 256, 512, 1024, 2048};
    int nM = (int)(sizeof(Ms)/sizeof(Ms[0]));
    if (getenv("TP_MAXM")) { int mm=atoi(getenv("TP_MAXM")); while(nM>0 && Ms[nM-1]>mm) nM--; }
    printf("\n%6s | %10s | %10s | %10s | %9s | %9s\n","tokens","single ms","tp_host ms","tp_peer ms","spd_host","spd_peer");
    printf("-------+------------+------------+------------+-----------+----------\n");
    for (int mi=0; mi<nM; mi++) {
        int64_t M = Ms[mi];
        double s = run(M,false,nullptr);
        unsetenv("GGML_METAL_PEER_ENABLE"); double th = run(M,true,nullptr);
        setenv("GGML_METAL_PEER_ENABLE","1",1); double tp = run(M,true,nullptr);
        printf("%6lld | %10.3f | %10.3f | %10.3f | %9.2f | %9.2f\n",(long long)M,s,th,tp,s/th,s/tp);
        fflush(stdout);
    }
    printf("\n# full layer incl. attention + 2 all-reduces + norms + residual.\n");
    printf("# spd_* > 1.0 => %d-die TP beats a single die for a whole layer.\n", N);

    for (int d=0; d<N; d++){ ggml_backend_buffer_free(bwd[d]); ggml_free(cwd[d]); ggml_backend_free(backend[d]); }
    ggml_backend_buffer_free(bw0); ggml_free(cw0); ggml_backend_free(cpu_backend);
    return 0;
}
