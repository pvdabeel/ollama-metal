// tp-moe-ggml.cpp
//
// Expert-parallel (EP) feasibility prototype for the MoE FFN of qwen3.5moe,
// on the real ggml-metal backend + scheduler (follow-on to the TP exploration,
// see docs/tensor-parallel-exploration.md).
//
// MoE FFN (qwen3.5moe): 256 experts, 8 used/token, expert ff = 1024, n_embd=3072.
// In ggml this is MUL_MAT_ID: experts are [in, out, n_expert] tensors and a
// per-token id list [n_used, n_tokens] selects which experts run.
//
//   single (today's behaviour): all 256 experts on one die; mul_mat_id over the
//     8 selected experts per token, weighted-sum -> [n_embd, n_tokens].
//   EP: each die owns 256/N experts and computes n_used/N of every token's
//     selected experts (best-case perfect load balance), then the per-die
//     [n_embd, n_tokens] partials are all-reduced. One combine per MoE layer.
//
// Routing is constructed so each token's 8 experts split evenly into N groups
// (n_used/N per die, all distinct globally) => single == Σ_die partials EXACTLY,
// so this doubles as a correctness check while measuring EP's compute scaling.
//
// Build:
//   LL=~/Desktop/ollama-metal/work/llama.cpp
//   clang++ -std=c++17 -O2 tp-moe-ggml.cpp -o tp-moe-ggml \
//       -I$LL/ggml/include -L$LL/build-test/bin \
//       -lggml -lggml-base -lggml-metal -lggml-cpu \
//       -Wl,-rpath,$LL/build-test/bin -framework Foundation
//
// Run:  ./tp-moe-ggml 2 40   |   ./tp-moe-ggml 4 40

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
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }
static double median(std::vector<double> v){ if(v.empty())return 0; std::sort(v.begin(),v.end()); return v[v.size()/2]; }

// qwen3.5moe MoE dims
static const int64_t E      = 3072;  // n_embd
static const int64_t FF     = 1024;  // expert ff (intermediate)
static const int64_t NEXP   = 256;   // experts
static const int64_t NUSED  = 8;     // experts used per token

static std::mt19937 rng(11);
static std::uniform_real_distribution<float> dist(-0.03f, 0.03f);
static std::vector<ggml_fp16_t> randf16(size_t n){ std::vector<ggml_fp16_t> v(n); for(auto&x:v)x=ggml_fp32_to_fp16(dist(rng)); return v; }

int main(int argc, char ** argv) {
    int N    = argc > 1 ? atoi(argv[1]) : 2;
    int reps = argc > 2 ? atoi(argv[2]) : 40;

    ggml_backend_reg_t reg = ggml_backend_metal_reg();
    int ndev = (int) ggml_backend_reg_dev_count(reg);
    if (ndev < 2) { fprintf(stderr,"need >=2 Metal devices\n"); return 1; }
    if (N > ndev) N = ndev;
    if (NUSED % N != 0) { fprintf(stderr,"n_used(%lld) must be divisible by N(%d)\n",(long long)NUSED,N); return 1; }

    std::vector<ggml_backend_t> backend(N); std::vector<ggml_backend_buffer_type_t> buft(N); std::vector<ggml_backend_dev_t> dev(N);
    for (int d=0; d<N; d++){ dev[d]=ggml_backend_reg_dev_get(reg,d); backend[d]=ggml_backend_dev_init(dev[d],NULL); buft[d]=ggml_backend_dev_buffer_type(dev[d]); }
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    std::vector<ggml_backend_t> sb=backend; sb.push_back(cpu_backend);
    std::vector<ggml_backend_buffer_type_t> sbt=buft; sbt.push_back(ggml_backend_cpu_buffer_type());
    const int n_sched=N+1; const bool par = getenv("TP_PARALLEL")!=NULL;

    const int64_t EpD = NEXP / N;       // experts per die
    const int64_t nu_local = NUSED / N; // selected experts handled per die

    printf("# EP MoE prototype (qwen3.5moe dims) — real ggml-metal + sched\n");
    printf("# dies=%d  E=%lld FF=%lld n_expert=%lld n_used=%lld  exp/die=%lld used/die=%lld  reps=%d\n",
           N,(long long)E,(long long)FF,(long long)NEXP,(long long)NUSED,(long long)EpD,(long long)nu_local,reps);
    for (int d=0; d<N; d++) printf("#   die %d: %s\n", d, ggml_backend_dev_description(dev[d]));

    // host expert masters: gate/up [E, FF, NEXP]; down [FF, E, NEXP]  (data[e*in*out + o*in + i])
    auto Mg = randf16((size_t)E*FF*NEXP), Mu = randf16((size_t)E*FF*NEXP), Md = randf16((size_t)FF*E*NEXP);
    auto ovh=[](int n){ return ggml_tensor_overhead()*n + 8192; };

    // single experts (all on die0)
    ggml_context * cw0 = ggml_init({ovh(3),NULL,true});
    ggml_tensor *Sg=ggml_new_tensor_3d(cw0,GGML_TYPE_F16,E,FF,NEXP),
                *Su=ggml_new_tensor_3d(cw0,GGML_TYPE_F16,E,FF,NEXP),
                *Sd=ggml_new_tensor_3d(cw0,GGML_TYPE_F16,FF,E,NEXP);
    ggml_backend_buffer_t bw0=ggml_backend_alloc_ctx_tensors_from_buft(cw0,buft[0]);
    ggml_backend_tensor_set(Sg,Mg.data(),0,ggml_nbytes(Sg));
    ggml_backend_tensor_set(Su,Mu.data(),0,ggml_nbytes(Su));
    ggml_backend_tensor_set(Sd,Md.data(),0,ggml_nbytes(Sd));

    // EP experts (per die: slice of NEXP -> EpD; expert dim is ne2 => contiguous block)
    std::vector<ggml_context*> cwd(N); std::vector<ggml_backend_buffer_t> bwd(N);
    std::vector<ggml_tensor*> Tg(N),Tu(N),Td(N);
    for (int d=0; d<N; d++){
        cwd[d]=ggml_init({ovh(3),NULL,true});
        Tg[d]=ggml_new_tensor_3d(cwd[d],GGML_TYPE_F16,E,FF,EpD);
        Tu[d]=ggml_new_tensor_3d(cwd[d],GGML_TYPE_F16,E,FF,EpD);
        Td[d]=ggml_new_tensor_3d(cwd[d],GGML_TYPE_F16,FF,E,EpD);
        bwd[d]=ggml_backend_alloc_ctx_tensors_from_buft(cwd[d],buft[d]);
        ggml_backend_tensor_set(Tg[d],Mg.data()+(size_t)d*EpD*E*FF,0,ggml_nbytes(Tg[d]));
        ggml_backend_tensor_set(Tu[d],Mu.data()+(size_t)d*EpD*E*FF,0,ggml_nbytes(Tu[d]));
        ggml_backend_tensor_set(Td[d],Md.data()+(size_t)d*EpD*FF*E,0,ggml_nbytes(Td[d]));
    }

    // sum the n_expert_used dim (ne1) of a [E, nu, M] tensor -> [E, M]
    auto sum_used = [&](ggml_context*c, ggml_tensor*t, int64_t nu, int64_t M)->ggml_tensor*{
        ggml_tensor*acc=nullptr;
        for (int64_t j=0;j<nu;j++){
            ggml_tensor*s=ggml_cont(c, ggml_view_2d(c,t,E,M,t->nb[2], j*t->nb[1]));
            acc = acc ? ggml_add(c,acc,s) : s;
        }
        return acc;
    };

    std::vector<float> hcur, hw;     // input + weights
    std::vector<int32_t> hids;       // ids

    auto run=[&](int64_t M,bool ep,std::vector<float>*out)->double{
        ggml_backend_sched_t sched=ggml_backend_sched_new(sb.data(),sbt.data(),n_sched,GGML_DEFAULT_GRAPH_SIZE,par,false);
        ggml_context*c=ggml_init({ovh(4096)+ggml_graph_overhead_custom(8192,false),NULL,true});

        // shared input activation [E,1,M]
        ggml_tensor*cur=ggml_new_tensor_3d(c,GGML_TYPE_F32,E,1,M); ggml_set_name(cur,"cur"); ggml_set_input(cur);

        ggml_tensor*o;
        if(!ep){
            ggml_tensor*ids=ggml_new_tensor_2d(c,GGML_TYPE_I32,NUSED,M); ggml_set_name(ids,"ids"); ggml_set_input(ids);
            ggml_tensor*w  =ggml_new_tensor_3d(c,GGML_TYPE_F32,1,NUSED,M); ggml_set_name(w,"w"); ggml_set_input(w);
            ggml_tensor*up =ggml_mul_mat_id(c,Su,cur,ids);          // [FF, NUSED, M]
            ggml_tensor*gt =ggml_mul_mat_id(c,Sg,cur,ids);
            ggml_tensor*act=ggml_mul(c,ggml_silu(c,gt),up);          // [FF, NUSED, M]
            ggml_tensor*dn =ggml_mul_mat_id(c,Sd,act,ids);          // [E, NUSED, M]
            ggml_tensor*wd =ggml_mul(c,dn,w);                        // broadcast weights
            o = sum_used(c,wd,NUSED,M);
        } else {
            ggml_tensor*part=nullptr;
            for(int d=0; d<N; d++){
                ggml_tensor*ids=ggml_new_tensor_2d(c,GGML_TYPE_I32,nu_local,M);
                ggml_tensor*w  =ggml_new_tensor_3d(c,GGML_TYPE_F32,1,nu_local,M);
                char nm[32]; snprintf(nm,32,"ids%d",d); ggml_set_name(ids,nm); ggml_set_input(ids);
                snprintf(nm,32,"w%d",d); ggml_set_name(w,nm); ggml_set_input(w);
                ggml_tensor*up =ggml_mul_mat_id(c,Tu[d],cur,ids);   // runs on die d (weights there)
                ggml_tensor*gt =ggml_mul_mat_id(c,Tg[d],cur,ids);
                ggml_tensor*act=ggml_mul(c,ggml_silu(c,gt),up);
                ggml_tensor*dn =ggml_mul_mat_id(c,Td[d],act,ids);
                ggml_tensor*wd =ggml_mul(c,dn,w);
                ggml_tensor*pd =sum_used(c,wd,nu_local,M);          // [E, M]
                part = part ? ggml_add(c,part,pd) : pd;             // all-reduce
            }
            o = part;
        }
        ggml_set_name(o,"o");
        ggml_cgraph*gf=ggml_new_graph_custom(c,8192,false);
        ggml_build_forward_expand(gf,o);
        ggml_backend_sched_reserve(sched,gf);
        ggml_backend_sched_alloc_graph(sched,gf);

        // ---- inputs: routing constructed for exact single==EP ----
        hcur.resize((size_t)E*M); for(auto&z:hcur)z=dist(rng);
        ggml_backend_tensor_set(cur,hcur.data(),0,ggml_nbytes(cur));
        // per token: for each die d pick nu_local distinct locals in [0,EpD); global=d*EpD+local
        // weights random per (slot) shared between single and EP
        std::vector<std::vector<int32_t>> locals(M), globals(M); std::vector<std::vector<float>> wt(M);
        std::uniform_int_distribution<int> ld(0,(int)EpD-1);
        for(int64_t t=0;t<M;t++){
            for(int d=0; d<N; d++){
                std::vector<int> chosen;
                while((int)chosen.size()<nu_local){ int l=ld(rng); if(std::find(chosen.begin(),chosen.end(),l)==chosen.end()) chosen.push_back(l); }
                for(int j=0;j<nu_local;j++){ locals[t].push_back(chosen[j]); globals[t].push_back(d*EpD+chosen[j]); wt[t].push_back(dist(rng)+0.05f); }
            }
        }
        auto set_tensor_by_name=[&](const char*nm,const void*data,size_t bytes){
            ggml_tensor*t=ggml_get_tensor(c,nm); if(t) ggml_backend_tensor_set(t,data,0,bytes);
        };
        if(!ep){
            hids.resize((size_t)NUSED*M); hw.resize((size_t)NUSED*M);
            for(int64_t t=0;t<M;t++) for(int s=0;s<NUSED;s++){ hids[t*NUSED+s]=globals[t][s]; hw[t*NUSED+s]=wt[t][s]; }
            set_tensor_by_name("ids",hids.data(),(size_t)NUSED*M*sizeof(int32_t));
            set_tensor_by_name("w",  hw.data(),  (size_t)NUSED*M*sizeof(float));
        } else {
            for(int d=0; d<N; d++){
                std::vector<int32_t> ids_d((size_t)nu_local*M); std::vector<float> w_d((size_t)nu_local*M);
                for(int64_t t=0;t<M;t++) for(int j=0;j<nu_local;j++){ ids_d[t*nu_local+j]=locals[t][d*nu_local+j]; w_d[t*nu_local+j]=wt[t][d*nu_local+j]; }
                char nm[32]; snprintf(nm,32,"ids%d",d); set_tensor_by_name(nm,ids_d.data(),ids_d.size()*sizeof(int32_t));
                snprintf(nm,32,"w%d",d); set_tensor_by_name(nm,w_d.data(),w_d.size()*sizeof(float));
            }
        }

        auto sync=[&]{ for(int d=0;d<N;d++) ggml_backend_synchronize(backend[d]); };
        for(int wmup=0;wmup<5;wmup++) ggml_backend_sched_graph_compute(sched,gf);
        sync();
        std::vector<double> tt; for(int r=0;r<reps;r++){ auto t0=clk::now(); ggml_backend_sched_graph_compute(sched,gf); sync(); tt.push_back(ms_since(t0)); }
        if(out){ out->resize((size_t)E*M); ggml_backend_tensor_get(o,out->data(),0,ggml_nbytes(o)); }
        ggml_free(c); ggml_backend_sched_free(sched);
        return median(tt);
    };

    setenv("GGML_METAL_PEER_ENABLE","1",1);
    { std::vector<float> os,oe; run(64,false,&os); run(64,true,&oe);
      double mab=0,mr=0; for(size_t i=0;i<os.size();i++){double a=std::fabs(os[i]-oe[i]); mab=std::max(mab,a); mr=std::max(mr,a/(std::fabs(os[i])+1e-3));}
      printf("\n# correctness (M=64): max|abs|=%.4g max rel=%.4g -> %s\n",mab,mr,mr<0.05?"OK":"CHECK"); }

    int64_t Ms[]={1,8,64,256,512,1024,2048};
    int nM = (int)(sizeof(Ms)/sizeof(Ms[0]));
    if (getenv("TP_MAXM")) { int mm=atoi(getenv("TP_MAXM")); while(nM>0 && Ms[nM-1]>mm) nM--; }
    printf("\n%6s | %10s | %10s | %10s | %9s | %9s\n","tokens","single ms","ep_host ms","ep_peer ms","spd_host","spd_peer");
    printf("-------+------------+------------+------------+-----------+----------\n");
    for(int mi=0; mi<nM; mi++){
        int64_t M=Ms[mi];
        double s=run(M,false,nullptr);
        unsetenv("GGML_METAL_PEER_ENABLE"); double eh=run(M,true,nullptr);
        setenv("GGML_METAL_PEER_ENABLE","1",1); double ep=run(M,true,nullptr);
        printf("%6lld | %10.3f | %10.3f | %10.3f | %9.2f | %9.2f\n",(long long)M,s,eh,ep,s/eh,s/ep); fflush(stdout);
    }
    printf("\n# best-case balanced EP; one combine all-reduce per MoE layer.\n");
    printf("# spd_* > 1.0 => %d-die expert-parallel beats all-experts-on-one-die.\n", N);

    for(int d=0;d<N;d++){ ggml_backend_buffer_free(bwd[d]); ggml_free(cwd[d]); ggml_backend_free(backend[d]); }
    ggml_backend_buffer_free(bw0); ggml_free(cw0); ggml_backend_free(cpu_backend);
    return 0;
}
