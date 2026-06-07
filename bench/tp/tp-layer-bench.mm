// tp-layer-bench.mm
//
// Tensor-parallel (row/column split) feasibility micro-benchmark for the
// AMD W6800X Metal stack. Step 1 of the TP exploration (see
// docs/tensor-parallel-exploration.md).
//
// It models ONE Megatron-style FFN block:
//
//     up:   a = x @ Wup      [M x H] @ [H x I]   -> [M x I]   (column-parallel)
//     act:  a = gelu(a)                                       (cheap, omitted)
//     down: o = a @ Wdown    [M x I] @ [I x H]   -> [M x H]   (row-parallel)
//     all-reduce(o) across dies                               (the TP tax)
//
// Single-die baseline: one die computes the full up+down GEMM, no comm.
// TP path:             each die computes its I/N column slice (up) and the
//                      matching I/N row slice (down) CONCURRENTLY, then the
//                      [M x H] partial outputs are all-reduced across dies.
//
// The all-reduce is measured both ways:
//   - HOST: bounce through a shared staging buffer (Phase C path)
//   - PEER: direct device-to-device via newRemoteBufferViewForDevice:
//           over Infinity Fabric (Phase D path)
//
// We sweep M (tokens) from 1 (decode/tg regime) to 2048 (prefill/pp regime)
// to find the crossover where N-die TP beats a single die for one FFN.
//
// GEMMs use MPSMatrixMultiplication (f16) for realistic per-die throughput.
//
// Build:
//   clang++ -std=c++17 -fobjc-arc -O2 tp-layer-bench.mm -o tp-layer-bench \
//       -framework Metal -framework MetalPerformanceShaders -framework Foundation
//
// Run:
//   ./tp-layer-bench                 # defaults: H=5120 I=13824 all dies
//   ./tp-layer-bench 4096 14336 4 80 # H I maxDies iters

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <vector>
#include <algorithm>

static double now_ms() {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// f16 matmul left[M x K] @ right[K x N] -> result[M x N], all on `dev`.
struct Gemm {
    MPSMatrixMultiplication *kern;
    MPSMatrix *L, *R, *C;
    static MPSMatrix *mat(id<MTLDevice> dev, id<MTLBuffer> buf, NSUInteger rows, NSUInteger cols) {
        NSUInteger rb = [MPSMatrixDescriptor rowBytesFromColumns:cols dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                       columns:cols
                                                                      rowBytes:rb
                                                                      dataType:MPSDataTypeFloat16];
        return [[MPSMatrix alloc] initWithBuffer:buf descriptor:d];
    }
    static id<MTLBuffer> buf(id<MTLDevice> dev, NSUInteger rows, NSUInteger cols) {
        NSUInteger rb = [MPSMatrixDescriptor rowBytesFromColumns:cols dataType:MPSDataTypeFloat16];
        return [dev newBufferWithLength:rows * rb options:MTLResourceStorageModePrivate];
    }
    Gemm(id<MTLDevice> dev, id<MTLBuffer> lb, id<MTLBuffer> rb, id<MTLBuffer> cb,
         NSUInteger M, NSUInteger K, NSUInteger N) {
        L = mat(dev, lb, M, K);
        R = mat(dev, rb, K, N);
        C = mat(dev, cb, M, N);
        kern = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                    resultRows:M
                                                 resultColumns:N
                                                interiorColumns:K];
    }
    void encode(id<MTLCommandBuffer> cb) {
        [kern encodeToCommandBuffer:cb leftMatrix:L rightMatrix:R resultMatrix:C];
    }
};

int main(int argc, char **argv) {
    @autoreleasepool {
        NSUInteger H     = argc > 1 ? atoi(argv[1]) : 5120;   // hidden dim
        NSUInteger I     = argc > 2 ? atoi(argv[2]) : 13824;  // ffn intermediate
        int        maxN  = argc > 3 ? atoi(argv[3]) : 0;      // 0 = all dies
        int        iters = argc > 4 ? atoi(argv[4]) : 60;

        NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
        std::vector<id<MTLDevice>> devs;
        for (id<MTLDevice> d in all) devs.push_back(d);
        if (maxN > 0 && (int)devs.size() > maxN) devs.resize(maxN);
        int N = (int)devs.size();
        if (N < 2) { NSLog(@"need >=2 devices, have %d", N); return 1; }
        NSUInteger Is = I / N;  // per-die intermediate slice

        bool peers = true;
        uint64_t pg = devs[0].peerGroupID;
        for (auto d : devs) { if (d.peerGroupID == 0 || d.peerGroupID != pg) peers = false; }

        printf("# TP FFN micro-benchmark\n");
        printf("# dies=%d  H=%lu  I=%lu  I/die=%lu  iters=%d  dtype=f16\n",
               N, (unsigned long)H, (unsigned long)I, (unsigned long)Is, iters);
        printf("# peer group: %s (0x%llx)\n", peers ? "SHARED (Infinity Fabric)" : "none",
               (unsigned long long)pg);
        for (int d = 0; d < N; d++)
            printf("#   die %d: %s\n", d, [devs[d].name UTF8String]);

        std::vector<id<MTLCommandQueue>> Q(N);
        for (int d = 0; d < N; d++) Q[d] = [devs[d] newCommandQueue];

        const NSUInteger elem = 2; // f16

        // Persistent weights (independent of M): full on die0, slices on every die.
        id<MTLBuffer> Wup_full   = Gemm::buf(devs[0], H, I);
        id<MTLBuffer> Wdown_full = Gemm::buf(devs[0], I, H);
        std::vector<id<MTLBuffer>> Wup_s(N), Wdown_s(N);
        for (int d = 0; d < N; d++) {
            Wup_s[d]   = Gemm::buf(devs[d], H, Is);
            Wdown_s[d] = Gemm::buf(devs[d], Is, H);
        }

        size_t tokens[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

        printf("\n%6s | %10s | %10s | %10s | %10s | %9s | %9s\n",
               "tokens", "single ms", "tp_cmp ms", "ar_host ms", "ar_peer ms",
               "spd_host", "spd_peer");
        printf("-------+------------+------------+------------+------------+-----------+----------\n");

        for (size_t M : tokens) {
            // ---- buffers for this M ----
            id<MTLBuffer> x0   = Gemm::buf(devs[0], M, H);   // single-die input
            id<MTLBuffer> a0   = Gemm::buf(devs[0], M, I);   // single-die up result
            id<MTLBuffer> o0   = Gemm::buf(devs[0], M, H);   // single-die down result
            Gemm up0(devs[0], x0, Wup_full, a0, M, H, I);
            Gemm dn0(devs[0], a0, Wdown_full, o0, M, I, H);

            std::vector<id<MTLBuffer>> xs(N), as(N), op(N);   // per-die x, a-slice, o-partial
            std::vector<Gemm*> ups, dns;
            for (int d = 0; d < N; d++) {
                xs[d] = Gemm::buf(devs[d], M, H);
                as[d] = Gemm::buf(devs[d], M, Is);
                op[d] = Gemm::buf(devs[d], M, H);
                ups.push_back(new Gemm(devs[d], xs[d], Wup_s[d], as[d], M, H, Is));
                dns.push_back(new Gemm(devs[d], as[d], Wdown_s[d], op[d], M, Is, H));
            }

            // staging buffers for host-mediated all-reduce (out of timed region)
            NSUInteger oBytes = M * [MPSMatrixDescriptor rowBytesFromColumns:H dataType:MPSDataTypeFloat16];
            std::vector<id<MTLBuffer>> stage(N), scratch0(N);
            for (int d = 0; d < N; d++) {
                stage[d]    = [devs[d] newBufferWithLength:oBytes options:MTLResourceStorageModeShared];
                scratch0[d] = [devs[0] newBufferWithLength:oBytes options:MTLResourceStorageModePrivate];
            }

            auto runSingle = [&]() {
                id<MTLCommandBuffer> cb = [Q[0] commandBuffer];
                up0.encode(cb); dn0.encode(cb);
                [cb commit]; [cb waitUntilCompleted];
            };
            auto runTPCompute = [&]() {
                std::vector<id<MTLCommandBuffer>> cbs(N);
                for (int d = 0; d < N; d++) {
                    cbs[d] = [Q[d] commandBuffer];
                    ups[d]->encode(cbs[d]); dns[d]->encode(cbs[d]);
                    [cbs[d] commit];
                }
                for (int d = 0; d < N; d++) [cbs[d] waitUntilCompleted];
            };
            // naive all-reduce: gather op[1..N-1] -> die0, then broadcast die0 -> op[1..N-1].
            // (the elementwise sum on die0 is cheap and identical for both paths; omitted.)
            auto runAR_host = [&]() {
                std::vector<id<MTLCommandBuffer>> cbs;
                // reduce: op[d] (deviceD) -> stage[d] (shared on D) -> scratch0[d] (die0)
                for (int d = 1; d < N; d++) {
                    id<MTLCommandBuffer> c1 = [Q[d] commandBuffer];
                    id<MTLBlitCommandEncoder> e1 = [c1 blitCommandEncoder];
                    [e1 copyFromBuffer:op[d] sourceOffset:0 toBuffer:stage[d] destinationOffset:0 size:oBytes];
                    [e1 endEncoding]; [c1 commit]; [c1 waitUntilCompleted];
                    id<MTLBuffer> view0 = [devs[0] newBufferWithBytesNoCopy:stage[d].contents
                                                                     length:oBytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
                    id<MTLCommandBuffer> c2 = [Q[0] commandBuffer];
                    id<MTLBlitCommandEncoder> e2 = [c2 blitCommandEncoder];
                    [e2 copyFromBuffer:view0 sourceOffset:0 toBuffer:scratch0[d] destinationOffset:0 size:oBytes];
                    [e2 endEncoding]; [c2 commit]; cbs.push_back(c2);
                }
                for (auto c : cbs) [c waitUntilCompleted];
                cbs.clear();
                // broadcast: op[0] (die0) -> stage[0] (shared on 0) -> op[d] (deviceD)
                id<MTLCommandBuffer> cg = [Q[0] commandBuffer];
                id<MTLBlitCommandEncoder> eg = [cg blitCommandEncoder];
                [eg copyFromBuffer:op[0] sourceOffset:0 toBuffer:stage[0] destinationOffset:0 size:oBytes];
                [eg endEncoding]; [cg commit]; [cg waitUntilCompleted];
                for (int d = 1; d < N; d++) {
                    id<MTLBuffer> viewd = [devs[d] newBufferWithBytesNoCopy:stage[0].contents
                                                                     length:oBytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
                    id<MTLCommandBuffer> c = [Q[d] commandBuffer];
                    id<MTLBlitCommandEncoder> e = [c blitCommandEncoder];
                    [e copyFromBuffer:viewd sourceOffset:0 toBuffer:op[d] destinationOffset:0 size:oBytes];
                    [e endEncoding]; [c commit]; cbs.push_back(c);
                }
                for (auto c : cbs) [c waitUntilCompleted];
            };
            auto runAR_peer = [&]() {
                if (!peers) return;
                std::vector<id<MTLCommandBuffer>> cbs;
                // reduce: op[d] viewed on die0, blit into scratch0[d]
                for (int d = 1; d < N; d++) {
                    id<MTLBuffer> remote = [op[d] newRemoteBufferViewForDevice:devs[0]];
                    if (!remote) return;
                    id<MTLCommandBuffer> c = [Q[0] commandBuffer];
                    id<MTLBlitCommandEncoder> e = [c blitCommandEncoder];
                    [e copyFromBuffer:remote sourceOffset:0 toBuffer:scratch0[d] destinationOffset:0 size:oBytes];
                    [e endEncoding]; [c commit]; cbs.push_back(c);
                }
                for (auto c : cbs) [c waitUntilCompleted];
                cbs.clear();
                // broadcast: op[0] viewed on die d, blit into op[d]
                for (int d = 1; d < N; d++) {
                    id<MTLBuffer> remote = [op[0] newRemoteBufferViewForDevice:devs[d]];
                    if (!remote) return;
                    id<MTLCommandBuffer> c = [Q[d] commandBuffer];
                    id<MTLBlitCommandEncoder> e = [c blitCommandEncoder];
                    [e copyFromBuffer:remote sourceOffset:0 toBuffer:op[d] destinationOffset:0 size:oBytes];
                    [e endEncoding]; [c commit]; cbs.push_back(c);
                }
                for (auto c : cbs) [c waitUntilCompleted];
            };

            // warmup
            for (int w = 0; w < 5; w++) { runSingle(); runTPCompute(); runAR_host(); runAR_peer(); }

            std::vector<double> ts, tc, th, tp;
            for (int it = 0; it < iters; it++) {
                double t0 = now_ms(); runSingle();    ts.push_back(now_ms() - t0);
                t0 = now_ms();        runTPCompute(); tc.push_back(now_ms() - t0);
                t0 = now_ms();        runAR_host();   th.push_back(now_ms() - t0);
                if (peers) { t0 = now_ms(); runAR_peer(); tp.push_back(now_ms() - t0); }
            }
            double s = median(ts), c = median(tc), h = median(th), p = peers ? median(tp) : 0;
            double spd_h = s / (c + h);
            double spd_p = peers ? s / (c + p) : 0;
            printf("%6zu | %10.3f | %10.3f | %10.3f | %10.3f | %9.2f | %9.2f\n",
                   M, s, c, h, p, spd_h, spd_p);
            fflush(stdout);

            for (auto g : ups) delete g;
            for (auto g : dns) delete g;
        }
        printf("\n# spd_* > 1.0  => N-die TP beats a single die for one FFN block.\n");
        printf("# A full transformer layer has ~2 all-reduces (attn + ffn).\n");
    }
    return 0;
}
