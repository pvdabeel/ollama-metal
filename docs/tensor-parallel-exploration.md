# Tensor parallelism on the AMD W6800X Metal stack — exploration

Status: **exploration complete (steps 1–6).** Verdict: TP/EP are not worth
building for any currently deployed model. `qwen3.5moe` loses (small dims + sparse
MoE); `gemma4:31b` (dense) shows a ~1.5–1.6× **decode** win at F16 but it
**collapses to ~1.0× at the deployed Q4_K_M** — the win was a weight-bandwidth
artifact that quantization removes. Separately, an in-stack copy benchmark plus a
real `qwen3.5:122b` layer-split A/B (step 6/6b) show **Infinity Fabric gives no
reproducible inference speedup** — not decode, not even long-prefill of an 80 GB
layer-split model; it only helps bulk cross-die staging. No production code
yet. This note records why row/tensor-parallel split is "CUDA-only" today, what
it would take on Metal, and the measured feasibility numbers that decide whether
it is worth building.

> **Correction — the `peer` columns and Infinity Fabric (read this first).**
> Only **step 1** (the MPS micro-bench `tp-layer-bench.mm`) measured Infinity
> Fabric for real: it implements its own all-reduce with explicit
> `newRemoteBufferViewForDevice:` peer blits vs a host-staged path, so its
> `ar_host`/`ar_peer` split is a genuine head-to-head.
> **Steps 2–5 do NOT exercise the fabric.** They run on `ggml_backend_sched`,
> and the scheduler moves the tiny cross-die partials by host-staging them
> (metal→host→metal through the CPU backend); it never calls the Phase-D
> `ggml_metal_buffer_cpy_tensor` peer path, so `GGML_METAL_PEER_ENABLE` has no
> effect there. Verified by instrumenting that function: it is never reached for
> these graphs. Consequently the `spd_peer ≈ spd_host` ties in steps 2–5 mean
> only "the combine cost was not the bottleneck" — they are **not** a
> fabric-vs-host comparison (both columns used the same host-staged path). This
> does not change any conclusion: the cross-die payloads are tiny and
> latency-bound, and the verdicts are driven by matmul size / weight-read
> bandwidth, not the copy path. Step 6 (`tp-copy-ggml.cpp`) measures the fabric
> for real (it bypasses the scheduler) and quantifies this: fabric is ~4.7×
> faster than host for bulk copies but both are latency-bound (~270 µs vs
> ~550 µs) at the tiny per-token payloads, so the fabric cannot rescue TP/EP.

## 1. Why row-split is CUDA-only today

It is not a missing kernel — it is a backend-topology difference.

| | CUDA | Metal (this stack) |
|---|---|---|
| Backend ↔ device | one backend owns **all** GPUs | one backend **per die** (`ggml_backend_metal_device_init` builds 4) |
| Who splits an op | the backend, *internally* | nobody — the **scheduler** assigns each whole op to one backend |
| Cross-device copy | `cudaMemcpyPeerAsync` | `newRemoteBufferViewForDevice:` (built in Phase D) |

In `ggml-cuda.cu`, `ggml_cuda_op_mul_mat` loops over every device, broadcasts
the activations (`src1`) to each via `cudaMemcpyPeerAsync`, runs each device's
row-slice matmul on its own stream, then gathers the partial results. The
`ggml_backend_sched` scheduler never sees this: the multi-GPU matmul is a single
op on a single backend. The split weight lives in a special
`ggml_backend_cuda_split_buffer_type` that physically shards rows across devices
at load time.

Metal is the opposite: `ggml_metal_device_t` wraps exactly one `id<MTLDevice>`,
`ggml_metal_op_mul_mat` encodes onto that one device's command buffer, and the
scheduler hands each op to one die. **There is no scheduler mechanism to fan one
op across backends** — that is the wall.

(A `ggml-cuda/allreduce.cuh` all-reduce pipeline exists in-tree but is
untracked/CUDA-only. No Metal equivalent — Phase D's peer copy is the building
block to write one.)

## 2. Three implementation paths

**A. CUDA-style "meta-backend".** A new Metal backend that internally owns all
dies + a `ggml_backend_metal_split_buffer_type`, with `ggml_metal_op_mul_mat`
rewritten to fan out across N command queues and gather via Phase D remote
buffer views.
- Pro: transparent to scheduler and llama.cpp; works for all architectures.
- Con: ggml-metal is built bottom-to-top around one-device-per-context (device,
  context, queue, pipeline cache, residency sets all per-die). Major rewrite.

**B. Split-buffer-type + scheduler.** Reuse the existing
`make_gpu_buft_list(... LLAMA_SPLIT_MODE_ROW ...)` in `llama-model.cpp`.
- Verdict: **does not work as-is.** That path assumes a single backend
  reinterprets the split buffer inside its op (the CUDA model). With per-die
  backends the scheduler still places the matmul on one backend. Dead end
  without also doing A.

**C. Graph-level sharding.** Modify llama.cpp graph builders (`build_attn`,
`build_ffn`, and `build_moe_ffn`/`MUL_MAT_ID` for this MoE model) to emit, in TP
mode, N explicit parallel sub-matmuls — each consuming a weight slice resident on
a different die's normal buffer — followed by a concat (column-parallel) or
sum/all-reduce (row-parallel) node.
- Pro: the scheduler naturally places each sub-matmul on the die holding its
  weight slice, and the **existing cross-die copies (Phase C host / Phase D
  fabric) handle broadcast + gather automatically.** No ggml-metal surgery.
- Con: invasive to architecture-agnostic model code, per-op-pattern, permanent
  fork divergence; needs custom weight placement at load.

## 3. Step 1 — measured feasibility (`bench/tp/tp-layer-bench.mm`)

Models one Megatron-style FFN block (up = column-parallel, down = row-parallel +
all-reduce) with real MPS f16 GEMMs. Single-die baseline = one die does the full
up+down (this is exactly what current per-layer placement does). TP path = each
die computes its `I/N` slice concurrently, then the `[tokens x H]` partials are
all-reduced. All-reduce measured both host-mediated (Phase C) and peer/fabric
(Phase D). Dims H=5120, I=13824 (dense ~13B-class proxy), 60 iters, median.

`spd_*` = single_ms / (tp_compute_ms + allreduce_ms). > 1.0 ⇒ TP wins.

### 4 dies

| tokens | single ms | tp_cmp ms | ar_host | ar_peer | spd_host | **spd_peer** |
|--------|-----------|-----------|---------|---------|----------|----------|
| 1      | 2.63      | 1.33      | 1.64    | 1.08    | 0.88     | **1.09** |
| 8      | 2.54      | 1.12      | 1.45    | 0.95    | 0.99     | **1.23** |
| 64     | 3.03      | 1.37      | 2.05    | 1.11    | 0.88     | **1.22** |
| 256    | 4.45      | 1.97      | 3.49    | 1.30    | 0.81     | **1.36** |
| 512    | 7.69      | 2.55      | 4.91    | 1.49    | 1.03     | **1.91** |
| 1024   | 14.20     | 3.70      | 8.52    | 2.03    | 1.16     | **2.48** |
| 2048   | 27.40     | 6.65      | 16.49   | 3.53    | 1.18     | **2.69** |

### 2 dies

| tokens | single ms | tp_cmp ms | ar_host | ar_peer | spd_host | **spd_peer** |
|--------|-----------|-----------|---------|---------|----------|----------|
| 1      | 2.56      | 1.03      | 0.84    | 0.85    | 1.37     | **1.37** |
| 64     | 3.00      | 1.35      | 1.22    | 0.94    | 1.17     | **1.31** |
| 512    | 7.81      | 3.99      | 3.37    | 1.26    | 1.06     | **1.49** |
| 1024   | 14.59     | 6.21      | 5.65    | 1.79    | 1.23     | **1.82** |
| 2048   | 27.42     | 11.34     | 10.05   | 2.67    | 1.28     | **1.96** |

### Findings

1. **Infinity Fabric (Phase D) is essential for TP.** Host-mediated all-reduce
   (`spd_host`) is often < 1.0 — the bounce-copy cost eats the compute win. Peer
   all-reduce stays cheap and roughly flat as batch grows, so `spd_peer` climbs.
   This is the first workload where Phase D pays off (layer-split copies were too
   tiny to matter; TP all-reduces are large in prefill).
2. **TP is a prefill (pp) accelerator.** 4-die: ~1.9–2.7× at 512–2048 tokens.
   2-die: ~1.5–2.0×. Compute parallelism (≈4× at 2048) dominates and the fabric
   all-reduce is cheap.
3. **TP is ~neutral for single-token decode (tg).** 4-die batch-1 ≈ 1.09×, 2-die
   ≈ 1.37×. More dies = more sync overhead per token, so for decode fewer dies
   are better. Net: TP does **not** meaningfully speed up a lone chat's decode.

### Caveats (read before trusting the decode numbers)

- The benchmark submits one command buffer per FFN; a real llama.cpp graph
  batches the whole layer into one submit, amortizing fixed overhead. So the
  batch-1 fixed costs here are **overstated for both sides**, and the real-graph
  decode picture is likely closer to neutral/negative than the 1.09–1.37× shown.
  The **prefill** win is robust to this (compute-dominated).
- All-reduce is naive reduce-to-root + broadcast; a ring all-reduce would make
  the peer path even cheaper at large batch (conservative here).
- The local elementwise sum of N partials is omitted (cheap, identical both
  paths).
- Dense FFN proxy. The real model is `qwen3.5moe` (`MUL_MAT_ID`): in prefill all
  experts are hit so GEMM-like (proxy holds); in decode only K experts fire per
  token (smaller, more latency-bound — reinforces "decode = neutral").

## 4. Step 2 — graph-level (Approach C) prototype on the real stack (`bench/tp/tp-ffn-ggml.cpp`)

Step 1 used MPS GEMMs (a *fast* matmul). Step 2 builds the TP FFN as a real
**ggml graph across N Metal backends** driven by `ggml_backend_sched` — i.e. the
actual kernels and scheduler Ollama/llama.cpp use. The graph contains N parallel
sub-matmuls (weights pinned per die) + an add tree (= all-reduce); the scheduler
places each matmul on the die holding its weight slice and inserts the cross-die
copies (our Phase C host / Phase D peer path) for the x-broadcast and the gather.

FFN is `o = Σ_d Wdown_s[d] @ (Wup_s[d] @ x)`, mathematically identical to the
single-die full FFN, so it doubles as a correctness check. H=5120, I=13824, f16.

**Correctness: bit-identical** (max abs diff = 0, max rel = 0) for both 2- and
4-die. The scheduler-driven sharding + cross-die-copy all-reduce is exact.

`spd` = single-die ms / TP ms (end-to-end, including all scheduler copies):

| tokens | single ms | **2-die spd** | **4-die spd** |
|--------|-----------|-----------|-----------|
| 1      | ~0.85     | 1.7–1.8   | 1.4–1.6   |
| 8      | 2.35      | 2.56      | 3.4       |
| 64     | 7.29      | 1.89      | 3.40      |
| 256    | 24.4      | 1.94      | 3.63      |
| 512    | 53.3      | 1.9–2.1   | **4.2**   |
| 1024   | 92.8      | 1.5       | 3.3–3.6   |
| 2048   | 184       | 1.65      | 3.3       |

### The big reversal vs step 1

Step 1 (MPS) said TP is comm-bound, fabric-dependent, prefill-only, neutral for
decode. **On the real ggml kernels the opposite is true:**

1. **The real matmuls are compute-bound, so TP scales nearly linearly** — 4-die
   ~3.3–4.2×, 2-die ~1.9×, *across all batch sizes including decode*. Reason: the
   W6800X has **no `simdgroup_matrix`** (`simdgroup matrix mul. = false` in the
   device log), so ggml's f16 GEMM is far slower than MPS's; single-die M=2048
   takes 184 ms here vs 27 ms in the MPS mock. When compute dominates,
   parallelising it across dies is a near-linear win.
2. **This is exactly the fix for the "every GPU at 30%" problem.** Today's
   layer-split runs one die at a time per token (pipeline). TP runs all dies on
   every layer concurrently, converting the idle 70% into single-stream speedup.
3. **Comm is not the bottleneck here** — `spd_host ≈ spd_peer`, because the
   cross-die copy is a small fraction of the (slow) compute. (Caveat: both
   columns actually ran the same host-staged path — see the correction note at
   the top; `ggml_backend_sched` never engaged the Phase-D fabric here.) So a
   first TP build can ship on the host-mediated copy path; the remote-view path even needs
   caching (it currently builds a new remote view per copy) before it helps.

### Issues found (must be handled by the full build)

- **`ggml_backend_sched` `parallel=true` aborts with >2 Metal backends**: it
  corrupts an operand type and trips the `op_bin` `GGML_TYPE_F32` assert
  (`ggml-metal-ops.cpp:3095`). `parallel=false` is correct and still overlaps the
  dies via async command buffers (hence the ~4×). The concurrent-execution path
  is a real ggml-metal multi-backend bug to fix/avoid.
- Per-FFN submit overhead inflates the batch-1 numbers somewhat; a real fused
  layer graph amortises it. End-to-end model gains will be below the per-FFN
  microbench because norms / routing / residuals / per-layer sync don't
  parallelise, and MoE `MUL_MAT_ID` needs its own split.

## 5. Step 3 — full transformer block at the REAL model dims (`bench/tp/tp-block-ggml.cpp`)

Step 2 measured an isolated FFN matmul pair at generous dense dims
(H=5120, I=13824). Step 3 measures a **whole layer** — RMSNorm + GQA attention
(head-parallel core + O-projection all-reduce) + residual + RMSNorm + FFN
(all-reduce) + residual — at the **actual `qwen3.5moe` dims**: `n_embd=3072`,
`n_head=32`, `n_head_kv=2`, `head_dim=256`, FFN proxy F=8192. Built on the real
ggml-metal backend + scheduler, with both all-reduces a real TP layer needs.

**Correctness: bit-identical** to single-die (max diff = 0), 2- and 4-die.

`spd` = single-die ms / TP ms, whole layer:

| tokens | single ms | 2-die spd | 4-die spd |
|--------|-----------|-----------|-----------|
| 1 (decode) | 1.11 | 1.09–1.11 | 1.19–1.22 |
| 8      | 1.84 | 1.00–1.03 | 1.05 |
| 64     | 8.28 | 0.85      | 0.67 |
| 256    | 26.0 | 0.96      | 0.85 |
| 512    | 49.6 | 0.96      | 0.89 |
| 1024   | 104  | 0.97      | 0.90 |
| 2048   | 257  | 0.97–0.98 | 0.90–0.94 |

### Finding: for THIS model, full-layer TP does not pay off

The big step-2 FFN numbers (1.9–4×) **do not survive** at the real dimensions
once the whole layer is included. End-to-end per-layer TP is **break-even at best
and a net slowdown during prefill** (down to 0.67× at 4-die). Reasons:

1. **`qwen3.5moe` is small + fine-grained MoE.** `n_embd=3072` (vs 5120 in
   step 2) and the real FFN is `MUL_MAT_ID` over 8 tiny active experts
   (intermediate 1024), not a big dense GEMM. The matmuls are small, so they are
   kernel-launch / overhead bound, and slicing them *smaller* across dies + two
   per-layer all-reduces (gather+broadcast cross-die copies + sync) + replicated
   K/V projections costs more than the parallelism saves.
2. **More dies = worse** in the prefill range (4-die < 2-die): extra sync and
   copy overhead with no extra usable compute.
3. **Only single-token decode shows a slight win** (~1.1× at 2-die, ~1.2× at
   4-die) — consistent with parallel weight-read bandwidth (each die streams its
   slice of the layer weights concurrently). That is the one regime where TP has
   a real, if modest, edge for this model.

Comm is again a non-factor here (`spd_host ≈ spd_peer`; both columns are in fact
the same host-staged path — fabric was not engaged, see the top correction): comm is not the
differentiator at these sizes.

This reverses the step-2 read: TP's large win was an artifact of large dense
matmuls. `qwen3.5moe` does not have them.

## 6. Step 4 — expert parallelism (EP) for the MoE FFN (`bench/tp/tp-moe-ggml.cpp`)

Step 3 closed the door on *tensor*-splitting `qwen3.5moe`. The natural follow-on
for an MoE model is **expert parallelism**: instead of slicing each weight matrix,
give each die a disjoint subset of the 256 experts and let the router fan tokens
out to whichever die owns the selected expert. The appeal is that EP needs only
**one combine per MoE layer** (sum the per-die partials) rather than TP's two
per-layer all-reduces, and the experts are independent.

The prototype models the MoE FFN at the real dims (`n_embd=3072`, expert
`ff=1024`, `n_expert=256`, `n_used=8`) on the real `ggml-metal` `MUL_MAT_ID`
path:

- **single** (today's behaviour): all 256 experts on one die; one `mul_mat_id`
  over the 8 selected experts/token, weighted-sum → `[n_embd, n_tokens]`.
- **EP**: each die owns `256/N` experts and handles `n_used/N` of every token's
  selected experts (**best-case perfect load balance**), then the per-die
  `[n_embd, n_tokens]` partials are all-reduced.

Routing is constructed so every token's 8 experts split evenly into `N` groups,
all globally distinct, so `single == Σ_die partials` **exactly** — the bench
doubles as a correctness check (`max|abs| = 0`, both 2- and 4-die).

```
            single        2 dies (EP)            4 dies (EP)
tokens        ms      host    peer   spd       host    peer   spd
     1       2.2      5.7     5.4   0.38       5.4     5.5   0.36
     8      13.4     39.3    40.6   0.34      39.4    40.8   0.34
    64      90.1    197.9   214.4   0.46     199.4   203.7   0.51
   256     397.5    511.7   537.4   0.78     489.2   485.4   0.81
   512     744.3    833.3   848.9   0.89     918.3   896.2   0.81
  1024    1606.8   1608.7  1698.1   1.00    1718.7  1743.8   0.85
  2048    3120.7   3347.1  3402.3   0.93    3423.0  3399.7   0.88
```
(`spd` = single / ep_host; >1.0 means EP wins. The `peer` column tried to opt into
the fabric but the scheduler host-stages anyway — see the top correction.)

### Finding: EP does not pay off either — and adding dies makes it worse

- **EP never beats single-die.** Best case is 2-die merely *reaching parity*
  (1.00×) at M=1024; everywhere else it is a slowdown. 4-die never reaches
  parity and **plateaus below 2-die** (~0.85×).
- **More dies = worse, not better.** This is the opposite of step-2 dense TP.
  Splitting `n_used=8` experts over 4 dies leaves each die only 2 tiny
  `3072×1024` expert matmuls/token — *more* overhead-bound — while the combine
  grows to a 4-way reduce. The compute you distribute is already sparse and
  small; you cannot make the per-expert matmuls bigger by spreading them out,
  only smaller.
- **Decode (small M) is hit hardest** (0.34–0.41×): pure dispatch + combine
  overhead with almost no compute to amortise it.
- **Combine cost is a non-factor** (`peer ≈ host` — but note both are the same
  host-staged path; the fabric was not engaged, see the top correction), consistent with
  Phase D and steps 1–3: cross-die data volume is never the bottleneck here.
- This is the **best case** (perfect balance, static routing). Real EP adds
  imbalance and a dynamic all-to-all, so it can only be worse.

The single-die path wins precisely because it keeps all experts local (no
cross-die combine) and batches one efficient `mul_mat_id` — MoE sparsity is
already the "savings", and there is nothing left for EP to parallelise
profitably on this stack.

## 7. Step 5 — `gemma4:31b`, a DENSE model: the case where TP finally wins (`tp-block-ggml.cpp`, parameterised)

Steps 3–4 ruled out `qwen3.5moe` (small dims + sparse MoE). The natural retest is
a *dense* model with big matmuls. `gemma4:31b` (from `ollama show` + gguf
metadata): `n_embd=5376`, `n_head=32`, `n_head_kv=16`, `head_dim=512`
(Q-proj 5376→16384), `ffn=21504`, 60 layers. The step-3 prototype was generalised
to take these dims via env vars (`TP_E/TP_NH/TP_NKV/TP_HD/TP_F`) and to split GQA
KV-heads across dies when divisible (Megatron-correct; replicate otherwise).

Per-layer median ms, **F16 weights**, correctness exact (`max|abs|=0`):

```
            single      2 dies          4 dies
tokens        ms     spd            spd
     1       3.15    1.07           1.50   <- decode
     8       5.79    1.22           1.62   <- decode
    64      29.4     0.95           0.87
   256     106.7    0.99           0.96
   512     210.8    1.00           0.99
  1024     428.2    1.00           1.00
  2048     954.2    1.02           1.04
```

### Finding: TP wins at *decode* for a big dense model — the first positive result

- **4-die TP gives ~1.5–1.6× at decode (M=1–8)** — the token-generation regime,
  and exactly where layer-split leaves GPUs idle ("30% per GPU"). Stable across
  reps; the `host` and `peer` columns are identical because they are literally
  the same host-staged copy path (the scheduler never engaged the fabric — see
  the top correction), not because fabric and host were measured equal.
- **Prefill is break-even (~1.0).** This is the mirror image of step 2's
  *isolated-FFN* prefill win: here the whole layer at large M is already
  compute-saturated on one die, and the two all-reduces + non-split parts
  (attention core, norms) wash out the compute split.
- **Why decode wins:** at M=1 the layer is **memory-bandwidth-bound on weight
  reads**. Splitting Wq/Wo/Wgate/Wup/Wdown across 4 dies multiplies aggregate
  VRAM bandwidth, so the (large) weights are read in parallel. The all-reduce
  payload at decode is tiny (`[n_embd,1]` ≈ 21 KB), so it barely costs anything.

### The F16 decode win is a bandwidth artifact — Q4_K_M (the deployed quant) erases it

The prototype was extended to quantize weights to **Q4_K** (`TP_QUANT=q4k`). For
gemma4's dims every contraction dim and every `IN/N` row-split is divisible by the
256-elem Q4_K block, so each split is quantized block-aligned and is bit-identical
to quantizing the full matrix (correctness stays `max|abs|=0`). Per-layer median ms:

```
            single Q4_K   2 dies         4 dies
tokens          ms      spd            spd
     1         1.45     1.01           1.04   <- decode (was 1.50x at F16!)
     8         7.09     0.98           0.96
    64        30.4      0.95           0.87
   256       109.9     0.99           0.96
   512       218.6     1.00           1.00
  1024       442.8     1.00           1.00
  2048       976       1.01           1.03
```

- **Quantization removes the win entirely.** Single-die decode drops 3.15 → 1.45 ms
  (F16 → Q4_K): with ~4× less weight data the layer is no longer bandwidth-starved
  on one die, so there is nothing for TP to recover. 2- and 4-die are a wash
  (1.0–1.06×) across the whole sweep.
- **The F16 1.5–1.6× was purely a weight-read-bandwidth effect**, not real headroom.
  The deployed model is Q4_K_M, so the production answer is: **TP does not help
  `gemma4:31b` either.**
- (Implementation note kept for the record: a quantized TP build must split Wo/Wdown
  on Q4_K block boundaries; gemma4's dims happen to satisfy this for N∈{2,4}.)

## 8. Step 6 — a REAL fabric-vs-host cross-die copy measurement (`tp-copy-ggml.cpp`)

The correction above says steps 2–5 never engaged the fabric. To measure it for
real, `tp-copy-ggml.cpp` calls `ggml_backend_tensor_copy(src_on_die_b, dst_on_die_0)`
directly — for two metal private buffers that lands in
`ggml_metal_buffer_cpy_tensor`, which takes the host bounce when
`GGML_METAL_PEER_ENABLE` is unset and the Phase-D Infinity-Fabric blit when it is
set. So the two env settings exercise exactly the two paths, end to end.

```
transfer        host GB/s   peer GB/s   speedup   (die 1->0; 2->0 and 3->0 identical)
   16 KB          0.03        0.06        2.5x
  256 KB          0.39        1.04        2.7x
    1 MB          1.21        3.69        3.0x
   16 MB          4.89       19.32        4.0x
  256 MB          5.85       27.68        4.7x
```
Correctness `ok/ok` on both paths at every size and every die pair.

### Findings — fabric works, and now we know exactly why it still can't help TP/EP

1. **The Phase-D fabric path is real and engaged.** `peer ≫ host` (up to 4.7×,
   saturating ~28 GB/s vs ~6 GB/s) proves `newRemoteBufferViewForDevice:` + blit
   actually ran. (Matches the ~26 GB/s noted in `ggml-metal-device.m`.)
2. **All four dies share one peer group / fabric** — 1→0, 2→0, 3→0 are all
   ~28 GB/s, so intra-card vs cross-card is equivalent; no topology penalty.
3. **The win is bandwidth, only for BULK transfers.** At the ~16 KB payload a
   per-token all-reduce actually moves, both paths are **latency-bound**:
   ~270 µs/copy on fabric, ~550 µs on host (submit + blit + `waitUntilCompleted`).
   That per-copy latency × ~120 copies/token (60 layers × 2 all-reduces) is
   ~32 ms/token of pure copy overhead — far more than the compute TP/EP would
   save. **Fabric halves the latency but it is still the dominant cost**, so the
   earlier "peer ≈ host wash" holds even with the fabric genuinely switched on.
4. So the fabric's real value is **one-off bulk movement** (e.g. staging a
   >32 GB model's cross-die layer split), not per-token inference copies — which
   is exactly what Phase D concluded, now backed by an in-stack measurement.

### Step 6b — does a real >32 GB layer-split model benefit? (`qwen3.5:122b` A/B)

The obvious "best case" for the fabric in real inference is a model too big for
one die (so it is layer-split) driven with a long prompt (so the cross-die
residual handoff `[n_embd, n_tokens]` is bulk). Tested directly with the
production `llama-server` on `qwen3.5:122b` (80 GB, split across all 4 dies),
6304-token prefill, `GGML_METAL_PEER_ENABLE` on vs off:

- **First A/B looked like a win:** prefill 135 (on) vs 98 (off) tok/s — a
  tempting +38%. **It did not reproduce.**
- **Clean within-process A/B** (4 iterations per server, prompt-cache busted, to
  remove the per-launch 80 GB disk-reload confound):
  ```
  fabric ON : 135.5 135.3 135.4 135.2 tok/s
  fabric OFF: 135.4 135.3 135.0 108.8 tok/s   (last = sporadic system-noise dip)
  decode    : ~19–22 tok/s both modes (unchanged)
  ```
  **ON ≈ OFF ≈ 135 tok/s.** The first run's 98 (and the later 108.8) are
  cold-cache / contention outliers that appear in both modes; they are not
  flag-correlated.

**Conclusion: no reproducible fabric benefit, even here.** Real layer-split has
only a handful of cross-die boundaries per forward, so the cross-die copy volume
is a negligible fraction of the MoE prefill compute over 6304 tokens — a 4.7×
faster copy of a negligible slice is still negligible. This confirms the original
analysis (the noisy first A/B had briefly seemed to overturn it). **No standard
inference workload on these models benefits from Infinity Fabric**: decode copies
are tiny/latency-bound, and even long-prefill layer-split copies are
compute-dominated. The fabric only wins when cross-die transfer is *itself* the
dominant cost (bulk staging / sharding), which inference is not.

## 9. Recommendation

The six steps together give a clear, dimension-dependent verdict:

- **TP helps only large *F16* dense matmuls — and quantization erases that.**
  Step 2 (H=5120/I=13824, FFN only, F16): 1.9–4×. Step 3 (`qwen3.5moe`): 0.67–1.22×.
  Step 5 (`gemma4:31b`, whole layer): **1.5–1.6× at decode at F16, but only
  1.0–1.06× at Q4_K_M** (the deployed quant). The F16 win is a weight-bandwidth
  artifact; once weights are 4× smaller, single-die decode is already fast and
  there is nothing to recover.
- **For `qwen3.5moe` specifically, full TP is not worth building.** Small
  `n_embd` + fine-grained MoE means the per-layer matmuls are overhead-bound;
  slicing them across dies plus two per-layer all-reduces costs more than it
  saves everywhere except a slim ~1.1–1.2× at single-token decode.
- **Approach C is validated as a *mechanism*** (correct, scheduler places the
  sub-matmuls, cross-die copies do the all-reduce) — the blocker is economic, not
  technical. (Phase D fabric was not even engaged by the scheduler in steps 2–5;
  the cross-die copies were host-staged — see the top correction.)

Recommended course:

1. **Do not invest in the full llama.cpp loader+graph TP surgery for this model.**
   Step 3 (cheap, standalone, real dims) was the de-risk that tells us the large
   integration would not pay off for `qwen3.5moe`. This is the payoff of
   measuring before building.
2. **Single-stream "30% per GPU" stays a property of layer-split**; the realistic
   lever for *aggregate* throughput remains `OLLAMA_NUM_PARALLEL` (pipeline
   overlap across requests), which needs no new code.
3. **Expert parallelism is also ruled out for this model** (step 4): even the
   best-case balanced EP never beats single-die, and more dies make it worse.
   MoE sparsity already leaves the per-expert matmuls too small to profit from
   distribution + a combine.
4. **Dense models (`gemma4:31b`) were the most promising case and still do not
   justify TP at the deployed quant.** Step 5's F16 decode win (1.5–1.6×)
   vanished to ~1.0× at Q4_K_M — the Q4 gate (≳1.3×) failed. Do not build the
   llama.cpp loader+graph TP surgery for any currently deployed model. Revisit
   only for a model run at F16/BF16 *and* with very large matmuls (e.g. a 70B+
   dense model served unquantized), where the bandwidth headroom actually exists.
5. **The realistic lever on this stack is concurrency** — multiple streams /
   multiple models across dies via the existing multi-die layer-split — not
   splitting a single stream. This needs no new code.

Known issue regardless: `ggml_backend_sched` `parallel=true` aborts with >2 Metal
backends (`op_bin` F32 assert); `parallel=false` is the correct setting and still
overlaps dies via async command buffers.

Reproduce:

```sh
cd bench/tp
LL=~/Desktop/ollama-metal/work/llama.cpp
# step 1 — MPS upper-bound mock (comm-bound regime)
clang++ -std=c++17 -fobjc-arc -O2 tp-layer-bench.mm -o tp-layer-bench \
    -framework Metal -framework MetalPerformanceShaders -framework Foundation
./tp-layer-bench                    # 4 dies
./tp-layer-bench 5120 13824 2 60    # 2 dies
# step 2 — real ggml-metal + scheduler (Approach C, the representative regime)
clang++ -std=c++17 -O2 tp-ffn-ggml.cpp -o tp-ffn-ggml \
    -I$LL/ggml/include -L$LL/build-test/bin -lggml -lggml-base -lggml-metal -lggml-cpu \
    -Wl,-rpath,$LL/build-test/bin -framework Foundation
./tp-ffn-ggml                 # 2 dies
./tp-ffn-ggml 5120 13824 4 40 # 4 dies
# step 3 — full transformer block at real qwen3.5moe dims (the decisive test)
clang++ -std=c++17 -O2 tp-block-ggml.cpp -o tp-block-ggml \
    -I$LL/ggml/include -L$LL/build-test/bin -lggml -lggml-base -lggml-metal -lggml-cpu \
    -Wl,-rpath,$LL/build-test/bin -framework Foundation
./tp-block-ggml 2 40   # 2 dies, qwen3.5moe default dims
./tp-block-ggml 4 40   # 4 dies
# step 5 — same prototype at gemma4:31b dense dims (F16 = the apparent win)
TP_E=5376 TP_NH=32 TP_NKV=16 TP_HD=512 TP_F=21504 ./tp-block-ggml 4 8
# step 5b — gemma4:31b at the DEPLOYED quant (TP_QUANT=q4k = the win disappears)
TP_QUANT=q4k TP_E=5376 TP_NH=32 TP_NKV=16 TP_HD=512 TP_F=21504 ./tp-block-ggml 4 8
# step 4 — expert parallelism on the real MUL_MAT_ID MoE path (the MoE-native test)
clang++ -std=c++17 -O2 tp-moe-ggml.cpp -o tp-moe-ggml \
    -I$LL/ggml/include -L$LL/build-test/bin -lggml -lggml-base -lggml-metal -lggml-cpu \
    -Wl,-rpath,$LL/build-test/bin -framework Foundation
./tp-moe-ggml 2 8      # 2 dies   (TP_MAXM=512 caps the token sweep)
./tp-moe-ggml 4 8      # 4 dies
# step 6 — REAL fabric-vs-host cross-die copy (bypasses the scheduler)
clang++ -std=c++17 -O2 tp-copy-ggml.cpp -o tp-copy-ggml \
    -I$LL/ggml/include -L$LL/build-test/bin -lggml -lggml-base -lggml-metal \
    -Wl,-rpath,$LL/build-test/bin -framework Foundation
./tp-copy-ggml         # all src dies -> die 0, host vs peer GB/s
```
