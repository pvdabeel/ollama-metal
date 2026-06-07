# Benchmarks

All numbers measured on the target machine:
- Mac Pro 2019 (MacPro7,1), Intel Xeon W-3275M (28 cores), macOS 26.5.1
- AMD Radeon PRO W6800X Duo (one die used), `recommendedMaxWorkingSetSize`
  ~34 GB, `simdgroup matrix mul. = false`, `has unified memory = false`
- Model: `llama3.2` 3B Q4_K_M (reused from the local Ollama blob store)
- Tool: Ollama `/api/generate` (`temperature=0`), t/s from the response's
  `eval_count / eval_duration` (gen) and `prompt_eval_*` (prompt)

## Single-die: backend comparison (measured via Ollama)

| Backend / config                          | prompt t/s | gen t/s |
|-------------------------------------------|-----------:|--------:|
| GPU, stock Ollama defaults (mmap on)      |         32 |     1.6 |
| GPU, concurrency on (stock)               |        ~2  |    ~2.3 | (garbage output)
| CPU (28 threads, `num_gpu=0`)             |        133 |      28 |
| **GPU, ollama-metal fixes (default)**     |    **211** | **~104**|

The ollama-metal default = concurrency-off (correctness) + mmap-off on
discrete (weights in VRAM) + tiled `mul_mm` (prompt throughput). No env vars
or request flags required.

Correctness: the fixed build produces coherent text (verified with multiple
prompts) - not the garbage seen on the stock Metal discrete-GPU path.

The README headline table reports the same model through the raw `llama-server`
`/completion` harness instead (512-token prompt, `n_predict=128`, temp 0), which
is the harness used for the cross-die section below. Longer prompts and the
larger batch push the patched-Metal single die to pp ~473 / tg ~73 and the CPU
to pp ~216 / tg ~24; absolute t/s differ from the Ollama-API numbers above
(shorter prompts) but the GPU ≫ CPU ≫ broken-stock ordering is identical.

## Takeaways

- Two independent "Metal == Apple Silicon UMA" bugs caused the problem, neither
  was the wavefront width:
  - **Concurrency** (correctness): the Metal backend's concurrent dispatch
    corrupts tensors on discrete AMD. Every op passes `test-backend-ops`
    individually; the race only shows in the full graph.
  - **mmap weights** (performance): Ollama left weights mmap'd in host RAM, so
    the GPU streamed ~1.9 GB over PCIe per token (1.6 t/s). Loading into VRAM
    is the ~65x generation win.
- Generation now beats the CPU ~3.7x; prompt ~1.6x.
- The threadgroup-tiled `mul_mm` kernel helps prompt eval (~12x vs stock
  `mul_mv`) but is NOT required for correctness.
- One die (~32 GB) easily fits 3B-30B (A3B) class models; multi-die is for
  capacity beyond 32 GB and concurrent multi-model throughput.

## Multi-die: one model per die (concurrent multi-model)

The Mac Pro's two W6800X Duo cards expose **4 distinct physical dies** (unique
Metal `registryID`s, 32 GiB each = 128 GiB total). With the multi-die patches a
single `ollama serve` discovers all four (`MTL0..MTL3`) and places one model per
die automatically (`bestSingleGPUFit` + per-die VRAM accounting + a
`GGML_METAL_DEVICE_INDEX` export that pins each runner to its die).

Two copies of llama3.2 3B pinned to die 0 and die 1, raw `llama-server`:

| Scenario              | die0 gen t/s | die1 gen t/s |
|-----------------------|-------------:|-------------:|
| solo (one die idle)   |          102 |           94 |
| concurrent (both)     |          101 |           99 |

Concurrent throughput equals solo -> the dies run fully in parallel (~2x
aggregate). A shared die would roughly halve each.

Three *different* models, one `ollama serve`, served concurrently:

| Model        | size | die  | gen t/s (concurrent) |
|--------------|-----:|------|---------------------:|
| llama3.2 3B  | 2 GB | MTL0 |                 99.8 |
| devstral     |14 GB | MTL1 |                 19.5 |
| qwen3-coder  |18 GB | MTL2 |                 72.6 |

Placement is automatic: each `/api/generate` for a new model lands on the next
free die. Set `OLLAMA_MAX_LOADED_MODELS=4` to cap at one per die. Models that do
not fit one die (e.g. with very large context) fall back to the stock
spread/split path.

## Cross-die layer split (single model >32 GB)

A single model can be split across dies (`-sm layer` / `OLLAMA_SCHED_SPREAD=1`),
so models larger than one die's 32 GB run by using up to ~128 GB across 4 dies.
Cross-die tensor transfers are host-mediated by default (a Metal blit cannot
cross physical `MTLDevice`s unless you use a peer remote view, see Phase D).

**Methodology caveat (important):** raw `llama-server` defaults to **mmap on**,
which on a discrete GPU streams weights over PCIe every token and floors *every*
config at ~3 t/s — masking compute *and* any cross-die cost. Always pass
`--no-mmap` (Ollama does this automatically via the sched patch). An earlier
revision of this table compared an mmap'd raw split (~20 t/s) against a no-mmap
single-die number and wrongly concluded the split was "4-5x slower, PCIe-bound".
The corrected, consistent (`--no-mmap`) numbers below show the split overhead is
actually small.

llama3.2 3B Q4_K_M (2 GB blob), `--no-mmap`, raw `llama-server`, measured via
`/completion` (512-token prompt, `n_predict=128`, temp 0; pp/tg from
`timings.*_per_second`, best of 2):

| Split                                  | dies        | pp  | tg host | tg peer (`GGML_METAL_PEER_ENABLE`) |
|----------------------------------------|-------------|----:|--------:|-----------------------------------:|
| single die (`-dev MTL0`)               | 1           | 473 |    73.0 |                                —   |
| 2-die (`-dev MTL2,MTL3 -ts 1,1`)       | MTL2 + MTL3 | 494 |    68.7 |                               68.0 |
| 4-way (`-dev MTL0..MTL3 -ts 1,1,1,1`)  | MTL0..MTL3  | 539 |    51.0 |                               50.8 |

All splits produce coherent output (verified). Takeaways:
- This 3B model fits one die; the split here only measures *overhead*. Token
  generation falls ~73 → ~69 (2-die) → ~51 (4-way) t/s as the residual stream
  hops more dies (latency-bound handoffs). Use the split only for models that
  do not fit one die's ~32 GB.
- Prompt eval *rises* with more dies (473 → 494 → 539 t/s): prefill is
  compute-bound, so spreading the big prompt batch across dies adds throughput
  that outweighs the per-layer handoff. (Larger models like devstral instead
  saw pp fall at 4-way, where their bigger per-layer tensors make the handoff
  dominate — the direction depends on model size.)
- **Phase D (Infinity Fabric peer copy) is a tie with the host path** at every
  split (68.0 vs 68.7; 50.8 vs 51.0). The fabric copy is several-x faster as a
  bulk 256 MB transfer (~26 vs ~3-6 GB/s) but layer-split moves only the
  residual stream (~10-20 KB/token), so the split cost is submission/sync
  latency, not bandwidth. Phase D is kept opt-in (`GGML_METAL_PEER_ENABLE`);
  default is the host path. There is no Ollama/llama.cpp flag for Infinity Fabric.
- For models that fit one die, prefer one-model-per-die (full speed, see above).

## Larger models: gemma4 31B (dense) and qwen3.5 122B (MoE)

Same `llama-server` `/completion` harness (512-token prompt, `n_predict=128`,
temp 0, `--no-mmap`, best of 2). gemma4 uses `-c 4096`, qwen3.5 uses `-c 8192`.

**gemma4 31B Q4_K_M** (19 GB blob) — dense; fits one die:

| Config                                 | dies        | pp   | tg host | tg peer |
|----------------------------------------|-------------|-----:|--------:|--------:|
| single die (`-dev MTL0`)               | 1           | 46.4 |    11.4 |     —   |
| 2-die (`-ts 1,1`)                       | MTL2 + MTL3 | 47.0 |    11.1 |    11.1 |
| 4-way (`-ts 1,1,1,1`)                   | MTL0..MTL3  | 42.2 |    10.4 |    10.4 |

A 31B dense model is firmly bandwidth-bound on one W6800X die (~11 tg), so
spreading its layers across dies barely changes the per-token cost (11.4 → 11.1
→ 10.4 t/s); pp is flat too. Unlike the small llama3.2 (where the split cost was
visible), here both the work and the residual hops are small relative to the
weight traffic. Fabric ties the host path.

**qwen3.5 122B Q4_K_M** (81 GB blob) — MoE, ~few-B active; **cannot fit one die**
(needs ≥3 of the 4 × 32 GB dies):

| Config                                 | dies        | pp   | tg host | tg peer |
|----------------------------------------|-------------|-----:|--------:|--------:|
| 3-die (`-ts 1,1,1`)                     | MTL0..MTL2  | ~80  |    27.6 |    27.6 |
| 4-way (`-ts 1,1,1,1`)                   | MTL0..MTL3  | ~75  |    26.5 |    25.6 |

Despite being the largest model here, qwen3.5 122B generates *fastest* (~27 t/s)
because it is a Mixture-of-Experts: only a small fraction of the 122B params is
active per token. Three dies (96 GB) is the minimum that holds the 81 GB of
weights + KV; a fourth die adds capacity/headroom but a touch more handoff, so
tg dips slightly (27.6 → 26.5). pp on this model is noisy across server restarts
(~75–105 observed); tg is stable. Fabric ties the host path at both splits.

## Reference points from upstream (iRon-Llama README)

The fork's headline "~87 t/s" on the W6800X is a **Vulkan (MoltenVK)** number.
Its actual **Metal** result (Qwen3-30B-A3B Q4_K_M, MoE, ~3B active) is
pp512 ~272 t/s, tg128 ~72 t/s on the older b6123 backend. Our b9509 Metal
result (104 t/s gen on dense 3B) is in the same ballpark and exceeds the
fork's Metal tg.

## How to reproduce

Single-die quick check (CPU baseline + correctness gate):

```sh
scripts/bench.sh llama3.2   # uses the patched build under work/
```

The single-die and cross-die tables above use the patched `llama-server`
directly so split flags (`-dev`, `-sm layer`, `-ts`) can be set explicitly.
For each config: start the server, then read `timings` from `/completion`.

```sh
SRV=work/build-amd64/llama-server-local/bin/llama-server
BLOB=~/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff

# single die / 2-die / 4-die (add GGML_METAL_PEER_ENABLE=1 for the fabric column)
"$SRV" -m "$BLOB" --no-mmap -c 2048 --port 18099 --no-webui -dev MTL0 -ngl 99 &
"$SRV" -m "$BLOB" --no-mmap -c 2048 --port 18099 --no-webui -dev MTL2,MTL3      -sm layer -ts 1,1     -ngl 99 &
"$SRV" -m "$BLOB" --no-mmap -c 2048 --port 18099 --no-webui -dev MTL0,MTL1,MTL2,MTL3 -sm layer -ts 1,1,1,1 -ngl 99 &

curl -s localhost:18099/completion -d '{"prompt":"<~512 tokens>","n_predict":128,"temperature":0,"cache_prompt":false}' \
  | python3 -c 'import json,sys;d=json.load(sys.stdin)["timings"];print("pp",d["prompt_per_second"],"tg",d["predicted_per_second"])'
```

Record new runs as a dated section here: model, quant, die index, build pin
(`OLLAMA_VERSION` / `LLAMA_CPP_VERSION`), and the t/s table.
