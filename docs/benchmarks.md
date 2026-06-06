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

## Reference points from upstream (iRon-Llama README)

The fork's headline "~87 t/s" on the W6800X is a **Vulkan (MoltenVK)** number.
Its actual **Metal** result (Qwen3-30B-A3B Q4_K_M, MoE, ~3B active) is
pp512 ~272 t/s, tg128 ~72 t/s on the older b6123 backend. Our b9509 Metal
result (104 t/s gen on dense 3B) is in the same ballpark and exceeds the
fork's Metal tg.

## How to reproduce

```sh
scripts/bench.sh llama3.2   # uses the patched build under work/
```

Record new runs as a dated section here: model, quant, die index, build pin
(`OLLAMA_VERSION` / `LLAMA_CPP_VERSION`), and the t/s table.
