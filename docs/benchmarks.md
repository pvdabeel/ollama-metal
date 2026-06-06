# Benchmarks

All numbers measured on the target machine:
- Mac Pro 2019 (MacPro7,1), Intel Xeon W-3275M (28 cores), macOS 26.5.1
- AMD Radeon PRO W6800X Duo (one die used), `recommendedMaxWorkingSetSize`
  ~34 GB, `simdgroup matrix mul. = false`, `has unified memory = false`
- Model: `llama3.2` 3B Q4_K_M (reused from the local Ollama blob store)
- Tool: `llama-bench` (`-p 64 -n 32 -r 2`)

## Single-die: backend comparison

| Backend                              | pp (prompt) t/s | tg (generation) t/s |
|--------------------------------------|----------------:|--------------------:|
| Stock llama.cpp Metal (master)       |            2.30 |                2.07 |
| Stock Metal + env tuning (N_CB, ub)  |            2.51 |                2.17 |
| CPU (28 threads, `-ngl 0`)           |          222.14 |               29.85 |
| **iRon-Llama patched Metal** (b6123) |      **338.74** |           **87.15** |

Tuning used for the patched run: `GGML_METAL_N_CB=4`, `-fa 0 -ub 32 -b 32`,
`GGML_METAL_DEVICE_INDEX=0`.

Correctness: the patched build produces coherent text (verified with a
"why is the sky blue" prompt) - not the `@@@@`/NaN garbage seen on the stock
Metal discrete-GPU path.

## Takeaways

- The patched Metal backend is ~165x faster than stock Metal on this card and
  ~3x faster than the CPU for generation.
- The win comes from the kernels (threadgroup-tiled GEMM), not env tuning:
  env tuning alone left stock Metal at ~2.5 t/s.
- One die (~32 GB) easily fits 3B-30B (A3B) class models; multi-die is for
  capacity beyond 32 GB and concurrent multi-model throughput.

## Reference points from upstream (iRon-Llama README / discussions)

On the same W6800X family, the fork reports (Qwen3-30B-A3B Q4_K_M, MoE):
pp512 ~250-310 t/s, tg128 ~60-85 t/s; dense Qwen3-4B Q4_0: pp512 ~331,
tg128 ~105. These corroborate our measured 3B numbers.

## How to reproduce

```sh
scripts/bench.sh llama3.2   # uses the patched build under work/
```

Record new runs as a dated section here: model, quant, die index, build pin
(`OLLAMA_VERSION` / `LLAMA_CPP_VERSION`), and the t/s table.
