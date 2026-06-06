# ollama-metal

GPU-accelerated GGUF inference in **Ollama** on **Intel Macs with discrete AMD
(Metal 3) GPUs**, plus multi-GPU (multi-die) support.

This is a small, re-appliable **patch set** layered on pinned upstream versions
of Ollama and llama.cpp - not a fork. It ports the AMD-friendly Metal kernels
from [`Basten7/iRon-Llama`](https://github.com/Basten7/iRon-Llama) onto the
llama.cpp version Ollama pins, re-enables the Metal backend for `x86_64`, and
teaches Ollama's GPU discovery/scheduler about the AMD dies.

## Why

On an Intel Mac Pro with AMD Radeon Pro W6800X GPUs, stock Ollama runs GGUF
models CPU-only:
- Ollama's Mac GPU path moved to MLX (Apple Silicon only); its ggml Metal
  backend is gated to arm64.
- Even stock llama.cpp Metal is ~2 t/s on these GPUs (no `simdgroup_matrix`).

The iRon-Llama kernels (threadgroup-tiled GEMM, `has_simdgroup_mm=false`) fix
this. Measured on the target machine (one W6800X die, llama-3.2-3B Q4_K_M):

| Backend                    | Prompt (pp) | Generation (tg) |
|----------------------------|------------:|----------------:|
| Stock llama.cpp Metal      |     2.3 t/s |         2.0 t/s |
| CPU (28-thread Xeon W)     |     222 t/s |        29.9 t/s |
| iRon-Llama patched Metal   | **339 t/s** |     **87 t/s**  |

See [docs/benchmarks.md](docs/benchmarks.md) for full data.

## Target hardware (only supported config)

- Mac Pro 2019 (MacPro7,1), Intel Xeon W, macOS 26.x
- 2x AMD Radeon PRO W6800X Duo = 4 dies, ~32 GB each, non-UMA
- Xcode + Metal, Go (MacPorts), MacPorts cmake/ninja/git

## Layout

```
LLAMA_CPP_VERSION   pinned upstream llama.cpp tag we patch (mirrors Ollama)
OLLAMA_VERSION      pinned Ollama tag + commit we integrate with
.cursorrules        architecture + repatching strategy (read this first)
docs/               architecture.md, repatching.md, benchmarks.md
patches/            git patches: patches/llama-cpp/ and patches/ollama/
reference/          verbatim iRon-Llama b6123 kernels (MIT) for provenance/port
scripts/            env / bootstrap / apply-patch / build / bench automation
bench/              curated benchmark results
```

## Quick start (once patches land)

```sh
scripts/bootstrap.sh      # clone Ollama + llama.cpp at the pinned versions
scripts/apply-patch.sh    # apply patches/ onto the work/ checkouts
scripts/build.sh          # build Ollama against the patched llama.cpp tree
scripts/bench.sh llama3.2 # correctness + speed vs CPU baseline
```

## Status

- [x] Phase 0: pins located, integration point identified
      (`OLLAMA_LLAMA_CPP_SOURCE` override), reference kernels vendored.
- [ ] Phase A: single-die AMD Metal inside Ollama (un-gate + kernel port +
      discovery).
- [ ] Phase B: multi-die concurrent serving (one model per die).
- [ ] Phase C (optional): single-model layer split across dies for >32 GB.

## License

MIT. Derives from MIT-licensed llama.cpp, Ollama, and iRon-Llama; their
attributions are retained (see `LICENSE` and
`reference/iron-llama-b6123/UPSTREAM-README.md`).
