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
this.

<table>
<tr>
<td width="44%" valign="top">
<img src="docs/images/mac-pro-w6800x-duo.png" width="100%" alt="Mac Pro (2019) with two AMD Radeon Pro W6800X Duo MPX modules = 4 GPU dies">
<sub>Mac Pro (2019), two Radeon Pro W6800X Duo MPX modules &mdash; <b>4 GPU dies</b>, ~32&nbsp;GB each, non-UMA, linked by Infinity Fabric.</sub>
</td>
<td valign="top">

**Single die** (llama-3.2-3B Q4_K_M):

<table>
<tr><th align="left">Backend</th><th align="right">pp</th><th align="right">tg</th></tr>
<tr><td>Stock llama.cpp Metal</td><td align="right">2.3</td><td align="right">2.0</td></tr>
<tr><td>CPU (28-thread Xeon W)</td><td align="right">222</td><td align="right">29.9</td></tr>
<tr><td>iRon-Llama patched Metal</td><td align="right"><b>339</b></td><td align="right"><b>87</b></td></tr>
</table>

**Cross-die layer split** (devstral 24B, VRAM-resident):

<table>
<tr><th align="left">Config</th><th align="right">tg</th></tr>
<tr><td>single die</td><td align="right">75</td></tr>
<tr><td>2-die split</td><td align="right">70</td></tr>
<tr><td>4-way split</td><td align="right">60</td></tr>
</table>

<sub>t/s @ temp 0. Full data: <a href="docs/benchmarks.md">docs/benchmarks.md</a></sub>

</td>
</tr>
</table>

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
- [x] Phase A: single-die AMD Metal inside Ollama (un-gate + kernel port +
      discovery); correct + fast by default.
- [x] Phase B: multi-die concurrent serving (one model per die); all 4 dies
      discovered as `MTL0..MTL3` (~128 GB total).
- [x] Phase C: single-model layer split across dies for >32 GB (host-mediated).
- [x] Phase D (opt-in): Infinity Fabric peer copy for cross-die transfers
      (`GGML_METAL_PEER_ENABLE`); correct, but no inference speedup vs Phase C.

## License

MIT. Derives from MIT-licensed llama.cpp, Ollama, and iRon-Llama; their
attributions are retained (see `LICENSE` and
`reference/iron-llama-b6123/UPSTREAM-README.md`).
