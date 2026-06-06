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

## What this patch set does

Getting these GPUs useful took four increments. Each is a small, isolated patch
(see [`patches/`](patches/) and [`docs/architecture.md`](docs/architecture.md)):

- **Phase A — correct + fast single-die.** Two root causes, both wrongly
  blamed on the kernels at first:
  - *Correctness.* The Metal backend's **concurrent command-buffer dispatch**
    corrupts intermediate tensors on discrete AMD GPUs (every op passes
    `test-backend-ops` individually; only the concurrent graph is wrong). We
    default `use_concurrency = false` on non-UMA devices.
  - *Performance.* Ollama assumed Metal ⇒ unified memory and left weights
    **mmap'd in host RAM**, so the GPU streamed them over PCIe every token
    (~2 t/s). Disabling mmap on discrete Metal loads weights into VRAM
    (~65× generation speedup). The iRon-Llama threadgroup-tiled `mul_mm` adds
    ~12× prompt-eval on top.
- **Phase B — multi-die, one model per die.** The dual W6800X Duo cards are
  **4 separate GPU dies**. Stock ggml-metal always bound to the system default
  device; we map each ggml device to a distinct `MTLCopyAllDevices()[i]`, teach
  Ollama's discovery to recognise the `MTL0..MTL3` dies, pin a runner to one die
  (`GGML_METAL_DEVICE_INDEX`), and fix per-die VRAM accounting. One
  `ollama serve` now runs a different model on each die (~128 GB total).
- **Phase C — cross-die layer split for models >32 GB.** A Metal blit can't
  cross physical `MTLDevice`s, so cross-die tensor copies fall back to a
  host-mediated path (bounce-buffered `get`/`set`). `-sm layer` /
  `OLLAMA_SCHED_SPREAD` then split one model across dies. Overhead is modest:
  ~75 → ~70 → ~60 t/s for 1 / 2 / 4-die splits.

### Infinity Fabric (Phase D, opt-in)

The two W6800X Duo MPX modules are linked by **AMD Infinity Fabric**, and macOS
exposes it: all four dies report an identical **Metal peer group**
(`peerGroupID`, `peerCount = 4`). That lets a die read another die's VRAM
directly over the fabric via a *remote buffer view*
(`newRemoteBufferViewForDevice:`) instead of bouncing through host RAM.

We implemented it and measured it honestly:

- It is **correct** and **~9× faster as a raw transfer** (~26 vs ~3 GB/s for a
  256 MB block).
- But it gives **no measurable inference speedup** (70.1 peer vs 70.3 host t/s
  on a 2-die split). Layer-split inference only moves the residual stream across
  the boundary (~10–20 KB/token), so the cost is submission/sync **latency, not
  bandwidth** — the fabric's bandwidth has nothing to bite on.

So the peer copy is kept **off by default** and enabled with
`GGML_METAL_PEER_ENABLE` (useful for copy-heavy splits or experimentation).
Note: there is **no Ollama or llama.cpp flag for Infinity Fabric** — the only
stock peer-to-peer knobs are CUDA-specific; Phase D is the only way it is
exercised here.

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
