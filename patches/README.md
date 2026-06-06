# patches/

Git patches applied onto clean upstream checkouts by `scripts/apply-patch.sh`.
Never edit `work/` trees in place and commit them; capture changes here.

## Layout

- `patches/llama-cpp/` - applied to llama.cpp @ `LLAMA_CPP_VERSION`.
  - AMD-Metal kernel port (device caps, `mul_mm`/`mul_mm_id` threadgroup
    kernels, `flash_attn_ext` gating, `GGML_METAL_DEVICE_INDEX` selection).
  - x86_64 build un-gating for the Metal backend.
  - Multi-device backend registration (Phase B+).
- `patches/ollama/` - applied to Ollama @ `OLLAMA_VERSION`.
  - `scripts/build_darwin.sh` amd64 Metal build/install.
  - `discover/` Metal device + VRAM reporting on darwin/amd64.
  - scheduler placement for multi-die (Phase B+).

## Conventions

- Name files `NN-short-description.patch` (e.g. `01-amd-metal-device-caps.patch`).
- One logical change per patch.
- Regenerate with `git -C work/<tree> diff > patches/.../NN-....patch` (or
  `git format-patch`), then verify a clean `apply-patch.sh` run from scratch.
- Kernels are version-sensitive: prefer semantic forward-porting over textual
  application when bumping `LLAMA_CPP_VERSION` (see ../docs/repatching.md).

## Status

**Phase A complete** — correct + fast inference on the AMD Radeon Pro W6800X
(Intel Mac Pro), automatic by default. See `../docs/pa-port.md` for the full
root-cause analysis (the original wave64 theory was disproven by measurement).

**Phase B complete** — multi-die. A single `ollama serve` discovers all 4 dies
(`MTL0..MTL3`, 128 GiB total) and places one model per die automatically;
validated 3 different models served concurrently on 3 dies. See
`../docs/benchmarks.md` (Multi-die section).

`patches/llama-cpp/`:
- `01-metal-concurrency-off-nonuma.patch` — **correctness.** Default
  `use_concurrency = false` on non-UMA (discrete) Metal devices. The backend's
  concurrent command-buffer dispatch corrupts intermediate tensors on discrete
  AMD GPUs (every op passes `test-backend-ops` individually).
- `02-metal-discrete-device.patch` — **device adaptations** (`ggml-metal-device.m`):
  Mac2 reduction caps (no-op on W6800X, helps older discrete GPUs); bounce-buffer
  fallback in `set_tensor`/`get_tensor` for unaligned host pointers (stock
  asserts + aborts on discrete GPUs); and **multi-die selection** — `device_init`
  binds the ggml device to `MTLCopyAllDevices()[idx]` (idx from the ggml slot or
  `GGML_METAL_DEVICE_INDEX`) instead of always the system default device.
- `03-metal-tiled-mul-mm.patch` — **perf (optional).** Threadgroup-tiled
  `mul_mm`/dispatch for non-UMA; ~12x faster prompt eval. NOT required for
  correctness (stock `mul_mv` is numerically correct here too).
- `04-metal-multi-die-registration.patch` — **multi-die.** Default the
  registered backend-device count to the physical Metal device count
  (`ggml_metal_device_count()`), so dual W6800X Duo cards expose `MTL0..MTL3`.
  `GGML_METAL_DEVICE_INDEX` pins a process to one die (count = 1);
  `GGML_METAL_DEVICES` still overrides. Apple Silicon / single-GPU = 1
  (unchanged).

`patches/ollama/`:
- `0001-ungate-metal-amd64.patch` — build the Metal backend on x86_64 macOS.
- `0002-discover-metal-amd64.patch` — Metal device + VRAM reporting on
  darwin/amd64.
- `0003-sched-disable-mmap-discrete-metal.patch` — **perf.** Disable mmap on
  darwin + discrete GPU so weights load into private VRAM instead of being
  streamed over PCIe each token (1.6 -> ~104 t/s generation on llama3.2).
- `0004-multi-die-discovery-and-pinning.patch` — **multi-die one-model-per-die.**
  (a) `discover/llama_server.go`: classify `MTL<n>`-named devices as the `Metal`
  library (discrete AMD descriptions contain neither "metal" nor "apple"), so
  `ByLibrary` grouping and pinning work. (b) `ml/device.go`: export
  `GGML_METAL_DEVICE_INDEX=<id>` for a single-device Metal placement, pinning the
  runner to that physical die. (c) `llm/llama_server.go`: attribute a pinned
  runner's whole GPU footprint to its sole device in `VRAMByGPU` (a pinned
  Metal runner labels buffers `MTL0` regardless of die, so the per-name lookup
  would otherwise return 0 and the scheduler would double-book one die).

The proven kernels referenced when porting live in
`../reference/iron-llama-b6123/`.
