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

**Phase C complete** — cross-die layer split for single models >32 GB. The
Metal backend's cross-device tensor copies are host-mediated on discrete dies
(a Metal blit cannot cross physical `MTLDevice`s by default), so `-sm layer` /
`OLLAMA_SCHED_SPREAD` splits a model across dies correctly (validated coherent
on a forced 2-die and an Ollama 4-die spread). The split overhead is small:
~75 t/s single-die vs ~70 t/s on a 2-die layer split (devstral 24B, weights
resident in VRAM), the trade for capacity up to ~128 GB.

**Phase D (opt-in, not a perf win)** — Infinity Fabric peer copy. All four
W6800X dies share one Metal peer group (`peerGroupID`, `peerCount=4`), so
cross-die copies can go directly over the AMD Infinity Fabric Link via a remote
buffer view (`newRemoteBufferViewForDevice:`) instead of through the host.
Verified correct and ~9x faster as a raw transfer (~26 vs ~3 GB/s for 256 MB),
**but it yields no measurable inference speedup** (70.1 peer vs 70.3 host t/s on
the 2-die split): layer-split cross-die copies are tiny (residual stream,
~10-20 KB/token) so the small split overhead is submission/sync latency, not
copy bandwidth. Kept inert by default; enable with `GGML_METAL_PEER_ENABLE` for
copy-heavy splits or experimentation. (There is no Ollama or llama.cpp flag for
Infinity Fabric; the only stock P2P knobs are CUDA-specific.)

`patches/llama-cpp/`:
- `01-metal-context-concurrency-and-cross-die.patch` (`ggml-metal-context.m`):
  - **correctness.** Default `use_concurrency = false` on non-UMA (discrete)
    Metal devices. The backend's concurrent command-buffer dispatch corrupts
    intermediate tensors on discrete AMD GPUs (every op passes
    `test-backend-ops` individually).
  - **cross-die copy.** `ggml_metal_cpy_tensor_async` returns false when src and
    dst are on different physical `MTLDevice`s (the async path also relies on a
    single-device event signal/wait), so ggml falls back to the synchronous
    `ggml_metal_buffer_cpy_tensor`. Required for multi-die layer split.
- `02-metal-discrete-device.patch` — **device adaptations** (`ggml-metal-device.m`):
  Mac2 reduction caps (no-op on W6800X, helps older discrete GPUs); bounce-buffer
  fallback in `set_tensor`/`get_tensor` for unaligned host pointers (stock
  asserts + aborts on discrete GPUs); **multi-die selection** — `device_init`
  binds the ggml device to `MTLCopyAllDevices()[idx]` (idx from the ggml slot or
  `GGML_METAL_DEVICE_INDEX`) instead of always the system default device; the
  **cross-die copy** in `ggml_metal_buffer_cpy_tensor`: host-mediated by default
  (Phase C), or — when `GGML_METAL_PEER_ENABLE` is set and both dies share a
  Metal peer group — a direct Infinity Fabric peer blit via a remote buffer view
  (Phase D, opt-in, see status above); and the opt-in AMD FA-vec gate
  (`ggml_metal_set_force_fa_vec`, `GGML_METAL_AMD_FA`).
- `03-metal-tiled-mul-mm.patch` — **perf (optional).** Threadgroup-tiled
  `mul_mm`/dispatch for non-UMA (`kernels/mul_mm.metal` +
  `ggml_metal_op_mul_mat_use_mm`); ~12x faster prompt eval. NOT required for
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
