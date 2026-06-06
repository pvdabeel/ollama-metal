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

`patches/llama-cpp/`:
- `01-metal-concurrency-off-nonuma.patch` — **correctness.** Default
  `use_concurrency = false` on non-UMA (discrete) Metal devices. The backend's
  concurrent command-buffer dispatch corrupts intermediate tensors on discrete
  AMD GPUs (every op passes `test-backend-ops` individually).
- `02-metal-discrete-caps-and-bounce-buffer.patch` — **robustness.** Mac2
  reduction caps (no-op on W6800X, helps older discrete GPUs) + bounce-buffer
  fallback in `set_tensor`/`get_tensor` for unaligned host pointers (stock
  asserts + aborts on discrete GPUs).
- `03-metal-tiled-mul-mm.patch` — **perf (optional).** Threadgroup-tiled
  `mul_mm`/dispatch for non-UMA; ~12x faster prompt eval. NOT required for
  correctness (stock `mul_mv` is numerically correct here too).

`patches/ollama/`:
- `0001-ungate-metal-amd64.patch` — build the Metal backend on x86_64 macOS.
- `0002-discover-metal-amd64.patch` — Metal device + VRAM reporting on
  darwin/amd64.
- `0003-sched-disable-mmap-discrete-metal.patch` — **perf.** Disable mmap on
  darwin + discrete GPU so weights load into private VRAM instead of being
  streamed over PCIe each token (1.6 -> ~104 t/s generation on llama3.2).

The proven kernels referenced when porting live in
`../reference/iron-llama-b6123/`.
