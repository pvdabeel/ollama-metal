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

Empty until Phase A. The proven kernels to port live in
`../reference/iron-llama-b6123/`.
