# reference/iron-llama-b6123

Verbatim copies of the two modified Metal backend files from
[`Basten7/iRon-Llama`](https://github.com/Basten7/iRon-Llama) (MIT), kept here
for provenance and as the semantic source when forward-porting the AMD-Metal
kernels to the llama.cpp version Ollama pins.

- `ggml-metal.m`     - device init + kernel registration (AMD-friendly caps).
- `ggml-metal.metal` - threadgroup-tiled GEMM kernels (no Apple `simdgroup_mm`).
- `UPSTREAM-README.md` - the fork's own README (install + benchmarks).

## Provenance

- Upstream fork base: llama.cpp tag `b6123`.
- Install per the fork: check out llama.cpp `b6123`, drop these two files into
  `ggml/src/ggml-metal/`, build with Metal.
- These files apply to the `b6123` monolithic Metal backend layout. Ollama's
  pin (`LLAMA_CPP_VERSION` at the repo root) is newer and may use a refactored
  multi-file Metal backend and/or `MetalPerformancePrimitives`; do not copy
  these files blindly - port the semantics (see ../../docs/repatching.md).

## What changed vs stock b6123 (summary)

- Report `has_simdgroup_reduction` for `MTLGPUFamilyMetal3`/`Mac2`.
- Force `has_simdgroup_mm = false`.
- Enable `mul_mm`/`mul_mm_id` kernels when `!has_unified_memory`
  (`allow_mm_kernels = has_simdgroup_mm || !has_unified_memory`).
- Disable `flash_attn_ext` on AMD.
- Device selection via `MTLCopyAllDevices()` + `GGML_METAL_DEVICE_INDEX`.
- ~1200 lines of rewritten tiled GEMM shaders in `ggml-metal.metal`.

## License

MIT (upstream). Copyright retained by the iRon-Llama / llama.cpp authors.
