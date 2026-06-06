# Repatching strategy

The goal is to make tracking upstream Ollama cheap and deterministic. All of
our changes live in this repo as patches + scripts; the upstream trees are
cloned fresh and patched, never edited in place and committed.

## Sources of truth

- `LLAMA_CPP_VERSION` - llama.cpp tag we patch (must equal Ollama's own file).
- `OLLAMA_VERSION` - Ollama tag + commit we integrate with.
- `patches/llama-cpp/*.patch` - applied to llama.cpp @ `LLAMA_CPP_VERSION`.
- `patches/ollama/*.patch` - applied to Ollama @ `OLLAMA_VERSION`.
- `reference/iron-llama-b6123/` - verbatim proven kernels (base `b6123`), the
  semantic reference when forward-porting to a newer llama.cpp.

## Routine: bump to a new Ollama release

1. Pick the new Ollama tag; set it in `OLLAMA_VERSION` (tag + full commit).
2. Read that release's `LLAMA_CPP_VERSION`; copy the value into ours.
3. `scripts/bootstrap.sh` to clone both at the new pins into `work/`.
4. `scripts/apply-patch.sh` which runs `git apply --3way`:
   - If patches apply cleanly: build + run the validation gates.
   - If they reject (kernels are version-sensitive): forward-port semantically
     (see below), regenerate the patch with `git -C work/llama.cpp diff`, and
     replace the file under `patches/llama-cpp/`.
5. Build (`scripts/build.sh`) and validate (`scripts/bench.sh`): both
   correctness and speed gates must pass.
6. Commit the updated pins + regenerated patches together.

## Forward-porting the kernels (semantic, not textual)

The `.metal`/`.m` kernels drift between llama.cpp versions, so prefer porting
the INTENT over applying a textual diff. The invariant changes are:

- Device init (`ggml-metal.m` / its refactored equivalent):
  - `has_simdgroup_reduction |= supportsFamily(MTLGPUFamilyMetal3 | Mac2)`
  - force `has_simdgroup_mm = false`
  - kernel gate: `allow_mm_kernels = has_simdgroup_mm || !has_unified_memory`
  - `allow_flash_attn_ext = has_unified_memory && has_simdgroup_mm`
  - device selection via `MTLCopyAllDevices()` + `GGML_METAL_DEVICE_INDEX`
- Shaders (`ggml-metal.metal`): ensure the threadgroup-tiled `mul_mm` /
  `mul_mm_id` kernels exist and are registered for all required quant types
  (Q4_K, Q6_K, etc.). Diff `reference/iron-llama-b6123/ggml-metal.metal`
  against upstream `b6123` to see exactly which kernels were rewritten, then
  apply the equivalent rewrite to the new shader file.
- Watch for newer upstream using `MetalPerformancePrimitives` cooperative
  tensor matmul (the `bfloat`/`half` static_assert path that fails on
  non-Apple GPUs). Disable it on AMD alongside the simdgroup-mm disable.

## Build un-gating (Ollama side)

- Ensure the Metal backend target is built and installed for `x86_64` in
  `scripts/build_darwin.sh` (amd64 branch) and any CMake guard that restricts
  Metal/metallib install to arm64.
- Add CGO link flags `-framework Metal -framework Foundation` for the amd64
  build if missing.

## Discovery / scheduler (Ollama side)

- Make `discover/` report the Metal device(s) on `darwin/amd64` with VRAM from
  `recommendedMaxWorkingSetSize` (~34 GB/die) so the scheduler offloads layers.
- For multi-die, report each die as a distinct GPU and rely on
  `OLLAMA_SCHED_SPREAD` / `OLLAMA_MAX_LOADED_MODELS` for placement.

## Safety net

The standalone llama.cpp `llama-server` (built from the same patched tree)
always provides a working ~87 t/s OpenAI-compatible endpoint, independent of
Ollama. Keep it buildable so a broken Ollama bump never leaves the machine
without GPU inference.

## What NOT to do

- Do not commit cloned Ollama/llama.cpp trees (they live in git-ignored
  `work/`).
- Do not hardcode tags in scripts; read `LLAMA_CPP_VERSION` / `OLLAMA_VERSION`.
- Do not run `sudo` from scripts; print the command for the user instead.
