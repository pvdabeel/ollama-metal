# pa-port: AMD-Metal kernel port (b6123 fork -> b9509)

Root-cause analysis and porting plan for making Ollama's pinned llama.cpp
(`b9509`) produce **correct, fast** inference on the AMD Radeon Pro W6800X
under Metal on Intel macOS.

## Validated symptom (control)

With `pa-ungate` + `pa-discover` applied (stock b9509 Metal kernels), Ollama
on MacPro7,1 offloads `llama3.2` 29/29 layers to the W6800X and generates at
**~2.3 tok/s with garbage output** (e.g. `KG [aireoviesorta coercquote...`).

So the GPU path works end-to-end; the *kernels* are both slow and numerically
wrong on this GPU.

## Root cause

AMD GCN/RDNA GPUs execute in **64-wide wavefronts** (the iRon-Llama fork's own
logs report `warp size: 64` for the W6800X). llama.cpp's Metal kernels assume
**32-wide SIMD groups**:

```
ggml/src/ggml-metal/ggml-metal.metal:28
#define N_SIMDWIDTH 32 // assuming SIMD group size is 32
```

Two consequences on the W6800X (`has_simdgroup_mm=false`, `has_unified_memory=false`,
`has_simdgroup_reduction=true`):

1. `GGML_OP_MUL_MAT` is claimed supported via `has_simdgroup_reduction`
   (`ggml-metal-device.m:1246-1248`) and, because `has_simdgroup_mm=false`,
   dispatches to the **vector `mul_mv`** path (`ggml-metal-ops.cpp:2167/2322`).
   Those vector kernels reduce across `N_SIMDWIDTH=32` lanes; on a 64-wide
   wavefront the simd-reduction spans the wrong number of lanes -> wrong sums
   -> garbage.
2. `GGML_OP_FLASH_ATTN_EXT` already returns `has_simdgroup_mm`
   (`ggml-metal-device.m:1236`), so it is *already disabled* on AMD and is NOT
   the source of the garbage. (Ollama also defaults `OLLAMA_FLASH_ATTENTION=false`.)

This is why a caps-only change cannot fix correctness: the vector path itself
is unsafe on wave64. The fork sidesteps it by routing matmuls through a
**threadgroup-tiled `mul_mm`** that does not depend on a 32-wide reduction or
on Apple's `simdgroup_float8x8` matrix intrinsics (which the W6800X lacks).

## What the iRon-Llama fork changes (vs stock b6123)

Caps / dispatch (`ggml-metal.m`):
- `has_simdgroup_reduction |= supportsFamily(Mac2)` (cover Intel Macs).
- When `!hasUnifiedMemory`: force `has_simdgroup_mm = false` (avoid compiling
  the Apple matrix MM/flash kernels).
- `allow_mm_kernels    = has_simdgroup_mm || !has_unified_memory`  (TRUE on AMD).
- `allow_flash_attn_ext = has_unified_memory && has_simdgroup_mm`  (FALSE on AMD).
- Register every `MUL_MM_*` / `MUL_MM_ID_*` kernel with `allow_mm_kernels`
  instead of `has_simdgroup_mm`.
- Device selection via `MTLCopyAllDevices()` + `GGML_METAL_DEVICE_INDEX`.

Kernels (`ggml-metal.metal`, ~1200 lines):
- Replacement threadgroup-tiled GEMM for `kernel_mul_mm` / `kernel_mul_mm_id`
  that loads A/B tiles into `threadgroup` memory with explicit
  `threadgroup_barrier`, accumulating per-thread without 32-lane simd
  reductions or `simdgroup_float8x8`.

## b9509 integration points (refactored, multi-file backend)

| Concern              | b6123 (fork)             | b9509 (target)                                  |
|----------------------|--------------------------|-------------------------------------------------|
| device caps          | `ggml-metal.m`           | `ggml-metal-device.m` (~696-723, 1052)          |
| op support gate      | `ggml-metal.m`           | `ggml-metal-device.m:ggml_metal_device_supports_op` (1051) |
| mm-vs-mv dispatch    | `ggml-metal.m`           | `ggml-metal-ops.cpp:2167, 2322`                 |
| kernel arg structs   | inline                   | `ggml-metal-impl.h`                             |
| shaders              | `ggml-metal.metal`       | `ggml-metal.metal` (452 KB; templated kernels)  |

b9509's `kernel_mul_mm` is templated and uses `simdgroup_load` /
`simdgroup_multiply_accumulate` / `make_filled_simdgroup_matrix<,8>`. The port
must add an AMD-safe tiled variant and dispatch to it when
`!has_unified_memory` (new `allow_mm_kernels` semantics), keeping the Apple
simdgroup path for arm64.

## Porting plan

1. **caps** (`pa-port-caps`): in `ggml-metal-device.m`, add
   `props.allow_mm_kernels = has_simdgroup_mm || !has_unified_memory` and
   `props.use_simdgroup_mm = has_simdgroup_mm` (Apple-matrix path only). Gate
   `MUL_MAT`/`MUL_MAT_ID` mm-path on `allow_mm_kernels`; keep `FLASH_ATTN_EXT`
   on `has_simdgroup_mm`. Mirror the gate in `ggml-metal-ops.cpp:2167/2322`.
2. **kernel** (`pa-port-kernel`): add `kernel_mul_mm_tiled` (+ `_id`) adapted to
   b9509's arg struct; dispatch when `!use_simdgroup_mm`.
3. **build** (`pa-port-build`): `BUILD_MODE=local` against the patched
   `work/llama.cpp` (Ollama compat patch applied first, then
   `-DOLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON`).
4. **validate** (`pa-port-validate`): `llama3.2` must produce coherent text and
   beat the ~30 t/s CPU baseline (fork reference: ~87 t/s tg on one die).

## Open risks

- b9509 templating: the tiled kernel must satisfy the same template
  instantiations (per quant type) the dispatch expects, or dispatch must be
  special-cased for the AMD path.
- `threads_per_threadgroup` / tile sizes tuned for wave64 (fork uses 64-wide
  friendly tiles; see `GGML_METAL_N_CB`, `-ub/-b` sweet spots in
  reference/iron-llama-b6123/UPSTREAM-README.md).
