# Flash Attention on discrete AMD (Metal) — status & findings

Goal: run attention on the GPU (instead of the CPU fallback) on the AMD
W6800X dies, which unlocks the biggest documented wins on this hardware:
long-context decode throughput and a **GPU-resident quantized KV cache**
(q8_0/q4_0), as reported by the `iRon-Llama-RC2` (MIT) and `ToshLLM` (GPL-3.0)
projects. This is patch `patches/llama-cpp/05-metal-amd-flash-attn-vec.patch`.

## Background: why FA is on the CPU by default

`llama.cpp`'s Metal backend has two flash-attention kernels:

- the **`mm`** (matrix) kernel — uses `simdgroup_matrix` (`simdgroup_half8x8`),
  which discrete AMD RDNA2 GPUs do **not** support (`has_simdgroup_mm = false`),
  so it crashes there; and
- the **`vec`** kernel — reduction-only (`simd_shuffle_down` / `simd_max` /
  `simd_sum`), needs only `has_simdgroup_reduction`, which the W6800X **has**.

Upstream `ggml_metal_device_supports_op()` gates `FLASH_ATTN_EXT` on
`has_simdgroup_mm` (with a literal `// TODO: over-restricted for vec-kernels`).
On AMD that returns false, so `--flash-attn auto` disables FA and attention
runs correctly on the CPU — but generation collapses at long context and any
quantized KV cache is forced onto the CPU.

## What patch 05 does

1. **Gate (device.m):** when `!has_simdgroup_mm && has_simdgroup_reduction`,
   report `FLASH_ATTN_EXT` as supported for the shapes the **vec** kernel is
   actually instantiated for in `b10091` — head dims `dk ∈
   {32,64,96,128,192,256,320,512,576}` with matching `dv` (only the MLA cases
   192→128, 320→256, 576→512 differ). All vec KV types (f16/bf16/q4_0/q4_1/
   q5_0/q5_1/q8_0) are covered upstream. Shapes outside that set keep the CPU
   fallback (no regression).
2. **Force the vec path (ops.cpp `..._use_vec`):** on such devices the `mm`
   kernel can't run, so the vec kernel must serve every batch size (prefill
   included), not just small decode batches.
3. **Single-workgroup dispatch:** default the vec dispatch to `nwg=1` (direct
   write to `dst`, no `vec_reduce` combine pass), with `GGML_METAL_FA_VEC_NSG` /
   `GGML_METAL_FA_VEC_NWG` runtime knobs (mirroring `iRon-Llama-RC2`).

No Ollama patch is needed: `ml.FlashAttentionSupported` already treats `Metal`
as FA-capable, so `--flash-attn auto` picks up the new `supports_op` result.
`OLLAMA_FLASH_ATTENTION=1` forces `--flash-attn on`.

## Current result: fast but INCORRECT — hence opt-in and OFF by default

With the gate enabled, FA runs **on the GPU** (`flash_attn = enabled`, all
layers offloaded, no "assigned to device CPU" warning) at ~**106 t/s** vs
~**79 t/s** for the non-FA baseline on Llama-3.2-3B (die 0) — but the **output
is garbage**.

Isolation performed (all still garbage):

- `nwg=1` (no cross-workgroup `vec_reduce` combine),
- `nsg=1` (no cross-simdgroup threadgroup-memory reduction),
- 1-token prompt (`NQPSG=1`, so prefill and decode take the identical
  per-query path).

So the fault is in the **core per-simdgroup online-softmax computation** of
`b10091`'s vec kernel on RDNA2 — not the combine/reduce passes. This matches
ToshLLM's note that "each SIMD primitive is correct in isolation but the vec
kernel miscompiles on RDNA2". Because it would corrupt output, the path is
**opt-in and OFF by default**:

- default (unset): FA on AMD → CPU fallback → correct, matches the validated
  bump (Llama-3.2-3B: ~79 t/s tg, ~180 t/s pp on one die);
- `GGML_METAL_AMD_FA=1`: enable the GPU vec path for kernel-correctness work;
  `GGML_METAL_FA_VEC_NSG` / `_NWG` tune the simdgroup/workgroup split.

## Next step (kernel correctness)

Port the proven **`iRon-Llama-RC2`** vec flash-attention kernel (MIT — license
compatible with this repo; do **not** copy from GPL-3.0 ToshLLM) onto
`b10091`'s newer function-constant infra
(`FC_flash_attn_ext_vec_nsg/nwg/ns10/ns20`). RC2's kernel is
`NSG`-templated with a different reduction layout and is validated bit-exact on
this exact W6800X hardware. Reference kept at `work/iron-rc2` during
development (base commit `4686a7095`, "CC_V26_mgpu-quant-stable").

When correct, expected wins (from the reference projects): long-context decode
+17–75%, and a GPU-resident quantized KV cache (q8_0 ≈ halves KV footprint with
no tg regression).
