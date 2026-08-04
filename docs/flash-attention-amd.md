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

## Current result: fast but INCORRECT — hence opt-in and OFF by default

With the gate enabled (`GGML_METAL_AMD_FA=1` + `--flash-attn on`), FA runs **on
the GPU** at ~**82–86 t/s** on Llama-3.2-3B (die 0) — but the **output is
garbage** (e.g. `"##_ _{. this was not only have been said…"`).

Isolation performed (all still garbage):

- `nwg=1` (no cross-workgroup `vec_reduce` combine),
- `nsg=1` (no cross-simdgroup threadgroup-memory reduction),
- 1-token prompt (`NQPSG=1`, so prefill and decode take the identical
  per-query path).

So the fault is in the **core per-simdgroup online-softmax computation** of
`b10091`'s vec kernel on RDNA2 — not the combine/reduce passes.

## Attempted fix (reverted): NSG templatization

The `iRon-Llama-RC2` vec kernel differs from `b10091`'s mainly in that it makes
`NSG` (simdgroups per threadgroup) a **compile-time template parameter** instead
of the `FC_flash_attn_ext_vec_nsg` **function constant**, via a thin
`switch (nsg)` wrapper that dispatches to an `NSG`-specialized `_impl`. The
hypothesis was that AMD's Metal compiler mis-propagates the function-constant
`NSG` into threadgroup-array sizing and reduction loops.

This was ported onto `b10091` and **did not fix correctness** (still garbage at
`nsg=1`), while it **tripled** the FA-vec shader instantiations (cases 1/2/4 ×
every head-dim/type), pushing the *cold* Metal-library compile past Ollama's
30 s GPU-discovery watchdog (→ silent CPU fallback for a fresh install). Net
negative, so it was reverted.

Further isolation ruled out the obvious suspects: the `pad` helper kernel is
**byte-identical** to RC2's; the `blk` (mask-block-skip) kernel differs but is
**only used by the non-vec path**; and the ordinary Metal softmax/attention
reductions (same `N_SIMDWIDTH=32`, `simd_max`/`simd_sum`) are **correct** on
this hardware (the non-FA path is coherent), so it is not a naive wave-width
bug. At `nsg=1,nqptg=1` the b10091 and RC2 vec kernels are semantically
equivalent (offsets and `FATTN_SMEM` match), yet only the non-FA path is
correct — the true root cause needs empirical buffer-level bisection (dump
K·Q^T / softmax / P·V intermediates), not a source diff. Reference kept at
`work/iron-rc2` (base `4686a7095`).

## Why this is low priority now

On this hardware FA is **not a speed win**: GPU FA (garbage) measured ~82 t/s
vs the correct non-FA path at ~**83–94 t/s** tg on Llama-3.2-3B (one die). The
only real upside would be a GPU-resident **quantized KV cache** (capacity for
long context / larger models), which requires the kernel to be correct first.

## Behaviour matrix (validated on W6800X die 0, Llama-3.2-3B)

| Config | runner flag | output | tg t/s |
|---|---|---|---|
| default (nothing set) | `--flash-attn auto` | coherent | ~83 |
| `OLLAMA_FLASH_ATTENTION=1` | `--flash-attn off` (guarded) | coherent | ~88 |
| `GGML_METAL_AMD_FA=1` (+ FA on) | `--flash-attn on` | **garbage** | ~86 |

## Ollama guard (patch 0005)

Because Ollama would otherwise honour `OLLAMA_FLASH_ATTENTION=1` verbatim and
force the broken kernel, `patches/ollama/0005-metal-amd-disable-flash-attn.patch`
makes `LlamaServerFlashAttention` return `Disabled` for discrete-AMD Metal
(any `Metal` device on non-`arm64`) **regardless** of `OLLAMA_FLASH_ATTENTION`,
unless `GGML_METAL_AMD_FA` is set (the true experimental opt-in). Apple-Silicon
Metal (arm64) is unaffected. This keeps the default *and* the common
`OLLAMA_FLASH_ATTENTION=1` recommendation correct and fast on this hardware.

## Next step (kernel correctness, if quantized-KV capacity is wanted)

Empirically bisect the `b10091` vec kernel on RDNA2 by dumping intermediate
tensors (scores, online-softmax `m`/`s`, accumulator) for a 1-token / 1-head
case and comparing against the CPU reference, rather than diffing against RC2
(which is semantically equivalent at `nsg=1` yet reportedly correct — implying
the divergence is compiler/codegen-specific and must be found empirically). Use
MIT `iRon-Llama-RC2` only (never GPL-3.0 ToshLLM) as a reference.
