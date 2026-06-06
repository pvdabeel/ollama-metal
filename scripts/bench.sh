#!/usr/bin/env bash
# Validation harness: correctness + speed vs the CPU baseline.
# Usage: scripts/bench.sh <model-gguf-or-ollama-blob> [device-index]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MODEL="${1:?usage: bench.sh <model.gguf|ollama-blob> [device-index]}"
DEV="${2:-0}"
BUILD_DIR="$WORK_DIR/build-amd64"
BENCH="$BUILD_DIR/bin/llama-bench"

[ -x "$BENCH" ] || { echo "llama-bench not found at $BENCH (run scripts/build.sh)"; exit 1; }

echo "=== GPU (die $DEV, patched Metal) ==="
GGML_METAL_DEVICE_INDEX="$DEV" GGML_METAL_N_CB=4 \
  "$BENCH" -m "$MODEL" -fa 0 -ub 32 -b 32 -p 64 -n 32 -r 2 -ngl 99 || true

echo "=== CPU baseline (-ngl 0) ==="
"$BENCH" -m "$MODEL" -p 64 -n 32 -r 2 -ngl 0 || true

cat <<'NOTE'

Validation gates (both must pass):
  1. CORRECTNESS: run a short generation and confirm coherent text:
       GGML_METAL_DEVICE_INDEX=0 GGML_METAL_N_CB=4 \
         work/build-amd64/bin/llama-cli -m <model> -ngl 99 -n 48 \
         -p "Explain in two sentences why the sky is blue."
  2. SPEED: GPU tg t/s must clearly exceed the CPU tg t/s above.
Record results in bench/ with model, quant, die index, and build pins.
NOTE
