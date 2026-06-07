#!/usr/bin/env bash
# Build Ollama for darwin/amd64 using Ollama's canonical CMake flow.
#
# Ollama's top-level CMake is the orchestration entrypoint: `cmake --build`
# drives BOTH the native llama-server payload (via llama/server/CMakeLists.txt
# + FetchContent of the pinned llama.cpp) AND the Go binary (ollama-go target).
# Do NOT call `make llama-server` or `go build` directly.
#
# Backend selection on Apple is decided inside cmake/local.cmake, NOT by a
# top-level -DGGML_METAL flag (that does not propagate to the inner
# llama-server sub-build). Stock Ollama gates Metal to arm64:
#
#     cmake/local.cmake:506  if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
#                                -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
#                            else() ... if(APPLE) -DGGML_METAL=OFF
#
# So on x86_64 a STOCK build is CPU-only (our Phase 0 control). Metal on
# x86_64 comes from the pa-ungate patch (patches/ollama/) which widens that
# condition to `if(APPLE)`. Hence this script needs no GGML_METAL flag.
#
# Modes:
#   BUILD_MODE=local    (default) build against our patched work/llama.cpp tree.
#                       Requires the Ollama compat patch already applied to
#                       that tree (apply-patch.sh) + SKIP_COMPAT flag, because
#                       setting OLLAMA_LLAMA_CPP_SOURCE disables auto-compat.
#   BUILD_MODE=control  stock build -> CPU-only x86_64 baseline for A/B.
#                       Lets FetchContent clone+compat-patch llama.cpp b9509.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

[ -d "$OLLAMA_DIR/.git" ] || { echo "Run scripts/bootstrap.sh first"; exit 1; }

BUILD_MODE="${BUILD_MODE:-local}"
BUILD_DIR="$WORK_DIR/build-amd64"
INSTALL_PREFIX="$WORK_DIR/dist/darwin-amd64"
mkdir -p "$BUILD_DIR" "$INSTALL_PREFIX"

CMAKE_EXTRA=()
if [ "$BUILD_MODE" = "local" ]; then
  echo "BUILD_MODE=local: linking patched llama.cpp at $LLAMA_DIR"
  echo "  (assumes scripts/apply-patch.sh already applied kernels + Ollama compat patch)"
  export OLLAMA_LLAMA_CPP_SOURCE="$LLAMA_DIR"
  # Compat sources are only auto-linked when we tell CMake the tree is prepared.
  CMAKE_EXTRA+=( -DOLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON )
else
  echo "BUILD_MODE=control: stock build (FetchContent llama.cpp b9509, CPU-only on x86_64)"
fi

echo "Configuring (x86_64) ..."
cmake -S "$OLLAMA_DIR" -B "$BUILD_DIR" \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  "${CMAKE_EXTRA[@]}"

echo "Building (native payload + Go binary) ..."
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "Installing into $INSTALL_PREFIX ..."
cmake --install "$BUILD_DIR" || true

echo
echo "Built ollama: $OLLAMA_DIR/ollama (Go binary)"
echo "Payload:      $BUILD_DIR/lib/ollama/"
echo "Next: scripts/bench.sh <model>   (or ./ollama serve to smoke-test)"
