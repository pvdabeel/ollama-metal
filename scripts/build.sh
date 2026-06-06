#!/usr/bin/env bash
# Build Ollama (x86_64) against the patched local llama.cpp tree.
# OLLAMA_LLAMA_CPP_SOURCE makes Ollama link our tree and skip the compat patch.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

[ -d "$OLLAMA_DIR/.git" ] || { echo "Run scripts/bootstrap.sh first"; exit 1; }

export OLLAMA_LLAMA_CPP_SOURCE="$LLAMA_DIR"

# Native CGO flags for the Metal-enabled amd64 build.
export CGO_ENABLED=1
export GOOS=darwin
export GOARCH=amd64
export CGO_CFLAGS="-O3 -mmacosx-version-min=14.0"
export CGO_LDFLAGS="-lc++ -framework Metal -framework Foundation -framework Accelerate -mmacosx-version-min=14.0"

BUILD_DIR="$WORK_DIR/build-amd64"
INSTALL_PREFIX="$WORK_DIR/dist/darwin-amd64"
mkdir -p "$BUILD_DIR" "$INSTALL_PREFIX"

echo "Configuring (x86_64, Metal on) against patched llama.cpp at $LLAMA_DIR ..."
cmake -S "$OLLAMA_DIR" -B "$BUILD_DIR" \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DOLLAMA_RUNNER_DIR=./

echo "Building llama-server (Metal) ..."
cmake --build "$BUILD_DIR" --target llama-server -j "$JOBS"
cmake --install "$BUILD_DIR" --component llama-server || true

echo "Building ollama binary ..."
( cd "$OLLAMA_DIR" && go build -o "$INSTALL_PREFIX/ollama" . )

echo
echo "Built: $INSTALL_PREFIX/ollama"
echo "       $INSTALL_PREFIX/lib/ollama/llama-server (if installed)"
echo "Next: scripts/bench.sh <model>"
