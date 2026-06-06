#!/usr/bin/env bash
# Shared environment for all scripts. Source this; do not execute.
# Reads the version pins (single source of truth) and locates the toolchain.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# --- Version pins (read from the root files, never hardcode) ---------------
export LLAMA_CPP_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/LLAMA_CPP_VERSION")"
export OLLAMA_TAG="$(sed -n '1p' "$REPO_ROOT/OLLAMA_VERSION" | tr -d '[:space:]')"
export OLLAMA_COMMIT="$(sed -n '2p' "$REPO_ROOT/OLLAMA_VERSION" | tr -d '[:space:]')"

# --- Work area (git-ignored) ------------------------------------------------
export WORK_DIR="${WORK_DIR:-$REPO_ROOT/work}"
export OLLAMA_DIR="$WORK_DIR/ollama"
export LLAMA_DIR="$WORK_DIR/llama.cpp"
mkdir -p "$WORK_DIR"

# --- Toolchain (MacPorts preferred; no Homebrew on this host) --------------
export PATH="/opt/local/bin:$PATH"
command -v go   >/dev/null || { echo "ERROR: go not found (sudo port install go)"; exit 1; }
command -v cmake>/dev/null || { echo "ERROR: cmake not found (sudo port install cmake)"; exit 1; }
command -v git  >/dev/null || { echo "ERROR: git not found"; exit 1; }

# Number of parallel build jobs.
export JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

echo "ollama-amd-metal env:"
echo "  OLLAMA   = $OLLAMA_TAG ($OLLAMA_COMMIT)"
echo "  LLAMACPP = $LLAMA_CPP_VERSION"
echo "  GO       = $(go version)"
echo "  WORK_DIR = $WORK_DIR"
echo "  JOBS     = $JOBS"
