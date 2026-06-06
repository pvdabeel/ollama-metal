#!/usr/bin/env bash
# Clone Ollama and llama.cpp at the pinned versions into work/ (git-ignored).
# Idempotent: re-running fetches/cleans to the pinned refs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

clone_at() {
  local url="$1" dir="$2" ref="$3"
  if [ ! -d "$dir/.git" ]; then
    echo "Cloning $url @ $ref ..."
    git clone "$url" "$dir"
  fi
  echo "Checking out $ref in $dir ..."
  git -C "$dir" fetch --tags --depth 1 origin "$ref" 2>/dev/null || git -C "$dir" fetch --tags origin
  git -C "$dir" checkout -f "$ref"
  git -C "$dir" clean -fdx >/dev/null 2>&1 || true
}

clone_at "https://github.com/ollama/ollama.git"        "$OLLAMA_DIR" "$OLLAMA_COMMIT"
clone_at "https://github.com/ggml-org/llama.cpp.git"   "$LLAMA_DIR"  "$LLAMA_CPP_VERSION"

# Sanity: the Ollama release must pin the llama.cpp tag we expect.
ACTUAL="$(tr -d '[:space:]' < "$OLLAMA_DIR/LLAMA_CPP_VERSION" 2>/dev/null || echo '?')"
if [ "$ACTUAL" != "$LLAMA_CPP_VERSION" ]; then
  echo "WARNING: Ollama pins llama.cpp '$ACTUAL' but our LLAMA_CPP_VERSION is '$LLAMA_CPP_VERSION'."
  echo "         Update LLAMA_CPP_VERSION to match before patching (see docs/repatching.md)."
fi

echo "Bootstrap done. Next: scripts/apply-patch.sh"
