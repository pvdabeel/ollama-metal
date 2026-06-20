#!/usr/bin/env bash
# Clone Ollama and llama.cpp at the pinned versions into work/ (git-ignored).
# Idempotent: re-running fetches/cleans to the pinned refs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

clone_at() {
  local url="$1" dir="$2" ref="$3"
  # Shallow fetch only the pinned ref (tag or commit SHA). A full-history clone
  # of Ollama/llama.cpp is large and flaky over slow links; GitHub supports
  # fetching a specific commit/tag, so depth-1 is both faster and more reliable.
  if [ ! -d "$dir/.git" ]; then
    echo "Initialising $dir for shallow fetch of $ref ..."
    git init -q "$dir"
    git -C "$dir" remote add origin "$url" 2>/dev/null || git -C "$dir" remote set-url origin "$url"
  fi
  echo "Fetching $ref from $url (shallow) ..."
  # Force HTTP/1.1 and a large post buffer: GitHub's HTTP/2 multiplexing is
  # prone to "stream reset by server (error 0x8 CANCEL) / early EOF" on slow or
  # lossy links, which aborts the packfile transfer. HTTP/1.1 is slower but
  # reliable here. Retry a few times before falling back to a full fetch.
  local git_robust=(-c http.version=HTTP/1.1 -c http.postBuffer=524288000)
  local ok=0 attempt
  for attempt in 1 2 3; do
    if git "${git_robust[@]}" -C "$dir" fetch --depth 1 --tags origin "$ref"; then
      ok=1; break
    fi
    echo "  shallow fetch attempt $attempt failed; retrying ..."
    sleep 3
  done
  if [ "$ok" -ne 1 ]; then
    echo "  shallow fetch failed 3x; retrying with full fetch ..."
    git "${git_robust[@]}" -C "$dir" fetch --tags origin
  fi
  git -C "$dir" checkout -f FETCH_HEAD
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
