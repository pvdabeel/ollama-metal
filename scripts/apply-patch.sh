#!/usr/bin/env bash
# Apply our patches onto the work/ checkouts. Uses --3way so version drift
# surfaces as normal git conflicts (see docs/repatching.md for forward-port).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

apply_dir() {
  local tree="$1" patchdir="$2"
  [ -d "$patchdir" ] || { echo "No patches in $patchdir (skipping)"; return 0; }
  shopt -s nullglob
  local applied=0
  for p in "$patchdir"/*.patch; do
    echo "Applying $(basename "$p") -> $tree"
    git -C "$tree" apply --3way --whitespace=nowarn "$p"
    applied=$((applied+1))
  done
  shopt -u nullglob
  echo "  ($applied patch(es) applied to $tree)"
}

apply_dir "$LLAMA_DIR"  "$REPO_ROOT/patches/llama-cpp"
apply_dir "$OLLAMA_DIR" "$REPO_ROOT/patches/ollama"

echo "Patches applied. Next: scripts/build.sh"
