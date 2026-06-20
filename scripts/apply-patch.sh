#!/usr/bin/env bash
# Apply our patches onto the work/ checkouts. Uses --3way so version drift
# surfaces as normal git conflicts (see docs/repatching.md for forward-port).
# Idempotent: a patch that reverse-applies cleanly is treated as already
# applied and skipped, so re-running is safe.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# Apply one patch file to a git tree, skipping if already applied.
apply_one() {
  local tree="$1" p="$2"
  if git -C "$tree" apply --reverse --check "$p" >/dev/null 2>&1; then
    echo "  already applied: $(basename "$p")"
    return 0
  fi
  echo "Applying $(basename "$p") -> $tree"
  git -C "$tree" apply --3way --whitespace=nowarn "$p"
}

apply_dir() {
  local tree="$1" patchdir="$2"
  [ -d "$patchdir" ] || { echo "No patches in $patchdir (skipping)"; return 0; }
  shopt -s nullglob
  local applied=0
  for p in "$patchdir"/*.patch; do
    apply_one "$tree" "$p"
    applied=$((applied+1))
  done
  shopt -u nullglob
  echo "  ($applied patch(es) processed for $tree)"
}

# Ollama ships an in-process llama.cpp "compat" layer: source files that are
# linked into the llama target PLUS an ordered patch set that adds the matching
# call-sites and model-class declarations (e.g. llama_model_laguna). Normally
# Ollama's build applies these via FetchContent's PATCH_COMMAND, but our local
# build points OLLAMA_LLAMA_CPP_SOURCE at work/llama.cpp and sets
# OLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON (which only links the sources, never
# patches). So we must apply the compat patches to our llama.cpp tree here, or
# the linked compat sources fail to compile. The patches live in the Ollama
# tree and are applied in basename order, mirroring llama/compat/apply-patch.cmake.
apply_compat() {
  local tree="$1" compatroot="$2"
  [ -d "$compatroot" ] || { echo "No compat dir $compatroot (skipping)"; return 0; }
  shopt -s nullglob globstar
  # Collect recursively, then sort by basename to match apply-patch.cmake order.
  local p base
  local -a entries=()
  for p in "$compatroot"/**/*.patch; do
    base="$(basename "$p")"
    entries+=("$base|$p")
  done
  shopt -u nullglob globstar
  local sorted
  sorted="$(printf '%s\n' "${entries[@]}" | sort)"
  local count=0
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    p="${line#*|}"
    apply_one "$tree" "$p"
    count=$((count+1))
  done <<< "$sorted"
  echo "  ($count compat patch(es) processed for $tree)"
}

apply_dir "$LLAMA_DIR"  "$REPO_ROOT/patches/llama-cpp"
apply_dir "$OLLAMA_DIR" "$REPO_ROOT/patches/ollama"

echo "Applying Ollama llama.cpp compat patches -> $LLAMA_DIR"
apply_compat "$LLAMA_DIR" "$OLLAMA_DIR/llama/compat"

echo "Patches applied. Next: scripts/build.sh"
