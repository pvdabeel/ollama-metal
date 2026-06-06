#!/usr/bin/env bash
# Revert scripts/install-app.sh: restore the original Ollama.app binaries from
# their *.orig backups.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

APP="${OLLAMA_APP:-/Applications/Ollama.app}"
RES="$APP/Contents/Resources"

[ -d "$APP" ] || { echo "ERROR: $APP not found"; exit 1; }

echo "Stopping Ollama ..."
osascript -e 'tell application "Ollama" to quit' >/dev/null 2>&1 || true
sleep 1
pkill -f "$RES/ollama" 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
sleep 1

restored=0
for f in ollama llama-server; do
  if [ -e "$RES/$f.orig" ]; then
    echo "Restoring $f from $f.orig"
    mv -f "$RES/$f.orig" "$RES/$f"
    restored=$((restored+1))
  else
    echo "No backup for $f (skipping)"
  fi
done

if [ "$restored" -gt 0 ]; then
  echo "Restored $restored original binary(ies). The app's own Developer-ID"
  echo "signatures are intact again. Relaunch Ollama from /Applications."
else
  echo "Nothing to restore."
fi
