#!/usr/bin/env bash
# Patch the official Ollama.app (the nice menubar UI from ollama.com) to use our
# Metal-enabled, AMD-friendly binaries.
#
# The desktop app (Contents/MacOS/Ollama) spawns Contents/Resources/ollama
# (the Go server), which in turn spawns Contents/Resources/llama-server (the
# runner). We replace BOTH with our patched x86_64 builds:
#   - llama-server : statically links our ggml-metal backend (Metal on x86_64,
#                    discrete-AMD concurrency + cross-die fixes, embedded shaders)
#   - ollama       : Go server with the discrete-Metal mmap disable (the big perf
#                    win) + multi-die discovery/pinning.
#
# Originals are backed up to *.orig (first run only). Re-run after any Ollama.app
# update. Reverse with scripts/uninstall-app.sh.
#
# The app is Developer-ID signed; we ad-hoc re-sign the two replaced binaries.
# This works on Intel (x86_64) macOS, which does not require notarised binaries.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

APP="${OLLAMA_APP:-/Applications/Ollama.app}"
RES="$APP/Contents/Resources"
DIST="$WORK_DIR/dist/darwin-amd64"
SRC_OLLAMA="$DIST/bin/ollama"
SRC_SERVER="$DIST/lib/ollama/llama-server"

[ -d "$APP" ] || { echo "ERROR: $APP not found (install Ollama.dmg from ollama.com first)"; exit 1; }
[ -x "$SRC_OLLAMA" ] || { echo "ERROR: $SRC_OLLAMA missing — run BUILD_MODE=local scripts/build.sh first"; exit 1; }
[ -x "$SRC_SERVER" ] || { echo "ERROR: $SRC_SERVER missing — run BUILD_MODE=local scripts/build.sh first"; exit 1; }

# --- version sanity: app must match the version we built against ---------------
APP_VER="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist" 2>/dev/null || echo '?')"
WANT_VER="${OLLAMA_TAG#v}"
echo "Ollama.app version : $APP_VER"
echo "Built against      : $WANT_VER ($OLLAMA_COMMIT)"
if [ "$APP_VER" != "$WANT_VER" ]; then
  echo
  echo "WARNING: version mismatch. The patched server may be incompatible with"
  echo "this app build. Re-pin OLLAMA_VERSION to $APP_VER and rebuild, or install"
  echo "the matching Ollama.app. Set OLLAMA_FORCE=1 to override."
  [ "${OLLAMA_FORCE:-}" = "1" ] || exit 1
fi

# --- stop any running server ---------------------------------------------------
echo "Stopping Ollama (app + server + runners) ..."
osascript -e 'tell application "Ollama" to quit' >/dev/null 2>&1 || true
sleep 1
pkill -f "$RES/ollama" 2>/dev/null || true
pkill -f "Ollama.app/Contents/Resources/ollama" 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
sleep 1

# --- back up originals (first run only) ----------------------------------------
for f in ollama llama-server; do
  if [ ! -e "$RES/$f.orig" ] && [ -e "$RES/$f" ]; then
    echo "Backing up $f -> $f.orig"
    cp -p "$RES/$f" "$RES/$f.orig"
  fi
done

# --- install patched binaries --------------------------------------------------
echo "Installing patched ollama  -> $RES/ollama"
cp -f "$SRC_OLLAMA" "$RES/ollama"
echo "Installing patched llama-server -> $RES/llama-server"
cp -f "$SRC_SERVER" "$RES/llama-server"

# --- ad-hoc sign + clear quarantine so they run as spawned children ------------
for f in ollama llama-server; do
  codesign --remove-signature "$RES/$f" 2>/dev/null || true
  codesign --force --sign - "$RES/$f"
  xattr -d com.apple.quarantine "$RES/$f" 2>/dev/null || true
done

echo
echo "Done. Patched binaries installed into $APP."
echo "Launch Ollama from /Applications (or it will auto-start at login), then verify:"
echo "  tail -f ~/.ollama/logs/server.log    # look for 'MTL0..MTL3' device lines"
echo "  ollama run llama3.2 'hi'              # then check Activity Monitor GPU history"
echo
echo "Revert with: scripts/uninstall-app.sh"
