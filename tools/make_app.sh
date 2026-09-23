#!/usr/bin/env bash
# Build the double-clickable "RGC SDR.app" launcher.
#
#   ./tools/make_app.sh              # onto the Desktop
#   ./tools/make_app.sh /Applications
#
# The bundle is a thin wrapper: it just runs run.sh from this repo, so the shortcut
# always launches the current working copy. Re-run this if the repo ever moves.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$HOME/Desktop}"
APP="$DEST/RGC SDR.app"

[ -x "$REPO/run.sh" ] || { echo "missing $REPO/run.sh" >&2; exit 1; }

# The icon is a build artifact, regenerated from tools/make_icon.py rather than committed.
if [ ! -f "$REPO/assets/AppIcon.icns" ]; then
  echo "generating icon ..."
  QT_QPA_PLATFORM=offscreen "$REPO/.venv/bin/python" "$REPO/tools/make_icon.py" >/dev/null
  "$REPO/tools/make_icns.sh" >/dev/null
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleName</key>
	<string>RGC SDR</string>
	<key>CFBundleDisplayName</key>
	<string>RGC SDR</string>
	<key>CFBundleIdentifier</key>
	<string>com.rgc-sdr.launcher</string>
	<key>CFBundleVersion</key>
	<string>0.3.0</string>
	<key>CFBundleShortVersionString</key>
	<string>0.3.0</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleExecutable</key>
	<string>RGCSDR</string>
	<key>CFBundleIconFile</key>
	<string>AppIcon</string>
	<key>LSMinimumSystemVersion</key>
	<string>11.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>LSUIElement</key>
	<false/>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/RGCSDR" <<LAUNCH
#!/bin/bash
# Thin wrapper -- all the logic lives in the repo so this shortcut tracks the
# current version. Rebuild with tools/make_app.sh if the repo moves.
exec "$REPO/run.sh"
LAUNCH
chmod +x "$APP/Contents/MacOS/RGCSDR"

if [ -f "$REPO/assets/AppIcon.icns" ]; then
  cp "$REPO/assets/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
fi

# Nudge Finder to pick up the new icon rather than showing a stale cached one.
touch "$APP"
echo "built $APP -> $REPO/run.sh"
