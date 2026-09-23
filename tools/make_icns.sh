#!/usr/bin/env bash
# Build assets/AppIcon.icns from assets/icon.png (regenerate that with make_icon.py).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/assets/icon.png"
SET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$SET"
for size in 16 32 128 256 512; do
  sips -z $size $size        "$SRC" --out "$SET/icon_${size}x${size}.png"    >/dev/null
  sips -z $((size*2)) $((size*2)) "$SRC" --out "$SET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$SET" -o "$HERE/assets/AppIcon.icns"
echo "wrote $HERE/assets/AppIcon.icns"
