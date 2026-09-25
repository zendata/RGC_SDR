#!/usr/bin/env bash
# Set up the Python environment for VK3RQ Super SDR (RGC_SDR).
#
# Needs the native SoapySDR stack from Homebrew first -- `./install.sh` does all of it on
# a new Mac. This script only builds .venv:
#
#   ./setup.sh
#
# The SoapySDR Python bindings are not on PyPI. Homebrew's soapysdr builds them for one
# particular Homebrew Python (3.14 as of 2026-09), so the venv is made with *that*
# Python and the two binding files are linked in. A venv made with macOS's own python3
# (3.9) cannot load them.
set -euo pipefail

cd "$(dirname "$0")"
VENV_DIR=".venv"

BREW="$(command -v brew || true)"
[ -n "$BREW" ] || { echo "Homebrew is needed first: run ./install.sh" >&2; exit 1; }
PREFIX="$("$BREW" --prefix)"

# The Python soapysdr was built against, e.g. python@3.14 -> 3.14.
PYVER="$("$BREW" deps soapysdr 2>/dev/null | sed -n 's/^python@\([0-9.]*\)$/\1/p' | head -1)"
[ -n "$PYVER" ] || { echo "soapysdr is not installed: brew install soapysdr" >&2; exit 1; }
PYTHON="${PYTHON:-$PREFIX/opt/python@$PYVER/bin/python$PYVER}"
BINDINGS="$PREFIX/lib/python$PYVER/site-packages"

if [ -d "$VENV_DIR" ] && ! "$VENV_DIR/bin/python" -c \
    "import sys; sys.exit(sys.version.startswith('$PYVER') is False)" 2>/dev/null; then
  echo "Existing $VENV_DIR is not Python $PYVER; rebuilding it ..."
  rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating $VENV_DIR with Python $PYVER ..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing Python packages ..."
pip install -q --upgrade pip
pip install -q numpy PyQt6 pyqtgraph scipy sounddevice pytest

echo "Linking the SoapySDR bindings from $BINDINGS ..."
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for f in SoapySDR.py _SoapySDR.so; do
  [ -e "$BINDINGS/$f" ] || { echo "missing $BINDINGS/$f -- reinstall soapysdr" >&2; exit 1; }
  ln -sf "$BINDINGS/$f" "$SITE/$f"
done

python - <<'PY'
import SoapySDR
kinds = sorted({str(dict(d).get("driver", "?")) for d in SoapySDR.Device.enumerate()})
print(f"  SoapySDR {SoapySDR.getAPIVersion()} OK; attached: {', '.join(kinds) or 'none'}")
PY

echo
echo "Done. Run the app with ./run.sh, or build the Desktop icon: ./tools/make_app.sh"
