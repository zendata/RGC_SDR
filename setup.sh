#!/usr/bin/env bash
# Set up the Python environment for RGC_SDR.
#
# Requires the SoapySDR native libraries and the Airspy HF+ driver module, which come
# from Homebrew rather than pip:
#     brew install soapysdr soapyairspyhf
#
# Usage:
#   ./setup.sh           # app + test deps
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip ..."
pip install -q --upgrade pip

echo "Installing app + test dependencies ..."
pip install -q numpy PyQt6 pyqtgraph scipy sounddevice pytest

echo
echo "Checking for the SoapySDR bindings ..."
if python -c "import SoapySDR" 2>/dev/null; then
  python - <<'PY'
import SoapySDR
found = SoapySDR.Device.enumerate()
print(f"  SoapySDR OK - {len(found)} device(s) attached")
for d in found:
    print("   ", dict(d).get("label", dict(d)))
PY
else
  echo "  SoapySDR bindings NOT found."
  echo "  Install the native side with:  brew install soapysdr soapyairspyhf"
  echo "  The venv must also see them; if brew's python differs, create the venv with"
  echo "  --system-site-packages or add the Soapy python path to PYTHONPATH."
fi

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
echo "List devices:         python -m src.rgc_sdr --list"
echo "Run the waterfall:    python -m src.rgc_sdr --freq 7.1e6"
