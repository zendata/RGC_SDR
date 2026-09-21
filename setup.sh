#!/usr/bin/env bash
# Set up the Python environment for RGC_SDR.
# Usage:
#   ./setup.sh           # core deps (numpy) + pytest
#   ./setup.sh --all     # also install UI (PyQt6, pyqtgraph) and DSP (scipy)
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

echo "Installing dependencies ..."
pip install -q numpy pytest

if [ "${1:-}" = "--all" ]; then
  echo "Installing UI + DSP extras ..."
  pip install -q PyQt6 pyqtgraph scipy
fi

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
echo "Then run the spike:   python -m src.rgc_sdr.device.spike --simulate"
