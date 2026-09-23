#!/usr/bin/env bash
# Launch RGC_SDR. Used both from a terminal and by the Desktop shortcut, so it reports
# problems through a dialog when there is no terminal to print to.
#
#   ./run.sh                 # open the waterfall at the default frequency
#   ./run.sh --freq 0.909e6  # any flag from `python -m src.rgc_sdr --help`
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

notify() {
  if [ -t 2 ]; then
    printf 'RGC_SDR: %s\n' "$1" >&2
  else
    # Launched from Finder: no terminal, so put it on screen.
    /usr/bin/osascript - "$1" >/dev/null 2>&1 <<'OSA'
on run argv
  display dialog (item 1 of argv) with title "RGC_SDR" buttons {"OK"} ¬
    default button 1 with icon caution
end run
OSA
  fi
}

PY="$HERE/.venv/bin/python"
if [ ! -x "$PY" ]; then
  notify "No Python environment found at .venv. Run ./setup.sh in $HERE first."
  exit 1
fi

# Fail with a clear message rather than a stack trace when the radio is unplugged.
if ! "$PY" -c 'import sys
from src.rgc_sdr.device.source import enumerate_devices
sys.exit(0 if enumerate_devices() else 1)' >/dev/null 2>&1; then
  notify "No SDR detected.

Plug in the Airspy HF+ and try again."
  exit 1
fi

exec "$PY" -m src.rgc_sdr "$@"
