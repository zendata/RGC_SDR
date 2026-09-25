#!/usr/bin/env bash
# Install VK3RQ Super SDR on a Mac from scratch: every dependency, the SDR drivers, the
# Python environment and the Desktop icon. Safe to run again: each step skips or
# refreshes what is already there.
#
#   ./install.sh                 # from a clone of the repository
#   ./install.sh --skip-tests    # without running the test suite at the end
#
# On a Mac with nothing installed, get the repository first (git comes with the Xcode
# Command Line Tools, which macOS offers to install the first time git is run):
#
#   git clone https://github.com/zendata/RGC_SDR.git ~/Code/RGC_SDR
#   cd ~/Code/RGC_SDR && ./install.sh
#
# Apple silicon macOS 11 or later. Needs an internet connection and, for Homebrew and
# the Command Line Tools, your login password once. Takes ~10-20 minutes the first time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
RUN_TESTS=1
[ "${1:-}" = "--skip-tests" ] && RUN_TESTS=0

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mInstall stopped: %s\033[0m\n' "$1" >&2; exit 1; }

step "1/7  Checking the Mac"
[ "$(uname -s)" = Darwin ] || fail "this installer is for macOS."
[ "$(uname -m)" = arm64 ] || echo "  Note: tested on Apple silicon only; this Mac is $(uname -m)."
echo "  macOS $(sw_vers -productVersion), $(uname -m)"

step "2/7  Xcode Command Line Tools (compiler, git)"
if xcode-select -p >/dev/null 2>&1; then
  echo "  already installed"
else
  xcode-select --install || true
  fail "a dialog has opened to install the Command Line Tools. When it finishes, run ./install.sh again."
fi

step "3/7  Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    [ -x "$candidate" ] && eval "$("$candidate" shellenv)" && break
  done
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "  installing Homebrew (it will ask for your password) ..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
  # So new terminals find brew too.
  if ! grep -q 'brew shellenv' "$HOME/.zprofile" 2>/dev/null; then
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
  fi
fi
echo "  $(brew --version | head -1) at $(brew --prefix)"

step "4/7  Homebrew packages"
# soapysdr brings its own Python (the one its bindings are built for); cmake, libusb and
# pkgconf are for the drivers built from source in the next step.
brew install soapysdr soapyhackrf airspy airspyhf libusb cmake pkgconf

step "5/7  SDR drivers (Airspy HF+, Airspy R2/Mini, ADALM-Pluto) from source"
./tools/install_drivers.sh all
SoapySDRUtil --info 2>/dev/null | grep "Available factories" | sed 's/^/  /'

step "6/7  Python environment"
./setup.sh

if [ "$RUN_TESTS" = 1 ]; then
  echo
  echo "  running the test suite (hardware tests skip themselves if no radio is attached) ..."
  .venv/bin/python -m pytest -q 2>&1 | tail -1 | sed 's/^/  /'
fi

step "7/7  Desktop icon"
./tools/make_app.sh

cat <<EOF

Installed. Double-click "VK3RQ Super SDR" on the Desktop, or run ./run.sh here.

  * Plug in a radio before launching; the SDR menu shows which ones are connected.
  * The first TX press asks macOS for microphone permission.
  * An RTL-SDR needs one more package:  brew install soapyrtlsdr
EOF
