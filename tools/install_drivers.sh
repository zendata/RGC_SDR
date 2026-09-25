#!/usr/bin/env bash
# Build and install the SoapySDR drivers Homebrew does not ship: Airspy R2/Mini and
# ADALM-Pluto. (Airspy HF+ and HackRF come from Homebrew; see README.)
#
#   ./tools/install_drivers.sh            # both
#   ./tools/install_drivers.sh airspy     # or just one: airspy | pluto
#
# Everything installs under /opt/homebrew, next to SoapySDR, so its module loader finds
# the drivers. Measured 2026-09-25 on macOS/arm64; three things had to be got right:
#  * libiio v0.25, not 1.x: SoapyPlutoSDR uses the v0 API.
#  * An older /Library/Frameworks/iio.framework (a libiio 1.x install) is preferred by
#    cmake and lacks v0 symbols, so frameworks are ignored (CMAKE_FIND_FRAMEWORK=NEVER).
#  * The libraries use @rpath names; without an install rpath SoapySDR cannot load the
#    Pluto module ("MISSING" in SoapySDRUtil --check=plutosdr).
set -euo pipefail

PREFIX=/opt/homebrew
WORK="${TMPDIR:-/tmp}/rgc-sdr-drivers"
WHICH="${1:-all}"
COMMON=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX"
        -DCMAKE_FIND_FRAMEWORK=NEVER -DCMAKE_INSTALL_RPATH="$PREFIX/lib")
mkdir -p "$WORK"

build() {   # build <dir> <git url> <branch or ""> [extra cmake args...]
  local dir=$1 url=$2 branch=$3; shift 3
  rm -rf "$WORK/$dir"
  if [ -n "$branch" ]; then
    git clone --depth 1 --branch "$branch" "$url" "$WORK/$dir"
  else
    git clone --depth 1 "$url" "$WORK/$dir"
  fi
  cmake -S "$WORK/$dir" -B "$WORK/$dir/build" "${COMMON[@]}" "$@"
  cmake --build "$WORK/$dir/build" -j 8
  cmake --install "$WORK/$dir/build"
}

if [ "$WHICH" = all ] || [ "$WHICH" = airspy ]; then
  brew install airspy
  build SoapyAirspy https://github.com/pothosware/SoapyAirspy.git ""
fi

if [ "$WHICH" = all ] || [ "$WHICH" = pluto ]; then
  brew install libusb
  build libiio https://github.com/analogdevicesinc/libiio.git v0.25 \
    -DOSX_FRAMEWORK=OFF -DOSX_PACKAGE=OFF -DWITH_USB_BACKEND=ON -DWITH_NETWORK_BACKEND=ON \
    -DHAVE_DNS_SD=OFF -DWITH_SERIAL_BACKEND=OFF -DWITH_TESTS=OFF -DWITH_DOC=OFF \
    -DPYTHON_BINDINGS=OFF -DCSHARP_BINDINGS=OFF
  build libad9361-iio https://github.com/analogdevicesinc/libad9361-iio.git "" \
    -DOSX_FRAMEWORK=OFF -DOSX_PACKAGE=OFF -DBUILD_TESTS=OFF -DWITH_DOC=OFF \
    -DPYTHON_BINDINGS=OFF -DLIBIIO_LIBRARIES="$PREFIX/lib/libiio.dylib" \
    -DLIBIIO_INCLUDEDIR="$PREFIX/include"
  build SoapyPlutoSDR https://github.com/pothosware/SoapyPlutoSDR.git ""
fi

SoapySDRUtil --info | grep "Available factories"
