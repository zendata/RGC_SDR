"""RGC_SDR entry point.

    python -m src.rgc_sdr --list
    python -m src.rgc_sdr --freq 7.1e6
    python -m src.rgc_sdr --driver hackrf --freq 100e6 --rate 8e6
"""

from __future__ import annotations

import argparse
import sys

from .device.source import SoapyIQSource, enumerate_devices
from .ui.main_window import FFT_SIZES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rgc_sdr", description="Live SDR spectrum and waterfall.")
    p.add_argument("--list", action="store_true", help="list attached SDRs and exit")
    p.add_argument("--driver", default="airspyhf", help="SoapySDR driver key (default: airspyhf)")
    p.add_argument("--serial", default=None, help="device serial, when several are attached")
    p.add_argument("--freq", type=float, default=7.1e6, help="centre frequency in Hz")
    p.add_argument("--rate", type=float, default=None, help="sample rate in Hz (default: 768k)")
    p.add_argument("--fft", type=int, default=4096, choices=FFT_SIZES, help="FFT size")
    p.add_argument("--fps", type=int, default=25, help="display frame rate")
    p.add_argument("--rows", type=int, default=512, help="waterfall history rows")
    p.add_argument("--bins", type=int, default=1024, help="waterfall width in bins")
    p.add_argument("--colormap", default="inferno", help="waterfall colour map")
    p.add_argument(
        "--min-db", type=float, default=None, help="colour range floor (default: auto-fit)"
    )
    p.add_argument(
        "--max-db", type=float, default=None, help="colour range ceiling (default: auto-fit)"
    )
    p.add_argument("--agc", dest="agc", action="store_true", default=None, help="enable AGC")
    p.add_argument("--no-agc", dest="agc", action="store_false", help="disable AGC")
    return p


def list_devices() -> int:
    devices = enumerate_devices()
    if not devices:
        print("No SDR devices found. Check the USB connection and Soapy modules.")
        return 1
    for i, d in enumerate(devices):
        detail = ", ".join(f"{k}={v}" for k, v in sorted(d.items()))
        print(f"[{i}] {detail}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return list_devices()

    try:
        source = SoapyIQSource(
            driver=args.driver,
            serial=args.serial,
            sample_rate=args.rate,
            center_freq=args.freq,
            agc=args.agc,
        )
    except RuntimeError as exc:  # missing bindings
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Could not open driver={args.driver}: {exc}", file=sys.stderr)
        print("Run with --list to see attached devices.", file=sys.stderr)
        return 2

    caps = source.caps
    if not caps.covers(args.freq):
        print(
            f"Warning: {args.freq / 1e6:g} MHz is outside this device's range "
            f"({caps.describe_ranges()}).",
            file=sys.stderr,
        )

    levels = None
    if args.min_db is not None or args.max_db is not None:
        levels = (
            args.min_db if args.min_db is not None else -120.0,
            args.max_db if args.max_db is not None else -60.0,
        )

    from .ui.main_window import run

    return run(
        source,
        fft_size=args.fft,
        fps=args.fps,
        history_rows=args.rows,
        waterfall_bins=args.bins,
        colormap=args.colormap,
        levels=levels,
    )


if __name__ == "__main__":
    raise SystemExit(main())
