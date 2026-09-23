"""RGC_SDR entry point.

    python -m src.rgc_sdr --list
    python -m src.rgc_sdr                      # resumes the last used settings
    python -m src.rgc_sdr --freq 7.1e6 --zoom 8

With no arguments the receiver comes back up where it was left. Any flag given overrides
the saved value for that one setting; `--no-restore` ignores the saved state entirely.
"""

from __future__ import annotations

import argparse
import sys

from .device.source import SoapyIQSource, enumerate_devices
from .dsp.demod import MODES
from .settings import Settings
from .ui.main_window import FFT_SIZES, ZOOM_FACTORS
from .ui.waterfall import COLORMAPS

DEFAULT_FREQ = 7.1e6


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rgc_sdr", description="Live SDR spectrum and waterfall.")
    p.add_argument("--list", action="store_true", help="list attached SDRs and exit")
    p.add_argument("--driver", default="airspyhf", help="SoapySDR driver key (default: airspyhf)")
    p.add_argument("--serial", default=None, help="device serial, when several are attached")
    p.add_argument("--freq", type=float, default=None, help="centre frequency in Hz")
    p.add_argument("--rate", type=float, default=None, help="sample rate in Hz")
    p.add_argument("--zoom", type=int, default=None, choices=ZOOM_FACTORS,
                   help="decimation factor: narrower span, finer resolution")
    p.add_argument("--fft", type=int, default=None, choices=FFT_SIZES, help="FFT size")
    p.add_argument("--fps", type=int, default=25, help="display frame rate")
    p.add_argument("--rows", type=int, default=512, help="waterfall history rows")
    p.add_argument("--bins", type=int, default=1024, help="waterfall width in bins")
    p.add_argument("--colormap", default=None, choices=COLORMAPS, help="waterfall colour map")
    p.add_argument("--min-db", type=float, default=None, help="colour floor (default: auto-fit)")
    p.add_argument("--max-db", type=float, default=None, help="colour ceiling (default: auto-fit)")
    p.add_argument("--agc", dest="agc", action="store_true", default=None, help="enable AGC")
    p.add_argument("--no-agc", dest="agc", action="store_false", help="disable AGC")
    p.add_argument("--mode", default=None, choices=("off",) + MODES,
                   help="demodulation mode (default: off)")
    p.add_argument("--volume", type=float, default=None, metavar="0..1",
                   help="audio volume")
    p.add_argument("--offset", type=float, default=None, metavar="HZ",
                   help="listen this far from the tuned centre")
    p.add_argument("--squelch", type=float, default=None, metavar="DBFS",
                   help="mute FM below this level")
    p.add_argument("--step", type=float, default=None, metavar="HZ",
                   help="tuning increment for arrow keys and sideways swipes")
    p.add_argument("--bandwidth", type=float, default=None, metavar="HZ",
                   help="channel filter width (default: the mode's own)")
    p.add_argument("--no-audio", action="store_true", help="do not open an audio device")
    p.add_argument("--recordings", default=None, metavar="DIR",
                   help="where to write recordings (default: ~/Documents/RGC_SDR)")
    p.add_argument("--debug-gestures", action="store_true",
                   help="log trackpad wheel and gesture events, to diagnose swipe tuning")
    p.add_argument("--no-restore", action="store_true", help="ignore saved settings this run")
    p.add_argument("--forget", action="store_true",
                   help="delete saved settings and memories, then exit")
    p.add_argument("--memory", default=None, metavar="NAME",
                   help="start from a saved memory")
    p.add_argument("--list-memories", action="store_true", help="list saved memories and exit")
    return p


def list_devices() -> int:
    devices = enumerate_devices()
    if not devices:
        print("No SDR devices found. Check the USB connection and Soapy modules.")
        return 1
    for i, d in enumerate(devices):
        print(f"[{i}] " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return list_devices()

    settings = Settings.load()

    if args.forget:
        fresh = Settings(settings.path)
        fresh.save()
        print(f"cleared {settings.path}")
        return 0

    if args.list_memories:
        if not settings.memories:
            print("No saved memories.")
            return 0
        for memory in settings.memories:
            print(f"{memory.name}\t{memory.snapshot.describe()}")
        return 0

    # Precedence: an explicit flag beats a recalled memory, which beats the last state.
    base = None
    if args.memory:
        memory = settings.get_memory(args.memory)
        if memory is None:
            print(f"No memory named {args.memory!r}. Try --list-memories.", file=sys.stderr)
            return 2
        base = memory.snapshot
    elif settings.last is not None and not args.no_restore:
        base = settings.last

    def pick(flag, attr, fallback):
        if flag is not None:
            return flag
        return getattr(base, attr) if base is not None else fallback

    freq = pick(args.freq, "freq_hz", DEFAULT_FREQ)
    rate = pick(args.rate, "sample_rate", None)
    zoom = pick(args.zoom, "decimation", 1)
    fft = pick(args.fft, "fft_size", 4096)
    colormap = pick(args.colormap, "colormap", "inferno")
    agc = pick(args.agc, "agc", None)
    peak_hold = base.peak_hold if base is not None else True
    mode = pick(args.mode, "mode", "off")
    volume = pick(args.volume, "volume", 0.4)
    offset = pick(args.offset, "offset_hz", 0.0)
    squelch = args.squelch if args.squelch is not None else (
        base.squelch_dbfs if base is not None else None
    )
    bandwidth = args.bandwidth if args.bandwidth is not None else (
        base.bandwidth_hz if base is not None else None
    )
    step = pick(args.step, "step_hz", 10e3)

    if args.min_db is not None or args.max_db is not None:
        levels = (args.min_db if args.min_db is not None else -120.0,
                  args.max_db if args.max_db is not None else -60.0)
    elif base is not None and base.min_db is not None and base.max_db is not None:
        levels = (base.min_db, base.max_db)
    else:
        levels = None

    try:
        source = SoapyIQSource(
            driver=args.driver, serial=args.serial, sample_rate=rate,
            center_freq=freq, agc=agc,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Could not open driver={args.driver}: {exc}", file=sys.stderr)
        print("Run with --list to see attached devices.", file=sys.stderr)
        return 2

    caps = source.caps
    if not caps.covers(freq):
        print(f"Warning: {freq / 1e6:g} MHz is outside this device's range "
              f"({caps.describe_ranges()}).", file=sys.stderr)

    from .ui.main_window import run

    return run(
        debug_gestures=args.debug_gestures,
        source,
        fft_size=fft,
        fps=args.fps,
        history_rows=args.rows,
        waterfall_bins=args.bins,
        colormap=colormap,
        levels=levels,
        decimation=zoom,
        peak_hold=peak_hold,
        mode=mode,
        volume=volume,
        offset_hz=offset,
        squelch_dbfs=squelch,
        bandwidth_hz=bandwidth,
        step_hz=step,
        enable_audio=not args.no_audio,
        recordings_dir=args.recordings,
        settings=settings,
    )


if __name__ == "__main__":
    raise SystemExit(main())
