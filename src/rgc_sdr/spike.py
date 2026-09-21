"""P0 diagnostic (app layer): open the radio, stream IQ, report capabilities and levels.

Real hardware only -- the simulated path from v1 is gone (PLANNING.md section 1).

    python -m src.rgc_sdr.spike --list
    python -m src.rgc_sdr.spike --freq 7.1e6 --duration 2
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .device.source import SoapyIQSource, enumerate_devices
from .dsp.spectrum import SpectrumAnalyzer


def report(driver: str, serial: str | None, freq: float, rate: float | None, duration: float) -> int:
    src = SoapyIQSource(driver=driver, serial=serial, sample_rate=rate, center_freq=freq)
    caps = src.caps
    print(f"Device:       {caps.label or caps.driver} (serial {caps.serial or 'n/a'})")
    print(f"Sample rates: {', '.join(f'{r / 1e3:g}k' for r in caps.sample_rates)}")
    print(f"Freq ranges:  {caps.describe_ranges()}")
    print(f"Formats:      {', '.join(caps.formats)}")
    print(f"AGC:          {caps.has_agc}")
    print(
        "Gain stages:  "
        + (", ".join(g.name for g in caps.gain_elements) if caps.gain_elements else "none")
    )
    print(f"Streaming {duration:g}s at {src.sample_rate / 1e3:g} kS/s, {freq / 1e6:g} MHz ...")

    analyzer = SpectrumAnalyzer(fft_size=4096)
    with src:
        time.sleep(duration)
        iq = src.read_latest(analyzer.samples_wanted())
        stats = dict(src.stats)

    print(f"Samples:      {stats['samples']:,}")
    print(f"Throughput:   {stats['samples'] / duration / 1e3:.1f} kS/s")
    print(f"Overflows:    {stats['overflows']}  timeouts: {stats['timeouts']}  errors: {stats['errors']}")
    if iq.size < analyzer.fft_size:
        print("Not enough samples for a spectrum -- is the device delivering data?")
        return 1
    if not np.all(np.isfinite(iq)):
        print("Non-finite samples in the stream.")
        return 1
    dbfs = analyzer.psd_dbfs(iq)
    print(f"Peak:         {dbfs.max():.1f} dBFS")
    print(f"Noise floor:  {float(np.median(dbfs)):.1f} dBFS (median)")
    print(f"Dynamic span: {dbfs.max() - float(np.median(dbfs)):.1f} dB")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="list attached SDRs and exit")
    p.add_argument("--driver", default="airspyhf")
    p.add_argument("--serial", default=None)
    p.add_argument("--freq", type=float, default=7.1e6)
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--duration", type=float, default=2.0)
    args = p.parse_args(argv)

    if args.list:
        devices = enumerate_devices()
        if not devices:
            print("No SDR devices found.")
            return 1
        for i, d in enumerate(devices):
            print(f"[{i}] " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))
        return 0

    try:
        return report(args.driver, args.serial, args.freq, args.rate, args.duration)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
