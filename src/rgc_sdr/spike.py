"""P0 spike: open Airspy HF+, stream IQ, report stats.

Usage:
    python -m src.rgc_sdr.device.spike [--duration SECONDS]

Without hardware available, a synthesised IQ stream is used when run with
--simulate, so the pipeline is testable off-device.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class StreamStats:
    samples: int = 0
    duration_s: float = 0.0
    start: float = 0.0
    overflows: int = 0

    def rate(self) -> float:
        return self.samples / self.duration_s if self.duration_s > 0 else 0.0

    def snr_db(self, last_block: np.ndarray) -> float:
        peak = np.max(np.abs(last_block))
        noise = np.std(last_block)
        return float(20 * np.log10(peak / noise)) if noise > 0 else float("inf")


def stream_device(duration_s: float, plant_samples=None) -> StreamStats:
    """Stream from the Airspy HF+ via SoapySDR if available.

    If SoapySDR is missing or no device is present and `plant_samples` is
    provided (simulation), those samples are streamed instead.
    """
    try:
        import SoapySDR  # type: ignore
        sdr = SoapySDR.Device("driver=airspyhf")
    except ImportError:
        raise RuntimeError(
            "SoapySDR Python bindings not found. Install with: brew install soapysdr soapyairspyhf"
        ) from None
    except Exception as exc:
        # SoapySDR is installed but the device failed to open
        print(f"Warning: Airspy HF+ detected but failed to open: {exc}")
        sdr = None

    stats = StreamStats(start=time.time())

    if sdr is not None:
        sample_rate = 768_000
        rx = sdr.setupStream(1, "CF32")
        sdr.activateStream(rx)
        buf = np.empty(65536, dtype=np.complex64)
        end = time.time() + duration_s
        while time.time() < end:
            ret = sdr.readStream(rx, [buf], len(buf))
            stats.samples += ret.ret
            stats.last_block = buf[: ret.ret]
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)
        stats.duration_s = time.time() - stats.start
        return stats

    if plant_samples is None:
        raise RuntimeError("Airspy HF+ not found and no simulated samples supplied.")

    virtual_rate = 768_000  # Hz, mimics Airspy HF+ delivery rate
    pos = 0
    end = time.time() + duration_s
    delivered = 0.0
    while time.time() < end:
        take = min(len(plant_samples) - pos, 65536) or len(plant_samples)
        block = plant_samples[pos : pos + take]
        pos = (pos + take) % len(plant_samples)
        stats.samples += take
        stats.last_block = block
        delivered += take
        target_elapsed = delivered / virtual_rate
        actual = time.time() - stats.start
        if target_elapsed > actual:
            time.sleep(target_elapsed - actual)
    stats.duration_s = time.time() - stats.start
    return stats


def synthesize_iq(n=1_000_000, fs=768_000) -> np.ndarray:
    t = np.arange(n) / fs
    signal = (0.6 * np.cos(2 * np.pi * 12_000 * t) + 0.25 * np.cos(2 * np.pi * 45_000 * t))
    noise = 0.05 * (np.random.randn(n) + 1j * np.random.randn(n))
    return (signal.astype(np.complex64) + noise.astype(np.complex64)).astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=2.0, help="seconds to stream")
    parser.add_argument("--simulate", action="store_true", help="use synthesized IQ")
    args = parser.parse_args()

    sim = synthesize_iq() if args.simulate else None
    stats = stream_device(args.duration, plant_samples=sim)
    print(f"Samples: {stats.samples:,}")
    print(f"Duration: {stats.duration_s:.2f} s")
    print(f"Rate: {stats.rate()/1e3:.1f} kS/s")
    print(f"SNR (last block): {stats.snr_db(stats.last_block):.1f} dB")


if __name__ == "__main__":
    main()
