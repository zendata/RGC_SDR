"""IQ sample sources.

Real hardware only -- see PLANNING.md section 1. SoapySDR is the abstraction seam, so a
single generic `SoapyIQSource` serves any installed driver (airspyhf, hackrf, ...) and
capabilities are probed at open time rather than hardcoded per radio.

This module must not import Qt or anything from `rgc_sdr.dsp` (layering rule, PLANNING.md
section 5).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# SoapySDR direction/error constants, redeclared so this module imports without the
# bindings present (they are only needed to actually open a device).
SOAPY_RX = 0
ERR_TIMEOUT = -1
ERR_OVERFLOW = -4

_SOAPY_HINT = (
    "SoapySDR Python bindings not found. Install with:\n"
    "    brew install soapysdr soapyairspyhf"
)


def _import_soapy():
    try:
        import SoapySDR  # type: ignore
    except ImportError:
        raise RuntimeError(_SOAPY_HINT) from None
    return SoapySDR


@dataclass(frozen=True)
class FreqRange:
    min_hz: float
    max_hz: float

    def contains(self, hz: float) -> bool:
        return self.min_hz <= hz <= self.max_hz


@dataclass(frozen=True)
class GainElement:
    """A named gain stage. The airspyhf driver reports none of these; HackRF does."""

    name: str
    min_db: float
    max_db: float
    step_db: float


@dataclass(frozen=True)
class DeviceCaps:
    """What a radio can actually do, probed from the driver.

    UI controls are built from this rather than from per-device assumptions, so adding a
    new SDR needs no code changes.
    """

    driver: str
    label: str
    serial: str
    sample_rates: tuple[float, ...]
    freq_ranges: tuple[FreqRange, ...]
    gain_elements: tuple[GainElement, ...]
    has_agc: bool
    formats: tuple[str, ...]

    def default_sample_rate(self, prefer: float = 768e3) -> float:
        """Pick `prefer` if offered, else the highest rate at or below it, else the lowest."""
        if not self.sample_rates:
            return prefer
        if prefer in self.sample_rates:
            return prefer
        below = [r for r in self.sample_rates if r <= prefer]
        return max(below) if below else min(self.sample_rates)

    def covers(self, hz: float) -> bool:
        return any(r.contains(hz) for r in self.freq_ranges)

    def describe_ranges(self) -> str:
        return ", ".join(f"{r.min_hz / 1e6:g}-{r.max_hz / 1e6:g} MHz" for r in self.freq_ranges)


def enumerate_devices(driver: str | None = None) -> list[dict[str, str]]:
    """List attached SDRs. Returns [] when none are present (not an error)."""
    SoapySDR = _import_soapy()
    args = f"driver={driver}" if driver else ""
    try:
        found = SoapySDR.Device.enumerate(args)
    except Exception:
        return []
    return [dict(d) for d in found]


class _Ring:
    """Single-producer/single-consumer ring buffer of complex samples.

    The reader thread writes; the GUI thread takes snapshots of the newest samples. The
    lock is held only around the index arithmetic and the copy, never around device I/O.
    """

    def __init__(self, capacity: int, dtype=np.complex64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buf = np.zeros(capacity, dtype=dtype)
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._buf.size

    def __len__(self) -> int:
        with self._lock:
            return self._filled

    def write(self, block: np.ndarray) -> None:
        n = block.size
        cap = self._buf.size
        if n == 0:
            return
        if n >= cap:
            # Block larger than the ring: keep only its newest `cap` samples.
            with self._lock:
                self._buf[:] = block[-cap:]
                self._write = 0
                self._filled = cap
            return
        with self._lock:
            end = self._write + n
            if end <= cap:
                self._buf[self._write : end] = block
            else:
                split = cap - self._write
                self._buf[self._write :] = block[:split]
                self._buf[: end - cap] = block[split:]
            self._write = end % cap
            self._filled = min(cap, self._filled + n)

    def read_latest(self, n: int) -> np.ndarray:
        """Newest `n` samples in chronological order, or fewer if not yet available."""
        cap = self._buf.size
        with self._lock:
            n = min(n, self._filled, cap)
            if n == 0:
                return np.empty(0, dtype=self._buf.dtype)
            start = (self._write - n) % cap
            if start + n <= cap:
                return self._buf[start : start + n].copy()
            split = cap - start
            out = np.empty(n, dtype=self._buf.dtype)
            out[:split] = self._buf[start:]
            out[split:] = self._buf[: n - split]
            return out


class IQSource(ABC):
    """A running stream of complex baseband samples."""

    @property
    @abstractmethod
    def caps(self) -> DeviceCaps: ...

    @property
    @abstractmethod
    def sample_rate(self) -> float: ...

    @property
    @abstractmethod
    def center_freq(self) -> float: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read_latest(self, n: int) -> np.ndarray:
        """Newest `n` samples, chronological. Short or empty if the stream is still filling."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class SoapyIQSource(IQSource):
    """Streams IQ from any SoapySDR-supported radio via a background reader thread.

    The reader thread is required, not an optimisation: the airspyhf driver returns 2048
    samples per `readStream` call (~375 calls/sec at 768 kHz), which cannot be serviced
    from the Qt event loop. See PLANNING.md section 3.
    """

    def __init__(
        self,
        driver: str = "airspyhf",
        serial: str | None = None,
        sample_rate: float | None = None,
        center_freq: float = 7.1e6,
        buffer_seconds: float = 0.5,
        read_size: int = 65536,
        agc: bool | None = None,
    ) -> None:
        SoapySDR = _import_soapy()
        self._soapy = SoapySDR

        # String args only: Device(dict(...)) fails with "Device::make() no match".
        spec = f"driver={driver}"
        if serial:
            spec += f",serial={serial}"
        self._dev = SoapySDR.Device(spec)

        self._caps = self._probe_caps(driver)
        self._rate = float(sample_rate or self._caps.default_sample_rate())
        self._dev.setSampleRate(SOAPY_RX, 0, self._rate)
        # Read back: the driver may quantise to a supported rate.
        self._rate = float(self._dev.getSampleRate(SOAPY_RX, 0))

        self._freq = float(center_freq)
        self._dev.setFrequency(SOAPY_RX, 0, self._freq)

        if agc is not None and self._caps.has_agc:
            self._dev.setGainMode(SOAPY_RX, 0, bool(agc))

        self._read_size = int(read_size)
        self._ring = _Ring(max(int(self._rate * buffer_seconds), self._read_size * 2))
        self._stream = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._overflows = 0
        self._timeouts = 0
        self._errors = 0
        self._total_samples = 0

    # -- capability probing -------------------------------------------------

    def _probe_caps(self, driver: str) -> DeviceCaps:
        d = self._dev

        def _safe(fn, default):
            try:
                return fn()
            except Exception:
                return default

        rates = tuple(float(r) for r in _safe(lambda: d.listSampleRates(SOAPY_RX, 0), ()))
        ranges = tuple(
            FreqRange(float(r.minimum()), float(r.maximum()))
            for r in _safe(lambda: d.getFrequencyRange(SOAPY_RX, 0), ())
        )
        gains = []
        for name in _safe(lambda: d.listGains(SOAPY_RX, 0), ()):
            gr = _safe(lambda: d.getGainRange(SOAPY_RX, 0, name), None)
            if gr is not None and gr.maximum() > gr.minimum():
                gains.append(
                    GainElement(str(name), float(gr.minimum()), float(gr.maximum()), float(gr.step()))
                )
        info = _safe(lambda: d.getHardwareInfo(), {})
        info = dict(info) if info else {}
        return DeviceCaps(
            driver=driver,
            label=str(info.get("label", _safe(lambda: d.getDriverKey(), driver))),
            serial=str(info.get("serial", "")),
            sample_rates=rates,
            freq_ranges=ranges,
            gain_elements=tuple(gains),
            has_agc=bool(_safe(lambda: d.hasGainMode(SOAPY_RX, 0), False)),
            formats=tuple(str(f) for f in _safe(lambda: d.getStreamFormats(SOAPY_RX, 0), ())),
        )

    # -- IQSource ----------------------------------------------------------

    @property
    def caps(self) -> DeviceCaps:
        return self._caps

    @property
    def sample_rate(self) -> float:
        return self._rate

    @property
    def center_freq(self) -> float:
        return self._freq

    def set_center_freq(self, hz: float) -> float:
        """Retune. Kept here for P2; the P1 UI only reads `center_freq`."""
        self._dev.setFrequency(SOAPY_RX, 0, float(hz))
        self._freq = float(self._dev.getFrequency(SOAPY_RX, 0))
        return self._freq

    def set_agc(self, enabled: bool) -> None:
        """Enable/disable hardware AGC. No-op when the driver has no gain mode."""
        if self._caps.has_agc:
            self._dev.setGainMode(SOAPY_RX, 0, bool(enabled))

    def get_agc(self) -> bool:
        if not self._caps.has_agc:
            return False
        try:
            return bool(self._dev.getGainMode(SOAPY_RX, 0))
        except Exception:
            return False

    def set_gain(self, name: str, db: float) -> None:
        """Set a named gain element. airspyhf exposes none; HackRF and others do."""
        self._dev.setGain(SOAPY_RX, 0, name, float(db))

    def get_gain(self, name: str) -> float:
        try:
            return float(self._dev.getGain(SOAPY_RX, 0, name))
        except Exception:
            return 0.0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "samples": self._total_samples,
            "overflows": self._overflows,
            "timeouts": self._timeouts,
            "errors": self._errors,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stream = self._dev.setupStream(SOAPY_RX, "CF32")
        self._dev.activateStream(self._stream)
        self._running.set()
        self._thread = threading.Thread(target=self._reader, name="iq-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        buf = np.empty(self._read_size, dtype=np.complex64)
        while self._running.is_set():
            try:
                sr = self._dev.readStream(self._stream, [buf], buf.size, timeoutUs=500_000)
            except Exception:
                self._errors += 1
                continue
            ret = sr.ret
            if ret > 0:
                # Only the first `ret` entries are valid; the tail is uninitialised.
                self._ring.write(buf[:ret])
                self._total_samples += ret
            elif ret == ERR_OVERFLOW:
                self._overflows += 1
            elif ret == ERR_TIMEOUT:
                self._timeouts += 1
            else:
                self._errors += 1

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._dev.deactivateStream(self._stream)
                self._dev.closeStream(self._stream)
            finally:
                self._stream = None

    def read_latest(self, n: int) -> np.ndarray:
        return self._ring.read_latest(n)
