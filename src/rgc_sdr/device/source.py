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
import math
from dataclasses import dataclass

import numpy as np

# SoapySDR direction/error constants, redeclared so this module imports without the
# bindings present (they are only needed to actually open a device). TX is 0 and RX is
# 1 (SOAPY_SDR_TX / SOAPY_SDR_RX). This was once 0: the Airspy HF+ driver ignores the
# direction so nothing showed, but the HackRF opened a *transmit* stream, every read
# failed with NOT_SUPPORTED, and the gains offered were its TX gains.
SOAPY_RX = 1
ERR_TIMEOUT = -1
ERR_OVERFLOW = -4

#: Floor on ring capacity, independent of sample rate. Deep zoom needs
#: `fft_size * factor` input samples for a single FFT -- 16384 x 32 plus filter overhead
#: is about 530k -- which exceeds a second's worth at the lower sample rates.
MIN_RING_SAMPLES = 1_200_000

class SoapyUnavailable(RuntimeError):
    """The SoapySDR bindings are not importable at all.

    Distinct from a failure to *open* a device, which Soapy also reports as a plain
    RuntimeError -- so without its own type, a mistyped driver name produced an
    "install SoapySDR" hint instead of "could not open".
    """


_SOAPY_HINT = (
    "SoapySDR Python bindings not found. Install with:\n"
    "    brew install soapysdr soapyairspyhf"
)


def _import_soapy():
    try:
        import SoapySDR  # type: ignore
    except ImportError:
        raise SoapyUnavailable(_SOAPY_HINT) from None
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
class SettingInfo:
    """An on/off driver setting, such as a bias-tee.

    Setting names differ between drivers, so rather than hardcode each one the UI offers
    whatever boolean settings the driver itself advertises.
    """

    key: str
    name: str
    description: str = ""
    default: bool = False


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
    # Probed, not assumed: libairspyhf ties IF bandwidth to sample rate and may expose
    # nothing here, while other radios offer a list. Empty means "no bandwidth control".
    bandwidths: tuple[float, ...] = ()
    #: Boolean driver settings (bias-tee and the like), offered as checkboxes.
    settings: tuple[SettingInfo, ...] = ()

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

    def clamp_freq(self, hz: float) -> float:
        """Nearest tunable frequency.

        The Airspy HF+ has two disjoint ranges (0.009-31 and 60-260 MHz), so a frequency
        can be between them rather than merely out of bounds. Snap to the nearest edge of
        the nearest range instead of silently accepting an untunable value.
        """
        if not self.freq_ranges or self.covers(hz):
            return float(hz)
        edges = [e for r in self.freq_ranges for e in (r.min_hz, r.max_hz)]
        return float(min(edges, key=lambda e: abs(e - hz)))

    def nearest_sample_rate(self, hz: float) -> float:
        if not self.sample_rates:
            return float(hz)
        return float(min(self.sample_rates, key=lambda r: abs(r - hz)))

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
        # Monotonic count of samples ever written, so a sequential reader can hold an
        # absolute cursor and tell whether it has been lapped. `_valid_from` marks where
        # usable data starts after a clear (retune), since the count never rewinds.
        self._total = 0
        self._valid_from = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._buf.size

    def __len__(self) -> int:
        with self._lock:
            return self._filled

    def clear(self) -> None:
        """Discard the contents without disturbing the write position.

        `_write` is deliberately left alone: SequentialReader maps an absolute sample
        index to `index % capacity`, and rewinding the pointer to 0 breaks that, so a
        reader resyncing after a retune would be handed pre-retune samples.
        """
        with self._lock:
            self._filled = 0
            self._valid_from = self._total

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total

    def write(self, block: np.ndarray) -> None:
        n = block.size
        cap = self._buf.size
        if n == 0:
            return
        if n >= cap:
            # Block larger than the ring: keep only its newest `cap` samples, placed
            # where their absolute indices require so `index % capacity` stays true.
            with self._lock:
                self._total += n
                start = self._total % cap
                tail = block[-cap:]
                split = cap - start
                self._buf[start:] = tail[:split]
                self._buf[:start] = tail[split:]
                self._write = start
                self._filled = cap
                self._valid_from = self._total - cap
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
            self._total += n

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


class SequentialReader:
    """Gapless reader over a ring, for a continuous consumer such as audio.

    `read_latest` is lossy on purpose: the display only ever wants the newest frame, and
    skipping is invisible. Audio cannot skip -- every dropped sample is an audible click --
    so this keeps its own absolute cursor and advances it one block at a time. If the
    writer laps it (the consumer fell behind, or a retune cleared the ring) it resyncs to
    the oldest still-valid sample and counts what it lost, rather than reading stale or
    torn data.
    """

    def __init__(self, ring: "_Ring") -> None:
        self._ring = ring
        self._cursor = ring.total_written
        self.lost = 0

    def available(self) -> int:
        ring = self._ring
        with ring._lock:
            self._resync_locked()
            return max(0, ring._total - self._cursor)

    def _resync_locked(self) -> None:
        """Move the cursor forward if it points at data that is gone. Caller holds lock."""
        ring = self._ring
        oldest = max(ring._valid_from, ring._total - ring._filled)
        if self._cursor < oldest:
            self.lost += oldest - self._cursor
            self._cursor = oldest

    def read(self, n: int) -> np.ndarray:
        """Next `n` samples in order, or empty if that many are not yet available."""
        ring = self._ring
        cap = ring._buf.size
        with ring._lock:
            self._resync_locked()
            if ring._total - self._cursor < n:
                return np.empty(0, dtype=ring._buf.dtype)
            start = self._cursor % cap
            if start + n <= cap:
                out = ring._buf[start : start + n].copy()
            else:
                split = cap - start
                out = np.empty(n, dtype=ring._buf.dtype)
                out[:split] = ring._buf[start:]
                out[split:] = ring._buf[: n - split]
            self._cursor += n
            return out

    def skip_to_latest(self) -> None:
        ring = self._ring
        with ring._lock:
            self._cursor = ring._total


#: How far a radio with a DC spike is tuned away from the wanted frequency. The stream is
#: shifted back in software, so the spike lands this far from centre instead of on top of
#: whatever is being listened to. Measured on a HackRF at 105.1 MHz: at centre the stereo
#: pilot was unusable (coherence 0.09); off the spike it locked.
LO_OFFSET_HZ = 200e3


class _Nco:
    """Phase-continuous frequency shift of a complex stream, in place.

    Uses one period of the complex exponential as a lookup table when the shift divides
    the rate into a short cycle -- 200 kHz does into every supported rate (4 MS/s: 20
    samples) -- which costs a multiply per sample instead of an exp. Otherwise falls back
    to computing the phasor from a running phase.
    """

    MAX_TABLE = 1 << 20

    def __init__(self, shift_hz: float, rate: float) -> None:
        self.shift_hz = float(shift_hz)
        self.rate = float(rate)
        self._n = 0
        self._table = None
        self._phase = 0.0
        if self.shift_hz == 0.0:
            return
        r, f = int(round(self.rate)), int(round(abs(self.shift_hz)))
        if r == self.rate and f == abs(self.shift_hz):
            period = r // math.gcd(r, f)
            if period <= self.MAX_TABLE:
                n = np.arange(period)
                self._table = np.exp(2j * np.pi * self.shift_hz * n / self.rate).astype(
                    np.complex64)

    def process(self, x: np.ndarray) -> None:
        if self.shift_hz == 0.0 or x.size == 0:
            return
        if self._table is not None:
            period = self._table.size
            idx = np.arange(self._n, self._n + x.size) % period
            x *= self._table[idx]
            self._n = (self._n + x.size) % period
            return
        step = 2 * np.pi * self.shift_hz / self.rate
        phases = self._phase + step * np.arange(x.size)
        x *= np.exp(1j * phases).astype(np.complex64)
        self._phase = float((self._phase + step * x.size) % (2 * np.pi))


class IQSource(ABC):
    """A running stream of complex baseband samples."""

    #: Where the radio's DC spike appears, relative to `center_freq`. Zero for a radio
    #: without one, or one listened to at centre.
    dc_spike_offset_hz: float = 0.0

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

    def set_center_freq(self, hz: float, flush: bool = True) -> float:
        """Retune. `flush` discards buffered samples; see SoapyIQSource for why a fine
        adjustment should not."""
        raise NotImplementedError

    def set_sample_rate(self, hz: float) -> float:
        raise NotImplementedError

    def sequential_reader(self) -> SequentialReader:
        """A gapless reader, for consumers that cannot skip samples.

        Part of the contract because both audio and IQ recording need it; `read_latest`
        is deliberately lossy and is not a substitute.
        """
        raise NotImplementedError

    #: The profile of the radio behind this source, when it is a known one.
    profile = None

    def close(self) -> None:
        """Release the underlying device. Stops streaming first."""
        self.stop()

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
        buffer_seconds: float = 1.0,
        read_size: int = 65536,
        agc: bool | None = None,
        settle_seconds: float = 0.15,
    ) -> None:
        SoapySDR = _import_soapy()
        self._soapy = SoapySDR

        # String args only: Device(dict(...)) fails with "Device::make() no match".
        spec = f"driver={driver}"
        if serial:
            spec += f",serial={serial}"
        self._dev = SoapySDR.Device(spec)

        from .profiles import profile_for, refine_caps

        self.profile = profile_for(driver)
        self._caps = refine_caps(self._probe_caps(driver), self.profile)
        prefer = self.profile.default_rate if self.profile else 768e3
        self._rate = float(sample_rate or self._caps.default_sample_rate(prefer))
        self._dev.setSampleRate(SOAPY_RX, 0, self._rate)
        # Read back: the driver may quantise to a supported rate.
        self._rate = float(self._dev.getSampleRate(SOAPY_RX, 0))

        # Radios with a DC spike are tuned LO_OFFSET_HZ away and shifted back (_Nco).
        self._lo_offset = LO_OFFSET_HZ if (self.profile and self.profile.dc_offset) else 0.0
        self._nco = _Nco(0.0, self._rate)
        self._freq = float(center_freq)
        self._tune_hardware(self._caps.clamp_freq(self._freq))

        if agc is not None and self._caps.has_agc:
            self._dev.setGainMode(SOAPY_RX, 0, bool(agc))
        names = {g.name for g in self._caps.gain_elements}
        for name, db in (self.profile.default_gains if self.profile else ()):
            if name in names:
                self._dev.setGain(SOAPY_RX, 0, name, float(db))

        self._read_size = int(read_size)
        self._buffer_seconds = float(buffer_seconds)
        self._settle_seconds = float(settle_seconds)
        self._ring = _Ring(self._ring_capacity())
        self._stream = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._overflows = 0
        self._timeouts = 0
        self._errors = 0
        self._total_samples = 0
        self._read_samples = 0
        self._dropped = 0
        # Measured 2026-09-23: the first ~4 frames after a stream restart read about 8 dB
        # hot broadband (peak -93.5 dBFS against -110 steady) while the front end settles,
        # which paints one bright row across the waterfall and skews auto-ranging.
        # `_drop_until` is only ever written from the calling thread and `_read_samples`
        # only from the reader, so no lock is needed.
        self._drop_until = 0

    # -- capability probing -------------------------------------------------

    def _probe_caps(self, driver: str) -> DeviceCaps:
        d, S = self._dev, self._soapy

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
        bandwidths = tuple(
            float(b) for b in _safe(lambda: d.listBandwidths(SOAPY_RX, 0), ()) if float(b) > 0
        )
        # Verified, not trusted. SoapyAirspyHF reports hasGainMode() == True and then
        # ignores setGainMode entirely: measured 2026-09-23, setting it False leaves
        # getGainMode() reporting True and the received level unchanged to within 0.05 dB.
        # A control that does nothing is worse than no control, so the claim is tested by
        # trying to change it and putting it back.
        has_agc = bool(_safe(lambda: d.hasGainMode(SOAPY_RX, 0), False))
        if has_agc:
            original = _safe(lambda: bool(d.getGainMode(SOAPY_RX, 0)), None)
            if original is None:
                has_agc = False
            else:
                try:
                    d.setGainMode(SOAPY_RX, 0, not original)
                    has_agc = bool(d.getGainMode(SOAPY_RX, 0)) != original
                    d.setGainMode(SOAPY_RX, 0, original)
                except Exception:
                    has_agc = False

        settings = []
        for arg in _safe(lambda: d.getSettingInfo(), ()):
            try:
                if arg.type != S.ArgInfo.BOOL:
                    continue
                settings.append(SettingInfo(
                    str(arg.key), str(arg.name or arg.key), str(arg.description or ""),
                    str(arg.value).strip().lower() == "true",
                ))
            except Exception:
                continue

        # Rates reported only as a continuous range (Pluto does this) come back empty
        # from listSampleRates; the profile fills that gap in refine_caps.
        info = _safe(lambda: d.getHardwareInfo(), {})
        info = dict(info) if info else {}
        return DeviceCaps(
            driver=driver,
            label=str(info.get("label", _safe(lambda: d.getDriverKey(), driver))),
            serial=str(info.get("serial", "")),
            sample_rates=rates,
            freq_ranges=ranges,
            gain_elements=tuple(gains),
            has_agc=has_agc,
            formats=tuple(str(f) for f in _safe(lambda: d.getStreamFormats(SOAPY_RX, 0), ())),
            bandwidths=bandwidths,
            settings=tuple(settings),
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

    def _ring_capacity(self) -> int:
        return max(
            int(self._rate * self._buffer_seconds), self._read_size * 2, MIN_RING_SAMPLES
        )

    def _arm_settle(self) -> None:
        """Discard the next `settle_seconds` of samples."""
        self._drop_until = self._read_samples + int(self._rate * self._settle_seconds)

    def set_center_freq(self, hz: float, flush: bool = True) -> float:
        """Retune, clamped to a tunable range.

        With `flush` the ring is dropped and a settle period armed, because whatever the
        buffer holds was received at the old frequency and rendering it smears stale
        signal across the new span.

        Pass `flush=False` for a small adjustment. Discarding ~150 ms on every step made
        fine tuning unusable: 30 nudges of 100 Hz cost 3.7 seconds of audio silence,
        measured. A 100 Hz error inside a several-kHz channel is inaudible, so the
        buffered samples stay valid and the audio runs on. (Measured at P2: an LO change
        on a live stream shows no transient; only a *rate* change, which restarts the
        stream, needs the settle.)
        """
        self._tune_hardware(self._caps.clamp_freq(float(hz)))
        if flush:
            self._ring.clear()
            self._arm_settle()
        return self._freq

    def _tune_hardware(self, wanted: float) -> None:
        """Tune so that `wanted` ends up at the centre of the stream we deliver.

        With an LO offset the hardware sits above the wanted frequency (below it at the
        top of the radio's range) and the NCO shifts the stream by the difference, so
        the spike lands at +/-LO_OFFSET_HZ on the display.
        """
        shift = self._lo_offset
        if shift and not self._caps.covers(wanted + shift):
            shift = -shift
        self._dev.setFrequency(SOAPY_RX, 0, wanted + shift)
        hardware = float(self._dev.getFrequency(SOAPY_RX, 0))
        self._freq = hardware - shift
        if shift != self._nco.shift_hz or self._nco.rate != self._rate:
            # Swapped whole, so the reader thread only ever sees a complete NCO.
            self._nco = _Nco(shift, self._rate)

    @property
    def dc_spike_offset_hz(self) -> float:
        return self._nco.shift_hz

    @property
    def bandwidth(self) -> float:
        try:
            return float(self._dev.getBandwidth(SOAPY_RX, 0))
        except Exception:
            return 0.0

    def set_bandwidth(self, hz: float) -> float:
        """Set IF bandwidth. No-op when the driver reports no bandwidth options."""
        if not self._caps.bandwidths:
            return self.bandwidth
        self._dev.setBandwidth(SOAPY_RX, 0, float(hz))
        return self.bandwidth

    def set_sample_rate(self, hz: float) -> float:
        """Change sample rate, restarting the stream around it.

        SoapySDR will not accept a rate change on an active stream, so the reader thread
        is stopped and restarted. The ring is rebuilt because its capacity is derived
        from the rate, and its contents were sampled at the old one.
        """
        target = self._caps.nearest_sample_rate(float(hz))
        if target == self._rate:
            return self._rate
        was_running = self._thread is not None
        if was_running:
            self.stop()
        self._dev.setSampleRate(SOAPY_RX, 0, target)
        self._rate = float(self._dev.getSampleRate(SOAPY_RX, 0))
        self._ring = _Ring(self._ring_capacity())
        self._nco = _Nco(self._nco.shift_hz, self._rate)
        if was_running:
            self.start()
        return self._rate

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

    def read_setting(self, key: str) -> bool:
        try:
            return str(self._dev.readSetting(key)).strip().lower() == "true"
        except Exception:
            return False

    def write_setting(self, key: str, enabled: bool) -> None:
        """Set a boolean driver setting such as a bias-tee."""
        self._dev.writeSetting(key, "true" if enabled else "false")

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
            "dropped": self._dropped,
            "overflows": self._overflows,
            "timeouts": self._timeouts,
            "errors": self._errors,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stream = self._dev.setupStream(SOAPY_RX, "CF32")
        self._dev.activateStream(self._stream)
        self._arm_settle()
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
                self._read_samples += ret
                if self._read_samples <= self._drop_until:
                    self._dropped += ret
                    continue
                self._nco.process(buf[:ret])       # undo the LO offset, if any
                self._ring.write(buf[:ret])
                self._total_samples += ret
            elif ret == ERR_OVERFLOW:
                self._overflows += 1
            elif ret == ERR_TIMEOUT:
                self._timeouts += 1
            else:
                self._errors += 1

    def stop(self) -> None:
        if self._dev is None:
            return
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

    def close(self) -> None:
        """Stop streaming and release the USB device.

        stop() alone leaves the device open until the object is garbage collected, and a
        second open of the same radio then fails with "Unable to open" -- which is what
        switching back to a radio would do.
        """
        self.stop()
        dev, self._dev = self._dev, None
        if dev is not None:
            # The binding's own close(), not Device.unmake(): its destructor calls
            # close() too, and unmaking behind its back made that a double free.
            try:
                dev.close()
            except Exception:
                pass

    def sequential_reader(self) -> SequentialReader:
        """A gapless reader for audio. Independent of the display's lossy reads."""
        return SequentialReader(self._ring)
