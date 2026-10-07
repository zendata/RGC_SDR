"""Classify everything strong on the screen at once: the "Classify" button.

Asked for by VK3RQ (2026-10-06): sweep the span the waterfall shows, wherever the radio
is tuned, and label up to five (twelve since 2026-10-07) strong, clean signals; skip what cannot be identified;
merge a trunked system's channels into one label, colour-coded.

1. **One capture** of the whole span (`classify.CAPTURE_S`, capped at MAX_SAMPLES).
2. **Candidates** from its 50 ms max-hold spectrum: peaks at least CANDIDATE_SNR_DB over
   the floor (a max-hold sits ~15 dB above the mean noise, so 20 dB keeps noise out),
   inside the visible span, away from the span's edges and the radio's own DC spike,
   each with its width; strongest first.
3. **One FFT channeliser**: the capture is transformed once, and each candidate becomes
   a narrow stream by taking its bins and transforming back -- a sharp filter for the
   price of a small inverse FFT, where mixing and decimating 12 million samples per
   candidate would take seconds each.
4. **The "?" classifier** (`classify.classify`) on each stream, strongest first, until
   MAX_LABELS labels are filled; "Unknown" is skipped.
5. **Merging**: P25 channels with one NAC, DMR channels with one colour code, become one
   label at the strongest, with markers in the same colour on the rest.

ADS-B is pulses a microsecond long: it never stands out in an averaged spectrum and
needs raw samples, so when 1090 MHz is on screen it is decoded from the capture itself.

Pure NumPy and the app's DSP; `SweepJob` drives a source on a thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .classify import (
    CAPTURE_S, MAX_SAMPLES, Classification, _smooth, _spectrum, _walk, classify,
)
from .dsp.adsb import FREQUENCY_HZ as ADSB_HZ
from .dsp.adsb import AdsbDecoder
from .dsp.demod import Mixer

#: Labels shown at most, after merging.
MAX_LABELS = 12
#: Candidates classified at most, strongest first, to fill those labels.
MAX_CANDIDATES = 24
CANDIDATE_SNR_DB = 20.0
#: Fraction of the span at each edge left out: the radio's own filter rolls off there.
EDGE_FRACTION = 0.05
#: Left out either side of the radio's DC spike.
SPIKE_GUARD_HZ = 5e3
#: Stream rate per candidate: room for a 12.5-25 kHz channel and its decoders; wide
#: signals (broadcast FM) get WIDE_RATE.
NARROW_RATE = 250e3
WIDE_RATE = 500e3
#: Label colours: the first for a signal on its own, the rest one per merged system.
COLOURS = ("#FFE45C", "#7FDBFF", "#FF9F80", "#B5E655", "#D7A6FF", "#FFB3D9")


#: Channel grids labels snap to: broadcast FM's 100 kHz, else land mobile's 6.25 kHz
#: (12.5 and 25 kHz channels sit on it too) when the measured centre is close. A centre
#: measured from the spectrum carries the radio's error and the signal's own asymmetry
#: (measured: 97.4948 for 97.5 MHz on an uncorrected HackRF).
BROADCAST_FM = (87.5e6, 108.0e6)
GRID_HZ = 6.25e3
GRID_SNAP_HZ = 1.5e3


#: Airband channels are 25 kHz apart, or 8.33 kHz (a third of 25) where split.
AIRBAND = (118.0e6, 137.0e6)
#: Two candidates closer than a channel are one signal (a P25 signal's spectral nulls
#: can dip to the floor and split it).
MIN_SEPARATION_HZ = 12.5e3


def on_grid(freq_hz: float, label: str) -> float:
    if label == "WBFM" and BROADCAST_FM[0] <= freq_hz <= BROADCAST_FM[1]:
        return round(freq_hz / 100e3) * 100e3
    grid = 25e3 / 3 if AIRBAND[0] <= freq_hz <= AIRBAND[1] else GRID_HZ
    nearest = round(freq_hz / grid) * grid
    return float(round(nearest if abs(nearest - freq_hz) <= GRID_SNAP_HZ
                       else round(freq_hz, -2)))


#: Judgements a sweep cannot stand behind, so it skips them: "Unknown", and the
#: sidebands -- which side of the carrier needs the carrier's frequency, which only a
#: tuned listener knows; measured from the centre of the occupied band, noise passed
#: for LSB.
UNNAMED = ("Unknown", "USB", "LSB")


@dataclass
class Candidate:
    freq_hz: float
    width_hz: float
    snr_db: float


@dataclass
class Label:
    freq_hz: float
    text: str
    colour: str
    #: Other channels of the same system: marked, not labelled.
    others: list[float] = field(default_factory=list)


def find_candidates(iq, rate, centre_hz, lo_hz, hi_hz, spike_offset_hz=0.0,
                    most=MAX_CANDIDATES) -> list[Candidate]:
    """Strong peaks inside [lo_hz, hi_hz], strongest first, each with its width."""
    mean_db, hold_db, offsets = _spectrum(iq, rate)
    step = offsets[1] - offsets[0]
    hold = _smooth(hold_db, 1000.0 / step)
    floor = float(np.median(mean_db))
    freqs = centre_hz + offsets
    allowed = ((freqs >= lo_hz) & (freqs <= hi_hz)
               & (np.abs(offsets) <= rate * (0.5 - EDGE_FRACTION)))
    if spike_offset_hz:
        allowed &= np.abs(offsets - spike_offset_hz) > SPIKE_GUARD_HZ
    out: list[Candidate] = []
    level = np.where(allowed, hold, -np.inf)
    while len(out) < most:
        k = int(np.argmax(level))
        snr = float(level[k] - floor)
        if not np.isfinite(snr) or snr < CANDIDATE_SNR_DB:
            break
        edge = max(hold[k] - 20.0, floor + 6.0)
        lo, hi = _walk(hold, k, edge, -1, floor), _walk(hold, k, edge, +1, floor)
        width = (hi - lo + 1) * step
        freq = float(centre_hz + (offsets[lo] + offsets[hi]) / 2)
        guard = int(max(2e3, 0.25 * width) / step)
        level[max(0, lo - guard): hi + guard + 1] = -np.inf
        if any(abs(freq - c.freq_hz) < max(MIN_SEPARATION_HZ, (width + c.width_hz) / 2)
               for c in out):
            continue
        out.append(Candidate(freq, float(width), snr))
    return out


class Channeliser:
    """One FFT of a whole capture; any stretch of it back out as a narrow stream."""

    def __init__(self, iq: np.ndarray, rate: float, centre_hz: float) -> None:
        self.rate, self.centre_hz = float(rate), float(centre_hz)
        self.n = int(iq.size)
        self._spectrum = np.fft.fft(np.asarray(iq, dtype=np.complex64))

    def channel(self, freq_hz: float, out_rate: float) -> tuple[np.ndarray, float]:
        """The signal at `freq_hz` moved to 0 Hz, at about `out_rate` (returned)."""
        m = min(self.n, int(round(out_rate * self.n / self.rate / 2)) * 2)
        k0 = int(round((freq_hz - self.centre_hz) * self.n / self.rate))
        bins = (k0 + np.arange(-m // 2, m // 2)) % self.n
        z = np.fft.ifft(np.fft.ifftshift(self._spectrum[bins])) * (m / self.n)
        return z.astype(np.complex64), self.rate * m / self.n


def _adsb(iq, rate, centre_hz) -> Classification | None:
    if not (abs(ADSB_HZ - centre_hz) < rate * (0.5 - EDGE_FRACTION) and rate >= 2e6
            and abs(rate / 1e6 - round(rate / 1e6)) < 1e-6):
        return None
    mixer, decoder = Mixer(rate, ADSB_HZ - centre_hz), AdsbDecoder(rate)
    found = []
    block = int(rate * 0.05)
    for i in range(0, iq.size, block):
        found += decoder.process(mixer.process(iq[i:i + block]))
    if not found:
        return None
    return Classification(ADSB_HZ, "ADS-B", f"{len({m.icao for m in found})} aircraft",
                          certain=True)


def merge(results: list[Classification]) -> list[Label]:
    """One label per signal, or per trunked system (its strongest channel, with the
    rest marked), each system in its own colour. `results` strongest first."""
    labels: list[Label] = []
    systems: dict[str, Label] = {}
    for r in results:
        if r.system and r.system in systems:
            systems[r.system].others.append(r.freq_hz)
            continue
        colour = COLOURS[0]
        if r.system:
            colour = COLOURS[1 + len(systems) % (len(COLOURS) - 1)]
        label = Label(r.freq_hz, r.text(), colour)
        labels.append(label)
        if r.system:
            systems[r.system] = label
    for label in systems.values():
        if label.others:
            count = len(label.others) + 1
            label.text += f"  +{count - 1} channel{'s' if count > 2 else ''}"
    return labels


def sweep(iq, rate, centre_hz, lo_hz, hi_hz, spike_offset_hz=0.0,
          on_result=None, cancelled=lambda: False) -> list[Classification]:
    """Classify the strong signals in [lo_hz, hi_hz]; strongest first, unknowns left
    out, stopping once MAX_LABELS labels are filled. `on_result` hears each as found."""
    iq = np.asarray(iq, dtype=np.complex64)
    results: list[Classification] = []

    def keep(r: Classification) -> None:
        results.append(r)
        if on_result is not None:
            on_result(r)

    if lo_hz <= ADSB_HZ <= hi_hz:
        found = _adsb(iq, rate, centre_hz)
        if found is not None:
            keep(found)
    channeliser = None
    for cand in find_candidates(iq, rate, centre_hz, lo_hz, hi_hz, spike_offset_hz):
        if cancelled() or len(merge(results)) >= MAX_LABELS:
            break
        if abs(cand.freq_hz - ADSB_HZ) < 1e6 and any(r.label == "ADS-B" for r in results):
            continue
        if channeliser is None:
            channeliser = Channeliser(iq, rate, centre_hz)
        out_rate = WIDE_RATE if cand.width_hz > 100e3 else NARROW_RATE
        z, fs = channeliser.channel(cand.freq_hz, min(out_rate, rate))
        found = classify(z, fs, cand.freq_hz, cand.freq_hz)
        if found is None or found.label in UNNAMED:
            continue
        found.snr_db = cand.snr_db
        found.freq_hz = on_grid(found.freq_hz, found.label)
        keep(found)
    return results


class SweepJob:
    """Capture the source's span and sweep [lo_hz, hi_hz] on a thread. `results` grows
    as signals are classified (strongest first); poll `done`, read `labels()`."""

    def __init__(self, source, lo_hz: float, hi_hz: float) -> None:
        import threading

        self.source = source
        self.lo_hz, self.hi_hz = float(lo_hz), float(hi_hz)
        self.centre_hz = float(source.center_freq)
        self.rate = float(source.sample_rate)
        self.spike_offset_hz = float(getattr(source, "dc_spike_offset_hz", 0.0) or 0.0)
        self.results: list[Classification] = []
        self.error = ""
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._cancel = threading.Event()
        self._reader = source.sequential_reader()
        self._thread = threading.Thread(target=self._run, name="sweep", daemon=True)
        self._thread.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def labels(self) -> list[Label]:
        with self._lock:
            return merge(list(self.results))

    def _found(self, result: Classification) -> None:
        with self._lock:
            self.results.append(result)

    def _run(self) -> None:
        import time

        try:
            wanted = min(int(CAPTURE_S * self.rate), MAX_SAMPLES)
            chunks, have = [], 0
            deadline = time.monotonic() + CAPTURE_S * 3 + 2
            while have < wanted and not self._cancel.is_set():
                if time.monotonic() > deadline:
                    raise TimeoutError("the radio sent too few samples")
                n = self._reader.available()
                if n == 0:
                    time.sleep(0.02)
                    continue
                piece = self._reader.read(min(n, wanted - have))
                chunks.append(piece)
                have += piece.size
            if not self._cancel.is_set():
                iq = np.concatenate(chunks)
                del chunks
                sweep(iq, self.rate, self.centre_hz, self.lo_hz, self.hi_hz,
                      self.spike_offset_hz, on_result=self._found,
                      cancelled=self._cancel.is_set)
        except Exception as exc:                      # shown, never fatal
            self.error = str(exc) or type(exc).__name__
        finally:
            self._done.set()
