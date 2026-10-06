"""What is this signal? A few seconds of IQ around the tuned frequency in, a name out.

Asked for by VK3RQ (2026-10-06): a button that classifies the signal at the listening
frequency and labels it on the waterfall. Two stages:

1. **Protocols, by decoding.** Each decoder the app has is run over the capture; one
   that finds frames passing their own error checks (a P25 NID, a DMR slot type, a
   POCSAG sync and batch, an AX.25, AIS or ACARS CRC, a Mode S parity) names the signal
   beyond doubt. Nothing decoded is shown -- pager text least of all.
2. **Modulation, by measurement**, when no decoder recognises it: occupied bandwidth,
   how constant the envelope is, how much power sits in a carrier, which side of the
   tuned frequency the sideband is on, and for broadcast FM the 19 kHz stereo pilot and
   57 kHz RDS. That separates WBFM, NBFM, FSK data, AM, USB, LSB, CW and a plain
   carrier. These are judgements from features, and say so less firmly.

Pure NumPy/SciPy and the app's own DSP; no Qt, no device access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp.acars import AcarsDecoder
from .dsp.adsb import FREQUENCY_HZ as ADSB_HZ
from .dsp.adsb import AdsbDecoder
from .dsp.ais import CHANNELS as AIS_CHANNELS
from .dsp.ais import AisDecoder
from .dsp.aprs import AprsDecoder
from .dsp.demod import DemodChain, Mixer
from .dsp.dmr import DmrDecoder
from .dsp.p25 import P25Decoder
from .dsp.pocsag import PocsagDecoder

#: Seconds of IQ to look at: long enough for a 512-baud POCSAG batch (1.06 s) to land
#: whole, and for a few P25 or DMR frames.
CAPTURE_S = 2.5
#: Most samples captured, whatever the rate (6 MS/s x 2 s, about 100 MB).
MAX_SAMPLES = 12_000_000
#: How far either side of the tuned frequency a signal is looked for.
SEARCH_HZ = 12.5e3
#: A signal must stand this far above the noise floor (50 ms max-hold, so the floor's own
#: excursions are small; see detect.py for why plain max-hold needs 20 dB).
MIN_SNR_DB = 12.0
#: Each 50 ms segment of the capture is averaged before the max-hold.
SEGMENT_S = 0.05
BLOCK_S = 0.05


@dataclass
class Classification:
    freq_hz: float
    label: str
    detail: str = ""
    bandwidth_hz: float = 0.0
    snr_db: float = 0.0
    #: True when an error-checked decode identified it; False for a judgement from
    #: features of the signal.
    certain: bool = False

    def text(self) -> str:
        detail = f" ({self.detail})" if self.detail else ""
        hedge = "" if self.certain else "?"
        return f"{self.freq_hz / 1e6:.4f} MHz  {self.label}{hedge}{detail}"


def _spectrum(iq: np.ndarray, rate: float, resolution_hz: float = 250.0):
    """Mean and 50 ms max-hold power spectra (dB), fftshifted, with bin offsets (Hz)."""
    nfft = int(2 ** np.ceil(np.log2(max(256.0, rate / resolution_hz))))
    nfft = min(nfft, 65536)
    frames = iq[: iq.size // nfft * nfft].reshape(-1, nfft)
    window = np.hanning(nfft).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft(frames * window), axes=1)) ** 2
    per = max(1, int(SEGMENT_S * rate / nfft))
    usable = power.shape[0] // per * per or power.shape[0]
    segments = power[:usable].reshape(-1, min(per, usable), nfft).mean(axis=1)
    offsets = (np.arange(nfft) - nfft // 2) * rate / nfft
    to_db = lambda p: 10 * np.log10(p + 1e-20)                  # noqa: E731
    return to_db(power.mean(axis=0)), to_db(segments.max(axis=0)), offsets


def _smooth(x: np.ndarray, bins: int) -> np.ndarray:
    bins = max(1, int(bins))
    return np.convolve(x, np.ones(bins) / bins, mode="same")


def find_signal(iq, rate, listen_offset_hz):
    """(centre offset, bandwidth, snr) of the strongest signal near the listening
    frequency, or None. Bandwidth is the span within 20 dB of its peak (or 6 dB over
    the floor, for a weak one)."""
    mean_db, hold_db, offsets = _spectrum(iq, rate)
    step = offsets[1] - offsets[0]
    hold = _smooth(hold_db, 1000.0 / step)
    floor = float(np.median(mean_db))
    near = np.flatnonzero(np.abs(offsets - listen_offset_hz) <= SEARCH_HZ)
    if near.size == 0:
        return None
    k = int(near[np.argmax(hold[near])])
    snr = float(hold[k] - floor)
    if snr < MIN_SNR_DB:
        return None
    edge = max(hold[k] - 20.0, floor + 6.0)
    lo, hi = _walk(hold, k, edge, -1, floor), _walk(hold, k, edge, +1, floor)
    width = (hi - lo + 1) * step
    return float((offsets[lo] + offsets[hi]) / 2), float(width), snr


def _walk(hold: np.ndarray, k: int, edge: float, step: int, floor: float) -> int:
    """From the peak outwards until the level falls below `edge`, or rises 6 dB out of a
    valley that reaches down near the noise: a neighbour's skirt must not count as this
    signal's width (measured at 131.55 MHz, where a stronger signal 20-90 kHz above made
    ACARS look 250 kHz wide), but a dip within one signal -- between an FSK signal's two
    tones, or in broadcast FM -- must."""
    i, lowest, at_lowest = k, hold[k], k
    while 0 < i + step < hold.size and hold[i + step] > edge:
        i += step
        if hold[i] < lowest:
            lowest, at_lowest = hold[i], i
        elif hold[i] > lowest + 6.0 and lowest < floor + 9.0:
            return at_lowest
    return i


def _run_blocks(iq, rate, consumers) -> None:
    block = max(4096, int(rate * BLOCK_S))
    for i in range(0, iq.size, block):
        piece = iq[i:i + block]
        for consume in consumers:
            consume(piece)


def _decode(iq, rate, centre_hz, offset_hz) -> Classification | None:
    """The app's decoders over the capture; the first with error-checked frames wins."""
    freq = centre_hz + offset_hz
    if abs(freq - ADSB_HZ) < rate / 2 and abs(rate / 1e6 - round(rate / 1e6)) < 1e-6 \
            and rate >= 2e6:
        mixer, adsb = Mixer(rate, ADSB_HZ - centre_hz), AdsbDecoder(rate)
        found = []
        _run_blocks(iq, rate, [lambda x: found.extend(adsb.process(mixer.process(x)))])
        if found:
            aircraft = len({m.icao for m in found})
            return Classification(ADSB_HZ, "ADS-B", f"{aircraft} aircraft", certain=True)

    fm = DemodChain(rate, "nbfm", offset_hz=offset_hz, bandwidth_hz=16e3, agc=False)
    wide = DemodChain(rate, "nbfm", offset_hz=offset_hz, bandwidth_hz=20e3, agc=False)
    am = DemodChain(rate, "am", offset_hz=offset_hz, bandwidth_hz=10e3, agc=False)
    p25, dmr = P25Decoder(fm.if_rate), DmrDecoder(fm.if_rate)
    pocsag, aprs = PocsagDecoder(fm.if_rate), AprsDecoder(fm.if_rate)
    ais, acars = AisDecoder(wide.if_rate), AcarsDecoder(am.if_rate)
    got: dict[str, list] = {name: [] for name in ("p25", "dmr", "aprs", "ais", "acars")}

    def narrow(x):
        fm.process(x)
        d = fm.last_detected
        got["p25"] += p25.process(d)
        got["dmr"] += dmr.process(d)
        pocsag.process(d)                  # messages discarded: batches are the evidence
        got["aprs"] += aprs.process(d)

    def ais_lane(x):
        wide.process(x)
        got["ais"] += ais.process(wide.last_detected)

    def am_lane(x):
        am.process(x)
        got["acars"] += acars.process(am.last_detected)

    lanes = [narrow, am_lane]
    if any(abs(freq - f) < 15e3 for f in AIS_CHANNELS.values()):
        lanes.append(ais_lane)
    _run_blocks(iq, rate, lanes)

    if p25.frames >= 2:
        nacs = sorted({f"{m.nac:03X}" for m in got["p25"]})
        control = any(m.kind == "TSBK" for m in got["p25"])
        detail = ", ".join(([f"NAC {', '.join(nacs)}"] if nacs else [])
                           + (["control channel"] if control else []))
        return Classification(freq, "P25", detail, certain=True)
    if dmr.bursts >= 3:
        codes = sorted({m.colour_code for m in got["dmr"] if m.colour_code is not None})
        # A preamble CSBK only announces data; other control blocks mean a control
        # channel (Motorola's own fill local ones).
        control = any(m.kind == "CSBK" and not (m.fields.get("opcode") == 0x3D
                                                and m.fields.get("fid") == 0)
                      for m in got["dmr"])
        detail = ", ".join(([f"CC {', '.join(map(str, codes))}"] if codes else [])
                           + (["control channel"] if control else []))
        return Classification(freq, "DMR", detail, certain=True)
    batches = {f.baud: f.batches for f in pocsag._framers if f.batches}
    if batches:
        baud = max(batches, key=batches.get)
        return Classification(freq, "POCSAG", f"{baud} baud", certain=True)
    if got["ais"]:
        return Classification(freq, "AIS", certain=True)
    if got["aprs"]:
        return Classification(freq, "AX.25 packet", "APRS, 1200 baud AFSK", certain=True)
    if got["acars"]:
        return Classification(freq, "ACARS", certain=True)
    return None


def _baseband(iq, rate, offset_hz, bandwidth_hz):
    """The signal alone: shifted to 0 Hz, decimated to a few times its width, and
    filtered to it, so neighbours do not beat with it (measured: FM stations 200 kHz
    away made a strong broadcast station's envelope look like AM)."""
    from .dsp.decimate import StreamDecimator, lowpass_taps
    from .dsp.filters import fir_length_for

    target = max(4 * bandwidth_hz, 24e3)
    factor = 1
    while rate / (factor * 2) >= target:
        factor *= 2
    z = Mixer(rate, offset_hz).process(iq.astype(np.complex128))
    if factor > 1:
        z = StreamDecimator(factor).process(z)
    fs = rate / factor
    cutoff = min(0.6 * bandwidth_hz / fs, 0.45)
    taps = lowpass_taps(cutoff, fir_length_for(cutoff, maximum=511))
    return np.convolve(z, taps, mode="same"), fs


def _measure(iq, rate, centre_hz, listen_offset_hz, offset_hz, bandwidth_hz, snr):
    """Name the modulation from its features, when no decoder knew it."""
    freq = centre_hz + offset_hz
    broadcast = 87.5e6 <= freq <= 108.0e6
    if broadcast:
        bandwidth_hz = max(bandwidth_hz, 180e3)       # a broadcast channel, whatever dips
    z, fs = _baseband(iq, rate, offset_hz, bandwidth_hz)
    z = _active(z[int(0.05 * fs):], fs)          # past the filters' start-up
    env = np.abs(z)
    spread = float(np.std(env) / (np.mean(env) + 1e-12))
    result = lambda label, detail="": Classification(            # noqa: E731
        freq, label, detail, bandwidth_hz, snr)

    if bandwidth_hz < 1000.0:
        if spread < 0.2:
            return result("Carrier", "unmodulated")
        on = env > 0.5 * np.percentile(env, 99)
        if 0.1 < on.mean() < 0.9:
            return result("CW", "on-off keyed")
        return result("Narrow carrier")

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(z * np.hanning(z.size)))) ** 2
    f = (np.arange(z.size) - z.size // 2) * fs / z.size
    total = spectrum.sum() + 1e-20
    # AM is a carrier with matching sidebands either side of it; SSB, even a two-tone
    # test, puts everything on one side of its strongest line.
    k = int(np.argmax(spectrum))
    guard = max(50.0, 3 * fs / z.size)
    line = spectrum[np.abs(f - f[k]) <= guard].sum() / total
    above = spectrum[f > f[k] + guard].sum()
    below = spectrum[f < f[k] - guard].sum()
    symmetric = max(above, below) < 3 * min(above, below) + 1e-20

    d = np.angle(z[1:] * np.conj(z[:-1])) * fs / (2 * np.pi)
    # Lightly modulated AM (an ATIS: measured at 119.8 MHz) has a steadier envelope than
    # the 0.3 below allows, so ask which one carries the audio: in AM the envelope is
    # shaped like speech and the frequency is noise; in FM the other way round.
    d_flat = _flatness(d, fs, bandwidth_hz)
    am_like = (not broadcast and spread > 0.03 and d_flat > 0.4
               and _flatness(env, fs, bandwidth_hz) < d_flat)
    # A weak broadcast station's noise makes its envelope vary; the band is FM's alone.
    if (spread < 0.3 and not am_like) or broadcast:   # constant envelope: FM of some kind
        dev = float(np.percentile(np.abs(d), 99))
        if broadcast or dev > 20e3:
            return _broadcast(iq, rate, centre_hz, offset_hz, bandwidth_hz, snr)
        # Two-level FSK sits at its two deviations: few samples near zero.
        busy = np.abs(d - np.median(d)) < 0.25 * dev
        if busy.mean() < 0.12:
            return result("FSK data", f"about +/-{dev / 1e3:.1f} kHz, not a known protocol")
        return result("NBFM", f"about +/-{dev / 1e3:.1f} kHz deviation")

    if line > 0.15 and symmetric:
        return result("AM")
    if am_like and spread < 0.3:
        return result("AM", "carrier not clear")
    # No carrier: single sideband, on whichever side of the tuned frequency it lies.
    rel = f + offset_hz - listen_offset_hz
    upper = spectrum[(rel > 200) & (rel < 3000)].sum()
    lower = spectrum[(rel < -200) & (rel > -3000)].sum()
    if upper > 4 * lower:
        return result("USB")
    if lower > 4 * upper:
        return result("LSB")
    return result("Unknown", "varying envelope, no carrier")


#: RDS blocks passing their check before RDS counts as there (a block is 26 bits, so
#: about 22 a second arrive from a station that sends it).
RDS_MIN_BLOCKS = 8


def _broadcast(iq, rate, centre_hz, offset_hz, bandwidth_hz, snr) -> Classification:
    """Broadcast FM, through the app's own WBFM chain: its stereo decoder says whether
    the pilot is there, and its RDS decoder whether data blocks pass their checks --
    which, with the station's name, settles it."""
    chain = DemodChain(rate, "wbfm", offset_hz=offset_hz, agc=False)
    _run_blocks(iq, rate, [chain.process])
    extras = ["stereo"] if chain.stereo else []
    rds = chain._rds
    good = rds.assembler.good_blocks if rds is not None else 0
    if good >= RDS_MIN_BLOCKS:
        name = (chain.rds.ps_name or "").strip()
        extras.append(f"RDS {name}".strip())
    return Classification(centre_hz + offset_hz, "WBFM", ", ".join(extras), bandwidth_hz,
                          snr, certain=good >= RDS_MIN_BLOCKS)


def _flatness(x: np.ndarray, fs: float, bandwidth_hz: float) -> float:
    """Spectral flatness (geometric over arithmetic mean) of `x` across the audio band:
    near 1 for noise, small for speech or a tone."""
    from scipy.signal import welch

    f, p = welch(np.asarray(x, dtype=np.float64) - np.mean(x), fs, nperseg=1024)
    band = (f >= 300.0) & (f <= max(600.0, min(2500.0, 0.45 * bandwidth_hz)))
    if band.sum() < 3:
        return 1.0
    p = p[band] + 1e-30
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def _active(z: np.ndarray, fs: float) -> np.ndarray:
    """Only the stretches where the signal is on: bursts (ACARS, a DMR mobile, a
    packet) would otherwise be measured mostly as the noise between them."""
    n = max(1, int(0.005 * fs))
    usable = z[: z.size // n * n]
    if usable.size < 4 * n:
        return z
    power = (np.abs(usable.reshape(-1, n)) ** 2).mean(axis=1)
    quiet = np.percentile(power, 10)
    on = power > 4 * quiet                       # 6 dB over the quietest stretches
    if on.mean() < 0.05 or on.all():
        return z
    return usable.reshape(-1, n)[on].ravel()


def classify(iq: np.ndarray, rate: float, centre_hz: float,
             listen_hz: float) -> Classification | None:
    """Classify the signal nearest `listen_hz` in a capture centred on `centre_hz`.
    None if there is no signal there."""
    iq = np.asarray(iq, dtype=np.complex64)
    listen_offset = listen_hz - centre_hz
    if abs(listen_hz - ADSB_HZ) < 1e6:
        # Pulses a microsecond long average away to nothing in a spectrum: decode first.
        found = _decode(iq, rate, centre_hz, listen_offset)
        if found is not None:
            return found
    signal = find_signal(iq, rate, listen_offset)
    if signal is None:
        return None
    offset, width, snr = signal
    # Decoders first where the user tuned -- that is the signal they mean, and a channel
    # filter there keeps neighbours out -- then at the measured centre if elsewhere.
    found = _decode(iq, rate, centre_hz, listen_offset)
    if found is None and abs(offset - listen_offset) > 2e3 and width < 40e3:
        found = _decode(iq, rate, centre_hz, offset)
    if found is not None:
        found.bandwidth_hz, found.snr_db = width, snr
    else:
        found = _measure(iq, rate, centre_hz, listen_offset, offset, width, snr)
    if abs(offset - listen_offset) <= width / 2:
        # Tuned onto it: the frequency the user chose is the channel's. The measured
        # centre carries the radio's own crystal error (+5 ppm on the Pluto, 2 kHz at
        # 420 MHz) and the signal's spectral shape.
        found.freq_hz = listen_hz
    return found


class ClassifyJob:
    """Capture CAPTURE_S of a source's IQ and classify it, on a thread of its own so the
    window keeps running. Poll `done`; then read `result` (None for no signal) or
    `error`."""

    def __init__(self, source, listen_hz: float) -> None:
        import threading

        self.source = source
        self.listen_hz = float(listen_hz)
        self.centre_hz = float(source.center_freq)
        self.rate = float(source.sample_rate)
        self.result: Classification | None = None
        self.error = ""
        self._done = threading.Event()
        self._cancel = threading.Event()
        self._reader = source.sequential_reader()
        self._thread = threading.Thread(target=self._run, name="classify", daemon=True)
        self._thread.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> None:
        self._cancel.set()

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
                self.result = classify(np.concatenate(chunks), self.rate, self.centre_hz,
                                       self.listen_hz)
        except Exception as exc:                      # shown, never fatal
            self.error = str(exc) or type(exc).__name__
        finally:
            self._done.set()
