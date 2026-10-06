"""Measure a radio's frequency error against a carrier whose frequency is known.

Asked for by VK3RQ (2026-10-06): the Pluto and HackRF read several ppm off, so each radio
keeps a correction (`SoapyIQSource.set_ppm`) found once, from a reference carrier -- an
AM transmitter's carrier (an ATIS) or a CW beacon -- measured as the strongest line
near its known frequency, to a fraction of a hertz.

A line counts only if it stands at the same radio frequency in two captures taken at
different tunings: the radio's own spurs (the HackRF has combs of them) move with the
tuning. And only within REFERENCE_PPM of the known frequency, where a radio's crystal
can put it: with +/-100 ppm, other signals 6-7 kHz away passed for the reference.

*Not broadcast FM:* the mean of a station's instantaneous frequency should be its
carrier, but over 4 s the programme's bass does not average away -- measured on the
Pluto, repeat readings of one station differed by up to 10 ppm and five stations
scattered from -22 to +51 ppm, against +4.66 ppm (to 0.01) from the 119.8 MHz carrier.

A reading `offset_hz` high at `freq_hz`, with `ppm_now` already applied, means the
radio's error is ppm_now + offset/freq. Applied again it converges.

Pure NumPy and the app's DSP; the job at the end drives a source on a thread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp.decimate import StreamDecimator, lowpass_taps
from .dsp.demod import Mixer
from .dsp.filters import fir_length_for

#: Known carriers, tried in turn when a radio is first connected (Melbourne). Measured
#: 2026-10-06: the Essendon ATIS carrier at 119.8 MHz stands 60 dB clear on the Pluto,
#: continuously. The 144.650 MHz CW beacon is VK3RQ's suggestion; keyed, so not always
#: on the air.
REFERENCES = ((119.8e6, "Essendon ATIS"), (144.65e6, "the 144.650 MHz beacon"))
#: A carrier is looked for this far either side of its known frequency: a crystal's
#: error, not further (other signals sit 6-7 kHz from 119.8 and 144.65 MHz).
CARRIER_SEARCH_PPM = 25.0
CARRIER_MIN_SNR_DB = 30.0
#: Two readings of a carrier taken at different tunings must agree this closely, or the
#: line was a spur of the radio's own (the HackRF has a comb of them). A spur moves with
#: the tuning -- here by a tenth of the sample rate, 200 kHz -- while the HackRF's own
#: readings of one carrier wander by about 20 Hz (measured), so 100 Hz separates them.
CARRIER_AGREE_HZ = 100.0


@dataclass
class Calibration:
    ppm: float
    reference_hz: float
    offset_hz: float
    method: str                 # what the reference was: "carrier"
    snr_db: float = 0.0

    def describe(self, label: str = "radio") -> str:
        return (f"{label}: {self.ppm:+.2f} ppm, from the {self.method} at "
                f"{self.reference_hz / 1e6:.4f} MHz (read {self.offset_hz:+.0f} Hz off)")


def new_ppm(ppm_now: float, offset_hz: float, freq_hz: float) -> float:
    return float(ppm_now + offset_hz / freq_hz * 1e6)


def _baseband(iq, rate, offset_hz, half_width_hz):
    """The stretch `offset_hz` from the centre, shifted to 0 Hz, decimated and filtered
    to +/- `half_width_hz`."""
    factor = 1
    while rate / (factor * 2) >= 2.5 * 2 * half_width_hz:
        factor *= 2
    z = Mixer(rate, offset_hz).process(np.asarray(iq, dtype=np.complex128))
    if factor > 1:
        z = StreamDecimator(factor).process(z)
    fs = rate / factor
    cutoff = min(half_width_hz / fs, 0.45)
    taps = lowpass_taps(cutoff, fir_length_for(cutoff, maximum=511))
    return np.convolve(z, taps, mode="valid"), fs


def carrier_lines(iq, rate, centre_hz, nominal_hz, most: int = 8) -> list[tuple[float, float]]:
    """Clear lines within CARRIER_SEARCH_PPM of `nominal_hz`, strongest first, as
    (offset from nominal to a fraction of a hertz, snr dB)."""
    span = max(2e3, nominal_hz * CARRIER_SEARCH_PPM * 1e-6)
    z, fs = _baseband(iq, rate, nominal_hz - centre_hz, span)
    nfft = int(2 ** np.floor(np.log2(max(2, min(z.size, int(fs * 4))))))
    if nfft < 1024:
        return []
    frames = z[: z.size // nfft * nfft].reshape(-1, nfft)
    p = (np.abs(np.fft.fftshift(np.fft.fft(frames * np.hanning(nfft)), axes=1)) ** 2
         ).mean(axis=0)
    f = (np.arange(nfft) - nfft // 2) * fs / nfft
    near = np.flatnonzero((np.abs(f) <= span) & (np.arange(nfft) > 0) & (np.arange(nfft) < nfft - 1))
    floor = np.median(p[near])
    peaks = near[(p[near] > p[near - 1]) & (p[near] >= p[near + 1])
                 & (p[near] > floor * 10 ** (CARRIER_MIN_SNR_DB / 10))]
    out = []
    for k in peaks[np.argsort(p[peaks])[::-1]][:most]:
        a, b, c = np.log(p[k - 1:k + 2] + 1e-30)
        shift = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c < 0 else 0.0
        out.append((float((k - nfft // 2 + shift) * fs / nfft),
                    float(10 * np.log10(p[k] / floor))))
    return out


def common_line(first, second) -> tuple[float, float] | None:
    """The strongest line found at the same frequency (within CARRIER_AGREE_HZ) in two
    captures taken at different tunings: a real signal. A spur moves with the tuning."""
    for offset, snr in first:
        match = [o for o, _ in second if abs(o - offset) <= CARRIER_AGREE_HZ]
        if match:
            return (offset + match[0]) / 2, snr
    return None


# -- driving a source ------------------------------------------------------------------

def _capture(source, seconds: float, cancel) -> np.ndarray:
    import time

    reader = source.sequential_reader()
    wanted = int(seconds * source.sample_rate)
    chunks, have = [], 0
    deadline = time.monotonic() + seconds * 3 + 2
    while have < wanted:
        if cancel.is_set():
            raise InterruptedError("cancelled")
        if time.monotonic() > deadline:
            raise TimeoutError("the radio sent too few samples")
        n = reader.available()
        if n == 0:
            time.sleep(0.02)
            continue
        piece = reader.read(min(n, wanted - have))
        chunks.append(piece)
        have += piece.size
    return np.concatenate(chunks)


class CalibrateJob:
    """Find the radio's error on a thread, against the first of `references` (Hz, or
    (Hz, name) pairs) where a clear carrier is found. The radio is retuned while it
    works and put back where it was. Poll `done`; then `result` or `error`."""

    def __init__(self, source, references) -> None:
        import threading

        self.source = source
        self.references = [r if isinstance(r, tuple) else (float(r), "") for r in references]
        self.home_hz = float(source.center_freq)
        self.ppm_before = float(getattr(source, "ppm", 0.0))
        self.result: Calibration | None = None
        self.error = ""
        self.stage = ""
        self._done = threading.Event()
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name="calibrate", daemon=True)
        self._thread.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def _tune(self, hz: float) -> None:
        import time

        self.source.set_center_freq(hz)
        time.sleep(0.25)

    def _run(self) -> None:
        problems = []
        try:
            for nominal, name in self.references:
                if not self.source.caps.covers(nominal):
                    continue
                try:
                    self.result = self._against_carrier(nominal, name)
                    break
                except LookupError as exc:
                    problems.append(str(exc))
            if self.result is None and not self._cancel.is_set():
                self.error = "; ".join(problems) or "no reference this radio can tune"
        except InterruptedError:
            pass
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
        finally:
            if not self._cancel.is_set():
                try:
                    self.source.set_center_freq(self.home_hz)
                except Exception:
                    pass
            self._done.set()

    def _against_carrier(self, nominal: float, name: str = "") -> Calibration:
        rate = self.source.sample_rate
        what = name or f"{nominal / 1e6:.4f} MHz"
        readings = []
        for shift in (0.2, 0.3):                  # two tunings: a spur moves, a signal not
            self.stage = f"measuring the carrier of {what}"
            self._tune(nominal + shift * rate)
            iq = _capture(self.source, 2.0, self._cancel)
            readings.append(carrier_lines(iq, rate, self.source.center_freq, nominal))
        line = common_line(*readings)
        if line is None:
            raise LookupError(f"no clear carrier from {what}"
                              + (" (only lines that moved with the tuning: the radio's "
                                 "own spurs)" if readings[0] else ""))
        offset, snr = line
        return Calibration(new_ppm(self.ppm_before, offset, nominal), nominal, offset,
                           f"carrier of {what}", snr)
