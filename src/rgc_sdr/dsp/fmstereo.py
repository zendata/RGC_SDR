"""FM stereo: recover left and right from the broadcast multiplex.

The multiplex (MPX) that comes out of the FM discriminator carries, by frequency:

    0-15 kHz    L+R, which is also the mono signal
    19 kHz      the pilot tone, present only on stereo transmissions
    23-53 kHz   L-R, double-sideband suppressed-carrier on 38 kHz (twice the pilot)
    57 kHz      RDS data (three times the pilot) -- see rds.py

The 38 kHz carrier is regenerated from the pilot without a phase-locked loop. A complex
band-pass isolates the pilot as a phasor e^(j theta); squaring that phasor gives
e^(j 2 theta), exactly twice the frequency with exactly twice the phase. A loop would
have to be a per-sample Python iteration, which is far too slow at MPX rates.

Every filter here has the same length, so every path has the same delay and the
regenerated carrier lines up with the signal it demodulates without any bookkeeping.

Pure NumPy, no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decimate import lowpass_taps
from .filters import Fir

PILOT_HZ = 19_000.0
AUDIO_HZ = 15_000.0

#: Common FIR length. At 192 kHz this gives a transition band of about 4 kHz, which fits
#: the 4 kHz gaps either side of the pilot (15 kHz audio below, 23 kHz L-R above).
TAPS = 255

#: How steady the pilot's phase must be before a transmission counts as stereo.
#:
#: Measured 2026-09-25. An amplitude test was used first and failed on air: the FM
#: discriminator turns noise into a multiplex with plenty of energy at 19 kHz, and pure
#: noise measured 0.087 against a 0.03 threshold, so noise was declared stereo. That is
#: worse than a wrong label -- stereo decoding then demodulates noise from 23-53 kHz into
#: the audio. A real pilot is a steady tone and measures 1.000 on this test; noise wanders
#: and measures 0.094.
PILOT_COHERENCE = 0.5


def _bandpass(centre_hz: float, width_hz: float, rate: float, taps: int) -> np.ndarray:
    half = width_hz / 2.0 / rate
    base = lowpass_taps(half, taps)
    n = np.arange(taps) - (taps - 1) / 2.0
    return base * np.exp(2j * np.pi * (centre_hz / rate) * n)


class _Delay:
    """Delays a stream by a whole number of samples, across block boundaries."""

    def __init__(self, samples: int) -> None:
        self._held = np.zeros(samples, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._held.size == 0:
            return x
        joined = np.concatenate([self._held, x])
        self._held = joined[x.size:]
        return joined[: x.size]


@dataclass
class StereoBlock:
    left: np.ndarray
    right: np.ndarray
    #: The multiplex and unit pilot phasor, delayed to line up with each other, for RDS.
    mpx: np.ndarray
    pilot: np.ndarray
    stereo: bool


class StereoDecoder:
    def __init__(self, rate: float, taps: int = TAPS) -> None:
        if rate < 2.0 * 57_000.0 + 10_000.0:
            raise ValueError("the multiplex needs at least ~125 kHz to carry RDS")
        self.rate = float(rate)
        self.taps = int(taps) | 1
        delay = (self.taps - 1) // 2
        self._pilot_filter = Fir(_bandpass(PILOT_HZ, 1500.0, self.rate, self.taps))
        self._sum_filter = Fir(lowpass_taps(AUDIO_HZ / self.rate, self.taps))
        self._diff_filter = Fir(lowpass_taps(AUDIO_HZ / self.rate, self.taps))
        self._delay = _Delay(delay)
        self.pilot_ratio = 0.0
        self.pilot_coherence = 0.0
        self.force_mono = False

    @property
    def stereo(self) -> bool:
        return not self.force_mono and self.pilot_coherence >= PILOT_COHERENCE

    def reset(self) -> None:
        self.__init__(self.rate, self.taps)

    def process(self, mpx: np.ndarray) -> StereoBlock:
        mpx = np.asarray(mpx, dtype=np.float64)
        pilot = self._pilot_filter.process(mpx)
        delayed = self._delay.process(mpx)

        magnitude = np.abs(pilot)
        rms = float(np.sqrt(np.mean(delayed**2))) if delayed.size else 0.0
        ratio = float(np.mean(magnitude)) / rms if rms > 0.0 else 0.0
        # Smoothed, so a quiet passage of music does not flick the display to mono.
        self.pilot_ratio += 0.3 * (ratio - self.pilot_ratio)

        unit = pilot / np.maximum(magnitude, 1e-12)
        # Phase coherence: derotate by the expected 19 kHz and see whether what is left
        # holds still. The reference restarts each block, which is fine because only the
        # magnitude of the mean is used and a constant phase offset does not change it.
        if unit.size:
            reference = np.exp(-2j * np.pi * PILOT_HZ * np.arange(unit.size) / self.rate)
            coherence = float(abs(np.mean(unit * reference)))
            self.pilot_coherence += 0.3 * (coherence - self.pilot_coherence)
        # The pilot is sin(wt); its analytic part is e^(j(wt - pi/2)). Squared that is
        # -e^(j 2wt), so the sin(2wt) subcarrier is -Im(unit^2).
        subcarrier = -np.imag(unit * unit)

        total = self._sum_filter.process(delayed)
        difference = self._diff_filter.process(2.0 * delayed * subcarrier)
        if self.stereo:
            left = 0.5 * (total + difference)
            right = 0.5 * (total - difference)
        else:
            left = right = 0.5 * total
        return StereoBlock(left, right, delayed, unit, self.stereo)
