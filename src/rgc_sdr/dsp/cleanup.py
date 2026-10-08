"""Cleaning up what is heard: a noise blanker on the IQ, and noise reduction and notches
on the audio (P10, PLANNING.md 7r).

* `NoiseBlanker` zeroes impulses -- ignition, electric fences, switching supplies -- in
  the IQ at the radio's rate, before any filter has spread them out. An impulse is a
  sample far above the block's median magnitude; the samples either side go too, as the
  radio's own filters have already smeared it a little.
* `AudioCleanup` works on the audio in short overlapping FFT frames. Noise reduction
  learns the noise spectrum from the quietest frames and turns each bin down by how much
  of it is noise; the automatic notch removes bins holding a steady tone (a carrier
  whistling in an SSB passband); manual notches remove chosen audio frequencies.

Pure NumPy; no Qt, no device access. Frames are looped over (a handful per block), never
samples.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

#: Impulse guard either side of a blanked sample, in seconds.
NB_GUARD_S = 20e-6
#: Blanker thresholds, as multiples of the median magnitude: level 1 to level 10.
NB_THRESHOLD_LEVEL1 = 15.5
NB_THRESHOLD_LEVEL10 = 2.0

#: Audio frame length: about this long, rounded up to a power of two.
FRAME_S = 0.01
#: Noise estimate: how fast it may rise per second, in dB (it falls at once).
NOISE_RISE_DB_PER_S = 1.0
#: Smoothing of each bin's power before its minimum is tracked, and the factor that
#: puts the smoothed minimum back at the noise's mean.
POWER_SMOOTHING = 0.8
NOISE_BIAS = 2.0
#: Bins either side whose noise median caps a bin's noise estimate.
NR_NEIGHBOURS = 16
#: Smoothing of each bin's gain from frame to frame, against "musical noise".
GAIN_SMOOTHING = 0.6
#: Automatic notch: time constant of the slow spectrum, and how far a bin must stand
#: over its neighbours' median to count as a steady tone.
NOTCH_AVERAGE_S = 1.0
NOTCH_RATIO_DB = 13.0
NOTCH_NEIGHBOURS = 8
#: Audio band the automatic notch looks in: tones below and above are not heterodynes
#: anyone needs removed, and the band edges are where the channel filter rolls off.
NOTCH_BAND_HZ = (150.0, 4000.0)
#: What a notched bin is turned down to.
NOTCH_GAIN = 0.02
#: Manual notches cover their frequency this far either side.
MANUAL_NOTCH_HALF_WIDTH_HZ = 40.0


def blanker_threshold(level: int) -> float:
    """Multiple of the median magnitude above which a sample is an impulse."""
    level = min(10, max(1, int(level)))
    step = (NB_THRESHOLD_LEVEL1 - NB_THRESHOLD_LEVEL10) / 9.0
    return NB_THRESHOLD_LEVEL1 - (level - 1) * step


class NoiseBlanker:
    """Zero impulses in complex IQ. Level 0 is off, 1 gentle, 10 aggressive."""

    def __init__(self, sample_rate: float, level: int = 0) -> None:
        self.sample_rate = float(sample_rate)
        self.level = int(level)
        self._guard = max(1, int(round(NB_GUARD_S * self.sample_rate)))
        self._reference: float | None = None
        #: Share of samples blanked lately (for the status line).
        self.blanked = 0.0

    def reset(self) -> None:
        self._reference = None
        self.blanked = 0.0

    def process(self, iq: np.ndarray) -> np.ndarray:
        if self.level <= 0 or iq.size == 0:
            return iq
        magnitude = np.abs(iq)
        # The median of a subsample: impulses are rare, so they cannot move it, and a
        # tenth of the samples gives the same answer for a tenth of the sorting.
        median = float(np.median(magnitude[:: max(1, iq.size // 4096)]))
        self._reference = (median if self._reference is None
                           else 0.8 * self._reference + 0.2 * median)
        if self._reference <= 0.0:
            return iq
        hot = magnitude > blanker_threshold(self.level) * self._reference
        if not hot.any():
            self.blanked *= 0.9
            return iq
        width = 2 * self._guard + 1
        mask = np.convolve(hot.astype(np.float32), np.ones(width, np.float32), mode="same") > 0
        self.blanked = 0.9 * self.blanked + 0.1 * float(mask.mean())
        out = iq.copy()
        out[mask] = 0
        return out


def _frame_length(rate: float) -> int:
    n = 64
    while n < FRAME_S * rate:
        n *= 2
    return n


class AudioCleanup:
    """Noise reduction and notches for mono audio, in 50%-overlapped FFT frames.

    `nr_level` 0 is off, 1-10 stronger. `auto_notch` removes steady tones.
    `set_notches` takes audio frequencies to remove. Output lags input by one frame.
    """

    def __init__(self, rate: float, nr_level: int = 0, auto_notch: bool = False,
                 notches_hz=()) -> None:
        self.rate = float(rate)
        self.n = _frame_length(self.rate)
        self.hop = self.n // 2
        # A periodic Hann's square root, used both ways: their product, overlapped by
        # half, sums to exactly one, so a frame with every gain 1 comes back unchanged.
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n) / self.n)
        self._window = np.sqrt(hann)
        self.freqs = np.fft.rfftfreq(self.n, 1.0 / self.rate)
        frame_s = self.hop / self.rate
        self._rise = 10.0 ** (NOISE_RISE_DB_PER_S * frame_s / 10.0)
        self._slow_alpha = min(1.0, frame_s / NOTCH_AVERAGE_S)
        self._band = (self.freqs >= NOTCH_BAND_HZ[0]) & (self.freqs <= NOTCH_BAND_HZ[1])
        self.nr_level = int(nr_level)
        self.auto_notch = bool(auto_notch)
        self._manual = np.ones(self.freqs.size)
        self.set_notches(notches_hz)
        self.reset()

    @property
    def active(self) -> bool:
        return self.nr_level > 0 or self.auto_notch or bool(self.notches_hz)

    def reset(self) -> None:
        self._pending = np.zeros(self.n - self.hop)      # input not yet framed
        self._overlap = np.zeros(self.n - self.hop)      # output tail to add
        self._noise: np.ndarray | None = None
        self._smooth: np.ndarray | None = None
        self._gain = np.ones(self.freqs.size)
        self._slow: np.ndarray | None = None
        #: Bins the automatic notch is removing now, as audio frequencies.
        self.notched_hz: list[float] = []

    def set_notches(self, notches_hz) -> None:
        self.notches_hz = tuple(float(f) for f in notches_hz if f > 0)
        manual = np.ones(self.freqs.size)
        bin_hz = self.rate / self.n
        for f in self.notches_hz:
            near = np.abs(self.freqs - f) <= max(MANUAL_NOTCH_HALF_WIDTH_HZ, bin_hz)
            manual[near] = 0.0
        self._manual = manual

    def _nr_gain(self, power: np.ndarray) -> np.ndarray:
        if self._noise is None:
            self._noise, self._smooth = power.copy(), power.copy()
        # A bin's power from one frame to the next swings widely even on steady noise,
        # and the quietest frame lies far below the mean (minimum statistics' bias). So
        # the minimum is taken of a smoothed power, and scaled back up by NOISE_BIAS.
        self._smooth = POWER_SMOOTHING * self._smooth + (1.0 - POWER_SMOOTHING) * power
        # Falls at once to anything quieter, creeps up otherwise: the noise floor is
        # what the quietest frames show, speech never pulls it up for long.
        self._noise = np.where(self._smooth < self._noise, self._smooth,
                               self._noise * self._rise)
        # But a steady tone -- a CW signal, a carrier -- is as stationary as noise, and
        # its bin would creep up until the tone itself counted as noise and was removed.
        # Noise is spread across frequency and a tone is not: no bin's noise may exceed
        # twice its neighbours' median.
        k = NR_NEIGHBOURS
        local = np.median(sliding_window_view(np.pad(self._noise, k, mode="edge"),
                                              2 * k + 1), axis=1)
        self._noise = np.minimum(self._noise, 2.0 * local)
        if self.nr_level <= 0:
            return np.ones_like(power)
        noise = self._noise * NOISE_BIAS
        over = 0.5 + 0.25 * self.nr_level                   # 0.75 .. 3: how hard
        floor = 10.0 ** (-(6.0 + 2.0 * self.nr_level) / 20.0)   # -8 .. -26 dB
        gain = np.maximum(floor, 1.0 - over * noise / np.maximum(power, 1e-30))
        self._gain = GAIN_SMOOTHING * self._gain + (1.0 - GAIN_SMOOTHING) * gain
        return self._gain

    def _notch_gain(self, power: np.ndarray) -> np.ndarray:
        self._slow = (power.copy() if self._slow is None
                      else self._slow + self._slow_alpha * (power - self._slow))
        if not self.auto_notch:
            self.notched_hz = []
            return np.ones_like(power)
        k = NOTCH_NEIGHBOURS
        padded = np.pad(self._slow, k, mode="edge")
        local = np.median(sliding_window_view(padded, 2 * k + 1), axis=1)
        tone = self._band & (self._slow > local * 10.0 ** (NOTCH_RATIO_DB / 10.0))
        # A tone spreads into the bins beside it through the window.
        tone = np.convolve(tone.astype(np.float32), np.ones(3, np.float32), "same") > 0
        self.notched_hz = [float(f) for f in self.freqs[tone]]
        return np.where(tone, NOTCH_GAIN, 1.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float64)
        if not self.active:
            # Off costs nothing, but a change of mind must not jump the stream: the
            # frame's lag is kept as delay even when nothing is cleaned up.
            stream = np.concatenate([self._pending, audio])
            out, self._pending = stream[:audio.size], stream[audio.size:]
            return out
        stream = np.concatenate([self._pending, audio])
        count = max(0, (stream.size - self.n) // self.hop + 1)
        if count == 0:
            self._pending = stream
            return np.zeros(0)
        frames = sliding_window_view(stream, self.n)[::self.hop][:count] * self._window
        spectra = np.fft.rfft(frames, axis=1)
        power = spectra.real ** 2 + spectra.imag ** 2
        for i in range(count):                       # a few frames a block
            gain = self._nr_gain(power[i]) * self._notch_gain(power[i]) * self._manual
            spectra[i] *= gain
        pieces = np.fft.irfft(spectra, n=self.n, axis=1) * self._window
        out = np.zeros(count * self.hop + self.n - self.hop)
        for i in range(count):
            out[i * self.hop:i * self.hop + self.n] += pieces[i]
        out[:self.n - self.hop] += self._overlap
        self._overlap = out[count * self.hop:].copy()
        self._pending = stream[count * self.hop:]
        return out[:count * self.hop]
