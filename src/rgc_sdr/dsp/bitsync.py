"""Bit recovery for two-level FSK data, from a demodulated baseband.

Bit timing comes from the transitions, without a per-sample loop (PLANNING.md 7p).
Each sign change is located to a fraction of a sample by interpolation, and given a bit
index `round((t - phase) / samples_per_bit)`. `phase` is a running mean of the
transition times modulo one bit, kept as a smoothed phasor and unwrapped so that clock
drift is followed without ever slipping a whole bit. Every bit between two transitions
has the same value, so the bit stream is `np.repeat(levels, diff(index))`; a glitch
shorter than half a bit gets the same index at both ends and simply vanishes.

Pure NumPy and scipy; no Qt, no device access.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .filters import Fir, fir_length_for
from .decimate import lowpass_taps

#: How many transitions the timing average spans (its time constant).
TIMING_TRANSITIONS = 24.0
#: Bits held back at most while no transition arrives, so output never stalls for long.
MAX_PENDING_BITS = 64
#: The DC tracker's time constant in bits. FSK slightly off tune sits off zero.
DC_BITS = 64.0


class BitSlicer:
    """Real samples in, hard bits (0/1) out: 1 where the signal is positive."""

    def __init__(self, sample_rate: float, baud: float, prefilter: bool = True,
                 track_dc: bool = True) -> None:
        self.sample_rate = float(sample_rate)
        self.baud = float(baud)
        self.sps = self.sample_rate / self.baud
        if self.sps < 2.0:
            raise ValueError("need at least two samples per bit")
        self._fir = None
        if prefilter:
            # Pass up to about the bit rate: enough for clean NRZ edges, and most of the
            # noise above it is gone.
            cutoff = min(0.8 * self.baud / self.sample_rate, 0.45)
            self._fir = Fir(lowpass_taps(cutoff, fir_length_for(cutoff, maximum=255)))
        self._dc_alpha = 1.0 / (DC_BITS * self.sps) if track_dc else 0.0
        self.reset()

    def reset(self) -> None:
        if self._fir is not None:
            self._fir.reset()
        self._dc_zi = None
        self._n = 0                      # absolute index of the next sample
        self._prev = 0.0                 # the previous sample's value
        self._phasor_zi = np.zeros(1, dtype=np.complex128)
        self._phase = 0.0                # unwrapped timing phase, in radians
        self._last_index = None          # bit index of the last transition
        self._level = 0                  # level since the last transition

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return np.zeros(0, dtype=np.uint8)
        if self._fir is not None:
            x = self._fir.process(x)
        if self._dc_alpha:
            a = self._dc_alpha
            if self._dc_zi is None:
                self._dc_zi = np.array([(1 - a) * x[0]])
            dc, self._dc_zi = lfilter([a], [1.0, a - 1.0], x, zi=self._dc_zi)
            x = x - dc

        prev = np.concatenate([[self._prev], x[:-1]])
        crossing = np.flatnonzero((prev > 0) != (x > 0))
        n0 = self._n
        self._n += x.size
        self._prev = float(x[-1])

        bits = []
        if crossing.size:
            a, b = prev[crossing], x[crossing]
            frac = np.where(a != b, a / (a - b), 0.5)
            t = n0 + crossing - 1 + frac             # absolute time of each sign change
            levels_after = (x[crossing] > 0).astype(np.uint8)

            # Timing phase: smoothed unit phasor of each transition's time within a bit.
            k = 1.0 / TIMING_TRANSITIONS
            p = np.exp(2j * np.pi * t / self.sps)
            smoothed, self._phasor_zi = lfilter([k], [1.0, k - 1.0], p, zi=self._phasor_zi)
            angle = np.unwrap(np.concatenate([[self._phase], np.angle(smoothed)]))[1:]
            self._phase = float(angle[-1])
            index = np.round(t / self.sps - angle / (2 * np.pi)).astype(np.int64)

            if self._last_index is None:
                # Nothing before the first transition is trustworthy.
                self._last_index = int(index[0])
                self._level = int(levels_after[0])
                index, levels_after = index[1:], levels_after[1:]
            if index.size:
                starts = np.concatenate([[self._last_index], index[:-1]])
                levels = np.concatenate([[self._level], levels_after[:-1]])
                bits.append(np.repeat(levels, np.maximum(index - starts, 0)).astype(np.uint8))
                self._last_index = int(max(index[-1], self._last_index))
                self._level = int(levels_after[-1])

        if self._last_index is not None:
            # A long run with no transition: hand over the bits it already holds.
            now = int(np.floor(self._n / self.sps - self._phase / (2 * np.pi)))
            if now - self._last_index > MAX_PENDING_BITS:
                count = now - self._last_index - 1
                bits.append(np.full(count, self._level, dtype=np.uint8))
                self._last_index += count
        return np.concatenate(bits) if bits else np.zeros(0, dtype=np.uint8)
