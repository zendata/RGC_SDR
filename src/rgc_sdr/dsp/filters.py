"""Stateful FIR filtering shared by the demodulators, stereo decoder and RDS.

Split out of demod.py so the stereo and RDS decoders can use it without a circular
import; demod.py re-exports these names, so existing imports keep working.
"""

from __future__ import annotations

import numpy as np

from .decimate import lowpass_taps


def fir_length_for(cutoff: float, minimum: int = 31, maximum: int = 511) -> int:
    """Odd tap count giving a transition band proportional to the cutoff."""
    transition = max(cutoff * 0.35, 0.004)
    n = int(4.0 / transition)
    n = max(minimum, min(maximum, n))
    return n | 1


class Fir:
    """Stateful FIR. Taps may be complex, for a single-sideband filter."""

    def __init__(self, taps: np.ndarray) -> None:
        self._taps = np.asarray(taps)
        self._history: np.ndarray | None = None

    @property
    def taps(self) -> np.ndarray:
        return self._taps

    def reset(self) -> None:
        self._history = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        dtype = np.result_type(x.dtype, self._taps.dtype)
        if self._history is None:
            self._history = np.zeros(self._taps.size - 1, dtype=dtype)
        padded = np.concatenate([self._history.astype(dtype), x.astype(dtype)])
        self._history = padded[-(self._taps.size - 1) :]
        return np.convolve(padded, self._taps, mode="valid")


def bandpass_taps(centre_hz: float, bandwidth_hz: float, sample_rate: float) -> np.ndarray:
    """Complex band-pass covering centre +/- bandwidth/2.

    Complex, and therefore asymmetric: a real low-pass passes mirror-image frequencies
    either side of zero, which is exactly what must not happen when one sideband or one
    audio pitch is wanted.
    """
    half = min(bandwidth_hz / 2.0 / sample_rate, 0.24)
    taps = lowpass_taps(half, fir_length_for(half))
    n = np.arange(taps.size) - (taps.size - 1) / 2.0
    return taps * np.exp(2j * np.pi * (centre_hz / sample_rate) * n)
