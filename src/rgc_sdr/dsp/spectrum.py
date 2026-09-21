"""Power-spectrum estimation for complex baseband IQ.

Pure NumPy, no Qt, no device access (layering rule, PLANNING.md section 5), so this is
verifiable headless with synthetic arrays.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-30  # floor so log10 never sees zero


def hann_periodic(n: int) -> np.ndarray:
    """Periodic (DFT-even) Hann window.

    `np.hanning` is the *symmetric* variant, which is for filter design; spectral
    analysis wants the periodic form or the tone leaks into neighbouring bins.
    """
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float64)


class SpectrumAnalyzer:
    """Welch-averaged periodogram in dBFS.

    A single FFT of 4096 points covers only ~13% of the samples that arrive between
    frames at 25 FPS, so several overlapping segments are averaged per frame. That
    smooths the noise floor and stops short bursts being missed. Power is averaged and
    *then* converted to dB -- averaging dB values is wrong.
    """

    def __init__(
        self,
        fft_size: int = 4096,
        overlap: float = 0.5,
        max_segments: int = 16,
    ) -> None:
        if fft_size < 8 or fft_size & (fft_size - 1):
            raise ValueError("fft_size must be a power of two >= 8")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        self.fft_size = int(fft_size)
        self.overlap = float(overlap)
        self.max_segments = max(1, int(max_segments))
        self._window = hann_periodic(self.fft_size)
        # Coherent gain. A full-scale complex tone sums coherently into one bin, so the
        # normaliser is sum(w) -- not sum(w)/2, which is the real-signal case and reads
        # 6 dB hot. Pinned by test_full_scale_tone_reads_zero_dbfs.
        self._coherent_gain = float(self._window.sum())

    @property
    def hop(self) -> int:
        return max(1, int(self.fft_size * (1.0 - self.overlap)))

    def samples_wanted(self) -> int:
        """Samples needed to fill `max_segments` segments -- what a frame should fetch."""
        return self.fft_size + (self.max_segments - 1) * self.hop

    def psd_dbfs(self, iq: np.ndarray) -> np.ndarray:
        """dBFS spectrum, fftshifted so DC sits in the centre.

        Returns `fft_size` bins. Raises ValueError if given less than one segment.
        """
        n = self.fft_size
        if iq.size < n:
            raise ValueError(f"need at least {n} samples, got {iq.size}")

        hop = self.hop
        nseg = min(self.max_segments, 1 + (iq.size - n) // hop)
        # One batched FFT over a strided view beats a Python loop over segments.
        idx = np.arange(nseg)[:, None] * hop + np.arange(n)[None, :]
        segs = iq[idx] * self._window
        spec = np.fft.fft(segs, axis=1)
        power = np.mean(spec.real**2 + spec.imag**2, axis=0)
        power = np.fft.fftshift(power)
        return (10.0 * np.log10(power / self._coherent_gain**2 + _TINY)).astype(np.float32)

    def freq_axis(self, center_hz: float, sample_rate: float) -> np.ndarray:
        """Absolute frequency of each bin, in Hz, matching `psd_dbfs` ordering."""
        n = self.fft_size
        return center_hz + (np.arange(n) - n // 2) * (sample_rate / n)
