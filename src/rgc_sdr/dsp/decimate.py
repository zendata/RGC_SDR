"""Integer decimation of complex baseband IQ.

Decimating by M narrows the visible span by M and sharpens resolution by M, which is what
a "zoom" control does on a spectrum display. It is also the front half of demodulation:
the audio chain decimates down to a few tens of kHz before detecting.

Pure NumPy, no Qt, no device access (layering rule, PLANNING.md section 5).

Implemented as a cascade of halving stages rather than one filter per factor. A single
filter sized for M=32 would need roughly 40*M taps to keep its transition band narrower
than the surviving passband -- about 1280 taps, whose polyphase temporaries run to tens of
megabytes. Each halving stage instead faces the same relative problem, so a fixed modest
filter is correct at every stage and the cost falls geometrically.
"""

from __future__ import annotations

import numpy as np

#: Taps per halving stage. Blackman-windowed sinc; see tests for the measured rejection.
STAGE_TAPS = 63


def lowpass_taps(cutoff: float, num_taps: int = STAGE_TAPS) -> np.ndarray:
    """Windowed-sinc low-pass. `cutoff` is normalised to the sample rate (0 < c < 0.5)."""
    if not 0.0 < cutoff < 0.5:
        raise ValueError("cutoff must be in (0, 0.5)")
    if num_taps < 3:
        raise ValueError("num_taps must be >= 3")
    m = np.arange(num_taps) - (num_taps - 1) / 2.0
    h = np.sinc(2.0 * cutoff * m) * np.blackman(num_taps)
    return (h / h.sum()).astype(np.float64)


class Decimator:
    """Decimate complex IQ by a power-of-two factor.

    Stateless between calls: each frame decimates a fresh block, which suits a display
    that only ever wants the newest samples. A continuous consumer such as audio will
    want a stateful variant that carries filter history across blocks.
    """

    def __init__(self, factor: int = 1, taps_per_stage: int = STAGE_TAPS) -> None:
        if factor < 1 or (factor & (factor - 1)):
            raise ValueError("factor must be a power of two >= 1")
        self._factor = int(factor)
        self._taps_per_stage = int(taps_per_stage)
        # Each stage halves the rate, so its output Nyquist is a quarter of its input
        # rate: the anti-alias cutoff is 0.25 normalised to that stage's input.
        stages = int(self._factor).bit_length() - 1
        self._taps = [lowpass_taps(0.25, self._taps_per_stage) for _ in range(stages)]

    @property
    def factor(self) -> int:
        return self._factor

    @property
    def stages(self) -> int:
        return len(self._taps)

    def effective_rate(self, sample_rate: float) -> float:
        return sample_rate / self._factor

    def input_for_output(self, n_out: int) -> int:
        """Input samples needed to yield at least `n_out` decimated samples."""
        need = int(n_out)
        for taps in reversed(self._taps):
            need = need * 2 + taps.size - 1
        return need

    def process(self, iq: np.ndarray) -> np.ndarray:
        """Decimate, dropping the filter's edge transient via 'valid' convolution."""
        if self._factor == 1:
            return iq
        out = iq
        for taps in self._taps:
            if out.size < taps.size:
                return out[:0]
            out = np.convolve(out, taps, mode="valid")[::2]
        return out
