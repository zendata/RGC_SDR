"""Measure how far a CW carrier is from where it should be, so it can be zero-beaten.

The display spectrum is far too coarse for this: 4096 bins across 768 kHz is 187.5 Hz per
bin, and the job is to place a carrier on a 700 Hz tone. So this runs its own narrow,
high-resolution measurement.

Resolution comes from a long transform plus parabolic interpolation rather than from a
huge FFT. A 32768-point transform gives 23.4 Hz bins, and fitting a parabola to the peak
and its two neighbours locates the true peak to a couple of hertz -- well inside what the
ear can detect on a beat note. It also covers only 43 ms, which matters for CW: a longer
window would average across the gaps between dots and smear the carrier it is looking for.

Pure NumPy, no Qt, no device access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Transform length. 43 ms at 768 kS/s: long enough to resolve, short enough to sit
#: inside a single CW element rather than spanning several.
DEFAULT_FFT = 32768


@dataclass(frozen=True)
class CarrierMeasurement:
    """Where a carrier actually is, relative to the tuned centre."""

    #: Baseband frequency of the carrier, in Hz relative to the tuned centre.
    carrier_hz: float
    #: How far it is from where it ought to be. Add this to the centre frequency.
    error_hz: float
    snr_db: float


def interpolate_peak(dbfs: np.ndarray, index: int) -> float:
    """Sub-bin position of a peak, by fitting a parabola to it and its neighbours.

    Returns an offset in bins, in [-0.5, 0.5]. Without this the best achievable tuning
    would be half a bin, which at 23.4 Hz bins is an audible 12 Hz error on a beat note.
    """
    if index <= 0 or index >= dbfs.size - 1:
        return 0.0
    left, centre, right = float(dbfs[index - 1]), float(dbfs[index]), float(dbfs[index + 1])
    denominator = left - 2.0 * centre + right
    if denominator == 0.0:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(np.clip(offset, -0.5, 0.5))


def measure_carrier(
    iq: np.ndarray,
    sample_rate: float,
    listen_hz: float = 0.0,
    search_hz: float = 500.0,
    min_snr_db: float = 8.0,
    fft_size: int = DEFAULT_FFT,
) -> CarrierMeasurement | None:
    """Find the strongest carrier within `search_hz` of `listen_hz`.

    Returns None when there is nothing convincing there -- no signal, or only noise --
    so the caller can leave the tuning alone rather than chasing a noise peak.
    """
    if search_hz <= 0.0:
        raise ValueError("search_hz must be positive")
    if iq.size < fft_size:
        return None

    segment = iq[-fft_size:] * np.hanning(fft_size)
    spectrum = np.fft.fftshift(np.fft.fft(segment))
    power = spectrum.real**2 + spectrum.imag**2
    dbfs = 10.0 * np.log10(power + 1e-30)
    bin_hz = sample_rate / fft_size
    freqs = (np.arange(fft_size) - fft_size // 2) * bin_hz

    window = np.abs(freqs - listen_hz) <= search_hz
    if not window.any():
        return None

    # The noise reference comes from a band wider than the search, so a strong carrier
    # filling the search window cannot raise its own reference and hide itself.
    reference = np.abs(freqs - listen_hz) <= max(search_hz * 4.0, 4000.0)
    noise_db = float(np.median(dbfs[reference])) if reference.any() else float(np.median(dbfs))

    indices = np.flatnonzero(window)
    peak = int(indices[int(np.argmax(dbfs[indices]))])
    snr = float(dbfs[peak]) - noise_db
    if snr < min_snr_db:
        return None

    carrier_hz = float(freqs[peak] + interpolate_peak(dbfs, peak) * bin_hz)
    return CarrierMeasurement(carrier_hz, carrier_hz - float(listen_hz), snr)
