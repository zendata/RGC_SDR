"""Measurements on a spectrum: markers, channel power, occupied bandwidth (P11).

All work on the display's own dBFS spectrum (`SpectrumAnalyzer.psd_dbfs`), so a marker
reads what the curve shows. Pure NumPy; no Qt, no device access.
"""

from __future__ import annotations

import numpy as np

#: A marker placed within this many bins of a peak moves onto it.
PEAK_SNAP_BINS = 4
#: Occupied bandwidth: the share of the power inside it (ITU-R SM.328: 99%).
OBW_FRACTION = 0.99


def level_at(freqs: np.ndarray, dbfs: np.ndarray, hz: float) -> float:
    """The spectrum's level at `hz` (nearest bin)."""
    return float(dbfs[int(np.argmin(np.abs(freqs - hz)))])


def peak_near(freqs: np.ndarray, dbfs: np.ndarray, hz: float,
              bins: int = PEAK_SNAP_BINS) -> float:
    """The frequency of the strongest bin within `bins` of `hz`."""
    k = int(np.argmin(np.abs(freqs - hz)))
    lo, hi = max(0, k - bins), min(freqs.size, k + bins + 1)
    return float(freqs[lo + int(np.argmax(dbfs[lo:hi]))])


def strongest(freqs: np.ndarray, dbfs: np.ndarray, low_hz: float, high_hz: float) -> float | None:
    """The frequency of the strongest bin between `low` and `high`, or None."""
    inside = (freqs >= low_hz) & (freqs <= high_hz)
    if not inside.any():
        return None
    return float(freqs[inside][int(np.argmax(dbfs[inside]))])


def channel_power_dbfs(freqs: np.ndarray, dbfs: np.ndarray, low_hz: float,
                       high_hz: float) -> float | None:
    """Total power between `low` and `high`, dBFS: the bins' powers summed."""
    inside = (freqs >= low_hz) & (freqs <= high_hz)
    if not inside.any():
        return None
    return float(10.0 * np.log10(np.sum(10.0 ** (dbfs[inside] / 10.0)) + 1e-30))


def occupied_bandwidth(freqs: np.ndarray, dbfs: np.ndarray, low_hz: float, high_hz: float,
                       fraction: float = OBW_FRACTION) -> float | None:
    """Width holding `fraction` of the power between `low` and `high`: half the rest cut
    from each side (the ITU's definition). None with fewer than two bins."""
    inside = (freqs >= low_hz) & (freqs <= high_hz)
    if np.count_nonzero(inside) < 2:
        return None
    f = freqs[inside]
    power = 10.0 ** (dbfs[inside] / 10.0)
    cumulative = np.cumsum(power) / np.sum(power)
    tail = (1.0 - fraction) / 2.0
    lo = int(np.searchsorted(cumulative, tail))
    hi = int(np.searchsorted(cumulative, 1.0 - tail))
    bin_hz = float(f[1] - f[0])
    return float(f[min(hi, f.size - 1)] - f[lo] + bin_hz)
