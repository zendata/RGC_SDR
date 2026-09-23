"""Signal detection for the scanner: plan sweep windows, find carriers in a spectrum.

Pure NumPy, no Qt, no device access, so the whole detection path is testable against
synthetic spectra.

The scanner is spectrum-based rather than step-and-dwell. A step-and-dwell scanner
retunes to every channel in turn, so airband's 19 MHz at 25 kHz spacing is 760 retunes.
This receiver already sees 768 kHz at once, so the same range is about 31 windows and one
FFT finds every active channel in each -- two orders of magnitude fewer retunes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One carrier found in a spectrum."""

    freq_hz: float
    level_dbfs: float
    snr_db: float


def snap_to_grid(freq_hz: float, step_hz: float) -> float:
    """Round to the nearest channel on a `step_hz` grid."""
    if step_hz <= 0:
        return float(freq_hz)
    return float(round(freq_hz / step_hz) * step_hz)


def plan_windows(
    start_hz: float,
    end_hz: float,
    span_hz: float,
    usable_fraction: float = 0.8,
    grid_hz: float | None = None,
) -> list[float]:
    """Centre frequencies whose usable portions cover [start_hz, end_hz].

    Only the middle `usable_fraction` of each window is trusted, because the receiver's
    own filter rolls off towards the span edges. Centres are deliberately placed *off*
    the channel grid (by half a step) so no real channel ever lands on DC, where a
    direct-conversion receiver may leak its local oscillator.
    """
    if end_hz <= start_hz:
        raise ValueError("end_hz must be above start_hz")
    if span_hz <= 0:
        raise ValueError("span_hz must be positive")
    if not 0.0 < usable_fraction <= 1.0:
        raise ValueError("usable_fraction must be in (0, 1]")

    usable = span_hz * usable_fraction
    count = max(1, math.ceil((end_hz - start_hz) / usable))
    centres = [start_hz + usable * (i + 0.5) for i in range(count)]
    if grid_hz:
        # Half a channel off the grid: never put a channel on DC.
        centres = [snap_to_grid(c, grid_hz) + grid_hz / 2.0 for c in centres]
    return centres


def usable_edges(
    centre_hz: float, span_hz: float, usable_fraction: float = 0.8
) -> tuple[float, float]:
    half = span_hz * usable_fraction / 2.0
    return (centre_hz - half, centre_hz + half)


def noise_floor_dbfs(dbfs: np.ndarray) -> float:
    """Median level, which is a robust noise estimate even with carriers present."""
    return float(np.median(dbfs)) if dbfs.size else -200.0


def find_peaks(dbfs: np.ndarray, threshold_dbfs: float) -> np.ndarray:
    """Indices of the strongest bin in each contiguous run above `threshold_dbfs`.

    One index per run rather than per bin: a carrier occupies several bins, and counting
    each would report one signal many times.
    """
    above = dbfs > threshold_dbfs
    indices = np.flatnonzero(above)
    if indices.size == 0:
        return np.empty(0, dtype=int)
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    return np.array([run[int(np.argmax(dbfs[run]))] for run in np.split(indices, breaks)])


def detect_channels(
    freqs: np.ndarray,
    dbfs: np.ndarray,
    *,
    threshold_db: float = 10.0,
    step_hz: float = 25e3,
    usable_lo: float | None = None,
    usable_hi: float | None = None,
    dc_guard_hz: float = 1.5e3,
    centre_hz: float | None = None,
    lockout: frozenset[float] | set[float] | tuple = (),
) -> list[Detection]:
    """Active channels in one spectrum.

    `threshold_db` is measured above the noise floor rather than as an absolute level, so
    the same setting works on a quiet band and a busy one, and on any antenna.
    """
    if freqs.size != dbfs.size:
        raise ValueError("freqs and dbfs must be the same length")
    if freqs.size == 0:
        return []

    window = np.ones(freqs.size, dtype=bool)
    if usable_lo is not None:
        window &= freqs >= usable_lo
    if usable_hi is not None:
        window &= freqs <= usable_hi
    if centre_hz is not None and dc_guard_hz > 0:
        window &= np.abs(freqs - centre_hz) > dc_guard_hz
    if not window.any():
        return []

    sub_freqs = freqs[window]
    sub_dbfs = dbfs[window]
    noise = noise_floor_dbfs(sub_dbfs)
    peaks = find_peaks(sub_dbfs, noise + threshold_db)

    locked = {snap_to_grid(f, step_hz) for f in lockout}
    best: dict[float, Detection] = {}
    for index in peaks:
        channel = snap_to_grid(float(sub_freqs[index]), step_hz)
        if channel in locked:
            continue
        level = float(sub_dbfs[index])
        found = Detection(channel, level, level - noise)
        previous = best.get(channel)
        if previous is None or found.level_dbfs > previous.level_dbfs:
            best[channel] = found
    return [best[key] for key in sorted(best)]
