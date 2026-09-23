"""Detection tests against synthetic spectra: no radio, no Qt."""

import numpy as np
import pytest

from src.rgc_sdr.dsp.detect import (
    detect_channels,
    find_peaks,
    noise_floor_dbfs,
    plan_windows,
    snap_to_grid,
    usable_edges,
)

SPAN = 768e3


def spectrum(centre_hz, span_hz=SPAN, bins=4096, floor=-118.0, carriers=(), noise=0.7, seed=3):
    """A plausible spectrum: flat noise plus a few carriers of given (freq, dB above floor)."""
    rng = np.random.default_rng(seed)
    freqs = centre_hz + (np.arange(bins) - bins // 2) * (span_hz / bins)
    dbfs = floor + rng.normal(0.0, noise, bins)
    for freq, excess in carriers:
        idx = int(np.argmin(np.abs(freqs - freq)))
        for offset, share in ((0, 1.0), (-1, 0.6), (1, 0.6), (-2, 0.25), (2, 0.25)):
            j = idx + offset
            if 0 <= j < bins:
                dbfs[j] = max(dbfs[j], floor + excess * share)
    return freqs, dbfs


# -- grid and window planning ------------------------------------------------

def test_snap_to_grid():
    assert snap_to_grid(118_247_000, 25e3) == pytest.approx(118_250_000)
    assert snap_to_grid(118_260_000, 25e3) == pytest.approx(118_250_000)
    assert snap_to_grid(118_263_000, 25e3) == pytest.approx(118_275_000)


def test_snap_to_grid_with_no_grid_is_identity():
    assert snap_to_grid(118_247_123, 0) == pytest.approx(118_247_123)


def test_plan_windows_covers_the_whole_range():
    start, end = 118e6, 137e6
    centres = plan_windows(start, end, SPAN, usable_fraction=0.8)
    covered = [usable_edges(c, SPAN, 0.8) for c in centres]
    assert covered[0][0] <= start + 1.0
    assert covered[-1][1] >= end - 1.0
    # No gaps between consecutive windows.
    for (_, hi), (lo, _) in zip(covered, covered[1:]):
        assert lo <= hi + 1.0, "gap between windows"


def test_plan_windows_is_efficient_for_airband():
    """Spectrum-based scanning should need tens of retunes, not hundreds."""
    centres = plan_windows(118e6, 137e6, SPAN, usable_fraction=0.8)
    assert 25 <= len(centres) <= 45
    # A step-and-dwell scanner at 25 kHz spacing would need this many:
    assert len(centres) < (137e6 - 118e6) / 25e3 / 10


def test_plan_windows_keeps_centres_off_the_channel_grid():
    """A channel must never land on DC, where some radios leak their LO."""
    for centre in plan_windows(118e6, 137e6, SPAN, grid_hz=25e3):
        offset = abs(centre % 25e3)
        assert min(offset, 25e3 - offset) > 25e3 * 0.4


def test_plan_windows_handles_a_range_narrower_than_one_span():
    centres = plan_windows(118e6, 118.1e6, SPAN)
    assert len(centres) == 1
    lo, hi = usable_edges(centres[0], SPAN)
    assert lo <= 118e6 and hi >= 118.1e6


def test_plan_windows_rejects_nonsense():
    for args in ((137e6, 118e6, SPAN), (118e6, 118e6, SPAN), (118e6, 137e6, 0.0)):
        with pytest.raises(ValueError):
            plan_windows(*args)
    with pytest.raises(ValueError):
        plan_windows(118e6, 137e6, SPAN, usable_fraction=1.5)


# -- peak finding ------------------------------------------------------------

def test_find_peaks_returns_one_index_per_carrier():
    """A carrier spans several bins; reporting each would be one signal counted many times."""
    dbfs = np.full(1000, -120.0)
    for centre in (100, 500, 800):
        dbfs[centre - 2 : centre + 3] = [-105, -100, -95, -100, -105]
    peaks = find_peaks(dbfs, -110.0)
    assert peaks.size == 3
    assert sorted(peaks.tolist()) == [100, 500, 800]


def test_find_peaks_empty_when_nothing_is_above():
    assert find_peaks(np.full(100, -120.0), -110.0).size == 0


def test_noise_floor_is_robust_to_strong_carriers():
    dbfs = np.full(1000, -118.0)
    dbfs[400:430] = -60.0                      # a big signal
    assert noise_floor_dbfs(dbfs) == pytest.approx(-118.0, abs=0.5)


# -- channel detection -------------------------------------------------------

def test_detects_carriers_on_the_channel_grid():
    freqs, dbfs = spectrum(125e6, carriers=[(125_050_000, 25), (125_200_000, 18)])
    found = detect_channels(freqs, dbfs, threshold_db=10.0, step_hz=25e3, centre_hz=125e6)
    assert [d.freq_hz for d in found] == [125_050_000, 125_200_000]
    assert all(d.snr_db > 10.0 for d in found)


def test_quiet_band_yields_nothing():
    freqs, dbfs = spectrum(125e6, carriers=[])
    assert detect_channels(freqs, dbfs, threshold_db=10.0, centre_hz=125e6) == []


def test_threshold_is_relative_to_the_noise_floor():
    """The same threshold must work on a quiet antenna and a loud one."""
    for floor in (-140.0, -118.0, -80.0):
        freqs, dbfs = spectrum(125e6, floor=floor, carriers=[(125_050_000, 20)])
        found = detect_channels(freqs, dbfs, threshold_db=10.0, centre_hz=125e6)
        assert [d.freq_hz for d in found] == [125_050_000], f"failed at floor {floor}"


def test_weak_signal_below_threshold_is_ignored():
    freqs, dbfs = spectrum(125e6, carriers=[(125_050_000, 5)])
    assert detect_channels(freqs, dbfs, threshold_db=12.0, centre_hz=125e6) == []


def test_window_edges_are_excluded():
    """Signals in the receiver's filter rolloff must not be reported."""
    freqs, dbfs = spectrum(125e6, carriers=[(125_370_000, 25)])   # near the span edge
    lo, hi = usable_edges(125e6, SPAN, 0.8)
    assert detect_channels(freqs, dbfs, usable_lo=lo, usable_hi=hi, centre_hz=125e6) == []
    # Without the restriction it is found, so the exclusion is what suppressed it.
    assert detect_channels(freqs, dbfs, centre_hz=125e6) != []


def test_dc_guard_suppresses_a_centre_artefact():
    freqs, dbfs = spectrum(125e6, carriers=[(125e6, 30)])
    assert detect_channels(freqs, dbfs, centre_hz=125e6, dc_guard_hz=5e3) == []
    assert detect_channels(freqs, dbfs, centre_hz=125e6, dc_guard_hz=0.0) != []


def test_locked_out_channels_are_not_reported():
    freqs, dbfs = spectrum(125e6, carriers=[(125_050_000, 25), (125_200_000, 20)])
    found = detect_channels(
        freqs, dbfs, step_hz=25e3, centre_hz=125e6, lockout={125_050_000},
    )
    assert [d.freq_hz for d in found] == [125_200_000]


def test_lockout_matches_on_the_grid_not_the_exact_value():
    """A lockout entered slightly off-channel must still suppress that channel."""
    freqs, dbfs = spectrum(125e6, carriers=[(125_050_000, 25)])
    found = detect_channels(
        freqs, dbfs, step_hz=25e3, centre_hz=125e6, lockout={125_047_300},
    )
    assert found == []


def test_two_bins_of_one_carrier_collapse_to_a_single_channel():
    freqs, dbfs = spectrum(125e6, carriers=[(125_050_000, 25), (125_053_000, 22)])
    found = detect_channels(freqs, dbfs, step_hz=25e3, centre_hz=125e6)
    assert len(found) == 1 and found[0].freq_hz == pytest.approx(125_050_000)


def test_results_are_sorted_by_frequency():
    freqs, dbfs = spectrum(125e6, carriers=[(125_200_000, 20), (124_900_000, 22), (125_050_000, 25)])
    found = detect_channels(freqs, dbfs, step_hz=25e3, centre_hz=125e6)
    assert [d.freq_hz for d in found] == sorted(d.freq_hz for d in found)


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError):
        detect_channels(np.zeros(10), np.zeros(11))


def test_empty_spectrum_is_handled():
    assert detect_channels(np.zeros(0), np.zeros(0)) == []
