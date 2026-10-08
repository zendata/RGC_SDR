"""P11: spectrum measurements (dsp/measure.py) and the band plan (bandplan.py)."""

import numpy as np
import pytest

from src.rgc_sdr.bandplan import BANDS, KIND_COLOURS, bands_in
from src.rgc_sdr.dsp.measure import (
    channel_power_dbfs, level_at, occupied_bandwidth, peak_near, strongest,
)


def _spectrum():
    freqs = np.linspace(-50e3, 50e3, 1001)               # 100 Hz bins
    dbfs = np.full(freqs.size, -120.0)
    dbfs[600] = -40.0                                     # a carrier at +10 kHz
    dbfs[300:310] = -60.0                                 # a 1 kHz-wide signal at -20 kHz
    return freqs, dbfs


def test_levels_and_peaks():
    freqs, dbfs = _spectrum()
    assert level_at(freqs, dbfs, 10_020) == -40.0
    assert peak_near(freqs, dbfs, 10_300) == pytest.approx(10_000)      # snapped on
    assert peak_near(freqs, dbfs, 30_000) != 10_000                     # too far
    assert strongest(freqs, dbfs, -50e3, 50e3) == pytest.approx(10_000)
    assert strongest(freqs, dbfs, -30e3, 0) == pytest.approx(-20_000)
    assert strongest(freqs, dbfs, 60e3, 70e3) is None


def test_channel_power_sums_the_bins():
    freqs, dbfs = _spectrum()
    ten_bins = channel_power_dbfs(freqs, dbfs, -20_050, -19_050)
    assert ten_bins == pytest.approx(-60.0 + 10.0, abs=0.01)            # ten bins of -60
    assert channel_power_dbfs(freqs, dbfs, 60e3, 70e3) is None


def test_occupied_bandwidth_holds_99_percent():
    freqs, dbfs = _spectrum()
    obw = occupied_bandwidth(freqs, dbfs, -25e3, -15e3)
    assert obw == pytest.approx(1000.0, abs=100.0)                       # the 1 kHz signal
    assert occupied_bandwidth(freqs, dbfs, 9_850, 10_150) == pytest.approx(100.0, abs=100.0)


def test_the_band_plan_finds_what_is_in_view():
    names = [b.name for b in bands_in(144.5e6, 145.5e6)]
    assert "2 m amateur" in names
    airband = [b.name for b in bands_in(118e6, 137e6)]
    assert "Airband (AM voice, ACARS)" in airband
    assert all(b.kind in KIND_COLOURS for b in BANDS)
    assert all(b.low_hz < b.high_hz for b in BANDS)
    marine = bands_in(161.9e6, 162.1e6)
    assert marine[0].name == "Marine VHF" and marine[-1].name == "AIS"   # widest first
    assert bands_in(1e9, 1.01e9) == []
