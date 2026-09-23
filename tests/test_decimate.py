"""Decimation tests, including the property that actually matters: alias rejection."""

import numpy as np
import pytest

from src.rgc_sdr.dsp.decimate import Decimator, lowpass_taps
from src.rgc_sdr.dsp.spectrum import SpectrumAnalyzer

FS = 768e3


def tone(n, freq_hz, fs=FS, amp=1.0):
    t = np.arange(n) / fs
    return (amp * np.exp(2j * np.pi * freq_hz * t)).astype(np.complex64)


def test_unity_factor_is_a_passthrough():
    d = Decimator(1)
    x = tone(1000, 10e3)
    assert d.process(x) is x
    assert d.stages == 0
    assert d.input_for_output(4096) == 4096


def test_rejects_non_power_of_two():
    for bad in (0, 3, 6, 12, -2):
        with pytest.raises(ValueError):
            Decimator(bad)


@pytest.mark.parametrize("factor,stages", [(2, 1), (4, 2), (8, 3), (16, 4), (32, 5)])
def test_stage_count_and_rate(factor, stages):
    d = Decimator(factor)
    assert d.stages == stages
    assert d.effective_rate(FS) == pytest.approx(FS / factor)


@pytest.mark.parametrize("factor", [2, 4, 8, 16, 32])
def test_input_for_output_yields_at_least_what_was_asked(factor):
    d = Decimator(factor)
    for n_out in (1024, 4096, 16384):
        got = d.process(tone(d.input_for_output(n_out), 1e3)).size
        assert got >= n_out, f"factor {factor}: asked {n_out}, got {got}"


@pytest.mark.parametrize("factor", [2, 4, 8, 16, 32])
def test_in_band_tone_survives_at_the_right_frequency(factor):
    """A tone inside the surviving passband must land on the same absolute frequency."""
    d = Decimator(factor)
    fs_out = FS / factor
    offset = fs_out * 0.25          # comfortably inside the passband
    sa = SpectrumAnalyzer(fft_size=2048, max_segments=4)
    x = tone(d.input_for_output(sa.samples_wanted()), offset)
    db = sa.psd_dbfs(d.process(x))
    axis = sa.freq_axis(0.0, fs_out)
    assert axis[int(np.argmax(db))] == pytest.approx(offset, abs=fs_out / 2048 * 2)


@pytest.mark.parametrize("factor", [2, 4, 8, 16, 32])
def test_out_of_band_tone_is_rejected_not_aliased(factor):
    """The point of the anti-alias filter.

    A full-scale tone well outside the surviving passband must not reappear inside it.
    Without filtering it would fold in at full strength and read as a real signal.
    """
    d = Decimator(factor)
    fs_out = FS / factor
    # Just past the far side of the first alias zone, where folding would be worst.
    offset = fs_out * 0.85
    sa = SpectrumAnalyzer(fft_size=2048, max_segments=4)
    x = tone(d.input_for_output(sa.samples_wanted()), offset)
    db = sa.psd_dbfs(d.process(x))
    assert db.max() < -45.0, f"factor {factor}: alias at {db.max():.1f} dBFS"


def test_measured_stopband_rejection_is_documented():
    """Pin the filter's actual rejection so a tap/window change cannot quietly weaken it."""
    h = lowpass_taps(0.25)
    H = np.abs(np.fft.rfft(h, 4096))
    freqs = np.fft.rfftfreq(4096)
    passband = 20 * np.log10(H[freqs < 0.20].max())
    stopband = 20 * np.log10(H[freqs > 0.30].max())
    assert passband == pytest.approx(0.0, abs=0.5)
    assert stopband < -60.0, f"stopband only {stopband:.1f} dB"


def test_decimation_sharpens_resolution():
    """Two tones unresolvable at full rate must separate once decimated."""
    sa = SpectrumAnalyzer(fft_size=1024, max_segments=1)
    spacing = FS / 1024 * 0.6          # finer than one full-rate bin
    d = Decimator(16)
    n = d.input_for_output(sa.samples_wanted())
    x = tone(n, 1e3) + tone(n, 1e3 + spacing)

    wide = sa.psd_dbfs(x[: sa.samples_wanted()])
    narrow = sa.psd_dbfs(d.process(x))
    # Count distinct strong peaks in each.
    def peaks(db):
        thresh = db.max() - 6.0
        above = db > thresh
        return int(np.sum(above[1:] & ~above[:-1])) + int(above[0])
    assert peaks(wide) == 1, "tones should be merged at full rate"
    assert peaks(narrow) == 2, "tones should resolve after decimation"


def test_lowpass_taps_validates_arguments():
    for bad in (0.0, 0.5, 0.7, -0.1):
        with pytest.raises(ValueError):
            lowpass_taps(bad)
    with pytest.raises(ValueError):
        lowpass_taps(0.25, num_taps=2)


def test_short_input_returns_empty_rather_than_raising():
    assert Decimator(8).process(np.zeros(4, dtype=np.complex64)).size == 0
