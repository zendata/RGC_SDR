"""Spectrum DSP tests: synthetic IQ arrays, no radio, no Qt."""

import numpy as np
import pytest

from src.rgc_sdr.dsp.spectrum import SpectrumAnalyzer, hann_periodic


def complex_tone(n, freq_hz, fs, amplitude=1.0):
    """A single-sided complex exponential -- what a real IQ receiver delivers."""
    t = np.arange(n) / fs
    return (amplitude * np.exp(2j * np.pi * freq_hz * t)).astype(np.complex64)


def test_hann_periodic_is_not_symmetric_variant():
    n = 64
    w = hann_periodic(n)
    assert w[0] == pytest.approx(0.0)
    # The periodic window does not return to zero at the last sample; np.hanning does.
    assert w[-1] > 0.0
    assert np.hanning(n)[-1] == pytest.approx(0.0)


def test_tone_lands_in_expected_bin():
    fs, fft = 768_000.0, 4096
    offset = 96_000.0  # exactly on a bin centre: 96000 / (768000/4096) = 512 bins
    sa = SpectrumAnalyzer(fft_size=fft)
    db = sa.psd_dbfs(complex_tone(sa.samples_wanted(), offset, fs))
    expected = fft // 2 + int(round(offset / (fs / fft)))
    assert abs(int(np.argmax(db)) - expected) <= 1


def test_full_scale_tone_reads_zero_dbfs():
    """Pins the sum(w) normalisation. sum(w)/2 would read +6 dB here."""
    fs, fft = 768_000.0, 4096
    sa = SpectrumAnalyzer(fft_size=fft)
    db = sa.psd_dbfs(complex_tone(sa.samples_wanted(), 96_000.0, fs, amplitude=1.0))
    assert db.max() == pytest.approx(0.0, abs=0.1)


def test_half_scale_tone_reads_minus_six_dbfs():
    fs = 768_000.0
    sa = SpectrumAnalyzer(fft_size=4096)
    db = sa.psd_dbfs(complex_tone(sa.samples_wanted(), 96_000.0, fs, amplitude=0.5))
    assert db.max() == pytest.approx(-6.02, abs=0.1)


def test_no_mirror_image_for_complex_input():
    """Catches fftshift/sideband errors and real-valued 'IQ'.

    A complex tone at +offset must leave the -offset bin at the noise floor. A
    real-valued signal (the v1 bug) would light up both.
    """
    fs, fft = 768_000.0, 4096
    offset = 96_000.0
    sa = SpectrumAnalyzer(fft_size=fft)
    db = sa.psd_dbfs(complex_tone(sa.samples_wanted(), offset, fs))
    bins = int(round(offset / (fs / fft)))
    pos, neg = fft // 2 + bins, fft // 2 - bins
    assert db[pos] - db[neg] > 100.0


def test_dc_lands_in_centre_bin():
    sa = SpectrumAnalyzer(fft_size=1024)
    db = sa.psd_dbfs(np.ones(sa.samples_wanted(), dtype=np.complex64))
    assert int(np.argmax(db)) == 512


def test_negative_frequency_is_left_of_centre():
    fs, fft = 768_000.0, 4096
    sa = SpectrumAnalyzer(fft_size=fft)
    db = sa.psd_dbfs(complex_tone(sa.samples_wanted(), -96_000.0, fs))
    assert int(np.argmax(db)) < fft // 2


def test_averaging_lowers_noise_variance():
    """More Welch segments must give a flatter floor, or the averaging is not working."""
    rng = np.random.default_rng(0)
    n = 4096 + 63 * 2048
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.01
    one = SpectrumAnalyzer(fft_size=4096, max_segments=1).psd_dbfs(noise)
    many = SpectrumAnalyzer(fft_size=4096, max_segments=32).psd_dbfs(noise)
    assert many.std() < one.std() * 0.6


def test_freq_axis_matches_bin_ordering():
    fs, center, fft = 768_000.0, 7.1e6, 4096
    sa = SpectrumAnalyzer(fft_size=fft)
    axis = sa.freq_axis(center, fs)
    assert axis.size == fft
    assert axis[fft // 2] == pytest.approx(center)
    assert axis[0] == pytest.approx(center - fs / 2)
    assert np.all(np.diff(axis) > 0)
    db = sa.psd_dbfs(complex_tone(sa.samples_wanted(), 96_000.0, fs))
    assert axis[int(np.argmax(db))] == pytest.approx(center + 96_000.0, abs=fs / fft)


def test_samples_wanted_is_sufficient_and_minimal():
    sa = SpectrumAnalyzer(fft_size=4096, overlap=0.5, max_segments=16)
    assert sa.samples_wanted() == 4096 + 15 * 2048
    sa.psd_dbfs(np.ones(sa.samples_wanted(), dtype=np.complex64))
    with pytest.raises(ValueError):
        sa.psd_dbfs(np.ones(4095, dtype=np.complex64))


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        SpectrumAnalyzer(fft_size=1000)
