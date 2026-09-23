"""Carrier measurement for CW zero-beating: synthetic signals, no radio."""

import numpy as np
import pytest

from src.rgc_sdr.dsp.zerobeat import (
    DEFAULT_FFT,
    interpolate_peak,
    measure_carrier,
)

FS = 768e3


def carrier(n, offset_hz, fs=FS, amplitude=0.3, noise=1e-3, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    signal = amplitude * np.exp(2j * np.pi * offset_hz * t)
    hiss = noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return (signal + hiss).astype(np.complex64)


def noise_only(n, fs=FS, level=1e-3, seed=9):
    rng = np.random.default_rng(seed)
    return (level * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)


# -- interpolation -----------------------------------------------------------

def test_interpolation_finds_the_centre_of_a_symmetric_peak():
    dbfs = np.array([-20.0, -10.0, -20.0])
    assert interpolate_peak(dbfs, 1) == pytest.approx(0.0)


def test_interpolation_leans_toward_the_taller_neighbour():
    assert interpolate_peak(np.array([-12.0, -10.0, -20.0]), 1) < 0.0
    assert interpolate_peak(np.array([-20.0, -10.0, -12.0]), 1) > 0.0


def test_interpolation_is_bounded_to_half_a_bin():
    for triple in ([-10.0, -10.0, -30.0], [-30.0, -10.0, -10.0], [0.0, 0.0, 0.0]):
        assert -0.5 <= interpolate_peak(np.array(triple), 1) <= 0.5


def test_interpolation_at_the_array_edges_is_safe():
    dbfs = np.array([-10.0, -20.0, -30.0])
    assert interpolate_peak(dbfs, 0) == 0.0
    assert interpolate_peak(dbfs, 2) == 0.0


# -- measurement -------------------------------------------------------------

@pytest.mark.parametrize("offset", [0.0, 37.0, -37.0, 150.0, -150.0, 480.0, -480.0])
def test_a_carrier_is_located_to_within_a_few_hertz(offset):
    """Accuracy is what makes zero-beating possible; bin width alone is not enough."""
    found = measure_carrier(carrier(DEFAULT_FFT * 2, offset), FS)
    assert found is not None, f"missed a carrier at {offset} Hz"
    assert found.carrier_hz == pytest.approx(offset, abs=4.0)
    assert found.error_hz == pytest.approx(offset, abs=4.0)


def test_accuracy_beats_the_bin_width():
    """A carrier deliberately between bins must still be found accurately."""
    bin_hz = FS / DEFAULT_FFT
    offset = bin_hz * 7.5          # exactly between two bins
    found = measure_carrier(carrier(DEFAULT_FFT * 2, offset), FS)
    assert abs(found.carrier_hz - offset) < bin_hz / 4.0


def test_nothing_is_reported_for_noise_alone():
    """No nearby signal means leave the tuning alone, not chase a noise peak."""
    assert measure_carrier(noise_only(DEFAULT_FFT * 2), FS) is None


def test_the_threshold_rejects_what_it_is_set_above():
    """Measured against the signal's own SNR rather than a guessed amplitude.

    Worth knowing: a carrier at amplitude 2e-4 in 1e-3 of noise is *negative* SNR in the
    time domain yet over 25 dB in the spectrum, because a 32768-point transform
    concentrates the coherent carrier into one bin while noise spreads across all of
    them. That processing gain is the whole reason narrowband detection works, and it is
    why "weak" has to be defined spectrally.
    """
    faint = carrier(DEFAULT_FFT * 2, 100.0, amplitude=2e-4, noise=1e-3)
    detected = measure_carrier(faint, FS, min_snr_db=0.0)
    assert detected is not None
    assert detected.snr_db > 10.0, "processing gain should make this easily visible"
    assert measure_carrier(faint, FS, min_snr_db=detected.snr_db + 3.0) is None


def test_a_carrier_outside_the_search_window_is_not_chased():
    """The search is deliberately bounded: a signal 2 kHz away is not this signal."""
    assert measure_carrier(carrier(DEFAULT_FFT * 2, 2000.0), FS, search_hz=500.0) is None


def test_the_search_window_is_centred_on_where_we_are_listening():
    signal = carrier(DEFAULT_FFT * 2, 3000.0)
    assert measure_carrier(signal, FS, listen_hz=0.0, search_hz=500.0) is None
    found = measure_carrier(signal, FS, listen_hz=3000.0, search_hz=500.0)
    assert found is not None
    assert found.carrier_hz == pytest.approx(3000.0, abs=5.0)
    assert found.error_hz == pytest.approx(0.0, abs=5.0)


def test_error_is_signed_so_the_caller_knows_which_way_to_go():
    above = measure_carrier(carrier(DEFAULT_FFT * 2, 200.0), FS)
    below = measure_carrier(carrier(DEFAULT_FFT * 2, -200.0), FS)
    assert above.error_hz > 0.0
    assert below.error_hz < 0.0


def test_the_stronger_of_two_carriers_wins():
    n = DEFAULT_FFT * 2
    both = carrier(n, 120.0, amplitude=0.3) + carrier(n, -300.0, amplitude=0.05, seed=11)
    found = measure_carrier(both, FS)
    assert found.carrier_hz == pytest.approx(120.0, abs=6.0)


def test_a_strong_carrier_cannot_mask_itself():
    """The noise reference spans wider than the search, so a big signal filling the
    window does not raise its own floor above the threshold."""
    found = measure_carrier(carrier(DEFAULT_FFT * 2, 0.0, amplitude=1.0), FS)
    assert found is not None and found.snr_db > 30.0


def test_too_little_data_returns_nothing():
    assert measure_carrier(carrier(1024, 100.0), FS) is None


def test_search_width_must_be_positive():
    with pytest.raises(ValueError):
        measure_carrier(carrier(DEFAULT_FFT * 2, 0.0), FS, search_hz=0.0)


def test_measurement_cost_is_acceptable_for_interactive_use():
    import time

    signal = carrier(DEFAULT_FFT * 2, 100.0)
    measure_carrier(signal, FS)                     # warm up
    start = time.perf_counter()
    for _ in range(10):
        measure_carrier(signal, FS)
    ms = (time.perf_counter() - start) / 10 * 1000
    assert ms < 30.0, f"{ms:.1f} ms per measurement is too slow to hold a button down"


def test_the_default_threshold_rejects_marginal_peaks():
    """Set from on-air measurement, not taste.

    Against a real 40 m CW signal the genuine carrier read 26-29 dB while noise peaks
    and marginal neighbours read 8-10 dB. An 8 dB default acted on those and dragged the
    tuning 770 Hz onto a different signal, so the default sits above them.
    """
    from src.rgc_sdr.dsp.zerobeat import MIN_SNR_DB

    assert MIN_SNR_DB >= 12.0, "a marginal peak will be chased"
    marginal = carrier(DEFAULT_FFT * 2, 120.0, amplitude=1.2e-5, noise=1e-3)
    detected = measure_carrier(marginal, FS, min_snr_db=0.0)
    assert detected is not None
    if detected.snr_db < MIN_SNR_DB:
        assert measure_carrier(marginal, FS) is None


def test_a_strong_carrier_still_passes_the_default_threshold():
    assert measure_carrier(carrier(DEFAULT_FFT * 2, 120.0, amplitude=0.3), FS) is not None
