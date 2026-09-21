"""Device-layer tests. Ring buffer and capability logic are pure; streaming needs hardware."""

import numpy as np
import pytest

from src.rgc_sdr.device.source import (
    DeviceCaps,
    FreqRange,
    GainElement,
    SoapyIQSource,
    _Ring,
    enumerate_devices,
)

AIRSPY_RATES = (912e3, 768e3, 650e3, 456e3, 384e3, 228e3, 192e3)


# -- ring buffer -------------------------------------------------------------

def test_ring_returns_newest_in_chronological_order():
    r = _Ring(10)
    r.write(np.arange(6, dtype=np.complex64))
    assert np.array_equal(r.read_latest(3).real, [3, 4, 5])


def test_ring_short_read_while_filling():
    r = _Ring(10)
    r.write(np.arange(2, dtype=np.complex64))
    out = r.read_latest(5)
    assert out.size == 2 and np.array_equal(out.real, [0, 1])


def test_ring_empty_read():
    assert _Ring(10).read_latest(4).size == 0


def test_ring_wraparound_keeps_order():
    r = _Ring(8)
    for start in range(0, 20, 5):
        r.write(np.arange(start, start + 5, dtype=np.complex64))
    # 20 samples written into an 8-slot ring: newest 8 are 12..19
    assert np.array_equal(r.read_latest(8).real, np.arange(12, 20))


def test_ring_block_larger_than_capacity_keeps_newest():
    r = _Ring(4)
    r.write(np.arange(100, dtype=np.complex64))
    assert np.array_equal(r.read_latest(4).real, [96, 97, 98, 99])


def test_ring_never_exceeds_capacity():
    r = _Ring(5)
    r.write(np.arange(50, dtype=np.complex64))
    assert len(r) == 5
    assert r.read_latest(999).size == 5


def test_ring_read_is_a_copy():
    r = _Ring(8)
    r.write(np.arange(8, dtype=np.complex64))
    out = r.read_latest(4)
    out[:] = 0
    assert r.read_latest(4).real[0] == 4.0


def test_ring_rejects_bad_capacity():
    with pytest.raises(ValueError):
        _Ring(0)


# -- capabilities ------------------------------------------------------------

def _caps(**kw):
    base = dict(
        driver="airspyhf",
        label="AirSpy HF+",
        serial="x",
        sample_rates=AIRSPY_RATES,
        freq_ranges=(FreqRange(9e3, 31e6), FreqRange(60e6, 260e6)),
        gain_elements=(),
        has_agc=True,
        formats=("CF32",),
    )
    base.update(kw)
    return DeviceCaps(**base)


def test_default_sample_rate_prefers_768k():
    assert _caps().default_sample_rate() == 768e3


def test_default_sample_rate_falls_back_below_preference():
    caps = _caps(sample_rates=(192e3, 384e3))
    assert caps.default_sample_rate(768e3) == 384e3


def test_default_sample_rate_when_all_rates_exceed_preference():
    caps = _caps(sample_rates=(2e6, 4e6))
    assert caps.default_sample_rate(768e3) == 2e6


def test_covers_the_airspy_gap_correctly():
    caps = _caps()
    assert caps.covers(7.1e6)
    assert caps.covers(100e6)
    assert not caps.covers(45e6)  # the 31-60 MHz hole
    assert not caps.covers(300e6)


def test_airspyhf_reports_no_gain_elements():
    """Documents the measured reality that P2 gain UI must handle (PLANNING.md section 3)."""
    assert _caps().gain_elements == ()
    assert _caps().has_agc is True


def test_gain_elements_supported_for_other_radios():
    caps = _caps(driver="hackrf", gain_elements=(GainElement("LNA", 0.0, 40.0, 8.0),))
    assert caps.gain_elements[0].name == "LNA"


def test_describe_ranges():
    assert _caps().describe_ranges() == "0.009-31 MHz, 60-260 MHz"


# -- hardware ----------------------------------------------------------------

def test_enumerate_does_not_raise():
    assert isinstance(enumerate_devices(), list)


@pytest.mark.hardware
def test_probed_caps_match_measured_airspy(sdr_devices):
    if not any(d.get("driver") == "airspyhf" for d in sdr_devices):
        pytest.skip("no airspyhf")
    src = SoapyIQSource(driver="airspyhf", center_freq=7.1e6)
    caps = src.caps
    assert set(caps.sample_rates) == set(AIRSPY_RATES)
    assert caps.gain_elements == ()
    assert caps.has_agc is True
    assert "CF32" in caps.formats
    assert caps.covers(7.1e6) and not caps.covers(45e6)


@pytest.mark.hardware
def test_streams_real_iq(sdr_devices):
    if not any(d.get("driver") == "airspyhf" for d in sdr_devices):
        pytest.skip("no airspyhf")
    import time

    from src.rgc_sdr.dsp.spectrum import SpectrumAnalyzer

    sa = SpectrumAnalyzer(fft_size=4096)
    with SoapyIQSource(driver="airspyhf", sample_rate=768e3, center_freq=7.1e6) as src:
        assert src.sample_rate == 768e3
        deadline = time.time() + 5.0
        iq = src.read_latest(sa.samples_wanted())
        while iq.size < sa.samples_wanted() and time.time() < deadline:
            time.sleep(0.05)
            iq = src.read_latest(sa.samples_wanted())
        assert iq.size == sa.samples_wanted(), "device did not deliver samples"
        assert np.all(np.isfinite(iq)), "uninitialised buffer tail leaked into the ring"
        db = sa.psd_dbfs(iq)
        assert -200.0 < db.min() and db.max() < 0.1
        assert src.stats["samples"] > 0
    assert src.stats["errors"] == 0
