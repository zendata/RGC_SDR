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


# -- P2: tuning, clamping, rate selection ------------------------------------

def test_clamp_freq_snaps_into_the_31_to_60_mhz_gap():
    """The Airspy HF+ has two disjoint ranges, so a frequency can be *between* them."""
    caps = _caps()
    assert caps.clamp_freq(45e6) == 31e6   # nearer the top of the HF range
    assert caps.clamp_freq(55e6) == 60e6   # nearer the bottom of the VHF range
    assert caps.clamp_freq(7.1e6) == 7.1e6  # in range, untouched


def test_clamp_freq_handles_out_of_range_ends():
    caps = _caps()
    assert caps.clamp_freq(1e3) == 9e3
    assert caps.clamp_freq(500e6) == 260e6


def test_clamp_freq_without_known_ranges_is_identity():
    assert _caps(freq_ranges=()).clamp_freq(123e6) == 123e6


def test_nearest_sample_rate_snaps_to_a_supported_rate():
    caps = _caps()
    assert caps.nearest_sample_rate(700e3) == 650e3
    assert caps.nearest_sample_rate(900e3) == 912e3
    assert caps.nearest_sample_rate(1e9) == 912e3


def test_airspyhf_reports_no_bandwidth_options():
    """Measured 2026-09-23: listBandwidths() is empty, so no bandwidth UI is built."""
    assert _caps().bandwidths == ()


def test_ring_clear_discards_stale_samples():
    r = _Ring(16)
    r.write(np.arange(8, dtype=np.complex64))
    r.clear()
    assert len(r) == 0 and r.read_latest(8).size == 0


@pytest.mark.hardware
def test_retune_and_rate_change_on_live_stream(sdr_devices):
    if not any(d.get("driver") == "airspyhf" for d in sdr_devices):
        pytest.skip("no airspyhf")
    import time

    src = SoapyIQSource(driver="airspyhf", sample_rate=768e3, center_freq=7.1e6)
    with src:
        time.sleep(0.4)
        assert src.set_center_freq(5.0e6) == pytest.approx(5.0e6, abs=1.0)
        # The ring is dropped on retune so stale audio from the old QRG cannot render.
        assert src.read_latest(4096).size < 4096
        time.sleep(0.5)
        assert src.read_latest(4096).size == 4096

        assert src.set_sample_rate(456e3) == pytest.approx(456e3, abs=1.0)
        assert src.sample_rate == pytest.approx(456e3, abs=1.0)
        time.sleep(0.5)
        assert src.read_latest(4096).size == 4096
        assert src.stats["errors"] == 0


@pytest.mark.hardware
def test_clamped_retune_never_leaves_an_untunable_frequency(sdr_devices):
    if not any(d.get("driver") == "airspyhf" for d in sdr_devices):
        pytest.skip("no airspyhf")
    src = SoapyIQSource(driver="airspyhf", center_freq=7.1e6)
    assert src.set_center_freq(45e6) == pytest.approx(31e6, abs=1.0)
    assert src.caps.covers(src.center_freq)


@pytest.mark.hardware
def test_set_bandwidth_is_a_safe_noop_when_unsupported(sdr_devices):
    """setBandwidth() is silently accepted by this driver but does nothing; guard on caps."""
    if not any(d.get("driver") == "airspyhf" for d in sdr_devices):
        pytest.skip("no airspyhf")
    src = SoapyIQSource(driver="airspyhf", center_freq=7.1e6)
    assert src.caps.bandwidths == ()
    assert src.set_bandwidth(200e3) == 0.0  # unchanged, and must not raise


@pytest.mark.hardware
def test_settle_period_discards_samples_after_restart(sdr_devices):
    """The first frames after a stream restart read ~8 dB hot; they must not be served."""
    if not any(d.get("driver") == "airspyhf" for d in sdr_devices):
        pytest.skip("no airspyhf")
    import time

    src = SoapyIQSource(driver="airspyhf", sample_rate=768e3, center_freq=0.909e6,
                        settle_seconds=0.15)
    with src:
        time.sleep(0.8)
        after_start = src.stats["dropped"]
        assert after_start > 0, "nothing discarded at stream start"
        # ~0.15 s at 768 kS/s, allowing for the 2048-sample read granularity.
        assert 0.5 * 0.15 * 768e3 < after_start < 2.0 * 0.15 * 768e3

        src.set_sample_rate(456e3)
        time.sleep(0.8)
        assert src.stats["dropped"] > after_start, "rate change did not re-arm settling"
        assert src.stats["errors"] == 0


@pytest.mark.hardware
def test_settle_can_be_disabled(sdr_devices):
    if not any(d.get("driver") == "airspyhf" for d in sdr_devices):
        pytest.skip("no airspyhf")
    import time

    src = SoapyIQSource(driver="airspyhf", center_freq=7.1e6, settle_seconds=0.0)
    with src:
        time.sleep(0.5)
        assert src.stats["dropped"] == 0


@pytest.mark.hardware
def test_unflushed_retune_keeps_the_buffer_and_discards_nothing(sdr_devices):
    """Fine tuning must not cost a settle period: that starved audio, measured."""
    if not any(d.get("driver") == "airspyhf" for d in sdr_devices):
        pytest.skip("no airspyhf")
    import time

    src = SoapyIQSource(driver="airspyhf", sample_rate=768e3, center_freq=7.1e6)
    with src:
        time.sleep(0.8)
        before = src.stats["dropped"]
        reader = src.sequential_reader()
        time.sleep(0.3)

        for i in range(10):
            src.set_center_freq(7.1e6 + (i + 1) * 100.0, flush=False)
        time.sleep(0.3)

        assert src.stats["dropped"] == before, "a fine nudge armed the settle period"
        # The audio-side reader must see a continuous stream throughout.
        assert reader.available() > 0
        assert reader.lost == 0, "samples were lost across fine nudges"
        assert src.center_freq == pytest.approx(7.1e6 + 1000.0, abs=1.0)

        # A flushing retune still settles, as a band change should.
        src.set_center_freq(14.2e6, flush=True)
        time.sleep(0.4)
        assert src.stats["dropped"] > before
