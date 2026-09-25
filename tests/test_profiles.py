"""SDR profiles and availability. No hardware needed."""

import pytest

from src.rgc_sdr.device.profiles import (
    APP_MAX_RATE,
    PROFILES,
    availability,
    caps_from_profile,
    profile_for,
    refine_caps,
    starting_frequency,
)
from src.rgc_sdr.device.source import DeviceCaps, FreqRange


def test_the_four_requested_radios_are_supported():
    labels = " ".join(p.label for p in PROFILES).lower()
    for name in ("pluto", "hackrf", "airspy", "rtl"):
        assert name in labels, f"{name} missing"


def test_both_airspy_families_are_distinct():
    """The HF+ and the R2/Mini are different radios with different drivers."""
    hf = profile_for("airspyhf")
    r2 = profile_for("airspy")
    assert hf is not None and r2 is not None and hf is not r2
    assert hf.module != r2.module


def test_keys_drivers_and_modules_are_unique():
    for attr in ("key", "driver", "module", "label"):
        values = [getattr(p, attr) for p in PROFILES]
        assert len(values) == len(set(values)), f"duplicate {attr}"


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
def test_profile_defaults_are_self_consistent(profile):
    assert profile.default_rate in profile.sample_rates
    assert profile.default_rate <= profile.max_rate <= APP_MAX_RATE
    assert profile.covers(profile.default_freq), "starts outside its own coverage"
    assert profile.install
    for gain in profile.gain_elements:
        assert gain.max_db > gain.min_db
    stages = {g.name: g for g in profile.gain_elements}
    for name, db in profile.default_gains:
        assert name in stages, f"default for unknown gain {name}"
        g = stages[name]
        assert g.min_db <= db <= g.max_db
        assert (db - g.min_db) % g.step_db == 0, f"{name} {db} dB is off its step"


def test_no_rate_above_the_dsp_ceiling_is_the_default():
    """Measured: the NumPy audio chain needs 74% of a core at 20 MS/s."""
    assert all(p.default_rate <= 6e6 for p in PROFILES)


def test_profile_lookup_by_key_or_driver():
    assert profile_for("hackrf").label == "HackRF One"
    assert profile_for("plutosdr").label == "ADALM-Pluto"
    assert profile_for("no-such-radio") is None


def test_availability_distinguishes_connected_installed_and_missing():
    modules = {"airspyhfSupport", "HackRFSupport"}
    devices = [{"driver": "airspyhf", "serial": "abc"}]
    by_key = {a.profile.key: a for a in availability(devices, modules)}
    assert by_key["airspyhf"].status == "connected"
    assert by_key["airspyhf"].serial == "abc"
    assert by_key["hackrf"].status == "not connected"
    assert by_key["rtlsdr"].status == "driver not installed"
    assert by_key["plutosdr"].status == "driver not installed"


def test_the_hf_plus_module_does_not_count_as_the_r2_driver():
    """"airspySupport" is a prefix of "airspyhfSupport" -- matching loosely would say
    the R2 driver is installed when only the HF+ one is."""
    by_key = {a.profile.key: a for a in availability([], {"airspyhfSupport"})}
    assert by_key["airspyhf"].installed
    assert not by_key["airspy"].installed


def test_every_profile_is_reported_even_with_nothing_present():
    assert len(availability([], set())) == len(PROFILES)


def test_refine_caps_drops_rates_the_dsp_cannot_sustain():
    probed = DeviceCaps("hackrf", "HackRF", "", (20e6, 10e6, 8e6, 2e6),
                        (FreqRange(1e6, 6e9),), (), False, ("CF32",))
    refined = refine_caps(probed, profile_for("hackrf"))
    assert max(refined.sample_rates) <= APP_MAX_RATE
    assert 20e6 not in refined.sample_rates


def test_refine_caps_supplies_rates_when_the_driver_reports_only_a_range():
    """Pluto reports a continuous range, so listSampleRates comes back empty."""
    probed = DeviceCaps("plutosdr", "Pluto", "", (), (FreqRange(325e6, 3.8e9),),
                        (), True, ("CF32",))
    refined = refine_caps(probed, profile_for("plutosdr"))
    assert refined.sample_rates
    assert profile_for("plutosdr").default_rate in refined.sample_rates


def test_probing_wins_over_the_profile():
    probed = DeviceCaps("rtlsdr", "RTL", "", (2.048e6,), (FreqRange(24e6, 1.7e9),),
                        (), True, ("CF32",))
    refined = refine_caps(probed, profile_for("rtlsdr"))
    assert refined.sample_rates == (2.048e6,)


def test_refine_caps_without_a_profile_is_identity():
    probed = DeviceCaps("mystery", "?", "", (1e6,), (), (), False, ("CF32",))
    assert refine_caps(probed, None) is probed


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
def test_caps_from_profile_is_usable(profile):
    caps = caps_from_profile(profile)
    assert caps.driver == profile.driver
    assert caps.sample_rates
    assert caps.covers(profile.default_freq)


def test_starting_frequency_keeps_what_the_new_radio_can_tune():
    assert starting_frequency(profile_for("hackrf"), 7.1e6) == pytest.approx(7.1e6)


def test_starting_frequency_moves_somewhere_useful_when_it_cannot():
    """7.1 MHz is below the RTL-SDR's range; the band edge would show nothing."""
    rtl = profile_for("rtlsdr")
    assert starting_frequency(rtl, 7.1e6) == pytest.approx(rtl.default_freq)
    assert rtl.covers(rtl.default_freq)
