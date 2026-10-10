"""The DMR/P25 survey (survey.py): every channel in one capture, decoded at once."""

import numpy as np
import pytest

from src.rgc_sdr.survey import find_channels, merge, survey_iq
from tests.test_dmr import DMR_UNIT, HEADER, PREAMBLE, burst
from tests.test_fsk4 import fsk4_wave
from tests.test_p25 import CONTROL

RATE = 1e6
CENTRE = 460e6


def _fm(dibits, unit, offset_hz, n, seed):
    """A four-level FSK channel at `offset_hz` in a capture of `n` samples at RATE."""
    wave = fsk4_wave(list(dibits) * 3, rate=RATE, unit=unit, dc=0.0, noise=0.0, lead=20000)
    wave = np.resize(wave, n)
    phase = 2 * np.pi * np.cumsum(wave * 2500.0 + offset_hz) / RATE
    return 0.1 * np.exp(1j * phase)


def _capture(seconds=1.0):
    n = int(RATE * seconds)
    dmr = burst(1, PREAMBLE) + burst(2, HEADER, data_type=1) + burst(1, None, voice=True) * 4
    rng = np.random.default_rng(0)
    noise = 0.002 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return (_fm(dmr, DMR_UNIT, 100e3, n, 1) + _fm(CONTROL, 600 / 2500, -200e3, n, 2)
            + noise).astype(np.complex64)


def test_both_channels_are_found_and_named():
    iq = _capture()
    found = find_channels(iq, RATE, CENTRE, CENTRE - 450e3, CENTRE + 450e3)
    freqs = sorted(f for f, _ in found)
    assert freqs == pytest.approx([CENTRE - 200e3, CENTRE + 100e3], abs=6.25e3)
    reports = {round(r["freq_hz"] - CENTRE, -3): r
               for r in survey_iq(iq, RATE, CENTRE, CENTRE - 450e3, CENTRE + 450e3)}
    dmr, p25 = reports[100e3], reports[-200e3]
    assert dmr["protocol"] == "DMR" and dmr["colour_codes"] == [7]
    assert dmr["voice"] > 0 and 1 in dmr["voice_slots"]
    assert p25["protocol"] == "P25" and p25["nacs"] == [0x2A7]


def test_noise_finds_nothing():
    rng = np.random.default_rng(1)
    iq = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)).astype(np.complex64)
    assert survey_iq(iq, RATE, CENTRE, CENTRE - 4e5, CENTRE + 4e5) == []


def test_passes_merge_into_one_table():
    table = {}
    merge(table, [{"freq_hz": 1.0, "protocol": "DMR", "colour_codes": [1], "nacs": [],
                   "voice": 0, "voice_slots": [], "encrypted": False, "snr_db": 30}], 10.0)
    merge(table, [{"freq_hz": 1.0, "protocol": "", "colour_codes": [], "nacs": [],
                   "voice": 12, "voice_slots": [2], "encrypted": False, "snr_db": 25}], 20.0)
    row = table[1.0]
    assert row["protocol"] == "DMR" and row["colour_codes"] == {1}
    assert row["voice"] == 12 and row["voice_slots"] == {2} and row["last_voice"] == 20.0
    assert row["heard"] == 2 and row["snr_db"] == 30


def test_a_channel_placed_a_grid_step_off_is_settled_by_its_decoder():
    """The decoders choose between grid points when the spectrum is not sure."""
    from src.rgc_sdr import survey as sv

    iq = _capture()
    real = sv.find_channels
    # As if the spectrum had put the DMR channel 6.25 kHz low.
    def off(*args, **kw):
        return [((f - 6250.0) if abs(f - CENTRE - 100e3) < 1e3 else f, s)
                for f, s in real(*args, **kw)]
    sv.find_channels = off
    try:
        reports = sv.survey_iq(iq, RATE, CENTRE, CENTRE - 450e3, CENTRE + 450e3)
    finally:
        sv.find_channels = real
    dmr = [r for r in reports if r["protocol"] == "DMR"]
    assert len(dmr) == 1 and dmr[0]["freq_hz"] == pytest.approx(CENTRE + 100e3)
