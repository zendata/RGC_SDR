"""ADS-B decoding.

Reference messages and their values are from Junzi Sun, "The 1090 Megahertz Riddle"
(the standard worked examples), so the parity, CPR and velocity decoding are checked
against an outside source rather than only an encoder written here.
"""

from pathlib import Path

import numpy as np
import pytest

from src.rgc_sdr.decoding import DecodeWorker
from src.rgc_sdr.device.source import SequentialReader, _Ring
from src.rgc_sdr.dsp.adsb import (
    AdsbDecoder, altitude_ft, bits_of_hex, cpr_local, crc, decode_extended,
)

IDENT = "8D4840D6202CC371C32CE0576098"          # KLM1023
EVEN = "8D40621D58C382D690C8AC2863A7"           # with ODD: 52.2572 N, 3.91937 E, 38000 ft
ODD = "8D40621D58C386435CC412692AD6"
VELOCITY = "8D485020994409940838175B284F"       # 159.20 kt, track 182.88, -832 ft/min
AIRSPEED = "8DA05F219B06B6AF189400CBC33F"       # heading 243.98, TAS 375 kt, -2304 ft/min


def test_published_messages_pass_the_parity_check():
    for h in (IDENT, EVEN, ODD, VELOCITY, AIRSPEED):
        assert crc(bits_of_hex(h)) == 0
    damaged = bits_of_hex(IDENT)
    damaged[40] ^= 1
    assert crc(damaged) != 0


def test_identification():
    m = decode_extended(bits_of_hex(IDENT), {}, 0.0)
    assert (m.icao, m.tc, m.fields["callsign"]) == ("4840D6", 4, "KLM1023")


def test_position_from_an_even_odd_pair():
    state = {}
    first = decode_extended(bits_of_hex(ODD), state, 1457996400.0)
    assert first.position is None and first.fields["altitude_ft"] == 38000
    second = decode_extended(bits_of_hex(EVEN), state, 1457996402.0)
    assert second.position == pytest.approx((52.25720, 3.91937), abs=1e-5)


def test_local_decoding_from_a_known_position():
    lat17, lon17 = 93000, 51372                          # the reference's even frame
    assert cpr_local((lat17, lon17), False, (52.258, 3.918)) == pytest.approx(
        (52.25720, 3.91937), abs=1e-4)


def test_ground_speed_and_track():
    f = decode_extended(bits_of_hex(VELOCITY), {}, 0.0).fields
    assert f["speed_kn"] == pytest.approx(159.20, abs=0.01)
    assert f["track"] == pytest.approx(182.88, abs=0.01)
    assert f["vertical_fpm"] == -832


def test_airspeed_and_heading():
    f = decode_extended(bits_of_hex(AIRSPEED), {}, 0.0).fields
    assert (f["speed_kn"], f["airspeed"], f["vertical_fpm"]) == (375, "TAS", -2304)
    assert f["heading"] == pytest.approx(243.98, abs=0.01)


def test_altitude_coding():
    assert altitude_ft(0b110000111000) == 38000
    assert altitude_ft(0) is None


# -- over the air: pulses at 2 MS/s -------------------------------------------------

RATE = 2e6


def ppm(hexes, rate=RATE, gap_us=300, noise=0.05, offset_hz=0.0, seed=8):
    """Messages as 1090 MHz pulses: preamble, then a pulse in the first or second half
    of each microsecond; `offset_hz` from the centre of the stream."""
    sps = int(rate / 1e6)
    out = [np.zeros(int(gap_us * sps))]
    for h in hexes:
        chips = np.zeros(8 * sps)                        # the 8 us preamble
        for t in (0.0, 1.0, 3.5, 4.5):
            chips[int(t * sps):int(t * sps) + sps // 2] = 1.0
        body = []
        for b in bits_of_hex(h):
            half = np.zeros(sps)
            half[: sps // 2] = 1.0 if b else 0.0
            half[sps // 2:] = 0.0 if b else 1.0
            body.append(half)
        out += [chips, np.concatenate(body), np.zeros(int(gap_us * sps))]
    amp = np.concatenate(out)
    rng = np.random.default_rng(seed)
    iq = amp * np.exp(2j * np.pi * offset_hz * np.arange(amp.size) / rate)
    iq = iq + noise * (rng.standard_normal(amp.size) + 1j * rng.standard_normal(amp.size))
    return iq.astype(np.complex64)


def run(decoder, iq, block=3001):
    found = []
    for i in range(0, iq.size, block):                    # boundaries fall mid-message
        found += decoder.process(iq[i:i + block])
    return found


def test_decodes_pulses_across_block_boundaries():
    found = run(AdsbDecoder(RATE), ppm([IDENT, ODD, EVEN, VELOCITY]))
    assert [m.hex for m in found] == [IDENT, ODD, EVEN, VELOCITY]
    assert found[2].position == pytest.approx((52.2572, 3.91937), abs=1e-4)


def test_decodes_real_pulses_off_air():
    # 1 ms of the Pluto at 2 MS/s on 1090 MHz, Melbourne, 2026-10-06: real pulse shapes,
    # noise and timing, not synthesised ones.
    iq = np.fromfile(Path(__file__).parent / "data" / "adsb_1090_2msps.cf32",
                     dtype=np.complex64)
    found = run(AdsbDecoder(RATE), iq, block=700)
    assert [(m.icao, m.fields["callsign"], m.hex) for m in found] == [
        ("7C72AC", "WXQ", "8D7C72AC215D84608208200797A5")]


def test_noise_alone_decodes_nothing():
    rng = np.random.default_rng(2)
    iq = (0.3 * (rng.standard_normal(400_000) + 1j * rng.standard_normal(400_000)))
    assert run(AdsbDecoder(RATE), iq.astype(np.complex64)) == []


def test_needs_a_whole_number_of_samples_per_microsecond():
    with pytest.raises(ValueError):
        AdsbDecoder(768e3)
    with pytest.raises(ValueError):
        AdsbDecoder(2.5e6)
    assert AdsbDecoder(4e6).sps == 4


class RingSource:
    def __init__(self, rate=RATE, centre=1090e6):
        self.sample_rate, self.center_freq = rate, centre
        self.ring = _Ring(2_000_000)

    def sequential_reader(self):
        return SequentialReader(self.ring)


def test_worker_follows_1090_when_tuned_off_it():
    worker = DecodeWorker(RingSource(centre=1089.8e6), "adsb")
    assert worker.problem == "" and worker.channels_in_view()[0][2]
    found = []
    iq = ppm([IDENT], offset_hz=200e3)
    for i in range(0, iq.size, 4096):
        found += worker.process(iq[i:i + 4096])
    assert [m.fields.get("callsign") for m in found] == ["KLM1023"]


def test_worker_explains_an_unusable_rate():
    worker = DecodeWorker(RingSource(rate=768e3), "adsb")
    assert "samples per microsecond" in worker.problem
    assert worker.process(np.zeros(1000, np.complex64)) == []
