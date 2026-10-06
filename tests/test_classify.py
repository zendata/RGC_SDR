"""The "?" button: what is this signal?

Protocols must be named by a decoder that found error-checked frames; modulations by
their shape. Signals are synthesised with the same generators the decoder tests use,
placed off the centre of the stream as the listening offset would put them.
"""

import numpy as np
import pytest

from src.rgc_sdr.classify import classify, find_signal
from tests.test_adsb import IDENT, ODD, ppm
from tests.test_dmr import HEADER, PREAMBLE, burst
from tests.test_fsk4 import fsk4_wave
from tests.test_p25 import CONTROL
from tests.test_pocsag import _alpha_chunks, encode, fsk_iq

RATE = 192e3
CENTRE = 160e6
OFFSET = 25e3                    # the listening offset: the signal sits here
LISTEN = CENTRE + OFFSET


def place(baseband, rate=RATE, offset=OFFSET, noise=0.01, seconds=None, seed=3):
    """A baseband signal moved `offset` from the centre, in a little noise, padded
    with quiet to `seconds`."""
    x = np.asarray(baseband, dtype=np.complex128)
    if seconds is not None and x.size < seconds * rate:
        x = np.concatenate([x, np.zeros(int(seconds * rate) - x.size)])
    n = np.arange(x.size)
    rng = np.random.default_rng(seed)
    hiss = noise * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    return (x * np.exp(2j * np.pi * offset * n / rate) + hiss).astype(np.complex64)


def fm(deviation_hz, rate=RATE):
    return np.exp(2j * np.pi * np.cumsum(deviation_hz) / rate)


def tone(hz, seconds, rate=RATE):
    return np.sin(2 * np.pi * hz * np.arange(int(seconds * rate)) / rate)


def test_p25_is_named_by_its_frames():
    dev = fsk4_wave(CONTROL * 3, rate=RATE, unit=600.0, dc=0.0, noise=0.0)
    found = classify(place(fm(dev), seconds=1.0), RATE, CENTRE, LISTEN)
    assert (found.label, found.certain) == ("P25", True)
    assert found.detail == "NAC 2A7, control channel"
    assert found.freq_hz == LISTEN
    assert found.text() == "160.0250 MHz  P25 (NAC 2A7, control channel)"


def test_dmr_is_named_with_its_colour_code():
    bursts = (burst(1, PREAMBLE) + burst(2, HEADER, data_type=1)) * 4
    dev = fsk4_wave(bursts, rate=RATE, unit=648.0, dc=0.0, noise=0.0)
    found = classify(place(fm(dev), seconds=1.0), RATE, CENTRE, LISTEN)
    assert (found.label, found.detail, found.certain) == ("DMR", "CC 7", True)


def test_pocsag_is_named_with_its_rate_and_never_its_text():
    bits = encode(4321, 3, _alpha_chunks("CLASSIFY ME"))
    iq = fsk_iq(np.concatenate([bits, bits]), 1200, rate=RATE)
    found = classify(place(iq, seconds=2.0), RATE, CENTRE, LISTEN)
    assert (found.label, found.detail) == ("POCSAG", "1200 baud")
    assert "CLASSIFY" not in found.text()


def test_adsb_is_named_at_1090():
    iq = ppm([IDENT, ODD] * 5, rate=2e6)
    found = classify(iq, 2e6, 1090e6, 1090e6)
    assert (found.label, found.detail) == ("ADS-B", "2 aircraft")   # KLM1023 and 40621D


def test_nbfm_voice_is_a_judgement():
    found = classify(place(fm(2500.0 * tone(800.0, 1.0))), RATE, CENTRE, LISTEN)
    assert (found.label, found.certain) == ("NBFM", False)
    assert "2.5 kHz" in found.detail
    assert found.text().startswith("160.0250 MHz  NBFM?")


def test_fsk_data_that_no_decoder_knows():
    bits = np.random.default_rng(4).integers(0, 2, 2400)
    found = classify(place(fsk_iq(bits, 2400, rate=RATE, deviation=3000.0)), RATE, CENTRE,
                     LISTEN)
    assert found.label == "FSK data"


def test_am_has_its_carrier():
    found = classify(place(1.0 + 0.5 * tone(1000.0, 1.0)), RATE, CENTRE, LISTEN)
    assert found.label == "AM"


def test_lightly_modulated_am_is_still_am():
    """An ATIS at 119.8 MHz has a steady enough envelope to pass for FM by that alone."""
    voice = 0.25 * tone(700.0, 1.0) + 0.1 * tone(1700.0, 1.0)
    found = classify(place(1.0 + voice, noise=0.02), RATE, CENTRE, LISTEN)
    assert found.label == "AM"


@pytest.mark.parametrize("side", ["USB", "LSB"])
def test_ssb_is_told_by_which_side_of_the_tuned_frequency_it_is(side):
    sign = 1 if side == "USB" else -1
    t = np.arange(int(RATE)) / RATE
    two_tone = np.exp(2j * np.pi * sign * 700 * t) + np.exp(2j * np.pi * sign * 1900 * t)
    found = classify(place(two_tone), RATE, CENTRE, LISTEN)
    assert found.label == side


def test_a_plain_carrier():
    found = classify(place(np.ones(int(RATE)), offset=OFFSET + 300), RATE, CENTRE, LISTEN)
    assert found.label == "Carrier"


def test_wideband_fm_with_a_stereo_pilot():
    rate = 1e6
    t = np.arange(int(rate * 1.0)) / rate
    mpx = 0.8 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 19e3 * t)
    iq = place(fm(75e3 * mpx, rate), rate=rate, offset=100e3, noise=0.001)
    found = classify(iq, rate, 100.0e6, 100.1e6)
    assert found.label == "WBFM" and "stereo" in found.detail


def test_noise_alone_is_no_signal():
    noise = place(np.zeros(int(RATE)), noise=0.1)
    assert classify(noise, RATE, CENTRE, LISTEN) is None


def test_a_strong_neighbour_does_not_widen_the_signal():
    """Measured at 131.55 MHz: a stronger signal 20-90 kHz above made ACARS look
    250 kHz wide until the width stopped at the valley between them."""
    t = np.arange(int(RATE)) / RATE
    near = 0.3 * np.exp(2j * np.pi * 300 * t)
    neighbour = 3.0 * fm(5000 * tone(500.0, 1.0)) * np.exp(2j * np.pi * 30e3 * t)
    _, width, _ = find_signal(place(near + neighbour, offset=0.0), RATE, 0.0)
    assert width < 10e3
