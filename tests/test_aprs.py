"""APRS / AX.25 decoding.

Checked against published values as well as the test encoder below: the CRC's standard
check value, and the APRS specification's own position examples.
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp.aprs import (
    AprsDecoder, compressed_position, crc_x25, mic_e_position, parse_ax25, plain_position,
    unstuff,
)
from src.rgc_sdr.dsp.demod import DemodChain

# -- the published references ------------------------------------------------------


def test_crc_matches_the_standard_check_value():
    assert crc_x25(b"123456789") == 0x906E           # CRC-16/X-25


def test_plain_position_from_the_spec():
    lat, lon = plain_position("4903.50N/07201.75W-")
    assert lat == pytest.approx(49 + 3.5 / 60) and lon == pytest.approx(-(72 + 1.75 / 60))


def test_compressed_position_from_the_spec():
    lat, lon = compressed_position("/5L!!<*e7>7P[")
    assert lat == pytest.approx(49.5, abs=1e-4) and lon == pytest.approx(-72.75, abs=1e-4)


def test_mic_e_latitude_from_the_spec():
    """APRS 1.0.1 chapter 10: destination S32U6T is 33 25.64 N, longitude West."""
    info = bytes([0x60, 112 + 28, 0 + 28, 0 + 28])      # 112 degrees, no offset
    lat, lon = mic_e_position("S32U6T", info)
    assert lat == pytest.approx(33 + 25.64 / 60)
    assert lon < 0


def test_mic_e_in_melbourne():
    """South and East with the +100 degree longitude offset: 37 49.10 S, 144 58.00 E."""
    # Digits 3 7 4 9 1 0; 4th char a digit (South), 5th P-Z (+100), 6th a digit (East).
    dest = "3749Q0"
    info = bytes([0x60, 44 + 28, 58 + 28, 0 + 28])
    lat, lon = mic_e_position(dest, info)
    assert lat == pytest.approx(-(37 + 49.10 / 60))
    assert lon == pytest.approx(144 + 58.0 / 60)


def test_unstuffing_removes_the_zero_after_five_ones():
    bits = np.array([1, 1, 1, 1, 1, 0, 1, 0], dtype=np.uint8)
    assert list(unstuff(bits)) == [1, 1, 1, 1, 1, 1, 0]
    assert unstuff(np.ones(7, dtype=np.uint8)) is None      # an abort, not data


# -- an encoder, for tests only ----------------------------------------------------


def _addr(call, last=False, repeated=False):
    name, _, ssid = call.partition("-")
    raw = bytes((ord(c) << 1) for c in name.ljust(6))
    return raw + bytes([0x60 | (int(ssid or 0) << 1) | (0x80 if repeated else 0) | int(last)])


def ax25(source, dest, path, info):
    calls = [dest, source, *path]
    frame = b"".join(_addr(c.rstrip("*"), last=(i == len(calls) - 1), repeated=c.endswith("*"))
                     for i, c in enumerate(calls))
    frame += b"\x03\xf0" + info
    crc = crc_x25(frame)
    return frame + bytes([crc & 0xFF, crc >> 8])


def hdlc_levels(frame, flags=40):
    bits = np.unpackbits(np.frombuffer(frame, np.uint8), bitorder="little")
    stuffed, ones = [], 0
    for b in bits:
        stuffed.append(int(b))
        ones = ones + 1 if b else 0
        if ones == 5:
            stuffed.append(0)
            ones = 0
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    data = flag * flags + stuffed + flag * 4
    level, out = 0, []
    for b in data:                        # NRZI: a 0 changes the tone
        if b == 0:
            level ^= 1
        out.append(level)
    return np.array(out)


def afsk_iq(levels, rate=48_000.0, deviation=3000.0, emphasis=1.0, noise=0.0, seed=2):
    """Bell 202 tones, phase continuous, FM-modulated onto a carrier. `emphasis` scales
    the space tone, as a transmitter's pre-emphasis does."""
    sps = rate / 1200
    idx = np.minimum((np.arange(int(levels.size * sps)) / sps).astype(int), levels.size - 1)
    space = levels[idx] == 1
    tone = np.where(space, 2200.0, 1200.0)
    audio = np.sin(2 * np.pi * np.cumsum(tone) / rate) * np.where(space, emphasis, 1.0)
    audio = np.concatenate([np.zeros(int(rate * 0.2)), audio, np.zeros(int(rate * 0.3))])
    iq = np.exp(1j * 2 * np.pi * np.cumsum(audio * deviation) / rate)
    if noise:
        rng = np.random.default_rng(seed)
        iq = iq + noise * (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size))
    return iq.astype(np.complex64)


def decode_iq(iq, rate=48_000.0):
    chain = DemodChain(rate, "nbfm", bandwidth_hz=12.5e3)
    dec = AprsDecoder(chain.if_rate)
    out = []
    for block in np.array_split(iq, max(1, iq.size // 2048)):
        chain.process(block)
        out += dec.process(chain.last_discriminator)
    return out


# -- end to end ----------------------------------------------------------------------


def test_decodes_a_position_report():
    frame = ax25("VK3RQ-9", "APRS", ["WIDE1-1", "WIDE2-1"], b"!3749.10S/14458.00E>Melbourne")
    (p,) = decode_iq(afsk_iq(hdlc_levels(frame)))
    assert (p.source, p.dest, p.path) == ("VK3RQ-9", "APRS", ["WIDE1-1", "WIDE2-1"])
    assert p.text == "!3749.10S/14458.00E>Melbourne"
    assert p.position == pytest.approx((-(37 + 49.1 / 60), 144 + 58 / 60))
    assert p.summary().split("  ", 1)[1].startswith("VK3RQ-9>APRS,WIDE1-1,WIDE2-1:!3749")


def test_digipeated_path_is_marked():
    frame = ax25("VK3ABC", "APRS", ["VK3RMD*", "WIDE2-1"], b">status text")
    (p,) = decode_iq(afsk_iq(hdlc_levels(frame)))
    assert p.path == ["VK3RMD*", "WIDE2-1"]


@pytest.mark.parametrize("emphasis", [0.5, 2.0])
def test_tone_level_tilt_does_not_matter(emphasis):
    """Pre- or de-emphasis makes one tone louder; slicing on frequency ignores that."""
    frame = ax25("VK3RQ", "APRS", [], b">tilted")
    assert [p.text for p in decode_iq(afsk_iq(hdlc_levels(frame), emphasis=emphasis))] == [">tilted"]


def test_survives_noise():
    frame = ax25("VK3RQ", "APRS", ["WIDE1-1"], b"!3749.10S/14458.00E-noisy")
    assert len(decode_iq(afsk_iq(hdlc_levels(frame), noise=0.3))) == 1


def test_a_corrupted_frame_is_rejected():
    frame = bytearray(ax25("VK3RQ", "APRS", [], b">will be damaged"))
    frame[20] ^= 0x10
    assert decode_iq(afsk_iq(hdlc_levels(bytes(frame)))) == []


def test_noise_alone_decodes_nothing():
    rng = np.random.default_rng(9)
    iq = 0.1 * (rng.standard_normal(48_000 * 3) + 1j * rng.standard_normal(48_000 * 3))
    assert decode_iq(iq.astype(np.complex64)) == []


def test_parse_rejects_a_truncated_address_field():
    assert parse_ax25(b"\x82" * 10) is None
