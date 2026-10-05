"""ACARS decoding.

The bit coding and the block check were settled on air (see dsp/acars.py); here the CRC
is also checked against its standard check value, and a test encoder exercises the
whole path from an AM carrier.
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp.acars import (
    ETX, NAK, SOH, STX, SYN, AcarsDecoder, crc_kermit, parse_block, position_in,
)
from src.rgc_sdr.dsp.demod import DemodChain


def test_crc_matches_the_standard_check_value():
    assert crc_kermit(b"123456789") == 0x2189            # CRC-16/KERMIT


def odd(c):
    """A 7-bit character with its odd parity bit."""
    return c | (0x80 if bin(c).count("1") % 2 == 0 else 0)


def block(registration, label, block_id, text, mode="2", ack=NAK, end=ETX):
    """Characters after SYN SYN SOH, as transmitted, with the block check."""
    header = [ord(mode)] + [ord(c) for c in registration.rjust(7, ".")] + [ack] \
        + [ord(c) for c in label] + [ord(block_id), STX] + [ord(c) for c in text] + [end]
    chars = [odd(c) for c in header]
    crc = crc_kermit(bytes(chars))
    return chars + [crc & 0xFF, crc >> 8, 0x7F]


def msk_am_iq(chars, rate=48_000.0, depth=0.5, noise=0.0, seed=4):
    """A whole transmission: pre-key ones, '+', '*', SYN SYN SOH, the block; MSK with
    2400 Hz for "same as the previous bit", 1200 Hz for "different"; AM on a carrier."""
    pre = [0xFF] * 16 + [odd(ord("+")), odd(ord("*")), SYN, SYN, SOH]
    bits = np.unpackbits(np.array(pre + chars, dtype=np.uint8), bitorder="little")
    same = np.r_[True, bits[1:] == bits[:-1]]
    sps = rate / 2400
    idx = np.minimum((np.arange(int(bits.size * sps)) / sps).astype(int), bits.size - 1)
    freq = np.where(same[idx], 2400.0, 1200.0)
    audio = np.sin(2 * np.pi * np.cumsum(freq) / rate)
    gap = np.zeros(int(rate * 0.2))
    audio = np.concatenate([gap, audio, gap])
    carrier = np.r_[np.zeros(gap.size // 2), np.ones(audio.size - gap.size), np.zeros(gap.size - gap.size // 2)]
    iq = (carrier * (1 + depth * audio)).astype(np.complex64)
    if noise:
        rng = np.random.default_rng(seed)
        iq = iq + noise * (rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size))
    return iq.astype(np.complex64)


def decode(iq, rate=48_000.0):
    chain = DemodChain(rate, "am", bandwidth_hz=10e3, agc=False)
    dec = AcarsDecoder(chain.if_rate)
    out = []
    for b in np.array_split(iq, max(1, iq.size // 2400)):
        chain.process(b)
        out += dec.process(chain.last_detected)
    return out, dec


def test_parses_a_downlink_with_a_position():
    m = parse_block(block("VH-ABC", "3L", "7", "M93AJQ0737S 37.894/E144.735 /UTC 0416"))
    assert (m.registration, m.label, m.block, m.msg_no, m.flight) == (
        "VH-ABC", "3L", "7", "M93A", "JQ0737")
    assert m.downlink and m.ack == "NAK"
    assert m.position == pytest.approx((-37.894, 144.735))


def test_ground_station_squitter():
    m = parse_block(block("", "SQ", "\x00", "02XSMELYMML03741S14451EV136975/", ack=NAK))
    assert m.ground_station == "YMML" and not m.downlink
    assert m.position == pytest.approx((-(37 + 41 / 60), 144 + 51 / 60))


def test_a_bad_block_check_is_rejected():
    chars = block("VH-ABC", "H1", "4", "HELLO")
    chars[15] ^= 0x01
    assert parse_block(chars) is None


def test_positions_need_both_halves():
    assert position_in("ALT 37000", "H1") is None
    assert position_in("N 51.47/W000.45", "15") == pytest.approx((51.47, -0.45))


@pytest.mark.parametrize("noise", [0.0, 0.05])
def test_decodes_from_an_am_carrier(noise):
    chars = block("VH-ABC", "5Z", "3", "M01AQF0001HELLO FROM THE TEST")
    msgs, dec = decode(msk_am_iq(chars, noise=noise))
    assert len(msgs) == 1 and dec.bad_blocks == 0
    m = msgs[0]
    assert (m.registration, m.flight, m.text) == ("VH-ABC", "QF0001", "HELLO FROM THE TEST")


def test_two_blocks_in_one_stream_and_an_uplink():
    a = block("VH-ABC", "H1", "1", "M02AQF0002FIRST")
    b = block("VH-DEF", "_\x7f", "A", "", ack=ord("1"))
    msgs, _ = decode(np.concatenate([msk_am_iq(a), msk_am_iq(b)]))
    assert [(m.registration, m.downlink) for m in msgs] == [("VH-ABC", True), ("VH-DEF", False)]


def test_noise_alone_decodes_nothing():
    rng = np.random.default_rng(1)
    iq = (1 + 0.3 * rng.standard_normal(48_000 * 3)).astype(np.complex64)
    msgs, _ = decode(iq)
    assert msgs == []


def test_acknowledgement_label_is_readable():
    m = parse_block(block("VH-DEF", "_\x7f", "A", "", ack=ord("1")))
    assert "label _DEL (acknowledgement)" in m.summary()
