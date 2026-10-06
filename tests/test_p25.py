"""P25 Phase 1 metadata: NID, trellis-coded TSBKs, channel plans and grants.

The BCH generator is checked against its published value and the CRC against a
published check value (tests/test_fsk4.py); the frame layout comes from TIA-102 and was
confirmed on air (PLANNING.md 7p). All IDs here are made up.
"""

import numpy as np
import pytest

from src.rgc_sdr.decoding import DecodeWorker
from src.rgc_sdr.dsp.p25 import (
    BCH_GENERATOR, FRAME_SYNC, P25Decoder, bch_decode, bch_encode, trellis_decode,
    trellis_encode, tsbk_crc,
)
from tests.test_decoding import RingSource
from tests.test_fsk4 import RATE, fsk4_wave, run, word_dibits

NAC = 0x2A7


def bits_of(value: int, n: int) -> list[int]:
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def tsbk(opcode: int, args: list[tuple[int, int]], last: bool, mfid: int = 0) -> np.ndarray:
    bits = [int(last), 0] + bits_of(opcode, 6) + bits_of(mfid, 8)
    for value, n in args:
        bits += bits_of(value, n)
    assert len(bits) == 80
    bits += bits_of(tsbk_crc(np.array(bits)), 16)
    return np.array(bits)


def frame_dibits(duid: int, blocks: list[np.ndarray] = (), nac: int = NAC) -> list[int]:
    """Frame sync, NID and trellis-coded TSBKs, with a status symbol every 36 dibits."""
    nid = bits_of(bch_encode(nac << 4 | duid), 63) + [0]
    content = [2 * nid[i] + nid[i + 1] for i in range(0, 64, 2)]
    for block in blocks:
        data = [2 * block[i] + block[i + 1] for i in range(0, 96, 2)]
        content += list(trellis_encode(np.array(data)))
    out = word_dibits(FRAME_SYNC, 24)
    for d in content:
        if len(out) % 36 == 35:
            out.append(0b10)                                   # status symbol
        out.append(d)
    return out


IDEN = tsbk(0x34, [(3, 4), (4, 4), (0, 14), (50, 10), (int(420.0125e6 / 5), 32)], False)
NETWORK = tsbk(0x3B, [(0, 8), (0xABCDE, 20), (0x123, 12), (3 << 12 | 8, 16), (0, 8)], False)
GRANT = tsbk(0x00, [(0, 8), (3 << 12 | 8, 16), (100, 16), (2000, 24)], True)
#: What follows a frame on air: more frames, or noise.
FILLER = list(np.random.default_rng(9).integers(0, 4, 1800))
CONTROL = frame_dibits(0x7, [IDEN, NETWORK, GRANT]) + FILLER


def test_bch_generator_is_the_published_one():
    assert oct(BCH_GENERATOR) == "0o6331141367235453"


def test_nid_corrects_eleven_errors():
    rng = np.random.default_rng(5)
    word = bch_encode(NAC << 4 | 0x7)
    for _ in range(20):
        damaged = word
        for b in rng.choice(63, 11, replace=False):
            damaged ^= 1 << int(b)
        assert bch_decode(damaged) == (NAC << 4 | 0x7, 11)


def test_trellis_round_trip_corrects_errors():
    rng = np.random.default_rng(6)
    data = rng.integers(0, 4, 48)
    sent = trellis_encode(data)
    received = sent.copy()
    received[[5, 40, 77]] ^= 0b01
    decoded, errors = trellis_decode(received)
    assert list(decoded) == list(data) and errors == 3


def test_control_channel_plan_network_and_grant():
    found = run(P25Decoder(RATE), fsk4_wave(CONTROL))
    texts = [m.text for m in found]
    assert texts == [
        "channel plan (VHF/UHF)  3: from 420.01250 MHz every 6.25 kHz",
        "network status  WACN ABCDE  system 123  ch 3-8 (420.06250 MHz)",
        "group voice grant  TG 100  from 2000  ch 3-8 (420.06250 MHz)",
    ]
    assert all(m.nac == NAC for m in found)
    assert found[2].fields["group"] == 100 and found[2].fields["source"] == 2000


def test_repeats_are_held_back_and_bad_blocks_counted():
    decoder = P25Decoder(RATE)
    damaged = GRANT.copy()
    damaged[40] ^= 1                              # the CRC no longer matches
    stream = CONTROL + CONTROL + frame_dibits(0x7, [damaged]) + FILLER
    found = run(decoder, fsk4_wave(stream))
    assert len(found) == 3                        # the second copy says nothing new
    assert decoder.frames == 3 and decoder.bad_blocks == 1


def test_voice_and_inverted_signals():
    stream = frame_dibits(0x5) + FILLER
    found = run(P25Decoder(RATE), -fsk4_wave(stream))
    assert [(m.nac, m.kind) for m in found] == [(NAC, "voice")]


def test_noise_decodes_nothing():
    decoder = P25Decoder(RATE)
    assert run(decoder, 0.3 * np.random.default_rng(8).standard_normal(int(RATE * 5))) == []


def test_worker_decodes_from_iq():
    rate = 192e3
    dev = fsk4_wave(CONTROL, rate=rate, unit=600.0, dc=0.0, noise=0.0)
    iq = np.exp(2j * np.pi * np.cumsum(dev) / rate).astype(np.complex64)
    worker = DecodeWorker(RingSource(rate), "p25")
    found = []
    for block in np.array_split(iq, 30):
        found += worker.process(block)
    assert [m.fields.get("source") for m in found][-1] == 2000


# -- packet data ---------------------------------------------------------------------

from src.rgc_sdr.dsp.p25 import (  # noqa: E402
    block_crc9, packet_crc32, trellis34_decode, trellis34_encode,
)
from src.rgc_sdr.targets import TargetStore  # noqa: E402


def udp_packet(port: int, body: bytes) -> bytes:
    """SNDCP's 2-byte header, then IPv4/UDP to `port` (checksums not checked here)."""
    udp = (40000).to_bytes(2, "big") + port.to_bytes(2, "big") + \
        (8 + len(body)).to_bytes(2, "big") + b"\x00\x00" + body
    ip = bytes([0x45, 0]) + (20 + len(udp)).to_bytes(2, "big") + bytes(5) + bytes([17]) + \
        bytes(10)
    return b"\x51\x00" + ip + udp


def pdu_dibits(user: bytes, llid: int = 1234, outbound: bool = False) -> list[int]:
    """A confirmed data packet: header, 16-byte blocks with serial and CRC-9, CRC-32."""
    blocks = -(-(len(user) + 4) // 16)
    pad = blocks * 16 - len(user) - 4
    data = user + bytes(pad)
    data += packet_crc32(data).to_bytes(4, "big")
    head = [0, 1, int(outbound)] + bits_of(22, 5) + [0, 0] + bits_of(0, 6) + bits_of(0, 8)
    head += bits_of(llid, 24) + [1] + bits_of(blocks, 7) + [0, 0, 0] + bits_of(pad, 5)
    head += [0] * (80 - len(head))
    head += bits_of(tsbk_crc(np.array(head)), 16)
    coded = [np.array(head)]
    for k in range(blocks):
        body = list(np.unpackbits(np.frombuffer(data[16 * k:16 * k + 16], np.uint8)))
        serial = bits_of(k, 7)
        coded.append(np.array(serial + bits_of(block_crc9(np.array(serial + body)), 9) + body))
    out = word_dibits(FRAME_SYNC, 24)
    nid = bits_of(bch_encode(NAC << 4 | 0xC), 63) + [0]
    content = [2 * nid[i] + nid[i + 1] for i in range(0, 64, 2)]
    content += list(trellis_encode(np.array([2 * coded[0][i] + coded[0][i + 1]
                                             for i in range(0, 96, 2)])))
    for block in coded[1:]:
        content += list(trellis34_encode(block))
    for d in content:
        if len(out) % 36 == 35:
            out.append(0b10)
        out.append(d)
    return out + FILLER


def test_rate_three_quarter_trellis_corrects_errors():
    bits = np.random.default_rng(12).integers(0, 2, 144)
    sent = trellis34_encode(bits)
    # One symbol a level off. Rate 3/4 has less to spare than 1/2: two errors close
    # together after deinterleaving can defeat it, so this checks one.
    sent[60] ^= 0b01
    assert list(trellis34_decode(sent)) == list(bits)


def test_a_text_message_shows_its_text():
    text = "MEET AT THE GATE"
    body = b"\x00\x10\x00\x00" + text.encode("utf-16-le")
    found = run(P25Decoder(RATE), fsk4_wave(pdu_dibits(udp_packet(4007, body))))
    (m,) = found
    assert m.text == "packet data from 1234: text message (TMS)"
    assert text in m.summary() and text not in m.summary(show_text=False)


def test_a_location_report_puts_the_radio_on_the_map():
    lat, lon = -37.8136, 144.9631
    point = (b"\x66" + round(lat / 90 * 2 ** 31).to_bytes(4, "big", signed=True)
             + round(lon / 180 * 2 ** 31).to_bytes(4, "big", signed=True) + b"\x00\x10")
    lrrp = b"\x0d\x10\x22\x03\x00\x00\x01" + point
    (m,) = run(P25Decoder(RATE), fsk4_wave(pdu_dibits(udp_packet(4001, lrrp))))
    assert m.position == pytest.approx((lat, lon), abs=1e-5)
    assert "location report (LRRP)" in m.text and "-37.81360, 144.96310" in m.summary()
    store = TargetStore()
    assert store.update([m]) == 1
    (radio,) = store.placed()
    assert (radio.kind, radio.ident) == ("radio", "1234")


def test_a_damaged_block_is_rejected_not_shown():
    dibits = pdu_dibits(udp_packet(4007, "HELLO".encode("utf-16-le")))
    for i in range(57 + 98 + 10, 57 + 98 + 40):                  # wreck the first block
        if i % 36 != 35:
            dibits[i] ^= 0b11
    decoder = P25Decoder(RATE)
    assert run(decoder, fsk4_wave(dibits)) == [] and decoder.bad_blocks == 1
