"""DMR metadata: slot types, CACH slot numbers, BPTC, CSBKs and call headers.

Checked against properties that do not depend on an encoder written here -- Golay(20,8),
shortened from the extended Golay(24,12), has minimum distance 8; a Reed-Solomon codeword's syndromes are zero, the CRC's
published check value (tests/test_fsk4.py) -- and the layout was confirmed on air
(PLANNING.md 7p). All IDs here are made up.
"""

import numpy as np
import pytest

from src.rgc_sdr.decoding import DecodeWorker
from src.rgc_sdr.dsp.dmr import (
    LC_MASKS, SYNCS, TACT_POSITIONS, DmrDecoder, _EXP, _gmul, bptc_data, bptc_encode,
    csbk_crc, golay20_decode, golay20_encode, rs129_parity, tact_encode,
)
from tests.test_decoding import RingSource
from tests.test_fsk4 import RATE, fsk4_wave, run
from tests.test_p25 import bits_of

DMR_UNIT = 648 / 2500             # DMR steps 648 Hz per symbol level (+/-1944 Hz outer)
BS_VOICE = 0x755FD7DF75F7


def burst(slot: int, payload: np.ndarray | None, colour: int = 7, data_type: int = 3,
          voice: bool = False) -> list[int]:
    """One base-station burst, CACH included, as 144 dibits."""
    cach = np.zeros(24, dtype=int)
    cach[TACT_POSITIONS] = bits_of(tact_encode(0b1000 | (slot - 1) << 2), 7)
    if voice:
        info = np.random.default_rng(slot).integers(0, 2, 216)
        middle = bits_of(BS_VOICE, 48)
        bits = list(cach) + list(info[:108]) + middle + list(info[108:])
    else:
        info = bptc_encode(payload)
        slot_type = bits_of(golay20_encode(colour << 4 | data_type), 20)
        bits = (list(cach) + list(info[:98]) + slot_type[:10] + bits_of(SYNCS["bs"], 48)
                + slot_type[10:] + list(info[98:]))
    return [2 * bits[i] + bits[i + 1] for i in range(0, 288, 2)]


def csbk_payload(opcode: int, fid: int, body: list[tuple[int, int]]) -> np.ndarray:
    bits = [1, 0] + bits_of(opcode, 6) + bits_of(fid, 8)
    for value, n in body:
        bits += bits_of(value, n)
    bits += bits_of(csbk_crc(np.array(bits)), 16)
    return np.array(bits)


def lc_payload(flco: int, target: int, source: int, data_type: int, options: int = 0):
    data = bits_of(flco, 8) + bits_of(0, 8) + bits_of(options, 8)
    data += bits_of(target, 24) + bits_of(source, 24)
    values = [int("".join(map(str, data[8 * i:8 * i + 8])), 2) for i in range(9)]
    parity = rs129_parity(values) ^ LC_MASKS[data_type]
    return np.array(data + bits_of(parity, 24))


PREAMBLE = csbk_payload(0x3D, 0, [(0x40, 8), (2, 8), (100, 24), (2000, 24)])
HEADER = lc_payload(0, 100, 2000, data_type=1)
FILLER = list(np.random.default_rng(9).integers(0, 4, 200))


def wave(dibits):
    return fsk4_wave(dibits + FILLER, unit=DMR_UNIT, dc=-0.05)


def test_golay_has_minimum_distance_eight_and_corrects_three():
    words = np.array([golay20_encode(d) for d in range(256)], dtype=np.uint32)
    distance = np.bitwise_count(words[:, None] ^ words[None, :])
    assert distance[~np.eye(256, dtype=bool)].min() == 8
    assert golay20_decode(golay20_encode(0x73) ^ 0b10010000000000000001) == (0x73, 3)


def test_reed_solomon_codewords_have_zero_syndromes():
    data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11]
    p = rs129_parity(data)
    word = data + [p >> 16, (p >> 8) & 255, p & 255]
    for root in (1, 2, 3):
        s = 0
        for c in word:                                    # Horner at alpha^root
            s = _gmul(s, _EXP[root]) ^ c
        assert s == 0


def test_bptc_round_trip_corrects_single_errors():
    data = np.random.default_rng(10).integers(0, 2, 96).astype(np.uint8)
    sent = bptc_encode(data)
    received = sent.copy()
    received[[3, 70, 150]] ^= 1                           # three separate errors
    assert list(bptc_data(received)) == list(data)


def test_bursts_give_colour_code_slot_and_call():
    stream = (burst(1, PREAMBLE) + burst(2, HEADER, data_type=1)
              + burst(1, None, voice=True))
    decoder = DmrDecoder(RATE)
    found = run(decoder, wave(stream))
    assert [m.summary()[10:] for m in found] == [
        "DMR  CC 7  slot 1  preamble  TG 100  from 2000",
        "DMR  CC 7  slot 2  voice header  group call  TG 100  from 2000",
        "DMR  slot 1  voice",
    ]
    assert decoder.bad_bursts == 0


def test_private_encrypted_call_and_manufacturer_blocks():
    terminator = lc_payload(3, 300, 2000, data_type=2, options=0x40)
    moto = csbk_payload(0x3E, 0x10, [(0, 64)])
    found = run(DmrDecoder(RATE), wave(burst(2, terminator, data_type=2) + burst(1, moto)))
    assert [m.text for m in found] == [
        "terminator  private call  to 300  from 2000  encrypted", "Motorola CSBK 3E"]


def test_failed_checks_are_counted_not_shown():
    bad = PREAMBLE.copy()
    bad[20:24] ^= 1                                       # beyond BPTC's reach in one row
    decoder = DmrDecoder(RATE)
    assert run(decoder, wave(burst(1, bad))) == []
    assert decoder.bad_bursts == 1


def test_noise_decodes_nothing():
    noise = 0.3 * np.random.default_rng(12).standard_normal(int(RATE * 5))
    assert run(DmrDecoder(RATE), noise) == []


def test_worker_decodes_from_iq():
    rate = 192e3
    dev = fsk4_wave(burst(2, HEADER, data_type=1) + FILLER, rate=rate, unit=648.0,
                    dc=0.0, noise=0.0)
    iq = np.exp(2j * np.pi * np.cumsum(dev) / rate).astype(np.complex64)
    worker = DecodeWorker(RingSource(rate), "dmr")
    found = []
    for block in np.array_split(iq, 20):
        found += worker.process(block)
    assert [m.fields.get("target") for m in found] == [100]



# -- packet data ---------------------------------------------------------------------

from src.rgc_sdr.dsp.dmr import crc9, data_header_crc, packet_crc32  # noqa: E402
from src.rgc_sdr.dsp.p25 import trellis34_encode  # noqa: E402
from src.rgc_sdr.targets import TargetStore  # noqa: E402
from tests.test_p25 import udp_packet  # noqa: E402


def data_burst(slot, info196, data_type, colour=7):
    """A burst carrying `info196` already coded (BPTC or trellis)."""
    cach = np.zeros(24, dtype=int)
    cach[TACT_POSITIONS] = bits_of(tact_encode(0b1000 | (slot - 1) << 2), 7)
    slot_type = bits_of(golay20_encode(colour << 4 | data_type), 20)
    bits = (list(cach) + list(info196[:98]) + slot_type[:10] + bits_of(SYNCS["bs"], 48)
            + slot_type[10:] + list(info196[98:]))
    return [2 * bits[i] + bits[i + 1] for i in range(0, 288, 2)]


def packet_bursts(user: bytes, sap=4, source=2000, target=100, slot=1):
    """A confirmed rate-3/4 data packet: header, then 16-byte blocks, CRC-32 last."""
    blocks = -(-(len(user) + 4) // 16)
    pad = blocks * 16 - len(user) - 4
    data = user + bytes(pad)
    data += packet_crc32(data)
    head = [0, 0, 0, pad >> 4 & 1] + bits_of(3, 4) + bits_of(sap, 4) + bits_of(pad & 15, 4)
    head += bits_of(target, 24) + bits_of(source, 24) + [1] + bits_of(blocks, 7)
    head += [0] * (80 - len(head))
    head += bits_of(data_header_crc(np.array(head)), 16)
    out = data_burst(slot, bptc_encode(np.array(head)), 6)
    for k in range(blocks):
        body = list(np.unpackbits(np.frombuffer(data[16 * k:16 * k + 16], np.uint8)))
        serial = bits_of(k, 7)
        coded = np.array(serial + bits_of(crc9(np.array(body + serial)), 9) + body)
        d98 = trellis34_encode(coded)
        info = np.ravel([[d >> 1, d & 1] for d in d98])
        out += data_burst(slot, info, 8)
    return out


def test_a_location_report_goes_on_the_map():
    lat, lon = -37.6, 144.9
    point = (b"\x51" + (0x80000000 | round(-lat / 90 * 2 ** 31)).to_bytes(4, "big")
             + round(lon / 180 * 2 ** 31).to_bytes(4, "big", signed=True))
    lrrp = b"\x0d\x15\x22\x03\x00\x00\x01" + point
    found = run(DmrDecoder(RATE), wave(packet_bursts(udp_packet(4001, lrrp)[2:])))
    (m,) = [m for m in found if m.kind == "data"]
    assert m.text == "packet data  to 100  from 2000: location report (LRRP)"
    assert m.position == pytest.approx((lat, lon), abs=1e-5)
    assert "-37.60000, 144.90000" in m.summary() and "-37.6" not in m.summary(show_text=False)
    store = TargetStore()
    assert store.update([m]) == 1 and store.placed()[0].ident == "2000"


def test_a_text_message_shows_its_text():
    text = "RADIO CHECK"
    body = b"\x00\x10\x00\x00" + text.encode("utf-16-le")
    found = run(DmrDecoder(RATE), wave(packet_bursts(udp_packet(4007, body)[2:])))
    (m,) = [m for m in found if m.kind == "data"]
    assert "text message (TMS)" in m.text and text in m.summary()


def test_a_damaged_packet_is_rejected():
    bursts = packet_bursts(udp_packet(4007, "HI".encode("utf-16-le"))[2:])
    bursts[144 + 20] ^= 0b11                           # inside the header's payload
    bursts[144 + 21] ^= 0b11
    bursts[144 + 40] ^= 0b11
    decoder = DmrDecoder(RATE)
    assert [m for m in run(decoder, wave(bursts)) if m.kind == "data"] == []
