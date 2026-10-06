"""P25 Phase 1 metadata: NID, trellis-coded TSBKs, channel plans and grants.

The BCH generator is checked against its published value and the CRC against a
published check value (tests/test_fsk4.py); the frame layout comes from TIA-102 and was
confirmed on air (PLANNING.md 7p). All IDs here are made up.
"""

import numpy as np

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
FILLER = list(np.random.default_rng(9).integers(0, 4, 300))
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
    stream = frame_dibits(0x5) + list(np.random.default_rng(7).integers(0, 4, 400))
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
