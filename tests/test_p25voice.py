"""P25 voice: LDU framing, link control (who is talking), and audio through mbelib.

The IMBE codec is mbelib's (an outside library); these check everything around it and
that frames reach it. The link control's Reed-Solomon order is checked against an
encoder written here until a real transmission confirms it (PLANNING.md 7p).
"""

import numpy as np
import pytest

from src.rgc_sdr.dsp.demod import DemodChain
from src.rgc_sdr.dsp.p25 import FRAME_SYNC, P25Decoder, bch_encode
from src.rgc_sdr.dsp import p25voice as V
from tests.test_fsk4 import RATE, fsk4_wave, run, word_dibits
from tests.test_p25 import FILLER, NAC, bits_of

needs_codec = pytest.mark.skipif(not V.codec_available(), reason="mbelib not installed")


def lc_words(talkgroup=10, source=3051234, options=0x00, lcf=0x00):
    bits = bits_of(lcf, 8) + bits_of(0, 8) + bits_of(options, 8) + bits_of(0, 8)
    bits += bits_of(talkgroup, 16) + bits_of(source, 24)
    data = [int("".join(map(str, bits[6 * i:6 * i + 6])), 2) for i in range(12)]
    return data + V.rs_24_12_parity(data)


def ldu(duid, words, frames):
    """An LDU: sync, NID, then the layout, hex words Hamming(10,6) coded."""
    content, w, f = [], iter(words), iter(frames)
    for kind, n in V.LDU_LAYOUT:
        if kind == "v":
            content += list(V.interleave(next(f)))
        elif kind == "h":
            for _ in range(4):
                b = bits_of(V.hamming_10_6_encode(next(w)), 10)
                content += [2 * b[i] + b[i + 1] for i in range(0, 10, 2)]
        else:
            content += [0] * n
    nid = bits_of(bch_encode(NAC << 4 | duid), 63) + [0]
    content = [2 * nid[i] + nid[i + 1] for i in range(0, 64, 2)] + content
    out = word_dibits(FRAME_SYNC, 24)
    for d in content:
        if len(out) % 36 == 35:
            out.append(0b10)
        out.append(d)
    while len(out) % 36 != 0:                         # the last status symbol
        out.append(0b10)
    return out


def random_frames(seed, count=9):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        f = np.zeros((8, 23), dtype=np.uint8)
        for r, width in enumerate((23, 23, 23, 23, 15, 15, 15, 7)):
            f[r, :width] = rng.integers(0, 2, width)
        frames.append(f)
    return frames


def play(voice, x, block=3125):
    """Audio from a stream, block by block."""
    parts = [voice.process(x[i:i + block]) for i in range(0, x.size, block)]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def call(options=0x00, ldus=4):
    stream = []
    for k in range(ldus):
        duid = 0x5 if k % 2 == 0 else 0xA
        words = lc_words(options=options) if duid == 0x5 else [0] * 24
        stream += ldu(duid, words, random_frames(k))
    return stream + FILLER


def test_the_interleave_fills_every_cell_once_and_round_trips():
    cells = set(zip(V.IW, V.IX)) | set(zip(V.IY, V.IZ))
    widths = (23, 23, 23, 23, 15, 15, 15, 7)
    assert cells == {(r, c) for r, w in enumerate(widths) for c in range(w)}
    frame = random_frames(1, 1)[0]
    assert (V.deinterleave(V.interleave(frame)) == frame).all()


def test_hamming_10_6_corrects_one_error():
    for data in (0, 0b101101, 0b111111):
        word = V.hamming_10_6_encode(data)
        assert V.hamming_10_6_decode(word) == (data, 0)
        for b in range(10):
            assert V.hamming_10_6_decode(word ^ (1 << b)) == (data, 1)


def test_reed_solomon_24_12_checks_the_link_control():
    words = lc_words()
    assert V.rs_24_12_ok(words[:12], words[12:])
    words[3] ^= 0b000100
    assert not V.rs_24_12_ok(words[:12], words[12:])


def test_link_control_gives_talkgroup_and_source():
    assert V.link_control(lc_words(10, 3051234)) == (0x00, 0x00, 10, 3051234)
    unit = lc_words(lcf=0x03)
    assert V.link_control(unit)[0] == 0x03
    damaged = lc_words()
    damaged[0] = None
    assert V.link_control(damaged) is None


@needs_codec
def test_a_call_says_who_is_talking_and_gives_audio():
    voice = V.P25Voice(RATE)
    audio = play(voice, fsk4_wave(call()))
    assert voice.frames == 4
    assert audio.size == 4 * 9 * V.FRAME_SAMPLES       # 180 ms per LDU
    (who,) = voice.take_calls()
    assert (who.nac, who.talkgroup, who.source, who.encrypted) == (NAC, 10, 3051234, False)
    assert "TG 10  from 3051234" in who.summary()


@needs_codec
def test_an_encrypted_call_is_silent():
    voice = V.P25Voice(RATE)
    audio = play(voice, fsk4_wave(call(options=0x40)))
    assert audio.size > 0 and not audio.any()
    assert voice.take_calls()[0].encrypted


def test_the_decoder_panel_shows_who_is_talking():
    found = run(P25Decoder(RATE), fsk4_wave(call(ldus=2)))
    assert any(m.text == "voice  TG 10  from 3051234" for m in found)


@needs_codec
def test_the_p25_mode_plays_48_khz_audio_from_fm():
    rate = 192e3
    dev = fsk4_wave(call(), rate=rate, unit=600.0, dc=0.0, noise=0.0)
    iq = np.exp(2j * np.pi * np.cumsum(dev) / rate).astype(np.complex64)
    chain = DemodChain(rate, "p25", agc=True)
    out = [chain.process(iq[i:i + 8192]) for i in range(0, iq.size, 8192)]
    audio = np.concatenate([o for o in out if o.size])
    assert chain.audio_rate == 48000.0
    assert audio.size >= 0.8 * iq.size / rate * 48000          # silence between frames
    assert chain.p25_voice.frames == 4
