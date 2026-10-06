"""DAB+ stage 2: sub-channel decoding to access units, against an encoder written here,
with the parts the standards fix checked directly (EEP sizes, RS correction limits)."""

import numpy as np
import pytest

from src.rgc_sdr.dsp import dab as D
from src.rgc_sdr.dsp import dabplus as P


def test_eep_profiles_fill_their_sub_channels_exactly():
    # Punctured length must be the sub-channel's size: 64 bits per CU.
    for option, level, per_n in ((0, 1, 12), (0, 2, 8), (0, 3, 6), (0, 4, 4),
                                 (1, 1, 27), (1, 2, 21), (1, 3, 18), (1, 4, 15)):
        for n in (1, 2, 3, 8):
            l1, l2, pi1, pi2, _ = P.eep_profile(option, level, per_n * n)
            assert P.eep_pattern(l1, l2, pi1, pi2).sum() == per_n * n * 64


def test_reed_solomon_corrects_five_bytes_and_refuses_six():
    rng = np.random.default_rng(1)
    for errors in range(7):
        data = rng.integers(0, 256, 110).astype(np.uint8).tobytes()
        word = bytearray(data + P.rs_encode(data))
        bad = bytearray(word)
        for pos in rng.choice(120, errors, replace=False):
            bad[pos] ^= int(rng.integers(1, 256))
        result = P.rs_decode(bad)
        if errors <= 5:
            assert result == errors and bad == word
        else:
            assert result == -1


def test_audio_specific_config_for_he_aac():
    # HE-AAC 24 -> 48 kHz stereo: explicit SBR (AOT 5), 960-sample frames.
    asc = P.audio_specific_config(1, True, True, False)
    bits = "".join(f"{b:08b}" for b in asc)
    assert bits[:5] == "00101" and bits[5:9] == "0110" and bits[9:13] == "0010"
    assert bits[13:17] == "0011" and bits[17:22] == "00010" and bits[22] == "1"


def superframe(s: int, rng) -> bytes:
    """An HE-AAC (48 kHz, SBR, stereo) superframe: 3 AUs of random bytes with CRCs."""
    size = 110 * s
    starts = [6, 6 + (size - 6) // 3, 6 + 2 * (size - 6) // 3, size]
    body = bytearray(size)
    body[2] = 0b0110_0000                                     # dac 48k, sbr, stereo
    au_bits = f"{starts[1]:012b}{starts[2]:012b}"
    body[3:6] = bytes(int(au_bits[i:i + 8], 2) for i in range(0, 24, 8))
    for k in range(3):
        a, b = starts[k], starts[k + 1]
        au = rng.integers(0, 256, b - a - 2).astype(np.uint8).tobytes()
        body[a:b - 2] = au
        body[b - 2:b] = P.au_crc(au).to_bytes(2, "big")
    body[0:2] = P.firecode(bytes(body[2:11])).to_bytes(2, "big")
    return bytes(body)


def protect(sf: bytes, s: int) -> bytes:
    """RS(120,110) over the virtual interleave: parity rows, sent after the data."""
    out = bytearray(sf) + bytearray(10 * s)
    for row in range(s):
        parity = P.rs_encode(bytes(sf[row + s * j] for j in range(110)))
        for r in range(10):
            out[110 * s + row + s * r] = parity[r]
    return bytes(out)


def sub_channel_stream(superframes: int, size_cu=48, option=0, level=3, seed=3):
    """Soft bits for each CIF as the radio would deliver them, and the superframes."""
    rng = np.random.default_rng(seed)
    l1, l2, pi1, pi2, rate = P.eep_profile(option, level, size_cu)
    s, pattern = rate // 8, P.eep_pattern(l1, l2, pi1, pi2)
    sent = [protect(superframe(s, rng), s) for _ in range(superframes)]
    stream = b"".join(sent)
    frame_bytes = len(stream) // (5 * superframes)
    coded = []
    prbs = D.prbs(frame_bytes * 8)
    for k in range(5 * superframes):
        bits = np.unpackbits(np.frombuffer(stream[k * frame_bytes:(k + 1) * frame_bytes],
                                           np.uint8)) ^ prbs
        coded.append(D.convolve(bits)[pattern].astype(np.uint8))
    delays = np.array(P.DELAYS)[np.arange(coded[0].size) % 16]
    cifs = []
    for r in range(len(coded) + 15):
        cif = np.zeros(coded[0].size, dtype=np.uint8)
        src = r - delays
        ok = (src >= 0) & (src < len(coded))
        idx = np.flatnonzero(ok)
        cif[idx] = np.array([coded[src[i]][i] for i in idx], dtype=np.uint8)
        cifs.append(1.0 - 2.0 * cif)
    return cifs, sent


def test_sub_channel_to_access_units():
    cifs, _ = sub_channel_stream(4)
    audio = P.DabPlusAudio(0, 48, 0, 3)
    assert audio.bitrate == 64 and audio.rows == 8
    rng = np.random.default_rng(9)
    noisy = [c + 0.5 * rng.standard_normal(c.size) for c in cifs]
    for k in range(0, len(noisy), 4):
        audio.push(noisy[k:k + 4])
    assert audio.superframes >= 3
    assert audio.bad_superframes == 0 and audio.bad_aus == 0


def test_superframes_are_found_mid_stream():
    cifs, _ = sub_channel_stream(4, seed=5)
    audio = P.DabPlusAudio(0, 48, 0, 3)
    for k in range(2, len(cifs), 4):                  # start two frames late
        audio.push(cifs[k:k + 4])
    assert audio.superframes >= 2 and audio.bad_aus == 0
