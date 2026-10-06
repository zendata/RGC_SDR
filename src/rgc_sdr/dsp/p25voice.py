"""P25 Phase 1 voice: who is talking, and what they say (VK3RQ, 2026-10-07).

A voice transmission is LDU1 and LDU2 frames, 180 ms each, alternating. After the frame
sync and NID, each carries nine IMBE voice frames (144 bits, 72 dibits, interleaved)
between which sit 6-bit "hex words" with Hamming(10,6,3) parity:

    V1 V2 [4 words] V3 [4] V4 [4] V5 [4 parity] V6 [4 parity] V7 [4 parity] V8 [LSD] V9

LDU1's twelve words are the link control -- talkgroup and the talking radio's ID --
protected by Reed-Solomon (24,12,13) over GF(64); LDU2's carry the encryption sync.

The voice itself is IMBE, decoded by mbelib (Homebrew's `mbelib`, loaded with ctypes:
an outside codec, as PLANNING.md always said voice would need). mbelib does the IMBE
error correction and synthesis; this module hands it each frame deinterleaved into
its 8 x 23 array.

Pure NumPy apart from the codec; no Qt, no device access.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fsk4 import SyncFinder, bits_to_int, dibits_to_bits, to_dibits
from .p25 import FRAME_SYNC, bch_decode

# -- the IMBE interleave ------------------------------------------------------------
#
# Where each of a voice frame's 72 dibits goes in mbelib's imbe_fr[8][23]: the first bit
# of dibit k to [IW[k]][IX[k]], the second to [IY[k]][IZ[k]]. The table is TIA-102.BABA's
# schedule as transcribed in DSD (src/p25p1_const.h), under this notice:
#
#   Copyright (C) 2010 DSD Author
#   GPG Key ID: 0x3F1D7FD0 (74EF 430D F7F2 0A48 FCE6  F630 FAA2 635D 3F1D 7FD0)
#
#   Permission to use, copy, modify, and/or distribute this software for any
#   purpose with or without fee is hereby granted, provided that the above
#   copyright notice and this permission notice appear in all copies.
#
#   THE SOFTWARE IS PROVIDED "AS IS" AND ISC DISCLAIMS ALL WARRANTIES WITH
#   REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
#   AND FITNESS.  IN NO EVENT SHALL ISC BE LIABLE FOR ANY SPECIAL, DIRECT,
#   INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
#   LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
#   OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
#   PERFORMANCE OF THIS SOFTWARE.

IW = (0, 2, 4, 1, 3, 5, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6,
      0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 5, 1, 3, 6, 0, 2, 5, 1, 3, 6,
      0, 2, 5, 1, 3, 7, 0, 2, 5, 1, 3, 7, 0, 2, 5, 1, 4, 7, 0, 3, 5, 2, 4, 7)
IX = (22, 20, 10, 20, 18, 0, 20, 18, 8, 18, 16, 13, 18, 16, 6, 16, 14, 11,
      16, 14, 4, 14, 12, 9, 14, 12, 2, 12, 10, 7, 12, 10, 0, 10, 8, 5,
      10, 8, 13, 8, 6, 3, 8, 6, 11, 6, 4, 1, 6, 4, 9, 4, 2, 6,
      4, 2, 7, 2, 0, 4, 2, 0, 5, 0, 13, 2, 0, 21, 3, 21, 11, 0)
IY = (1, 3, 5, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 4,
      1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 5, 1, 3, 6, 0, 2, 5, 1, 3, 6, 0, 2, 5,
      1, 3, 6, 0, 2, 5, 1, 3, 7, 0, 2, 5, 1, 4, 7, 0, 3, 5, 2, 4, 7, 1, 3, 5)
IZ = (21, 19, 1, 21, 19, 9, 19, 17, 14, 19, 17, 7, 17, 15, 12, 17, 15, 5,
      15, 13, 10, 15, 13, 3, 13, 11, 8, 13, 11, 1, 11, 9, 6, 11, 9, 14,
      9, 7, 4, 9, 7, 12, 7, 5, 2, 7, 5, 10, 5, 3, 0, 5, 3, 8,
      3, 1, 5, 3, 1, 6, 1, 14, 3, 1, 22, 4, 22, 12, 1, 22, 20, 2)

#: An LDU after the NID: what each stretch of dibits is (v = voice frame, h = hex word,
#: l = low-speed data), status symbols not counted. 784 dibits.
LDU_LAYOUT = (("v", 72), ("v", 72), ("h", 20), ("v", 72), ("h", 20), ("v", 72), ("h", 20),
              ("v", 72), ("h", 20), ("v", 72), ("h", 20), ("v", 72), ("h", 20), ("v", 72),
              ("l", 16), ("v", 72))
LDU_DIBITS = sum(n for _, n in LDU_LAYOUT)
#: Symbols from the sync to the end of an LDU, status symbols included.
LDU_SYMBOLS = next(n for n in range(24, 2000)
                   if sum(1 for i in range(24, n) if i % 36 != 35) == 32 + LDU_DIBITS)
#: The audio mbelib makes: 8 kHz, 160 samples (20 ms) a frame.
AUDIO_RATE = 8000.0
FRAME_SAMPLES = 160


def deinterleave(dibits72: np.ndarray) -> np.ndarray:
    """72 voice-frame dibits -> mbelib's imbe_fr[8][23] (unused cells zero)."""
    frame = np.zeros((8, 23), dtype=np.uint8)
    d = np.asarray(dibits72, dtype=np.uint8)
    frame[IW, IX] = (d >> 1) & 1
    frame[IY, IZ] = d & 1
    return frame


def interleave(frame: np.ndarray) -> np.ndarray:
    """The inverse, for tests."""
    return (frame[IW, IX] << 1 | frame[IY, IZ]).astype(np.uint8)


# -- hex words: Hamming(10,6,3) -------------------------------------------------------

#: Parity (4 bits) contributed by each data bit, MSB first (TIA-102.BAAA's G matrix, as
#: DSD's Hamming.hpp gives it).
_HAMMING_10_6 = (0b1110, 0b1101, 0b1011, 0b0111, 0b0011, 0b1100)


def hamming_10_6_encode(data6: int) -> int:
    parity = 0
    for i, p in enumerate(_HAMMING_10_6):
        if data6 >> (5 - i) & 1:
            parity ^= p
    return data6 << 4 | parity


_HAMMING_TABLE = {}
for _d in range(64):
    _c = hamming_10_6_encode(_d)
    _HAMMING_TABLE[_c] = (_d, 0)
for _d in range(64):
    _c = hamming_10_6_encode(_d)
    for _b in range(10):
        _HAMMING_TABLE.setdefault(_c ^ (1 << _b), (_d, 1))


def hamming_10_6_decode(word10: int) -> tuple[int, int] | None:
    """(data, bits corrected), or None for more than one error."""
    return _HAMMING_TABLE.get(word10)


# -- Reed-Solomon (24,12,13) over GF(64) ------------------------------------------------

_EXP = [0] * 126
_LOG = [0] * 64
_v = 1
for _i in range(63):
    _EXP[_i], _LOG[_v] = _v, _i
    _v <<= 1
    if _v & 64:
        _v ^= 0b1000011                     # x^6 + x + 1
for _i in range(63, 126):
    _EXP[_i] = _EXP[_i - 63]


def _gmul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def _rs_generator(roots: int) -> list[int]:
    g = [1]
    for i in range(1, roots + 1):
        ng = [0] * (len(g) + 1)
        for k, c in enumerate(g):
            ng[k] ^= c
            ng[k + 1] ^= _gmul(c, _EXP[i])
        g = ng
    return g


_RS_G12 = _rs_generator(12)


def rs_24_12_parity(data12: list[int]) -> list[int]:
    """Twelve 6-bit parity symbols for twelve data symbols, highest degree first."""
    reg = [0] * 12
    for d in data12:
        fb = d ^ reg[0]
        reg = [reg[i + 1] ^ _gmul(fb, _RS_G12[i + 1]) for i in range(11)] + [
            _gmul(fb, _RS_G12[12])]
    return reg


def rs_24_12_ok(data12: list[int], parity12: list[int]) -> bool:
    """Whether the 24 symbols form a codeword (all twelve syndromes zero)."""
    word = list(data12) + list(parity12)
    for i in range(1, 13):
        s = 0
        for c in word:
            s = _gmul(s, _EXP[i]) ^ c
        if s:
            return False
    return True


# -- the codec ---------------------------------------------------------------------------

class _MbeParms(ctypes.Structure):
    _fields_ = [("w0", ctypes.c_float), ("L", ctypes.c_int), ("K", ctypes.c_int),
                ("Vl", ctypes.c_int * 57), ("Ml", ctypes.c_float * 57),
                ("log2Ml", ctypes.c_float * 57), ("PHIl", ctypes.c_float * 57),
                ("PSIl", ctypes.c_float * 57), ("gamma", ctypes.c_float),
                ("un", ctypes.c_int), ("repeat", ctypes.c_int)]


def _load_mbelib():
    candidates = [ctypes.util.find_library("mbe"), "/opt/homebrew/lib/libmbe.dylib",
                  "/usr/local/lib/libmbe.dylib"]
    for path in candidates:
        if path and (Path(path).exists() or "/" not in path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    return None


_MBE = None


def codec_available() -> bool:
    global _MBE
    if _MBE is None:
        _MBE = _load_mbelib() or False
    return bool(_MBE)


class ImbeDecoder:
    """mbelib's IMBE 7200x4400 decoder, one voice frame in, 20 ms of audio out."""

    #: mbelib's float output is in 16-bit units (DSD scales it to shorts).
    SCALE = 1.0 / 32768.0
    #: Its voiced/unvoiced quality setting; DSD's default.
    UV_QUALITY = 3

    def __init__(self) -> None:
        if not codec_available():
            raise RuntimeError("P25 voice needs mbelib: brew install mbelib")
        self._mbe = _MBE
        self._cur, self._prev, self._enh = _MbeParms(), _MbeParms(), _MbeParms()
        self._mbe.mbe_initMbeParms(ctypes.byref(self._cur), ctypes.byref(self._prev),
                                   ctypes.byref(self._enh))
        self._out = (ctypes.c_float * FRAME_SAMPLES)()
        self._fr = ((ctypes.c_char * 23) * 8)()
        self._d = (ctypes.c_char * 88)()
        self._err_str = ctypes.create_string_buffer(64)
        self.errors = 0

    def decode(self, frame: np.ndarray) -> np.ndarray:
        """imbe_fr[8][23] (0/1) -> 160 float samples, about -1..1."""
        flat = np.asarray(frame, dtype=np.uint8).reshape(8, 23)
        for r in range(8):
            ctypes.memmove(self._fr[r], flat[r].tobytes(), 23)
        errs, errs2 = ctypes.c_int(0), ctypes.c_int(0)
        self._mbe.mbe_processImbe7200x4400Framef(
            self._out, ctypes.byref(errs), ctypes.byref(errs2), self._err_str, self._fr,
            self._d, ctypes.byref(self._cur), ctypes.byref(self._prev),
            ctypes.byref(self._enh), self.UV_QUALITY)
        self.errors = int(errs2.value)
        return np.frombuffer(self._out, dtype=np.float32).copy() * self.SCALE


# -- frames ----------------------------------------------------------------------------

@dataclass
class VoiceCall:
    """Who is talking, from LDU1's link control."""

    nac: int
    talkgroup: int | None
    source: int
    encrypted: bool
    private: bool = False
    received: float = field(default_factory=time.time)

    def summary(self, show_text: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.received))
        to = f"to {self.talkgroup}" if self.private else f"TG {self.talkgroup}"
        secret = "  encrypted" if self.encrypted else ""
        return f"{stamp}  P25  NAC {self.nac:03X}  voice  {to}  from {self.source}{secret}"


def split_ldu(dibits: np.ndarray):
    """An LDU's dibits after the NID (status symbols removed) -> nine voice-frame
    dibit arrays, the hex words (6 data bits each, None where uncorrectable)."""
    voice, words = [], []
    pos = 0
    for kind, n in LDU_LAYOUT:
        part = dibits[pos:pos + n]
        pos += n
        if kind == "v":
            voice.append(part)
        elif kind == "h":
            bits = dibits_to_bits(part)
            for w in range(4):
                decoded = hamming_10_6_decode(bits_to_int(bits[10 * w:10 * w + 10]))
                words.append(None if decoded is None else decoded[0])
    return voice, words


def link_control(words: list) -> tuple[int, int, int, int] | None:
    """(LCF, options, destination, source) from LDU1's 24 words, if the Reed-Solomon
    check passes."""
    if any(w is None for w in words) or len(words) < 24:
        return None
    data, parity = list(words[:12]), list(words[12:24])
    if not rs_24_12_ok(data, parity):
        return None
    bits = np.ravel([[(w >> (5 - i)) & 1 for i in range(6)] for w in data])
    lcf, options = bits_to_int(bits[0:8]), bits_to_int(bits[16:24])
    if lcf & 0x3F == 0x03:                       # unit to unit
        return lcf, options, bits_to_int(bits[24:48]), bits_to_int(bits[48:72])
    return lcf, options, bits_to_int(bits[32:48]), bits_to_int(bits[48:72])


class P25Voice:
    """FM discriminator samples in; voice audio (8 kHz) and who is talking out."""

    def __init__(self, sample_rate: float) -> None:
        self._finder = SyncFinder(sample_rate, {"fs": FRAME_SYNC}, LDU_SYMBOLS + 2)
        self._codec = ImbeDecoder() if codec_available() else None
        #: Muted while the current call is encrypted (no key, so it would be noise).
        self.encrypted = False
        self.frames = 0
        self.calls: list[VoiceCall] = []
        self._last_call: tuple | None = None

    def reset(self) -> None:
        self._finder.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        audio = []
        for _hit, symbols in self._finder.process(x):
            dibits = to_dibits(symbols)
            nid = dibits_to_bits(np.concatenate([dibits[24:35], dibits[36:57]]))
            value, errors = bch_decode(bits_to_int(nid[:63]))
            if errors > 11:
                continue
            nac, duid = value >> 4, value & 0xF
            if duid not in (0x5, 0xA):
                if duid in (0x3, 0xF):              # terminator: the call is over
                    self.encrypted = False
                    self._last_call = None
                continue
            body = np.array([dibits[i] for i in range(57, dibits.size) if i % 36 != 35])
            if body.size < LDU_DIBITS:
                continue
            self.frames += 1
            voice, words = split_ldu(body[:LDU_DIBITS])
            if duid == 0x5:
                lc = link_control(words)
                if lc is not None:
                    lcf, options, dest, source = lc
                    self.encrypted = bool(options & 0x40)
                    key = (nac, dest, source, self.encrypted)
                    if key != self._last_call:
                        self._last_call = key
                        self.calls.append(VoiceCall(nac, dest, source, self.encrypted,
                                                    private=lcf & 0x3F == 0x03))
            if self._codec is None or self.encrypted:
                audio.append(np.zeros(9 * FRAME_SAMPLES, dtype=np.float32))
                continue
            for part in voice:
                audio.append(self._codec.decode(deinterleave(part)))
        return np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)

    def take_calls(self) -> list[VoiceCall]:
        out, self.calls = self.calls, []
        return out
