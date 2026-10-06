"""DAB+ audio (ETSI TS 102 563): one station's sub-channel to sound.

Stage 2 of DAB+ (VK3RQ, 2026-10-07). From the main service channel's soft bits:

1. **Time deinterleaving** (EN 300 401 clause 12): each bit of a 24 ms logical frame is
   delayed by 0-15 frames according to its position mod 16 (table 21: a bit reversal).
2. **EEP depuncturing** (tables 18/20) and the K=7 Viterbi decoder (`dab.viterbi`), then
   energy dispersal removed.
3. **Superframes**: five logical frames, Reed-Solomon (120,110) over GF(256) across rows
   of a virtual interleave (TS 102 563 clause 6), found by the header's Fire code.
4. **Access units** (20-60 ms of audio each), each with a CRC, decoded by **FAAD2**
   (Homebrew, through ctypes) -- HE-AAC v2 with 960-sample frames, an outside codec as
   mbelib is for P25.

Pure NumPy apart from the codec; no Qt, no device access.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path

import numpy as np

from .dab import PI, PI_TAIL, depuncture, prbs, viterbi

CU_BITS = 64
#: Delay (in logical frames) of a bit at position i, by i mod 16 (EN 300 401 table 21).
DELAYS = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)


# -- EEP ------------------------------------------------------------------------------------

def eep_profile(option: int, level: int, size_cu: int) -> tuple[int, int, int, int, int]:
    """(L1, L2, PI1, PI2, bit rate in kbit/s) for an EEP sub-channel of `size_cu` CUs:
    option 0 is set A (tables 17/18), 1 is set B (tables 19/20)."""
    if option == 0:
        n = size_cu // {1: 12, 2: 8, 3: 6, 4: 4}[level]
        rate = 8 * n
        if level == 1:
            return 6 * n - 3, 3, 24, 23, rate
        if level == 2:
            return (5, 1, 13, 12, rate) if n == 1 else (2 * n - 3, 4 * n + 3, 14, 13, rate)
        if level == 3:
            return 6 * n - 3, 3, 8, 7, rate
        return 4 * n - 3, 2 * n + 3, 3, 2, rate
    n = size_cu // {1: 27, 2: 21, 3: 18, 4: 15}[level]
    pi1, pi2 = {1: (10, 9), 2: (6, 5), 3: (4, 3), 4: (2, 1)}[level]
    return 24 * n - 3, 3, pi1, pi2, 32 * n


def eep_pattern(l1: int, l2: int, pi1: int, pi2: int) -> np.ndarray:
    """Which mother-code bits are sent, for L1 blocks of PI1, L2 of PI2, and the tail."""
    return np.concatenate([np.tile(PI[pi1], 4)] * l1 + [np.tile(PI[pi2], 4)] * l2 + [PI_TAIL])


# -- Reed-Solomon (120, 110) over GF(256) -----------------------------------------------

_EXP = [0] * 512
_LOG = [0] * 256
_v = 1
for _i in range(255):
    _EXP[_i], _LOG[_v] = _v, _i
    _v <<= 1
    if _v & 0x100:
        _v ^= 0x11D                          # x^8 + x^4 + x^3 + x^2 + 1
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _gmul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def _ginv(a: int) -> int:
    return _EXP[255 - _LOG[a]]


_RS_G = [1]
for _i in range(10):                         # roots alpha^0 .. alpha^9
    _g = [0] * (len(_RS_G) + 1)
    for _k, _c in enumerate(_RS_G):
        _g[_k] ^= _c
        _g[_k + 1] ^= _gmul(_c, _EXP[_i])
    _RS_G = _g


def rs_encode(data110: bytes) -> bytes:
    """Ten parity bytes for 110 data bytes."""
    reg = [0] * 10
    for d in data110:
        fb = d ^ reg[0]
        reg = [reg[i + 1] ^ _gmul(fb, _RS_G[i + 1]) for i in range(9)] + [_gmul(fb, _RS_G[10])]
    return bytes(reg)


def _gpow(x: int, power: int) -> int:
    return _EXP[(_LOG[x] * power) % 255]


def _peval(poly: list[int], x: int) -> int:
    """A polynomial, highest degree first, at x."""
    y = poly[0]
    for c in poly[1:]:
        y = _gmul(y, x) ^ c
    return y


def _pmul(p: list[int], q: list[int]) -> list[int]:
    r = [0] * (len(p) + len(q) - 1)
    for j, b in enumerate(q):
        for i, a in enumerate(p):
            r[i + j] ^= _gmul(a, b)
    return r


def _padd(p: list[int], q: list[int]) -> list[int]:
    r = [0] * max(len(p), len(q))
    for i, c in enumerate(p):
        r[i + len(r) - len(p)] = c
    for i, c in enumerate(q):
        r[i + len(r) - len(q)] ^= c
    return r


def rs_decode(word: bytearray) -> int:
    """Correct a 120-byte RS(120,110) word in place, up to five bad bytes: syndromes,
    Berlekamp-Massey, Chien search, Forney (first consecutive root alpha^0). Returns the
    number corrected, or -1 if it could not be."""
    nsym, n = 10, len(word)
    synd = [_peval(list(word), _EXP[i]) for i in range(nsym)]
    if not any(synd):
        return 0
    # Berlekamp-Massey: the error locator, lowest degree last.
    loc, old = [1], [1]
    for i in range(nsym):
        delta = synd[i]
        for j in range(1, len(loc)):
            delta ^= _gmul(loc[-(j + 1)], synd[i - j])
        old = old + [0]
        if delta:
            if len(old) > len(loc):
                new = [_gmul(c, delta) for c in old]
                old = [_gmul(c, _ginv(delta)) for c in loc]
                loc = new
            loc = _padd(loc, [_gmul(c, delta) for c in old])
    while loc and loc[0] == 0:
        loc.pop(0)
    errors = len(loc) - 1
    if errors * 2 > nsym:
        return -1
    # Chien search: byte k (degree n-1-k) is bad where the locator has a root at its
    # inverse position.
    rev = loc[::-1]
    positions = [n - 1 - i for i in range(n) if _peval(rev, _EXP[i % 255]) == 0]
    if len(positions) != errors:
        return -1
    # Forney.
    coef = [n - 1 - p for p in positions]
    errata = [1]
    for c in coef:
        errata = _pmul(errata, [_EXP[c], 1])
    # The error evaluator: syndromes times the locator, mod x^errors.
    product = _pmul(synd[::-1], errata)
    evaluator = product[-errors:]
    xs = [_EXP[c] for c in coef]
    for i, x in enumerate(xs):
        x_inv = _ginv(x)
        prime = 1
        for j, other in enumerate(xs):
            if j != i:
                prime = _gmul(prime, 1 ^ _gmul(x_inv, other))
        if prime == 0:
            return -1
        y = _peval(evaluator, x_inv)
        word[positions[i]] ^= _gmul(y, _ginv(prime))
    if any(_peval(list(word), _EXP[i]) for i in range(nsym)):
        return -1
    return errors


# -- CRCs ------------------------------------------------------------------------------------

def _crc16(data: bytes, poly: int, init: int) -> int:
    r = init
    for byte in data:
        for i in range(7, -1, -1):
            fb = ((r >> 15) ^ (byte >> i)) & 1
            r = (r << 1) & 0xFFFF
            if fb:
                r ^= poly
    return r


def firecode(data9: bytes) -> int:
    """The superframe header's Fire code (x^16+x^14+x^13+x^12+x^11+x^5+x^3+x^2+x+1)."""
    return _crc16(data9, 0x782F, 0)


def au_crc(data: bytes) -> int:
    """An access unit's CRC-CCITT, from all ones, complemented."""
    return _crc16(data, 0x1021, 0xFFFF) ^ 0xFFFF


# -- FAAD2 --------------------------------------------------------------------------------

class _FrameInfo(ctypes.Structure):
    _fields_ = [("bytesconsumed", ctypes.c_ulong), ("samples", ctypes.c_ulong),
                ("channels", ctypes.c_ubyte), ("error", ctypes.c_ubyte),
                ("samplerate", ctypes.c_ulong), ("sbr", ctypes.c_ubyte),
                ("object_type", ctypes.c_ubyte), ("header_type", ctypes.c_ubyte),
                ("num_front_channels", ctypes.c_ubyte), ("num_side_channels", ctypes.c_ubyte),
                ("num_back_channels", ctypes.c_ubyte), ("num_lfe_channels", ctypes.c_ubyte),
                ("channel_position", ctypes.c_ubyte * 64), ("ps", ctypes.c_ubyte)]


class _Config(ctypes.Structure):
    _fields_ = [("defObjectType", ctypes.c_ubyte), ("defSampleRate", ctypes.c_ulong),
                ("outputFormat", ctypes.c_ubyte), ("downMatrix", ctypes.c_ubyte),
                ("useOldADTSFormat", ctypes.c_ubyte),
                ("dontUpSampleImplicitSBR", ctypes.c_ubyte)]


_FAAD = None


def codec_available() -> bool:
    global _FAAD
    if _FAAD is None:
        _FAAD = False
        for path in (ctypes.util.find_library("faad"), "/opt/homebrew/lib/libfaad.dylib",
                     "/usr/local/lib/libfaad.dylib"):
            if path and (Path(path).exists() or "/" not in path):
                try:
                    lib = ctypes.CDLL(path)
                except OSError:
                    continue
                lib.NeAACDecOpen.restype = ctypes.c_void_p
                lib.NeAACDecGetCurrentConfiguration.argtypes = [ctypes.c_void_p]
                lib.NeAACDecGetCurrentConfiguration.restype = ctypes.POINTER(_Config)
                lib.NeAACDecSetConfiguration.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Config)]
                lib.NeAACDecInit2.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ulong,
                                              ctypes.POINTER(ctypes.c_ulong),
                                              ctypes.POINTER(ctypes.c_ubyte)]
                lib.NeAACDecInit2.restype = ctypes.c_byte
                lib.NeAACDecDecode.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FrameInfo),
                                               ctypes.c_char_p, ctypes.c_ulong]
                lib.NeAACDecDecode.restype = ctypes.POINTER(ctypes.c_short)
                lib.NeAACDecClose.argtypes = [ctypes.c_void_p]
                _FAAD = lib
                break
    return bool(_FAAD)


#: MPEG-4 sampling frequency indices.
_SR_INDEX = {48000: 3, 32000: 5, 24000: 6, 16000: 8}


def audio_specific_config(dac_rate: int, sbr: bool, stereo: bool, ps: bool) -> bytes:
    """The AudioSpecificConfig a DAB+ superframe header implies: AAC-LC with 960-sample
    frames, signalled explicitly as SBR (and PS) when used."""
    dac = 48000 if dac_rate else 32000
    core = dac // 2 if sbr else dac
    channels = 2 if stereo else 1
    bits = ""
    if sbr:
        bits += f"{29 if ps else 5:05b}{_SR_INDEX[core]:04b}{channels:04b}"
        bits += f"{_SR_INDEX[dac]:04b}{2:05b}"
    else:
        bits += f"{2:05b}{_SR_INDEX[core]:04b}{channels:04b}"
    bits += "100"                      # GASpecificConfig: 960 frame, no core coder, no ext
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


class AacDecoder:
    """FAAD2, configured from a DAB+ superframe header; access units in, float stereo
    samples out."""

    def __init__(self, asc: bytes) -> None:
        if not codec_available():
            raise RuntimeError("DAB+ audio needs FAAD2: brew install faad2")
        self._lib = _FAAD
        self._h = self._lib.NeAACDecOpen()
        # FAAD2's default 16-bit output, scaled here. (Its float output is already
        # normalised; scaling that again gave silence, the first time on air.)
        rate, ch = ctypes.c_ulong(), ctypes.c_ubyte()
        if self._lib.NeAACDecInit2(self._h, asc, len(asc), ctypes.byref(rate),
                                   ctypes.byref(ch)) < 0:
            raise RuntimeError("FAAD2 refused the stream's configuration")
        self.sample_rate = float(rate.value)
        self.errors = 0

    def decode(self, au: bytes) -> np.ndarray:
        """One access unit -> [samples, 2] float32, about -1..1 (empty on an error)."""
        info = _FrameInfo()
        out = self._lib.NeAACDecDecode(self._h, ctypes.byref(info), au, len(au))
        if info.error or not out or info.samples == 0:
            self.errors += 1
            return np.zeros((0, 2), dtype=np.float32)
        if info.samplerate:
            self.sample_rate = float(info.samplerate)
        pcm = np.ctypeslib.as_array(out, shape=(info.samples,)).astype(np.float32) / 32768.0
        if info.channels == 1:
            return np.repeat(pcm[:, None], 2, axis=1)
        return pcm.reshape(-1, info.channels)[:, :2]

    def close(self) -> None:
        if self._h:
            self._lib.NeAACDecClose(self._h)
            self._h = None


# -- the sub-channel ------------------------------------------------------------------------

class DabPlusAudio:
    """One DAB+ sub-channel's soft bits, a logical frame (24 ms) at a time, to audio."""

    def __init__(self, start_cu: int, size_cu: int, option: int, level: int) -> None:
        self.start_cu, self.size_cu = start_cu, size_cu
        l1, l2, pi1, pi2, self.bitrate = eep_profile(option, level, size_cu)
        self._pattern = eep_pattern(l1, l2, pi1, pi2)
        self.bits = 32 * (l1 + l2)                          # I: data bits per logical frame
        self._prbs = prbs(self.bits)
        self._delay = np.array(DELAYS)[np.arange(size_cu * CU_BITS) % 16]
        self._history: list[np.ndarray] = []                # the last 16 frames received
        self._frames: list[bytes] = []                      # decoded logical frames
        self.rows = self.bitrate // 8                       # s, RS rows per superframe
        self._aac: AacDecoder | None = None
        self._header: tuple | None = None
        self.superframes = 0
        self.bad_superframes = 0
        self.bad_aus = 0
        self.sample_rate = 48000.0

    def push(self, cifs: list[np.ndarray]) -> np.ndarray:
        """This frame's soft bits for the sub-channel, one array per 24 ms CIF, oldest
        first. Returns the audio decoded so far, [n, 2]."""
        ready = []
        for soft in cifs:
            self._history.append(soft)
            self._history = self._history[-16:]
            if len(self._history) < 16:
                continue
            # B_{r-15}: bit i comes from the frame received d(i) frames after it.
            stack = np.stack(self._history)                 # [16, bits], oldest first
            ready.append(stack[self._delay, np.arange(stack.shape[1])])
        if not ready:
            return np.zeros((0, 2), dtype=np.float32)
        decoded = viterbi(depuncture(np.stack(ready), self._pattern))[:, :self.bits]
        for frame in decoded:
            self._frames.append(np.packbits(frame ^ self._prbs).tobytes())
        return self._superframes()

    def _superframes(self) -> np.ndarray:
        audio = []
        while len(self._frames) >= 5:
            block = bytearray(b"".join(self._frames[:5]))
            s = self.rows
            fixed = bytearray(block)
            ok = True
            for row in range(s):
                word = bytearray(fixed[row + s * j] for j in range(120))
                if rs_decode(word) < 0:
                    ok = False
                for j in range(120):
                    fixed[row + s * j] = word[j]
            data = bytes(fixed[:110 * s])
            if firecode(data[2:11]) != int.from_bytes(data[:2], "big"):
                self._frames.pop(0)                          # not aligned yet: slide
                continue
            del self._frames[:5]
            self.superframes += 1
            if not ok:
                self.bad_superframes += 1
            audio.append(self._access_units(data))
        return np.concatenate(audio) if audio else np.zeros((0, 2), dtype=np.float32)

    def _access_units(self, sf: bytes) -> np.ndarray:
        p = sf[2]
        dac, sbr, stereo, ps = p >> 6 & 1, p >> 5 & 1, p >> 4 & 1, p >> 3 & 1
        count = {(0, 1): 2, (1, 1): 3, (0, 0): 4, (1, 0): 6}[(dac, sbr)]
        starts = [{2: 5, 3: 6, 4: 8, 6: 11}[count]]
        bits = "".join(f"{b:08b}" for b in sf[3:3 + 9])
        for k in range(count - 1):
            starts.append(int(bits[12 * k:12 * k + 12], 2))
        starts.append(len(sf))
        header = (dac, sbr, stereo, ps)
        if self._header != header or self._aac is None:
            if self._aac is not None:
                self._aac.close()
            self._aac = AacDecoder(audio_specific_config(dac, bool(sbr), bool(stereo), bool(ps)))
            self._header = header
        out = []
        for k in range(count):
            a, b = starts[k], starts[k + 1]
            if not 0 < a < b <= len(sf) or b - a < 3:
                self.bad_aus += 1
                continue
            au, crc = sf[a:b - 2], int.from_bytes(sf[b - 2:b], "big")
            if au_crc(au) != crc:
                self.bad_aus += 1
                continue
            pcm = self._aac.decode(au)
            if pcm.size:
                out.append(pcm)
        self.sample_rate = self._aac.sample_rate
        return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float32)
