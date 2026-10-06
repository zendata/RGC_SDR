"""DMR (ETSI TS 102 361): bursts, slot types, control blocks and call headers.

Metadata only -- no voice, which would need an AMBE+2 codec from outside (PLANNING.md
7p). DMR is two-slot TDMA: a repeater sends a 30 ms burst for each slot in turn, each
132 symbols with a 24-symbol sync in the middle, and ahead of it 12 symbols of CACH that
say which slot the burst belongs to. A data sync's negative is the matching voice sync,
so one correlation finds both: positive for data and control, negative for voice.

Around the sync, a data burst carries a 20-bit slot type (colour code and data type,
Golay(20,8)) and 196 bits of payload, BPTC(196,96) coded. The payloads read here:
- CSBK (control): opcode, manufacturer, 64 bits, CRC-CCITT;
- voice LC header and terminator: the full link control -- group or private call,
  talkgroup or destination, source -- with Reed-Solomon (12,9) parity.

Settled on air, 2026-10-06 (Pluto, Melbourne), over four channels: the slot type is
the extended Golay(24,12) code (generator 0xC75) shortened to 8 data bits; the BPTC
deinterleave takes position i from received (i * 181) mod 196, and its Hamming
row and column parities held on every clean block; the CSBK CRC is inverted
and XORed with 0xA5A5; the LC parity is RS(12,9) over GF(256) (x^8+x^4+x^3+x^2+1),
roots alpha^1..alpha^3, masked with 0x969696 (header) and 0x999999 (terminator); and
the CACH slot bit alternated on all 997 bursts in a row of a continuous channel, its
Hamming(7,4) parities holding on 99.9 %.

Pure NumPy; no Qt, no device access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .fsk4 import RepeatFilter, SyncFinder, bits_to_int, crc_ccitt, dibits_to_bits, to_dibits

#: Data syncs; each one's negative is the voice sync of the same source.
SYNCS = {"bs": 0xDFF57D75DF5D, "ms": 0xD5D7F77FD757,
         "dm1": 0xF7FDD5DDFD55, "dm2": 0xD7557F5FF7F5}
#: CACH (12) and the first payload half and slot type (54) ahead of the sync.
LEAD_SYMBOLS = 66
BURST_SYMBOLS = 144
DATA_TYPES = {0: "privacy header", 1: "voice header", 2: "terminator", 3: "CSBK",
              4: "MBC header", 5: "MBC continuation", 6: "data header",
              7: "rate 1/2 data", 8: "rate 3/4 data", 9: "idle", 10: "rate 1 data"}
CSBK_OPCODES = {0x04: "unit-to-unit voice request", 0x05: "unit-to-unit answer",
                0x07: "channel timing", 0x19: "Aloha", 0x1C: "Ahoy", 0x28: "announcement",
                0x30: "private voice grant", 0x31: "talkgroup voice grant",
                0x32: "broadcast voice grant", 0x33: "private data grant",
                0x34: "talkgroup data grant", 0x38: "BS outbound activation",
                0x3D: "preamble"}
MANUFACTURERS = {0x00: "", 0x10: "Motorola", 0x68: "Hytera", 0x08: "Hytera"}
#: CACH bits holding the TACT word (AT, TC, LCSS x2, three Hamming parities).
TACT_POSITIONS = [0, 4, 8, 12, 14, 18, 22]
#: Golay(20,8) corrects up to three errors.
SLOT_TYPE_MAX_ERRORS = 3


# -- Golay(20,8): the extended Golay(24,12) shortened by four --------------------

def golay20_encode(data8: int) -> int:
    r = data8 << 11
    for i in range(18, 10, -1):
        if r >> i & 1:
            r ^= 0xC75 << (i - 11)
    w = (data8 << 11) | r
    return (w << 1) | (bin(w).count("1") & 1)


_GOLAY = np.array([golay20_encode(d) for d in range(256)], dtype=np.uint32)


def golay20_decode(word: int) -> tuple[int, int]:
    d = np.bitwise_count(_GOLAY ^ np.uint32(word))
    k = int(np.argmin(d))
    return k, int(d[k])


# -- Hamming(7,4) for the TACT ----------------------------------------------------

def tact_encode(data4: int) -> int:
    d = [(data4 >> (3 - i)) & 1 for i in range(4)]
    p = [d[0] ^ d[1] ^ d[2], d[1] ^ d[2] ^ d[3], d[0] ^ d[1] ^ d[3]]
    return bits_to_int(d + p)


_TACT = {tact_encode(d): d for d in range(16)}
for _d in range(16):
    for _b in range(7):
        _TACT.setdefault(tact_encode(_d) ^ (1 << _b), _d)       # one error corrected


# -- BPTC(196,96) -------------------------------------------------------------------

_BPTC_FROM = (np.arange(196) * 181) % 196
_BPTC_ROWS = np.r_[np.arange(3, 11), (np.arange(1, 9)[:, None] * 15 + np.arange(11)).ravel()]


#: Parity equations, as data positions, of the Hamming(15,11) rows and the Hamming(13,9)
#: columns. Read off 1622 clean control blocks on air, where each held every time; the
#: row equations are also those published by open DMR decoders.
_ROW_PARITY = [(0, 1, 2, 3, 5, 7, 8), (1, 2, 3, 4, 6, 8, 9), (2, 3, 4, 5, 7, 9, 10),
               (0, 1, 2, 4, 6, 7, 10)]
_COL_PARITY = [(0, 1, 3, 5, 6), (0, 1, 2, 4, 6, 7), (0, 1, 2, 3, 5, 7, 8), (0, 2, 4, 5, 8)]


def _check_matrix(parity: list[tuple[int, ...]], k: int) -> np.ndarray:
    h = np.zeros((len(parity), k + len(parity)), dtype=np.int64)
    for j, positions in enumerate(parity):
        h[j, list(positions)] = 1
        h[j, k + j] = 1
    return h


_H_ROW = _check_matrix(_ROW_PARITY, 11)
_H_COL = _check_matrix(_COL_PARITY, 9)


def _correct(words: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Fix one error in each row of `words` whose syndrome names a position."""
    weights = 1 << np.arange(h.shape[0])
    position = {int(s): i for i, s in enumerate(weights @ h)}
    syndromes = (words @ h.T % 2) @ weights
    out = words.copy()
    for r in np.flatnonzero(syndromes):
        i = position.get(int(syndromes[r]))
        if i is not None:
            out[r, i] ^= 1
    return out


def bptc_data(info196: np.ndarray) -> np.ndarray:
    """196 received payload bits -> the 96 data bits, after correcting single errors
    in each row and column, twice over. The CRC or Reed-Solomon check after it decides
    whether the result stands."""
    de = np.asarray(info196, dtype=np.int64)[_BPTC_FROM]
    m = de[1:].reshape(13, 15)
    for _ in range(2):
        m = _correct(m, _H_ROW)
        m = _correct(m.T, _H_COL).T
    return m.ravel()[_BPTC_ROWS].astype(np.uint8)


def bptc_encode(data96: np.ndarray) -> np.ndarray:
    """96 data bits -> 196 transmitted bits, parities included (for tests)."""
    m = np.zeros(195, dtype=np.int64)
    m[_BPTC_ROWS] = data96
    m = m.reshape(13, 15)
    for j, positions in enumerate(_ROW_PARITY):
        m[:9, 11 + j] = m[:9, list(positions)].sum(axis=1) % 2
    for j, positions in enumerate(_COL_PARITY):
        m[9 + j, :] = m[list(positions), :].sum(axis=0) % 2
    de = np.concatenate([[0], m.ravel()])
    out = np.zeros(196, dtype=np.uint8)
    out[_BPTC_FROM] = de
    return out


# -- Reed-Solomon (12,9) over GF(256) ------------------------------------------------

_EXP = [0] * 512
_LOG = [0] * 256
_v = 1
for _i in range(255):
    _EXP[_i], _LOG[_v] = _v, _i
    _v <<= 1
    if _v & 0x100:
        _v ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _gmul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


_RS_G = [1]
for _i in range(1, 4):
    _g = [0] * (len(_RS_G) + 1)
    for _k, _c in enumerate(_RS_G):
        _g[_k] ^= _c
        _g[_k + 1] ^= _gmul(_c, _EXP[_i])
    _RS_G = _g

LC_MASKS = {1: 0x969696, 2: 0x999999}


def rs129_parity(data9: list[int]) -> int:
    reg = [0, 0, 0]
    for d in data9:
        fb = d ^ reg[0]
        reg = [reg[1] ^ _gmul(fb, _RS_G[1]), reg[2] ^ _gmul(fb, _RS_G[2]), _gmul(fb, _RS_G[3])]
    return reg[0] << 16 | reg[1] << 8 | reg[2]


def csbk_crc(bits80: np.ndarray) -> int:
    return crc_ccitt(bits80) ^ 0xFFFF ^ 0xA5A5


# -- Messages ----------------------------------------------------------------------

@dataclass
class DmrMessage:
    colour_code: int | None
    slot: int | None
    kind: str
    text: str
    fields: dict = field(default_factory=dict)
    channel: str = ""
    received: float = field(default_factory=time.time)

    def summary(self, show_text: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.received))
        where = []
        if self.colour_code is not None:
            where.append(f"CC {self.colour_code}")
        if self.slot is not None:
            where.append(f"slot {self.slot}")
        return f"{stamp}  DMR  {'  '.join(where + [self.text])}"


def _field(bits: np.ndarray, start: int, n: int) -> int:
    return bits_to_int(bits[start:start + n])


def parse_csbk(bits: np.ndarray) -> tuple[str, dict]:
    op, fid = _field(bits, 2, 6), _field(bits, 8, 8)
    f = {"opcode": op, "fid": fid}
    if fid != 0:
        maker = MANUFACTURERS.get(fid, f"manufacturer {fid:02X}")
        return f"{maker} CSBK {op:02X}", f
    text = CSBK_OPCODES.get(op, f"CSBK {op:02X}")
    if op == 0x3D:
        f.update(group=bool(bits[17]), target=_field(bits, 32, 24), source=_field(bits, 56, 24))
        text += (f"  {'TG' if f['group'] else 'to'} {f['target']}  from {f['source']}")
    return text, f


def parse_lc(bits: np.ndarray) -> tuple[str, dict]:
    flco, fid = _field(bits, 2, 6), _field(bits, 8, 8)
    options = _field(bits, 16, 8)
    f = {"flco": flco, "fid": fid, "options": options,
         "target": _field(bits, 24, 24), "source": _field(bits, 48, 24)}
    secret = "  encrypted" if options & 0x40 else ""
    if fid == 0 and flco == 0:
        return f"group call  TG {f['target']}  from {f['source']}{secret}", f
    if fid == 0 and flco == 3:
        return f"private call  to {f['target']}  from {f['source']}{secret}", f
    maker = MANUFACTURERS.get(fid, f"manufacturer {fid:02X}")
    return f"{maker} link control {flco:02X}".strip(), f


class DmrDecoder:
    """FM discriminator samples in, DMR messages out."""

    name = "DMR"

    def __init__(self, sample_rate: float, channel: str = "") -> None:
        self.sample_rate = float(sample_rate)
        self.channel = channel
        self._finder = SyncFinder(sample_rate, SYNCS, BURST_SYMBOLS, LEAD_SYMBOLS,
                                  upright=False)
        self.bursts = 0
        self.bad_bursts = 0
        self._repeats = RepeatFilter()
        self.reset()

    def reset(self) -> None:
        self._finder.reset()

    def process(self, x: np.ndarray) -> list[DmrMessage]:
        out = []
        for hit, symbols in self._finder.process(x):
            bits = dibits_to_bits(to_dibits(symbols))
            message = self._burst(hit.name, hit.sign, bits)
            if message is None:
                continue
            message.channel = self.channel
            if self._repeats.fresh(message.summary()[10:]):
                out.append(message)
        return out

    def _slot(self, source: str, bits: np.ndarray) -> int | None:
        if source == "dm1":
            return 1
        if source == "dm2":
            return 2
        if source != "bs":
            return None
        tact = _TACT.get(bits_to_int(bits[TACT_POSITIONS]))
        return None if tact is None else (tact >> 2 & 1) + 1

    def _burst(self, source: str, sign: int, bits: np.ndarray) -> DmrMessage | None:
        slot = self._slot(source, bits)
        if sign < 0:
            self.bursts += 1
            return DmrMessage(None, slot, "voice", "voice")
        value, errors = golay20_decode(_field(bits, 122, 10) << 10 | _field(bits, 180, 10))
        if errors > SLOT_TYPE_MAX_ERRORS:
            self.bad_bursts += 1
            return None
        self.bursts += 1
        cc, dtype = value >> 4, value & 0xF
        name = DATA_TYPES.get(dtype, f"data type {dtype}")
        if dtype == 9:
            return None                                   # idle: nothing to say
        payload = bptc_data(np.concatenate([bits[24:122], bits[190:288]]))
        if dtype == 3:
            if csbk_crc(payload[:80]) != _field(payload, 80, 16):
                self.bad_bursts += 1
                return None
            text, f = parse_csbk(payload)
            return DmrMessage(cc, slot, "CSBK", text, f)
        if dtype in LC_MASKS:
            data = [_field(payload, 8 * i, 8) for i in range(12)]
            parity = data[9] << 16 | data[10] << 8 | data[11]
            if rs129_parity(data[:9]) ^ LC_MASKS[dtype] != parity:
                self.bad_bursts += 1
                return None
            text, f = parse_lc(payload)
            return DmrMessage(cc, slot, name, f"{name}  {text}", f)
        return DmrMessage(cc, slot, name, name, {"data_type": dtype})
