"""P25 Phase 1 (TIA-102): frame sync, network ID, and trunking control blocks.

Metadata only -- no voice, which would need an IMBE codec from outside (PLANNING.md 7p).
Every frame starts with the 48-bit frame sync and a 64-bit network ID: a 12-bit NAC
(network access code) and a 4-bit DUID (what kind of frame), protected by BCH(63,16),
which corrects up to 11 bit errors. A status symbol is woven into the stream after
every 35 dibits and is dropped on reading.

Control channels send TSDUs: one to three trunking signalling blocks (TSBKs), each 96
bits -- opcode, manufacturer, 64 bits of arguments, CRC-16 -- carried by a rate 1/2
trellis code and interleaved over 98 dibits. Those give the system's identity, its
channel plan (so channel numbers become frequencies), and the voice grants: which
talkgroup, which radio, on which channel.

Settled on air, 2026-10-06 (Pluto, Melbourne): the BCH generator, built here from its
roots, is the published 6331141367235453 (octal), and every NID on four channels decoded
with no errors; the TSBK CRC is CRC-CCITT inverted; the deinterleave table runs from
received position to decoded position (the other way round, no block passed).

Pure NumPy; no Qt, no device access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .fsk4 import RepeatFilter, SyncFinder, bits_to_int, crc_ccitt, dibits_to_bits, to_dibits

FRAME_SYNC = 0x5575F5FF77FF
DUIDS = {0x0: "header", 0x3: "terminator", 0x5: "voice LDU1", 0xA: "voice LDU2",
         0x7: "TSDU", 0xC: "packet data", 0xF: "terminator with LC"}
def _symbols_for(blocks: int) -> int:
    """Symbols from the frame sync to the end of `blocks` 98-dibit blocks after the NID:
    24 of sync, 32 of NID, then the blocks, with a status symbol every 36th."""
    return next(n for n in range(24, 20000)
                if sum(1 for i in range(24, n) if i % 36 != 35) == 32 + blocks * 98)


#: Enough for a header and PDU_MAX_BLOCKS data blocks; a TSDU needs only three.
PDU_MAX_BLOCKS = 16
FRAME_SYMBOLS = _symbols_for(1 + PDU_MAX_BLOCKS)
#: Errors BCH(63,16,23) can correct.
NID_MAX_ERRORS = 11


# -- BCH(63,16) over GF(64) -------------------------------------------------------

def _bch_generator(prim: int = 0b1000011, designed_distance: int = 23) -> int:
    """Product of the minimal polynomials of alpha^1..alpha^22 over GF(2)."""
    exp, log = [0] * 126, [0] * 64
    v = 1
    for i in range(63):
        exp[i], log[v] = v, i
        v <<= 1
        if v & 64:
            v ^= prim
    for i in range(63, 126):
        exp[i] = exp[i - 63]
    g, seen = 1, set()
    for i in range(1, designed_distance):
        conj = tuple(sorted({(i << k) % 63 for k in range(6)}))
        if conj in seen:
            continue
        seen.add(conj)
        p = [1]                                   # coefficients in GF(64), lowest first
        for c in conj:
            q = [0] * (len(p) + 1)
            for k, a in enumerate(p):
                q[k + 1] ^= a
                if a:
                    q[k] ^= exp[log[a] + c]
            p = q
        m = sum(a << k for k, a in enumerate(p))  # binary, as minimal polynomials are
        r = 0
        while m:                                  # g *= m over GF(2)
            if m & 1:
                r ^= g
            g <<= 1
            m >>= 1
        g = r
    return g


BCH_GENERATOR = _bch_generator()


def bch_encode(data16: int) -> int:
    r = data16 << 47
    for i in range(62, 46, -1):
        if r >> i & 1:
            r ^= BCH_GENERATOR << (i - 47)
    return (data16 << 47) | r


_CODEWORDS: np.ndarray | None = None


def bch_decode(word63: int) -> tuple[int, int]:
    """(NAC << 4 | DUID, bit errors) of the nearest codeword. Brute force over all 65536,
    vectorised: well under a millisecond, and it corrects whatever is correctable."""
    global _CODEWORDS
    if _CODEWORDS is None:
        _CODEWORDS = np.array([bch_encode(m) for m in range(65536)], dtype=np.uint64)
    d = np.bitwise_count(_CODEWORDS ^ np.uint64(word63))
    k = int(np.argmin(d))
    return k, int(d[k])


# -- TSBK: rate 1/2 trellis ------------------------------------------------------

#: Output (two dibits) for [state][input dibit]; the next state is the input.
TRELLIS = np.array([[0x2, 0xC, 0x1, 0xF], [0xE, 0x0, 0xD, 0x3],
                    [0x9, 0x7, 0xA, 0x4], [0x5, 0xB, 0x6, 0x8]])
#: Received dibit i belongs at position DEINTERLEAVE[i].
DEINTERLEAVE = np.array([
    0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64, 65, 72, 73, 80, 81,
    88, 89, 96, 97, 2, 3, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 50, 51, 58, 59, 66, 67,
    74, 75, 82, 83, 90, 91, 4, 5, 12, 13, 20, 21, 28, 29, 36, 37, 44, 45, 52, 53, 60, 61,
    68, 69, 76, 77, 84, 85, 92, 93, 6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55,
    62, 63, 70, 71, 78, 79, 86, 87, 94, 95])
#: How far apart two dibit pairs are, in symbol-level steps: noise moves a symbol to the
#: next level (+1 read as +3), which flips one bit or two depending on the pair, so bit
#: counting misjudges it -- with it, the 3/4-rate code could not correct even single
#: errors. Index [sent ^ ...] is not enough; the table is [sent][received].
_LEVEL = {0b01: 3, 0b00: 2, 0b10: 1, 0b11: 0}
_DISTANCE = np.array([[abs(_LEVEL[a >> 2] - _LEVEL[b >> 2]) + abs(_LEVEL[a & 3] - _LEVEL[b & 3])
                       for b in range(16)] for a in range(16)])


def trellis_encode(dibits48: np.ndarray) -> np.ndarray:
    """48 data dibits -> 98 transmitted dibits (the inverse of `trellis_decode`)."""
    state, nibbles = 0, []
    for d in list(dibits48) + [0]:
        nibbles.append(TRELLIS[state, d])
        state = int(d)
    pairs = np.ravel([[n >> 2, n & 3] for n in nibbles])
    return pairs[DEINTERLEAVE]


def trellis_decode(dibits98: np.ndarray) -> tuple[np.ndarray, int]:
    """Viterbi over 49 steps of four states: 98 dibits -> 48 dibits, and the distance
    (symbol-level steps) to the nearest valid sequence. A loop over symbols, not
    samples."""
    de = np.empty(98, dtype=np.int64)
    de[DEINTERLEAVE] = dibits98
    nibbles = (de[0::2] << 2) | de[1::2]
    big = 1 << 20
    metric = np.array([0, big, big, big])
    back = []
    for r in nibbles:
        cost = metric[:, None] + _DISTANCE[TRELLIS, r]       # [state, input]
        prev = np.argmin(cost, axis=0)
        metric = cost[prev, np.arange(4)]
        back.append(prev)
    state = int(np.argmin(metric))
    errors = int(metric[state])
    out = []
    for prev in reversed(back):
        out.append(state)
        state = int(prev[state])
    return np.array(out[::-1][:48]), errors


def tsbk_crc(bits80: np.ndarray) -> int:
    return crc_ccitt(bits80) ^ 0xFFFF


# -- packet data: rate 3/4 trellis ----------------------------------------------------

#: TIA-102.BAAA's constellation points for [state][input]: 1/2 rate (dibits) and 3/4 rate
#: (tribits). The point -> dibit-pair mapping is read off the 1/2-rate table confirmed on
#: air (TRELLIS), so the 3/4 one is built from the same points. Confirmed on air
#: 2026-10-06: all 48 data blocks in a capture passed their CRC-9, every packet its
#: CRC-32.
_POINTS_12 = ((0, 15, 12, 3), (4, 11, 8, 7), (13, 2, 1, 14), (9, 6, 5, 10))
_POINTS_34 = ((0, 8, 4, 12, 2, 10, 6, 14), (4, 12, 2, 10, 6, 14, 0, 8),
              (1, 9, 5, 13, 3, 11, 7, 15), (5, 13, 3, 11, 7, 15, 1, 9),
              (3, 11, 7, 15, 1, 9, 5, 13), (7, 15, 1, 9, 5, 13, 3, 11),
              (2, 10, 6, 14, 0, 8, 4, 12), (6, 14, 0, 8, 4, 12, 2, 10))
_POINT_DIBITS = [0] * 16
for _s in range(4):
    for _d in range(4):
        _POINT_DIBITS[_POINTS_12[_s][_d]] = int(TRELLIS[_s][_d])
TRELLIS_34 = np.array([[_POINT_DIBITS[p] for p in row] for row in _POINTS_34])


def trellis34_encode(bits144: np.ndarray) -> np.ndarray:
    """144 data bits -> 98 transmitted dibits (for tests)."""
    tribits = [int(bits144[i]) << 2 | int(bits144[i + 1]) << 1 | int(bits144[i + 2])
               for i in range(0, 144, 3)]
    state, nibbles = 0, []
    for t in tribits + [0]:
        nibbles.append(TRELLIS_34[state, t])
        state = t
    pairs = np.ravel([[n >> 2, n & 3] for n in nibbles])
    return pairs[DEINTERLEAVE]


def trellis34_decode(dibits98: np.ndarray) -> np.ndarray:
    """Viterbi over eight states: 98 dibits -> 144 bits."""
    de = np.empty(98, dtype=np.int64)
    de[DEINTERLEAVE] = dibits98
    nibbles = (de[0::2] << 2) | de[1::2]
    big = 1 << 20
    metric = np.array([0] + [big] * 7)
    back = []
    for r in nibbles:
        cost = metric[:, None] + _DISTANCE[TRELLIS_34, r]
        prev = np.argmin(cost, axis=0)
        metric = cost[prev, np.arange(8)]
        back.append(prev)
    state, out = int(np.argmin(metric)), []
    for prev in reversed(back):
        out.append(state)
        state = int(prev[state])
    return np.ravel([[(t >> 2) & 1, (t >> 1) & 1, t & 1] for t in out[::-1][:48]])


def _crc(bits, width: int, poly: int) -> int:
    r, top, mask = 0, width - 1, (1 << width) - 1
    for b in bits:
        fb = ((r >> top) ^ int(b)) & 1
        r = (r << 1) & mask
        if fb:
            r ^= poly
    return r


def block_crc9(bits) -> int:
    """A confirmed data block's CRC-9 (x^9+x^6+x^4+x^3+1), inverted (on air)."""
    return _crc(bits, 9, 0x059) ^ 0x1FF


def packet_crc32(data: bytes) -> int:
    """A packet's CRC-32 (the Ethernet polynomial, MSB first), inverted (on air)."""
    return _crc(np.unpackbits(np.frombuffer(data, np.uint8)), 32, 0x04C11DB7) ^ 0xFFFFFFFF


#: What well-known UDP ports carry on Motorola P25 data (and DMR) networks.
UDP_SERVICES = {4001: "location report (LRRP)", 4005: "registration (ARS)",
                4007: "text message (TMS)", 4008: "telemetry", 4012: "over-the-air rekeying"}


#: Motorola LRRP message types (from open-source decoders; not a published standard).
LRRP_TYPES = {0x04: "location request", 0x05: "location", 0x09: "start reporting",
              0x0A: "start acknowledged", 0x0B: "location", 0x0D: "location",
              0x0F: "stop reporting", 0x10: "stop acknowledged", 0x14: "protocol request",
              0x15: "protocol response"}


def _content(body: bytes) -> str:
    """Readable runs of 4 characters or more, else the bytes in hex (48 at most)."""
    runs, current = [], []
    for c in body:
        if 32 <= c < 127:
            current.append(chr(c))
        else:
            if len(current) >= 4:
                runs.append("".join(current))
            current = []
    if len(current) >= 4:
        runs.append("".join(current))
    if runs:
        return " | ".join(r.strip() for r in runs if r.strip())
    return body[:48].hex(" ") + (" ..." if len(body) > 48 else "")


def _lrrp(body: bytes) -> tuple[str, tuple[float, float] | None]:
    """An LRRP message: its kind, request id, and a position if it carries one (the
    0x51/0x54/0x66/0x69 tokens). Layout from open-source decoders, *not yet checked on
    air*: the label says so."""
    if len(body) < 2:
        return _content(body), None
    kind = LRRP_TYPES.get(body[0], f"type {body[0]:02X}")
    parts, position = [kind], None
    i = 2
    while i < len(body):
        token = body[i]
        if token == 0x22 and i + 1 < len(body):                    # request id
            n = body[i + 1]
            parts.append(f"request {body[i + 2:i + 2 + n].hex()}")
            i += 2 + n
        elif token in (0x51, 0x54, 0x66, 0x69) and i + 9 <= len(body):
            lat = int.from_bytes(body[i + 1:i + 5], "big", signed=True) * 90.0 / 2 ** 31
            lon = int.from_bytes(body[i + 5:i + 9], "big", signed=True) * 180.0 / 2 ** 31
            parts.append(f"position {lat:.5f}, {lon:.5f} (layout unverified)")
            if abs(lat) <= 90 and abs(lon) <= 180:
                position = (lat, lon)
            break
        else:
            break
    if len(parts) == 1:
        parts.append(_content(body[2:]))
    return "  ".join(parts), position


def describe_packet(user: bytes) -> tuple[str, str, tuple[float, float] | None]:
    """(what, content) for a packet's user data: SNDCP's 2-byte header, then IPv4. The
    content is what can be shown: a text message's text, a registration's identifier, a
    location message, else readable text or the bytes in hex. Encrypted data has none.
    Third, a position (lat, lon) when the packet reports one."""
    if len(user) > 2 and user[0] >> 4 == 5 and user[2] >> 4 == 4:
        user = user[2:]
    if len(user) < 20 or user[0] >> 4 != 4:
        return f"{len(user)} bytes", _content(user), None
    ihl, proto = (user[0] & 15) * 4, user[9]
    if proto == 50:
        return "encrypted (IPsec)", "", None
    if proto != 17 or len(user) < ihl + 8:
        return f"IP protocol {proto}", _content(user[ihl:]), None
    dport = int.from_bytes(user[ihl + 2:ihl + 4], "big")
    sport = int.from_bytes(user[ihl:ihl + 2], "big")
    port = dport if dport in UDP_SERVICES else sport
    what = UDP_SERVICES.get(port, f"UDP port {dport}")
    body = user[ihl + 8:]
    if port == 4007:
        # Motorola's text messages are UTF-16LE after a short header: keep the longest
        # printable run.
        decoded = body[: len(body) // 2 * 2].decode("utf-16-le", errors="replace")
        runs = "".join(c if c.isprintable() else "\0" for c in decoded).split("\0")
        return what, max(runs, key=len, default="").strip(), None
    if port == 4001:
        return (what, *_lrrp(body))
    return what, _content(body), None


# -- TSBK contents (TIA-102.AABC) --------------------------------------------------

OPCODES = {
    0x00: "group voice grant", 0x02: "group voice grant update",
    0x03: "group voice grant update", 0x04: "unit-to-unit voice grant",
    0x06: "unit-to-unit grant update", 0x14: "data grant",
    0x16: "data channel announcement", 0x20: "acknowledge", 0x24: "extended function",
    0x27: "deny", 0x28: "group affiliation", 0x29: "secondary control channel",
    0x2B: "location registration", 0x2C: "unit registration", 0x2F: "deregistration",
    0x30: "TDMA sync", 0x33: "channel plan (TDMA)", 0x34: "channel plan (VHF/UHF)",
    0x35: "time and date", 0x38: "system services", 0x39: "secondary control channel",
    0x3A: "site status", 0x3B: "network status", 0x3C: "adjacent site",
    0x3D: "channel plan",
}
MANUFACTURERS = {0x00: "", 0x01: "", 0x90: "Motorola", 0xA4: "Harris", 0xD8: "Tait"}
#: Slots per carrier for each TDMA channel type (IDEN_UP_TDMA): 2-slot types are the
#: Phase 2 ones.
TDMA_SLOTS = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}


@dataclass
class ChannelPlan:
    base_hz: float
    spacing_hz: float
    slots: int = 1

    def frequency(self, number: int) -> float:
        return self.base_hz + (number // self.slots) * self.spacing_hz


@dataclass
class P25Message:
    nac: int
    kind: str                     # "TSBK", or a frame name for other traffic
    text: str
    fields: dict = field(default_factory=dict)
    channel: str = ""
    received: float = field(default_factory=time.time)
    #: What a data packet carried (text, a location, bytes): shown unless hidden.
    content: str = field(default="", repr=False)
    #: (lat, lon) a radio reported, for the map; and which radio (its LLID).
    position: tuple[float, float] | None = field(default=None, repr=False)
    radio: int | None = None

    def summary(self, show_text: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.received))
        line = f"{stamp}  P25  NAC {self.nac:03X}  {self.text}"
        if self.content:
            line += f"  {self.content}" if show_text else f"  [{len(self.content)} characters hidden]"
        return line


def _field(bits: np.ndarray, start: int, n: int) -> int:
    return bits_to_int(bits[start:start + n])


class P25Decoder:
    """FM discriminator samples in, P25 messages out."""

    name = "P25"

    def __init__(self, sample_rate: float, channel: str = "") -> None:
        self.sample_rate = float(sample_rate)
        self.channel = channel
        self._finder = SyncFinder(sample_rate, {"fs": FRAME_SYNC}, FRAME_SYMBOLS)
        self.plans: dict[int, ChannelPlan] = {}
        self.frames = 0
        self.bad_frames = 0
        self.bad_blocks = 0
        self._repeats = RepeatFilter()
        self.reset()

    def reset(self) -> None:
        self._finder.reset()

    def process(self, x: np.ndarray) -> list[P25Message]:
        out = []
        for _hit, symbols in self._finder.process(x):
            dibits = to_dibits(symbols)
            # The status symbol at dibit 35 sits inside the NID.
            nid_bits = dibits_to_bits(np.concatenate([dibits[24:35], dibits[36:57]]))
            value, errors = bch_decode(bits_to_int(nid_bits[:63]))
            if errors > NID_MAX_ERRORS:
                self.bad_frames += 1
                continue
            self.frames += 1
            nac, duid = value >> 4, value & 0xF
            payload = np.array([dibits[i] for i in range(57, dibits.size) if i % 36 != 35])
            if duid == 0x7:
                messages = self._tsdu(nac, payload)
            elif duid == 0xC:
                messages = self._pdu(nac, payload)
            else:
                name = DUIDS.get(duid, f"DUID {duid:X}")
                kind = "voice" if duid in (0x0, 0x5, 0xA) else name
                messages = [P25Message(nac, kind, "voice" if kind == "voice" else name,
                                       {"duid": duid})]
            for m in messages:
                m.channel = self.channel
                if self._repeats.fresh(m.text + "|" + m.content):
                    out.append(m)
        return out

    def _tsdu(self, nac: int, payload: np.ndarray) -> list[P25Message]:
        out = []
        for k in range(3):
            block = payload[98 * k: 98 * (k + 1)]
            if block.size < 98:
                break
            data, _ = trellis_decode(block)
            bits = dibits_to_bits(data)
            if tsbk_crc(bits[:80]) != _field(bits, 80, 16):
                self.bad_blocks += 1
                break
            message = self._tsbk(nac, bits)
            if message is not None:
                out.append(message)
            if bits[0]:                                 # last block
                break
        return out

    def _pdu(self, nac: int, payload: np.ndarray) -> list[P25Message]:
        """A data packet: its header (coded as a TSBK is), then confirmed data blocks of
        16 bytes (rate 3/4, each with a serial number and CRC-9) and a CRC-32 over all."""
        data, _ = trellis_decode(payload[:98])
        head = dibits_to_bits(data)
        if tsbk_crc(head[:80]) != _field(head, 80, 16):
            self.bad_blocks += 1
            return []
        outbound, fmt = bool(head[2]), _field(head, 3, 5)
        sap, llid = _field(head, 10, 6), _field(head, 24, 24)
        blocks, pad = _field(head, 49, 7), _field(head, 59, 5)
        f = {"format": fmt, "sap": sap, "llid": llid, "outbound": outbound}
        way = "to" if outbound else "from"
        if fmt == 3:
            return [P25Message(nac, "data", f"data acknowledgement {way} {llid}", f)]
        if fmt != 22:
            return [P25Message(nac, "data", f"packet data {way} {llid} (format {fmt})", f)]
        chunks = []
        for k in range(blocks):
            block = payload[98 * (k + 1): 98 * (k + 2)]
            if block.size < 98:
                return [P25Message(nac, "data", f"packet data {way} {llid}, "
                                   f"{blocks} blocks (too long to read)", f)]
            bits = trellis34_decode(block)
            if block_crc9(np.concatenate([bits[:7], bits[16:]])) != _field(bits, 7, 9):
                self.bad_blocks += 1
                return []
            chunks.append(np.packbits(bits[16:]).tobytes())
        packet = b"".join(chunks)
        if len(packet) < pad + 4 or packet_crc32(packet[:-4]) != int.from_bytes(packet[-4:], "big"):
            self.bad_blocks += 1
            return []
        what, content, position = describe_packet(packet[:len(packet) - pad - 4])
        f.update(service=what)
        return [P25Message(nac, "data", f"packet data {way} {llid}: {what}", f,
                           content=content, position=position, radio=llid)]

    def _channel(self, value: int) -> str:
        iden, number = value >> 12, value & 0xFFF
        plan = self.plans.get(iden)
        text = f"ch {iden}-{number}"
        if plan is not None:
            text += f" ({plan.frequency(number) / 1e6:.5f} MHz"
            text += f", slot {number % plan.slots + 1})" if plan.slots > 1 else ")"
        return text

    def _tsbk(self, nac: int, bits: np.ndarray) -> P25Message | None:
        op, mfid = _field(bits, 2, 6), _field(bits, 8, 8)
        a = bits[16:80]
        f: dict = {"opcode": op, "mfid": mfid}
        if mfid not in (0x00, 0x01):
            maker = MANUFACTURERS.get(mfid, f"manufacturer {mfid:02X}")
            return P25Message(nac, "TSBK", f"{maker} opcode {op:02X}", f)
        name = OPCODES.get(op, f"opcode {op:02X}")
        text = name
        if op == 0x00:
            f.update(channel=_field(a, 8, 16), group=_field(a, 24, 16),
                     source=_field(a, 40, 24))
            text += (f"  TG {f['group']}  from {f['source']}  "
                     f"{self._channel(f['channel'])}")
        elif op == 0x02:
            f.update(channel=_field(a, 0, 16), group=_field(a, 16, 16),
                     channel_b=_field(a, 32, 16), group_b=_field(a, 48, 16))
            text += f"  TG {f['group']} {self._channel(f['channel'])}"
            if f["group_b"] != f["group"]:
                text += f"; TG {f['group_b']} {self._channel(f['channel_b'])}"
        elif op == 0x03:
            f.update(channel=_field(a, 16, 16), group=_field(a, 48, 16))
            text += f"  TG {f['group']} {self._channel(f['channel'])}"
        elif op in (0x04, 0x06):
            f.update(channel=_field(a, 0, 16), target=_field(a, 16, 24),
                     source=_field(a, 40, 24))
            text += (f"  to {f['target']}  from {f['source']}  "
                     f"{self._channel(f['channel'])}")
        elif op in (0x33, 0x34, 0x3D):
            iden = _field(a, 0, 4)
            spacing = _field(a, 22, 10) * 125.0
            base = _field(a, 32, 32) * 5.0
            slots = TDMA_SLOTS.get(_field(a, 4, 4), 1) if op == 0x33 else 1
            self.plans[iden] = ChannelPlan(base, spacing, slots)
            f.update(iden=iden, base_hz=base, spacing_hz=spacing, slots=slots)
            text += (f"  {iden}: from {base / 1e6:.5f} MHz every {spacing / 1e3:g} kHz"
                     + (f", {slots} slots" if slots > 1 else ""))
        elif op in (0x3A, 0x3C):
            f.update(system=_field(a, 12, 12), rfss=_field(a, 24, 8), site=_field(a, 32, 8),
                     channel=_field(a, 40, 16))
            text += (f"  system {f['system']:03X}  RFSS {f['rfss']}  site {f['site']}  "
                     f"{self._channel(f['channel'])}")
        elif op == 0x3B:
            f.update(wacn=_field(a, 8, 20), system=_field(a, 28, 12),
                     channel=_field(a, 40, 16))
            text += (f"  WACN {f['wacn']:05X}  system {f['system']:03X}  "
                     f"{self._channel(f['channel'])}")
        elif op == 0x28:
            f.update(group=_field(a, 24, 16), target=_field(a, 40, 24))
            text += f"  {f['target']} to TG {f['group']}"
        elif op == 0x2C:
            f.update(system=_field(a, 4, 12), source=_field(a, 16, 24))
            text += f"  {f['source']}"
        return P25Message(nac, "TSBK", text, f)
