"""The IC-705's memory channels (CI-V 1A 00), decoded and encoded.

Layout from Icom's CI-V reference guide ("Memory content"), checked against VK3RQ's
radio 2026-09-28. After `1A 00`, group (2 BCD bytes: 0000-0099, 0100 = call channels)
and channel (2 BCD bytes), a channel is 111 bytes:

    split/select (1) | receive side (47) | transmit side (47) | name (16 ASCII)

Each side: frequency (5, BCD low byte first), mode + filter (2), data mode (1),
duplex/tone (1: high nibble 0 off / 1 DUP- / 2 DUP+, low nibble the tone mode),
digital squelch (1), tone (3), TSQL tone (3), DTCS (3: polarity, code), DV code
squelch (1), duplex offset (3, BCD in 100 Hz, low byte first), UR/R1/R2 call signs
(3 x 8 ASCII). A blank channel answers with a single FF.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from . import civ

CALL_GROUP = 100
SIDE_LEN = 47
CHANNEL_LEN = 1 + 2 * SIDE_LEN + 16

#: Tone modes. The guide lists 0-3; the radio also stores 4 on a channel set to transmit
#: DTCS only. 4-7 are taken to follow the TONE key's order (DTCS(T), TONE(T)/DTCS(R),
#: DTCS(T)/TSQL(R), TONE(T)/TSQL(R)) -- inferred, not documented.
TONE_MODES = {0: "OFF", 1: "TONE", 2: "TSQL", 3: "DTCS", 4: "DTCS(T)",
              5: "TONE(T)/DTCS(R)", 6: "DTCS(T)/TSQL(R)", 7: "TONE(T)/TSQL(R)"}
DUPLEX = {0: "", 1: "DUP−", 2: "DUP+"}


def _bcd(data: bytes) -> int:
    return int(data.hex())


def _to_bcd(value: int, nbytes: int) -> bytes:
    return bytes.fromhex(f"{int(value):0{2 * nbytes}d}")


def _text(data: bytes) -> str:
    return data.decode("ascii", "replace").rstrip()


def _pad(text: str, n: int) -> bytes:
    return text.encode("ascii", "replace")[:n].ljust(n, b" ")


@dataclass
class MemorySide:
    """The receive or transmit half of a channel."""

    freq_hz: int = 0
    mode: str = "fm"
    filter: int = 1
    data: bool = False
    duplex: int = 0
    tone_mode: int = 0
    dsql: int = 0
    tone_hz: float = 88.5
    tsql_hz: float = 88.5
    dtcs: str = "023"
    dtcs_polarity: int = 0
    csql: int = 0
    offset_hz: int = 0
    ur: str = "CQCQCQ"
    r1: str = ""
    r2: str = ""

    @classmethod
    def decode(cls, b: bytes) -> "MemorySide":
        mode = civ.MODES.get(b[5], "dv" if b[5] == 0x17 else f"{b[5]:02x}")
        return cls(
            freq_hz=civ.decode_freq(b[0:5]), mode=mode, filter=b[6], data=bool(b[7]),
            duplex=b[8] >> 4, tone_mode=b[8] & 0x0F, dsql=b[9] >> 4,
            tone_hz=_bcd(b[10:13]) / 10, tsql_hz=_bcd(b[13:16]) / 10,
            dtcs=f"{_bcd(b[17:19]):03d}", dtcs_polarity=b[16], csql=b[19],
            offset_hz=civ.decode_freq(b[20:23]) * 100,
            ur=_text(b[23:31]), r1=_text(b[31:39]), r2=_text(b[39:47]))

    def encode(self) -> bytes:
        mode = civ.MODE_CODES.get(self.mode, 0x17 if self.mode == "dv" else 0x05)
        out = (civ.encode_freq(self.freq_hz) + bytes([mode, self.filter, int(self.data),
               (self.duplex << 4) | self.tone_mode, self.dsql << 4])
               + _to_bcd(round(self.tone_hz * 10), 3) + _to_bcd(round(self.tsql_hz * 10), 3)
               + bytes([self.dtcs_polarity]) + _to_bcd(int(self.dtcs), 2) + bytes([self.csql])
               + civ.encode_freq(self.offset_hz // 100, nbytes=3)
               + _pad(self.ur, 8) + _pad(self.r1, 8) + _pad(self.r2, 8))
        assert len(out) == SIDE_LEN
        return out

    @property
    def tone_text(self) -> str:
        """What the radio's screen would show: "TSQL 91.5", "DTCS 023", ""."""
        mode = TONE_MODES.get(self.tone_mode, f"tone {self.tone_mode}")
        if self.tone_mode == 0:
            return ""
        if self.tone_mode == 1:
            return f"TONE {self.tone_hz:.1f}"
        if self.tone_mode == 2:
            return f"TSQL {self.tsql_hz:.1f}"
        if self.tone_mode in (3, 4):
            return f"{mode} {self.dtcs}"
        return mode


@dataclass
class MemoryChannel:
    group: int
    channel: int
    rx: MemorySide = field(default_factory=MemorySide)
    tx: MemorySide = field(default_factory=MemorySide)
    name: str = ""
    split: bool = False
    select: int = 0

    @property
    def label(self) -> str:
        """The channel as the radio names it: "00-12", or "144 C1" for a call channel."""
        if self.group == CALL_GROUP:
            return ("144 C1", "144 C2", "430 C1", "430 C2")[self.channel] \
                if self.channel < 4 else f"C{self.channel}"
        return f"{self.group:02d}-{self.channel:02d}"

    @classmethod
    def decode(cls, group: int, channel: int, b: bytes) -> "MemoryChannel | None":
        """From the 111 bytes after group and channel; None for a blank channel."""
        if len(b) < CHANNEL_LEN:
            return None
        return cls(group, channel, split=bool(b[0] & 0x0F), select=b[0] >> 4,
                   rx=MemorySide.decode(b[1:1 + SIDE_LEN]),
                   tx=MemorySide.decode(b[1 + SIDE_LEN:1 + 2 * SIDE_LEN]),
                   name=_text(b[1 + 2 * SIDE_LEN:CHANNEL_LEN]))

    def encode(self) -> bytes:
        return (bytes([(self.select << 4) | int(self.split)]) + self.rx.encode()
                + self.tx.encode() + _pad(self.name, 16))

    @classmethod
    def simplex(cls, group: int, channel: int, side: MemorySide, name: str = "") \
            -> "MemoryChannel":
        """A new channel with split off: the guide asks for the same data on both sides."""
        return cls(group, channel, rx=side, tx=replace(side), name=name)


def address(group: int, channel: int) -> bytes:
    """The group and channel as the radio takes them: 00 00 00 12 for 00-12."""
    return _to_bcd(group, 2) + _to_bcd(channel, 2)
