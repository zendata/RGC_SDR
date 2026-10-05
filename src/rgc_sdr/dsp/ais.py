"""AIS ship reports (ITU-R M.1371), from the NBFM discriminator.

GMSK at 9600 baud with +/-2.4 kHz deviation, so an FM discriminator recovers the bits
and the GMSK filtering only softens the edges. The framing is HDLC exactly as in AX.25
(NRZI, flags, bit stuffing, the same CRC-16/X.25), so `aprs.HdlcFramer` is shared. The
one twist: AIS fields run most-significant-bit first, while HDLC sends each byte least
significant bit first, so every byte's bits are reversed back.

Positions (types 1-3, 18), base stations (4), names and voyages (5, 24) and aids to
navigation (21) are read; every message is also given as the standard `!AIVDM`
sentence, which chart plotters understand.

Pure NumPy; no Qt, no device access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .aprs import HdlcFramer
from .bitsync import BitSlicer

BAUD = 9600
#: AIS 1 and AIS 2, worldwide.
CHANNELS = {"A": 161.975e6, "B": 162.025e6}
#: The shortest AIS message (type 10, 72 bits) and its FCS.
MIN_AIS_BYTES = 11
#: A burst lasts 27 ms, so the DC (the carrier's offset) must be found within its
#: 24-bit training sequence. Bit stuffing caps a steady level at 6 bits, so this fast
#: a tracker does not follow the data.
DC_BITS = 16.0

NAV_STATUS = (
    "under way using engine", "at anchor", "not under command",
    "restricted manoeuvrability", "constrained by draught", "moored", "aground",
    "engaged in fishing", "under way sailing", "", "", "", "", "", "AIS-SART active", "",
)
AID_TYPES = {1: "reference point", 3: "fixed light", 5: "light", 6: "light",
             9: "beacon", 20: "cardinal mark N", 21: "cardinal mark E",
             22: "cardinal mark S", 23: "cardinal mark W", 24: "port hand mark",
             25: "starboard hand mark", 30: "special mark"}


# -- bit fields --------------------------------------------------------------------


def _uint(bits: np.ndarray, start: int, n: int) -> int | None:
    if start + n > bits.size:
        return None
    return int("".join("1" if b else "0" for b in bits[start:start + n]) or "0", 2)


def _sint(bits: np.ndarray, start: int, n: int) -> int | None:
    v = _uint(bits, start, n)
    if v is None:
        return None
    return v - (1 << n) if v >> (n - 1) else v


def _text(bits: np.ndarray, start: int, n: int) -> str:
    """AIS 6-bit text: 0-31 are '@'-'_', 32-63 are ' '-'?'; '@' pads."""
    out = []
    for i in range(start, min(start + n, bits.size - 5), 6):
        v = _uint(bits, i, 6)
        out.append(chr(v + 64) if v < 32 else chr(v))
    return "".join(out).split("@")[0].strip()


def _position(bits: np.ndarray, lon_at: int, lat_at: int) -> tuple[float, float] | None:
    lon, lat = _sint(bits, lon_at, 28), _sint(bits, lat_at, 27)
    if lon is None or lat is None:
        return None
    lon, lat = lon / 600000.0, lat / 600000.0
    if abs(lon) > 180 or abs(lat) > 90:        # 181 and 91 mean "not available"
        return None
    return lat, lon


def nmea(bits: np.ndarray, channel: str = "A") -> list[str]:
    """The message as `!AIVDM` sentences: 6-bit armoured, split to stay under the NMEA
    0183 sentence length."""
    fill = -bits.size % 6
    padded = np.concatenate([bits, np.zeros(fill, dtype=bits.dtype)])
    chars = []
    for i in range(0, padded.size, 6):
        v = _uint(padded, i, 6) + 48
        chars.append(chr(v + 8 if v > 87 else v))
    payload = "".join(chars)
    parts = [payload[i:i + 60] for i in range(0, len(payload), 60)] or [""]
    out = []
    for n, part in enumerate(parts, 1):
        body = (f"AIVDM,{len(parts)},{n},{'1' if len(parts) > 1 else ''},{channel},"
                f"{part},{fill if n == len(parts) else 0}")
        check = 0
        for c in body:
            check ^= ord(c)
        out.append(f"!{body}*{check:02X}")
    return out


# -- messages ----------------------------------------------------------------------


@dataclass
class AisMessage:
    msg_type: int
    mmsi: int
    fields: dict
    sentences: list[str]
    channel: str = ""
    name: str = ""                     # from an earlier type 5 or 24, when known
    received: float = field(default_factory=time.time)

    @property
    def position(self) -> tuple[float, float] | None:
        return self.fields.get("position")

    def summary(self, show_text: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.received))
        f = self.fields
        who = f"{self.mmsi:09d}" + (f" {self.name}" if self.name else "")
        parts = [f"{stamp}  AIS{self.channel}  {who}  type {self.msg_type}"]
        if self.position:
            parts.append(f"{self.position[0]:.5f}, {self.position[1]:.5f}")
        if f.get("sog") is not None:
            parts.append(f"{f['sog']:.1f} kn")
        if f.get("cog") is not None:
            parts.append(f"COG {f['cog']:.1f}")
        if f.get("heading") is not None:
            parts.append(f"HDG {f['heading']}")
        for key in ("status", "callsign", "destination", "ship_type", "aid_type", "time"):
            if f.get(key):
                parts.append(str(f[key]))
        return "  ".join(parts) + "\n    " + "  ".join(self.sentences)


def parse(bits: np.ndarray) -> tuple[int, int, dict] | None:
    """(type, MMSI, fields) of an AIS message's bits, most significant first."""
    msg_type, mmsi = _uint(bits, 0, 6), _uint(bits, 8, 30)
    if msg_type is None or mmsi is None or not 1 <= msg_type <= 27:
        return None
    f: dict = {}
    if msg_type in (1, 2, 3):
        status = _uint(bits, 38, 4)
        f["status"] = NAV_STATUS[status] if status is not None else ""
        sog, cog, hdg = _uint(bits, 50, 10), _uint(bits, 116, 12), _uint(bits, 128, 9)
        f["position"] = _position(bits, 61, 89)
        f["sog"] = sog / 10 if sog is not None and sog != 1023 else None
        f["cog"] = cog / 10 if cog is not None and cog < 3600 else None
        f["heading"] = hdg if hdg is not None and hdg != 511 else None
        f["second"] = _uint(bits, 137, 6)
    elif msg_type == 18:
        sog, cog, hdg = _uint(bits, 46, 10), _uint(bits, 112, 12), _uint(bits, 124, 9)
        f["position"] = _position(bits, 57, 85)
        f["sog"] = sog / 10 if sog is not None and sog != 1023 else None
        f["cog"] = cog / 10 if cog is not None and cog < 3600 else None
        f["heading"] = hdg if hdg is not None and hdg != 511 else None
    elif msg_type == 4:
        y, mo, d = _uint(bits, 38, 14), _uint(bits, 52, 4), _uint(bits, 56, 5)
        h, mi, s = _uint(bits, 61, 5), _uint(bits, 66, 6), _uint(bits, 72, 6)
        if None not in (y, mo, d, h, mi, s) and y:
            f["time"] = f"base station {y:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}:{s:02d} UTC"
        f["position"] = _position(bits, 79, 107)
    elif msg_type == 5:
        f["callsign"] = _text(bits, 70, 42)
        f["name"] = _text(bits, 112, 120)
        f["ship_type"] = f"ship type {_uint(bits, 232, 8)}"
        f["destination"] = _text(bits, 302, 120)
        if f["destination"]:
            f["destination"] = "to " + f["destination"]
    elif msg_type == 24:
        part = _uint(bits, 38, 2)
        if part == 0:
            f["name"] = _text(bits, 40, 120)
        elif part == 1:
            f["ship_type"] = f"ship type {_uint(bits, 40, 8)}"
            f["callsign"] = _text(bits, 90, 42)
    elif msg_type == 21:
        f["aid_type"] = AID_TYPES.get(_uint(bits, 38, 5) or 0, "aid to navigation")
        f["name"] = _text(bits, 43, 120)
        f["position"] = _position(bits, 164, 192)
    return msg_type, mmsi, f


class AisDecoder:
    """Discriminator samples of one AIS channel in, messages out."""

    name = "AIS"

    def __init__(self, sample_rate: float, channel: str = "") -> None:
        self.sample_rate = float(sample_rate)
        self.channel = channel
        self._slicer = BitSlicer(self.sample_rate, BAUD, dc_bits=DC_BITS)
        self._framer = HdlcFramer(min_bytes=MIN_AIS_BYTES)
        #: Names learnt from types 5, 21 and 24, so later position reports carry them.
        self.names: dict[int, str] = {}

    def reset(self) -> None:
        self._slicer.reset()
        self._framer.reset()

    @property
    def bad_frames(self) -> int:
        return self._framer.bad_frames

    def process(self, x: np.ndarray) -> list[AisMessage]:
        out = []
        for frame in self._framer.feed(self._slicer.process(x)):
            bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))  # MSB first again
            parsed = parse(bits)
            if parsed is None:
                continue
            msg_type, mmsi, fields = parsed
            if fields.get("name"):
                self.names[mmsi] = fields["name"]
            out.append(AisMessage(msg_type, mmsi, fields, nmea(bits, self.channel or "A"),
                                  self.channel, self.names.get(mmsi, "")))
        return out
