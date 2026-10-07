"""APRS: AX.25 packets over 1200 baud Bell 202 AFSK, from the NBFM discriminator.

Mark is 1200 Hz and space 2200 Hz. The discriminator's audio is mixed down by their
midpoint, 1700 Hz, low-passed and FM-detected a second time, so the sign says mark or
space whatever the transmitter's pre-emphasis did to the two tones' levels. Then NRZI
(a change is 0, no change is 1), HDLC flags and bit unstuffing, least-significant-bit
first bytes, and the CRC-16/X.25 frame check.

Positions are read in all three APRS formats: plain ("!3749.10S/14458.00E"), compressed
(base 91), and Mic-E, which most Kenwood and Yaesu APRS radios send.

Pure NumPy; no Qt, no device access.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .bitsync import BitSlicer
from .decimate import lowpass_taps
from .filters import Fir, fir_length_for

BAUD = 1200
MARK_HZ, SPACE_HZ = 1200.0, 2200.0
CENTRE_HZ = (MARK_HZ + SPACE_HZ) / 2
FLAG = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
#: Shortest frame worth checking: two addresses, control and the FCS.
MIN_FRAME_BYTES = 17
MAX_FRAME_BYTES = 400
#: Bits kept while waiting for a closing flag before giving up on a frame.
MAX_PENDING_BITS = MAX_FRAME_BYTES * 10


def _crc_table() -> np.ndarray:
    table = np.arange(256, dtype=np.uint32)
    for _ in range(8):
        table = np.where(table & 1, (table >> 1) ^ 0x8408, table >> 1)
    return table.astype(np.uint16)


_CRC_TABLE = _crc_table()


def crc_x25(data: bytes) -> int:
    """CRC-16/X.25, the AX.25 frame check: reflected 0x1021, init and xorout 0xFFFF.
    A loop over a frame's bytes, which are few, not over samples."""
    crc = 0xFFFF
    for byte in data:
        crc = (crc >> 8) ^ int(_CRC_TABLE[(crc ^ byte) & 0xFF])
    return crc ^ 0xFFFF


def unstuff(bits: np.ndarray) -> np.ndarray | None:
    """Remove the 0 a sender inserts after five 1s. None if six or more 1s appear,
    which no valid frame contains."""
    idx = np.arange(bits.size)
    last_zero = np.maximum.accumulate(np.where(bits == 0, idx, -1))
    ones = np.where(bits == 1, idx - last_zero, 0)        # 1s in a row ending here
    if ones.size and ones.max() >= 6:
        return None
    stuffed = np.zeros(bits.size, dtype=bool)
    stuffed[1:] = (bits[1:] == 0) & (ones[:-1] == 5)
    return bits[~stuffed]


def frame_bytes(bits: np.ndarray, min_bytes: int = MIN_FRAME_BYTES) -> bytes | None:
    """Unstuffed bits between two flags -> bytes, if they make a checked frame."""
    bits = unstuff(bits)
    if bits is None or bits.size % 8 or not (min_bytes * 8 <= bits.size
                                              <= MAX_FRAME_BYTES * 8):
        return None
    data = np.packbits(bits.reshape(-1, 8), axis=1, bitorder="little").ravel().tobytes()
    body, fcs = data[:-2], data[-2] | data[-1] << 8
    return body if crc_x25(body) == fcs else None


class AfskDemod:
    """Discriminator audio in, a signal whose sign is space (+) or mark (-) out."""

    def __init__(self, sample_rate: float) -> None:
        self.rate = float(sample_rate)
        self._step = -2.0 * np.pi * CENTRE_HZ / self.rate
        self._phase = 0.0
        cutoff = 1200.0 / self.rate
        self._lpf = Fir(lowpass_taps(cutoff, fir_length_for(cutoff, maximum=255)))
        self._last = None

    def reset(self) -> None:
        self._phase = 0.0
        self._lpf.reset()
        self._last = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0)
        phase = self._phase + self._step * np.arange(x.size)
        self._phase = float((self._phase + self._step * x.size) % (2 * np.pi))
        z = self._lpf.process(x * np.exp(1j * phase))
        prev = np.empty_like(z)
        prev[0] = z[0] if self._last is None else self._last
        prev[1:] = z[:-1]
        self._last = complex(z[-1])
        return np.angle(z * np.conj(prev))


class HdlcFramer:
    """NRZI levels in, checked HDLC frames (without their FCS) out. AX.25 and AIS both
    frame this way."""

    def __init__(self, min_bytes: int = MIN_FRAME_BYTES) -> None:
        self.min_bytes = int(min_bytes)
        self.reset()

    def reset(self) -> None:
        self._prev_level = None
        self._bits = np.zeros(0, dtype=np.uint8)
        self.bad_frames = 0

    def feed(self, levels: np.ndarray) -> list[bytes]:
        if levels.size == 0:
            return []
        prev = levels[0] if self._prev_level is None else self._prev_level
        shifted = np.concatenate([[prev], levels[:-1]])
        self._prev_level = int(levels[-1])
        data = (levels == shifted).astype(np.uint8)          # NRZI: no change is a 1
        buf = np.concatenate([self._bits, data])
        if buf.size < 8:
            self._bits = buf
            return []
        windows = sliding_window_view(buf, 8)
        flags = np.flatnonzero((windows == FLAG).all(axis=1))
        frames = []
        for start, end in zip(flags[:-1], flags[1:]):
            inner = buf[start + 8:end]
            if inner.size >= self.min_bytes * 8:
                frame = frame_bytes(inner, self.min_bytes)
                if frame is None:
                    self.bad_frames += 1
                else:
                    frames.append(frame)
        keep_from = int(flags[-1]) if flags.size else max(0, buf.size - 7)
        self._bits = buf[keep_from:][-MAX_PENDING_BITS:]
        return frames


def _address(chunk: bytes) -> tuple[str, bool]:
    call = bytes(b >> 1 for b in chunk[:6]).decode("ascii", "replace").strip()
    ssid = (chunk[6] >> 1) & 0xF
    return (f"{call}-{ssid}" if ssid else call), bool(chunk[6] & 0x80)


def _printable(info: bytes) -> str:
    return "".join(chr(b) if 32 <= b < 127 else f"<0x{b:02x}>" for b in info)


def _base91(s: str) -> int:
    value = 0
    for c in s:
        value = value * 91 + ord(c) - 33
    return value


_PLAIN = re.compile(r"^(\d{2})([\d ]{2}\.[\d ]{2})([NS])(.)(\d{3})([\d ]{2}\.[\d ]{2})([EW])(.)")


def plain_position(body: str) -> tuple[float, float] | None:
    m = _PLAIN.match(body)
    if not m:
        return None
    lat = int(m[1]) + float(m[2].replace(" ", "0")) / 60
    lon = int(m[5]) + float(m[6].replace(" ", "0")) / 60
    return (-lat if m[3] == "S" else lat), (-lon if m[7] == "W" else lon)


def compressed_position(body: str) -> tuple[float, float] | None:
    if len(body) < 13 or not (body[0] in "/\\" or body[0].isalnum()):
        return None
    y, x = body[1:5], body[5:9]
    if not all(33 <= ord(c) <= 123 for c in y + x):
        return None
    return 90 - _base91(y) / 380926, -180 + _base91(x) / 190463


def mic_e_position(dest: str, info: bytes) -> tuple[float, float] | None:
    """Mic-E: latitude, N/S, E/W and a longitude offset hide in the destination."""
    call = dest.split("-")[0]
    if len(call) != 6 or len(info) < 4:
        return None
    digits = []
    for c in call:
        if c.isdigit():
            digits.append(int(c))
        elif "A" <= c <= "J":
            digits.append(ord(c) - ord("A"))
        elif "P" <= c <= "Y":
            digits.append(ord(c) - ord("P"))
        elif c in "KLZ":
            digits.append(0)
        else:
            return None
    lat = digits[0] * 10 + digits[1] + (digits[2] * 10 + digits[3] + (digits[4] * 10 + digits[5]) / 100) / 60
    north = "P" <= call[3] <= "Z"
    offset = 100 if "P" <= call[4] <= "Z" else 0
    west = "P" <= call[5] <= "Z"
    d = info[1] - 28 + offset
    if 180 <= d <= 189:
        d -= 80
    elif 190 <= d <= 199:
        d -= 190
    m = info[2] - 28
    if m >= 60:
        m -= 60
    h = info[3] - 28
    lon = d + (m + h / 100) / 60
    return (lat if north else -lat), (-lon if west else lon)


@dataclass
class AprsPacket:
    source: str
    dest: str
    path: list[str]
    info: bytes
    position: tuple[float, float] | None = None
    received: float = field(default_factory=time.time)
    #: An object's or item's own name (";" and ")" packets): what is placed on the map
    #: is the thing named -- a repeater, a net, an event -- not the station sending it.
    object_name: str | None = None
    #: Degrees true and knots, when the position report carries them.
    course: float | None = None
    speed_kn: float | None = None

    @property
    def text(self) -> str:
        return _printable(self.info)

    def summary(self, show_text: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.received))
        head = ">".join([self.source, ",".join([self.dest, *self.path])])
        where = (f"  [{self.position[0]:.4f}, {self.position[1]:.4f}]"
                 if self.position else "")
        return f"{stamp}  {head}:{self.text}{where}"


def parse_ax25(frame: bytes) -> AprsPacket | None:
    """An AX.25 UI frame -> packet; None if the address field is malformed."""
    addresses = []
    i = 0
    while i + 7 <= len(frame) and len(addresses) < 10:
        call, repeated = _address(frame[i:i + 7])
        addresses.append((call, repeated))
        last = frame[i + 6] & 1
        i += 7
        if last:
            break
    else:
        return None
    if len(addresses) < 2 or i >= len(frame):
        return None
    control = frame[i]
    i += 1
    if control == 0x03 and i < len(frame):
        i += 1                                              # the PID, 0xF0 for APRS
    info = frame[i:]
    dest = addresses[0][0]
    source = addresses[1][0]
    path = [c + ("*" if rep else "") for c, rep in addresses[2:]]
    packet = AprsPacket(source, dest, path, info)
    packet.position = aprs_position(dest, info)
    packet.object_name = aprs_object_name(info)
    if packet.position is not None:
        packet.course, packet.speed_kn = aprs_motion(dest, info)
    return packet


def aprs_object_name(info: bytes) -> str | None:
    """The name of an object (";NAME     *...") or item (")NAME!..."), else None."""
    text = info.decode("ascii", "replace")
    if text[:1] == ";" and len(text) >= 11 and text[10] in "*_":
        return text[1:10].strip() or None
    if text[:1] == ")":
        m = _ITEM.match(text)
        if m:
            return m[1].strip() or None
    return None


_ITEM = re.compile(r"^\)([^!_]{3,9})[!_]")
#: Course and speed after an uncompressed position: "...E>088/036".
_MOTION = re.compile(r"^(\d{3})/(\d{3})")


def _position_body(info: bytes) -> tuple[str, str] | None:
    """(kind, the text where the position starts) for the uncompressed-family reports."""
    text = info.decode("ascii", "replace")
    kind = text[:1]
    if kind in "!=":
        return kind, text[1:]
    if kind in "/@":
        return kind, text[8:]                          # after the timestamp
    if kind == ";" and len(text) >= 11 and text[10] in "*_":
        return kind, text[18:]                         # name, live flag, timestamp
    if kind == ")":
        m = _ITEM.match(text)
        if m:
            return kind, text[m.end():]
    return None


def aprs_motion(dest: str, info: bytes) -> tuple[float | None, float | None]:
    """(course degrees, speed knots) if the report gives them, else Nones."""
    try:
        if info[:1] in (b"`", b"'"):
            if len(info) < 7:
                return None, None
            sp, dc, se = info[4] - 28, info[5] - 28, info[6] - 28
            speed = sp * 10 + dc // 10
            course = (dc % 10) * 100 + se
            speed -= 800 if speed >= 800 else 0
            course -= 400 if course >= 400 else 0
            return (float(course) if 0 < course <= 360 else None), float(speed)
        found = _position_body(info)
        if found is None or not _PLAIN.match(found[1]):
            return None, None
        m = _MOTION.match(found[1][19:])
        if not m:
            return None, None
        course, speed = int(m[1]), int(m[2])
        return (float(course) if 0 < course <= 360 else None), float(speed)
    except (ValueError, IndexError):
        return None, None


def aprs_position(dest: str, info: bytes) -> tuple[float, float] | None:
    if not info:
        return None
    kind = chr(info[0])
    try:
        if kind in "`'":
            return mic_e_position(dest, info)
        found = _position_body(info)
        if found is None:
            return None
        body = found[1]
        return plain_position(body) or compressed_position(body)
    except (ValueError, IndexError):
        return None


class AprsDecoder:
    """Discriminator samples in, APRS packets out."""

    name = "APRS"

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = float(sample_rate)
        self._afsk = AfskDemod(self.sample_rate)
        self._slicer = BitSlicer(self.sample_rate, BAUD, track_dc=False)
        self._framer = HdlcFramer()

    def reset(self) -> None:
        self._afsk.reset()
        self._slicer.reset()
        self._framer.reset()

    @property
    def bad_frames(self) -> int:
        return self._framer.bad_frames

    def process(self, x: np.ndarray) -> list[AprsPacket]:
        levels = self._slicer.process(self._afsk.process(x))
        packets = []
        for frame in self._framer.feed(levels):
            packet = parse_ax25(frame)
            if packet is not None:
                packets.append(packet)
        return packets
