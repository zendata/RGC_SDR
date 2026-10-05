"""ACARS (ARINC 618) aircraft messages, from the AM detector.

2400 baud MSK on an AM carrier: tones of 1200 and 2400 Hz. The tone does not carry a
bit directly: **2400 Hz means "the same as the previous bit", 1200 Hz means "the
opposite"**, so the data is the running XOR of the 1200 Hz decisions. Measured on air
(131.550 MHz, Melbourne, 2026-10-05): that reading found SYN SYN SOH in all 30 bursts of
a 4-minute capture, where reading the tone as the bit, or a tone change as a bit, found
none. Its polarity depends on where decoding started, so both are tried.

Characters are 7-bit ASCII with odd parity, least significant bit first. A block is
SYN SYN SOH, mode, aircraft address (7), acknowledgement, label (2), block id, STX,
text, ETX or ETB, then a CRC-16/KERMIT block check -- the variant that validated all 26
complete frames of that capture, where X.25, XMODEM and CCITT-FALSE validated none.

The AM detector removes the carrier only slowly, so the audio is high-passed first: in
a burst a fraction of a second long, the residual DC otherwise swamps the tones.

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

BAUD = 2400
CENTRE_HZ = 1800.0
HIGHPASS_HZ = 600.0
SYN, SOH, STX, ETX, ETB, NAK = 0x16, 0x01, 0x02, 0x03, 0x17, 0x15
#: SYN SYN SOH as transmitted (with their parity bits), least significant bit first.
_SYNC_BITS = np.unpackbits(np.array([SYN, SYN, SOH], dtype=np.uint8), bitorder="little")
MAX_CHARS = 240                    # 220 text characters plus the header and check
#: Labels worth naming; there are hundreds, most airline-specific.
LABELS = {"SQ": "squitter", "_\x7f": "acknowledgement", "Q0": "link test",
          "H1": "message to/from a terminal", "5Z": "airline downlink",
          "80": "position report", "15": "position report", "16": "position report",
          "C1": "uplink to cockpit printer", "QQ": "free text", "10": "free text"}

# Positions: decimal degrees ("S 37.894/E144.735"), and ground stations' own in
# squitters, degrees and minutes ("03741S14451E").
_DECIMAL = re.compile(r"([NS])\s?(\d{1,2}\.\d+)\s?/\s?([EW])\s?(\d{1,3}\.\d+)")
_SQUITTER = re.compile(r"(\d{3})(\d{2})([NS])(\d{3})(\d{2})([EW])")


def crc_kermit(data: bytes) -> int:
    """CRC-16/KERMIT, ACARS's block check: reflected 0x1021, initial 0, no final XOR.
    A loop over a block's characters, which are few, not over samples."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc


def _text(values) -> str:
    return "".join(chr(v) if 32 <= v < 127 else ("\n" if v == 10 else "")
                   for v in values).replace("\r", "")


def position_in(text: str, label: str) -> tuple[float, float] | None:
    m = _DECIMAL.search(text)
    if m:
        lat, lon = float(m[2]), float(m[4])
        if lat <= 90 and lon <= 180:
            return (-lat if m[1] == "S" else lat), (-lon if m[3] == "W" else lon)
    if label == "SQ":
        m = _SQUITTER.search(text)
        if m:
            lat = int(m[1]) + int(m[2]) / 60
            lon = int(m[4]) + int(m[5]) / 60
            return (-lat if m[3] == "S" else lat), (-lon if m[6] == "W" else lon)
    return None


@dataclass
class AcarsMessage:
    mode: str
    registration: str
    ack: str
    label: str
    block: str
    text: str
    flight: str = ""
    msg_no: str = ""
    channel: str = ""
    position: tuple[float, float] | None = None
    received: float = field(default_factory=time.time)

    @property
    def downlink(self) -> bool:
        """From an aircraft (block id a digit), rather than from the ground."""
        return self.block.isdigit()

    @property
    def ground_station(self) -> str:
        """The station's ICAO code, from a squitter ("...MELYMML..." -> YMML)."""
        if self.label != "SQ":
            return ""
        m = re.search(r"^\d\dX[A-Z]([A-Z]{3})([A-Z]{4})", self.text)
        return m[2] if m else ""

    def summary(self, show_text: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.received))
        who = self.registration or self.ground_station or "ground"
        way = "from" if self.downlink else "to"
        head = [f"{stamp}  ACARS  {way} {who}"]
        if self.flight:
            head.append(self.flight)
        name = LABELS.get(self.label, "")
        shown = self.label.replace("\x7f", "DEL")         # the acknowledgement label
        head.append(f"label {shown}" + (f" ({name})" if name else ""))
        if self.position:
            head.append(f"[{self.position[0]:.4f}, {self.position[1]:.4f}]")
        body = self.text.strip()
        return "  ".join(head) + (("\n    " + body.replace("\n", "\n    ")) if body else "")


def parse_block(values: list[int]) -> AcarsMessage | None:
    """Characters after SYN SYN SOH, parity bits still on, through the check: a message,
    or None if the block check fails."""
    end = next((i for i, v in enumerate(values) if i >= 11 and v & 0x7F in (ETX, ETB)), None)
    if end is None or end + 2 >= len(values):
        return None
    body = bytes(values[:end + 1])
    if crc_kermit(body) != values[end + 1] | values[end + 2] << 8:
        return None
    chars = [v & 0x7F for v in values[:end]]
    mode = chr(chars[0])
    registration = _text(chars[1:8]).lstrip(".").strip()
    ack = "NAK" if chars[8] == NAK else _text(chars[8:9])
    label = "".join(chr(c) for c in chars[9:11])
    block = chr(chars[11]) if chars[11] else ""
    text_chars = chars[13:] if len(chars) > 12 and chars[12] == STX else []
    text = _text(text_chars)
    msg_no = flight = ""
    if block.isdigit() and len(text) >= 10:
        msg_no, flight, text = text[:4], text[4:10].strip(), text[10:]
    message = AcarsMessage(mode, registration, ack, label, block, text, flight, msg_no)
    message.position = position_in(text, label)
    return message


class AcarsDecoder:
    """AM detector samples in, ACARS messages out."""

    name = "ACARS"

    def __init__(self, sample_rate: float, channel: str = "") -> None:
        self.sample_rate = float(sample_rate)
        self.channel = channel
        c = HIGHPASS_HZ / self.sample_rate
        n = fir_length_for(c, maximum=511)
        highpass = -lowpass_taps(c, n)
        highpass[n // 2] += 1.0
        self._highpass = Fir(highpass)
        self._step = -2.0 * np.pi * CENTRE_HZ / self.sample_rate
        cutoff = 1300.0 / self.sample_rate
        self._lowpass = Fir(lowpass_taps(cutoff, fir_length_for(cutoff, maximum=255)))
        self._slicer = BitSlicer(self.sample_rate, BAUD, track_dc=False)
        self.bad_blocks = 0
        self.reset()

    def reset(self) -> None:
        self._highpass.reset()
        self._lowpass.reset()
        self._slicer.reset()
        self._phase = 0.0
        self._last = None
        self._data_bit = 0
        self._bits = np.zeros(0, dtype=np.uint8)

    def _tones(self, x: np.ndarray) -> np.ndarray:
        """1 where the tone is 2400 Hz, 0 where 1200 Hz, one per bit."""
        x = self._highpass.process(np.asarray(x, dtype=np.float64))
        phase = self._phase + self._step * np.arange(x.size)
        self._phase = float((self._phase + self._step * x.size) % (2 * np.pi))
        z = self._lowpass.process(x * np.exp(1j * phase))
        prev = np.empty_like(z)
        prev[0] = z[0] if self._last is None else self._last
        prev[1:] = z[:-1]
        self._last = complex(z[-1])
        return self._slicer.process(np.angle(z * np.conj(prev)))

    def process(self, x: np.ndarray) -> list[AcarsMessage]:
        if np.asarray(x).size == 0:
            return []
        hi = self._tones(x)
        if hi.size:
            # 1200 Hz flips the data; 2400 Hz keeps it.
            data = np.bitwise_xor.accumulate(1 - hi) ^ self._data_bit
            self._data_bit = int(data[-1])
            self._bits = np.concatenate([self._bits, data.astype(np.uint8)])
        return self._frames()

    def _frames(self) -> list[AcarsMessage]:
        out = []
        buf = self._bits
        need = _SYNC_BITS.size + 8 * MAX_CHARS
        pos = 0
        while buf.size - pos >= _SYNC_BITS.size:
            windows = sliding_window_view(buf[pos:], _SYNC_BITS.size)
            match = np.flatnonzero((windows == _SYNC_BITS).all(axis=1) |
                                   (windows == 1 - _SYNC_BITS).all(axis=1))
            if match.size == 0:
                pos = buf.size - _SYNC_BITS.size + 1
                break
            start = pos + int(match[0])
            inverted = buf[start] != _SYNC_BITS[0]
            body = buf[start + _SYNC_BITS.size:start + need]
            if inverted:
                body = 1 - body
            usable = body.size // 8 * 8
            values = list(np.packbits(body[:usable].reshape(-1, 8), axis=1,
                                      bitorder="little").ravel())
            end = next((i for i, v in enumerate(values)
                        if i >= 11 and v & 0x7F in (ETX, ETB) and i + 2 < len(values)), None)
            if end is None and body.size < 8 * MAX_CHARS:
                pos = start                           # wait for the rest of the block
                break
            message = parse_block([int(v) for v in values])
            if message is None:
                self.bad_blocks += 1
                pos = start + 1
                continue
            message.channel = self.channel
            out.append(message)
            pos = start + _SYNC_BITS.size + 8 * (end + 3)     # past the check characters
        self._bits = buf[pos:]
        return out
