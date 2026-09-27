"""Icom CI-V: the control protocol of Icom transceivers such as the IC-705.

A CI-V message is

    FE FE <to> <from> <command> [<sub-command>] [<data>...] FD

addressed radio <-> controller. Frequencies are 5 bytes of BCD, least significant byte
first; levels are 2 bytes of BCD, most significant first (0000-0255). The radio answers
a setting with FB (OK) or FA (NG), and with "CI-V Transceive" on it also reports changes
made on its own front panel, unasked.

The spectrum scope streams as command 27 00, one scope line split over several
messages (IC-705 over USB, measured 2026-09-27: 11 per line, 475 one-byte amplitudes,
4.3 lines/s whatever the span, scope speed or host baud rate). `ScopeAssembler` puts a
line back together.

Pure protocol: no serial port, no Qt, no DSP. The transport lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PREAMBLE = 0xFE
END = 0xFD
OK = 0xFB
NG = 0xFA
#: Sent by a device that detected a collision on the shared CI-V bus.
JAMMER = 0xFC

IC705_ADDRESS = 0xA4
CONTROLLER_ADDRESS = 0xE0

#: A generous ceiling: the longest IC-705 message (a scope division) is 60 bytes.
MAX_FRAME = 256

CMD_READ_FREQ = 0x03
CMD_READ_MODE = 0x04
CMD_SET_FREQ = 0x05
CMD_SET_MODE = 0x06
CMD_TRANSCEIVE_FREQ = 0x00      # unasked frequency report (CI-V Transceive)
CMD_TRANSCEIVE_MODE = 0x01      # unasked mode report
CMD_SCOPE = 0x27

SCOPE_WAVE = 0x00               # 27 00: waveform data
SCOPE_ON = 0x10                 # 27 10: scope on/off
SCOPE_OUTPUT = 0x11             # 27 11: waveform output over CI-V on/off
SCOPE_CENTRE_FIXED = 0x14       # 27 14 <main/sub>: 00 centre, 01 fixed
SCOPE_SPAN = 0x15               # 27 15 <main/sub> <5-byte BCD half-span>
SCOPE_SPEED = 0x1A              # 27 1A <main/sub>: 00 fast, 01 mid, 02 slow

#: CI-V mode codes (command 04/06). D-STAR (DV, 0x17) is left out for now.
MODES: dict[int, str] = {
    0x00: "lsb", 0x01: "usb", 0x02: "am", 0x03: "cw", 0x04: "rtty",
    0x05: "fm", 0x06: "wfm", 0x07: "cw-r", 0x08: "rtty-r",
}
MODE_CODES = {name: code for code, name in MODES.items()}

#: The scope spans the IC-705 offers, as half-spans in Hz.
SCOPE_SPANS_HZ = (2.5e3, 5e3, 10e3, 25e3, 50e3, 100e3, 250e3, 500e3)


@dataclass(frozen=True)
class Frame:
    to: int
    frm: int
    cmd: int
    #: Everything after the command byte, sub-command included.
    payload: bytes

    @property
    def sub(self) -> int | None:
        return self.payload[0] if self.payload else None

    @property
    def is_ok(self) -> bool:
        return self.cmd == OK

    @property
    def is_ng(self) -> bool:
        return self.cmd == NG


def encode(cmd: int, payload: bytes | tuple | list = b"", to: int = IC705_ADDRESS,
           frm: int = CONTROLLER_ADDRESS) -> bytes:
    """One CI-V message."""
    body = bytes(payload)
    if END in body:
        raise ValueError("CI-V data cannot contain FD")
    return bytes([PREAMBLE, PREAMBLE, to, frm, cmd]) + body + bytes([END])


class FrameParser:
    """Turns a byte stream into CI-V frames, however it was split across reads.

    Discards noise before a preamble, frames that overrun MAX_FRAME, and anything a
    jammer (FC) announces as a collision.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.discarded = 0

    def feed(self, data: bytes) -> list[Frame]:
        self._buffer.extend(data)
        frames: list[Frame] = []
        buf = self._buffer
        while True:
            start = buf.find(bytes([PREAMBLE, PREAMBLE]))
            if start < 0:
                # Keep a lone trailing FE: it may be half of the next preamble.
                keep = 1 if buf.endswith(bytes([PREAMBLE])) else 0
                self.discarded += len(buf) - keep
                del buf[: len(buf) - keep]
                break
            if start:
                self.discarded += start
                del buf[:start]
            # Collapse extra FEs: FE FE FE A4 ... is a preamble followed by data.
            while len(buf) > 2 and buf[2] == PREAMBLE:
                del buf[0]
                self.discarded += 1
            end = buf.find(bytes([END]), 2)
            if end < 0:
                if len(buf) > MAX_FRAME:
                    self.discarded += len(buf)
                    buf.clear()
                break
            raw = bytes(buf[: end + 1])
            del buf[: end + 1]
            if len(raw) < 6 or JAMMER in raw[2:-1]:
                self.discarded += len(raw)
                continue
            frames.append(Frame(to=raw[2], frm=raw[3], cmd=raw[4], payload=raw[5:-1]))
        return frames


# -- BCD -----------------------------------------------------------------------------

def _bcd_byte(value: int) -> int:
    return ((value // 10) << 4) | (value % 10)


def _from_bcd_byte(b: int) -> int:
    hi, lo = b >> 4, b & 0x0F
    if hi > 9 or lo > 9:
        raise ValueError(f"not BCD: {b:02x}")
    return hi * 10 + lo


def encode_freq(hz: float, nbytes: int = 5) -> bytes:
    """Frequency as BCD, least significant byte first: 145.65 MHz -> 00 00 65 45 01."""
    value = int(round(hz))
    if value < 0 or value >= 10 ** (2 * nbytes):
        raise ValueError(f"{hz} Hz does not fit in {nbytes} BCD bytes")
    out = []
    for _ in range(nbytes):
        out.append(_bcd_byte(value % 100))
        value //= 100
    return bytes(out)


def decode_freq(data: bytes) -> int:
    value = 0
    for b in reversed(data):
        value = value * 100 + _from_bcd_byte(b)
    return value


def encode_level(value: int) -> bytes:
    """0-255 as the 2-byte BCD Icom uses for levels, most significant first: 128 -> 01 28."""
    if not 0 <= value <= 255:
        raise ValueError("level must be 0-255")
    return bytes([_bcd_byte(value // 100), _bcd_byte(value % 100)])


def decode_level(data: bytes) -> int:
    return _from_bcd_byte(data[0]) * 100 + _from_bcd_byte(data[1])


# -- scope -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScopeLine:
    """One sweep of the radio's spectrum scope."""

    #: True in centre mode (centre +/- span), False in fixed mode (lower..upper edge).
    centre_mode: bool
    low_hz: float
    high_hz: float
    #: The radio's own amplitudes, 0-160 as documented; one per scope pixel.
    amplitudes: np.ndarray
    out_of_range: bool

    @property
    def centre_hz(self) -> float:
        return (self.low_hz + self.high_hz) / 2.0

    @property
    def span_hz(self) -> float:
        return self.high_hz - self.low_hz


class ScopeAssembler:
    """Collects 27 00 divisions into whole scope lines.

    Division 1 carries the header -- centre/fixed, two frequencies, out-of-range -- and
    the rest carry amplitudes. Sequence numbers are BCD. A missing or out-of-order
    division drops the partial line rather than emitting a torn one.
    """

    def __init__(self) -> None:
        self._header: tuple[bool, float, float, bool] | None = None
        self._parts: list[bytes] = []
        self._expect = 1
        self.dropped = 0

    def feed(self, frame: Frame) -> ScopeLine | None:
        if frame.cmd != CMD_SCOPE or frame.sub != SCOPE_WAVE or len(frame.payload) < 4:
            return None
        body = frame.payload[1:]            # after the 00 sub-command
        # body: <main/sub> <seq> <seq max> <data...>
        try:
            seq, total = _from_bcd_byte(body[1]), _from_bcd_byte(body[2])
        except ValueError:
            self._reset(dropped=True)
            return None
        data = body[3:]
        if seq == 1:
            if self._parts or self._header:
                self._reset(dropped=True)
            if len(data) < 12:
                return None
            centre_mode = data[0] == 0x00
            first, second = decode_freq(data[1:6]), decode_freq(data[6:11])
            if centre_mode:                 # centre and half-span
                low, high = first - second, first + second
            else:                           # lower and upper edges
                low, high = first, second
            self._header = (centre_mode, float(low), float(high), bool(data[11]))
            self._expect = 2
            if total == 1:                  # everything in one message (network form)
                return self._line(data[12:])
            return None
        if self._header is None or seq != self._expect:
            self._reset(dropped=self._header is not None)
            return None
        self._parts.append(bytes(data))
        self._expect += 1
        if seq == total:
            return self._line(b"".join(self._parts))
        return None

    def _line(self, amplitudes: bytes) -> ScopeLine:
        centre_mode, low, high, oor = self._header
        line = ScopeLine(centre_mode, low, high,
                         np.frombuffer(amplitudes, dtype=np.uint8).copy(), oor)
        self._reset()
        return line

    def _reset(self, dropped: bool = False) -> None:
        if dropped:
            self.dropped += 1
        self._header = None
        self._parts = []
        self._expect = 1
