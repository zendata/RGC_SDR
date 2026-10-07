"""The network radio server's wire format, shared by the server and the app (PLANNING.md 7q).

One TCP connection carries frames of `[type u8][length u32 LE][payload]`: JSON control
both ways, and IQ from the server as block-scaled int16 pairs. Also the link-rate plan:
which rates the server offers for a radio, and how it makes each.

No Qt and no DSP imports (layering rule, PLANNING.md section 5), and Python 3.13-clean:
the server runs on the Pi's Debian Python.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

#: The server's TCP port, next to SoapyRemote's 55132.
DEFAULT_PORT = 55133
#: Bumped when the wire format changes; the server refuses a client that differs.
PROTOCOL_VERSION = 1

FRAME_JSON = 1
FRAME_IQ = 2

_HEADER = struct.Struct("<BI")
_IQ_HEADER = struct.Struct("<IfI")
#: Largest frame accepted: a corrupt length must not allocate gigabytes.
MAX_FRAME = 16 << 20

#: The link-rate window (PLANNING.md 7q): below this nothing useful fits, above it the
#: measured Tailscale link (about 30 Mbit/s, 4 bytes a sample) runs out.
MIN_LINK_RATE = 48e3
MAX_LINK_RATE = 1.0e6
#: Highest radio rate the Pi is asked to decimate from, as the app's own ceiling.
MAX_RADIO_RATE = 10e6


class ProtocolError(RuntimeError):
    """The other end sent something that is not this protocol."""


def frame(kind: int, payload: bytes) -> bytes:
    return _HEADER.pack(kind, len(payload)) + payload


def json_frame(message: dict) -> bytes:
    return frame(FRAME_JSON, json.dumps(message, separators=(",", ":")).encode())


def encode_iq(iq: np.ndarray, generation: int) -> bytes:
    """An IQ frame: int16 pairs scaled to this block's peak, so a quiet block keeps all
    16 bits rather than sitting at the bottom of a fixed full scale."""
    iq = np.asarray(iq, dtype=np.complex64)
    pairs = iq.view(np.float32)
    peak = float(np.max(np.abs(pairs))) if pairs.size else 0.0
    scale = peak / 32767.0 if peak > 0 else 1.0
    ints = np.round(pairs / scale).astype("<i2")
    return frame(FRAME_IQ, _IQ_HEADER.pack(int(generation) & 0xFFFFFFFF, scale, iq.size)
                 + ints.tobytes())


def decode_iq(payload: bytes) -> tuple[int, np.ndarray]:
    """(generation, complex64 samples) from an IQ frame's payload."""
    if len(payload) < _IQ_HEADER.size:
        raise ProtocolError("short IQ frame")
    generation, scale, count = _IQ_HEADER.unpack_from(payload)
    body = np.frombuffer(payload, dtype="<i2", offset=_IQ_HEADER.size)
    if body.size != 2 * count:
        raise ProtocolError("IQ frame length does not match its count")
    out = body.astype(np.float32) * np.float32(scale)
    return generation, out.view(np.complex64)


def read_exact(sock: socket.socket, n: int) -> bytes:
    """Exactly `n` bytes, or ConnectionError if the other end closes first."""
    chunks = []
    while n:
        chunk = sock.recv(min(n, 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    kind, length = _HEADER.unpack(read_exact(sock, _HEADER.size))
    if kind not in (FRAME_JSON, FRAME_IQ) or length > MAX_FRAME:
        raise ProtocolError(f"bad frame (type {kind}, {length} bytes)")
    return kind, read_exact(sock, length)


def parse_json(payload: bytes) -> dict:
    try:
        message = json.loads(payload)
    except ValueError as exc:
        raise ProtocolError(f"bad JSON: {exc}") from None
    if not isinstance(message, dict):
        raise ProtocolError("a control message must be an object")
    return message


def link_plan(radio_rates, default_rate: float,
              max_radio_rate: float = MAX_RADIO_RATE) -> dict[float, tuple[float, int]]:
    """Link rate -> (radio rate, decimation factor): each rate between MIN_LINK_RATE and
    MAX_LINK_RATE that some radio rate up to `max_radio_rate` reaches by halving, made
    from the radio rate nearest `default_rate` (the profile's choice, known to work)."""
    plan: dict[float, tuple[float, int]] = {}
    for rate in sorted({float(r) for r in radio_rates if 0 < r <= max_radio_rate}):
        factor = 1
        while rate / factor >= MIN_LINK_RATE:
            link = rate / factor
            if link <= MAX_LINK_RATE:
                best = plan.get(link)
                if best is None or abs(rate - default_rate) < abs(best[0] - default_rate):
                    plan[link] = (rate, factor)
            factor *= 2
    return dict(sorted(plan.items(), reverse=True))


# -- capabilities as JSON ------------------------------------------------------------


def caps_to_dict(caps) -> dict:
    """A DeviceCaps as JSON, without its transmitter (P9 is receive only)."""
    return {
        "driver": caps.driver, "label": caps.label, "serial": caps.serial,
        "sample_rates": list(caps.sample_rates),
        "freq_ranges": [[r.min_hz, r.max_hz] for r in caps.freq_ranges],
        "gain_elements": [[g.name, g.min_db, g.max_db, g.step_db] for g in caps.gain_elements],
        "has_agc": caps.has_agc, "formats": list(caps.formats),
        "bandwidths": list(caps.bandwidths),
        "settings": [[s.key, s.name, s.description, s.default] for s in caps.settings],
    }


def caps_from_dict(data: dict):
    from .source import DeviceCaps, FreqRange, GainElement, SettingInfo

    return DeviceCaps(
        driver=str(data["driver"]), label=str(data.get("label", "")),
        serial=str(data.get("serial", "")),
        sample_rates=tuple(float(r) for r in data.get("sample_rates", ())),
        freq_ranges=tuple(FreqRange(float(a), float(b)) for a, b in data.get("freq_ranges", ())),
        gain_elements=tuple(GainElement(str(n), float(a), float(b), float(s))
                            for n, a, b, s in data.get("gain_elements", ())),
        has_agc=bool(data.get("has_agc", False)),
        formats=tuple(str(f) for f in data.get("formats", ())),
        bandwidths=tuple(float(b) for b in data.get("bandwidths", ())),
        settings=tuple(SettingInfo(str(k), str(n), str(d), bool(v))
                       for k, n, d, v in data.get("settings", ())),
        tx=None,
    )
