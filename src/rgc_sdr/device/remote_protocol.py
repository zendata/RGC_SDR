"""The network radio server's wire format, shared by the server and the app (PLANNING.md 7q).

One TCP connection carries frames of `[type u8][length u32 LE][payload]`: JSON control
both ways; from the server, IQ as block-scaled int16 pairs and spectrum lines of the
radio's whole span as a byte a bin. Also the rules both ends share: the IQ window's rate
for a radio rate, and when tuning has to move the radio rather than the window.

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
PROTOCOL_VERSION = 2

FRAME_JSON = 1
FRAME_IQ = 2
FRAME_SPECTRUM = 3

_HEADER = struct.Struct("<BI")
_IQ_HEADER = struct.Struct("<IfI")
_SPECTRUM_HEADER = struct.Struct("<ddffI")
#: Largest frame accepted: a corrupt length must not allocate gigabytes.
MAX_FRAME = 16 << 20

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


#: The IQ window (P9b): the radio's rate halved until it is at most this. Wide enough
#: for broadcast FM's 200 kHz, about 8 Mbit/s at 256 kS/s.
MAX_IQ_RATE = 400e3
#: The IQ window keeps this fraction of the span away from the edge, where the radio's
#: own filter rolls off; nearer, the radio is retuned instead.
EDGE_MARGIN = 0.05
#: A spectrum line's bytes: the step between levels is never finer than this.
MIN_DB_STEP = 0.05


def iq_rate_for(radio_rate: float) -> tuple[float, int]:
    """(IQ window rate, decimation factor) for a radio running at `radio_rate`."""
    factor = 1
    while radio_rate / factor > MAX_IQ_RATE:
        factor *= 2
    return radio_rate / factor, factor


def needs_recentre(radio_centre: float, radio_rate: float, iq_rate: float,
                   tuned: float) -> bool:
    """Whether an IQ window at `tuned` would reach past the usable span of a radio
    centred on `radio_centre`, so the radio itself has to move."""
    reach = radio_rate / 2 - iq_rate / 2 - radio_rate * EDGE_MARGIN
    return abs(tuned - radio_centre) > max(0.0, reach)


def encode_spectrum(centre_hz: float, span_hz: float, dbfs: np.ndarray) -> bytes:
    """A spectrum line, a byte a bin between its own floor and peak."""
    dbfs = np.asarray(dbfs, dtype=np.float32)
    low = float(dbfs.min()) if dbfs.size else 0.0
    step = max(MIN_DB_STEP, (float(dbfs.max()) - low) / 255.0) if dbfs.size else 1.0
    q = np.clip(np.round((dbfs - low) / step), 0, 255).astype(np.uint8)
    return frame(FRAME_SPECTRUM,
                 _SPECTRUM_HEADER.pack(centre_hz, span_hz, low, step, dbfs.size) + q.tobytes())


def decode_spectrum(payload: bytes) -> tuple[float, float, np.ndarray]:
    """(centre Hz, span Hz, dBFS per bin) from a spectrum frame's payload."""
    if len(payload) < _SPECTRUM_HEADER.size:
        raise ProtocolError("short spectrum frame")
    centre, span, low, step, count = _SPECTRUM_HEADER.unpack_from(payload)
    body = np.frombuffer(payload, dtype=np.uint8, offset=_SPECTRUM_HEADER.size)
    if body.size != count:
        raise ProtocolError("spectrum frame length does not match its count")
    return centre, span, (low + body.astype(np.float32) * np.float32(step))


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
    if kind not in (FRAME_JSON, FRAME_IQ, FRAME_SPECTRUM) or length > MAX_FRAME:
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
