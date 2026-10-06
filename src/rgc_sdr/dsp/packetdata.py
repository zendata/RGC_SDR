"""What a data packet carries, for P25 and DMR alike: IPv4/UDP to Motorola's well-known
services -- text messages (TMS), registrations (ARS), location reports (LRRP) -- shown
with their content; positions for the map.

Pure Python; no Qt, no device access.
"""

from __future__ import annotations

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
    0x51/0x54/0x66/0x69 tokens). Token layout from open-source decoders; the position
    (0x51) checked on air 2026-10-07 against DMR reports from Victoria."""
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
            # Latitude is sign and magnitude, longitude two's complement: measured on
            # DMR location reports, which land in Victoria only that way.
            raw = int.from_bytes(body[i + 1:i + 5], "big")
            lat = (-1 if raw & 0x80000000 else 1) * (raw & 0x7FFFFFFF) * 90.0 / 2 ** 31
            lon = int.from_bytes(body[i + 5:i + 9], "big", signed=True) * 180.0 / 2 ** 31
            parts.append(f"position {lat:.5f}, {lon:.5f}")
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
