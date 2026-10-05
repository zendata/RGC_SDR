"""ADS-B (Mode S extended squitter) on 1090 MHz, from raw IQ.

Pulse-position modulation at 1 Mbit/s: an 8 us preamble of four pulses (0, 1, 3.5 and
4.5 us), then 56 or 112 bits, each a pulse in the first half of its microsecond for 1
or the second half for 0. It is read from the IQ magnitude directly, without an AM or
FM demodulator, at a whole number of samples per microsecond (2 MS/s and up).

Every message carries a 24-bit parity check (generator 0xFFF409), so a noisy candidate
is rejected rather than guessed. Extended squitters (DF17/18) give identity, airborne
position (CPR, decoded from an even/odd pair, then locally from the last fix -- no home
location is needed or stored), altitude, and velocity.

Pure NumPy; no Qt, no device access. Reference: Junzi Sun, "The 1090 Megahertz Riddle".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

FREQUENCY_HZ = 1090e6
GENERATOR = 0x1FFF409                      # x^24 + ... : the Mode S parity polynomial
LONG_BITS, SHORT_BITS = 112, 56
CALLSIGN_CHARS = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"
#: Seconds within which an even and an odd position may be paired.
CPR_PAIR_S = 10.0
NZ = 15
#: A preamble's pulses must stand this far above the quiet spaces between them.
PREAMBLE_RATIO = 2.0


def crc(bits: np.ndarray) -> int:
    """Mode S parity remainder over all bits; 0 for an intact DF17/18 message."""
    value = int("".join("1" if b else "0" for b in bits), 2)
    n = bits.size
    for i in range(n - 24):
        if value >> (n - 1 - i) & 1:
            value ^= GENERATOR << (n - 25 - i)
    return value & 0xFFFFFF


def _uint(bits: np.ndarray, start: int, n: int) -> int:
    return int("".join("1" if b else "0" for b in bits[start:start + n]), 2)


def bits_of_hex(text: str) -> np.ndarray:
    """A message written as hex, as the references write them -> bits."""
    return np.array([int(b) for b in bin(int(text, 16))[2:].zfill(len(text) * 4)],
                    dtype=np.uint8)


def altitude_ft(code: int) -> int | None:
    """12-bit altitude field: with the Q bit set, 25 ft steps from -1000 ft. The older
    Gillham coding (Q clear) is not decoded."""
    if code == 0 or not code & 0x10:
        return None
    n = ((code & 0xFE0) >> 1) | (code & 0xF)
    return n * 25 - 1000


def _nl(lat: float) -> int:
    """Longitude zones at a latitude, for CPR."""
    if abs(lat) >= 87.0:
        return 1
    a = 1 - math.cos(math.pi / (2 * NZ))
    b = math.cos(math.radians(lat)) ** 2
    return int(math.floor(2 * math.pi / math.acos(1 - a / b)))


def cpr_global(even: tuple[int, int], odd: tuple[int, int],
               latest_odd: bool) -> tuple[float, float] | None:
    """Airborne position from an even and an odd CPR pair (17-bit lat, lon)."""
    lat_e, lon_e = even[0] / 131072, even[1] / 131072
    lat_o, lon_o = odd[0] / 131072, odd[1] / 131072
    j = math.floor(59 * lat_e - 60 * lat_o + 0.5)
    lat0 = 360 / 60 * ((j % 60) + lat_e)
    lat1 = 360 / 59 * ((j % 59) + lat_o)
    lat0 = lat0 - 360 if lat0 >= 270 else lat0
    lat1 = lat1 - 360 if lat1 >= 270 else lat1
    if _nl(lat0) != _nl(lat1):
        return None                       # straddles a zone boundary; wait for another
    lat = lat1 if latest_odd else lat0
    nl = _nl(lat)
    m = math.floor(lon_e * (nl - 1) - lon_o * nl + 0.5)
    if latest_odd:
        n = max(nl - 1, 1)
        lon = 360 / n * ((m % n) + lon_o)
    else:
        n = max(nl, 1)
        lon = 360 / n * ((m % n) + lon_e)
    lon = lon - 360 if lon >= 180 else lon
    return lat, lon


def cpr_local(cpr: tuple[int, int], odd: bool, ref: tuple[float, float]) -> tuple[float, float]:
    """Airborne position from one CPR message and a nearby known position."""
    lat_c, lon_c = cpr[0] / 131072, cpr[1] / 131072
    dlat = 360 / (59 if odd else 60)
    j = math.floor(ref[0] / dlat) + math.floor((ref[0] % dlat) / dlat - lat_c + 0.5)
    lat = dlat * (j + lat_c)
    n = max(_nl(lat) - (1 if odd else 0), 1)
    dlon = 360 / n
    m = math.floor(ref[1] / dlon) + math.floor((ref[1] % dlon) / dlon - lon_c + 0.5)
    return lat, dlon * (m + lon_c)


@dataclass
class AdsbMessage:
    icao: str
    df: int
    tc: int
    fields: dict
    hex: str
    received: float = field(default_factory=time.time)

    @property
    def position(self) -> tuple[float, float] | None:
        return self.fields.get("position")

    def summary(self, show_text: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.received))
        f = self.fields
        parts = [f"{stamp}  ADS-B  {self.icao}"]
        if f.get("callsign"):
            parts.append(f["callsign"])
        if self.position:
            parts.append(f"{self.position[0]:.4f}, {self.position[1]:.4f}")
        if f.get("altitude_ft") is not None:
            parts.append(f"{f['altitude_ft']} ft")
        if f.get("speed_kn") is not None:
            parts.append(f"{f['speed_kn']:.0f} kn")
        if f.get("track") is not None:
            parts.append(f"track {f['track']:.0f}")
        if f.get("heading") is not None:
            parts.append(f"heading {f['heading']:.0f}")
        if f.get("vertical_fpm") is not None:
            parts.append(f"{f['vertical_fpm']:+d} ft/min")
        parts.append(f"*{self.hex};")
        return "  ".join(parts)


class _Cpr:
    """Per-aircraft CPR state: the latest even and odd frames, and the last fix."""

    def __init__(self) -> None:
        self.even = self.odd = None          # (lat17, lon17, time)
        self.fix: tuple[float, float, float] | None = None   # lat, lon, time


def decode_extended(bits: np.ndarray, cpr_state: dict[str, _Cpr],
                    received: float) -> AdsbMessage | None:
    """A 112-bit DF17/18 message, parity already checked."""
    df = _uint(bits, 0, 5)
    icao = f"{_uint(bits, 8, 24):06X}"
    me = bits[32:88]
    tc = _uint(me, 0, 5)
    f: dict = {}
    if 1 <= tc <= 4:
        f["callsign"] = "".join(CALLSIGN_CHARS[_uint(me, 8 + 6 * i, 6)]
                                for i in range(8)).replace("#", "").strip()
    elif 9 <= tc <= 18:
        f["altitude_ft"] = altitude_ft(_uint(me, 8, 12))
        odd = bool(me[21])
        lat17, lon17 = _uint(me, 22, 17), _uint(me, 39, 17)
        state = cpr_state.setdefault(icao, _Cpr())
        if odd:
            state.odd = (lat17, lon17, received)
        else:
            state.even = (lat17, lon17, received)
        position = None
        if state.fix is not None and received - state.fix[2] < 60:
            position = cpr_local((lat17, lon17), odd, state.fix[:2])
        elif state.even and state.odd and abs(state.even[2] - state.odd[2]) < CPR_PAIR_S:
            position = cpr_global(state.even[:2], state.odd[:2], latest_odd=odd)
        if position is not None:
            state.fix = (position[0], position[1], received)
            f["position"] = position
    elif tc == 19:
        subtype = _uint(me, 5, 3)
        vr_sign, vr = me[36], _uint(me, 37, 9)
        if vr:
            f["vertical_fpm"] = (vr - 1) * 64 * (-1 if vr_sign else 1)
        if subtype in (1, 2):
            scale = 4 if subtype == 2 else 1
            ew_sign, ew = me[13], _uint(me, 14, 10)
            ns_sign, ns = me[24], _uint(me, 25, 10)
            if ew and ns:
                vx = (ew - 1) * scale * (-1 if ew_sign else 1)
                vy = (ns - 1) * scale * (-1 if ns_sign else 1)
                f["speed_kn"] = math.hypot(vx, vy)
                f["track"] = math.degrees(math.atan2(vx, vy)) % 360
        elif subtype in (3, 4):
            if me[13]:
                f["heading"] = _uint(me, 14, 10) * 360 / 1024
            airspeed = _uint(me, 25, 10)
            if airspeed:
                f["speed_kn"] = (airspeed - 1) * (4 if subtype == 4 else 1)
                f["airspeed"] = "TAS" if me[24] else "IAS"
    hexed = f"{_uint(bits, 0, 112):028X}"
    return AdsbMessage(icao, df, tc, f, hexed, received)


class AdsbDecoder:
    """Raw IQ at a whole number of samples per microsecond in, ADS-B messages out."""

    name = "ADS-B"

    def __init__(self, sample_rate: float, channel: str = "") -> None:
        self.sample_rate = float(sample_rate)
        sps = self.sample_rate / 1e6
        if sps < 2 or abs(sps - round(sps)) > 1e-6:
            raise ValueError("ADS-B needs a whole number (2 or more) of samples per "
                             f"microsecond; {self.sample_rate / 1e6:g} MS/s is not")
        self.sps = int(round(sps))
        self.channel = channel
        self.cpr: dict[str, _Cpr] = {}
        self.bad_messages = 0
        self.reset()

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)

    def process(self, iq: np.ndarray) -> list[AdsbMessage]:
        mag = np.abs(np.asarray(iq)).astype(np.float32)
        m = np.concatenate([self._tail, mag])
        s = self.sps
        frame = (8 + LONG_BITS) * s
        if m.size < frame:
            self._tail = m
            return []
        # Preamble: pulses at 0, 1, 3.5 and 4.5 us, quiet at 2-3 and 5.5-7.5 us.
        n = m.size - frame
        pulse = [m[k:k + n] for k in (0, 1 * s, int(3.5 * s), int(4.5 * s))]
        quiet = [m[k:k + n] for k in (int(0.5 * s), int(2 * s), int(2.5 * s),
                                      int(5.5 * s), int(6.5 * s))]
        low = np.minimum.reduce(pulse)
        high_quiet = np.maximum.reduce(quiet)
        candidates = np.flatnonzero(low > PREAMBLE_RATIO * high_quiet)
        out = []
        skip_to = -1
        now = time.time()
        for c in candidates:
            if c < skip_to:
                continue
            base = c + 8 * s
            halves = m[base:base + LONG_BITS * s].reshape(LONG_BITS, s)
            first = halves[:, : s // 2].sum(axis=1)
            second = halves[:, s // 2: (s // 2) * 2].sum(axis=1)
            bits = (first > second).astype(np.uint8)
            df = _uint(bits, 0, 5)
            if df not in (17, 18) or crc(bits) != 0:
                self.bad_messages += df in (17, 18)
                continue
            message = decode_extended(bits, self.cpr, now)
            if message is not None:
                out.append(message)
                skip_to = c + frame
        # Start positions not yet examined: the last `frame` samples, kept for next time.
        self._tail = m[-frame:]
        return out
