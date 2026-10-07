"""Things with a position: ships, navigation marks, shore stations and (later) aircraft.

Decoded messages are folded into one `Target` per identity, which keeps its latest
position, what is known about it, and a short trail. The map window draws these. No
Qt here, so the bookkeeping is testable on its own.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

#: Points kept per trail.
TRAIL_POINTS = 200
#: How long each kind stays on the map after it was last heard, in seconds. Ships
#: report every few seconds to every few minutes; marks and shore stations are fixed;
#: an aircraft's ACARS position reports can be ten minutes or more apart.
EXPIRY_S = {"ship": 30 * 60, "aid": 60 * 60, "base": 60 * 60, "aircraft": 20 * 60,
            "radio": 30 * 60, "station": 60 * 60}


@dataclass
class Target:
    key: str                       # "ais:503020660", later "adsb:7c6b2d"
    kind: str                      # "ship", "aid", "base", "aircraft", "radio", "station"
    ident: str                     # MMSI or ICAO address, as text
    name: str = ""
    lat: float | None = None
    lon: float | None = None
    course: float | None = None    # degrees true, over the ground
    heading: float | None = None   # degrees true, where the bow points
    speed_kn: float | None = None
    altitude_ft: float | None = None
    details: dict = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    messages: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_POINTS))

    @property
    def label(self) -> str:
        return self.name or self.ident

    @property
    def bearing(self) -> float | None:
        """Which way to draw it pointing: the heading if known, else the course."""
        return self.heading if self.heading is not None else self.course

    def describe(self) -> str:
        lines = [f"{self.label}  ({self.kind}, {self.ident})"]
        if self.lat is not None:
            lines.append(f"{self.lat:.5f}, {self.lon:.5f}")
        if self.speed_kn is not None:
            lines.append(f"{self.speed_kn:.1f} kn" + (f", course {self.course:.0f}"
                                                      if self.course is not None else ""))
        if self.heading is not None:
            lines.append(f"heading {self.heading:.0f}")
        if self.altitude_ft is not None:
            lines.append(f"{self.altitude_ft:.0f} ft")
        lines += [f"{k}: {v}" for k, v in self.details.items() if v]
        age = time.time() - self.last_seen
        lines.append(f"heard {age:.0f} s ago, {self.messages} messages")
        return "\n".join(lines)


class TargetStore:
    """Every target heard, by key."""

    def __init__(self) -> None:
        self.targets: dict[str, Target] = {}
        #: Bumped on every change, so a view can tell whether to redraw.
        self.version = 0

    def update(self, messages) -> int:
        """Fold in decoded messages of any kind; those without a position or identity
        are ignored. Returns how many targets changed."""
        changed = 0
        for message in messages:
            if getattr(message, "mmsi", None) is not None:
                changed += self._from_ais(message)
            elif getattr(message, "registration", None) is not None:
                changed += self._from_acars(message)
            elif getattr(message, "icao", None) is not None:
                changed += self._from_adsb(message)
            elif getattr(message, "nac", None) is not None:
                changed += self._from_p25(message)
            elif hasattr(message, "colour_code"):
                changed += self._from_dmr(message)
            elif hasattr(message, "object_name"):
                changed += self._from_aprs(message)
        if changed:
            self.version += 1
        return changed

    def _get(self, key: str, kind: str, ident: str) -> Target:
        target = self.targets.get(key)
        if target is None:
            target = self.targets[key] = Target(key, kind, ident)
        return target

    def _from_ais(self, m) -> int:
        f = m.fields
        kind = "aid" if m.msg_type == 21 else "base" if m.msg_type in (4, 11) else "ship"
        t = self._get(f"ais:{m.mmsi}", kind, f"{m.mmsi:09d}")
        t.last_seen = m.received
        t.messages += 1
        name = f.get("name") or m.name
        if name:
            t.name = name
        for key in ("callsign", "destination", "ship_type", "status", "aid_type"):
            if f.get(key):
                t.details[key] = f[key]
        if f.get("sog") is not None:
            t.speed_kn = f["sog"]
        if f.get("cog") is not None:
            t.course = f["cog"]
        if "heading" in f:
            t.heading = f["heading"]
        position = f.get("position")
        if position is not None:
            t.lat, t.lon = position
            if not t.trail or t.trail[-1] != position:
                t.trail.append(position)
        return 1

    def _from_acars(self, m) -> int:
        """Aircraft by registration, ground stations by ICAO code; only with a position
        or an already-placed target, since most ACARS messages carry none."""
        if m.registration:
            key, kind, ident = f"acars:{m.registration}", "aircraft", m.registration
        elif m.ground_station:
            key, kind, ident = f"acars:{m.ground_station}", "base", m.ground_station
        else:
            return 0
        if m.position is None and key not in self.targets:
            return 0
        t = self._get(key, kind, ident)
        t.last_seen = m.received
        t.messages += 1
        if m.flight:
            t.name = m.flight
            t.details["registration"] = m.registration
        if m.position is not None:
            t.lat, t.lon = m.position
            if not t.trail or t.trail[-1] != m.position:
                t.trail.append(m.position)
        return 1

    def _from_adsb(self, m) -> int:
        f = m.fields
        t = self._get(f"adsb:{m.icao}", "aircraft", m.icao)
        t.last_seen = m.received
        t.messages += 1
        if f.get("callsign"):
            t.name = f["callsign"]
        if f.get("altitude_ft") is not None:
            t.altitude_ft = f["altitude_ft"]
        if f.get("speed_kn") is not None:
            t.speed_kn = f["speed_kn"]
        if f.get("track") is not None:
            t.course = f["track"]
        if f.get("heading") is not None:
            t.heading = f["heading"]
        if f.get("vertical_fpm") is not None:
            t.details["climb"] = f"{f['vertical_fpm']:+d} ft/min"
        position = f.get("position")
        if position is not None:
            t.lat, t.lon = position
            if not t.trail or t.trail[-1] != position:
                t.trail.append(position)
        return 1

    def _from_p25(self, m) -> int:
        """A P25 radio that reported its position (LRRP), by its radio ID; placed only
        with a position, as a radio has no other identity worth a map entry."""
        if m.position is None or m.radio is None:
            return 0
        t = self._get(f"p25:{m.nac:03X}:{m.radio}", "radio", str(m.radio))
        t.last_seen = m.received
        t.messages += 1
        t.details["network"] = f"P25 NAC {m.nac:03X}"
        t.details["position"] = "location reported by the radio (LRRP)"
        t.lat, t.lon = m.position
        if not t.trail or t.trail[-1] != m.position:
            t.trail.append(m.position)
        return 1

    def _from_dmr(self, m) -> int:
        """A DMR radio that reported its position (LRRP), by its radio ID."""
        if getattr(m, "position", None) is None or m.radio is None:
            return 0
        t = self._get(f"dmr:{m.colour_code}:{m.radio}", "radio", str(m.radio))
        t.last_seen = m.received
        t.messages += 1
        t.details["network"] = f"DMR CC {m.colour_code}"
        t.details["position"] = "location reported by the radio (LRRP)"
        t.lat, t.lon = m.position
        if not t.trail or t.trail[-1] != m.position:
            t.trail.append(m.position)
        return 1

    def _from_aprs(self, m) -> int:
        """An APRS station by callsign, or an object or item by its name; placed only
        with a position."""
        if m.position is None or not m.source:
            return 0
        name = m.object_name
        t = self._get(f"aprs:{name or m.source}", "station", name or m.source)
        t.last_seen = m.received
        t.messages += 1
        if name:
            t.details["sent by"] = m.source
        t.details["via"] = ",".join(m.path) or "direct"
        t.details["last"] = m.text[:80]
        t.course, t.speed_kn = m.course, m.speed_kn
        t.lat, t.lon = m.position
        if not t.trail or t.trail[-1] != m.position:
            t.trail.append(m.position)
        return 1

    def expire(self, now: float | None = None) -> int:
        """Forget targets not heard for their kind's expiry time."""
        now = time.time() if now is None else now
        old = [k for k, t in self.targets.items()
               if now - t.last_seen > EXPIRY_S.get(t.kind, 30 * 60)]
        for k in old:
            del self.targets[k]
        if old:
            self.version += 1
        return len(old)

    def placed(self) -> list[Target]:
        """Targets with a known position."""
        return [t for t in self.targets.values() if t.lat is not None]
