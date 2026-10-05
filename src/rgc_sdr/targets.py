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
#: report every few seconds to every few minutes; marks and shore stations are fixed.
EXPIRY_S = {"ship": 30 * 60, "aid": 60 * 60, "base": 60 * 60, "aircraft": 3 * 60}


@dataclass
class Target:
    key: str                       # "ais:503020660", later "adsb:7c6b2d"
    kind: str                      # "ship", "aid", "base" or "aircraft"
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
