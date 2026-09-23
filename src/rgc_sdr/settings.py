"""Persistent receiver settings: named memories and the last-used state.

"Memory" here is the radio sense -- a named preset of frequency, rate, zoom and display
choices, like the memory channels on a receiver's front panel.

Stored as one JSON file under ~/Library/Application Support/RGC_SDR/. Loading is
deliberately forgiving: a missing, truncated or hand-edited file falls back to defaults
rather than stopping the application from starting, since these are conveniences and
losing them must never cost you the radio.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = 1

APP_DIR = Path.home() / "Library" / "Application Support" / "RGC_SDR"
SETTINGS_FILE = APP_DIR / "settings.json"


def _coerce(value, default):
    """Best-effort conversion to the default's type; fall back on nonsense."""
    if default is None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    try:
        if isinstance(default, bool):
            return bool(value)
        return type(default)(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Snapshot:
    """Everything needed to put the receiver back where it was."""

    freq_hz: float = 7.1e6
    sample_rate: float = 768e3
    decimation: int = 1
    fft_size: int = 4096
    colormap: str = "inferno"
    #: None means "fit the colour range to the signal", which is the default behaviour.
    min_db: float | None = None
    max_db: float | None = None
    agc: bool = True
    peak_hold: bool = True
    #: "off" or one of dsp.demod.MODES.
    mode: str = "off"
    volume: float = 0.4
    offset_hz: float = 0.0
    #: None means squelch disabled.
    squelch_dbfs: float | None = None
    #: None means use the mode's default channel width.
    bandwidth_hz: float | None = None
    #: Tuning increment for the arrow keys and sideways swipes.
    step_hz: float = 10e3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "Snapshot":
        snap = cls()
        if not isinstance(data, dict):
            return snap
        known = {f.name: f.default for f in fields(cls)}
        for name, default in known.items():
            if name in data:
                setattr(snap, name, _coerce(data[name], default))
        if snap.decimation < 1:
            snap.decimation = 1
        return snap

    def describe(self) -> str:
        text = f"{self.freq_hz / 1e6:.4f} MHz"
        if self.mode and self.mode != "off":
            text += f" {self.mode.upper()}"
        if self.decimation > 1:
            text += f" ({self.decimation}x)"
        return text


@dataclass
class Memory:
    name: str
    snapshot: Snapshot = field(default_factory=Snapshot)


@dataclass
class FoundChannel:
    """A channel the scanner discovered.

    Kept in its own list, separate from `Memory`: memories are deliberate choices the
    user made and named, while these accumulate automatically and get cleared between
    sweeps. Mixing them would mean a scan could bury a hand-saved frequency.
    """

    freq_hz: float
    level_dbfs: float = -200.0
    snr_db: float = 0.0
    count: int = 1
    last_seen: str = ""
    #: Optional user note, e.g. "Tower" or "Approach".
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "FoundChannel | None":
        if not isinstance(data, dict) or "freq_hz" not in data:
            return None
        try:
            freq = float(data["freq_hz"])
        except (TypeError, ValueError):
            return None
        return cls(
            freq_hz=freq,
            level_dbfs=_coerce(data.get("level_dbfs", -200.0), -200.0),
            snr_db=_coerce(data.get("snr_db", 0.0), 0.0),
            count=_coerce(data.get("count", 1), 1),
            last_seen=str(data.get("last_seen", "")),
            label=str(data.get("label", "")),
        )

    def describe(self) -> str:
        text = f"{self.freq_hz / 1e6:.4f} MHz"
        if self.label:
            text += f"  {self.label}"
        text += f"   {self.snr_db:.0f} dB S/N"
        if self.count > 1:
            text += f"   x{self.count}"
        return text


@dataclass
class ScanSettings:
    """The scan range, remembered between sessions."""

    start_hz: float = 118e6
    end_hz: float = 137e6
    step_hz: float = 25e3
    threshold_db: float = 10.0
    stop_on_signal: bool = True
    min_sightings: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "ScanSettings":
        out = cls()
        if not isinstance(data, dict):
            return out
        for f in fields(cls):
            if f.name in data:
                setattr(out, f.name, _coerce(data[f.name], f.default))
        if out.end_hz <= out.start_hz or out.step_hz <= 0:
            return cls()
        return out


class Settings:
    """The settings file: the last-used state plus any named memories."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else SETTINGS_FILE
        self.last: Snapshot | None = None
        self.memories: list[Memory] = []
        #: Scanner results, separate from `memories` by design.
        self.found: list[FoundChannel] = []
        #: Channels the scanner must skip on later passes.
        self.lockout: set[float] = set()
        self.scan = ScanSettings()

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        settings = cls(path)
        try:
            raw = json.loads(settings.path.read_text())
        except (OSError, ValueError):
            return settings
        if not isinstance(raw, dict):
            return settings
        if isinstance(raw.get("last"), dict):
            settings.last = Snapshot.from_dict(raw["last"])
        entries = raw.get("memories")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and str(entry.get("name", "")).strip():
                    settings.memories.append(
                        Memory(str(entry["name"]).strip(),
                               Snapshot.from_dict(entry.get("snapshot")))
                    )
            settings._sort()

        entries = raw.get("found")
        if isinstance(entries, list):
            for entry in entries:
                channel = FoundChannel.from_dict(entry)
                if channel is not None:
                    settings.found.append(channel)
            settings.found.sort(key=lambda c: c.freq_hz)

        locked = raw.get("lockout")
        if isinstance(locked, list):
            for value in locked:
                try:
                    settings.lockout.add(float(value))
                except (TypeError, ValueError):
                    continue

        settings.scan = ScanSettings.from_dict(raw.get("scan"))
        return settings

    def save(self) -> None:
        """Write atomically, so a crash mid-write cannot leave a corrupt file."""
        payload = {
            "version": SCHEMA_VERSION,
            "last": self.last.to_dict() if self.last is not None else None,
            "memories": [
                {"name": m.name, "snapshot": m.snapshot.to_dict()} for m in self.memories
            ],
            "found": [c.to_dict() for c in self.found],
            "lockout": sorted(self.lockout),
            "scan": self.scan.to_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, self.path)
            tmp = None
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)

    # -- memories ----------------------------------------------------------

    def _sort(self) -> None:
        self.memories.sort(key=lambda m: m.name.lower())

    def names(self) -> list[str]:
        return [m.name for m in self.memories]

    def get_memory(self, name: str) -> Memory | None:
        key = name.strip().lower()
        for memory in self.memories:
            if memory.name.lower() == key:
                return memory
        return None

    def add_memory(self, name: str, snapshot: Snapshot) -> bool:
        """Store a memory. Returns True if it replaced one of the same name."""
        clean = name.strip()
        if not clean:
            raise ValueError("a memory needs a name")
        existing = self.get_memory(clean)
        if existing is not None:
            existing.snapshot = snapshot
            existing.name = clean
            self._sort()
            return True
        self.memories.append(Memory(clean, snapshot))
        self._sort()
        return False

    def remove_memory(self, name: str) -> bool:
        existing = self.get_memory(name)
        if existing is None:
            return False
        self.memories.remove(existing)
        return True

    # -- scanner results ---------------------------------------------------

    def find_channel(self, freq_hz: float, tolerance_hz: float = 1.0) -> FoundChannel | None:
        for channel in self.found:
            if abs(channel.freq_hz - freq_hz) <= tolerance_hz:
                return channel
        return None

    def record_found(
        self, freq_hz: float, level_dbfs: float, snr_db: float, seen: str = ""
    ) -> FoundChannel:
        """Add or update a scanner hit, keeping the strongest reading seen."""
        existing = self.find_channel(freq_hz)
        if existing is not None:
            existing.level_dbfs = max(existing.level_dbfs, level_dbfs)
            existing.snr_db = max(existing.snr_db, snr_db)
            existing.count += 1
            if seen:
                existing.last_seen = seen
            return existing
        channel = FoundChannel(float(freq_hz), level_dbfs, snr_db, 1, seen)
        self.found.append(channel)
        self.found.sort(key=lambda c: c.freq_hz)
        return channel

    def remove_found(self, freq_hz: float, tolerance_hz: float = 1.0) -> bool:
        channel = self.find_channel(freq_hz, tolerance_hz)
        if channel is None:
            return False
        self.found.remove(channel)
        return True

    def clear_found(self) -> None:
        self.found.clear()

    def add_lockout(self, freq_hz: float) -> None:
        self.lockout.add(float(freq_hz))
        self.remove_found(freq_hz)

    def remove_lockout(self, freq_hz: float) -> bool:
        if freq_hz in self.lockout:
            self.lockout.discard(freq_hz)
            return True
        return False
