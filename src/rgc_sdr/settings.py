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
        if self.decimation > 1:
            text += f" ({self.decimation}x)"
        return text


@dataclass
class Memory:
    name: str
    snapshot: Snapshot = field(default_factory=Snapshot)


class Settings:
    """The settings file: the last-used state plus any named memories."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else SETTINGS_FILE
        self.last: Snapshot | None = None
        self.memories: list[Memory] = []

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
        return settings

    def save(self) -> None:
        """Write atomically, so a crash mid-write cannot leave a corrupt file."""
        payload = {
            "version": SCHEMA_VERSION,
            "last": self.last.to_dict() if self.last is not None else None,
            "memories": [
                {"name": m.name, "snapshot": m.snapshot.to_dict()} for m in self.memories
            ],
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
