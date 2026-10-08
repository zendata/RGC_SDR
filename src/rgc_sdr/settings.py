"""Persistent receiver settings: named memories and the last-used state.

"Memory" here is the radio sense -- a named preset of frequency, rate, zoom and display
choices, like the memory channels on a receiver's front panel.

Settings that belong to a *radio* rather than a station -- sample rate, zoom, gains, AGC,
IF bandwidth, bias-tee, colour levels -- are kept per radio type (`RadioSettings`), both
as each radio's last-used state and inside each memory. So a station saved on one radio
recalls on another with that radio's own settings: a HackRF's gains mean nothing to an
Airspy, and its noise floor sits some 50 dB higher.

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

SCHEMA_VERSION = 2

#: Every memory saved before per-radio settings existed was made on the Airspy HF+ (the
#: only radio that streamed), so that is where their rate and zoom are filed.
LEGACY_RADIO = "airspyhf"

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
    #: Round tuning to a multiple of the step, for channelised bands.
    snap: bool = False
    #: CW beat-note pitch.
    pitch_hz: float = 500.0
    #: Broadcast FM in stereo when the station sends it; False forces mono.
    stereo: bool = True
    #: NBFM repeater split for transmit: "simplex", "plus" or "minus".
    repeater_shift: str = "simplex"
    #: The split; None means the band's standard (600 kHz on 2 m, 5 MHz on 70 cm).
    repeater_offset_hz: float | None = None
    #: NBFM signalling: "off", "tone" (CTCSS sent), "tsql" (CTCSS sent and required on
    #: receive) or "dcs" (DCS sent and required on receive).
    tone_mode: str = "off"
    ctcss_hz: float = 88.5
    dcs_code: str = "023"
    #: Data decoder to run ("" for none, else a key of decoding.DECODERS), so a memory
    #: such as "APRS 2m" brings its decoder with it.
    decoder: str = ""
    #: The DSP row (P10, PLANNING.md 7r): passband shift, noise blanker and reduction
    #: levels (0 off), notches (absolute Hz), AGC mode and its fixed gain, SAM sideband,
    #: and the NBFM noise squelch.
    if_shift_hz: float = 0.0
    nb_level: int = 0
    nr_level: int = 0
    auto_notch: bool = False
    notches_hz: tuple = ()
    agc_mode: str = "medium"
    manual_gain_db: float = 60.0
    sam_sideband: str = "both"
    noise_squelch: bool = False
    quieting_db: float = 10.0

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
        try:
            snap.notches_hz = tuple(float(f) for f in snap.notches_hz)
        except (TypeError, ValueError):
            snap.notches_hz = ()
        return snap

    def describe(self) -> str:
        text = f"{self.freq_hz / 1e6:.4f} MHz"
        if self.mode and self.mode != "off":
            text += f" {self.mode.upper()}"
        if self.decimation > 1:
            text += f" ({self.decimation}x)"
        return text


@dataclass
class RadioSettings:
    """What belongs to one radio type rather than to a station."""

    #: None means the radio's default.
    sample_rate: float | None = None
    decimation: int = 1
    #: Gain stage name -> dB, as the driver names them.
    gains: dict[str, float] = field(default_factory=dict)
    agc: bool | None = None
    if_bandwidth_hz: float | None = None
    #: Boolean driver settings (bias-tee and the like), by driver key.
    driver_settings: dict[str, bool] = field(default_factory=dict)
    #: Colour range; None means fit to the signal.
    min_db: float | None = None
    max_db: float | None = None
    #: Transmit gain stage name -> dB, for radios that transmit.
    tx_gains: dict[str, float] = field(default_factory=dict)
    #: Microphone level sent to a transceiver (0-1); None means the default.
    tx_audio_level: float | None = None
    #: Microphone gain (dB) for an SDR that transmits; None means the default.
    tx_mic_gain_db: float | None = None
    #: The radio's frequency error, ppm (+ reads high); None until it is calibrated.
    ppm: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "RadioSettings":
        out = cls()
        if not isinstance(data, dict):
            return out
        for name in ("sample_rate", "if_bandwidth_hz", "min_db", "max_db", "tx_audio_level",
                     "tx_mic_gain_db", "ppm"):
            if name in data:
                setattr(out, name, _coerce(data[name], None))
        out.decimation = max(1, _coerce(data.get("decimation", 1), 1))
        if data.get("agc") is not None:
            out.agc = bool(data["agc"])
        for attr in ("gains", "tx_gains"):
            gains = data.get(attr)
            if isinstance(gains, dict):
                for key, value in gains.items():
                    db = _coerce(value, None)
                    if db is not None:
                        getattr(out, attr)[str(key)] = db
        flags = data.get("driver_settings")
        if isinstance(flags, dict):
            out.driver_settings = {str(k): bool(v) for k, v in flags.items()}
        return out

    @classmethod
    def from_snapshot(cls, snap: Snapshot) -> "RadioSettings":
        """The radio half of a pre-version-2 snapshot."""
        return cls(sample_rate=snap.sample_rate, decimation=snap.decimation,
                   agc=snap.agc, min_db=snap.min_db, max_db=snap.max_db)


def _radios_from(data: object) -> dict[str, RadioSettings]:
    if not isinstance(data, dict):
        return {}
    return {str(k): RadioSettings.from_dict(v) for k, v in data.items() if k}


@dataclass
class Memory:
    name: str
    snapshot: Snapshot = field(default_factory=Snapshot)
    #: Per radio type: how this station was set up on that radio.
    radios: dict[str, RadioSettings] = field(default_factory=dict)
    #: The memory group it is filed under ("" until placed: see `default_group`).
    group: str = ""


#: The memory groups to start with (VK3RQ, 2026-10-07), in this order.
DEFAULT_GROUPS = ("LW", "AM broadcast", "FM broadcast", "HF", "VHF", "UHF", "Airband",
                  "Satellites", "Air Nav/Data", "P25", "DMR", "DAB+")


def default_group(snapshot: Snapshot) -> str:
    """Where a memory belongs, from its mode, decoder and frequency: the kind of signal
    first (a P25 control channel is P25 whatever its band), then the band."""
    f, mode, decoder = snapshot.freq_hz, snapshot.mode, snapshot.decoder
    if mode == "dab":
        return "DAB+"
    if mode == "p25" or decoder == "p25":
        return "P25"
    if decoder == "dmr":
        return "DMR"
    if decoder in ("acars", "adsb"):
        return "Air Nav/Data"
    if f < 300e3:
        return "LW"
    if f < 1.7e6:
        return "AM broadcast"
    if f < 30e6:
        return "HF"
    if 87.5e6 <= f <= 108e6 and mode in ("wbfm", "off"):
        return "FM broadcast"
    if 108e6 <= f < 118e6:
        return "Air Nav/Data"                 # VOR and ILS
    if 118e6 <= f < 137e6:
        return "Airband"
    # Weather satellites at 137 MHz; the amateur satellite sub-bands on 2 m and 70 cm.
    if 137e6 <= f < 138e6 or 145.8e6 <= f < 146e6 or 435e6 <= f < 438e6:
        return "Satellites"
    if f < 300e6:
        return "VHF"
    return "UHF"


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
        #: Memory groups, in display order (`DEFAULT_GROUPS` to start).
        self.memory_groups: list[str] = list(DEFAULT_GROUPS)
        #: Which panels the tab row had open (settings, decode, classify, memory, map,
        #: scan).
        self.open_panels: list[str] = ["settings"]
        #: Spectrum overlays (P11): the band plan, and memory names.
        self.show_band_plan = True
        self.show_memory_names = False
        #: Seconds of IQ kept for Replay (P12); 0 keeps none.
        self.replay_seconds = 30.0
        #: The radio last used, by profile key.
        self.device: str | None = None
        #: The radio last chosen from the list: first choice wherever it reaches.
        self.preferred_device: str | None = None
        #: What each radio reported it could tune when last opened, as (low, high) Hz
        #: pairs: trusted over the profile's nominal ranges (a Pluto may have had the
        #: 70-6000 MHz firmware change).
        self.radio_ranges: dict[str, list[tuple[float, float]]] = {}
        #: Each radio type's last-used settings, restored when it is opened again.
        self.radios: dict[str, RadioSettings] = {}

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
                    snap = Snapshot.from_dict(entry.get("snapshot"))
                    if "radios" in entry:
                        radios = _radios_from(entry["radios"])
                    else:
                        radios = {LEGACY_RADIO: RadioSettings.from_snapshot(snap)}
                    group = str(entry.get("group", "")).strip() or default_group(snap)
                    settings.memories.append(Memory(str(entry["name"]).strip(), snap, radios,
                                                    group))
            settings._sort()
        for key in ("show_band_plan", "show_memory_names"):
            if isinstance(raw.get(key), bool):
                setattr(settings, key, raw[key])
        replay = raw.get("replay_seconds")
        if isinstance(replay, (int, float)) and not isinstance(replay, bool) and replay >= 0:
            settings.replay_seconds = float(replay)
        panels = raw.get("open_panels")
        if isinstance(panels, list):
            settings.open_panels = [str(p) for p in panels]
        groups = raw.get("memory_groups")
        if isinstance(groups, list):
            settings.memory_groups = [str(g) for g in groups if str(g).strip()]
        for memory in settings.memories:                 # a group a memory names is kept
            settings._ensure_group(memory.group)

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
        device = raw.get("device")
        settings.device = str(device) if isinstance(device, str) and device else None
        settings.radios = _radios_from(raw.get("radios"))
        preferred = raw.get("preferred_device")
        settings.preferred_device = (str(preferred) if isinstance(preferred, str) and preferred
                                     else None)
        ranges = raw.get("radio_ranges")
        if isinstance(ranges, dict):
            for key, pairs in ranges.items():
                try:
                    settings.radio_ranges[str(key)] = [(float(lo), float(hi)) for lo, hi in pairs]
                except (TypeError, ValueError):
                    continue
        if (not settings.radios and settings.last is not None and settings.device
                and raw.get("version", 1) < 2):
            # Only when the file says which radio the last state came from: a HackRF's
            # 2 MS/s filed under the Airspy would be nonsense.
            settings.radios[settings.device] = RadioSettings.from_snapshot(settings.last)
        return settings

    def save(self) -> None:
        """Write atomically, so a crash mid-write cannot leave a corrupt file."""
        payload = {
            "version": SCHEMA_VERSION,
            "last": self.last.to_dict() if self.last is not None else None,
            "memories": [
                {"name": m.name, "snapshot": m.snapshot.to_dict(),
                 "radios": {k: r.to_dict() for k, r in m.radios.items()}, "group": m.group}
                for m in self.memories
            ],
            "memory_groups": list(self.memory_groups),
            "open_panels": list(self.open_panels),
            "show_band_plan": self.show_band_plan,
            "show_memory_names": self.show_memory_names,
            "replay_seconds": self.replay_seconds,
            "found": [c.to_dict() for c in self.found],
            "lockout": sorted(self.lockout),
            "scan": self.scan.to_dict(),
            "device": self.device,
            "preferred_device": self.preferred_device,
            "radio_ranges": {k: [list(r) for r in v] for k, v in self.radio_ranges.items()},
            "radios": {k: r.to_dict() for k, r in self.radios.items()},
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

    def add_memory(
        self,
        name: str,
        snapshot: Snapshot,
        radio_key: str | None = None,
        radio: RadioSettings | None = None,
    ) -> bool:
        """Store a memory. Returns True if it replaced one of the same name.

        Saving over an existing memory replaces the station and *this* radio's settings,
        and keeps every other radio's: re-saving a station on the HackRF must not throw
        away how it was set up on the Airspy.
        """
        clean = name.strip()
        if not clean:
            raise ValueError("a memory needs a name")
        existing = self.get_memory(clean)
        if existing is not None:
            existing.snapshot = snapshot
            existing.name = clean
            if radio_key and radio is not None:
                existing.radios[radio_key] = radio
            self._sort()
            return True
        radios = {radio_key: radio} if radio_key and radio is not None else {}
        group = default_group(snapshot)
        self.memories.append(Memory(clean, snapshot, radios, group))
        self._ensure_group(group)
        self._sort()
        return False

    # -- memory groups -----------------------------------------------------

    def _ensure_group(self, group: str) -> None:
        if group and group not in self.memory_groups:
            self.memory_groups.append(group)

    def memories_in(self, group: str) -> list[Memory]:
        return [m for m in self.memories if m.group == group]

    def move_memory(self, name: str, group: str) -> bool:
        """File memory `name` under `group` (made if new). False if no such memory."""
        memory = self.get_memory(name)
        if memory is None or not group.strip():
            return False
        memory.group = group.strip()
        self._ensure_group(memory.group)
        return True

    def add_group(self, group: str) -> bool:
        clean = group.strip()
        if not clean or clean in self.memory_groups:
            return False
        self.memory_groups.append(clean)
        return True

    def rename_group(self, old: str, new: str) -> bool:
        clean = new.strip()
        if old not in self.memory_groups or not clean or clean in self.memory_groups:
            return False
        self.memory_groups[self.memory_groups.index(old)] = clean
        for memory in self.memories:
            if memory.group == old:
                memory.group = clean
        return True

    def remove_group(self, group: str) -> bool:
        """Drop an empty group. False if it holds memories, or does not exist."""
        if group not in self.memory_groups or self.memories_in(group):
            return False
        self.memory_groups.remove(group)
        return True

    def rename_memory(self, old: str, new: str) -> bool:
        memory, clean = self.get_memory(old), new.strip()
        if memory is None or not clean or (self.get_memory(clean) not in (None, memory)):
            return False
        memory.name = clean
        self._sort()
        return True

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
