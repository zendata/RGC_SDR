"""Memories in and out as CSV (P13, PLANNING.md 7u): the app's own, and CHIRP's.

* **The app's CSV** has one memory a row, readable columns first -- name, group,
  frequency, mode, width, decoder and the repeater settings -- then the whole station
  and every radio's setup as JSON, so nothing is lost on the way out and back. A file
  edited in a spreadsheet keeps working: the readable columns win over the JSON, and a
  row without JSON is built from the columns alone.
* **CHIRP's CSV** (the "generic CSV" CHIRP imports and exports) is how memories move to
  and from the many radios CHIRP programs, the IC-705 among them. Only what CHIRP can
  hold goes out: analogue modes, repeater shift, CTCSS and DCS; the group goes in the
  comment. DMR channels come in with the DMR decoder chosen.

No Qt; the memory panel's Import and Export call these.
"""

from __future__ import annotations

import csv
import json

from .repeater import MINUS, PLUS, SIMPLEX, band_for
from .settings import Memory, RadioSettings, Settings, Snapshot, default_group

OWN_COLUMNS = ("name", "group", "frequency_mhz", "mode", "bandwidth_hz", "decoder",
               "repeater_shift", "repeater_offset_hz", "tone_mode", "ctcss_hz",
               "dcs_code", "station_json", "radios_json")
CHIRP_COLUMNS = ("Location", "Name", "Frequency", "Duplex", "Offset", "Tone", "rToneFreq",
                 "cToneFreq", "DtcsCode", "DtcsPolarity", "RxDtcsCode", "CrossMode", "Mode",
                 "TStep", "Skip", "Power", "Comment", "URCALL", "RPT1CALL", "RPT2CALL",
                 "DVCODE")
#: Tuning steps CHIRP accepts, kHz.
CHIRP_STEPS = (5.0, 6.25, 8.33, 10.0, 12.5, 15.0, 20.0, 25.0, 50.0, 100.0)
_TONE_TO_CHIRP = {"tone": "Tone", "tsql": "TSQL", "dcs": "DTCS"}
_TONE_FROM_CHIRP = {"Tone": "tone", "TSQL": "tsql", "DTCS": "dcs", "Cross": "tone"}
#: CHIRP's mode -> (the app's mode, channel width, decoder).
_MODE_FROM_CHIRP = {
    "FM": ("nbfm", 16e3, ""), "NFM": ("nbfm", 12.5e3, ""), "WFM": ("wbfm", 200e3, ""),
    "AM": ("am", 9e3, ""), "NAM": ("am", 6e3, ""), "USB": ("usb", 2.7e3, ""),
    "LSB": ("lsb", 2.7e3, ""), "CW": ("cw", 500.0, ""), "DMR": ("nbfm", 12.5e3, "dmr"),
    "DV": ("nbfm", 12.5e3, ""), "DN": ("nbfm", 12.5e3, ""),
}


def is_chirp(path) -> bool:
    """Whether a CSV file is CHIRP's (its header starts with Location)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    return bool(header) and header[0].strip() == "Location"


# -- the app's own CSV -------------------------------------------------------------------


def export_csv(memories: list[Memory], path) -> int:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OWN_COLUMNS)
        writer.writeheader()
        for m in memories:
            snap = m.snapshot
            writer.writerow({
                "name": m.name, "group": m.group,
                "frequency_mhz": f"{snap.freq_hz / 1e6:.6f}", "mode": snap.mode,
                "bandwidth_hz": "" if snap.bandwidth_hz is None else f"{snap.bandwidth_hz:g}",
                "decoder": snap.decoder, "repeater_shift": snap.repeater_shift,
                "repeater_offset_hz": ("" if snap.repeater_offset_hz is None
                                       else f"{snap.repeater_offset_hz:g}"),
                "tone_mode": snap.tone_mode, "ctcss_hz": f"{snap.ctcss_hz:g}",
                "dcs_code": snap.dcs_code,
                "station_json": json.dumps(snap.to_dict(), separators=(",", ":")),
                "radios_json": json.dumps({k: r.to_dict() for k, r in m.radios.items()},
                                          separators=(",", ":")),
            })
    return len(memories)


def _number(text, default=None):
    try:
        return float(text) if str(text).strip() != "" else default
    except ValueError:
        return default


def import_csv(path) -> list[Memory]:
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("name") or "").strip()
            mhz = _number(row.get("frequency_mhz"))
            if not name or mhz is None:
                continue
            try:
                snap = Snapshot.from_dict(json.loads(row.get("station_json") or "{}"))
            except ValueError:
                snap = Snapshot()
            # The readable columns win: they are what a spreadsheet edits.
            snap.freq_hz = mhz * 1e6
            for key in ("mode", "decoder", "repeater_shift", "tone_mode", "dcs_code"):
                if (row.get(key) or "").strip():
                    setattr(snap, key, row[key].strip())
            for key in ("bandwidth_hz", "repeater_offset_hz", "ctcss_hz"):
                value = _number(row.get(key))
                if value is not None:
                    setattr(snap, key, value)
            radios = {}
            try:
                radios = {str(k): RadioSettings.from_dict(v)
                          for k, v in json.loads(row.get("radios_json") or "{}").items()}
            except (ValueError, AttributeError):
                pass
            group = (row.get("group") or "").strip() or default_group(snap)
            out.append(Memory(name, snap, radios, group))
    return out


# -- CHIRP's CSV ----------------------------------------------------------------------------


def _chirp_mode(snap: Snapshot) -> str | None:
    if snap.mode == "nbfm":
        return "NFM" if (snap.bandwidth_hz or 12.5e3) <= 12.5e3 else "FM"
    return {"wbfm": "WFM", "am": "AM", "sam": "AM", "usb": "USB", "lsb": "LSB",
            "cw": "CW"}.get(snap.mode)


def export_chirp(memories: list[Memory], path) -> int:
    """CHIRP's generic CSV; returns how many went out (digital modes stay behind)."""
    rows = []
    for m in memories:
        snap = m.snapshot
        mode = _chirp_mode(snap)
        if mode is None:
            continue
        offset = snap.repeater_offset_hz
        if offset is None:
            band = band_for(snap.freq_hz)
            offset = band.offset_hz if band is not None else 0.0
        duplex = {PLUS: "+", MINUS: "-"}.get(snap.repeater_shift, "")
        step = min(CHIRP_STEPS, key=lambda s: abs(s - snap.step_hz / 1e3))
        rows.append({
            "Location": len(rows), "Name": m.name,
            "Frequency": f"{snap.freq_hz / 1e6:.6f}", "Duplex": duplex,
            "Offset": f"{(offset if duplex else 0.0) / 1e6:.6f}",
            "Tone": _TONE_TO_CHIRP.get(snap.tone_mode, ""),
            "rToneFreq": f"{snap.ctcss_hz:.1f}", "cToneFreq": f"{snap.ctcss_hz:.1f}",
            "DtcsCode": snap.dcs_code, "DtcsPolarity": "NN", "RxDtcsCode": snap.dcs_code,
            "CrossMode": "Tone->Tone", "Mode": mode, "TStep": f"{step:.2f}", "Skip": "",
            "Power": "", "Comment": m.group, "URCALL": "", "RPT1CALL": "", "RPT2CALL": "",
            "DVCODE": "",
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHIRP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def import_chirp(path, groups=()) -> list[Memory]:
    """Memories from CHIRP's CSV. A comment that names one of `groups` files it there."""
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mhz = _number(row.get("Frequency"))
            mode_name = (row.get("Mode") or "FM").strip()
            if mhz is None or mhz <= 0 or mode_name not in _MODE_FROM_CHIRP:
                continue
            mode, width, decoder = _MODE_FROM_CHIRP[mode_name]
            snap = Snapshot(freq_hz=mhz * 1e6, mode=mode, bandwidth_hz=width, decoder=decoder)
            duplex = (row.get("Duplex") or "").strip()
            offset = _number(row.get("Offset"), 0.0)
            if duplex in ("+", "-") and offset:
                snap.repeater_shift = PLUS if duplex == "+" else MINUS
                snap.repeater_offset_hz = offset * 1e6
            else:
                snap.repeater_shift = SIMPLEX
            tone = _TONE_FROM_CHIRP.get((row.get("Tone") or "").strip())
            if tone is not None:
                snap.tone_mode = tone
                key = "cToneFreq" if tone == "tsql" else "rToneFreq"
                snap.ctcss_hz = _number(row.get(key), 88.5)
                if tone == "dcs":
                    snap.dcs_code = (row.get("DtcsCode") or "023").strip().zfill(3)
            step = _number(row.get("TStep"))
            if step:
                snap.step_hz = step * 1e3
            name = (row.get("Name") or "").strip() or f"CH{row.get('Location', '')} {mhz:.4f}"
            comment = (row.get("Comment") or "").strip()
            group = comment if comment in groups else default_group(snap)
            out.append(Memory(name, snap, {}, group))
    return out


def merge(settings: Settings, memories: list[Memory]) -> int:
    """Add imported memories; one with a name already used replaces it, keeping the
    setups of radios the import does not mention. Returns how many were added or
    replaced."""
    for incoming in memories:
        existing = settings.get_memory(incoming.name)
        if existing is not None:
            existing.snapshot = incoming.snapshot
            existing.radios.update(incoming.radios)
            existing.group = incoming.group or existing.group
            settings.move_memory(existing.name, existing.group)
        else:
            settings.memories.append(incoming)
            settings.move_memory(incoming.name, incoming.group)
    settings.memories.sort(key=lambda m: m.name.lower())
    return len(memories)


def export_any(memories: list[Memory], path, chirp: bool = False) -> int:
    return export_chirp(memories, path) if chirp else export_csv(memories, path)


def import_any(path, groups=()) -> list[Memory]:
    return import_chirp(path, groups) if is_chirp(path) else import_csv(path)

