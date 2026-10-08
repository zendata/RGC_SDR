"""IQ recordings on disk: SigMF, and the app's earlier `.cf32` + JSON format (P12).

SigMF (sigmf.org) is the open standard for recorded signals: a raw `.sigmf-data` file
and a `.sigmf-meta` JSON file beside it, with the sample format, rate and centre
frequency. Recordings made here are SigMF from P12 on, so they open in SDR++, GNU Radio,
inspectrum and the rest; and SigMF recordings from those tools play here.

The data formats read are the common ones: complex float32 (`cf32_le`), complex int16
(`ci16_le`, most SDR tools), and complex int8 / uint8 (`ci8`, `cu8`, the HackRF's and
the RTL-SDR's native formats). Writing is always `cf32_le`, which is what the app holds.

No Qt and no DSP imports (layering rule, PLANNING.md section 5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SIGMF_VERSION = "1.0.0"
DATA_SUFFIX = ".sigmf-data"
META_SUFFIX = ".sigmf-meta"
LEGACY_SUFFIX = ".cf32"

#: SigMF datatype -> (NumPy dtype of one component, scale to +/-1, offset).
DATATYPES = {
    "cf32_le": ("<f4", 1.0, 0.0),
    "ci16_le": ("<i2", 1.0 / 32768.0, 0.0),
    "ci8": ("i1", 1.0 / 128.0, 0.0),
    "cu8": ("u1", 1.0 / 128.0, -127.5),
}


@dataclass(frozen=True)
class Recording:
    """What is needed to play a recording."""

    data_path: Path
    datatype: str
    sample_rate: float
    center_freq: float
    #: Everything the metadata said, for display.
    meta: dict

    @property
    def component_dtype(self) -> np.dtype:
        return np.dtype(DATATYPES[self.datatype][0])

    @property
    def samples(self) -> int:
        """Complex samples in the file."""
        return self.data_path.stat().st_size // (2 * self.component_dtype.itemsize)

    def open(self) -> np.ndarray:
        """The raw components, memory-mapped: I, Q, I, Q ..."""
        return np.memmap(self.data_path, dtype=self.component_dtype, mode="r")

    def to_complex(self, components: np.ndarray) -> np.ndarray:
        """Raw I, Q components to complex64 at the app's full scale of 1.0."""
        _, scale, offset = DATATYPES[self.datatype]
        if self.datatype == "cf32_le":
            return np.ascontiguousarray(components).view(np.complex64).copy()
        x = (components.astype(np.float32) + np.float32(offset)) * np.float32(scale)
        return x.view(np.complex64).copy()


def meta_path_for(data_path: Path) -> Path:
    return Path(data_path).with_suffix(META_SUFFIX)


def data_path_for(meta_path: Path) -> Path:
    return Path(meta_path).with_suffix(DATA_SUFFIX)


def sigmf_meta(sample_rate: float, center_freq: float, started: datetime | None = None,
               hw: str = "", extras: dict | None = None) -> dict:
    """A SigMF metadata document for one capture of cf32_le samples."""
    started = started or datetime.now(timezone.utc)
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(sample_rate),
            "core:version": SIGMF_VERSION,
            "core:recorder": "RGC_SDR",
        },
        "captures": [{
            "core:sample_start": 0,
            "core:frequency": float(center_freq),
            "core:datetime": started.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"),
        }],
        "annotations": [],
    }
    if hw:
        meta["global"]["core:hw"] = hw
    for key, value in (extras or {}).items():
        meta["global"][f"rgc_sdr:{key}"] = value
    return meta


def write_sigmf(data_path: Path, samples: np.ndarray, sample_rate: float,
                center_freq: float, hw: str = "", extras: dict | None = None) -> Path:
    """Write `samples` as a SigMF recording; returns the data file's path."""
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(samples, dtype=np.complex64).tofile(data_path)
    meta = sigmf_meta(sample_rate, center_freq, hw=hw, extras=extras)
    meta_path_for(data_path).write_text(json.dumps(meta, indent=2))
    return data_path


def _from_sigmf(meta_path: Path) -> Recording:
    try:
        meta = json.loads(meta_path.read_text())
    except FileNotFoundError:
        raise ValueError(f"no {meta_path.name} beside the recording") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{meta_path.name} is not valid JSON: {exc}") from None
    glob = meta.get("global", {}) if isinstance(meta, dict) else {}
    datatype = str(glob.get("core:datatype", ""))
    if datatype not in DATATYPES:
        raise ValueError(f"unsupported SigMF datatype {datatype!r}; "
                         f"plays {', '.join(DATATYPES)}")
    rate = glob.get("core:sample_rate")
    captures = meta.get("captures") or [{}]
    freq = captures[0].get("core:frequency") if isinstance(captures[0], dict) else None
    if not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError("the SigMF metadata has no usable core:sample_rate")
    if not isinstance(freq, (int, float)) or freq <= 0:
        raise ValueError("the SigMF metadata has no usable core:frequency")
    data = data_path_for(meta_path)
    if not data.is_file():
        raise ValueError(f"no {data.name} beside the metadata")
    return Recording(data, datatype, float(rate), float(freq), meta)


def _from_legacy(path: Path) -> Recording:
    sidecar = path.with_suffix(path.suffix + ".json")
    try:
        meta = json.loads(sidecar.read_text())
    except FileNotFoundError:
        raise ValueError(f"no sidecar {sidecar.name} beside the recording") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"sidecar is not valid JSON: {exc}") from None
    if meta.get("format") != "complex64" or meta.get("byte_order", "little") != "little":
        raise ValueError(f"unsupported format {meta.get('format')!r}; expected complex64")
    for key in ("sample_rate_hz", "center_freq_hz"):
        if not isinstance(meta.get(key), (int, float)) or meta[key] <= 0:
            raise ValueError(f"sidecar has no usable {key}")
    return Recording(path, "cf32_le", float(meta["sample_rate_hz"]),
                     float(meta["center_freq_hz"]), meta)


def read_recording(path) -> Recording:
    """A recording from either of its SigMF files, or a `.cf32` with its JSON sidecar.
    Raises ValueError for anything that cannot be played."""
    path = Path(path)
    if path.suffix in (DATA_SUFFIX, META_SUFFIX):
        return _from_sigmf(meta_path_for(path))
    return _from_legacy(path)
