"""Playback of IQ recordings, as a source.

Plays back what a real radio captured (PLANNING.md section 1 allows this; it is not a
simulated device). A reader thread writes the file into the same `_Ring` a radio uses,
paced to the recorded sample rate, so the display, audio, decoders and recorder see it
just as they would see the radio.

Tuning moves a virtual centre within the recorded span with the same `_Nco` that offsets
a spiky radio's LO. Outside the recorded band there is nothing to hear, and the span's
edges wrap round.

Must not import Qt or `rgc_sdr.dsp` (PLANNING.md section 5).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .source import MIN_RING_SAMPLES, DeviceCaps, FreqRange, IQSource, SequentialReader, _Nco, _Ring

#: Driver name a playback source reports, so the app can tell it from a radio.
PLAYBACK_DRIVER = "file"
#: Samples written per pacing step: 20 ms, small enough that the display never waits.
BLOCK_SECONDS = 0.02


def sidecar_for(path: Path) -> Path:
    """`x.cf32` -> `x.cf32.json`, as the IQ recorder writes it."""
    return path.with_suffix(path.suffix + ".json")


def read_sidecar(path: Path) -> dict:
    """The recording's description. Raises ValueError for anything not playable."""
    try:
        meta = json.loads(sidecar_for(path).read_text())
    except FileNotFoundError:
        raise ValueError(f"no sidecar {sidecar_for(path).name} beside the recording") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"sidecar is not valid JSON: {exc}") from None
    if meta.get("format") != "complex64" or meta.get("byte_order", "little") != "little":
        raise ValueError(f"unsupported format {meta.get('format')!r}; expected complex64")
    for key in ("sample_rate_hz", "center_freq_hz"):
        if not isinstance(meta.get(key), (int, float)) or meta[key] <= 0:
            raise ValueError(f"sidecar has no usable {key}")
    return meta


class FileIQSource(IQSource):
    """Replays a `.cf32` recording in real time, looping at the end."""

    def __init__(self, path: str | Path, loop: bool = True) -> None:
        self.path = Path(path)
        meta = read_sidecar(self.path)
        self.meta = meta
        self._samples = np.memmap(self.path, dtype=np.complex64, mode="r")
        if self._samples.size == 0:
            raise ValueError("the recording is empty")
        self._rate = float(meta["sample_rate_hz"])
        self.recorded_center = float(meta["center_freq_hz"])
        half = self._rate / 2.0
        self._caps = DeviceCaps(
            driver=PLAYBACK_DRIVER,
            label=f"Recording {self.path.name}",
            serial="",
            sample_rates=(self._rate,),
            freq_ranges=(FreqRange(self.recorded_center - half, self.recorded_center + half),),
            gain_elements=(),
            has_agc=False,
            formats=("CF32",),
        )
        self.loop = bool(loop)
        self._freq = self.recorded_center
        self._nco = _Nco(0.0, self._rate)
        self._ring = _Ring(max(int(self._rate), MIN_RING_SAMPLES))
        self._block = max(1, int(self._rate * BLOCK_SECONDS))
        self._lock = threading.Lock()
        self._position = 0               # next sample of the file to play
        self._paused = threading.Event()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._played = 0
        self._loops = 0
        self.finished = False            # reached the end with looping off

    # -- IQSource ----------------------------------------------------------

    @property
    def caps(self) -> DeviceCaps:
        return self._caps

    @property
    def sample_rate(self) -> float:
        return self._rate

    @property
    def center_freq(self) -> float:
        return self._freq

    @property
    def dc_spike_offset_hz(self) -> float:
        # The recording's own centre is where the radio's DC lay, if it had any.
        return self.recorded_center - self._freq

    def set_center_freq(self, hz: float, flush: bool = True) -> float:
        """Move the virtual centre: the recording is shifted so `hz` lands at 0 Hz."""
        wanted = self._caps.clamp_freq(float(hz))
        with self._lock:
            self._freq = wanted
            self._nco = _Nco(self.recorded_center - wanted, self._rate)
        if flush:
            self._ring.clear()
        return self._freq

    def set_sample_rate(self, hz: float) -> float:
        return self._rate               # the recording has the rate it has

    def set_bandwidth(self, hz: float) -> float:
        return 0.0

    def set_agc(self, enabled: bool) -> None:
        pass

    def set_gain(self, name: str, db: float) -> None:
        pass

    def get_gain(self, name: str) -> float:
        return 0.0

    def read_latest(self, n: int) -> np.ndarray:
        return self._ring.read_latest(n)

    def sequential_reader(self) -> SequentialReader:
        return SequentialReader(self._ring)

    @property
    def stats(self) -> dict[str, int]:
        return {"samples": self._played, "dropped": 0, "overflows": 0, "timeouts": 0,
                "errors": 0, "loops": self._loops}

    # -- transport ---------------------------------------------------------

    @property
    def duration_s(self) -> float:
        return self._samples.size / self._rate

    @property
    def position_s(self) -> float:
        return self._position / self._rate

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def seek(self, seconds: float) -> None:
        """Jump to `seconds` into the recording; what was buffered is dropped."""
        with self._lock:
            self._position = int(min(max(seconds, 0.0), self.duration_s) * self._rate)
            self._position = min(self._position, self._samples.size - 1)
            self.finished = False
        self._ring.clear()

    def next_block(self) -> np.ndarray | None:
        """The next block of the recording, shifted to the current tuning, advancing the
        play position. None at the end when not looping. The pacing thread's step, and
        directly callable so tests need no thread or clock."""
        with self._lock:
            if self._position >= self._samples.size:
                if not self.loop:
                    self.finished = True
                    return None
                self._position = 0
                self._loops += 1
            end = min(self._position + self._block, self._samples.size)
            block = np.array(self._samples[self._position:end])   # a copy, out of the map
            self._position = end
            self._nco.process(block)
        self._ring.write(block)
        self._played += block.size
        return block

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._pace, name="iq-playback", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _pace(self) -> None:
        """Write blocks at the recorded rate, catching up after any stall."""
        origin = time.monotonic()
        written = 0
        while self._running.is_set():
            if self._paused.is_set() or self.finished:
                time.sleep(0.02)
                origin, written = time.monotonic(), 0     # resume without a burst
                continue
            due = (time.monotonic() - origin) * self._rate
            if written + self._block > due:
                time.sleep(BLOCK_SECONDS / 4)
                continue
            if self.next_block() is None:
                continue
            written += self._block
