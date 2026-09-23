"""Recording: demodulated audio to WAV, and raw IQ to a flat complex64 file.

Both write on their own thread. Neither the audio worker nor the Qt event loop may ever
block on a disk write -- a stalled write in the audio path is a dropout, and in the UI
thread it is a freeze. Each recorder therefore owns a bounded queue and drops (counting
what it dropped) rather than applying back-pressure to the signal path.

IQ recording is heavy: complex64 at 768 kS/s is about 6.1 MB/s, or 368 MB a minute, so a
byte limit is enforced and the recorder stops itself rather than filling the disk.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: Where recordings go unless told otherwise.
DEFAULT_DIR = Path.home() / "Documents" / "RGC_SDR"

#: Safety stop for IQ captures. 2 GiB is about 5.5 minutes at 768 kS/s.
DEFAULT_MAX_BYTES = 2 * 1024**3

IQ_DTYPE = np.complex64


def timestamp_name(center_hz: float, suffix: str, mode: str | None = None) -> str:
    """A sortable, self-describing filename."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    parts = [stamp, f"{center_hz / 1e6:.4f}MHz"]
    if mode:
        parts.append(mode)
    return "_".join(parts) + suffix


class _ThreadedWriter:
    """Common lifecycle: a bounded queue drained by one background thread."""

    def __init__(self, path: Path, queue_blocks: int = 64) -> None:
        self.path = Path(path)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_blocks)
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self.bytes_written = 0
        #: Counted at submit time rather than at write time, so a size limit cannot be
        #: overshot by whatever is still sitting in the queue.
        self.submitted_bytes = 0
        self.dropped_blocks = 0
        self.error: str | None = None
        self.started_at: float | None = None
        self._stopped_reason: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._running.is_set()

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    @property
    def stopped_reason(self) -> str | None:
        return self._stopped_reason

    def start(self) -> None:
        if self._thread is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open()
        self.started_at = time.monotonic()
        self._running.set()
        self._thread = threading.Thread(target=self._drain, name="recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._running.clear()
        self._thread.join(timeout=3.0)
        self._thread = None
        self._close()

    def submit(self, block: np.ndarray) -> None:
        """Hand a block over without ever blocking the caller."""
        if not self._running.is_set() or block.size == 0:
            return
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            self.dropped_blocks += 1
            return
        self.submitted_bytes += block.nbytes

    def _drain(self) -> None:
        while self._running.is_set() or not self._queue.empty():
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.bytes_written += self._write(block)
            except OSError as exc:
                self.error = str(exc)
                self._stopped_reason = f"write failed: {exc}"
                self._running.clear()
                return

    # Subclass hooks.
    def _open(self) -> None: ...
    def _close(self) -> None: ...
    def _write(self, block: np.ndarray) -> int:
        raise NotImplementedError


class AudioRecorder(_ThreadedWriter):
    """Demodulated audio as a 16-bit mono WAV.

    16-bit PCM rather than float32: the stdlib `wave` module cannot write
    WAVE_FORMAT_IEEE_FLOAT, and 16 bits is plenty for AGC-levelled audio that plays
    anywhere without conversion.
    """

    def __init__(self, path: Path, sample_rate: float, queue_blocks: int = 64) -> None:
        super().__init__(path, queue_blocks)
        self.sample_rate = int(round(sample_rate))
        self._wav: wave.Wave_write | None = None

    @property
    def seconds_recorded(self) -> float:
        return self.bytes_written / 2.0 / max(self.sample_rate, 1)

    def _open(self) -> None:
        self._wav = wave.open(str(self.path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.sample_rate)

    def _close(self) -> None:
        if self._wav is not None:
            try:
                self._wav.close()
            finally:
                self._wav = None

    def _write(self, block: np.ndarray) -> int:
        samples = np.clip(block, -1.0, 1.0)
        pcm = (samples * 32767.0).astype("<i2")
        self._wav.writeframes(pcm.tobytes())
        return pcm.nbytes


class IQRecorder(_ThreadedWriter):
    """Raw complex64 IQ, with a JSON sidecar describing how to read it.

    Pulls from its own gapless reader rather than being fed, so a recording is continuous
    regardless of what the display or audio path happens to be doing. The sidecar is
    written at start as well as at stop, so an interrupted capture is still readable.
    """

    def __init__(
        self,
        path: Path,
        source,
        block_samples: int = 65536,
        max_bytes: int = DEFAULT_MAX_BYTES,
        queue_blocks: int = 32,
    ) -> None:
        super().__init__(path, queue_blocks)
        self.source = source
        self.block_samples = int(block_samples)
        self.max_bytes = int(max_bytes)
        self.sample_rate = float(source.sample_rate)
        self.center_freq = float(source.center_freq)
        self.lost_samples = 0
        self._file = None
        self._reader = None
        self._feeder: threading.Thread | None = None

    @property
    def sidecar_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".json")

    @property
    def samples_written(self) -> int:
        return self.bytes_written // np.dtype(IQ_DTYPE).itemsize

    @property
    def seconds_recorded(self) -> float:
        return self.samples_written / max(self.sample_rate, 1.0)

    def _metadata(self) -> dict:
        return {
            "format": "complex64",
            "byte_order": "little",
            "sample_rate_hz": self.sample_rate,
            "center_freq_hz": self.center_freq,
            "driver": getattr(self.source.caps, "driver", "unknown"),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "samples": self.samples_written,
            "seconds": round(self.seconds_recorded, 3),
            "lost_samples": self.lost_samples,
            "note": "Interleaved float32 I/Q pairs, no header.",
        }

    def _write_sidecar(self) -> None:
        try:
            self.sidecar_path.write_text(json.dumps(self._metadata(), indent=2))
        except OSError as exc:
            self.error = str(exc)

    def _open(self) -> None:
        self._file = open(self.path, "wb", buffering=1024 * 1024)
        self._reader = self.source.sequential_reader()
        self._write_sidecar()

    def _close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
        self._write_sidecar()

    def _write(self, block: np.ndarray) -> int:
        data = np.ascontiguousarray(block, dtype=IQ_DTYPE)
        self._file.write(data.tobytes())
        return data.nbytes

    def start(self) -> None:
        super().start()
        self._feeder = threading.Thread(target=self._feed, name="iq-record-feed", daemon=True)
        self._feeder.start()

    def stop(self) -> None:
        self._running.clear()
        if self._feeder is not None:
            self._feeder.join(timeout=3.0)
            self._feeder = None
        super().stop()

    def _feed(self) -> None:
        while self._running.is_set():
            if self.submitted_bytes >= self.max_bytes:
                self._stopped_reason = f"reached the {self.max_bytes / 1024**3:.1f} GiB limit"
                self._running.clear()
                break
            if self._reader.available() < self.block_samples:
                time.sleep(0.01)
                continue
            block = self._reader.read(self.block_samples)
            if block.size:
                self.submit(block)
            self.lost_samples = self._reader.lost
