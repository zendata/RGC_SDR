"""The Icom IC-705 over its USB cable: a transceiver the app views and controls over CI-V.

Not an SDR: the radio demodulates itself and sends no IQ. What it sends is its spectrum
scope, 475 points a line at 4.3 lines/s (measured; section 7o), which the window draws
as the spectrum and waterfall. So this presents itself as an `IQSource` whose "sample
rate" is the scope's span and whose centre is the scope's centre -- enough for the
window's geometry, tuning and memories to work unchanged -- and `take_scope_line`
replaces `read_latest`.

Scope output is switched on while the source runs and put back as found on close.

The radio is found by its USB ID (Icom 0x0C26, IC-705 0x0036). Of its two serial ports
the first answers CI-V; the second (USB B, GPS/decode) is silent to it.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import civ
from .source import DeviceCaps, FreqRange, IQSource

ICOM_VID = 0x0C26
IC705_PID = 0x0036

#: The IC-705's receive coverage.
IC705_RANGES = (FreqRange(30e3, 199.999999e6), FreqRange(400e6, 470e6))

#: Measured: the 705 sends this many scope lines a second over USB, whatever the settings.
SCOPE_LINES_PER_S = 4.3

#: Seconds to wait for the radio to answer a query.
REPLY_TIMEOUT_S = 0.5


def find_ic705_ports() -> list[str]:
    """The IC-705's serial ports, CI-V first. Empty if none is attached."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    ports = [p.device for p in list_ports.comports()
             if p.vid == ICOM_VID and p.pid == IC705_PID]
    return sorted(ports)


class IcomSource(IQSource):
    """An IC-705 on USB, seen through its scope. Receive-side view only, for now."""

    kind = "transceiver"
    profile = None

    def __init__(self, port: str | None = None, address: int = civ.IC705_ADDRESS,
                 center_freq: float | None = None, **_ignored) -> None:
        import serial

        ports = [port] if port else find_ic705_ports()
        if not ports:
            raise RuntimeError("no IC-705 found on USB")
        self.port = ports[0]
        self.address = address
        self._serial = serial.Serial(self.port, 115200, timeout=0.05)
        self._parser = civ.FrameParser()
        self._assembler = civ.ScopeAssembler()
        self._lock = threading.Lock()
        self._replies: list[civ.Frame] = []
        self._reply_event = threading.Event()
        self._latest: civ.ScopeLine | None = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._lines = 0
        self._skipped = 0
        self._errors = 0
        self._restore: dict[int, bytes] = {}

        from .profiles import profile_for

        self.profile = profile_for("icom705")
        # Opened where the radio's own dial is: a transceiver is not retuned just
        # because the app last looked somewhere else. `center_freq` is ignored.
        self._freq = float(self._query_freq() or 145e6)
        self._span = 20e3
        self._caps = DeviceCaps(
            driver="icom705", label="Icom IC-705", serial="", sample_rates=(),
            freq_ranges=IC705_RANGES, gain_elements=(), has_agc=False, formats=(),
        )

    # -- CI-V plumbing --------------------------------------------------------------

    def _send(self, cmd: int, payload: bytes | tuple = b"") -> None:
        self._serial.write(civ.encode(cmd, payload, to=self.address))

    def _handle(self, frame: civ.Frame) -> None:
        if frame.frm != self.address:
            return
        if frame.cmd == civ.CMD_SCOPE and frame.sub == civ.SCOPE_WAVE:
            line = self._assembler.feed(frame)
            if line is not None:
                with self._lock:
                    if self._latest is not None:
                        self._skipped += 1
                    self._latest = line
                    self._lines += 1
                    self._span = line.span_hz
                    if line.centre_mode:
                        self._freq = line.centre_hz
            return
        if frame.cmd in (civ.CMD_TRANSCEIVE_FREQ, civ.CMD_READ_FREQ) and len(frame.payload) == 5:
            self._freq = float(civ.decode_freq(frame.payload))   # the radio's own dial
        with self._lock:
            self._replies.append(frame)
            # Unasked OKs (after every tune) and reports would otherwise pile up.
            del self._replies[:-32]
        self._reply_event.set()

    def _pump_once(self) -> None:
        try:
            data = self._serial.read(4096)
        except Exception:
            self._errors += 1
            time.sleep(0.05)
            return
        for frame in self._parser.feed(data):
            self._handle(frame)

    def _reader(self) -> None:
        while self._running.is_set():
            self._pump_once()

    def _ask(self, cmd: int, payload: bytes | tuple = b"") -> civ.Frame | None:
        """Send a query and wait for the answer with the same command."""
        with self._lock:
            self._replies.clear()
        self._reply_event.clear()
        self._send(cmd, payload)
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._thread is None:
                self._pump_once()           # no reader yet: read inline
            else:
                self._reply_event.wait(0.05)
            with self._lock:
                for frame in self._replies:
                    if frame.cmd == cmd or frame.is_ng:
                        return frame
        return None

    def _query_freq(self) -> int | None:
        frame = self._ask(civ.CMD_READ_FREQ)
        if frame is None or len(frame.payload) != 5:
            return None
        return civ.decode_freq(frame.payload)

    def _read_setting(self, *sub: int) -> bytes | None:
        frame = self._ask(civ.CMD_SCOPE, bytes(sub))
        if frame is None or frame.is_ng:
            return None
        return frame.payload[len(sub):]

    # -- IQSource ---------------------------------------------------------------------

    @property
    def caps(self) -> DeviceCaps:
        return self._caps

    @property
    def sample_rate(self) -> float:
        """The scope's full width: what the display spans."""
        return self._span

    @property
    def center_freq(self) -> float:
        return self._freq

    @property
    def stats(self) -> dict:
        return {"lines": self._lines, "skipped": self._skipped,
                "dropped": self._assembler.dropped, "errors": self._errors,
                "overflows": 0, "timeouts": 0}

    def start(self) -> None:
        if self._thread is not None:
            return
        # Remember the scope's on/off and output settings, to put back on close.
        for sub in (civ.SCOPE_ON, civ.SCOPE_OUTPUT):
            was = self._read_setting(sub)
            if was:
                self._restore[sub] = was
        self._send(civ.CMD_SCOPE, bytes([civ.SCOPE_ON, 0x01]))
        self._send(civ.CMD_SCOPE, bytes([civ.SCOPE_OUTPUT, 0x01]))
        self._running.set()
        self._thread = threading.Thread(target=self._reader, name="ic705-civ", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._running.clear()
        self._thread.join(timeout=1.0)
        self._thread = None
        # Output first: stop the stream, then the scope as it was.
        for sub in (civ.SCOPE_OUTPUT, civ.SCOPE_ON):
            value = self._restore.get(sub, b"\x00" if sub == civ.SCOPE_OUTPUT else None)
            if value is not None:
                self._send(civ.CMD_SCOPE, bytes([sub]) + value[:1])
        self._restore.clear()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._serial.close()

    def read_latest(self, n: int) -> np.ndarray:
        return np.zeros(0, dtype=np.complex64)      # no IQ from a transceiver

    def take_scope_line(self) -> civ.ScopeLine | None:
        """The newest whole scope line since the last call, or None."""
        with self._lock:
            line, self._latest = self._latest, None
        return line

    def set_center_freq(self, hz: float, flush: bool = True) -> float:
        target = self._caps.clamp_freq(float(hz))
        self._send(civ.CMD_SET_FREQ, civ.encode_freq(target))
        self._freq = target
        return self._freq

    def set_sample_rate(self, hz: float) -> float:
        return self._span                           # the span is the radio's to set

    def sequential_reader(self):
        raise NotImplementedError("the IC-705 sends no IQ; its audio comes over USB audio")
