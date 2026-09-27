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
from dataclasses import dataclass

import numpy as np

from . import civ
from .source import DeviceCaps, FreqRange, IQSource, TxCaps

ICOM_VID = 0x0C26
IC705_PID = 0x0036

#: The IC-705's receive coverage.
IC705_RANGES = (FreqRange(30e3, 199.999999e6), FreqRange(400e6, 470e6))

#: Measured: the 705 sends this many scope lines a second over USB, whatever the settings.
SCOPE_LINES_PER_S = 4.3

#: Seconds to wait for the radio to answer a query.
REPLY_TIMEOUT_S = 0.5

#: How often to ask for the S-meter, and for every other setting. CI-V Transceive
#: reports frequency and mode changes on its own, but not level or switch changes made
#: on the radio's front panel, so those are polled.
METER_POLL_S = 0.25
SETTINGS_POLL_S = 1.5


@dataclass(frozen=True)
class RadioControl:
    """One of the radio's own settings, as the window should offer it."""

    key: str
    label: str
    #: "level" (0-255, shown as %), "choice" (one of `choices`) or "switch" (on/off).
    kind: str
    choices: tuple[tuple[str, int], ...] = ()
    tooltip: str = ""


#: The IC-705 settings offered in the window, each confirmed on the radio (7o).
IC705_CONTROLS: tuple[RadioControl, ...] = (
    RadioControl("af", "AF", "level", tooltip="The radio's own speaker volume"),
    RadioControl("rf", "RF", "level", tooltip="RF gain"),
    RadioControl("sql", "SQL", "level", tooltip="The radio's squelch"),
    RadioControl("preamp", "Pre", "choice", (("Off", 0), ("P.AMP1", 1), ("P.AMP2", 2)),
                 tooltip="Preamplifier"),
    RadioControl("att", "ATT", "switch", tooltip="20 dB attenuator"),
    RadioControl("agc", "AGC", "choice", (("Fast", 1), ("Mid", 2), ("Slow", 3)),
                 tooltip="The 705 fixes AGC in FM, and refuses a change there"),
    RadioControl("nb", "NB", "switch", tooltip="Noise blanker"),
    RadioControl("nr", "NR", "switch", tooltip="Noise reduction"),
    RadioControl("power", "Power", "level", tooltip="TX power, 0.1-10 W"),
)

# key -> (command, sub-command or None, form): "level" is 2-byte BCD, "byte" one byte,
# "att" is 00 (off) or 20 (the 20 dB attenuator).
_CONTROL_CI_V = {
    "af": (0x14, 0x01, "level"), "rf": (0x14, 0x02, "level"), "sql": (0x14, 0x03, "level"),
    "power": (0x14, 0x0A, "level"), "preamp": (0x16, 0x02, "byte"),
    "agc": (0x16, 0x12, "byte"), "nb": (0x16, 0x22, "byte"), "nr": (0x16, 0x40, "byte"),
    "att": (0x11, None, "att"),
}
_BY_COMMAND = {(cmd, sub): key for key, (cmd, sub, _form) in _CONTROL_CI_V.items()}


def _piecewise(raw: int, points: tuple[tuple[int, float], ...]) -> float:
    """Linear between the calibration points Icom lists for a meter."""
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if raw <= x1:
            return y0 + (y1 - y0) * (max(raw, x0) - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return y1 + (y1 - y0) * (raw - x1) / (x1 - x0)


#: Icom's documented meter calibration (IC-7300/705 CI-V guides): raw -> % power, SWR.
#: Not yet checked against the 705 itself -- verify at the first transmission.
PO_POINTS = ((0, 0.0), (143, 50.0), (213, 100.0))
SWR_POINTS = ((0, 1.0), (48, 1.5), (80, 2.0), (120, 3.0))


def power_percent(raw: int) -> float:
    return min(_piecewise(raw, PO_POINTS), 120.0)


def swr_value(raw: int) -> float:
    return _piecewise(raw, SWR_POINTS)


def s_meter_text(raw: int) -> str:
    """Icom's 0-255 meter scale: 0 = S0, 120 = S9, 241 = S9+60 dB."""
    if raw <= 120:
        return f"S{round(raw / 120 * 9)}"
    return f"S9+{round((raw - 120) / 121 * 60)} dB"


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
                 center_freq: float | None = None, transport=None, **_ignored) -> None:
        if transport is None:
            import serial

            ports = [port] if port else find_ic705_ports()
            if not ports:
                raise RuntimeError("no IC-705 found on USB")
            self.port = ports[0]
            transport = serial.Serial(self.port, 115200, timeout=0.05)
        else:
            self.port = port or "injected"
        self.address = address
        #: Anything with read(n), write(bytes) and close(): a serial port, or a test's
        #: stand-in. Written from both the GUI and reader threads, hence the lock.
        self._serial = transport
        self._write_lock = threading.Lock()
        #: The radio's settings as last reported, by control key; plus mode and meter.
        self.state: dict[str, int] = {}
        self.mode: str | None = None
        self.filter: int | None = None
        self.smeter: int | None = None
        #: Writes awaiting FB/FA, oldest first, and the latest one the radio refused.
        self._pending: list[str] = []
        self.refused: str | None = None
        self._next_meter = 0.0
        self._next_settings = 0.0
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
        # The radio modulates and keeps to its own licensed bands and power; the app
        # only keys it and supplies audio. Half duplex, as any transceiver.
        self._caps = DeviceCaps(
            driver="icom705", label="Icom IC-705", serial="", sample_rates=(),
            freq_ranges=IC705_RANGES, gain_elements=(), has_agc=False, formats=(),
            tx=TxCaps(freq_ranges=IC705_RANGES, gain_elements=(), sample_rates=(),
                      full_duplex=False),
        )
        #: Keyed by the app, and the transmit meters while keyed (raw 0-255).
        self.transmitting = False
        self.po: int | None = None
        self.swr: int | None = None

    # -- CI-V plumbing --------------------------------------------------------------

    def _send(self, cmd: int, payload: bytes | tuple = b"", label: str | None = None) -> None:
        """Send; `label` marks a write, whose FB/FA answer is matched to it in order."""
        with self._write_lock:
            if label is not None:
                self._pending.append(label)
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
        self._update_state(frame)
        with self._lock:
            self._replies.append(frame)
            # Unasked OKs (after every tune) and reports would otherwise pile up.
            del self._replies[:-32]
        self._reply_event.set()

    def _update_state(self, frame: civ.Frame) -> None:
        cmd, payload = frame.cmd, frame.payload
        if frame.is_ok or frame.is_ng:
            with self._write_lock:
                label = self._pending.pop(0) if self._pending else None
            if frame.is_ng and label is not None:
                self.refused = label
                if label in _CONTROL_CI_V:
                    self._read_control(label)      # show what the radio really has
            return
        if cmd in (civ.CMD_READ_MODE, civ.CMD_TRANSCEIVE_MODE) and payload:
            self.mode = civ.MODES.get(payload[0], self.mode)
            if len(payload) > 1:
                self.filter = payload[1]
            return
        if cmd == 0x15 and len(payload) == 3:
            value = civ.decode_level(payload[1:3])
            if payload[0] == 0x02:
                self.smeter = value
            elif payload[0] == 0x11:
                self.po = value
            elif payload[0] == 0x12:
                self.swr = value
            return
        if cmd == 0x11 and len(payload) == 1:
            self.state["att"] = 1 if payload[0] else 0
            return
        if payload:
            key = _BY_COMMAND.get((cmd, payload[0]))
            if key is None:
                return
            _cmd, _sub, form = _CONTROL_CI_V[key]
            value = payload[1:]
            if form == "level" and len(value) == 2:
                self.state[key] = civ.decode_level(value)
            elif form == "byte" and len(value) == 1:
                self.state[key] = value[0]

    def _read_control(self, key: str) -> None:
        cmd, sub, _form = _CONTROL_CI_V[key]
        self._send(cmd, b"" if sub is None else bytes([sub]))

    def poll(self, now: float | None = None) -> None:
        """Ask for the meter, and now and then everything else. Called by the reader."""
        now = time.monotonic() if now is None else now
        if now >= self._next_meter:
            self._next_meter = now + METER_POLL_S
            if self.transmitting:
                self._send(0x15, b"\x11")        # power out
                self._send(0x15, b"\x12")        # SWR
            else:
                self._send(0x15, b"\x02")        # S-meter
        if now >= self._next_settings:
            self._next_settings = now + SETTINGS_POLL_S
            self._send(civ.CMD_READ_MODE)
            for key in _CONTROL_CI_V:
                self._read_control(key)

    # -- control ------------------------------------------------------------------------

    @property
    def controls(self) -> tuple[RadioControl, ...]:
        return IC705_CONTROLS

    def set_control(self, key: str, value: int) -> None:
        """Change one of the radio's settings. A refusal (FA) sets `refused` and the
        radio's real value is read back into `state`."""
        cmd, sub, form = _CONTROL_CI_V[key]
        if form == "level":
            data = civ.encode_level(int(value))
        elif form == "att":
            data = bytes([0x20 if value else 0x00])
        else:
            data = bytes([int(value)])
        self.state[key] = int(value)            # optimistic; a refusal corrects it
        self._send(cmd, (b"" if sub is None else bytes([sub])) + data, label=key)

    def set_ptt(self, on: bool) -> None:
        """Key or unkey the transmitter (CI-V 1C 00)."""
        self.transmitting = bool(on)
        if not on:
            self.po = self.swr = None
        self._send(0x1C, bytes([0x00, 0x01 if on else 0x00]), label="ptt")

    def set_mode(self, mode: str, filter_number: int | None = None) -> None:
        """Mode by name ("usb", "fm", ...) and filter FIL1-3 (kept if not given)."""
        code = civ.MODE_CODES[mode]
        filt = int(filter_number or self.filter or 1)
        self.mode, self.filter = mode, filt
        self._send(civ.CMD_SET_MODE, bytes([code, filt]), label="mode")

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
            self.poll()
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
        self._send(civ.CMD_SCOPE, bytes([civ.SCOPE_ON, 0x01]), label="scope")
        self._send(civ.CMD_SCOPE, bytes([civ.SCOPE_OUTPUT, 0x01]), label="scope")
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
                self._send(civ.CMD_SCOPE, bytes([sub]) + value[:1], label="scope")
        self._restore.clear()

    def close(self) -> None:
        try:
            if self.transmitting:
                self.set_ptt(False)             # never leave the radio keyed
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
        self._send(civ.CMD_SET_FREQ, civ.encode_freq(target), label="frequency")
        self._freq = target
        return self._freq

    def set_sample_rate(self, hz: float) -> float:
        return self._span                           # the span is the radio's to set

    def sequential_reader(self):
        raise NotImplementedError("the IC-705 sends no IQ; its audio comes over USB audio")
