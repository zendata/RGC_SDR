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

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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
#: Settings the app changes while it has the radio, and puts back afterwards (menu items
#: from Icom's IC-705 CI-V reference guide, `1A 05`; checked on the radio 2026-09-28):
#: the speaker is silenced (AF 0 -- the USB audio does not follow AF: -31.7 dBFS at AF 0)
#: and TX audio is taken from USB only. DATA OFF MOD was MIC,USB, so the radio's own
#: microphone was on the air alongside the Mac's.
#: USB AF Output Level goes to 100% (measured +11.6 dB over the default 50%, so the app
#: needs almost no gain of its own) and USB AF SQL on: the radio then mutes its USB audio
#: itself when the squelch closes (measured -34 -> -87 dBFS), instantly.
#: DATA MOD is the TX audio source in the data modes (USB-D, FM-D ...); VK3RQ's was MIC,
#: so there the radio's microphone would have gone out instead of the Mac's.
DATA_OFF_MOD = (0x01, 0x18)          # 00 MIC, 01 USB, 02 MIC+USB, 03 WLAN
#: `1A 05` menu items the app takes over: name -> item number.
MENU_ITEMS = {
    "data_off_mod": DATA_OFF_MOD,
    "data_mod": (0x01, 0x19),        # 00 MIC, 01 USB, 02 MIC+USB, 03 WLAN
    "usb_af_level": (0x01, 0x10),    # 0000-0255 = 0-100 %
    "usb_af_sql": (0x01, 0x11),      # 00 off (always open), 01 follows the squelch
}
TAKEOVER = {"af": b"\x00\x00", "data_off_mod": b"\x01", "data_mod": b"\x01",
            "usb_af_level": b"\x02\x55", "usb_af_sql": b"\x01"}

#: SET-menu items the app has taken over: item number -> TAKEOVER name. The menu window
#: shows the radio's own value for these, not the app's.
MENU_TAKEN = {110: "usb_af_level", 111: "usb_af_sql", 118: "data_off_mod",
              119: "data_mod"}
#: SET-menu items the menu window will not change, and why.
MENU_LOCKED = {
    110: "Set by the app while it has the radio (put back when it lets go)",
    111: "Set by the app while it has the radio (put back when it lets go)",
    118: "Set by the app while it has the radio (put back when it lets go)",
    119: "Set by the app while it has the radio (put back when it lets go)",
    131: "The app needs CI-V Transceive on to follow the radio's dial",
    132: "Echo Back on would confuse the app's reading of the radio's replies",
}

#: Where the radio's own settings are kept while the app has changed them, so that a
#: crash cannot leave the radio with its speaker and microphone switched off.
RESTORE_FILE = Path.home() / "Library" / "Application Support" / "RGC_SDR" / "ic705_restore.json"

#: The 705 does not squelch its USB audio (VK3RQ, 2026-09-28: noise heard with SQL up),
#: so the app gates it on the radio's squelch state, asked for this often.
SQUELCH_POLL_S = 0.1


@dataclass(frozen=True)
class RadioControl:
    """One of the radio's own settings, as the window should offer it."""

    key: str
    label: str
    #: "level" (0-255, shown as %), "choice" (one of `choices`) or "switch" (on/off).
    kind: str
    choices: tuple[tuple[str, int], ...] = ()
    tooltip: str = ""
    #: Where the window shows it: "row" (the Radio row), "function" (a button on the
    #: function panel, like the 705's FUNCTION screen) or "popup" (a level opened from
    #: a function button, as a long touch does on the radio).
    placement: str = "row"
    #: For a function button: the levels its long press opens (the radio's function menu).
    levels: tuple[str, ...] = ()
    #: For a level: what 0-255 means on the radio, (at 0, at 255, unit); None shows %.
    scale: tuple[float, float, str] | None = None


#: The IC-705 settings offered in the window. Every command read on the radio first
#: (2026-09-28); from Icom's IC-705 CI-V reference guide, command table pp. 3-4.
IC705_CONTROLS: tuple[RadioControl, ...] = (
    # Radio row: the levels in constant use.
    RadioControl("af", "AF", "level", tooltip="The radio's own speaker volume (the app "
                 "sets it to 0 while it has the radio)"),
    RadioControl("rf", "RF", "level", tooltip="RF gain"),
    RadioControl("sql", "SQL", "level", tooltip="The radio's squelch"),
    RadioControl("power", "Power", "level", tooltip="TX power, 0.1-10 W"),
    # Function panel: the FUNCTION screen's keys.
    # The FUNCTION screen's keys (IC-705 Basic Manual p. 2-6), long press -> its menu.
    RadioControl("preamp", "P.AMP", "choice", (("OFF", 0), ("P.AMP1", 1), ("P.AMP2", 2)),
                 tooltip="Preamplifier (144/430 MHz: on/off only)", placement="function"),
    RadioControl("att", "ATT", "switch", tooltip="20 dB attenuator (HF and 50 MHz)",
                 placement="function"),
    RadioControl("agc", "AGC", "choice", (("FAST", 1), ("MID", 2), ("SLOW", 3)),
                 tooltip="AGC time constant; fixed in FM, WFM and DV", placement="function"),
    RadioControl("notch", "NOTCH", "choice", (("OFF", 0), ("AN", 1), ("MN", 2)),
                 tooltip="Auto notch (SSB/AM/FM) or manual notch (SSB/CW/RTTY/AM)",
                 placement="function", levels=("notch_pos",)),
    RadioControl("nb", "NB", "switch", tooltip="Noise blanker (SSB/CW/RTTY/AM)",
                 placement="function", levels=("nb_level",)),
    RadioControl("nr", "NR", "switch", tooltip="Noise reduction", placement="function",
                 levels=("nr_level",)),
    RadioControl("split", "SPLIT", "switch", tooltip="Split: transmit on the other VFO",
                 placement="function"),
    RadioControl("vox", "VOX", "switch", tooltip="VOX", placement="function",
                 levels=("vox_gain", "anti_vox")),
    RadioControl("comp", "COMP", "switch", tooltip="Speech compressor (SSB)",
                 placement="function", levels=("comp_level",)),
    RadioControl("moni", "MONI", "switch", tooltip="Transmit monitor", placement="function",
                 levels=("moni_level",)),
    RadioControl("bkin", "BKIN", "choice", (("OFF", 0), ("BKIN", 1), ("F-BKIN", 2)),
                 tooltip="CW break-in: semi or full", placement="function",
                 levels=("bkin_delay",)),
    RadioControl("tone", "TONE", "choice",
                 (("OFF", 0), ("TONE", 1), ("TSQL", 2), ("DTCS", 3), ("DTCS(T)", 6),
                  ("TONE(T)/DTCS(R)", 7), ("DTCS(T)/TSQL(R)", 8), ("TONE(T)/TSQL(R)", 9)),
                 tooltip="Repeater tone, tone squelch, DTCS", placement="function"),
    RadioControl("dup", "DUP", "choice", (("OFF", 0x10), ("DUP\u2212", 0x11), ("DUP+", 0x12)),
                 tooltip="Repeater duplex", placement="function"),
    RadioControl("rit", "RIT", "switch", tooltip="Receive incremental tuning",
                 placement="function"),
    RadioControl("dtx", "\u0394TX", "switch", tooltip="Transmit offset", placement="function"),
    RadioControl("lock", "LOCK", "switch", tooltip="Dial lock", placement="function"),
    # The levels in those menus.
    RadioControl("nb_level", "NB level", "level", placement="popup"),
    RadioControl("nr_level", "NR level", "level", placement="popup", scale=(0, 15, "")),
    RadioControl("notch_pos", "Notch position", "level", placement="popup"),
    RadioControl("comp_level", "COMP level", "level", placement="popup", scale=(0, 10, "")),
    RadioControl("vox_gain", "VOX gain", "level", placement="popup"),
    RadioControl("anti_vox", "Anti-VOX", "level", placement="popup"),
    RadioControl("moni_level", "MONI level", "level", placement="popup"),
    RadioControl("bkin_delay", "BK-IN delay", "level", placement="popup",
                 scale=(2.0, 13.0, "d")),
    # The MULTI knob's menu (Basic Manual p. 2-7), and twin PBT.
    RadioControl("mic_gain", "MIC gain", "level", placement="popup"),
    RadioControl("key_speed", "Key speed", "level", placement="popup",
                 scale=(6, 48, " WPM")),
    RadioControl("cw_pitch", "CW pitch", "level", placement="popup",
                 scale=(300, 900, " Hz")),
    RadioControl("pbt1", "PBT1", "level", placement="popup", scale=(-100, 100, "")),
    RadioControl("pbt2", "PBT2", "level", placement="popup", scale=(-100, 100, "")),
)

# key -> (command, sub-command or None, form): "level" is 2-byte BCD, "byte" one byte,
# "att" is 00 (off) or 20 (the 20 dB attenuator); "split" and "dup" share command 0F;
# "notch" is made of the auto and manual notch switches.
_CONTROL_CI_V = {
    "af": (0x14, 0x01, "level"), "rf": (0x14, 0x02, "level"), "sql": (0x14, 0x03, "level"),
    "power": (0x14, 0x0A, "level"), "preamp": (0x16, 0x02, "byte"),
    "agc": (0x16, 0x12, "byte"), "nb": (0x16, 0x22, "byte"), "nr": (0x16, 0x40, "byte"),
    "att": (0x11, None, "att"),
    "anotch": (0x16, 0x41, "byte"), "mnotch": (0x16, 0x48, "byte"),
    "comp": (0x16, 0x44, "byte"), "moni": (0x16, 0x45, "byte"), "vox": (0x16, 0x46, "byte"),
    "bkin": (0x16, 0x47, "byte"), "lock": (0x16, 0x50, "byte"), "tone": (0x16, 0x5D, "byte"),
    "rit": (0x21, 0x01, "byte"), "dtx": (0x21, 0x02, "byte"),
    "split": (0x0F, None, "split"), "dup": (0x0F, None, "dup"),
    "notch": (None, None, "notch"),
    "nb_level": (0x14, 0x12, "level"), "nr_level": (0x14, 0x06, "level"),
    "notch_pos": (0x14, 0x0D, "level"), "comp_level": (0x14, 0x0E, "level"),
    "vox_gain": (0x14, 0x16, "level"), "anti_vox": (0x14, 0x17, "level"),
    "moni_level": (0x14, 0x15, "level"), "bkin_delay": (0x14, 0x0F, "level"),
    "mic_gain": (0x14, 0x0B, "level"), "key_speed": (0x14, 0x0C, "level"),
    "cw_pitch": (0x14, 0x09, "level"),
    # Twin PBT: 128 is centre. Refused (FA) in FM, WFM and DV, where it does not apply.
    "pbt1": (0x14, 0x07, "level"), "pbt2": (0x14, 0x08, "level"),
}

#: What the MULTI knob offers in each mode (Basic Manual p. 2-7), plus twin PBT where it
#: works (SSB, CW, RTTY, AM -- p. 4-4).
MULTI_BY_MODE = {
    "lsb": ("power", "mic_gain", "comp_level", "moni_level", "pbt1", "pbt2"),
    "usb": ("power", "mic_gain", "comp_level", "moni_level", "pbt1", "pbt2"),
    "cw": ("power", "key_speed", "cw_pitch", "moni_level", "pbt1", "pbt2"),
    "cw-r": ("power", "key_speed", "cw_pitch", "moni_level", "pbt1", "pbt2"),
    "rtty": ("power", "moni_level", "pbt1", "pbt2"),
    "rtty-r": ("power", "moni_level", "pbt1", "pbt2"),
    "am": ("power", "mic_gain", "moni_level", "pbt1", "pbt2"),
    "fm": ("power", "mic_gain", "moni_level"),
    "wfm": ("power",),
}
#: RIT / dTX offset limit, Hz (Basic Manual p. 11-2).
RIT_LIMIT_HZ = 9999
#: Keys read through others: DUP comes with SPLIT's 0F, NOTCH from the two notch switches.
_READ_VIA = {"dup": ("split",), "notch": ("anotch", "mnotch")}

#: Read-only display state, asked for with the settings: VFO A/B, RIT, duplex offset,
#: tones and step. (command, payload) pairs.
_STATUS_READS = (
    (0x25, b"\x00"), (0x25, b"\x01"), (0x26, b"\x00"), (0x26, b"\x01"),
    (0x21, b"\x00"), (0x0C, b""), (0x1B, b"\x00"), (0x1B, b"\x01"), (0x1B, b"\x02"),
    (0x15, b"\x15"),                                  # Vd: battery / supply volts
)


def _bcd_int(data: bytes) -> int:
    """Big-endian BCD digits as an integer: 00 08 85 -> 885."""
    value = 0
    for b in data:
        value = value * 100 + ((b >> 4) * 10 + (b & 0x0F))
    return value


VD_POINTS = ((0, 0.0), (75, 5.0), (241, 16.0))
ID_POINTS = ((0, 0.0), (121, 2.0), (241, 4.0))
COMP_POINTS = ((0, 0.0), (130, 15.0), (210, 25.5))


def supply_volts(raw: int) -> float:
    return _piecewise(raw, VD_POINTS)


def drain_amps(raw: int) -> float:
    return _piecewise(raw, ID_POINTS)


def comp_db(raw: int) -> float:
    return _piecewise(raw, COMP_POINTS)
_BY_COMMAND = {(cmd, sub): key for key, (cmd, sub, _form) in _CONTROL_CI_V.items()
               if key not in _READ_VIA}


def _piecewise(raw: int, points: tuple[tuple[int, float], ...]) -> float:
    """Linear between the calibration points Icom lists for a meter."""
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if raw <= x1:
            return y0 + (y1 - y0) * (max(raw, x0) - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return y1 + (y1 - y0) * (raw - x1) / (x1 - x0)


#: Icom's documented meter calibration (IC-7300/705 CI-V guides): raw -> % power, SWR.
#: Confirmed on VK3RQ's IC-705 on air (2026-09-28): power and SWR agree with the radio.
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
                 center_freq: float | None = None, transport=None,
                 restore_file: Path | None = None, **_ignored) -> None:
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
        #: Every command sent and not yet answered, oldest first: (command, label, is a
        #: write). Replies come back in order, so a data reply settles the entry for its
        #: command, FB the oldest write, FA the oldest entry -- a read can be refused too
        #: (twin PBT in WFM, measured). `refused` is the latest refused write.
        self._pending: list[tuple[int, str | None, bool]] = []
        #: The radio's display state: VFOs, RIT, offset, tones, meters. See _STATUS_READS.
        self.status: dict[str, object] = {}
        #: VFO / memory state, as the app last set it: the radio does not report it over
        #: CI-V (07 and 08 only set; checked 2026-09-29), so None until chosen from here.
        #: "A", "B", "MEMO" or "CALL".
        self.vfo_mode: str | None = None
        self.last_vfo = "A"
        self.memo_group = 0
        self.memo_channel = 0
        self.call_channel = 0
        #: The selected memory channel as read back (an ic705_memory.MemoryChannel), or
        #: "blank"; None while unknown.
        self.memo_contents = None
        self.refused: str | None = None
        self._next_meter = 0.0
        self._next_settings = 0.0
        self._next_squelch = 0.0
        #: The radio's squelch: True open, False closed, None not yet known.
        self.squelch_open: bool | None = None
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
        self.restore_file = Path(restore_file) if restore_file else RESTORE_FILE
        #: The radio's own values of the TAKEOVER settings, while the app has them.
        self._taken: dict[str, bytes] = {}
        #: The scope's own centre (differs from the dial in fixed mode) and half-span.
        self.display_center_freq: float | None = None
        self.half_span: float | None = None
        self.centre_mode = True

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
        """Send; `label` marks a write, so that a refusal can be reported against it."""
        with self._write_lock:
            self._pending.append((cmd, label, label is not None))
            del self._pending[:-256]                  # a lost reply must not grow it forever
            self._serial.write(civ.encode(cmd, payload, to=self.address))

    def _settle_pending(self, frame: civ.Frame) -> str | None:
        """Match a reply to what it answers. Returns the label of a refused write."""
        with self._write_lock:
            pending = self._pending
            if frame.is_ok:
                while pending:
                    if pending.pop(0)[2]:
                        break
                return None
            if frame.is_ng:
                if pending:
                    _cmd, label, is_write = pending.pop(0)
                    return label if is_write else None
                return None
            for i, (cmd, _label, _write) in enumerate(pending):
                if cmd == frame.cmd:
                    del pending[: i + 1]
                    break
            return None

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
                    self.display_center_freq = line.centre_hz
                    self.half_span = line.span_hz / 2.0
                    self.centre_mode = line.centre_mode
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
        if frame.to != 0x00:                       # a reply to us, not a transceive report
            refused = self._settle_pending(frame)
            if refused is not None:
                self.refused = refused
                if refused in _CONTROL_CI_V:
                    self._read_control(refused)    # show what the radio really has
        if frame.is_ok or frame.is_ng:
            return
        if self._update_status(cmd, payload):
            return
        if cmd in (civ.CMD_READ_MODE, civ.CMD_TRANSCEIVE_MODE) and payload:
            self.mode = civ.MODES.get(payload[0], self.mode)
            if len(payload) > 1:
                self.filter = payload[1]
            return
        if cmd == 0x15 and len(payload) == 2 and payload[0] == 0x01:
            self.squelch_open = bool(payload[1])
            return
        if cmd == 0x15 and len(payload) == 3:
            value = civ.decode_level(payload[1:3])
            if payload[0] == 0x02:
                self.smeter = value
            elif payload[0] == 0x11:
                self.po = value
            elif payload[0] == 0x12:
                self.swr = value
            else:
                name = {0x13: "alc", 0x14: "comp_meter", 0x15: "vd", 0x16: "id"}.get(payload[0])
                if name:
                    self.status[name] = value
            return
        if cmd == 0x0F and len(payload) == 1:
            # One value for both: 00 neither, 01 split, 11 DUP-, 12 DUP+.
            self.state["split"] = 1 if payload[0] == 0x01 else 0
            self.state["dup"] = payload[0] if payload[0] in (0x11, 0x12) else 0x10
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
            if key in ("anotch", "mnotch"):
                self.state["notch"] = 1 if self.state.get("anotch") else \
                    2 if self.state.get("mnotch") else 0

    def _update_status(self, cmd: int, payload: bytes) -> bool:
        """The display-only replies. True if the frame was one of them."""
        n = len(payload)
        if cmd == 0x25 and n == 6:
            self.status["vfo_sel_hz" if payload[0] == 0 else "vfo_other_hz"] = \
                civ.decode_freq(payload[1:6])
            return True
        if cmd == 0x26 and n >= 2:
            which = "vfo_sel" if payload[0] == 0 else "vfo_other"
            self.status[which + "_mode"] = civ.MODES.get(payload[1], "dv" if payload[1] == 0x17
                                                         else f"{payload[1]:02x}")
            if n >= 3:
                self.status[which + "_data"] = bool(payload[2])
            if n >= 4:
                self.status[which + "_filter"] = payload[3]
            return True
        if cmd == 0x21 and n == 4 and payload[0] == 0x00:
            hz = civ.decode_freq(payload[1:3])
            self.status["rit_hz"] = -hz if payload[3] == 0x01 else hz
            return True
        if cmd == 0x0C and n == 3:
            self.status["offset_hz"] = civ.decode_freq(payload) * 100    # 100 Hz units
            return True
        if cmd == 0x1B and n == 4 and payload[0] in (0x00, 0x01):
            self.status["tone_hz" if payload[0] == 0 else "tsql_hz"] = _bcd_int(payload[1:4]) / 10
            return True
        if cmd == 0x1B and n == 4 and payload[0] == 0x02:
            self.status["dtcs"] = f"{_bcd_int(payload[2:4]):03d}"
            self.status["dtcs_polarity"] = payload[1]
            return True
        if cmd == 0x15 and n == 2 and payload[0] == 0x07:
            self.status["ovf"] = bool(payload[1])
            return True
        return False

    def _read_control(self, key: str) -> None:
        if key in _READ_VIA:
            for other in _READ_VIA[key]:
                self._read_control(other)
            return
        cmd, sub, _form = _CONTROL_CI_V[key]
        self._send(cmd, b"" if sub is None else bytes([sub]))

    def poll(self, now: float | None = None) -> None:
        """Ask for the meter, and now and then everything else. Called by the reader."""
        now = time.monotonic() if now is None else now
        if now >= self._next_squelch and not self.transmitting:
            self._next_squelch = now + SQUELCH_POLL_S
            self._send(0x15, b"\x01")            # squelch open?
        if now >= self._next_meter:
            self._next_meter = now + METER_POLL_S
            if self.transmitting:
                for meter in (0x11, 0x12, 0x13, 0x14, 0x15, 0x16):  # Po SWR ALC COMP Vd Id
                    self._send(0x15, bytes([meter]))
            else:
                self._send(0x15, b"\x02")        # S-meter
                self._send(0x15, b"\x07")        # overflow
        if now >= self._next_settings:
            self._next_settings = now + SETTINGS_POLL_S
            self._send(civ.CMD_READ_MODE)
            for key in _CONTROL_CI_V:
                if key not in _READ_VIA:
                    self._read_control(key)
            for cmd, payload in _STATUS_READS:
                self._send(cmd, payload)

    # -- control ------------------------------------------------------------------------

    @property
    def controls(self) -> tuple[RadioControl, ...]:
        return IC705_CONTROLS

    def set_control(self, key: str, value: int) -> None:
        """Change one of the radio's settings. A refusal (FA) sets `refused` and the
        radio's real value is read back into `state`."""
        cmd, sub, form = _CONTROL_CI_V[key]
        self.state[key] = int(value)            # optimistic; a refusal corrects it
        if form == "split":
            # 0F writes: 00 split off, 01 split on, 10 simplex, 11 DUP-, 12 DUP+.
            self._send(0x0F, b"\x01" if value else b"\x00", label=key)
            if value:
                self.state["dup"] = 0x10
            return
        if form == "dup":
            self._send(0x0F, bytes([int(value)]), label=key)
            if value != 0x10:
                self.state["split"] = 0
            return
        if form == "notch":
            # One key on the radio: OFF -> AN -> MN. Off goes first so both are never on.
            auto, manual = int(value) == 1, int(value) == 2
            self.state["anotch"], self.state["mnotch"] = int(auto), int(manual)
            for sub_cmd, on in sorted(((0x41, auto), (0x48, manual)), key=lambda t: t[1]):
                self._send(0x16, bytes([sub_cmd, 0x01 if on else 0x00]), label=key)
            return
        if form == "level":
            data = civ.encode_level(int(value))
        elif form == "att":
            data = bytes([0x20 if value else 0x00])
        else:
            data = bytes([int(value)])
        self._send(cmd, (b"" if sub is None else bytes([sub])) + data, label=key)

    # -- the SET menu (1A 05) ------------------------------------------------------------

    @staticmethod
    def _menu_item(number: int) -> bytes:
        return bytes.fromhex(f"{number:04d}")          # item 0110 -> 01 10

    def read_menu(self, number: int) -> int | None:
        """One SET-menu item's value (its BCD data as a number), or None if the radio did
        not answer. Blocks for up to REPLY_TIMEOUT_S -- call it off the UI thread."""
        head = b"\x05" + self._menu_item(number)
        frame = self._ask(0x1A, head, prefix=head)
        if frame is None or len(frame.payload) <= 3:
            return None
        data = bytes(frame.payload[3:])
        if number in MENU_TAKEN and MENU_TAKEN[number] in self._taken:
            data = self._taken[MENU_TAKEN[number]]      # the radio's own, not the app's
        try:
            return int(data.hex())
        except ValueError:
            return None

    def write_menu(self, number: int, value: int, digits: int) -> None:
        """Set a SET-menu item. Refused if it is one the app itself depends on."""
        if number in MENU_LOCKED:
            raise PermissionError(MENU_LOCKED[number])
        data = bytes.fromhex(f"{int(value):0{digits + digits % 2}d}")
        self._send(0x1A, b"\x05" + self._menu_item(number) + data, label=f"menu {number:04d}")

    # -- memory channels (1A 00) ----------------------------------------------------------

    def read_memory(self, group: int, channel: int):
        """A memory channel: an `ic705_memory.MemoryChannel`, "blank", or None if the
        radio did not answer. Blocks -- call it off the UI thread."""
        from .ic705_memory import MemoryChannel, address

        head = b"\x00" + address(group, channel)
        frame = self._ask(0x1A, head, prefix=head)
        if frame is None:
            return None
        data = bytes(frame.payload[5:])
        if data == b"\xff":
            return "blank"
        return MemoryChannel.decode(group, channel, data)

    def write_memory(self, memory) -> None:
        from .ic705_memory import address

        self._send(0x1A, b"\x00" + address(memory.group, memory.channel) + memory.encode(),
                   label=f"memory {memory.label}")

    def clear_memory(self, group: int, channel: int) -> None:
        from .ic705_memory import address

        self._send(0x1A, b"\x00" + address(group, channel) + b"\xff", label="memory clear")

    @property
    def tx_freq_hz(self) -> float:
        """Where the radio transmits: the other VFO in SPLIT (or a memory channel's
        transmit side, which 25 01 also reads), the offset away in DUP-/DUP+, else the
        operating frequency. The radio does the switching itself; this is for display."""
        if self.state.get("split") and self.status.get("vfo_other_hz"):
            return float(self.status["vfo_other_hz"])
        offset = float(self.status.get("offset_hz") or 0)
        dup = self.state.get("dup")
        if dup == 0x11:
            return self.center_freq - offset
        if dup == 0x12:
            return self.center_freq + offset
        return self.center_freq

    # -- VFO / memory keys (the radio's VFO/MEMORY screen) ----------------------------------

    def select_vfo(self, which: str | None = None) -> None:
        """VFO mode (07), on VFO `which` ("A"/"B") or the last one used."""
        which = which or self.last_vfo
        self._send(0x07, b"", label="vfo")
        self._send(0x07, b"\x00" if which == "A" else b"\x01", label="vfo")
        self.vfo_mode = self.last_vfo = which

    def swap_vfo(self) -> None:
        """A/B: the other VFO."""
        self.select_vfo("B" if self.last_vfo == "A" else "A")

    def equalize_vfo(self) -> None:
        """A=B: copy the displayed VFO to the other (07 A0; the A/B key held)."""
        self._send(0x07, b"\xa0", label="vfo")

    def select_memory(self, group: int | None = None, channel: int | None = None) -> None:
        """Memory mode (08), optionally on a group (08 A0) and channel (08 xxxx). Entering
        memory mode comes first: selecting a group or channel alone does not switch."""
        from .ic705_memory import CALL_GROUP

        if group is not None and group != CALL_GROUP:
            self.memo_group = group
        if channel is not None:
            self.memo_channel = channel
        self._send(0x08, b"", label="memory")
        self._send(0x08, b"\xa0" + bytes.fromhex(f"{self.memo_group:04d}"), label="memory")
        self._send(0x08, bytes.fromhex(f"{self.memo_channel:04d}"), label="memory")
        self.vfo_mode = "MEMO"
        self._fetch_memo(self.memo_group, self.memo_channel)

    def select_call(self, channel: int | None = None) -> None:
        """Call channel mode: the call-channel group (0100) in memory mode."""
        from .ic705_memory import CALL_GROUP

        if channel is not None:
            self.call_channel = channel
        self._send(0x08, b"", label="memory")
        self._send(0x08, b"\xa0\x01\x00", label="memory")
        self._send(0x08, bytes.fromhex(f"{self.call_channel:04d}"), label="memory")
        self.vfo_mode = "CALL"
        self._fetch_memo(CALL_GROUP, self.call_channel)

    def step_channel(self, step: int) -> None:
        """Next or previous memory (or call) channel, as the dial does in memory mode."""
        if self.vfo_mode == "CALL":
            self.select_call((self.call_channel + step) % 4)
        elif self.vfo_mode == "MEMO":
            self.select_memory(channel=(self.memo_channel + step) % 100)

    def memory_write(self) -> None:
        """MW: the VFO into the selected memory channel (09)."""
        self._send(0x09, b"", label="memory write")
        self._fetch_memo(self.memo_group, self.memo_channel)

    def memory_clear(self) -> None:
        """M-CLR: clear the selected memory channel (0B)."""
        self._send(0x0B, b"", label="memory clear")
        self._fetch_memo(self.memo_group, self.memo_channel)

    def memory_to_vfo(self) -> None:
        """M->VFO: copy the memory channel to the VFO and go to VFO mode (0A)."""
        self._send(0x0A, b"", label="memory to vfo")
        self.vfo_mode = self.last_vfo

    def set_select(self, number: int) -> None:
        """SELECT: mark the memory channel as select channel 1-3, or 0 to clear (0E B1 / B0)."""
        if number:
            self._send(0x0E, b"\xb1" + bytes([number]), label="select")
        else:
            self._send(0x0E, b"\xb0", label="select")
        if hasattr(self.memo_contents, "select"):
            self.memo_contents.select = number

    def _fetch_memo(self, group: int, channel: int) -> None:
        """Read the selected channel's contents in the background, for its name."""
        self.memo_contents = None

        def fetch() -> None:
            self._confirmed(1.0)
            result = self.read_memory(group, channel)
            if (group, channel) in ((self.memo_group, self.memo_channel),
                                    (100, self.call_channel)):
                self.memo_contents = result

        if self._thread is None:
            fetch()                     # no poller running: nothing else reads the port
        else:
            threading.Thread(target=fetch, name="memo-read", daemon=True).start()

    def set_rit(self, hz: int) -> None:
        """Set the RIT/dTX offset (CI-V 21 00: two BCD bytes, low first, then the sign;
        checked on the radio: +120 -> 20 01 00, -1234 -> 34 12 01)."""
        hz = max(-RIT_LIMIT_HZ, min(RIT_LIMIT_HZ, int(round(hz))))
        self.status["rit_hz"] = hz
        self._send(0x21, b"\x00" + civ.encode_freq(abs(hz), nbytes=2)
                   + (b"\x01" if hz < 0 else b"\x00"), label="rit_hz")

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

    def _ask(self, cmd: int, payload: bytes | tuple = b"",
             prefix: bytes | None = None) -> civ.Frame | None:
        """Send a query and wait for the answer with the same command. With `prefix`, only
        an answer whose data starts with it counts, and refusals are not taken as the
        answer (the poller's own reads can be refused meanwhile): a refused query just
        times out."""
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
                    if prefix is not None:
                        if frame.cmd == cmd and bytes(frame.payload[:len(prefix)]) == prefix:
                            return frame
                    elif frame.cmd == cmd or frame.is_ng:
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

    # -- taking over the speaker and the TX audio source --------------------------------

    def _read_takeover(self, name: str) -> bytes | None:
        # Matched on the exact reply: while the poller runs, other 14 xx and FA answers
        # arrive too, and taking one of those for AF would restore the wrong level.
        if name == "af":
            frame = self._ask(0x14, b"\x01", prefix=b"\x01")
            return frame.payload[1:3] if frame is not None and len(frame.payload) == 3 else None
        head = bytes([0x05, *MENU_ITEMS[name]])
        frame = self._ask(0x1A, head, prefix=head)
        return frame.payload[3:] if frame is not None and len(frame.payload) > 3 else None

    def _write_takeover(self, name: str, value: bytes) -> None:
        if name == "af":
            self._send(0x14, b"\x01" + value, label="af")
            self.state["af"] = civ.decode_level(value)
        else:
            self._send(0x1A, bytes([0x05, *MENU_ITEMS[name]]) + value, label=name)

    def _take_over(self) -> None:
        """Silence the speaker and take TX audio from USB, remembering the radio's own
        settings -- on disk too, so a crash does not lose them. If the file is already
        there, a previous run did not finish: its values are the radio's real ones."""
        saved: dict[str, str] = {}
        try:
            saved = json.loads(self.restore_file.read_text())
        except (OSError, ValueError):
            saved = {}
        for name in TAKEOVER:
            if name in saved:
                self._taken[name] = bytes.fromhex(saved[name])
            else:
                value = self._read_takeover(name)
                if value is not None:
                    self._taken[name] = value
        try:
            self.restore_file.parent.mkdir(parents=True, exist_ok=True)
            self.restore_file.write_text(json.dumps({k: v.hex() for k, v in self._taken.items()}))
        except OSError:
            pass
        for name, value in TAKEOVER.items():
            if name in self._taken:
                self._write_takeover(name, value)

    def _confirmed(self, timeout: float = REPLY_TIMEOUT_S) -> bool:
        """Wait, reading inline, until every write sent has been answered. The radio
        drops commands sent back-to-back while it is busy (found 2026-09-28: the last two
        hand-back writes were lost when the port closed straight after), so the hand-back
        goes one write at a time."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._write_lock:
                if not any(write for _cmd, _label, write in self._pending):
                    return True
            if self._thread is None:
                self._pump_once()
            else:
                time.sleep(0.01)
        return False

    def _hand_back(self) -> None:
        for name, value in self._taken.items():
            for _attempt in range(3):
                self._write_takeover(name, value)
                if self._confirmed():
                    break
        self._taken.clear()
        try:
            self.restore_file.unlink()
        except OSError:
            pass

    def set_span(self, half_span_hz: float) -> None:
        """The scope's span in centre mode, as a half-span (+/-): 2.5 kHz to 500 kHz."""
        self._send(civ.CMD_SCOPE, bytes([civ.SCOPE_SPAN, 0x00]) + civ.encode_freq(half_span_hz),
                   label="span")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._take_over()
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
        # The scope stream off first, so the radio is not busy sending while the
        # settings go back; each one is confirmed before the next.
        with self._write_lock:
            self._pending.clear()
        out = self._restore.get(civ.SCOPE_OUTPUT, b"\x00")
        self._send(civ.CMD_SCOPE, bytes([civ.SCOPE_OUTPUT]) + out[:1], label="scope")
        self._confirmed()
        self._hand_back()
        on = self._restore.get(civ.SCOPE_ON)
        if on is not None:
            self._send(civ.CMD_SCOPE, bytes([civ.SCOPE_ON]) + on[:1], label="scope")
            self._confirmed()
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
