"""The IC-705's own display, rebuilt on the Mac from its CI-V state.

Not a picture of the radio's screen -- CI-V carries no image -- but the same information
laid out the same way: the operating VFO large, the other small, mode and filter, TX/RX,
split/duplex, tone, RIT, the function indicators, and a meter chosen like the radio's.

Reads what `IcomSource` has gathered (`state`, `status`, `mode`, meters); a stand-in with
the same attributes drives it in tests.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from .vfo_memory_panel import _INDICATOR, VfoMemoryPanel, indicator_text
from ..device.icom import comp_db, drain_amps, power_percent, s_meter_text, supply_volts, swr_value

_LIT = "color: #10141a; background: #f0b429; border-radius: 3px; padding: 1px 5px;"
_DIM = "color: #5c6773; background: #1a2029; border-radius: 3px; padding: 1px 5px;"
_TX = "color: white; background: #d62828; border-radius: 3px; padding: 2px 8px; font-weight: bold;"
_RX = "color: #10141a; background: #37b24d; border-radius: 3px; padding: 2px 8px; font-weight: bold;"
_WARN = "color: white; background: #d62828; border-radius: 3px; padding: 1px 5px;"

#: Meters as the radio names them; the first is shown on receive, the rest on transmit.
METERS = ("S", "Po", "SWR", "ALC", "COMP", "Vd", "Id")


def format_freq(hz: float | None) -> str:
    """The radio's style: 145.650.00 (MHz, kHz, then tens of hertz)."""
    if hz is None:
        return "---.---.--"
    hz = int(round(hz))
    mhz, rest = divmod(hz, 1_000_000)
    khz, hertz = divmod(rest, 1_000)
    return f"{mhz}.{khz:03d}.{hertz // 10:02d}"


def meter_reading(meter: str, src) -> tuple[float, str] | None:
    """(fraction of full scale, text) for a meter, from what the radio last reported."""
    status = getattr(src, "status", {}) or {}
    if meter == "S":
        raw = getattr(src, "smeter", None)
        return None if raw is None else (raw / 241, s_meter_text(raw))
    if meter == "Po":
        raw = getattr(src, "po", None)
        return None if raw is None else (power_percent(raw) / 100, f"{power_percent(raw):.0f}%")
    if meter == "SWR":
        raw = getattr(src, "swr", None)
        return None if raw is None else (min(raw / 120, 1.0), f"{swr_value(raw):.1f}")
    if meter == "ALC":
        raw = status.get("alc")
        return None if raw is None else (min(raw / 120, 1.0), f"{raw / 120 * 100:.0f}%")
    if meter == "COMP":
        raw = status.get("comp_meter")
        return None if raw is None else (min(raw / 210, 1.0), f"{comp_db(raw):.1f} dB")
    if meter == "Vd":
        raw = status.get("vd")
        return None if raw is None else (min(raw / 241, 1.0), f"{supply_volts(raw):.1f} V")
    if meter == "Id":
        raw = status.get("id")
        return None if raw is None else (min(raw / 241, 1.0), f"{drain_amps(raw):.2f} A")
    return None


def tone_text(mode, status) -> str:
    """The tone indicator for the 705's tone setting (16 5D), with its tone or code."""
    tone = f"{status['tone_hz']:.1f}" if status.get("tone_hz") else "?"
    tsql = f"{status['tsql_hz']:.1f}" if status.get("tsql_hz") else "?"
    dtcs = status.get("dtcs") or "?"
    return {
        1: f"TONE {tone}", 2: f"TSQL {tsql}", 3: f"DTCS {dtcs}", 6: f"DTCS(T) {dtcs}",
        7: f"TONE {tone} / DTCS {dtcs}", 8: f"DTCS {dtcs} / TSQL {tsql}",
        9: f"TONE {tone} / TSQL {tsql}",
    }.get(mode, "")


class _Bar(QtWidgets.QWidget):
    """A plain level bar, as the radio draws its meter."""

    def __init__(self) -> None:
        super().__init__()
        self.fraction = 0.0
        self.setFixedHeight(10)
        self.setMinimumWidth(220)

    def set_fraction(self, fraction: float) -> None:
        self.fraction = max(0.0, min(1.0, fraction))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#1a2029"))
        filled = QtCore.QRectF(0, 0, self.width() * self.fraction, self.height())
        grad = QtGui.QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, QtGui.QColor("#37b24d"))
        grad.setColorAt(0.7, QtGui.QColor("#f0b429"))
        grad.setColorAt(1.0, QtGui.QColor("#d62828"))
        p.fillRect(filled, QtGui.QBrush(grad))
        p.end()


class RadioDisplay(QtWidgets.QFrame):
    """The 705's main screen, as information."""

    #: Indicator chips: (state key, label, how to tell it is on).
    CHIPS = (
        ("preamp", None), ("att", "ATT"), ("agc", None), ("nb", "NB"), ("nr", "NR"),
        ("notch", None), ("comp", "COMP"), ("vox", "VOX"), ("bkin", None),
        ("moni", "MONI"), ("lock", "LOCK"), ("rfg", "RFG"),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("radioDisplay")
        self.setStyleSheet("#radioDisplay { background: #0b0f14; border: 1px solid #263040;"
                           " border-radius: 6px; } QLabel { color: #d8dee9; }")
        mono = QtGui.QFont("Menlo")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)

        self._src = None
        self.vfo_panel: VfoMemoryPanel | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        # Line 1: TX/RX, the operating VFO large, mode/filter, then the other VFO.
        top = QtWidgets.QHBoxLayout()
        self.txrx = QtWidgets.QLabel("RX")
        self.txrx.setStyleSheet(_RX)
        top.addWidget(self.txrx)
        self.freq = QtWidgets.QLabel(format_freq(None))
        big = QtGui.QFont(mono)
        big.setPointSize(26)
        big.setBold(True)
        self.freq.setFont(big)
        self.freq.setStyleSheet("color: #f5f7fa;")
        top.addWidget(self.freq)
        # Right of the frequency, as on the radio: VFO A/B or MEMO and the channel.
        # Clicking it opens the VFO/MEMORY screen; the arrows step memory channels.
        vfo_box = QtWidgets.QHBoxLayout()
        vfo_box.setSpacing(2)
        self.vfo_indicator = QtWidgets.QPushButton("VFO/MEMO ?")
        self.vfo_indicator.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.vfo_indicator.setStyleSheet(_INDICATOR)
        self.vfo_indicator.setToolTip(
            "VFO/MEMORY: click for the radio's VFO/MEMORY keys.\nThe radio does not report "
            "this over CI-V, so it shows what was last chosen from the Mac.")
        self.vfo_indicator.clicked.connect(self.open_vfo_panel)
        vfo_box.addWidget(self.vfo_indicator)
        arrows = QtWidgets.QVBoxLayout()
        arrows.setSpacing(0)
        self.channel_up = QtWidgets.QToolButton()
        self.channel_up.setArrowType(QtCore.Qt.ArrowType.UpArrow)
        self.channel_down = QtWidgets.QToolButton()
        self.channel_down.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        for button, step in ((self.channel_up, 1), (self.channel_down, -1)):
            button.setFixedSize(18, 14)
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            button.setToolTip("Next / previous memory channel")
            button.clicked.connect(lambda _c=False, st=step: self._step(st))
            arrows.addWidget(button)
        vfo_box.addLayout(arrows)
        top.addLayout(vfo_box)
        self.mode = QtWidgets.QLabel("")
        mode_font = QtGui.QFont(mono)
        mode_font.setPointSize(15)
        mode_font.setBold(True)
        self.mode.setFont(mode_font)
        self.mode.setStyleSheet("color: #66d9ef;")
        top.addWidget(self.mode)
        self.filter = QtWidgets.QLabel("")
        self.filter.setStyleSheet(_DIM)
        top.addWidget(self.filter)
        top.addSpacing(18)
        self.duplex = QtWidgets.QLabel("")
        top.addWidget(self.duplex)
        self.tone = QtWidgets.QLabel("")
        top.addWidget(self.tone)
        self.rit = QtWidgets.QLabel("")
        top.addWidget(self.rit)
        top.addStretch(1)
        self.other = QtWidgets.QLabel("")
        small = QtGui.QFont(mono)
        small.setPointSize(12)
        self.other.setFont(small)
        self.other.setStyleSheet("color: #8b98a8;")
        top.addWidget(self.other)
        outer.addLayout(top)

        # Line 2: the function indicators, lit when on.
        chips = QtWidgets.QHBoxLayout()
        chips.setSpacing(4)
        self.chips: dict[str, QtWidgets.QLabel] = {}
        for key, label in self.CHIPS:
            chip = QtWidgets.QLabel(label or key.upper())
            chip.setStyleSheet(_DIM)
            chips.addWidget(chip)
            self.chips[key] = chip
        self.ovf = QtWidgets.QLabel("OVF")
        self.ovf.setStyleSheet(_DIM)
        chips.addWidget(self.ovf)
        chips.addStretch(1)
        self.volts = QtWidgets.QLabel("")
        self.volts.setFont(small)
        chips.addWidget(self.volts)
        outer.addLayout(chips)

        # Line 3: the meter, chosen as on the radio.
        meter_row = QtWidgets.QHBoxLayout()
        self.meter_buttons: dict[str, QtWidgets.QPushButton] = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for name in METERS:
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            button.setMinimumWidth(64)
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            group.addButton(button)
            meter_row.addWidget(button)
            self.meter_buttons[name] = button
        self.meter_buttons["S"].setChecked(True)
        self.bar = _Bar()
        meter_row.addWidget(self.bar, 1)
        self.meter_text = QtWidgets.QLabel("--")
        self.meter_text.setFont(small)
        self.meter_text.setMinimumWidth(90)
        meter_row.addWidget(self.meter_text)
        outer.addLayout(meter_row)

    @property
    def meter(self) -> str:
        for name, button in self.meter_buttons.items():
            if button.isChecked():
                return name
        return "S"

    def open_vfo_panel(self) -> None:
        if self._src is None or not hasattr(self._src, "select_vfo"):
            return
        self.vfo_panel = VfoMemoryPanel(self._src, self)
        self.vfo_panel.move(self.vfo_indicator.mapToGlobal(
            QtCore.QPoint(0, self.vfo_indicator.height())))
        self.vfo_panel.show()

    def _step(self, step: int) -> None:
        if self._src is not None and hasattr(self._src, "step_channel"):
            self._src.step_channel(step)

    def update_from(self, src) -> None:
        self._src = src
        top, second = indicator_text(src)
        text = f"{top}\n{second}" if second else top
        if self.vfo_indicator.text() != text:
            self.vfo_indicator.setText(text)
        in_memory = getattr(src, "vfo_mode", None) in ("MEMO", "CALL")
        self.channel_up.setEnabled(in_memory)
        self.channel_down.setEnabled(in_memory)
        if self.vfo_panel is not None and self.vfo_panel.isVisible():
            self.vfo_panel.sync()
        state = getattr(src, "state", {}) or {}
        status = getattr(src, "status", {}) or {}
        transmitting = bool(getattr(src, "transmitting", False))

        self.txrx.setText("TX" if transmitting else "RX")
        self.txrx.setStyleSheet(_TX if transmitting else _RX)
        self.freq.setText(format_freq(status.get("vfo_sel_hz") or getattr(src, "center_freq", None)))
        mode = (getattr(src, "mode", None) or status.get("vfo_sel_mode") or "").upper()
        if status.get("vfo_sel_data"):
            mode += "-D"
        self.mode.setText(mode)
        flt = getattr(src, "filter", None) or status.get("vfo_sel_filter")
        self.filter.setText(f"FIL{flt}" if flt else "")
        other = status.get("vfo_other_hz")
        other_mode = (status.get("vfo_other_mode") or "").upper()
        self.other.setText(f"⇄ {format_freq(other)}  {other_mode}" if other else "")

        dup = state.get("dup")
        offset = status.get("offset_hz")
        if state.get("split"):
            self.duplex.setText("SPLIT")
            self.duplex.setStyleSheet(_LIT)
        elif dup in (0x11, 0x12):
            sign = "−" if dup == 0x11 else "+"
            text = f"DUP{sign}"
            if offset:
                text += f" {offset / 1e3:g} kHz"
            self.duplex.setText(text)
            self.duplex.setStyleSheet(_LIT)
        else:
            self.duplex.setText("")

        self.tone.setText(tone_text(state.get("tone"), status))
        self.tone.setStyleSheet(_LIT if self.tone.text() else "")

        rit_hz = status.get("rit_hz")
        parts = []
        if state.get("rit") and rit_hz is not None:
            parts.append(f"RIT {rit_hz / 1e3:+.2f}")
        if state.get("dtx") and rit_hz is not None:
            parts.append(f"ΔTX {rit_hz / 1e3:+.2f}")
        self.rit.setText("  ".join(parts))
        self.rit.setStyleSheet(_LIT if parts else "")

        for key, chip in self.chips.items():
            value = state.get(key)
            if key == "preamp":
                chip.setText({1: "P.AMP1", 2: "P.AMP2"}.get(value, "P.AMP"))
                lit = bool(value)
            elif key == "agc":
                chip.setText({1: "AGC-F", 2: "AGC-M", 3: "AGC-S"}.get(value, "AGC"))
                lit = value is not None
            elif key == "bkin":
                chip.setText({1: "BK-IN", 2: "F-BKIN"}.get(value, "BK-IN"))
                lit = bool(value)
            elif key == "notch":
                chip.setText({1: "AN", 2: "MN"}.get(value, "NOTCH"))
                lit = bool(value)
            elif key == "rfg":
                # "RFG" shows on the radio whenever the RF gain is turned down.
                rf = state.get("rf")
                lit = rf is not None and rf < 255
            else:
                lit = bool(value)
            chip.setStyleSheet(_LIT if lit else _DIM)
        self.ovf.setStyleSheet(_WARN if status.get("ovf") else _DIM)
        vd = status.get("vd")
        self.volts.setText(f"{supply_volts(vd):.1f} V" if vd is not None else "")

        # Receive always shows S, as the radio does; the chosen meter shows on transmit.
        reading = meter_reading("S" if not transmitting else self.meter, src)
        if reading is None:
            self.bar.set_fraction(0.0)
            self.meter_text.setText("--")
        else:
            self.bar.set_fraction(reading[0])
            label = "S" if not transmitting else self.meter
            self.meter_text.setText(f"{label} {reading[1]}")
