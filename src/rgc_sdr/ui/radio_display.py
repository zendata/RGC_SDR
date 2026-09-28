"""The IC-705's own display, rebuilt on the Mac from its CI-V state.

Not a picture of the radio's screen -- CI-V carries no image -- but the same information
laid out the same way: the operating VFO large, the other small, mode and filter, TX/RX,
split/duplex, tone, RIT, the function indicators, and a meter chosen like the radio's.

Reads what `IcomSource` has gathered (`state`, `status`, `mode`, meters); a stand-in with
the same attributes drives it in tests.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from ..device.icom import comp_db, drain_amps, power_percent, s_meter_text, supply_volts, swr_value

_LIT = "color: #10141a; background: #f0b429; border-radius: 3px; padding: 1px 5px;"
_DIM = "color: #5c6773; background: #1a2029; border-radius: 3px; padding: 1px 5px;"
_TX = "color: white; background: #d62828; border-radius: 3px; padding: 2px 8px; font-weight: bold;"
_RX = "color: #10141a; background: #37b24d; border-radius: 3px; padding: 2px 8px; font-weight: bold;"
_WARN = "color: white; background: #d62828; border-radius: 3px; padding: 1px 5px;"

#: Meters as the radio names them; the first is shown on receive, the rest on transmit.
METERS = ("S", "Po", "SWR", "ALC", "COMP", "Id")


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
    if meter == "Id":
        raw = status.get("id")
        return None if raw is None else (min(raw / 241, 1.0), f"{drain_amps(raw):.2f} A")
    return None


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
        ("anotch", "A-NOTCH"), ("mnotch", "NOTCH"), ("comp", "COMP"), ("vox", "VOX"),
        ("bkin", None), ("moni", "MONI"), ("lock", "LOCK"),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("radioDisplay")
        self.setStyleSheet("#radioDisplay { background: #0b0f14; border: 1px solid #263040;"
                           " border-radius: 6px; } QLabel { color: #d8dee9; }")
        mono = QtGui.QFont("Menlo")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        # Line 1: TX/RX, the operating VFO large, mode/filter, then the other VFO.
        top = QtWidgets.QHBoxLayout()
        self.txrx = QtWidgets.QLabel("RX")
        self.txrx.setStyleSheet(_RX)
        top.addWidget(self.txrx)
        self.vfo_name = QtWidgets.QLabel("VFO")
        top.addWidget(self.vfo_name)
        self.freq = QtWidgets.QLabel(format_freq(None))
        big = QtGui.QFont(mono)
        big.setPointSize(26)
        big.setBold(True)
        self.freq.setFont(big)
        self.freq.setStyleSheet("color: #f5f7fa;")
        top.addWidget(self.freq)
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
            button.setFixedWidth(52)
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

    def update_from(self, src) -> None:
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

        split = state.get("split")
        offset = status.get("offset_hz")
        if split == 0x01:
            self.duplex.setText("SPLIT")
            self.duplex.setStyleSheet(_LIT)
        elif split in (0x11, 0x12):
            sign = "−" if split == 0x11 else "+"
            text = f"DUP{sign}"
            if offset:
                text += f" {offset / 1e3:g} kHz"
            self.duplex.setText(text)
            self.duplex.setStyleSheet(_LIT)
        else:
            self.duplex.setText("")

        tone_mode = state.get("tone")
        if tone_mode == 1 and status.get("tone_hz"):
            self.tone.setText(f"TONE {status['tone_hz']:.1f}")
        elif tone_mode == 2 and status.get("tsql_hz"):
            self.tone.setText(f"TSQL {status['tsql_hz']:.1f}")
        elif tone_mode == 3 and status.get("dtcs"):
            self.tone.setText(f"DTCS {status['dtcs']}")
        else:
            self.tone.setText("")
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
