"""The IC-705's FUNCTION screen as buttons on the Mac.

On/off keys light when on; multi-setting keys (P.AMP, AGC, BK-IN, TONE, SPLIT) step to
the next setting on each click and show it. A right-click -- or pressing and holding, as
a long touch on the radio -- opens the levels behind a key, as the radio's function menu
does (NB, NR, NOTCH, VOX, COMP, MONI, BKIN). The buttons follow the radio: a change made
on its own screen shows here.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from ..device.icom import MULTI_BY_MODE, RIT_LIMIT_HZ

_OFF = ("QPushButton { color: #c8d1dc; background: #1a2029; border: 1px solid #2c3645;"
        " border-radius: 4px; padding: 4px 6px; }")
_ON = ("QPushButton { color: #10141a; background: #f0b429; border: 1px solid #f0b429;"
       " border-radius: 4px; padding: 4px 6px; font-weight: bold; }")

#: How long a press has to be held to count as a long touch.
LONG_PRESS_MS = 500


class FunctionButton(QtWidgets.QPushButton):
    """One function key. `long_pressed` fires on a held press or a right-click."""

    long_pressed = QtCore.pyqtSignal()

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setMinimumWidth(78)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(LONG_PRESS_MS)
        self._timer.timeout.connect(self._held)
        self._was_long = False

    def _held(self) -> None:
        self._was_long = True
        self.long_pressed.emit()

    def mousePressEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.long_pressed.emit()
            return
        self._was_long = False
        self._timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        self._timer.stop()
        if self._was_long:
            # A long touch opens the level; it is not also a click.
            self.setDown(False)
            self._was_long = False
            return
        super().mouseReleaseEvent(event)


def level_text(raw: int, scale) -> str:
    """A 0-255 level as the radio shows it: in its own units, or as a percentage."""
    if scale is None:
        return f"{round(raw / 255 * 100)}%"
    low, high, unit = scale
    value = low + raw / 255 * (high - low)
    return f"{value:.1f}{unit}" if isinstance(low, float) else f"{round(value):+d}" \
        if low < 0 else f"{round(value)}{unit}"


class LevelPopup(QtWidgets.QFrame):
    """Sliders for a menu of levels, shown under its key. `levels` is
    ((key, title, raw 0-255, scale), ...); `on_change(key, raw)` is called when a slider
    is let go or stepped."""

    def __init__(self, levels, on_change, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self.setStyleSheet("QFrame { background: #11161d; border: 1px solid #2c3645; }"
                           " QLabel { color: #d8dee9; }")
        grid = QtWidgets.QGridLayout(self)
        self.sliders: dict[str, QtWidgets.QSlider] = {}
        for row, (key, title, raw, scale) in enumerate(levels):
            grid.addWidget(QtWidgets.QLabel(title), row, 0)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, 255)
            slider.setValue(int(raw))
            slider.setFixedWidth(220)
            grid.addWidget(slider, row, 1)
            value = QtWidgets.QLabel(level_text(int(raw), scale))
            value.setMinimumWidth(64)
            grid.addWidget(value, row, 2)
            slider.valueChanged.connect(
                lambda v, lab=value, sc=scale: lab.setText(level_text(v, sc)))
            # Sent when let go, not on every step of a drag.
            slider.sliderReleased.connect(
                lambda k=key, sl=slider: on_change(k, sl.value()))
            slider.valueChanged.connect(
                lambda v, k=key, sl=slider: None if sl.isSliderDown() else on_change(k, v))
            self.sliders[key] = slider

    @property
    def slider(self) -> QtWidgets.QSlider:
        """The first slider -- most keys have only one."""
        return next(iter(self.sliders.values()))


class FunctionPanel(QtWidgets.QWidget):
    """Buttons for the source's `function` controls; levels from its `popup` ones."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(4)
        self.buttons: dict[str, FunctionButton] = {}
        self._controls: dict = {}
        self._source = None
        self._syncing = False
        self.popup: LevelPopup | None = None
        self.multi_button: FunctionButton | None = None
        self.rit_spin: QtWidgets.QSpinBox | None = None

    def build(self, source, columns: int = 9) -> None:
        """Lay out a button per function control of `source`, then MULTI and RIT."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.buttons.clear()
        self._source = source
        controls = getattr(source, "controls", ()) or ()
        self._controls = {c.key: c for c in controls}
        functions = [c for c in controls if c.placement == "function"]
        for i, control in enumerate(functions):
            button = FunctionButton(control.label)
            tip = control.tooltip
            if control.levels:
                tip += "\nRight-click or hold for its levels"
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, k=control.key: self._clicked(k))
            if control.levels:
                button.long_pressed.connect(lambda k=control.key: self.open_level(k))
            self._grid.addWidget(button, i // columns, i % columns)
            self.buttons[control.key] = button
        n = len(functions)
        # MULTI: the radio's multi-function knob menu for the current mode.
        multi = FunctionButton("MULTI")
        multi.setToolTip("The MULTI knob's menu for this mode: RF power, MIC gain, COMP, "
                         "key speed, CW pitch, monitor, twin PBT")
        multi.clicked.connect(self.open_multi)
        multi.long_pressed.connect(self.open_multi)
        multi.setStyleSheet(_OFF)
        self._grid.addWidget(multi, n // columns, n % columns)
        self.multi_button = multi
        n += 1
        # RIT / dTX offset, as the MULTI knob sets it when RIT or dTX is on.
        rit_box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(rit_box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.rit_spin = QtWidgets.QSpinBox()
        self.rit_spin.setRange(-RIT_LIMIT_HZ, RIT_LIMIT_HZ)
        self.rit_spin.setSingleStep(10)
        self.rit_spin.setSuffix(" Hz")
        self.rit_spin.setKeyboardTracking(False)
        self.rit_spin.setToolTip("RIT / \u0394TX offset, \u00b19.999 kHz")
        self.rit_spin.valueChanged.connect(self._rit_changed)
        clear = QtWidgets.QPushButton("CLR")
        clear.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        clear.setToolTip("Clear the RIT / \u0394TX offset")
        clear.clicked.connect(lambda: self.rit_spin.setValue(0))
        row.addWidget(self.rit_spin)
        row.addWidget(clear)
        self._grid.addWidget(rit_box, n // columns, n % columns, 1, 2)
        self.sync()

    def _value(self, key: str):
        return (getattr(self._source, "state", {}) or {}).get(key)

    def _clicked(self, key: str) -> None:
        control = self._controls[key]
        value = self._value(key)
        if control.kind == "switch":
            new = 0 if value else 1
        else:
            codes = [code for _label, code in control.choices]
            # Not yet reported: take it as the first setting, so a click still moves it.
            now = codes.index(value) if value in codes else 0
            new = codes[(now + 1) % len(codes)]
        self._source.set_control(key, new)
        self.sync()

    def _rit_changed(self, hz: int) -> None:
        if self._syncing:
            return
        self._source.set_rit(hz)

    def open_level(self, key: str) -> None:
        self._popup(self._controls[key].levels, self.buttons[key])

    def open_multi(self) -> None:
        mode = (getattr(self._source, "mode", None) or "").lower()
        self._popup(MULTI_BY_MODE.get(mode, ("power",)), self.multi_button)

    def _popup(self, keys, button) -> None:
        levels = [(k, self._controls[k].label, self._value(k) or 0, self._controls[k].scale)
                  for k in keys if k in self._controls]
        if not levels:
            return
        popup = LevelPopup(levels, lambda k, raw: self._source.set_control(k, int(raw)), self)
        popup.move(button.mapToGlobal(QtCore.QPoint(0, button.height())))
        popup.show()
        self.popup = popup

    def sync(self) -> None:
        """Show the radio's current settings on the buttons."""
        rit = (getattr(self._source, "status", {}) or {}).get("rit_hz")
        if self.rit_spin is not None and rit is not None and not self.rit_spin.hasFocus() \
                and self.rit_spin.value() != rit:
            self._syncing = True
            self.rit_spin.setValue(int(rit))
            self._syncing = False
        for key, button in self.buttons.items():
            control = self._controls[key]
            value = self._value(key)
            if control.kind == "choice":
                setting = {code: text for text, code in control.choices}.get(value)
                if setting is None:
                    text = control.label
                elif control.key in ("dup", "tone", "notch") and \
                        value != control.choices[0][1]:
                    text = setting               # "DUP-", "TSQL", "AN": says it all
                else:
                    text = f"{control.label}\n{setting}"      # "AGC" over "FAST"
                # Lit when not off. AGC has no off: lit whenever the radio reports it.
                on = value is not None if control.key == "agc" \
                    else value not in (None, control.choices[0][1])
            else:
                text = control.label
                on = bool(value)
            if button.text() != text:
                button.setText(text)
            button.setStyleSheet(_ON if on else _OFF)
