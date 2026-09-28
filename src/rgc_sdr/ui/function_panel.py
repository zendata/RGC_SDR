"""The IC-705's FUNCTION screen as buttons on the Mac.

On/off keys light when on; multi-setting keys (P.AMP, AGC, BK-IN, TONE, SPLIT) step to
the next setting on each click and show it. A right-click -- or pressing and holding, as
a long touch on the radio -- opens the level behind a key (NB, NR, NOTCH, COMP, VOX,
MONI). The buttons follow the radio: a change made on its own screen shows here.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

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


class LevelPopup(QtWidgets.QFrame):
    """A slider for one level, shown under its function key."""

    def __init__(self, title: str, percent: int, on_change, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self.setStyleSheet("QFrame { background: #11161d; border: 1px solid #2c3645; }"
                           " QLabel { color: #d8dee9; }")
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(title))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(percent)
        self.slider.setFixedWidth(200)
        layout.addWidget(self.slider)
        self.value = QtWidgets.QLabel(f"{percent}%")
        self.value.setMinimumWidth(40)
        layout.addWidget(self.value)
        self.slider.valueChanged.connect(lambda v: self.value.setText(f"{v}%"))
        # Sent when let go, not on every step of a drag.
        self.slider.sliderReleased.connect(lambda: on_change(self.slider.value()))
        self.slider.valueChanged.connect(
            lambda v: None if self.slider.isSliderDown() else on_change(v))


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
        self.popup: LevelPopup | None = None

    def build(self, source, columns: int = 9) -> None:
        """Lay out a button per function control of `source`."""
        for button in self.buttons.values():
            button.deleteLater()
        self.buttons.clear()
        self._source = source
        controls = getattr(source, "controls", ()) or ()
        self._controls = {c.key: c for c in controls}
        functions = [c for c in controls if c.placement == "function"]
        for i, control in enumerate(functions):
            button = FunctionButton(control.label)
            tip = control.tooltip
            if control.level:
                tip += "\nRight-click or hold for its level"
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, k=control.key: self._clicked(k))
            if control.level:
                button.long_pressed.connect(lambda k=control.key: self.open_level(k))
            self._grid.addWidget(button, i // columns, i % columns)
            self.buttons[control.key] = button
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
            new = codes[(codes.index(value) + 1) % len(codes)] if value in codes else codes[0]
        self._source.set_control(key, new)
        self.sync()

    def open_level(self, key: str) -> None:
        level_key = self._controls[key].level
        level = self._controls.get(level_key)
        if level is None:
            return
        raw = self._value(level_key)
        percent = round((raw or 0) / 255 * 100)
        popup = LevelPopup(level.label, percent,
                           lambda pct, k=level_key: self._source.set_control(
                               k, round(pct / 100 * 255)), self)
        button = self.buttons[key]
        popup.move(button.mapToGlobal(QtCore.QPoint(0, button.height())))
        popup.show()
        self.popup = popup

    def sync(self) -> None:
        """Show the radio's current settings on the buttons."""
        for key, button in self.buttons.items():
            control = self._controls[key]
            value = self._value(key)
            if control.kind == "choice":
                setting = {code: text for text, code in control.choices}.get(value)
                if setting is None:
                    text = control.label
                elif control.key in ("split", "tone"):
                    text = setting               # "DUP-", "TSQL": the setting says it all
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
