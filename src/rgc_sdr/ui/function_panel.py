"""The IC-705's FUNCTION screen as buttons on the Mac.

On/off keys light when on; multi-setting keys (P.AMP, AGC, BK-IN, TONE, SPLIT) step to
the next setting on each click and show it. A right-click -- or pressing and holding, as
a long touch on the radio -- opens the levels behind a key, as the radio's function menu
does (NB, NR, NOTCH, VOX, COMP, MONI, BKIN). The buttons follow the radio: a change made
on its own screen shows here.
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
    """Sliders for a key's levels, shown under it. `levels` is ((key, title, percent), ...);
    `on_change(key, percent)` is called when a slider is let go or stepped."""

    def __init__(self, levels, on_change, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self.setStyleSheet("QFrame { background: #11161d; border: 1px solid #2c3645; }"
                           " QLabel { color: #d8dee9; }")
        grid = QtWidgets.QGridLayout(self)
        self.sliders: dict[str, QtWidgets.QSlider] = {}
        for row, (key, title, percent) in enumerate(levels):
            grid.addWidget(QtWidgets.QLabel(title), row, 0)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(percent)
            slider.setFixedWidth(200)
            grid.addWidget(slider, row, 1)
            value = QtWidgets.QLabel(f"{percent}%")
            value.setMinimumWidth(40)
            grid.addWidget(value, row, 2)
            slider.valueChanged.connect(lambda v, lab=value: lab.setText(f"{v}%"))
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
            if control.levels:
                tip += "\nRight-click or hold for its levels"
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, k=control.key: self._clicked(k))
            if control.levels:
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
            # Not yet reported: take it as the first setting, so a click still moves it.
            now = codes.index(value) if value in codes else 0
            new = codes[(now + 1) % len(codes)]
        self._source.set_control(key, new)
        self.sync()

    def open_level(self, key: str) -> None:
        levels = [(k, self._controls[k].label, round((self._value(k) or 0) / 255 * 100))
                  for k in self._controls[key].levels if k in self._controls]
        if not levels:
            return
        popup = LevelPopup(levels, lambda k, pct: self._source.set_control(
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
