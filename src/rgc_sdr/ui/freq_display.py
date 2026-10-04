"""The SDR's frequency readout: a large spin box whose digits tune by dragging.

Press on any digit and drag up or down to step that digit, as on a radio's VFO knob
with a selectable step. Typing a frequency still works; a click without a drag puts the
text cursor there.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

#: Vertical pixels of drag per step of the grabbed digit.
PIXELS_PER_STEP = 8
FONT_POINTS = 26
COLOUR = "#ffd600"


def digit_place_mhz(text: str, index: int) -> float | None:
    """The value in MHz of one step of the digit at `index` in `text`, e.g. 0.001 for
    the kHz digit of "144.775000 MHz". None when that character is not a digit of the
    number."""
    if not 0 <= index < len(text) or not text[index].isdigit():
        return None
    end = index
    while end < len(text) and (text[end].isdigit() or text[end] == "."):
        end += 1
    start = index
    while start > 0 and (text[start - 1].isdigit() or text[start - 1] == "."):
        start -= 1
    number = text[start:end]
    dot = number.find(".")
    if dot < 0:
        dot = len(number)
    at = index - start
    exponent = dot - at - 1 if at < dot else dot - at
    return 10.0 ** exponent


class FrequencyDisplay(QtWidgets.QDoubleSpinBox):
    """Frequency in MHz. Drag a digit vertically to tune by that digit's place value.

    A drag emits `digitDragged` rather than `valueChanged`: it is an exact frequency the
    user dialled digit by digit, so the owner should tune to it without snapping it back
    onto the step grid -- otherwise dragging the kHz digit with a 25 kHz step would do
    nothing at all.
    """

    #: The new frequency in MHz, once per step of a digit drag.
    digitDragged = QtCore.pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        font = QtGui.QFont("Menlo")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        font.setPointSize(FONT_POINTS)
        font.setBold(True)
        self.setFont(font)
        self.setStyleSheet(f"QDoubleSpinBox {{ color: {COLOUR}; background: #0b0f14; }}")
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setToolTip("Drag a digit up or down to tune it; or type a frequency")
        self._drag: tuple[float, float, float] | None = None  # (place, start value, start y)
        self._dragged = False
        self._press_index = 0
        self.lineEdit().installEventFilter(self)
        self.lineEdit().setMouseTracking(True)

    def char_index_at(self, x: float) -> int:
        """The character under horizontal position `x` in the line edit, or -1.

        Measured from the font rather than `cursorPositionAt`, which gives the nearest
        gap between characters and so is wrong for the right half of each digit.
        """
        edit = self.lineEdit()
        metrics = QtGui.QFontMetricsF(edit.font())
        # QLineEdit's text starts after its contents margin, text margin and a fixed
        # 2-pixel inner margin; nothing here scrolls, as the box fits its text.
        left = edit.contentsRect().left() + edit.textMargins().left() + 2
        text = edit.text()
        offset = x - left
        if offset < 0:
            return -1
        for i in range(len(text)):
            if offset < metrics.horizontalAdvance(text[: i + 1]):
                return i
        return -1

    def _place_at(self, x: float) -> float | None:
        return digit_place_mhz(self.lineEdit().text(), self.char_index_at(x))

    def eventFilter(self, obj, event) -> bool:  # noqa: N802  (Qt naming)
        if obj is not self.lineEdit():
            return super().eventFilter(obj, event)
        kind = event.type()
        if kind == QtCore.QEvent.Type.MouseButtonPress and \
                event.button() == QtCore.Qt.MouseButton.LeftButton:
            place = self._place_at(event.position().x())
            if place is None:
                return False
            self._drag = (place, self.value(), event.position().y())
            self._dragged = False
            self._press_index = self.lineEdit().cursorPositionAt(event.position().toPoint())
            return True
        if kind == QtCore.QEvent.Type.MouseMove:
            if self._drag is None:
                over_digit = self._place_at(event.position().x()) is not None
                self.lineEdit().setCursor(QtCore.Qt.CursorShape.SizeVerCursor if over_digit
                                          else QtCore.Qt.CursorShape.IBeamCursor)
                return False
            place, start, y0 = self._drag
            steps = int((y0 - event.position().y()) / PIXELS_PER_STEP)   # up is more
            if steps:
                self._dragged = True
            self._set_dragged(start + steps * place)
            return True
        if kind == QtCore.QEvent.Type.MouseButtonRelease and self._drag is not None:
            self._drag = None
            if not self._dragged:
                # A plain click: behave like a text box so the frequency can be typed.
                self.lineEdit().setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                self.lineEdit().setCursorPosition(self._press_index)
            return True
        return super().eventFilter(obj, event)

    def _set_dragged(self, mhz: float) -> None:
        # Rounded to the displayed precision, so float error never leaves a stray 1 Hz.
        mhz = round(min(max(mhz, self.minimum()), self.maximum()), self.decimals())
        if mhz == round(self.value(), self.decimals()):
            return
        self.blockSignals(True)
        self.setValue(mhz)
        self.blockSignals(False)
        self.digitDragged.emit(mhz)
