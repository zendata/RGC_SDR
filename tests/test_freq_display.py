"""The frequency display: digit place values and drag-to-tune."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from src.rgc_sdr.ui.freq_display import PIXELS_PER_STEP, FrequencyDisplay, digit_place_mhz  # noqa: E402


def test_digit_place_values():
    text = "144.775000 MHz"
    assert digit_place_mhz(text, 0) == pytest.approx(100.0)
    assert digit_place_mhz(text, 2) == pytest.approx(1.0)
    assert digit_place_mhz(text, 4) == pytest.approx(0.1)
    assert digit_place_mhz(text, 6) == pytest.approx(0.001)     # kHz
    assert digit_place_mhz(text, 9) == pytest.approx(1e-6)      # Hz
    assert digit_place_mhz(text, 3) is None                     # the point
    assert digit_place_mhz(text, 11) is None                    # the units
    assert digit_place_mhz(text, -1) is None
    assert digit_place_mhz("7.100000 MHz", 0) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _mouse(widget, kind, x, y, buttons):
    pos = QtCore.QPointF(x, y)
    event = QtGui.QMouseEvent(kind, pos, widget.mapToGlobal(pos), QtCore.Qt.MouseButton.LeftButton,
                              buttons, QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, event)


def _drag(display, index, dy):
    """Press on character `index`, drag vertically by `dy` pixels (negative is up)."""
    edit = display.lineEdit()
    metrics = QtGui.QFontMetricsF(edit.font())
    text = edit.text()
    left = edit.contentsRect().left() + edit.textMargins().left() + 2
    x = left + (metrics.horizontalAdvance(text[:index]) + metrics.horizontalAdvance(text[: index + 1])) / 2
    y = edit.height() / 2
    held = QtCore.Qt.MouseButton.LeftButton
    _mouse(edit, QtCore.QEvent.Type.MouseButtonPress, x, y, held)
    for step in range(1, 11):
        _mouse(edit, QtCore.QEvent.Type.MouseMove, x, y + dy * step / 10, held)
    _mouse(edit, QtCore.QEvent.Type.MouseButtonRelease, x, y + dy, QtCore.Qt.MouseButton.NoButton)


def _display():
    display = FrequencyDisplay()
    display.setDecimals(6)
    display.setSuffix(" MHz")
    display.setRange(0.009, 260.0)
    display.setValue(144.775)
    display.resize(display.sizeHint())
    display.show()
    return display


def test_dragging_a_digit_up_steps_it(qapp):
    display = _display()
    seen = []
    changed = []
    display.digitDragged.connect(seen.append)
    display.valueChanged.connect(changed.append)
    _drag(display, 6, -3 * PIXELS_PER_STEP)           # kHz digit, three steps up
    assert display.value() == pytest.approx(144.778)
    assert seen[-1] == pytest.approx(144.778)
    assert len(seen) == 3                             # one emission per step
    assert changed == []                              # not the snapping path


def test_dragging_down_and_on_the_megahertz_digit(qapp):
    display = _display()
    _drag(display, 2, 2 * PIXELS_PER_STEP)            # 1 MHz digit, two steps down
    assert display.value() == pytest.approx(142.775)


def test_drag_is_clamped_to_the_range(qapp):
    display = _display()
    _drag(display, 0, -5 * PIXELS_PER_STEP)           # +500 MHz would pass the top
    assert display.value() == pytest.approx(260.0)


def test_a_click_without_drag_tunes_nothing(qapp):
    display = _display()
    seen = []
    display.digitDragged.connect(seen.append)
    _drag(display, 6, 0)
    assert seen == [] and display.value() == pytest.approx(144.775)
