"""Live spectrum curve with optional peak hold."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

from .gestures import (
    TUNE_DIRECTION,
    SwipeAccumulator,
    horizontal_dominates,
    wheel_deltas,
)

#: Native pan gestures report a small scalar rather than pixels, so they need their own
#: scaling to reach the same accumulator units.
PAN_TO_UNITS = 40.0


class SpectrumView(pg.PlotWidget):
    """Instantaneous spectrum, lightly smoothed, with a decaying peak-hold trace.

    Smoothing is applied here and *not* to the waterfall: the curve benefits from a
    steadier noise floor, while the waterfall must keep transients intact.
    """

    #: Emitted with a frequency in Hz when the user clicks to tune.
    frequencySelected = QtCore.pyqtSignal(float)
    #: Emitted with a signed number of tuning steps on a sideways swipe.
    frequencyNudged = QtCore.pyqtSignal(int)
    #: The passband's edges were dragged: (low Hz, high Hz), absolute (P10).
    passbandEdited = QtCore.pyqtSignal(float, float)
    #: Cmd-click (Control on other systems): add or remove a notch at this frequency.
    notchToggled = QtCore.pyqtSignal(float)

    def __init__(self, alpha: float = 0.3, peak_decay_db: float = 0.5, parent=None) -> None:
        super().__init__(parent=parent)
        self.alpha = float(alpha)
        self.peak_decay_db = float(peak_decay_db)
        self._smoothed: np.ndarray | None = None
        self._peak: np.ndarray | None = None

        self.setLabel("left", "power", units="dBFS")
        self.setLabel("bottom", "frequency", units="Hz")
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMenuEnabled(False)
        self.getViewBox().setMouseEnabled(x=True, y=False)
        self.getViewBox().setDefaultPadding(0.0)

        self._peak_curve = self.plot(
            pen=pg.mkPen("#ff8a65", width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        self._curve = self.plot(pen=pg.mkPen("#4fc3f7", width=1))
        self._center_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#9e9e9e", width=1, style=QtCore.Qt.PenStyle.DotLine)
        )
        self.addItem(self._center_line, ignoreBounds=True)
        self._peak_enabled = True
        self._swipe = SwipeAccumulator()
        # Shaded band showing what the demodulator is actually listening to, so the
        # offset and channel width are visible rather than abstract numbers.
        # Its edges drag (P10): the width and IF shift follow, as on a radio's
        # twin passband tuning.
        self._passband = pg.LinearRegionItem(
            values=(0.0, 0.0), movable=False,
            brush=pg.mkBrush(79, 195, 247, 40), pen=pg.mkPen(79, 195, 247, 90),
            hoverPen=pg.mkPen(79, 195, 247, 220, width=3),
        )
        self._passband.setZValue(-10)
        self._passband.setVisible(False)
        self._passband_editable = False
        for line in self._passband.lines:
            line.setMovable(False)
        self._passband.sigRegionChangeFinished.connect(self._on_passband_dragged)
        self.addItem(self._passband, ignoreBounds=True)
        self._setting_passband = False
        #: Manual notches drawn as dashed red lines.
        self._notch_lines: list[pg.InfiniteLine] = []
        self.scene().sigMouseClicked.connect(self._on_click)

    def _on_click(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        hz = float(vb.mapSceneToView(event.scenePos()).x())
        # Qt calls the Mac's Command key Control.
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.notchToggled.emit(hz)
        else:
            self.frequencySelected.emit(hz)
        event.accept()

    def _on_passband_dragged(self) -> None:
        if self._setting_passband or not self._passband_editable:
            return
        low, high = self._passband.getRegion()
        self.passbandEdited.emit(float(low), float(high))

    def set_passband_edges(self, low_hz: float, high_hz: float, editable: bool = True) -> None:
        """Shade what is heard, low..high in absolute Hz; `editable` lets its edges be
        dragged."""
        self._setting_passband = True
        try:
            self._passband.setRegion((float(low_hz), float(high_hz)))
        finally:
            self._setting_passband = False
        self._passband_editable = bool(editable)
        for line in self._passband.lines:
            line.setMovable(self._passband_editable)
        self._passband.setVisible(True)

    def set_notches(self, freqs_hz) -> None:
        """Draw manual notches at these absolute frequencies."""
        for line in self._notch_lines:
            self.removeItem(line)
        self._notch_lines = []
        for hz in freqs_hz:
            line = pg.InfiniteLine(pos=float(hz), angle=90, movable=False,
                                   pen=pg.mkPen("#ef5350", width=1,
                                                style=QtCore.Qt.PenStyle.DashLine))
            self.addItem(line, ignoreBounds=True)
            self._notch_lines.append(line)


    def wheelEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        """Tune on a sideways swipe or a shift-held vertical one; otherwise zoom.

        Shift plus vertical exists because macOS may never deliver a horizontal swipe at
        all: with "Swipe between pages" enabled the system claims the gesture and the
        application never sees it. A vertical swipe always arrives, so shift gives a
        route to fine tuning that cannot be intercepted.
        """
        dx, dy = wheel_deltas(event)
        shift = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if shift:
            delta = dy if abs(dy) >= abs(dx) else dx
        elif horizontal_dominates(dx, dy):
            delta = dx
        else:
            self._swipe.reset()
            super().wheelEvent(event)
            return
        steps = self._swipe.add(delta * TUNE_DIRECTION)
        if steps:
            self.frequencyNudged.emit(steps)
        event.accept()

    def event(self, ev):
        """Catch macOS trackpad pans, which do not arrive as wheel events.

        A sideways two-finger swipe can be delivered as a native pan gesture rather than
        a horizontal scroll, depending on the trackpad settings. Handling both is why
        this is here as well as `wheelEvent`.
        """
        if ev.type() == QtCore.QEvent.Type.NativeGesture:
            try:
                gesture = ev.gestureType()
                if gesture == QtCore.Qt.NativeGestureType.PanNativeGesture:
                    delta = ev.value()
                    steps = self._swipe.add(float(delta) * PAN_TO_UNITS * TUNE_DIRECTION)
                    if steps:
                        self.frequencyNudged.emit(steps)
                    ev.accept()
                    return True
            except (AttributeError, TypeError):
                pass
        return super().event(ev)

    def set_center_marker(self, hz: float) -> None:
        """Show where the receiver is actually tuned."""
        self._center_line.setPos(hz)

    def set_passband(self, center_hz: float, bandwidth_hz: float) -> None:
        """Shade the demodulator's channel. Zero bandwidth hides it."""
        if bandwidth_hz <= 0.0:
            self._passband.setVisible(False)
            return
        half = bandwidth_hz / 2.0
        self.set_passband_edges(center_hz - half, center_hz + half, editable=False)

    def clear_passband(self) -> None:
        self._passband.setVisible(False)

    def set_levels(self, low: float, high: float) -> None:
        self.setYRange(float(low), float(high), padding=0.02)

    def set_peak_hold(self, enabled: bool) -> None:
        self._peak_enabled = bool(enabled)
        self._peak_curve.setVisible(self._peak_enabled)
        if not enabled:
            self._peak = None

    def reset(self) -> None:
        self._smoothed = None
        self._peak = None

    def update_spectrum(self, freqs: np.ndarray, dbfs: np.ndarray) -> None:
        if self._smoothed is None or self._smoothed.shape != dbfs.shape:
            self._smoothed = dbfs.astype(np.float32).copy()
        else:
            a = self.alpha
            self._smoothed = (a * dbfs + (1.0 - a) * self._smoothed).astype(np.float32)
        self._curve.setData(freqs, self._smoothed)

        if not self._peak_enabled:
            return
        if self._peak is None or self._peak.shape != dbfs.shape:
            self._peak = dbfs.astype(np.float32).copy()
        else:
            self._peak = np.maximum(self._peak - self.peak_decay_db, dbfs).astype(np.float32)
        self._peak_curve.setData(freqs, self._peak)
