"""Live spectrum curve with optional peak hold."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui

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

    #: Markers further apart than this on screen are separate; a click nearer one
    #: removes it (P11).
    MARKER_CLICK_PIXELS = 8.0
    MARKER_COLOURS = ("#ffee58", "#80deea", "#ffab91", "#ce93d8", "#a5d6a7", "#f48fb1")

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
        #: Markers (P11): [frequency Hz, line, label], the first the delta reference.
        self._markers: list[list] = []
        self._freqs: np.ndarray | None = None
        #: Overlays (P11): the band plan along the bottom, memory names along the top.
        self._band_items: list = []
        self._memory_items: list = []
        self._show_bands = False
        self._memories: list[tuple[float, str]] = []
        self.getViewBox().sigXRangeChanged.connect(self._refresh_overlays)
        self.scene().sigMouseClicked.connect(self._on_click)

    def _on_click(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        hz = float(vb.mapSceneToView(event.scenePos()).x())
        modifiers = event.modifiers()
        # Qt calls the Mac's Command key Control, and Option Alt.
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.notchToggled.emit(hz)
        elif modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
            self.toggle_marker(hz)
        else:
            self.frequencySelected.emit(hz)
        event.accept()

    # -- markers (P11) ---------------------------------------------------------------

    @property
    def markers(self) -> list[float]:
        return [m[0] for m in self._markers]

    def _hz_per_pixel(self) -> float:
        (lo, hi), _ = self.getViewBox().viewRange()
        return (hi - lo) / max(1.0, self.getViewBox().width())

    def toggle_marker(self, hz: float) -> None:
        """Remove a marker near `hz`, else add one on the strongest bin nearby."""
        near = self.MARKER_CLICK_PIXELS * self._hz_per_pixel()
        for marker in self._markers:
            if abs(marker[0] - hz) <= near:
                self._remove_marker(marker)
                self._relabel()
                return
        if self._freqs is not None and self._smoothed is not None:
            from ..dsp.measure import peak_near

            hz = peak_near(self._freqs, self._smoothed, hz)
        self.add_marker(hz)

    def add_marker(self, hz: float) -> None:
        colour = self.MARKER_COLOURS[len(self._markers) % len(self.MARKER_COLOURS)]
        line = pg.InfiniteLine(pos=float(hz), angle=90, movable=False,
                               pen=pg.mkPen(colour, width=1))
        label = pg.TextItem(color=colour, anchor=(0, 1))
        self.addItem(line, ignoreBounds=True)
        self.addItem(label, ignoreBounds=True)
        self._markers.append([float(hz), line, label])
        self._relabel()

    def _remove_marker(self, marker) -> None:
        self.removeItem(marker[1])
        self.removeItem(marker[2])
        self._markers.remove(marker)

    def clear_markers(self) -> None:
        for marker in list(self._markers):
            self._remove_marker(marker)

    def peak_marker(self) -> float | None:
        """A marker on the strongest signal in view; its frequency, or None."""
        if self._freqs is None or self._smoothed is None:
            return None
        from ..dsp.measure import strongest

        (lo, hi), _ = self.getViewBox().viewRange()
        hz = strongest(self._freqs, self._smoothed, lo, hi)
        if hz is not None:
            self.add_marker(hz)
        return hz

    def marker_text(self, index: int) -> str:
        """M1 145.0000 MHz -62.1 dBFS; later ones with their difference from M1."""
        from ..dsp.measure import level_at

        hz = self._markers[index][0]
        if self._freqs is None or self._smoothed is None:
            return f"M{index + 1} {hz / 1e6:.4f} MHz"
        level = level_at(self._freqs, self._smoothed, hz)
        text = f"M{index + 1} {hz / 1e6:.4f} MHz {level:.1f} dBFS"
        if index:
            ref = self._markers[0][0]
            delta = level - level_at(self._freqs, self._smoothed, ref)
            text += f"\n\u0394 {(hz - ref) / 1e3:+.3f} kHz {delta:+.1f} dB"
        return text

    def _relabel(self) -> None:
        if self._smoothed is None or self._freqs is None:
            return
        from ..dsp.measure import level_at

        for i, (hz, _line, label) in enumerate(self._markers):
            label.setText(self.marker_text(i))
            label.setPos(hz, level_at(self._freqs, self._smoothed, hz))

    # -- overlays (P11) ----------------------------------------------------------------

    def set_band_plan(self, on: bool) -> None:
        self._show_bands = bool(on)
        self._refresh_overlays()

    def set_memories(self, memories) -> None:
        """(frequency Hz, name) pairs to name along the top; empty for none."""
        self._memories = [(float(f), str(n)) for f, n in memories]
        self._refresh_overlays()

    def _refresh_overlays(self, *_args) -> None:
        from ..bandplan import KIND_COLOURS, bands_in

        for item in self._band_items + self._memory_items:
            self.removeItem(item)
        self._band_items, self._memory_items = [], []
        (lo, hi), (bottom, top) = self.getViewBox().viewRange()
        hz_per_px = self._hz_per_pixel()
        if self._show_bands:
            for band in bands_in(lo, hi):
                colour = QtGui.QColor(KIND_COLOURS.get(band.kind, "#9e9e9e"))
                fill = QtGui.QColor(colour)
                fill.setAlpha(120)
                region = pg.LinearRegionItem(values=(band.low_hz, band.high_hz),
                                             movable=False, span=(0.0, 0.035),
                                             brush=pg.mkBrush(fill), pen=pg.mkPen(None))
                region.setZValue(-20)
                region.setToolTip(f"{band.name}: {band.low_hz / 1e6:g}-{band.high_hz / 1e6:g} MHz")
                self.addItem(region, ignoreBounds=True)
                self._band_items.append(region)
                visible = min(hi, band.high_hz) - max(lo, band.low_hz)
                if visible / hz_per_px > 8 * len(band.name):         # room for the name
                    label = pg.TextItem(band.name, color=colour, anchor=(0, 1))
                    label.setPos(max(lo, band.low_hz), bottom + 0.035 * (top - bottom))
                    self.addItem(label, ignoreBounds=True)
                    self._band_items.append(label)
        shown = [(f, n) for f, n in self._memories if lo <= f <= hi]
        if len(shown) <= 40:                 # a crowd of names is no help
            last_x = None
            for hz, name in sorted(shown):
                tick = pg.InfiniteLine(pos=hz, angle=90, movable=False, span=(0.93, 1.0),
                                       pen=pg.mkPen("#ce93d8", width=2))
                self.addItem(tick, ignoreBounds=True)
                self._memory_items.append(tick)
                if last_x is None or (hz - last_x) / hz_per_px > 7 * len(name):
                    label = pg.TextItem(name, color="#ce93d8", anchor=(0, 0))
                    label.setPos(hz, top)
                    self.addItem(label, ignoreBounds=True)
                    self._memory_items.append(label)
                    last_x = hz

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
        if self._band_items or self._memory_items:
            self._refresh_overlays()        # the labels sit at the top and bottom

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
        self._freqs = freqs
        if self._markers:
            self._relabel()

        if not self._peak_enabled:
            return
        if self._peak is None or self._peak.shape != dbfs.shape:
            self._peak = dbfs.astype(np.float32).copy()
        else:
            self._peak = np.maximum(self._peak - self.peak_decay_db, dbfs).astype(np.float32)
        self._peak_curve.setData(freqs, self._peak)
