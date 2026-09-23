"""Live spectrum curve with optional peak hold."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore


class SpectrumView(pg.PlotWidget):
    """Instantaneous spectrum, lightly smoothed, with a decaying peak-hold trace.

    Smoothing is applied here and *not* to the waterfall: the curve benefits from a
    steadier noise floor, while the waterfall must keep transients intact.
    """

    #: Emitted with a frequency in Hz when the user clicks to tune.
    frequencySelected = QtCore.pyqtSignal(float)

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
        # Shaded band showing what the demodulator is actually listening to, so the
        # offset and channel width are visible rather than abstract numbers.
        self._passband = pg.LinearRegionItem(
            values=(0.0, 0.0), movable=False,
            brush=pg.mkBrush(79, 195, 247, 40), pen=pg.mkPen(79, 195, 247, 90),
        )
        self._passband.setZValue(-10)
        self._passband.setVisible(False)
        self.addItem(self._passband, ignoreBounds=True)
        self.scene().sigMouseClicked.connect(self._on_click)

    def _on_click(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        self.frequencySelected.emit(float(vb.mapSceneToView(event.scenePos()).x()))
        event.accept()

    def set_center_marker(self, hz: float) -> None:
        """Show where the receiver is actually tuned."""
        self._center_line.setPos(hz)

    def set_passband(self, center_hz: float, bandwidth_hz: float) -> None:
        """Shade the demodulator's channel. Zero bandwidth hides it."""
        if bandwidth_hz <= 0.0:
            self._passband.setVisible(False)
            return
        half = bandwidth_hz / 2.0
        self._passband.setRegion((center_hz - half, center_hz + half))
        self._passband.setVisible(True)

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
