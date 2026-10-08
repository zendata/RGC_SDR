"""A recording at a glance: its whole length as a strip, clicked to seek (P12).

The picture is `device.playback.overview` -- a coarse spectrogram, time left to right
and frequency bottom to top -- coloured with the waterfall's own map, with a line where
playback has reached.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets


class RecordingOverview(QtWidgets.QLabel):
    """Emits `seekRequested` with a fraction 0..1 of the recording when clicked."""

    seekRequested = QtCore.pyqtSignal(float)

    HEIGHT = 44

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setMinimumWidth(200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setScaledContents(True)
        self.setToolTip("The whole recording, time left to right: click to go there")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._image: QtGui.QImage | None = None
        self.position = 0.0

    def set_picture(self, picture: np.ndarray, colormap: str = "inferno") -> None:
        """`picture` is [bins, columns] in dB, as `overview` returns it."""
        if picture.size == 0:
            self._image = None
            self.clear()
            return
        low, high = np.percentile(picture, [5, 99.5])
        scaled = np.clip((picture - low) / max(high - low, 1e-6), 0, 1)
        lut = pg.colormap.get(colormap).getLookupTable(nPts=256, alpha=False)
        rgb = lut[(scaled[::-1] * 255).astype(np.uint8)]          # high frequency on top
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        self._image = QtGui.QImage(rgb.data, w, h, 3 * w,
                                   QtGui.QImage.Format.Format_RGB888).copy()
        self._redraw()

    def set_position(self, fraction: float) -> None:
        self.position = min(1.0, max(0.0, float(fraction)))
        self._redraw()

    def _redraw(self) -> None:
        if self._image is None:
            return
        pixmap = QtGui.QPixmap.fromImage(self._image)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QPen(QtGui.QColor("#00e5ff"), 1))
        x = int(self.position * (pixmap.width() - 1))
        painter.drawLine(x, 0, x, pixmap.height())
        painter.end()
        self.setPixmap(pixmap)

    def mousePressEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.width() > 0:
            self.seekRequested.emit(min(1.0, max(0.0, event.position().x() / self.width())))
