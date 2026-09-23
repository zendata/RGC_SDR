"""Signal strength meter.

Reads in dBFS with an SNR estimate, deliberately **not** in S-units. S9 is defined as
-73 dBm at the antenna terminals, and converting to it needs the whole gain chain
calibrated -- antenna factor, feedline loss and receiver gain. The Airspy HF+ driver
reports no gain at all (PLANNING.md section 3), so any S-reading would be invented. dBFS
plus SNR is honest and, on an uncalibrated receiver, more useful.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets


class SMeter(QtWidgets.QWidget):
    """A horizontal bar with a decaying peak marker and a numeric readout."""

    def __init__(
        self,
        low_dbfs: float = -140.0,
        high_dbfs: float = -20.0,
        peak_decay_db: float = 0.4,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.low = float(low_dbfs)
        self.high = float(high_dbfs)
        self.peak_decay_db = float(peak_decay_db)
        self._level: float | None = None
        self._peak: float | None = None
        self._snr: float | None = None
        # Wide enough for the full readout: "-100.0 dBFS   S/N 12.3 dB" was being
        # clipped mid-word at the old width.
        self.setMinimumWidth(360)
        self.setFixedHeight(26)
        self.setToolTip("In-channel power (dBFS) and signal-to-noise estimate")

    @property
    def level_dbfs(self) -> float | None:
        return self._level

    @property
    def peak_dbfs(self) -> float | None:
        return self._peak

    def reset(self) -> None:
        self._level = None
        self._peak = None
        self._snr = None
        self.update()

    def set_level(self, dbfs: float | None, snr_db: float | None = None) -> None:
        if dbfs is None:
            self.reset()
            return
        self._level = float(dbfs)
        self._snr = None if snr_db is None else float(snr_db)
        if self._peak is None or self._level > self._peak:
            self._peak = self._level
        else:
            self._peak -= self.peak_decay_db
            self._peak = max(self._peak, self._level)
        self.update()

    def _fraction(self, dbfs: float) -> float:
        span = self.high - self.low
        if span <= 0:
            return 0.0
        return min(1.0, max(0.0, (dbfs - self.low) / span))

    def paintEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        track = QtCore.QRectF(0, 7, self.width() * 0.45, 12)
        radius = 3.0
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#1b1b21"))
        painter.drawRoundedRect(track, radius, radius)

        if self._level is not None:
            fraction = self._fraction(self._level)
            if fraction > 0.0:
                filled = QtCore.QRectF(track)
                filled.setWidth(track.width() * fraction)
                gradient = QtGui.QLinearGradient(track.left(), 0, track.right(), 0)
                gradient.setColorAt(0.0, QtGui.QColor("#2e7d32"))
                gradient.setColorAt(0.65, QtGui.QColor("#f9a825"))
                gradient.setColorAt(1.0, QtGui.QColor("#c62828"))
                painter.setBrush(QtGui.QBrush(gradient))
                painter.drawRoundedRect(filled, radius, radius)

        if self._peak is not None:
            x = track.left() + track.width() * self._fraction(self._peak)
            painter.setPen(QtGui.QPen(QtGui.QColor("#eceff1"), 2.0))
            painter.drawLine(QtCore.QPointF(x, track.top() - 1),
                             QtCore.QPointF(x, track.bottom() + 1))

        painter.setPen(QtGui.QPen(QtGui.QColor("#cfd8dc")))
        font = painter.font()
        font.setPointSizeF(max(9.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        if self._level is None:
            text = "--"
        else:
            text = f"{self._level:6.1f} dBFS"
            if self._snr is not None:
                text += f"   S/N {self._snr:4.1f} dB"
        painter.drawText(
            QtCore.QRectF(track.right() + 8, 0, self.width() - track.right() - 8, self.height()),
            int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft),
            text,
        )
        painter.end()
