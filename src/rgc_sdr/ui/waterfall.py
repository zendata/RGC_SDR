"""pyqtgraph waterfall view: a scrolling spectrogram image.

Wraps `dsp.waterfall.WaterfallBuffer`; all history and reduction logic lives there so it
stays testable without Qt.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6 import QtCore

from ..dsp.waterfall import WaterfallBuffer


# Rows of the array must map to y, columns to x. Without this, pyqtgraph's legacy default
# treats the first axis as x and the waterfall renders transposed.
pg.setConfigOption("imageAxisOrder", "row-major")

COLORMAPS = ("inferno", "magma", "plasma", "viridis", "turbo", "CET-L9", "CET-R4")


class WaterfallView(pg.PlotWidget):
    """Scrolling spectrogram, newest row at the top."""

    #: Emitted with a frequency in Hz when the user clicks to tune.
    frequencySelected = QtCore.pyqtSignal(float)

    def __init__(
        self,
        rows: int = 512,
        cols: int = 1024,
        colormap: str = "inferno",
        levels: tuple[float, float] = (-115.0, -40.0),
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.buffer = WaterfallBuffer(rows=rows, cols=cols)
        self._levels = levels
        self._history_s = 1.0
        self._rect: QtCore.QRectF | None = None

        self._img = pg.ImageItem(axisOrder="row-major")
        self.addItem(self._img)
        self.set_colormap(colormap)

        vb = self.getViewBox()
        vb.invertY(True)  # row 0 (newest) at the top
        vb.setMouseEnabled(x=True, y=False)
        vb.setDefaultPadding(0.0)
        self.setLabel("left", "age", units="s")
        self.setLabel("bottom", "frequency", units="Hz")
        self.showGrid(x=True, y=False, alpha=0.2)
        self.setMenuEnabled(False)
        self.scene().sigMouseClicked.connect(self._on_click)

    def _on_click(self, event) -> None:
        """Click-to-tune. pyqtgraph only raises this for a click without a drag, so it
        does not fight panning."""
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        self.frequencySelected.emit(float(vb.mapSceneToView(event.scenePos()).x()))
        event.accept()


    def set_colormap(self, name: str) -> None:
        self._img.setColorMap(pg.colormap.get(name))

    def set_levels(self, low: float, high: float) -> None:
        self._levels = (float(low), float(high))
        self._img.setLevels(self._levels)

    def set_geometry(
        self,
        center_hz: float,
        sample_rate: float,
        history_s: float,
        preserve_span: bool = False,
    ) -> None:
        """Place the image on the frequency/age axes.

        With `preserve_span`, an existing zoom is kept and simply recentred on the new
        tuning. Resetting to the full span on every retune threw away the zoom just as
        it became useful -- zoom in to inspect a crowded patch, click the station next
        door, and the view would snap back to the whole span.
        """
        self._history_s = float(history_s)
        f0 = center_hz - sample_rate / 2.0
        self._rect = QtCore.QRectF(f0, 0.0, float(sample_rate), self._history_s)
        self._apply_rect()

        view = self.getViewBox()
        lo, hi = f0, f0 + sample_rate
        if preserve_span:
            (was_lo, was_hi), _ = view.viewRange()
            width = was_hi - was_lo
            # Only when actually zoomed in; a full-span view has nothing to preserve.
            if 0.0 < width < sample_rate * 0.999:
                half = width / 2.0
                lo = min(max(center_hz - half, f0), f0 + sample_rate - width)
                hi = lo + width
        view.setRange(xRange=(lo, hi), yRange=(0.0, self._history_s), padding=0.0)

    def _apply_rect(self) -> None:
        """Map image pixels onto the frequency/age axes.

        `ImageItem.setRect` derives its scale from the image size *at call time*, so
        calling it while `image is None` silently scales as if the image were 1x1 -- the
        waterfall then renders far off-screen and the pane looks blank. Re-applying it
        after every `setImage` keeps the mapping correct across shape changes; building
        one QTransform per frame costs nothing.
        """
        if self._rect is not None and self._img.image is not None:
            self._img.setRect(self._rect)

    def clear_history(self) -> None:
        self.buffer.clear()

    def resize_bins(self, cols: int) -> None:
        self.buffer.resize_cols(cols)

    def push(self, row_dbfs) -> None:
        """Add one spectrum row (max-pooled to the buffer width by WaterfallBuffer)."""
        self.buffer.push(row_dbfs)
        self._img.setImage(
            self.buffer.image, autoLevels=False, levels=self._levels, autoDownsample=False
        )
        self._apply_rect()

    def auto_levels(self) -> tuple[float, float]:
        low, high = self.buffer.percentile_levels()
        self.set_levels(low, high)
        return (low, high)
