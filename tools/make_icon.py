"""Generate the macOS app icon: a miniature of the app's own spectrum + waterfall.

    python tools/make_icon.py            # writes assets/icon.png
    ./tools/make_icns.sh                 # turns that into assets/AppIcon.icns

Uses the same `inferno` colour map the waterfall uses, so the icon matches the app.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui

SIZE = 1024
OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "icon.png"


def synth_waterfall(rows: int, cols: int, seed: int = 7) -> np.ndarray:
    """A plausible-looking spectrogram: noise floor, band roll-off, a few carriers."""
    rng = np.random.default_rng(seed)
    wf = rng.normal(-112.0, 2.0, size=(rows, cols))
    # Receiver passband roll-off at both edges.
    x = np.linspace(-1.0, 1.0, cols)
    wf += -22.0 * np.clip(np.abs(x) - 0.82, 0.0, None) / 0.18
    # Steady carriers, plus one that fades in partway down.
    for pos, strength in ((0.18, 26), (0.34, 18), (0.52, 30), (0.71, 14), (0.86, 22)):
        col = int(pos * cols)
        wf[:, col] += strength
        wf[:, col - 1] += strength * 0.35
        wf[:, col + 1] += strength * 0.35
    fade = int(rows * 0.45)
    col = int(0.62 * cols)
    wf[fade:, col] += 24
    wf[fade:, col + 1] += 9
    return wf


def to_rgb(values: np.ndarray, low: float, high: float) -> np.ndarray:
    lut = pg.colormap.get("inferno").getLookupTable(nPts=256, alpha=False)
    idx = np.clip((values - low) / (high - low), 0.0, 1.0)
    return lut[(idx * 255).astype(np.uint8)]


def main() -> int:
    from PyQt6 import QtWidgets

    QtWidgets.QApplication(sys.argv)  # QImage/QPainter need an application object

    rows, cols = 320, 512
    wf = synth_waterfall(rows, cols)
    rgb = np.ascontiguousarray(to_rgb(wf, -119.0, -88.0))
    img = QtGui.QImage(rgb.tobytes(), cols, rows, 3 * cols, QtGui.QImage.Format.Format_RGB888)

    canvas = QtGui.QImage(SIZE, SIZE, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(canvas)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)

    # macOS icons sit inside a rounded square with a margin.
    margin = SIZE * 0.085
    body = QtCore.QRectF(margin, margin, SIZE - 2 * margin, SIZE - 2 * margin)
    radius = body.width() * 0.22
    path = QtGui.QPainterPath()
    path.addRoundedRect(body, radius, radius)
    p.setClipPath(path)
    p.fillPath(path, QtGui.QColor("#0d0d11"))

    # Waterfall fills the lower part.
    wf_rect = QtCore.QRectF(body.left(), body.top() + body.height() * 0.42,
                            body.width(), body.height() * 0.58)
    p.drawImage(wf_rect, img)

    # Spectrum trace above it, from the newest waterfall row.
    row = wf[0]
    lo, hi = -122.0, -80.0
    norm = np.clip((row - lo) / (hi - lo), 0.0, 1.0)
    trace = QtCore.QRectF(body.left(), body.top() + body.height() * 0.06,
                          body.width(), body.height() * 0.34)
    poly = QtGui.QPolygonF([
        QtCore.QPointF(trace.left() + trace.width() * i / (cols - 1),
                       trace.bottom() - trace.height() * float(v))
        for i, v in enumerate(norm)
    ])
    p.setPen(QtGui.QPen(QtGui.QColor("#4fc3f7"), SIZE * 0.012,
                        QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
    p.drawPolyline(poly)

    # Keep the clip on so only the inner half of the stroke shows: a faint inner edge
    # rather than a heavy ring straddling the boundary.
    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 22), SIZE * 0.008))
    p.drawPath(path)
    p.end()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not canvas.save(str(OUT)):
        print(f"failed to write {OUT}", file=sys.stderr)
        return 1
    print(f"wrote {OUT} ({SIZE}x{SIZE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
