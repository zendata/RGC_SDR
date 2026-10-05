"""A map of what the decoders have located: ships now, aircraft later.

OpenStreetMap tiles, drawn by a small slippy-map widget rather than an embedded browser,
so no web engine is needed. OSM's tile usage policy is kept: an identifying User-Agent,
tiles cached on disk for a week, at most two downloads at a time, and the copyright
notice on the map. Setting RGC_SDR_NO_TILES (the tests do) keeps it off the network;
the targets are still drawn, on a plain grid.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtNetwork, QtWidgets

from .. import __version__ as APP_VERSION
from ..targets import Target, TargetStore

TILE = 256
MIN_ZOOM, MAX_ZOOM = 2, 18
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = f"RGC_SDR/{APP_VERSION} (+https://github.com/zendata/RGC_SDR)"
CACHE_DIR = Path.home() / "Library" / "Caches" / "RGC_SDR" / "tiles"
CACHE_DAYS = 7
MAX_DOWNLOADS = 2
ATTRIBUTION = "© OpenStreetMap contributors"
#: Port Phillip, until something has been heard.
HOME = (-37.95, 144.85)
HOME_ZOOM = 10

COLOURS = {"ship": QtGui.QColor("#e53935"), "aid": QtGui.QColor("#fdd835"),
           "base": QtGui.QColor("#1e88e5"), "aircraft": QtGui.QColor("#8e24aa")}


# -- Web Mercator ------------------------------------------------------------------


def world_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Pixel position on the whole-world map at `zoom`."""
    scale = TILE * (1 << zoom)
    lat = max(min(lat, 85.0511), -85.0511)
    x = (lon + 180.0) / 360.0 * scale
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * scale
    return x, y


def lat_lon(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = TILE * (1 << zoom)
    lon = x / scale * 360.0 - 180.0
    n = math.pi - 2 * math.pi * y / scale
    return math.degrees(math.atan(math.sinh(n))), lon


# -- tiles ---------------------------------------------------------------------------


class TileSource(QtCore.QObject):
    """OSM tiles from memory, then disk, then the network, politely."""

    tileReady = QtCore.pyqtSignal()

    def __init__(self, parent=None, cache_dir: Path = CACHE_DIR) -> None:
        super().__init__(parent)
        self.offline = bool(os.environ.get("RGC_SDR_NO_TILES"))
        self.cache_dir = cache_dir
        self._memory: dict[tuple[int, int, int], QtGui.QPixmap] = {}
        self._queue: list[tuple[int, int, int]] = []
        self._pending: set[tuple[int, int, int]] = set()
        self._in_flight = 0
        self._failed: dict[tuple[int, int, int], float] = {}
        self._net = None if self.offline else QtNetwork.QNetworkAccessManager(self)

    def _path(self, key) -> Path:
        z, x, y = key
        return self.cache_dir / str(z) / str(x) / f"{y}.png"

    def tile(self, z: int, x: int, y: int) -> QtGui.QPixmap | None:
        key = (z, x, y)
        pixmap = self._memory.get(key)
        if pixmap is not None:
            return pixmap
        path = self._path(key)
        fresh = path.exists() and time.time() - path.stat().st_mtime < CACHE_DAYS * 86400
        if path.exists():
            pixmap = QtGui.QPixmap(str(path))
            if not pixmap.isNull():
                self._memory[key] = pixmap
                if fresh or self.offline:
                    return pixmap
        if not self.offline and key not in self._pending and \
                time.time() - self._failed.get(key, 0) > 60:
            self._pending.add(key)
            self._queue.append(key)
            self._pump()
        return pixmap

    def forget_queue(self) -> None:
        """Drop tiles asked for but not yet requested: the view has moved on."""
        for key in self._queue:
            self._pending.discard(key)
        self._queue.clear()

    def _pump(self) -> None:
        while self._queue and self._in_flight < MAX_DOWNLOADS:
            key = self._queue.pop()               # newest first: what is on screen now
            z, x, y = key
            request = QtNetwork.QNetworkRequest(QtCore.QUrl(TILE_URL.format(z=z, x=x, y=y)))
            request.setHeader(QtNetwork.QNetworkRequest.KnownHeaders.UserAgentHeader,
                              USER_AGENT)
            reply = self._net.get(request)
            self._in_flight += 1
            reply.finished.connect(lambda r=reply, k=key: self._done(r, k))

    def _done(self, reply, key) -> None:
        self._in_flight -= 1
        self._pending.discard(key)
        data = bytes(reply.readAll())
        ok = reply.error() == QtNetwork.QNetworkReply.NetworkError.NoError
        reply.deleteLater()
        pixmap = QtGui.QPixmap()
        if ok and pixmap.loadFromData(data):
            self._memory[key] = pixmap
            path = self._path(key)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            except OSError:
                pass
            self.tileReady.emit()
        else:
            self._failed[key] = time.time()
        self._pump()


# -- the map -------------------------------------------------------------------------


class MapView(QtWidgets.QWidget):
    """Tiles, targets, trails; drag to pan, scroll to zoom about the pointer."""

    targetClicked = QtCore.pyqtSignal(str)

    def __init__(self, store: TargetStore, tiles: TileSource | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.tiles = tiles or TileSource(self)
        self.tiles.tileReady.connect(self.update)
        self.zoom = HOME_ZOOM
        self.centre = HOME
        self.selected: str | None = None
        self._drag_from: QtCore.QPointF | None = None
        self._drag_centre = None
        self._wheel = 0.0
        #: Set once the user pans or zooms; until then the window keeps everything in view.
        self.user_moved = False
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

    # -- coordinates
    def _origin(self) -> tuple[float, float]:
        cx, cy = world_xy(*self.centre, self.zoom)
        return cx - self.width() / 2, cy - self.height() / 2

    def to_screen(self, lat: float, lon: float) -> QtCore.QPointF:
        ox, oy = self._origin()
        x, y = world_xy(lat, lon, self.zoom)
        return QtCore.QPointF(x - ox, y - oy)

    def to_lat_lon(self, point: QtCore.QPointF) -> tuple[float, float]:
        ox, oy = self._origin()
        return lat_lon(point.x() + ox, point.y() + oy, self.zoom)

    def set_zoom(self, zoom: int, about: QtCore.QPointF | None = None) -> None:
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, int(zoom)))
        if zoom == self.zoom:
            return
        about = about or QtCore.QPointF(self.width() / 2, self.height() / 2)
        fixed = self.to_lat_lon(about)
        self.zoom = zoom
        # Keep the point under the pointer where it was.
        p = self.to_screen(*fixed)
        ox, oy = self._origin()
        self.centre = lat_lon(ox + self.width() / 2 + p.x() - about.x(),
                              oy + self.height() / 2 + p.y() - about.y(), zoom)
        self.tiles.forget_queue()
        self.update()

    def centre_on(self, lat: float, lon: float) -> None:
        self.centre = (lat, lon)
        self.update()

    def fit(self, targets: list[Target]) -> None:
        """Zoom to show all of `targets`."""
        if not targets:
            return
        lats = [t.lat for t in targets]
        lons = [t.lon for t in targets]
        self.centre = ((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)
        for zoom in range(MAX_ZOOM - 4, MIN_ZOOM - 1, -1):
            x0, y0 = world_xy(max(lats), min(lons), zoom)
            x1, y1 = world_xy(min(lats), max(lons), zoom)
            if x1 - x0 < self.width() * 0.85 and y1 - y0 < self.height() * 0.85:
                break
        self.zoom = zoom
        self.update()

    def target_at(self, point: QtCore.QPointF, radius: float = 10.0) -> Target | None:
        best, best_d = None, radius
        for t in self.store.placed():
            p = self.to_screen(t.lat, t.lon)
            d = math.hypot(p.x() - point.x(), p.y() - point.y())
            if d < best_d:
                best, best_d = t, d
        return best

    # -- painting
    def paintEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#cfd8dc"))
        self._paint_tiles(painter)
        self._paint_targets(painter)
        painter.setPen(QtGui.QColor("#37474f"))
        painter.drawText(self.rect().adjusted(4, 0, -4, -3),
                         QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom,
                         ATTRIBUTION)
        painter.end()

    def _paint_tiles(self, painter: QtGui.QPainter) -> None:
        ox, oy = self._origin()
        n = 1 << self.zoom
        grid = QtGui.QPen(QtGui.QColor("#b0bec5"))
        for ty in range(int(oy // TILE), int((oy + self.height()) // TILE) + 1):
            if not 0 <= ty < n:
                continue
            for tx in range(int(ox // TILE), int((ox + self.width()) // TILE) + 1):
                rect = QtCore.QRectF(tx * TILE - ox, ty * TILE - oy, TILE, TILE)
                pixmap = self.tiles.tile(self.zoom, tx % n, ty)
                if pixmap is not None:
                    painter.drawPixmap(rect.toRect(), pixmap)
                else:
                    painter.setPen(grid)
                    painter.drawRect(rect)

    def _paint_targets(self, painter: QtGui.QPainter) -> None:
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        metrics = QtGui.QFontMetricsF(font)
        labels: list[QtCore.QRectF] = []
        # The selected target last, so its label always wins a crowded spot.
        for t in sorted(self.store.placed(), key=lambda t: t.key == self.selected):
            colour = COLOURS.get(t.kind, QtGui.QColor("black"))
            if len(t.trail) > 1:
                painter.setPen(QtGui.QPen(colour.darker(130), 1.5))
                path = QtGui.QPolygonF([self.to_screen(*p) for p in t.trail])
                painter.drawPolyline(path)
            p = self.to_screen(t.lat, t.lon)
            painter.save()
            painter.translate(p)
            painter.setPen(QtGui.QPen(QtGui.QColor("black"), 1))
            painter.setBrush(colour)
            if t.kind in ("ship", "aircraft") and t.bearing is not None:
                painter.rotate(t.bearing)
                painter.drawPolygon(QtGui.QPolygonF([
                    QtCore.QPointF(0, -9), QtCore.QPointF(5, 6), QtCore.QPointF(0, 3),
                    QtCore.QPointF(-5, 6)]))
            elif t.kind == "aid":
                painter.drawPolygon(QtGui.QPolygonF([
                    QtCore.QPointF(0, -6), QtCore.QPointF(6, 0), QtCore.QPointF(0, 6),
                    QtCore.QPointF(-6, 0)]))
            elif t.kind == "base":
                painter.drawRect(QtCore.QRectF(-5, -5, 10, 10))
            else:
                painter.drawEllipse(QtCore.QPointF(0, 0), 5, 5)
            painter.restore()
            if t.key == self.selected:
                painter.setPen(QtGui.QPen(QtGui.QColor("#00e5ff"), 2))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawEllipse(p, 12, 12)
            if self.zoom >= 11 or t.key == self.selected:
                # Labels that would overlap are left out; hovering still names a target.
                box = metrics.boundingRect(t.label).translated(p + QtCore.QPointF(9, -6))
                if t.key == self.selected or not any(box.intersects(b) for b in labels):
                    labels.append(box)
                    painter.setPen(QtGui.QColor("#102027"))
                    painter.drawText(p + QtCore.QPointF(9, -6), t.label)

    # -- mouse
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.user_moved = True
            self._drag_from = event.position()
            self._drag_centre = world_xy(*self.centre, self.zoom)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is not None:
            d = event.position() - self._drag_from
            cx, cy = self._drag_centre
            self.centre = lat_lon(cx - d.x(), cy - d.y(), self.zoom)
            self.update()
            return
        t = self.target_at(event.position())
        self.setToolTip(t.describe() if t else "")

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is not None:
            moved = (event.position() - self._drag_from).manhattanLength()
            self._drag_from = None
            if moved < 4:
                t = self.target_at(event.position())
                if t is not None:
                    self.selected = t.key
                    self.targetClicked.emit(t.key)
                    self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        # Trackpads send many small steps; zoom one level per notch's worth.
        self.user_moved = True
        self._wheel += event.angleDelta().y()
        while abs(self._wheel) >= 120:
            step = 1 if self._wheel > 0 else -1
            self._wheel -= 120 * step
            self.set_zoom(self.zoom + step, event.position())
        event.accept()


class MapWindow(QtWidgets.QWidget):
    """The map beside a list of everything heard, in its own window."""

    COLUMNS = ("Name", "ID", "Kind", "Speed kn", "Heard s ago")

    def __init__(self, store: TargetStore, tiles: TileSource | None = None,
                 parent=None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Map — ships and aircraft")
        self.store = store
        self.map = MapView(store, tiles)
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        self.table.cellClicked.connect(self._on_row)
        self.map.targetClicked.connect(self._select_row)

        fit = QtWidgets.QPushButton("Fit all")
        fit.clicked.connect(self.fit_all)
        self.count = QtWidgets.QLabel("")
        side = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(fit)
        row.addWidget(self.count, 1)
        side.addLayout(row)
        side.addWidget(self.table, 1)
        panel = QtWidgets.QWidget()
        panel.setLayout(side)
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.map)
        splitter.addWidget(panel)
        splitter.setSizes([800, 380])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(splitter)
        self.resize(1200, 750)
        self._shown_version = -1
        self._fitted = False

    def fit_all(self) -> None:
        self.map.fit(self.store.placed())
        self.map.user_moved = False                 # and keep fitting as more arrive

    def refresh(self) -> None:
        """Redraw if anything changed; called by the main window's frame timer."""
        self.store.expire()
        if self.store.version != self._shown_version:
            self._shown_version = self.store.version
            placed = self.store.placed()
            if not self.map.user_moved and placed:
                # Until the user takes over, keep everything heard in view.
                self.map.fit(placed)
                self._fitted = True
            self._fill_table()
            self.map.update()

    def _fill_table(self) -> None:
        now = time.time()
        targets = sorted(self.store.targets.values(), key=lambda t: t.label)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(targets))
        for row, t in enumerate(targets):
            values = (t.label, t.ident, t.kind,
                      "" if t.speed_kn is None else f"{t.speed_kn:.1f}",
                      f"{now - t.last_seen:.0f}")
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, t.key)
                if col in (3, 4) and value:
                    item.setData(QtCore.Qt.ItemDataRole.DisplayRole, float(value))
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.count.setText(f"{len(targets)} heard, {len(self.store.placed())} on the map")

    def _on_row(self, row: int, _col: int) -> None:
        key = self.table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        t = self.store.targets.get(key)
        if t is not None:
            self.map.selected = key
            if t.lat is not None:
                self.map.centre_on(t.lat, t.lon)
            self.map.update()

    def _select_row(self, key: str) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(QtCore.Qt.ItemDataRole.UserRole) == key:
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                return
