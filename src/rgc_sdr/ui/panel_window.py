"""A panel in a window of its own (VK3RQ, 2026-10-10).

Panels opened from the tab row stack above the spectrum and waterfall, which on a laptop
screen leaves the waterfall a strip once Memory or Map is open. Any panel can instead
live in its own window, moved and sized freely; where it was is remembered for the next
time it opens. The window only holds the panel: the main window still owns it, and takes
it back when the panel is docked again.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets


class PanelWindow(QtWidgets.QWidget):
    #: The window was closed by its own close button (the tab should uncheck).
    closed = QtCore.pyqtSignal()
    #: It moved or was resized: its geometry is worth saving.
    moved = QtCore.pyqtSignal()

    def __init__(self, title: str, parent=None) -> None:
        # A real window, but owned by the main one: it closes with it and is never
        # left behind.
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle(title)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)

    def hold(self, panel: QtWidgets.QWidget) -> None:
        self._layout.addWidget(panel)

    def release(self, panel: QtWidgets.QWidget) -> None:
        self._layout.removeWidget(panel)

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        # Its own close button (hide() from the tab row does not come here).
        self.closed.emit()
        super().closeEvent(event)

    def moveEvent(self, event) -> None:  # noqa: N802
        self.moved.emit()
        super().moveEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self.moved.emit()
        super().resizeEvent(event)
