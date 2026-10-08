"""Which radio to start with, when the one asked for cannot be opened (VK3RQ, 2026-10-08).

The last radio used is reopened at start. When that fails -- the IC-705 over WiFi is
switched off, say, which can only be found out by trying to log in -- this lists every
radio that is there instead: SDRs on this Mac, radios on a network radio server, the
IC-705 on USB or WiFi. Choosing one opens it; if that fails too, the reason is shown and
the list stays, with Look again for a radio just plugged in, and Quit.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

KEY_ROLE = QtCore.Qt.ItemDataRole.UserRole


class StartupRadioDialog(QtWidgets.QDialog):
    """Lists the radios that are there; `exec` returns the source opened, or None."""

    def __init__(self, problem: str, opener, availability_fn, failed: str = "",
                 parent=None) -> None:
        """`opener(key)` opens radio `key` (raising on failure); `availability_fn()`
        lists `Availability` entries, as `device.profiles.availability`. The radio that
        `failed` is listed last, so Open does not simply try it again."""
        super().__init__(parent)
        self._failed = failed
        self.setWindowTitle("VK3RQ Super SDR — choose a radio")
        self._opener = opener
        self._availability_fn = availability_fn
        self.source = None
        layout = QtWidgets.QVBoxLayout(self)
        self.problem = QtWidgets.QLabel(problem)
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #c62828;")
        layout.addWidget(self.problem)
        layout.addWidget(QtWidgets.QLabel("Start with one of these instead:"))
        self.radios = QtWidgets.QListWidget()
        self.radios.itemDoubleClicked.connect(lambda _item: self._open())
        self.radios.currentItemChanged.connect(lambda *_: self._sync_buttons())
        layout.addWidget(self.radios, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.again_button = QtWidgets.QPushButton("Look again")
        self.again_button.setToolTip("Search for radios again, after plugging one in")
        self.again_button.clicked.connect(self.refresh)
        buttons.addWidget(self.again_button)
        buttons.addStretch(1)
        self.quit_button = QtWidgets.QPushButton("Quit")
        self.quit_button.clicked.connect(self.reject)
        buttons.addWidget(self.quit_button)
        self.open_button = QtWidgets.QPushButton("Open")
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self._open)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)
        self.resize(460, 340)
        self.refresh()

    def refresh(self) -> None:
        """List the radios there now: attached ones, network ones, and the IC-705 over
        WiFi, which cannot be seen without logging in, so is always offered."""
        self.radios.clear()
        entries = sorted(self._availability_fn(),
                         key=lambda e: e.profile.key == self._failed)     # stable
        for entry in entries:
            profile = entry.profile
            if not entry.connected and profile.key != "icom705net":
                continue
            label = profile.label
            if profile.key == "icom705net":
                label += "  — log in…"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(KEY_ROLE, profile.key)
            item.setToolTip(profile.notes or profile.describe_ranges())
            self.radios.addItem(item)
        if self.radios.count():
            self.radios.setCurrentRow(0)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        self.open_button.setEnabled(self.radios.currentItem() is not None)

    def chosen_key(self) -> str | None:
        item = self.radios.currentItem()
        return item.data(KEY_ROLE) if item is not None else None

    def _open(self) -> None:
        key = self.chosen_key()
        if key is None:
            return
        self.setCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            source = self._opener(key)
        except Exception as exc:
            label = self.radios.currentItem().text().split("  —")[0]
            self.problem.setText(f"Could not open {label}: {exc}")
            return
        finally:
            self.unsetCursor()
        if source is None:                  # a login was cancelled: stay here
            return
        self.source = source
        self.accept()


def choose_radio(problem: str, opener, availability_fn, failed: str = "", parent=None):
    """Show the chooser; the source opened, or None if the user quit."""
    dialog = StartupRadioDialog(problem, opener, availability_fn, failed, parent)
    if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        return dialog.source
    return None
