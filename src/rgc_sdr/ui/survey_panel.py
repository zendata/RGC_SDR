"""The DMR/P25 survey's controls and results, beside the scanner (VK3RQ, 2026-10-10).

The range is the scanner's From and To. Each pass steps the radio across it a span at a
time; every active channel in each span is decoded at once (survey.py), and the table
gathers what the passes found -- protocol, colour codes or NACs, voice heard and when,
encryption. It keeps surveying, pass after pass, until stopped: voice comes and goes,
and the longer it runs the more channels show it.

The panel owns no surveying logic: it emits intent and shows rows. Nothing decoded but
the counts is kept -- no IDs, no talkgroups.
"""

from __future__ import annotations

import time

from PyQt6 import QtCore, QtGui, QtWidgets

COLUMNS = ("Frequency", "Protocol", "CC / NAC", "Voice", "Last voice", "Encrypted", "Seen")


class SurveyPanel(QtWidgets.QWidget):
    startRequested = QtCore.pyqtSignal()
    stopRequested = QtCore.pyqtSignal()
    #: A row double-clicked or Tune: (frequency Hz, protocol).
    tuneRequested = QtCore.pyqtSignal(float, str)
    #: To memory: (frequency Hz, protocol, colour code or NAC text).
    saveRequested = QtCore.pyqtSignal(float, str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("DMR / P25 survey")
        title.setStyleSheet("font-weight: bold;")
        head.addWidget(title)
        self.start_button = QtWidgets.QPushButton("Survey")
        self.start_button.setCheckable(True)
        self.start_button.setToolTip(
            "Find the DMR and P25 channels between the scanner's From and To, and which\n"
            "carry voice: each span is decoded all at once, pass after pass until\n"
            "stopped. On a network radio the survey runs on its server.")
        self.start_button.toggled.connect(self._on_toggled)
        head.addWidget(self.start_button)
        self.voice_only = QtWidgets.QCheckBox("Voice only")
        self.voice_only.setToolTip("Show only channels where voice has been heard")
        self.voice_only.toggled.connect(self._refilter)
        head.addWidget(self.voice_only)
        head.addStretch(1)
        layout.addLayout(head)
        self.status = QtWidgets.QLabel("idle")
        self.status.setStyleSheet("color: #888;")
        layout.addWidget(self.status)
        self.table = QtWidgets.QTreeWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHeaderLabels(COLUMNS)
        self.table.setRootIsDecorated(False)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        # Bound methods, not lambdas: a lambda holding `self` makes a cycle with the Qt
        # object, and the garbage collector freeing it after Qt had deleted the widget
        # crashed the test run (2026-10-10).
        self.table.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table, 1)
        buttons = QtWidgets.QHBoxLayout()
        tune = QtWidgets.QPushButton("Tune")
        tune.clicked.connect(self._tune_current)
        buttons.addWidget(tune)
        save = QtWidgets.QPushButton("To memory")
        save.setToolTip("Save the selected channel as a memory, in DMR or P25 mode")
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.clear)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._rows: dict[float, QtWidgets.QTreeWidgetItem] = {}

    # -- state from the window

    def set_running(self, running: bool) -> None:
        self.start_button.blockSignals(True)
        self.start_button.setChecked(running)
        self.start_button.setText("Stop" if running else "Survey")
        self.start_button.blockSignals(False)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def show_table(self, table: dict) -> None:
        """`table` is survey.merge's: frequency -> record. Digital channels only."""
        for freq, row in table.items():
            if not row["protocol"]:
                continue
            item = self._rows.get(freq)
            if item is None:
                item = _SortItem()
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, freq)
                self.table.addTopLevelItem(item)
                self._rows[freq] = item
            codes = (", ".join(f"CC {c}" for c in sorted(row["colour_codes"]))
                     if row["protocol"] == "DMR"
                     else ", ".join(f"NAC {n:03X}" for n in sorted(row["nacs"])))
            slots = row.get("voice_slots") or set()
            voice = str(row["voice"]) + (f" (slot {', '.join(map(str, sorted(slots)))})"
                                         if slots else "")
            last = (time.strftime("%H:%M:%S", time.localtime(row["last_voice"]))
                    if row["last_voice"] else "")
            values = (f"{freq / 1e6:.5f}", row["protocol"], codes, voice if row["voice"] else "",
                      last, "yes" if row["encrypted"] else "", str(row["heard"]))
            for column, text in enumerate(values):
                item.setText(column, text)
            item.setData(3, QtCore.Qt.ItemDataRole.UserRole, row["voice"])
            colour = QtGui.QColor("#a5d6a7" if row["voice"] and not row["encrypted"]
                                  else "#e0e0e0" if not row["encrypted"] else "#9e9e9e")
            for column in range(len(COLUMNS)):
                item.setForeground(column, colour)
        self._refilter()

    def clear(self) -> None:
        self.table.clear()
        self._rows = {}

    # -- internals

    def _refilter(self) -> None:
        only = self.voice_only.isChecked()
        for item in self._rows.values():
            item.setHidden(only and not item.data(3, QtCore.Qt.ItemDataRole.UserRole))

    def _on_toggled(self, on: bool) -> None:
        self.set_running(on)
        (self.startRequested if on else self.stopRequested).emit()

    def _on_double_click(self, item, _column) -> None:
        self._tune(item)

    def _tune_current(self) -> None:
        self._tune(self.table.currentItem())

    def _tune(self, item) -> None:
        if item is not None:
            self.tuneRequested.emit(float(item.data(0, QtCore.Qt.ItemDataRole.UserRole)),
                                    item.text(1))

    def _save(self) -> None:
        item = self.table.currentItem()
        if item is not None:
            self.saveRequested.emit(float(item.data(0, QtCore.Qt.ItemDataRole.UserRole)),
                                    item.text(1), item.text(2))


class _SortItem(QtWidgets.QTreeWidgetItem):
    """Frequency and voice sort as numbers, not text."""

    def __lt__(self, other) -> bool:
        column = self.treeWidget().sortColumn() if self.treeWidget() else 0
        if column == 0:
            return (self.data(0, QtCore.Qt.ItemDataRole.UserRole) or 0) < \
                (other.data(0, QtCore.Qt.ItemDataRole.UserRole) or 0)
        if column == 3:
            return (self.data(3, QtCore.Qt.ItemDataRole.UserRole) or 0) < \
                (other.data(3, QtCore.Qt.ItemDataRole.UserRole) or 0)
        return super().__lt__(other)
