"""The IC-705's memory channels in a Mac window.

One group at a time (00-99, and the call channels), read in the background: 100
channels take about five seconds. From here a channel can be tuned, copied into the
app's own memories, renamed or retuned in place, cleared, or filled from what the radio
is on now. Every write is read back, so the table always shows what the radio holds.
"""

from __future__ import annotations

import threading
from dataclasses import replace

from PyQt6 import QtCore, QtWidgets

from ..device.ic705_memory import CALL_GROUP, DUPLEX, MemoryChannel, MemorySide

COLUMNS = ("Ch", "Name", "Frequency (MHz)", "Mode", "Duplex", "Tone")
#: Columns that can be edited in place.
EDITABLE = {1: "name", 2: "freq"}


class _GroupReader(QtCore.QObject):
    read = QtCore.pyqtSignal(int, int, object)      # group, channel, MemoryChannel | "blank" | None
    finished = QtCore.pyqtSignal()

    def __init__(self, source, group: int, channels) -> None:
        super().__init__()
        self._source, self._group, self._channels = source, group, list(channels)
        self._stop = threading.Event()

    def run(self) -> None:
        for channel in self._channels:
            if self._stop.is_set():
                break
            self.read.emit(self._group, channel, self._source.read_memory(self._group, channel))
        self.finished.emit()

    def stop(self) -> None:
        self._stop.set()


class MemoryWindow(QtWidgets.QDialog):
    """A group of the radio's memory channels as a table, with actions on the selection."""

    #: Tune the app (and so the radio) to a channel: frequency, radio mode.
    tune_requested = QtCore.pyqtSignal(float, str)
    #: Copy a channel into the app's memories: name, frequency, radio mode.
    save_requested = QtCore.pyqtSignal(str, float, str)

    def __init__(self, source, parent=None, read_now: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("IC-705 memories")
        self.resize(760, 620)
        self._source = source
        self.channels: dict[int, MemoryChannel | str | None] = {}
        self._filling = False
        self._thread: QtCore.QThread | None = None
        self._reader: _GroupReader | None = None

        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Group"))
        self.group_combo = QtWidgets.QComboBox()
        for g in range(100):
            self.group_combo.addItem(f"{g:02d}", g)
        self.group_combo.addItem("Call channels", CALL_GROUP)
        self.group_combo.currentIndexChanged.connect(lambda _i: self.read_group())
        top.addWidget(self.group_combo)
        self.hide_blank = QtWidgets.QCheckBox("Hide blank channels")
        self.hide_blank.setChecked(True)
        self.hide_blank.toggled.connect(self._apply_hiding)
        top.addWidget(self.hide_blank)
        top.addStretch(1)
        self.reload = QtWidgets.QPushButton("Read from radio")
        self.reload.clicked.connect(self.read_group)
        top.addWidget(self.reload)
        layout.addLayout(top)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(1, 170)
        self.table.setColumnWidth(2, 140)
        self.table.itemChanged.connect(self._edited)
        self.table.cellDoubleClicked.connect(lambda row, col: self.tune_selected()
                                             if col not in EDITABLE else None)
        layout.addWidget(self.table, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.tune_button = QtWidgets.QPushButton("Tune")
        self.tune_button.setToolTip("Tune to this channel (double-click a row does the same)")
        self.tune_button.clicked.connect(self.tune_selected)
        self.save_button = QtWidgets.QPushButton("Save to app memories")
        self.save_button.clicked.connect(self.save_selected)
        self.store_button = QtWidgets.QPushButton("Store radio's frequency here")
        self.store_button.setToolTip("Write what the radio is on now into this channel")
        self.store_button.clicked.connect(self.store_current)
        self.clear_button = QtWidgets.QPushButton("Clear channel")
        self.clear_button.clicked.connect(self.clear_selected)
        for b in (self.tune_button, self.save_button, self.store_button, self.clear_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

        self._fill_empty_rows()
        if read_now:
            self.read_group()

    # -- reading ----------------------------------------------------------------------------

    @property
    def group(self) -> int:
        return int(self.group_combo.currentData())

    def _channel_numbers(self) -> range:
        return range(4) if self.group == CALL_GROUP else range(100)

    def _fill_empty_rows(self) -> None:
        self._filling = True
        self.table.setRowCount(len(self._channel_numbers()))
        for row, ch in enumerate(self._channel_numbers()):
            label = MemoryChannel(self.group, ch).label
            for col, text in enumerate((label, "", "", "", "", "")):
                item = QtWidgets.QTableWidgetItem(text)
                flags = item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
                self.table.setItem(row, col, item)
        self._filling = False

    def read_group(self) -> None:
        """Read the chosen group from the radio, in the background."""
        self._stop_reader()
        self.channels.clear()
        self._fill_empty_rows()
        self._thread = QtCore.QThread(self)
        self._reader = _GroupReader(self._source, self.group, self._channel_numbers())
        self._reader.moveToThread(self._thread)
        self._thread.started.connect(self._reader.run)
        self._reader.read.connect(self.show_channel)
        self._reader.finished.connect(self._read_done)
        self.status.setText(f"Reading group {self.group_combo.currentText()} …")
        self._thread.start()

    def _stop_reader(self) -> None:
        if self._reader is not None:
            self._reader.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = self._reader = None

    def _read_done(self) -> None:
        used = sum(isinstance(m, MemoryChannel) for m in self.channels.values())
        self.status.setText(f"Group {self.group_combo.currentText()}: {used} channel"
                            f"{'' if used == 1 else 's'} in use.")
        self._stop_reader()
        self._apply_hiding()

    def show_channel(self, group: int, channel: int, memory) -> None:
        if group != self.group:
            return                                       # a late answer for another group
        self.channels[channel] = memory
        row = list(self._channel_numbers()).index(channel)
        self._filling = True
        try:
            if isinstance(memory, MemoryChannel):
                rx = memory.rx
                dup = DUPLEX.get(rx.duplex, "")
                if dup and rx.offset_hz:
                    dup += f" {rx.offset_hz / 1e6:g} MHz"
                texts = (memory.label, memory.name, f"{rx.freq_hz / 1e6:.6f}",
                         rx.mode.upper() + ("-D" if rx.data else ""), dup, rx.tone_text)
            else:
                texts = (MemoryChannel(group, channel).label,
                         "" if memory == "blank" else "(no answer)", "", "", "", "")
            for col, text in enumerate(texts):
                item = self.table.item(row, col)
                item.setText(text)
                editable = isinstance(memory, MemoryChannel) and col in EDITABLE
                flags = item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable if editable \
                    else item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
        finally:
            self._filling = False
        if self._thread is None:
            self._apply_hiding()

    def _apply_hiding(self) -> None:
        hide = self.hide_blank.isChecked()
        for row, ch in enumerate(self._channel_numbers()):
            self.table.setRowHidden(row, hide and self.channels.get(ch) == "blank")

    # -- actions ----------------------------------------------------------------------------

    def _selected(self) -> tuple[int, MemoryChannel | str | None]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return -1, None
        channel = list(self._channel_numbers())[rows[0].row()]
        return channel, self.channels.get(channel)

    def tune_selected(self) -> None:
        _ch, memory = self._selected()
        if isinstance(memory, MemoryChannel):
            self.tune_requested.emit(float(memory.rx.freq_hz), memory.rx.mode)
            self.status.setText(f"Tuned to {memory.label} {memory.name}")

    def save_selected(self) -> None:
        _ch, memory = self._selected()
        if not isinstance(memory, MemoryChannel):
            return
        default = memory.name or f"{memory.rx.freq_hz / 1e6:.4f} MHz"
        name, ok = QtWidgets.QInputDialog.getText(self, "Save to app memories",
                                                  "Name for this memory:", text=default)
        if ok and name.strip():
            self.save_requested.emit(name.strip(), float(memory.rx.freq_hz), memory.rx.mode)
            self.status.setText(f"Saved “{name.strip()}” to the app's memories")

    def _write(self, memory: MemoryChannel) -> None:
        """Write a channel, then read it back so the table shows what the radio holds."""
        self._source.write_memory(memory)
        confirmed = getattr(self._source, "_confirmed", lambda: True)()
        back = self._source.read_memory(memory.group, memory.channel)
        self.show_channel(memory.group, memory.channel, back)
        if not confirmed or getattr(self._source, "refused", None) == f"memory {memory.label}":
            self.status.setText(f"The radio did not accept the change to {memory.label}")
        else:
            self.status.setText(f"Wrote {memory.label} {memory.name}")

    def store_current(self) -> None:
        channel, memory = self._selected()
        if channel < 0:
            return
        src = self._source
        state, status = getattr(src, "state", {}) or {}, getattr(src, "status", {}) or {}
        dup = {0x11: 1, 0x12: 2}.get(state.get("dup"), 0)
        tone = state.get("tone") or 0
        side = MemorySide(
            freq_hz=int(src.center_freq), mode=getattr(src, "mode", None) or "fm",
            filter=getattr(src, "filter", None) or 1, duplex=dup,
            tone_mode=tone if tone <= 3 else {6: 4, 7: 5, 8: 6, 9: 7}.get(tone, 0),
            tone_hz=status.get("tone_hz") or 88.5, tsql_hz=status.get("tsql_hz") or 88.5,
            dtcs=status.get("dtcs") or "023", offset_hz=int(status.get("offset_hz") or 0))
        name = memory.name if isinstance(memory, MemoryChannel) else ""
        if isinstance(memory, MemoryChannel):
            answer = QtWidgets.QMessageBox.question(
                self, "Overwrite channel", f"Replace {memory.label} {memory.name} with "
                f"{side.freq_hz / 1e6:.4f} MHz {side.mode.upper()}?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._write(MemoryChannel.simplex(self.group, channel, side, name=name))

    def clear_selected(self) -> None:
        channel, memory = self._selected()
        if not isinstance(memory, MemoryChannel):
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Clear channel", f"Clear {memory.label} {memory.name} on the radio?")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._source.clear_memory(self.group, channel)
        getattr(self._source, "_confirmed", lambda: True)()
        self.show_channel(self.group, channel, self._source.read_memory(self.group, channel))
        self.status.setText(f"Cleared {memory.label}")

    def _edited(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._filling or item.column() not in EDITABLE:
            return
        channel = list(self._channel_numbers())[item.row()]
        memory = self.channels.get(channel)
        if not isinstance(memory, MemoryChannel):
            return
        text = item.text().strip()
        if EDITABLE[item.column()] == "name":
            changed = replace(memory, name=text[:16])
        else:
            try:
                hz = round(float(text) * 1e6)
            except ValueError:
                self.show_channel(self.group, channel, memory)
                self.status.setText("Enter the frequency in MHz, e.g. 146.900")
                return
            changed = replace(memory, rx=replace(memory.rx, freq_hz=hz))
            if not memory.split:
                changed = replace(changed, tx=replace(memory.tx, freq_hz=hz))
        self._write(changed)

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        self._stop_reader()
        super().closeEvent(event)
