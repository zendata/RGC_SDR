"""The Decode dock: choose a data decoder and read what it finds.

Every decoder shows its text, pager messages included (VK3RQ, 2026-10-06; at first they
were hidden unless asked for). "Show text" can hide pager text again, for this session.
Nothing shown here is written to disk (PLANNING.md 7p).
"""

from __future__ import annotations

import time

from PyQt6 import QtCore, QtGui, QtWidgets

from ..decoding import DECODERS

#: Messages kept for redisplay when "Show text" changes.
KEEP_MESSAGES = 500


class DecoderPanel(QtWidgets.QWidget):
    #: The chosen decoder's key, or "" for off.
    decoderChanged = QtCore.pyqtSignal(str)
    #: The Map button: show the map window.
    mapRequested = QtCore.pyqtSignal()
    #: A voice call double-clicked: (frequency Hz) to listen there in P25 mode.
    callTuneRequested = QtCore.pyqtSignal(float)

    #: Voice calls kept in the table, most recent first.
    CALLS_KEPT = 60
    CALL_COLUMNS = ("Heard", "Talkgroup / to", "From", "Frequency", "Phase", "Encrypted",
                    "Hearable")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Decode"))
        self.combo = QtWidgets.QComboBox()
        self.combo.addItem("Off", "")
        for key, spec in DECODERS.items():
            self.combo.addItem(spec.label, key)
        self.combo.currentIndexChanged.connect(self._on_choice)
        top.addWidget(self.combo)
        self.show_text = QtWidgets.QCheckBox("Show text")
        self.show_text.setChecked(True)
        self.show_text.setToolTip(
            "Pager messages can carry names, addresses and medical details.\n"
            "Untick to hide their text; nothing decoded is ever saved.")
        self.show_text.toggled.connect(self._redraw)
        top.addWidget(self.show_text)
        self.map_button = QtWidgets.QPushButton("Map")
        self.map_button.setToolTip("Show what has been located on a map")
        self.map_button.clicked.connect(self.mapRequested)
        top.addWidget(self.map_button)
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        top.addWidget(self.clear_button)
        outer.addLayout(top)

        # On the controls line, not a line of its own: across the top of the window the
        # panel is short, and every line goes to the messages.
        self.status = QtWidgets.QLabel("")
        top.insertWidget(top.indexOf(self.show_text) + 1, self.status, 1)
        # Room between "Show text" and the status (VK3RQ, 2026-10-07).
        top.insertSpacing(top.indexOf(self.status), 24)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(KEEP_MESSAGES)
        mono = QtGui.QFont("Menlo")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.log.setFont(mono)
        # Wrapped to the panel, so a long message or APRS comment is read without
        # scrolling sideways.
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        # P25 voice calls the control channel grants, beside the messages (VK3RQ,
        # 2026-10-08): where each call is, and whether the app can play it.
        self.calls = QtWidgets.QTreeWidget()
        self.calls.setColumnCount(len(self.CALL_COLUMNS))
        self.calls.setHeaderLabels(self.CALL_COLUMNS)
        self.calls.setRootIsDecorated(False)
        self.calls.setToolTip(
            "Voice calls the control channel has granted. Hearable: Phase 1 and not\n"
            "encrypted -- double-click one to listen on its frequency. Phase 2 (TDMA)\n"
            "and encrypted calls cannot be played.")
        self.calls.itemDoubleClicked.connect(self._on_call_double_clicked)
        self._call_items: dict = {}
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self.log)
        split.addWidget(self.calls)
        split.setSizes([600, 500])
        outer.removeWidget(self.log)
        outer.addWidget(split, 1)
        self.calls.hide()
        self._messages: list = []
        self._sync_show_text()

    @property
    def decoder(self) -> str:
        return self.combo.currentData() or ""

    def _on_choice(self) -> None:
        self._sync_show_text()
        self.decoderChanged.emit(self.decoder)

    def _sync_show_text(self) -> None:
        spec = DECODERS.get(self.decoder)
        private = bool(spec and spec.private)
        self.show_text.setVisible(private)
        self.calls.setVisible(self.decoder == "p25")

    def add(self, messages: list) -> None:
        if not messages:
            return
        self._messages = (self._messages + list(messages))[-KEEP_MESSAGES:]
        show = self.show_text.isChecked()
        for message in messages:
            self.log.appendPlainText(message.summary(show_text=show))
            for call in getattr(message, "fields", {}).get("calls", ()):
                self._note_call(getattr(message, "nac", 0), call, message.received)

    def _redraw(self) -> None:
        show = self.show_text.isChecked()
        self.log.setPlainText("\n".join(m.summary(show_text=show) for m in self._messages))
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def clear(self) -> None:
        self._messages = []
        self.log.clear()
        self.calls.clear()
        self._call_items = {}

    @staticmethod
    def hearable(call: dict) -> str:
        """Whether the app can play a call, or why not."""
        if call.get("encrypted"):
            return "no: encrypted"
        if call.get("phase") == "2":
            return "no: Phase 2"
        if call.get("phase") == "?":
            return "?"                       # no channel plan heard yet
        return "yes" if call.get("encrypted") is False else "yes, unless encrypted"

    def _note_call(self, nac: int, call: dict, received: float) -> None:
        """Add a call to the table or refresh it: one row per talkgroup (or pair of
        radios), moved to the top each time it is granted."""
        who = (f"TG {call['group']}" if call.get("group") is not None
               else f"to {call.get('target', '?')}")
        key = (nac, who)
        item = self._call_items.get(key)
        if item is None:
            item = QtWidgets.QTreeWidgetItem()
            self._call_items[key] = item
        else:
            self.calls.takeTopLevelItem(self.calls.indexOfTopLevelItem(item))
        source = call.get("source")
        if source is None and item.text(2):
            source_text = item.text(2)              # an update does not name the talker
        else:
            source_text = "" if source is None else str(source)
        freq = call.get("freq_hz")
        slot = f" slot {call['slot']}" if call.get("slot") else ""
        encrypted = call.get("encrypted")
        values = (time.strftime("%H:%M:%S", time.localtime(received)), who, source_text,
                  (f"{freq / 1e6:.5f} MHz{slot}" if freq else f"ch {call['channel']}"),
                  {"1": "1", "2": "2 (TDMA)"}.get(call.get("phase"), "?"),
                  {True: "yes", False: "no"}.get(encrypted, item.text(5) or "?"),
                  self.hearable(call))
        for column, text in enumerate(values):
            item.setText(column, text)
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, freq)
        playable = values[-1].startswith("yes")
        colour = QtGui.QColor("#a5d6a7" if playable else "#9e9e9e")
        for column in range(len(values)):
            item.setForeground(column, colour)
        self.calls.insertTopLevelItem(0, item)
        while self.calls.topLevelItemCount() > self.CALLS_KEPT:
            old = self.calls.takeTopLevelItem(self.calls.topLevelItemCount() - 1)
            self._call_items = {k: v for k, v in self._call_items.items() if v is not old}

    def _on_call_double_clicked(self, item, _column) -> None:
        freq = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if freq:
            self.callTuneRequested.emit(float(freq))

    def set_status(self, text: str) -> None:
        self.status.setText(text)
