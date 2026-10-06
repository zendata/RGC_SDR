"""The Decode dock: choose a data decoder and read what it finds.

Every decoder shows its text, pager messages included (VK3RQ, 2026-10-06; at first they
were hidden unless asked for). "Show text" can hide pager text again, for this session.
Nothing shown here is written to disk (PLANNING.md 7p).
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from ..decoding import DECODERS

#: Messages kept for redisplay when "Show text" changes.
KEEP_MESSAGES = 500


class DecoderPanel(QtWidgets.QWidget):
    #: The chosen decoder's key, or "" for off.
    decoderChanged = QtCore.pyqtSignal(str)
    #: The Map button: show the map window.
    mapRequested = QtCore.pyqtSignal()

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
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(KEEP_MESSAGES)
        mono = QtGui.QFont("Menlo")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.log.setFont(mono)
        # Wrapped to the panel, so a long message or APRS comment is read without
        # scrolling sideways.
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        outer.addWidget(self.log, 1)
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

    def add(self, messages: list) -> None:
        if not messages:
            return
        self._messages = (self._messages + list(messages))[-KEEP_MESSAGES:]
        show = self.show_text.isChecked()
        for message in messages:
            self.log.appendPlainText(message.summary(show_text=show))

    def _redraw(self) -> None:
        show = self.show_text.isChecked()
        self.log.setPlainText("\n".join(m.summary(show_text=show) for m in self._messages))
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def clear(self) -> None:
        self._messages = []
        self.log.clear()

    def set_status(self, text: str) -> None:
        self.status.setText(text)
