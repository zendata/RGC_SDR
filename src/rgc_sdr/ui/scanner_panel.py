"""Scanner dock: range controls, the found list, and the lockout list.

A dock rather than another control row -- the found list needs vertical space, and the
whole panel can be hidden when not scanning.

The panel owns no scanning logic. It emits intent and displays state; `MainWindow` drives
the `Scanner` state machine from the frame loop.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

#: Presets for the bands this receiver can actually reach, with their channel spacing.
BAND_PRESETS: tuple[tuple[str, float, float, float], ...] = (
    ("Airband (VHF AM)", 118e6, 137e6, 25e3),
    ("Airband 8.33 kHz", 118e6, 137e6, 8.33e3),
    ("Marine VHF", 156e6, 163e6, 25e3),
    ("2 m amateur", 144e6, 146e6, 12.5e3),
    ("VHF FM broadcast", 87.5e6, 108e6, 100e3),
    ("MW broadcast", 0.531e6, 1.602e6, 9e3),
    ("49 m shortwave", 5.9e6, 6.2e6, 5e3),
    ("40 m amateur", 7.0e6, 7.2e6, 1e3),
)


class ScannerPanel(QtWidgets.QWidget):
    """Controls and results for a band sweep."""

    scanRequested = QtCore.pyqtSignal()
    stopRequested = QtCore.pyqtSignal()
    skipRequested = QtCore.pyqtSignal()
    lockOutCurrentRequested = QtCore.pyqtSignal()
    #: A found channel was chosen; tune to it (Hz).
    channelActivated = QtCore.pyqtSignal(float)
    lockOutRequested = QtCore.pyqtSignal(float)
    unlockRequested = QtCore.pyqtSignal(float)
    clearFoundRequested = QtCore.pyqtSignal()
    saveFoundRequested = QtCore.pyqtSignal(float)
    configChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # -- band preset ---------------------------------------------------
        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.addItem("Band preset…", None)
        for name, start, end, step in BAND_PRESETS:
            self._preset_combo.addItem(name, (start, end, step))
        self._preset_combo.activated.connect(self._on_preset)
        layout.addWidget(self._preset_combo)

        # -- range ---------------------------------------------------------
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        self._start_spin = self._freq_spin(118.0)
        self._end_spin = self._freq_spin(137.0)
        self._step_spin = QtWidgets.QDoubleSpinBox()
        self._step_spin.setRange(0.1, 1000.0)
        self._step_spin.setDecimals(2)
        self._step_spin.setValue(25.0)
        self._step_spin.setSuffix(" kHz")
        self._step_spin.setToolTip("Channel spacing the results are snapped to")
        self._threshold_spin = QtWidgets.QDoubleSpinBox()
        self._threshold_spin.setRange(3.0, 60.0)
        self._threshold_spin.setDecimals(0)
        self._threshold_spin.setValue(10.0)
        self._threshold_spin.setSuffix(" dB")
        self._threshold_spin.setToolTip("How far above the noise floor counts as a signal")
        for row, (label, widget) in enumerate((
            ("From", self._start_spin),
            ("To", self._end_spin),
            ("Step", self._step_spin),
            ("Threshold", self._threshold_spin),
        )):
            form.addWidget(QtWidgets.QLabel(label), row, 0)
            form.addWidget(widget, row, 1)
        layout.addLayout(form)

        for widget in (self._start_spin, self._end_spin, self._step_spin, self._threshold_spin):
            widget.valueChanged.connect(lambda *_: self.configChanged.emit())

        self._confirm_spin = QtWidgets.QSpinBox()
        self._confirm_spin.setRange(1, 10)
        self._confirm_spin.setValue(2)
        self._confirm_spin.setSuffix(" passes")
        self._confirm_spin.setToolTip(
            "How many passes a channel must appear on before it is stored.\n"
            "1 stores everything including noise spikes; 2 filters almost all of them."
        )
        form.addWidget(QtWidgets.QLabel("Confirm"), 4, 0)
        form.addWidget(self._confirm_spin, 4, 1)
        self._confirm_spin.valueChanged.connect(lambda *_: self.configChanged.emit())

        self._stop_on_signal = QtWidgets.QCheckBox("Stop on signal")
        self._stop_on_signal.setChecked(True)
        self._stop_on_signal.setToolTip("Unchecked: survey the range without pausing")
        self._stop_on_signal.toggled.connect(lambda *_: self.configChanged.emit())
        layout.addWidget(self._stop_on_signal)

        # -- run controls --------------------------------------------------
        buttons = QtWidgets.QHBoxLayout()
        self._scan_button = QtWidgets.QPushButton("Scan")
        self._scan_button.setCheckable(True)
        self._scan_button.clicked.connect(self._on_scan_clicked)
        buttons.addWidget(self._scan_button)
        self._skip_button = QtWidgets.QPushButton("Skip")
        self._skip_button.setToolTip("Leave this transmission and carry on")
        self._skip_button.clicked.connect(self.skipRequested.emit)
        buttons.addWidget(self._skip_button)
        self._lock_button = QtWidgets.QPushButton("Lock out")
        self._lock_button.setToolTip("Never stop here again")
        self._lock_button.clicked.connect(self.lockOutCurrentRequested.emit)
        buttons.addWidget(self._lock_button)
        layout.addLayout(buttons)

        self._status = QtWidgets.QLabel("idle")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        layout.addWidget(self._progress)

        # -- found ---------------------------------------------------------
        layout.addWidget(QtWidgets.QLabel("Found"))
        self._found_list = QtWidgets.QListWidget()
        self._found_list.setToolTip("Double-click to tune. These are separate from memories.")
        self._found_list.itemActivated.connect(self._on_found_activated)
        self._found_list.itemDoubleClicked.connect(self._on_found_activated)
        self._found_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._found_list.customContextMenuRequested.connect(self._on_found_menu)
        layout.addWidget(self._found_list, 3)

        found_buttons = QtWidgets.QHBoxLayout()
        tune = QtWidgets.QPushButton("Tune")
        tune.clicked.connect(self._tune_selected)
        found_buttons.addWidget(tune)
        lock_selected = QtWidgets.QPushButton("Lock")
        lock_selected.clicked.connect(self._lock_selected)
        found_buttons.addWidget(lock_selected)
        save_selected = QtWidgets.QPushButton("To memory")
        save_selected.setToolTip("Copy this into your named memories")
        save_selected.clicked.connect(self._save_selected)
        found_buttons.addWidget(save_selected)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.clearFoundRequested.emit)
        found_buttons.addWidget(clear)
        layout.addLayout(found_buttons)

        # -- lockout -------------------------------------------------------
        layout.addWidget(QtWidgets.QLabel("Locked out"))
        self._lock_list = QtWidgets.QListWidget()
        self._lock_list.setToolTip("Double-click to unlock")
        self._lock_list.itemDoubleClicked.connect(self._on_lock_activated)
        self._lock_list.setMaximumHeight(110)
        layout.addWidget(self._lock_list, 1)

        unlock = QtWidgets.QPushButton("Unlock selected")
        unlock.clicked.connect(self._unlock_selected)
        layout.addWidget(unlock)

        self.set_running(False)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _freq_spin(value_mhz: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 6000.0)
        spin.setDecimals(4)
        spin.setValue(value_mhz)
        spin.setSuffix(" MHz")
        spin.setKeyboardTracking(False)
        return spin

    def _on_preset(self, index: int) -> None:
        data = self._preset_combo.itemData(index)
        if not data:
            return
        start, end, step = data
        for spin, value in ((self._start_spin, start / 1e6), (self._end_spin, end / 1e6)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._step_spin.blockSignals(True)
        self._step_spin.setValue(step / 1e3)
        self._step_spin.blockSignals(False)
        self.configChanged.emit()

    def _on_scan_clicked(self, checked: bool) -> None:
        if checked:
            self.scanRequested.emit()
        else:
            self.stopRequested.emit()

    # -- configuration -----------------------------------------------------

    @property
    def start_hz(self) -> float:
        return self._start_spin.value() * 1e6

    @property
    def end_hz(self) -> float:
        return self._end_spin.value() * 1e6

    @property
    def step_hz(self) -> float:
        return self._step_spin.value() * 1e3

    @property
    def threshold_db(self) -> float:
        return self._threshold_spin.value()

    @property
    def stop_on_signal(self) -> bool:
        return self._stop_on_signal.isChecked()

    @property
    def min_sightings(self) -> int:
        return int(self._confirm_spin.value())

    def apply_config(
        self, start_hz, end_hz, step_hz, threshold_db, stop_on_signal, min_sightings=2
    ) -> None:
        for spin, value in (
            (self._start_spin, start_hz / 1e6),
            (self._end_spin, end_hz / 1e6),
            (self._step_spin, step_hz / 1e3),
            (self._threshold_spin, threshold_db),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._stop_on_signal.blockSignals(True)
        self._stop_on_signal.setChecked(bool(stop_on_signal))
        self._stop_on_signal.blockSignals(False)
        self._confirm_spin.blockSignals(True)
        self._confirm_spin.setValue(int(min_sightings))
        self._confirm_spin.blockSignals(False)

    def set_coverage(self, freq_ranges) -> None:
        """Grey out band presets this radio cannot tune."""
        model = self._preset_combo.model()
        for row in range(1, self._preset_combo.count()):
            data = self._preset_combo.itemData(row)
            if not data:
                continue
            start, end, _ = data
            covered = any(r.contains(start) and r.contains(end) for r in freq_ranges)
            item = model.item(row)
            if item is not None:
                item.setEnabled(covered)

    # -- display -----------------------------------------------------------

    def set_running(self, running: bool) -> None:
        self._scan_button.blockSignals(True)
        self._scan_button.setChecked(running)
        self._scan_button.blockSignals(False)
        self._scan_button.setText("Stop" if running else "Scan")
        self._skip_button.setEnabled(running)
        self._lock_button.setEnabled(running)
        if not running:
            self._progress.setValue(0)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_progress(self, index: int, total: int) -> None:
        self._progress.setMaximum(max(1, total))
        self._progress.setValue(max(0, min(index, total)))

    def set_found(self, channels) -> None:
        """Replace the found list. `channels` are settings.FoundChannel."""
        selected = self.selected_found_hz()
        self._found_list.clear()
        for channel in channels:
            item = QtWidgets.QListWidgetItem(channel.describe())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, channel.freq_hz)
            self._found_list.addItem(item)
            if selected is not None and abs(channel.freq_hz - selected) < 1.0:
                self._found_list.setCurrentItem(item)

    def set_lockout(self, frequencies) -> None:
        self._lock_list.clear()
        for freq in sorted(frequencies):
            item = QtWidgets.QListWidgetItem(f"{freq / 1e6:.4f} MHz")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, float(freq))
            self._lock_list.addItem(item)

    def selected_found_hz(self) -> float | None:
        item = self._found_list.currentItem()
        return None if item is None else float(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def selected_lock_hz(self) -> float | None:
        item = self._lock_list.currentItem()
        return None if item is None else float(item.data(QtCore.Qt.ItemDataRole.UserRole))

    # -- list actions ------------------------------------------------------

    def _on_found_activated(self, item) -> None:
        self.channelActivated.emit(float(item.data(QtCore.Qt.ItemDataRole.UserRole)))

    def _on_lock_activated(self, item) -> None:
        self.unlockRequested.emit(float(item.data(QtCore.Qt.ItemDataRole.UserRole)))

    def _tune_selected(self) -> None:
        freq = self.selected_found_hz()
        if freq is not None:
            self.channelActivated.emit(freq)

    def _lock_selected(self) -> None:
        freq = self.selected_found_hz()
        if freq is not None:
            self.lockOutRequested.emit(freq)

    def _save_selected(self) -> None:
        freq = self.selected_found_hz()
        if freq is not None:
            self.saveFoundRequested.emit(freq)

    def _unlock_selected(self) -> None:
        freq = self.selected_lock_hz()
        if freq is not None:
            self.unlockRequested.emit(freq)

    def _on_found_menu(self, point) -> None:
        item = self._found_list.itemAt(point)
        if item is None:
            return
        freq = float(item.data(QtCore.Qt.ItemDataRole.UserRole))
        menu = QtWidgets.QMenu(self)
        menu.addAction("Tune", lambda: self.channelActivated.emit(freq))
        menu.addAction("Lock out", lambda: self.lockOutRequested.emit(freq))
        menu.addAction("Save to memory…", lambda: self.saveFoundRequested.emit(freq))
        menu.exec(self._found_list.mapToGlobal(point))
