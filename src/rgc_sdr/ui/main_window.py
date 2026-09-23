"""Main window: spectrum above waterfall, driven by a frame timer.

Device controls are generated from `DeviceCaps`, not hardcoded, so a radio with gain
elements (HackRF) grows sliders while the Airspy HF+ -- which reports none -- shows only
its AGC toggle. See PLANNING.md sections 3 and 4.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from ..audio import AudioSink, audio_available
from ..device.source import IQSource
from ..dsp.decimate import Decimator
from ..dsp.demod import BANDWIDTH_PRESETS, MODE_SPECS, MODES
from ..recorder import DEFAULT_DIR, AudioRecorder, IQRecorder, timestamp_name
from ..scanner import ScanAction, ScanConfig, Scanner
from ..dsp.spectrum import SpectrumAnalyzer
from ..settings import Settings, Snapshot
from .scanner_panel import ScannerPanel
from .smeter import SMeter
from .spectrum_view import SpectrumView
from .waterfall import COLORMAPS, WaterfallView

FFT_SIZES = (1024, 2048, 4096, 8192, 16384)
#: Decimation factors offered as a zoom control. Powers of two, matching Decimator.
ZOOM_FACTORS = (1, 2, 4, 8, 16, 32)
#: Fraction of the ring a single frame may consume, so deep zoom cannot starve itself.
FRAME_INPUT_BUDGET = 0.6


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        source: IQSource,
        fft_size: int = 4096,
        fps: int = 25,
        history_rows: int = 512,
        waterfall_bins: int = 1024,
        colormap: str = "inferno",
        levels: tuple[float, float] | None = None,
        decimation: int = 1,
        peak_hold: bool = True,
        mode: str = "off",
        volume: float = 0.4,
        offset_hz: float = 0.0,
        squelch_dbfs: float | None = None,
        bandwidth_hz: float | None = None,
        step_hz: float = 10e3,
        enable_audio: bool = True,
        recordings_dir=None,
        settings: Settings | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.source = source
        self.fps = max(1, int(fps))
        self.analyzer = SpectrumAnalyzer(fft_size=fft_size)
        self.decimator = Decimator(decimation)
        # Only persist when a store was supplied, so tests never touch the real file.
        self._persist = settings is not None
        self.settings = settings if settings is not None else Settings()
        # `levels=None` means "fit to whatever this antenna is actually receiving once a
        # little history exists". Signal levels vary by tens of dB with antenna and band
        # (measured: -120..-85 dBFS on this setup, but -47 dBFS with a strong carrier), so
        # a fixed default range renders the waterfall uniformly blank as often as not.
        self._auto_pending = levels is None
        # Remember whether the range was *chosen* or merely fitted: a fitted range suits
        # this antenna today, so it is not worth restoring on a later launch.
        self._levels_explicit = levels is not None
        self._levels = levels if levels is not None else (-120.0, -60.0)
        self._initial_peak_hold = bool(peak_hold)
        self._initial_mode = mode if mode in MODES else "off"
        self._initial_volume = float(volume)
        self._initial_offset = float(offset_hz)
        self._initial_squelch = squelch_dbfs
        self._initial_step_hz = float(step_hz)
        self._initial_bandwidth = bandwidth_hz
        self.recordings_dir = Path(recordings_dir) if recordings_dir else DEFAULT_DIR
        self.audio_recorder: AudioRecorder | None = None
        self.iq_recorder: IQRecorder | None = None
        self.scanner: Scanner | None = None
        # Audio is optional: without a usable device the rest of the app still works.
        self._audio_ok = bool(enable_audio) and audio_available()
        self.audio: AudioSink | None = None
        self._rows_pushed = 0
        self._frames = 0
        self._fps_mark = time.perf_counter()
        self._measured_fps = 0.0

        levels = self._levels
        caps = source.caps
        self.setWindowTitle(f"RGC_SDR - {caps.label or caps.driver}")

        self.spectrum = SpectrumView()
        self.waterfall = WaterfallView(
            rows=history_rows, cols=waterfall_bins, colormap=colormap, levels=levels
        )
        self.spectrum.set_levels(*levels)
        self.waterfall.setXLink(self.spectrum)  # one shared frequency axis
        self.spectrum.frequencySelected.connect(self._retune)
        self.waterfall.frequencySelected.connect(self._retune)
        self.spectrum.frequencyNudged.connect(self.nudge_frequency)
        self.waterfall.frequencyNudged.connect(self.nudge_frequency)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.spectrum)
        splitter.addWidget(self.waterfall)
        splitter.setSizes([250, 550])

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._build_controls(), 0)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self._status = self.statusBar()
        self._build_scanner_dock()
        self._apply_geometry()
        self.spectrum.set_center_marker(source.center_freq)

        # Debounced, so dragging a spinbox does not rewrite the file on every step.
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)
        self._save_timer.timeout.connect(self._save_state)

        if self._initial_mode != "off":
            self.set_mode(self._initial_mode)
        self._update_passband()

        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._on_frame)
        self._timer.start(int(1000 / self.fps))

    # -- scanner -----------------------------------------------------------

    def _build_scanner_dock(self) -> None:
        self.scanner_panel = ScannerPanel()
        scan = self.settings.scan
        self.scanner_panel.apply_config(
            scan.start_hz, scan.end_hz, scan.step_hz, scan.threshold_db,
            scan.stop_on_signal, scan.min_sightings,
        )
        self.scanner_panel.set_found(self.settings.found)
        self.scanner_panel.set_lockout(self.settings.lockout)

        self.scanner_panel.scanRequested.connect(self.start_scan)
        self.scanner_panel.stopRequested.connect(self.stop_scan)
        self.scanner_panel.skipRequested.connect(self.skip_current)
        self.scanner_panel.lockOutCurrentRequested.connect(self.lock_out_current)
        self.scanner_panel.channelActivated.connect(self._tune_to_found)
        self.scanner_panel.lockOutRequested.connect(self.lock_out)
        self.scanner_panel.unlockRequested.connect(self.unlock)
        self.scanner_panel.clearFoundRequested.connect(self.clear_found)
        self.scanner_panel.saveFoundRequested.connect(self._save_found_to_memory)
        self.scanner_panel.configChanged.connect(self._on_scan_config_changed)

        self._scanner_dock = QtWidgets.QDockWidget("Scanner", self)
        self._scanner_dock.setObjectName("scannerDock")
        self._scanner_dock.setWidget(self.scanner_panel)
        self._scanner_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._scanner_dock)

    def _scan_config(self) -> ScanConfig:
        panel = self.scanner_panel
        return ScanConfig(
            start_hz=panel.start_hz,
            end_hz=panel.end_hz,
            step_hz=panel.step_hz,
            threshold_db=panel.threshold_db,
            stop_on_signal=panel.stop_on_signal,
            min_sightings=panel.min_sightings,
        )

    def _on_scan_config_changed(self) -> None:
        panel = self.scanner_panel
        scan = self.settings.scan
        scan.start_hz, scan.end_hz = panel.start_hz, panel.end_hz
        scan.step_hz, scan.threshold_db = panel.step_hz, panel.threshold_db
        scan.stop_on_signal = panel.stop_on_signal
        scan.min_sightings = panel.min_sightings
        self._schedule_save()

    def start_scan(self) -> bool:
        """Begin sweeping the configured range."""
        try:
            config = self._scan_config()
            self.scanner = Scanner(config, lockout=set(self.settings.lockout))
        except ValueError as exc:
            self.scanner = None
            self.scanner_panel.set_running(False)
            self.scanner_panel.set_status(str(exc))
            return False
        caps = self.source.caps
        if not (caps.covers(config.start_hz) and caps.covers(config.end_hz)):
            self.scanner_panel.set_status(
                f"range is outside this receiver ({caps.describe_ranges()})"
            )
            self.scanner = None
            self.scanner_panel.set_running(False)
            return False
        step = self.scanner.start(self.effective_rate)
        self._apply_scan_step(step)
        self.scanner_panel.set_running(True)
        if config.stop_on_signal and self.audio is None:
            self.scanner_panel.set_status(
                "scanning \u2014 choose an audio mode to hear what it stops on"
            )
        return True

    def stop_scan(self) -> None:
        if self.scanner is not None:
            self.scanner.stop()
            self.scanner = None
        self.scanner_panel.set_running(False)
        self.scanner_panel.set_status("idle")
        if self.audio is not None:
            self.audio.set_offset(self._offset_spin.value() * 1e3)

    def skip_current(self) -> None:
        if self.scanner is not None:
            self._apply_scan_step(self.scanner.skip())

    def lock_out_current(self) -> None:
        if self.scanner is None:
            return
        frequency = self.scanner.lock_out_current()
        if frequency is None:
            self.scanner_panel.set_status("nothing to lock out")
            return
        self.settings.add_lockout(frequency)
        self._refresh_scan_lists()
        self._save_state()

    def lock_out(self, freq_hz: float) -> None:
        """Lock out a channel whether or not a scan is running."""
        self.settings.add_lockout(freq_hz)
        if self.scanner is not None:
            self.scanner.lock_out(freq_hz)
        self._refresh_scan_lists()
        self._save_state()

    def unlock(self, freq_hz: float) -> None:
        self.settings.remove_lockout(freq_hz)
        if self.scanner is not None:
            self.scanner.unlock(freq_hz)
        self._refresh_scan_lists()
        self._save_state()

    def clear_found(self) -> None:
        self.settings.clear_found()
        if self.scanner is not None:
            self.scanner.clear_hits()
        self._refresh_scan_lists()
        self._save_state()

    def _refresh_scan_lists(self) -> None:
        self.scanner_panel.set_found(self.settings.found)
        self.scanner_panel.set_lockout(self.settings.lockout)

    def _tune_to_found(self, freq_hz: float) -> None:
        """Tune a discovered channel, stopping the sweep so it stays put."""
        if self.scanner is not None:
            self.stop_scan()
        self._offset_spin.setValue(0.0)
        self._retune(freq_hz)

    def _save_found_to_memory(self, freq_hz: float) -> None:
        """Promote a scanner hit into the named memories, which are a separate list."""
        channel = self.settings.find_channel(freq_hz)
        default = f"{freq_hz / 1e6:.4f} MHz"
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save to memory", "Name for this memory:",
            QtWidgets.QLineEdit.EchoMode.Normal, channel.label or default if channel else default,
        )
        if not ok or not name.strip():
            return
        snapshot = self.current_snapshot()
        snapshot.freq_hz = freq_hz
        snapshot.offset_hz = 0.0
        self.settings.add_memory(name, snapshot)
        self._refresh_memories()
        self._save_state()
        self._status.showMessage(f"saved {name.strip()} to memories", 4000)

    def _apply_scan_step(self, step) -> None:
        """Carry out whatever the scanner asked for."""
        if step.new_hits:
            from datetime import datetime

            stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            for hit in step.new_hits:
                self.settings.record_found(hit.freq_hz, hit.level_dbfs, hit.snr_db, stamp)
            self._refresh_scan_lists()
            self._schedule_save()

        if step.action is ScanAction.TUNE and step.centre_hz is not None:
            if self.audio is not None:
                self.audio.set_offset(0.0)
            self._retune(step.centre_hz, from_scan=True)
        elif step.action is ScanAction.DWELL and step.dwell_hz is not None:
            # Listening is an audio offset, not a retune: the hit is already inside the
            # window being received, so nothing needs to move.
            if self.audio is not None:
                self.audio.set_offset(step.dwell_hz - self.source.center_freq)
                self.audio.reset()
            self._update_passband()

    def _scan_frame(self, freqs, dbfs) -> None:
        if self.scanner is None:
            return
        step = self.scanner.on_frame(self.source.center_freq, freqs, dbfs)
        self._apply_scan_step(step)
        scanner = self.scanner
        if scanner is None:
            return
        self.scanner_panel.set_progress(scanner.window_index, max(1, scanner.window_count))
        if scanner.dwell_hz is not None:
            text = f"listening {scanner.dwell_hz / 1e6:.4f} MHz"
        else:
            text = (
                f"sweeping {scanner.window_index + 1}/{scanner.window_count}"
                f"  pass {scanner.passes + 1}"
            )
        self.scanner_panel.set_status(f"{text}  —  {len(self.settings.found)} found")

    # -- controls ----------------------------------------------------------

    def _build_controls(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(bar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(self._build_tuning_row())
        outer.addWidget(self._build_display_row())
        outer.addWidget(self._build_audio_row())
        outer.addWidget(self._build_memory_row())
        return bar

    def _build_audio_row(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)

        row.addWidget(QtWidgets.QLabel("Audio"))
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItem("Off", "off")
        for name in MODES:
            self._mode_combo.addItem(name.upper(), name)
        self._mode_combo.setCurrentIndex(
            max(0, self._mode_combo.findData(self._initial_mode))
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row.addWidget(self._mode_combo)

        row.addWidget(QtWidgets.QLabel("BW"))
        self._bw_audio_combo = QtWidgets.QComboBox()
        self._bw_audio_combo.setToolTip("Channel filter width")
        self._bw_audio_combo.currentIndexChanged.connect(self._on_audio_bandwidth_changed)
        row.addWidget(self._bw_audio_combo)

        row.addWidget(QtWidgets.QLabel("Vol"))
        self._volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(int(self._initial_volume * 100))
        self._volume_slider.setFixedWidth(110)
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        row.addWidget(self._volume_slider)

        row.addWidget(QtWidgets.QLabel("Offset"))
        self._offset_spin = QtWidgets.QDoubleSpinBox()
        self._offset_spin.setDecimals(2)
        self._offset_spin.setSuffix(" kHz")
        self._offset_spin.setSingleStep(1.0)
        self._offset_spin.setToolTip(
            "Listen this far from the tuned centre, without moving the radio"
        )
        self._offset_spin.setValue(self._initial_offset / 1e3)
        self._offset_spin.valueChanged.connect(self._on_offset_changed)
        row.addWidget(self._offset_spin)

        self._squelch_check = QtWidgets.QCheckBox("Squelch")
        self._squelch_check.setChecked(self._initial_squelch is not None)
        self._squelch_check.toggled.connect(self._on_squelch_changed)
        row.addWidget(self._squelch_check)

        self._squelch_spin = QtWidgets.QDoubleSpinBox()
        self._squelch_spin.setRange(-160.0, 0.0)
        self._squelch_spin.setDecimals(0)
        self._squelch_spin.setSuffix(" dBFS")
        self._squelch_spin.setValue(
            self._initial_squelch if self._initial_squelch is not None else -100.0
        )
        self._squelch_spin.valueChanged.connect(self._on_squelch_changed)
        row.addWidget(self._squelch_spin)

        if not self._audio_ok:
            self._mode_combo.setEnabled(False)
            note = QtWidgets.QLabel("no audio device")
            note.setEnabled(False)
            row.addWidget(note)

        row.addSpacing(16)
        self.smeter = SMeter()
        row.addWidget(self.smeter)

        row.addStretch(1)
        self._update_offset_range()
        self._sync_squelch_enabled()
        self._refresh_bandwidths()
        return box

    def _build_tuning_row(self) -> QtWidgets.QWidget:
        """Frequency and sample rate, both driven by probed capabilities."""
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        caps = self.source.caps

        row.addWidget(QtWidgets.QLabel("Freq"))
        self._freq_spin = QtWidgets.QDoubleSpinBox()
        self._freq_spin.setDecimals(6)
        self._freq_spin.setSuffix(" MHz")
        lo = min((r.min_hz for r in caps.freq_ranges), default=0.0) / 1e6
        hi = max((r.max_hz for r in caps.freq_ranges), default=6000.0) / 1e6
        self._freq_spin.setRange(lo, hi)
        self._freq_spin.setValue(self.source.center_freq / 1e6)
        self._freq_spin.setKeyboardTracking(False)
        self._freq_spin.setToolTip(f"Tunable: {caps.describe_ranges()}")
        self._freq_spin.valueChanged.connect(
            lambda mhz: self._retune(mhz * 1e6, from_spin=True)
        )
        row.addWidget(self._freq_spin)

        row.addWidget(QtWidgets.QLabel("Step"))
        self._step_combo = QtWidgets.QComboBox()
        # 10 Hz and 100 Hz matter for SSB, where a few hundred hertz is the
        # difference between intelligible speech and a comedy voice.
        for label, hz in (
            ("10 Hz", 10.0), ("100 Hz", 100.0), ("500 Hz", 500.0),
            ("1 kHz", 1e3), ("5 kHz", 5e3), ("9 kHz", 9e3), ("10 kHz", 10e3),
            ("25 kHz", 25e3), ("100 kHz", 100e3), ("1 MHz", 1e6),
        ):
            self._step_combo.addItem(label, hz)
        index = self._step_combo.findData(self._initial_step_hz)
        self._step_combo.setCurrentIndex(index if index >= 0 else
                                         self._step_combo.findData(10e3))
        self._step_combo.currentIndexChanged.connect(self._on_step_changed)
        row.addWidget(self._step_combo)
        self._on_step_changed()

        if len(caps.sample_rates) > 1:
            row.addWidget(QtWidgets.QLabel("Rate"))
            self._rate_combo = QtWidgets.QComboBox()
            for rate in sorted(caps.sample_rates, reverse=True):
                self._rate_combo.addItem(f"{rate / 1e3:g} kS/s", rate)
            self._rate_combo.setCurrentText(f"{self.source.sample_rate / 1e3:g} kS/s")
            self._rate_combo.currentIndexChanged.connect(self._on_rate_changed)
            row.addWidget(self._rate_combo)
        else:
            self._rate_combo = None

        # No bandwidth control: measured 2026-09-23, the airspyhf driver reports
        # listBandwidths() == () and getBandwidthRange() == [], and setBandwidth() is
        # silently accepted while getBandwidth() stays 0.0. Built only if a driver
        # actually offers options.
        if caps.bandwidths:
            row.addWidget(QtWidgets.QLabel("BW"))
            self._bw_combo = QtWidgets.QComboBox()
            for bw in sorted(caps.bandwidths):
                self._bw_combo.addItem(f"{bw / 1e3:g} kHz", bw)
            self._bw_combo.currentIndexChanged.connect(
                lambda: self.source.set_bandwidth(self._bw_combo.currentData())
            )
            row.addWidget(self._bw_combo)
        else:
            self._bw_combo = None

        row.addWidget(QtWidgets.QLabel("Zoom"))
        self._zoom_combo = QtWidgets.QComboBox()
        for factor in ZOOM_FACTORS:
            self._zoom_combo.addItem(f"{factor}x", factor)
        self._zoom_combo.setCurrentText(f"{self.decimator.factor}x")
        self._zoom_combo.setToolTip(
            "Decimate the IQ stream: narrower span, proportionally finer resolution"
        )
        self._zoom_combo.currentIndexChanged.connect(self._on_zoom_changed)
        row.addWidget(self._zoom_combo)

        row.addSpacing(12)
        row.addWidget(self._build_device_controls())
        row.addStretch(1)
        return box

    def _build_memory_row(self) -> QtWidgets.QWidget:
        """Named presets, in the radio sense: recall a frequency/rate/zoom combination."""
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)

        row.addWidget(QtWidgets.QLabel("Memory"))
        self._memory_combo = QtWidgets.QComboBox()
        self._memory_combo.setMinimumWidth(220)
        self._memory_combo.activated.connect(self._on_memory_activated)
        row.addWidget(self._memory_combo)

        self._save_button = QtWidgets.QPushButton("Save\u2026")
        self._save_button.setToolTip("Store the current settings under a name")
        self._save_button.clicked.connect(self._on_save_memory)
        row.addWidget(self._save_button)

        self._delete_button = QtWidgets.QPushButton("Delete")
        self._delete_button.clicked.connect(self._on_delete_memory)
        row.addWidget(self._delete_button)

        row.addSpacing(20)
        row.addWidget(QtWidgets.QLabel("Record"))
        self._rec_audio_button = QtWidgets.QPushButton("Audio")
        self._rec_audio_button.setCheckable(True)
        self._rec_audio_button.setToolTip("Record demodulated audio to a WAV file")
        self._rec_audio_button.clicked.connect(self._on_record_audio)
        row.addWidget(self._rec_audio_button)

        self._rec_iq_button = QtWidgets.QPushButton("IQ")
        self._rec_iq_button.setCheckable(True)
        self._rec_iq_button.setToolTip("Record raw IQ (about 6 MB/s at 768 kS/s)")
        self._rec_iq_button.clicked.connect(self._on_record_iq)
        row.addWidget(self._rec_iq_button)

        self._rec_label = QtWidgets.QLabel("")
        self._rec_label.setMinimumWidth(260)
        row.addWidget(self._rec_label)

        row.addStretch(1)
        self._refresh_memories()
        return box

    def _build_display_row(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)

        row.addWidget(QtWidgets.QLabel("FFT"))
        self._fft_combo = QtWidgets.QComboBox()
        for size in FFT_SIZES:
            self._fft_combo.addItem(str(size), size)
        self._fft_combo.setCurrentText(str(self.analyzer.fft_size))
        self._fft_combo.currentIndexChanged.connect(self._on_fft_changed)
        row.addWidget(self._fft_combo)

        row.addWidget(QtWidgets.QLabel("Colour"))
        self._cmap_combo = QtWidgets.QComboBox()
        self._cmap_combo.addItems(COLORMAPS)
        self._cmap_combo.currentTextChanged.connect(self._on_colormap_changed)
        row.addWidget(self._cmap_combo)

        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("dBFS"))
        self._min_spin = self._make_level_spin(self._levels[0])
        self._max_spin = self._make_level_spin(self._levels[1])
        row.addWidget(self._min_spin)
        row.addWidget(QtWidgets.QLabel("to"))
        row.addWidget(self._max_spin)

        auto = QtWidgets.QPushButton("Auto")
        auto.setToolTip("Fit colour range to the visible history")
        auto.clicked.connect(self._on_auto_levels)
        row.addWidget(auto)

        self._peak_check = QtWidgets.QCheckBox("Peak hold")
        self._peak_check.setChecked(self._initial_peak_hold)
        self._peak_check.toggled.connect(self._on_peak_hold_toggled)
        self.spectrum.set_peak_hold(self._initial_peak_hold)
        row.addWidget(self._peak_check)

        row.addStretch(1)
        return bar

    def _build_device_controls(self) -> QtWidgets.QWidget:
        """Gain UI derived from probed capabilities."""
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        caps = self.source.caps

        self._agc_check = None
        if caps.has_agc:
            self._agc_check = QtWidgets.QCheckBox("AGC")
            self._agc_check.setChecked(getattr(self.source, "get_agc", lambda: False)())
            self._agc_check.toggled.connect(self._on_agc_toggled)
            row.addWidget(self._agc_check)

        for element in caps.gain_elements:
            row.addWidget(QtWidgets.QLabel(element.name))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(element.min_db, element.max_db)
            spin.setSingleStep(element.step_db or 1.0)
            spin.setSuffix(" dB")
            spin.setValue(self.source.get_gain(element.name))
            spin.valueChanged.connect(
                lambda value, name=element.name: self.source.set_gain(name, value)
            )
            row.addWidget(spin)

        if not caps.has_agc and not caps.gain_elements:
            label = QtWidgets.QLabel("no gain controls")
            label.setEnabled(False)
            row.addWidget(label)
        return box

    def _make_level_spin(self, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-200.0, 20.0)
        spin.setDecimals(0)
        spin.setSingleStep(5.0)
        spin.setValue(value)
        spin.valueChanged.connect(self._on_levels_changed)
        return spin

    # -- handlers ----------------------------------------------------------

    @property
    def effective_rate(self) -> float:
        """Sample rate after decimation: the span actually on screen."""
        return self.decimator.effective_rate(self.source.sample_rate)

    def _apply_geometry(self) -> None:
        history_s = self.waterfall.buffer.rows / float(self.fps)
        self.waterfall.set_geometry(self.source.center_freq, self.effective_rate, history_s)

    @property
    def step_hz(self) -> float:
        return float(self._step_combo.currentData() or 10e3)

    def _on_step_changed(self) -> None:
        self._freq_spin.setSingleStep(self.step_hz / 1e6)
        self._schedule_save()

    def nudge_frequency(self, steps: int) -> float:
        """Tune by `steps` of the selected step size, for a sideways swipe."""
        if not steps:
            return self.source.center_freq
        target = self.source.center_freq + steps * self.step_hz
        self._retune(target)
        return self.source.center_freq

    def fine_tune_limit(self) -> float:
        """Below this much movement, a tune is an adjustment rather than a change.

        One display bin wide. Smaller than that and the waterfall would shift by under a
        pixel, so its history is still honest; it is also small against any channel
        filter, so the demodulator's state remains valid.
        """
        return self.effective_rate / max(1, self.waterfall.buffer.cols)

    def _retune(self, hz: float, from_spin: bool = False, from_scan: bool = False) -> None:
        """Tune, dropping whatever described the old frequency.

        A large move invalidates everything: the ring, the waterfall history, the
        spectrum smoothing and the audio in flight all describe the previous tuning, and
        keeping any of it smears a stale signal across the new span.

        A *fine* move does not. Nudging 100 Hz to pitch an SSB voice would otherwise
        clear the display and interrupt the audio on every step, which defeats the point
        of tuning by ear -- so below `fine_tune_limit` the history and the audio are left
        running and only the axis moves.
        """
        if not from_scan and self.scanner is not None:
            # A manual tune means the user wants to stay here, so stop sweeping rather
            # than fighting them for the dial.
            self.stop_scan()

        previous = self.source.center_freq
        # Decided before tuning, so the source can be told whether to flush.
        target = self.source.caps.clamp_freq(float(hz))
        fine = abs(target - previous) < self.fine_tune_limit()
        actual = self.source.set_center_freq(hz, flush=not fine)

        if self.iq_recorder is not None and not fine:
            # The sidecar records one centre frequency, so a real retune would make the
            # file a lie about itself. A sub-bin nudge is not worth killing a capture.
            self.stop_iq_recording("frequency changed")
            self._rec_iq_button.setChecked(False)

        if not fine:
            self.spectrum.reset()
            self.waterfall.clear_history()
            if self.audio is not None:
                self.audio.reset()

        self.spectrum.set_center_marker(actual)
        self._apply_geometry()
        self._update_passband()

        if not from_spin or abs(actual - hz) > 1.0:
            # Clamped to a tunable range, or tuned from a click: reflect reality.
            self._freq_spin.blockSignals(True)
            self._freq_spin.setValue(actual / 1e6)
            self._freq_spin.blockSignals(False)
        if not from_scan:
            # Sweeping rewrites the frequency many times a second; that is not a
            # preference worth persisting.
            self._schedule_save()

    def _on_rate_changed(self) -> None:
        if self._rate_combo is None:
            return
        self.source.set_sample_rate(self._rate_combo.currentData())
        self.spectrum.reset()
        self.waterfall.clear_history()
        self._apply_geometry()
        self._update_offset_range()
        if self.audio is not None:
            # The chain's decimation and audio rate both derive from the sample rate.
            self.audio.restart()
        self._update_passband()
        self._schedule_save()

    def set_decimation(self, factor: int) -> None:
        """Change zoom. The span changes, so everything derived from it is dropped."""
        self.decimator = Decimator(factor)
        self.spectrum.reset()
        self.waterfall.clear_history()
        self._apply_geometry()
        self._rows_pushed = 0
        self._update_offset_range()

    def _on_zoom_changed(self) -> None:
        self.set_decimation(self._zoom_combo.currentData())
        self._sync_zoom_combo()
        self._schedule_save()

    def _sync_zoom_combo(self) -> None:
        self._zoom_combo.blockSignals(True)
        self._zoom_combo.setCurrentText(f"{self.decimator.factor}x")
        self._zoom_combo.blockSignals(False)

    # -- audio -------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode_combo.currentData() or "off"

    def _update_offset_range(self) -> None:
        """Offset can only reach the edges of what is actually being received."""
        limit = self.effective_rate / 2.0 / 1e3
        self._offset_spin.setRange(-limit, limit)

    def _sync_squelch_enabled(self) -> None:
        mode = self.mode
        capable = mode in MODE_SPECS and MODE_SPECS[mode].squelch_capable
        self._squelch_check.setEnabled(capable)
        self._squelch_spin.setEnabled(capable and self._squelch_check.isChecked())

    def _squelch_value(self) -> float | None:
        if not self._squelch_check.isChecked():
            return None
        return float(self._squelch_spin.value())

    def _update_passband(self) -> None:
        mode = self.mode
        if mode == "off" or mode not in MODE_SPECS:
            self.spectrum.clear_passband()
            return
        width = self._channel_bandwidth()
        centre = self.source.center_freq + self._offset_spin.value() * 1e3
        if mode in ("usb", "lsb"):
            # One-sided: shade only the sideband actually being demodulated.
            lower = centre if mode == "usb" else centre - width
            self.spectrum.set_passband(lower + width / 2.0, width)
        else:
            self.spectrum.set_passband(centre, width)

    def set_mode(self, mode: str) -> None:
        """Start, stop or switch demodulation.

        The display side always follows the chosen mode, even when audio cannot be
        started: seeing the channel width and offset on the spectrum is useful in its own
        right, and the status bar explains why nothing is audible.
        """
        if mode == "off":
            if self.audio is not None:
                self.audio.stop()
                self.audio = None
        elif not self._audio_ok:
            self._status.showMessage("no audio device available", 4000)
        elif self.audio is None:
            sink = AudioSink(
                self.source, mode=mode,
                offset_hz=self._offset_spin.value() * 1e3,
                volume=self._volume_slider.value() / 100.0,
                squelch_dbfs=self._squelch_value(),
            )
            try:
                sink.start()
                self.audio = sink
            except Exception as exc:
                self._status.showMessage(f"could not start audio: {exc}", 6000)
        else:
            try:
                self.audio.set_mode(mode)
            except Exception as exc:
                self._status.showMessage(f"could not switch mode: {exc}", 6000)

        self._sync_squelch_enabled()
        self._refresh_bandwidths()
        self._update_passband()

    def _on_mode_changed(self) -> None:
        self.set_mode(self.mode)
        self._schedule_save()

    def _on_volume_changed(self, value: int) -> None:
        if self.audio is not None:
            self.audio.set_volume(value / 100.0)
        self._schedule_save()

    def _on_offset_changed(self, khz: float) -> None:
        if self.audio is not None:
            self.audio.set_offset(khz * 1e3)
            self.audio.reset()
        self._update_passband()
        self._schedule_save()

    def _on_squelch_changed(self, *_args) -> None:
        self._sync_squelch_enabled()
        if self.audio is not None:
            self.audio.set_squelch(self._squelch_value())
        self._schedule_save()

    def _refresh_bandwidths(self) -> None:
        """Offer the widths that make sense for the current mode."""
        mode = self.mode
        self._bw_audio_combo.blockSignals(True)
        self._bw_audio_combo.clear()
        presets = BANDWIDTH_PRESETS.get(mode, ())
        for width in presets:
            label = f"{width / 1e3:g} kHz"
            self._bw_audio_combo.addItem(label, width)
        if presets:
            wanted = self._initial_bandwidth
            if wanted is None or wanted not in presets:
                wanted = MODE_SPECS[mode].bandwidth_hz if mode in MODE_SPECS else presets[0]
            index = self._bw_audio_combo.findData(wanted)
            self._bw_audio_combo.setCurrentIndex(index if index >= 0 else 0)
        self._bw_audio_combo.blockSignals(False)
        self._bw_audio_combo.setEnabled(bool(presets))

    def bandwidth_hz(self) -> float | None:
        data = self._bw_audio_combo.currentData()
        return float(data) if data is not None else None

    def _on_audio_bandwidth_changed(self) -> None:
        width = self.bandwidth_hz()
        if width is None:
            return
        if self.audio is not None:
            self.audio.set_bandwidth(width)
        self._update_passband()
        self._schedule_save()

    # -- S-meter -----------------------------------------------------------

    def _channel_bandwidth(self) -> float:
        width = self.bandwidth_hz()
        if width is not None:
            return width
        mode = self.mode
        return MODE_SPECS[mode].bandwidth_hz if mode in MODE_SPECS else 9e3

    def _update_smeter(self, dbfs, freqs) -> None:
        """Signal level in the channel, plus an SNR estimate.

        When audio is running the level comes from the demodulator, measured after the
        channel filter, which is exactly the signal being listened to. With audio off it
        is integrated from the displayed spectrum over the same width, so the meter still
        works as a tuning aid.
        """
        linear = np.power(10.0, dbfs / 10.0)
        noise_per_bin = float(np.median(linear))
        centre = self.source.center_freq + self._offset_spin.value() * 1e3
        half = self._channel_bandwidth() / 2.0
        mask = np.abs(freqs - centre) <= half
        bins = int(np.count_nonzero(mask))
        if bins == 0:
            self.smeter.set_level(None)
            return

        noise_in_channel = noise_per_bin * bins
        spectrum_power = float(linear[mask].sum())

        # The displayed level prefers the demodulator's own measurement, which is taken
        # after the channel filter and so is exactly what is being listened to.
        if self.audio is not None and self.audio.channel_dbfs is not None:
            level_db = float(self.audio.channel_dbfs)
        else:
            level_db = 10.0 * np.log10(spectrum_power + 1e-30)

        # SNR is always computed from the spectrum, on both sides of the ratio. Mixing
        # the chain's level with a spectrum-derived noise figure compares two different
        # normalisations and produced readings like "-203 dB". Clamped at zero because
        # a channel at or below the noise floor has no measurable SNR -- 0 dB states
        # "indistinguishable from the noise" rather than dressing noise up as a number.
        excess = spectrum_power - noise_in_channel
        if excess <= 0.0 or noise_in_channel <= 0.0:
            snr_db = 0.0
        else:
            snr_db = min(100.0, max(0.0, 10.0 * np.log10(excess / noise_in_channel)))
        self.smeter.set_level(level_db, snr_db)

    # -- recording ---------------------------------------------------------

    def _recording_active(self) -> bool:
        return (self.audio_recorder is not None) or (self.iq_recorder is not None)

    def start_audio_recording(self) -> Path | None:
        """Record demodulated audio. Needs a running demodulator to record."""
        if self.audio_recorder is not None:
            return self.audio_recorder.path
        if self.audio is None:
            self._status.showMessage("choose an audio mode before recording audio", 4000)
            return None
        name = timestamp_name(self.source.center_freq, ".wav", self.mode)
        recorder = AudioRecorder(self.recordings_dir / name, self.audio.audio_rate)
        try:
            recorder.start()
        except OSError as exc:
            self._status.showMessage(f"could not start recording: {exc}", 6000)
            return None
        self.audio.on_audio = recorder.submit
        self.audio_recorder = recorder
        self._status.showMessage(f"recording audio to {recorder.path}", 5000)
        return recorder.path

    def stop_audio_recording(self, reason: str | None = None) -> None:
        """Stop and report. `reason` is folded into the same message as the file path, so
        neither fact is lost to the other overwriting the status bar."""
        if self.audio_recorder is None:
            return
        if self.audio is not None:
            self.audio.on_audio = None
        self.audio_recorder.stop()
        prefix = f"{reason} - " if reason else ""
        self._status.showMessage(f"{prefix}saved {self.audio_recorder.path}", 8000)
        self.audio_recorder = None

    def start_iq_recording(self) -> Path | None:
        if self.iq_recorder is not None:
            return self.iq_recorder.path
        name = timestamp_name(self.source.center_freq, ".cf32")
        recorder = IQRecorder(self.recordings_dir / name, self.source)
        try:
            recorder.start()
        except OSError as exc:
            self._status.showMessage(f"could not start recording: {exc}", 6000)
            return None
        self.iq_recorder = recorder
        self._status.showMessage(f"recording IQ to {recorder.path}", 5000)
        return recorder.path

    def stop_iq_recording(self, reason: str | None = None) -> None:
        if self.iq_recorder is None:
            return
        self.iq_recorder.stop()
        prefix = f"{reason} - " if reason else ""
        self._status.showMessage(f"{prefix}saved {self.iq_recorder.path}", 8000)
        self.iq_recorder = None

    def _on_record_audio(self, checked: bool) -> None:
        if checked:
            if self.start_audio_recording() is None:
                self._rec_audio_button.setChecked(False)
        else:
            self.stop_audio_recording()

    def _on_record_iq(self, checked: bool) -> None:
        if checked:
            if self.start_iq_recording() is None:
                self._rec_iq_button.setChecked(False)
        else:
            self.stop_iq_recording()

    def _update_recording_label(self) -> None:
        parts = []
        for tag, rec in (("audio", self.audio_recorder), ("IQ", self.iq_recorder)):
            if rec is None:
                continue
            parts.append(
                f"{tag} {rec.seconds_recorded:.0f}s "
                f"{rec.bytes_written / 1024**2:.1f} MB"
                + (f" drop {rec.dropped_blocks}" if rec.dropped_blocks else "")
            )
            if not rec.running and rec.stopped_reason:
                # Stopped itself: hit the size limit, or the disk complained.
                if tag == "audio":
                    self.stop_audio_recording(rec.stopped_reason)
                    self._rec_audio_button.setChecked(False)
                else:
                    self.stop_iq_recording(rec.stopped_reason)
                    self._rec_iq_button.setChecked(False)
        self._rec_label.setText("   ".join(parts))

    # -- memories ----------------------------------------------------------

    def current_snapshot(self) -> Snapshot:
        """The settings worth restoring or storing under a name."""
        explicit = self._levels_explicit
        return Snapshot(
            freq_hz=self.source.center_freq,
            sample_rate=self.source.sample_rate,
            decimation=self.decimator.factor,
            fft_size=self.analyzer.fft_size,
            colormap=self._cmap_combo.currentText(),
            min_db=self._levels[0] if explicit else None,
            max_db=self._levels[1] if explicit else None,
            agc=self._agc_check.isChecked() if self._agc_check is not None else False,
            peak_hold=self._peak_check.isChecked(),
            mode=self.mode,
            volume=self._volume_slider.value() / 100.0,
            offset_hz=self._offset_spin.value() * 1e3,
            squelch_dbfs=self._squelch_value(),
            bandwidth_hz=self.bandwidth_hz(),
            step_hz=self.step_hz,
        )

    def apply_snapshot(self, snap: Snapshot) -> None:
        """Put the receiver back into a stored state.

        Rate first: changing it restarts the stream, which would otherwise undo the
        frequency and zoom set afterwards.
        """
        if self._rate_combo is not None and snap.sample_rate != self.source.sample_rate:
            self.source.set_sample_rate(snap.sample_rate)
            self._rate_combo.blockSignals(True)
            self._rate_combo.setCurrentText(f"{self.source.sample_rate / 1e3:g} kS/s")
            self._rate_combo.blockSignals(False)

        if snap.fft_size != self.analyzer.fft_size and snap.fft_size in FFT_SIZES:
            self.analyzer = SpectrumAnalyzer(fft_size=snap.fft_size)
            self._fft_combo.blockSignals(True)
            self._fft_combo.setCurrentText(str(snap.fft_size))
            self._fft_combo.blockSignals(False)

        if snap.colormap in COLORMAPS:
            self._cmap_combo.blockSignals(True)
            self._cmap_combo.setCurrentText(snap.colormap)
            self._cmap_combo.blockSignals(False)
            self.waterfall.set_colormap(snap.colormap)

        factor = snap.decimation if snap.decimation in ZOOM_FACTORS else 1
        self.set_decimation(factor)
        self._sync_zoom_combo()

        if snap.min_db is None or snap.max_db is None:
            # Was auto-fitted rather than chosen, so fit again for today's conditions.
            self._auto_pending = True
            self._levels_explicit = False
        else:
            self._levels_explicit = True
            self._set_levels(snap.min_db, snap.max_db)

        self._peak_check.setChecked(snap.peak_hold)
        if self._agc_check is not None:
            self._agc_check.setChecked(snap.agc)

        self._volume_slider.blockSignals(True)
        self._volume_slider.setValue(int(snap.volume * 100))
        self._volume_slider.blockSignals(False)
        self._squelch_check.blockSignals(True)
        self._squelch_check.setChecked(snap.squelch_dbfs is not None)
        self._squelch_check.blockSignals(False)
        if snap.squelch_dbfs is not None:
            self._squelch_spin.blockSignals(True)
            self._squelch_spin.setValue(snap.squelch_dbfs)
            self._squelch_spin.blockSignals(False)
        self._update_offset_range()
        self._offset_spin.blockSignals(True)
        self._offset_spin.setValue(snap.offset_hz / 1e3)
        self._offset_spin.blockSignals(False)

        self._retune(snap.freq_hz)

        self._initial_bandwidth = snap.bandwidth_hz
        index = self._step_combo.findData(snap.step_hz)
        if index >= 0:
            self._step_combo.blockSignals(True)
            self._step_combo.setCurrentIndex(index)
            self._step_combo.blockSignals(False)
            self._on_step_changed()
        wanted = snap.mode if snap.mode in MODES else "off"
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(max(0, self._mode_combo.findData(wanted)))
        self._mode_combo.blockSignals(False)
        self.set_mode(wanted)

    def _refresh_memories(self) -> None:
        self._memory_combo.blockSignals(True)
        self._memory_combo.clear()
        self._memory_combo.addItem("\u2014 recall \u2014", None)
        for name in self.settings.names():
            self._memory_combo.addItem(name, name)
        self._memory_combo.blockSignals(False)
        has_any = bool(self.settings.names())
        self._memory_combo.setEnabled(has_any)
        self._delete_button.setEnabled(has_any)

    def save_memory(self, name: str) -> bool:
        """Store the current settings. Returns True if an existing name was replaced."""
        replaced = self.settings.add_memory(name, self.current_snapshot())
        self._refresh_memories()
        index = self._memory_combo.findData(self.settings.get_memory(name).name)
        if index >= 0:
            self._memory_combo.blockSignals(True)
            self._memory_combo.setCurrentIndex(index)
            self._memory_combo.blockSignals(False)
        self._save_state()
        return replaced

    def recall_memory(self, name: str) -> bool:
        memory = self.settings.get_memory(name)
        if memory is None:
            return False
        self.apply_snapshot(memory.snapshot)
        self._status.showMessage(f"recalled {memory.name}", 3000)
        self._schedule_save()
        return True

    def delete_memory(self, name: str) -> bool:
        removed = self.settings.remove_memory(name)
        if removed:
            self._refresh_memories()
            self._save_state()
        return removed

    def _on_memory_activated(self, index: int) -> None:
        name = self._memory_combo.itemData(index)
        if name:
            self.recall_memory(name)

    def _on_save_memory(self) -> None:
        suggestion = self.current_snapshot().describe()
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save memory", "Name for this memory:",
            QtWidgets.QLineEdit.EchoMode.Normal, suggestion,
        )
        if not ok or not name.strip():
            return
        if self.settings.get_memory(name) is not None:
            answer = QtWidgets.QMessageBox.question(
                self, "Replace memory?",
                f'"{name.strip()}" already exists. Replace it?',
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.save_memory(name)

    def _on_delete_memory(self) -> None:
        name = self._memory_combo.currentData()
        if not name:
            self._status.showMessage("pick a memory to delete first", 3000)
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete memory?", f'Delete "{name}"?',
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.delete_memory(name)

    # -- persistence -------------------------------------------------------

    def _schedule_save(self) -> None:
        # Defensive about the timer: control handlers fire while the widgets are still
        # being built, which is before the timer exists.
        if self._persist and getattr(self, "_save_timer", None) is not None:
            self._save_timer.start()

    def _save_state(self) -> None:
        if not self._persist:
            return
        self.settings.last = self.current_snapshot()
        try:
            self.settings.save()
        except OSError as exc:
            # Losing a preference must never take the radio down with it.
            self._status.showMessage(f"could not save settings: {exc}", 5000)

    def _set_levels(self, low: float, high: float) -> None:
        self._levels = (float(low), float(high))
        self.waterfall.set_levels(low, high)
        self.spectrum.set_levels(low, high)
        for spin, value in ((self._min_spin, low), (self._max_spin, high)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _on_levels_changed(self) -> None:
        low, high = self._min_spin.value(), self._max_spin.value()
        if high <= low:
            return
        self._levels_explicit = True   # chosen, so worth restoring next launch
        self._auto_pending = False
        self._set_levels(low, high)
        self._schedule_save()

    def _on_auto_levels(self) -> None:
        low, high = self.waterfall.auto_levels()
        self._set_levels(low, high)
        self._levels_explicit = False  # fitted to today's signal, not a preference
        self._schedule_save()

    def _on_colormap_changed(self, name: str) -> None:
        self.waterfall.set_colormap(name)
        self._schedule_save()

    def _on_peak_hold_toggled(self, enabled: bool) -> None:
        self.spectrum.set_peak_hold(enabled)
        self._schedule_save()

    def _on_agc_toggled(self, enabled: bool) -> None:
        self.source.set_agc(enabled)
        self._schedule_save()

    def _on_fft_changed(self) -> None:
        self.analyzer = SpectrumAnalyzer(fft_size=self._fft_combo.currentData())
        self.spectrum.reset()
        self.waterfall.clear_history()
        self._rows_pushed = 0
        self._schedule_save()

    def _frame_request(self) -> int:
        """Input samples to pull for one frame.

        Deep zoom needs `fft_size * factor` input samples for a single segment, so the
        Welch segment count is traded away as zoom increases rather than asking the ring
        for more than it holds.
        """
        factor = self.decimator.factor
        budget = int(FRAME_INPUT_BUDGET * self.source.sample_rate)
        affordable = max(self.analyzer.fft_size, budget // factor)
        wanted = min(self.analyzer.samples_wanted(), affordable)
        return self.decimator.input_for_output(wanted)

    def _on_frame(self) -> None:
        iq = self.source.read_latest(self._frame_request())
        if iq.size < self.decimator.input_for_output(self.analyzer.fft_size):
            self._status.showMessage("waiting for samples...")
            return

        decimated = self.decimator.process(iq)
        if decimated.size < self.analyzer.fft_size:
            self._status.showMessage("waiting for samples...")
            return

        dbfs = self.analyzer.psd_dbfs(decimated)
        freqs = self.analyzer.freq_axis(self.source.center_freq, self.effective_rate)
        self.spectrum.update_spectrum(freqs, dbfs)
        self.waterfall.push(dbfs)
        self._update_smeter(dbfs, freqs)
        self._scan_frame(freqs, dbfs)
        self._rows_pushed += 1

        # One-shot auto-range once there is enough history to be representative.
        if self._auto_pending and self._rows_pushed >= 20:
            self._auto_pending = False
            self._on_auto_levels()

        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_mark >= 1.0:
            self._measured_fps = self._frames / (now - self._fps_mark)
            self._frames = 0
            self._fps_mark = now
            self._update_status(dbfs)
            if self._recording_active():
                self._update_recording_label()

    def _update_status(self, dbfs) -> None:
        stats = getattr(self.source, "stats", {})
        span = self.effective_rate
        self._status.showMessage(
            f"{self.source.center_freq / 1e6:.4f} MHz  |  "
            f"{self.source.sample_rate / 1e3:.0f} kS/s  |  "
            f"zoom {self.decimator.factor}x  |  "
            f"span {span / 1e3:.1f} kHz  |  "
            f"{span / self.analyzer.fft_size:.1f} Hz/bin  |  "
            f"FFT {self.analyzer.fft_size}  |  "
            f"{self._measured_fps:.1f} FPS  |  "
            f"peak {float(dbfs.max()):.1f} dBFS  "
            f"floor {float(dbfs.min()):.1f} dBFS  |  "
            f"ovf {stats.get('overflows', 0)}  "
            f"to {stats.get('timeouts', 0)}  "
            f"err {stats.get('errors', 0)}"
            + self._audio_status()
        )

    def _audio_status(self) -> str:
        if self.audio is None:
            return ""
        a = self.audio.stats
        text = (f"  |  {self.mode.upper()} {a['audio_rate'] / 1e3:.1f} kHz"
                f"  agc x{a['agc_gain']:.0f}"
                f"  ur {int(a['underrun_samples'])}")
        if a["lost_iq"]:
            text += f"  lost {int(a['lost_iq'])}"
        if a["muted_blocks"]:
            text += f"  sq {int(a['muted_blocks'])}"
        return text

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        self._timer.stop()
        self._save_timer.stop()
        self._save_state()   # immediately, not debounced: there is no later
        if self.scanner is not None:
            self.scanner.stop()
            self.scanner = None
        self.stop_audio_recording()
        self.stop_iq_recording()
        if self.audio is not None:
            self.audio.stop()
            self.audio = None
        self.source.stop()
        super().closeEvent(event)


def run(source: IQSource, **kwargs) -> int:
    """Start the Qt app against an already-configured source."""
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Conventional Qt app identity. Note this does *not* rename the macOS Dock tile: the
    # desktop shortcut execs a system interpreter, so macOS attributes the running process
    # to Python.framework and the Dock says "Python". Fixing that needs the interpreter
    # bundled inside the .app, which is more than a launcher shortcut warrants.
    app.setApplicationName("RGC SDR")
    app.setApplicationDisplayName("RGC SDR")
    icon_path = Path(__file__).resolve().parents[3] / "assets" / "icon.png"
    if icon_path.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    source.start()
    window = MainWindow(source, **kwargs)
    window.resize(1280, 800)
    window.show()
    try:
        return app.exec()
    finally:
        source.stop()
