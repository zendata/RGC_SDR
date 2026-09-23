"""Main window: spectrum above waterfall, driven by a frame timer.

Device controls are generated from `DeviceCaps`, not hardcoded, so a radio with gain
elements (HackRF) grows sliders while the Airspy HF+ -- which reports none -- shows only
its AGC toggle. See PLANNING.md sections 3 and 4.
"""

from __future__ import annotations

import pathlib
import time

import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from ..device.source import IQSource
from ..dsp.spectrum import SpectrumAnalyzer
from .spectrum_view import SpectrumView
from .waterfall import COLORMAPS, WaterfallView

FFT_SIZES = (1024, 2048, 4096, 8192, 16384)


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
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.source = source
        self.fps = max(1, int(fps))
        self.analyzer = SpectrumAnalyzer(fft_size=fft_size)
        # `levels=None` means "fit to whatever this antenna is actually receiving once a
        # little history exists". Signal levels vary by tens of dB with antenna and band
        # (measured: -120..-85 dBFS on this setup, but -47 dBFS with a strong carrier), so
        # a fixed default range renders the waterfall uniformly blank as often as not.
        self._auto_pending = levels is None
        self._levels = levels if levels is not None else (-120.0, -60.0)
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
        self._apply_geometry()
        self.spectrum.set_center_marker(source.center_freq)

        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._on_frame)
        self._timer.start(int(1000 / self.fps))

    # -- controls ----------------------------------------------------------

    def _build_controls(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(bar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(self._build_tuning_row())
        outer.addWidget(self._build_display_row())
        return bar

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
        for label, hz in (
            ("1 kHz", 1e3), ("5 kHz", 5e3), ("9 kHz", 9e3), ("10 kHz", 10e3),
            ("25 kHz", 25e3), ("100 kHz", 100e3), ("1 MHz", 1e6),
        ):
            self._step_combo.addItem(label, hz)
        self._step_combo.setCurrentText("10 kHz")
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

        row.addSpacing(12)
        row.addWidget(self._build_device_controls())
        row.addStretch(1)
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
        self._cmap_combo.currentTextChanged.connect(self.waterfall.set_colormap)
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
        self._peak_check.setChecked(True)
        self._peak_check.toggled.connect(self.spectrum.set_peak_hold)
        row.addWidget(self._peak_check)

        row.addStretch(1)
        return bar

    def _build_device_controls(self) -> QtWidgets.QWidget:
        """Gain UI derived from probed capabilities."""
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        caps = self.source.caps

        if caps.has_agc:
            check = QtWidgets.QCheckBox("AGC")
            check.setChecked(getattr(self.source, "get_agc", lambda: False)())
            check.toggled.connect(lambda on: self.source.set_agc(on))
            row.addWidget(check)

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

    def _apply_geometry(self) -> None:
        history_s = self.waterfall.buffer.rows / float(self.fps)
        self.waterfall.set_geometry(self.source.center_freq, self.source.sample_rate, history_s)

    def _on_step_changed(self) -> None:
        self._freq_spin.setSingleStep(self._step_combo.currentData() / 1e6)

    def _retune(self, hz: float, from_spin: bool = False) -> None:
        """Tune, then drop everything that described the old frequency.

        The ring, the waterfall history, the smoothing state and the peak hold all
        describe the previous tuning; keeping any of them smears stale signal across the
        new span.
        """
        actual = self.source.set_center_freq(hz)
        self.spectrum.reset()
        self.spectrum.set_center_marker(actual)
        self.waterfall.clear_history()
        self._apply_geometry()
        if not from_spin or abs(actual - hz) > 1.0:
            # Clamped to a tunable range, or tuned from a click: reflect reality.
            self._freq_spin.blockSignals(True)
            self._freq_spin.setValue(actual / 1e6)
            self._freq_spin.blockSignals(False)

    def _on_rate_changed(self) -> None:
        if self._rate_combo is None:
            return
        self.source.set_sample_rate(self._rate_combo.currentData())
        self.spectrum.reset()
        self.waterfall.clear_history()
        self._apply_geometry()

    def _on_levels_changed(self) -> None:
        low, high = self._min_spin.value(), self._max_spin.value()
        if high <= low:
            return
        self._levels = (low, high)
        self.waterfall.set_levels(low, high)
        self.spectrum.set_levels(low, high)

    def _on_auto_levels(self) -> None:
        low, high = self.waterfall.auto_levels()
        self.spectrum.set_levels(low, high)
        for spin, value in ((self._min_spin, low), (self._max_spin, high)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._levels = (low, high)

    def _on_fft_changed(self) -> None:
        self.analyzer = SpectrumAnalyzer(fft_size=self._fft_combo.currentData())
        self.spectrum.reset()
        self.waterfall.clear_history()
        self._rows_pushed = 0

    def _on_frame(self) -> None:
        iq = self.source.read_latest(self.analyzer.samples_wanted())
        if iq.size < self.analyzer.fft_size:
            self._status.showMessage("waiting for samples...")
            return

        dbfs = self.analyzer.psd_dbfs(iq)
        freqs = self.analyzer.freq_axis(self.source.center_freq, self.source.sample_rate)
        self.spectrum.update_spectrum(freqs, dbfs)
        self.waterfall.push(dbfs)
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

    def _update_status(self, dbfs) -> None:
        stats = getattr(self.source, "stats", {})
        self._status.showMessage(
            f"{self.source.center_freq / 1e6:.4f} MHz  |  "
            f"{self.source.sample_rate / 1e3:.0f} kS/s  |  "
            f"FFT {self.analyzer.fft_size}  |  "
            f"{self._measured_fps:.1f} FPS  |  "
            f"peak {float(dbfs.max()):.1f} dBFS  "
            f"floor {float(dbfs.min()):.1f} dBFS  |  "
            f"ovf {stats.get('overflows', 0)}  "
            f"to {stats.get('timeouts', 0)}  "
            f"err {stats.get('errors', 0)}"
        )

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        self._timer.stop()
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
    icon_path = pathlib.Path(__file__).resolve().parents[3] / "assets" / "icon.png"
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
