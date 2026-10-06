"""Main window: spectrum above waterfall, driven by a frame timer.

Device controls are generated from `DeviceCaps`, not hardcoded, so a radio with gain
elements (HackRF) grows sliders while the Airspy HF+ -- which reports none -- shows only
its AGC toggle. See PLANNING.md sections 3 and 4.
"""

from __future__ import annotations

import dataclasses

import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from ..audio import AudioSink, audio_available
from ..device.profiles import (
    PROFILES,
    availability,
    profile_for,
    starting_frequency,
)
from ..device.playback import PLAYBACK_DRIVER
from ..device.source import IQSource
from ..dsp.decimate import Decimator
from ..dsp.demod import BANDWIDTH_PRESETS, CW_PITCHES, MODE_SPECS, MODES
from ..recorder import DEFAULT_DIR, AudioRecorder, IQRecorder, timestamp_name
from ..scanner import ScanAction, ScanConfig, Scanner
from ..dsp.spectrum import SpectrumAnalyzer
from ..dsp.zerobeat import DEFAULT_FFT as ZEROBEAT_FFT
from ..dsp.zerobeat import measure_carrier
from ..settings import RadioSettings, Settings, Snapshot
from .scanner_panel import ScannerPanel
from ..decoding import DecodeWorker
from .decoder_panel import DecoderPanel
from .map_window import MapWindow
from ..targets import TargetStore
from .freq_display import FrequencyDisplay
from .function_panel import FunctionPanel
from .radio_display import RadioDisplay
from ..dsp.modulate import TX_MODES
from ..transmit import TX_IQ_RATE, TX_TIMEOUT_S, RadioTransmitter, Transmitter
from ..repeater import CTCSS_TONES, MINUS, PLUS, SIMPLEX, band_for, tx_frequency
from ..dsp.tones import DCS_CODES
from .smeter import SMeter
from .spectrum_view import SpectrumView
from .waterfall import COLORMAPS, WaterfallView

FFT_SIZES = (1024, 2048, 4096, 8192, 16384)
#: How far either side of the current tuning zero-beat will look for a carrier.
ZEROBEAT_SEARCH_HZ = 500.0
#: How often it re-measures while the button is held.
ZEROBEAT_INTERVAL_MS = 150
#: Below this much error it stops nudging: further correction is inaudible and would
#: only jitter the tuning.
ZEROBEAT_DEADBAND_HZ = 3.0
#: Decimation factors offered as a zoom control. Powers of two, matching Decimator.
ZOOM_FACTORS = (1, 2, 4, 8, 16, 32)

#: Shown in the title bar, with the radio in use after it.
APP_TITLE = "VK3RQ Super SDR"
#: Background of radios detected now, in the radio list (VK3RQ, 2026-10-06).
DETECTED_COLOUR = "#FFE45C"
#: Seconds a status notice without its own timeout stays before the frame line returns.
NOTICE_HOLD_S = 8.0

#: A transceiver scope's width in points (IC-705: 475, measured) and line rate.
SCOPE_POINTS = 475
from ..device.icom import SCOPE_LINES_PER_S, power_percent, s_meter_text, swr_value  # noqa: E402
from ..device import civ  # noqa: E402

#: The IC-705 modes offered in the Mode list (D-STAR left out for now).
TRANSCEIVER_MODES = ("lsb", "usb", "am", "cw", "rtty", "fm", "wfm")
#: The same modes, named the app's way and the radio's way.
_TO_RADIO_MODE = {"nbfm": "fm", "wbfm": "wfm"}
_TO_SDR_MODE = {"fm": "nbfm", "wfm": "wbfm"}


def scope_level_db(amplitudes: np.ndarray) -> np.ndarray:
    """Icom scope amplitudes (0-160) on a dB-like axis, 0.5 dB a step, top at 0 dB.

    Icom does not publish the scale; 160 steps over the scope's 80 dB display is the
    working assumption, to be checked against a known signal level.
    """
    return (amplitudes.astype(np.float32) - 160.0) * 0.5
#: Fraction of the ring a single frame may consume, so deep zoom cannot starve itself.
FRAME_INPUT_BUDGET = 0.6


class DeviceCombo(QtWidgets.QComboBox):
    """SDR selector that refreshes connection status each time it opens.

    Radios get plugged in and out while the application runs, so a status worked out at
    startup would go stale.
    """

    aboutToShow = QtCore.pyqtSignal()

    def showPopup(self) -> None:  # noqa: N802  (Qt naming)
        self.aboutToShow.emit()
        super().showPopup()


def _open_soapy(driver: str, centre_hz: float) -> IQSource:
    from ..device.profiles import profile_for

    profile = profile_for(driver)
    if profile is not None and profile.kind == "transceiver":
        from ..device.icom import open_ic705

        return open_ic705(driver)     # opens wherever the radio's own dial is
    from ..device.source import SoapyIQSource

    return SoapyIQSource(driver=driver, center_freq=centre_hz)


class _SpaceTogglesTransmit(QtCore.QObject):
    """The space bar toggles TX anywhere in the window, as the TX button does.

    An application-wide filter rather than a shortcut, because the focused widget sees a
    key first: space would otherwise click whichever button or tick box had focus. Left
    alone only while text is being typed -- a memory name, say -- or in another window.
    """

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self._window = window

    def _typing(self) -> bool:
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            return True
        if isinstance(focus, QtWidgets.QLineEdit):
            # A spin box's editor is a line edit too, but nobody types spaces into one.
            return not isinstance(focus.parentWidget(), QtWidgets.QAbstractSpinBox)
        return False

    def eventFilter(self, obj, event) -> bool:  # noqa: N802  (Qt naming)
        if event.type() not in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease):
            return False
        if event.key() != QtCore.Qt.Key.Key_Space or event.modifiers() not in (
                QtCore.Qt.KeyboardModifier.NoModifier,):
            return False
        if QtWidgets.QApplication.activeWindow() is not self._window or self._typing():
            return False
        if event.type() == QtCore.QEvent.Type.KeyPress and not event.isAutoRepeat():
            self._window._tx_button.toggle()
        return True          # swallow presses, repeats and releases alike


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
        snap: bool = False,
        pitch_hz: float = 500.0,
        stereo: bool = True,
        enable_audio: bool = True,
        recordings_dir=None,
        settings: Settings | None = None,
        source_factory=None,
        availability_fn=None,
        transmitter_factory=None,
        auto_calibrate: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.source = source
        #: An SDR's waterfall width; a transceiver's scope sets its own.
        self._sdr_waterfall_bins = int(waterfall_bins)
        self._scope_geometry: tuple[float, float] | None = None
        #: Builds a Transmitter for a mode. Injectable so tests never open the microphone.
        self._transmitter_factory = transmitter_factory or self._make_transmitter
        self.transmitter: Transmitter | None = None
        from ..audio import DEFAULT_TX_AUDIO_LEVEL

        #: Mac microphone level sent to a transceiver for TX (0-1), saved per radio.
        self._tx_audio_level = DEFAULT_TX_AUDIO_LEVEL
        from ..dsp.modulate import MIC_GAIN_DB

        #: Microphone gain (dB) for an SDR's own modulator, saved per radio.
        self._tx_mic_gain_db = MIC_GAIN_DB
        #: This radio's measured frequency error (ppm), None until calibrated.
        self._ppm_measured: float | None = None
        #: Measure a new radio's error when it is first connected (the app does; tests
        #: do not, so stand-in radios are never retuned behind their backs).
        self._auto_calibrate = bool(auto_calibrate)
        self._calibration_tried: set[str] = set()
        #: A transceiver's received audio on the Mac (RadioAudio), while one is in use.
        #: Injectable so tests never open a sound device.
        self.radio_audio = None
        self._radio_audio_factory = None
        #: Transmit gains for the current radio, by stage. Start at each stage's minimum:
        #: a low first transmission, raised by the user as wanted.
        self._tx_gains: dict[str, float] = {}
        #: Opens a radio by Soapy driver key. Injectable so tests can switch devices
        #: without hardware.
        self._source_factory = source_factory or _open_soapy
        self._availability_fn = availability_fn or availability
        #: The last look for attached radios (it takes about 0.6 s, so it is not repeated
        #: on every retune): refreshed when the radio list opens, when radios change, and
        #: when no radio known to be there reaches a requested frequency.
        self._detected: list = []
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
        self._initial_snap = bool(snap)
        self._initial_pitch = float(pitch_hz)
        self._initial_stereo = bool(stereo)
        self._initial_bandwidth = bandwidth_hz
        self.recordings_dir = Path(recordings_dir) if recordings_dir else DEFAULT_DIR
        self.audio_recorder: AudioRecorder | None = None
        self.iq_recorder: IQRecorder | None = None
        #: The radio (key, frequency) to go back to when a recording stops playing.
        self._radio_before_playback: tuple[str, float] | None = None
        self.scanner: Scanner | None = None
        self.decode_worker: DecodeWorker | None = None
        #: Everything the decoders have located, for the map window.
        self.targets = TargetStore()
        self.map_window: MapWindow | None = None
        # Audio is optional: without a usable device the rest of the app still works.
        self._audio_ok = bool(enable_audio) and audio_available()
        self.audio: AudioSink | None = None
        self._audio_problem = ""          # why a chosen mode is silent, for the status bar
        self._rows_pushed = 0
        self._frames = 0
        self._fps_mark = time.perf_counter()
        self._measured_fps = 0.0

        levels = self._levels
        caps = source.caps
        self.setWindowTitle(f"{APP_TITLE} \u2014 {caps.label or caps.driver}")

        self.spectrum = SpectrumView()
        self.waterfall = WaterfallView(
            rows=history_rows, cols=waterfall_bins, colormap=colormap, levels=levels
        )
        self.spectrum.set_levels(*levels)
        self.waterfall.setXLink(self.spectrum)  # one shared frequency axis
        self.spectrum.frequencySelected.connect(self._retune)
        self.waterfall.frequencySelected.connect(self._retune)
        # Spectrum only: tuning from the waterfall while reading back through history
        # is more confusing than useful.
        self.spectrum.frequencyNudged.connect(self.nudge_frequency)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.spectrum)
        splitter.addWidget(self.waterfall)
        splitter.setSizes([250, 550])

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._build_controls(), 0)
        # A transceiver's own display and function keys, rebuilt from its CI-V state.
        # Hidden for the SDRs.
        self._radio_panel = QtWidgets.QWidget()
        panel = QtWidgets.QVBoxLayout(self._radio_panel)
        panel.setContentsMargins(0, 1, 0, 1)
        panel.setSpacing(2)
        self.radio_display = RadioDisplay()
        self.function_panel = FunctionPanel()
        self.function_panel.memory_tune.connect(self.tune_radio_memory)
        self.function_panel.memory_save.connect(self.save_radio_memory)
        panel.addWidget(self.radio_display)
        panel.addWidget(self.function_panel)
        self._radio_panel.hide()
        layout.addWidget(self._radio_panel, 0)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self._status = self.statusBar()
        self._build_scanner_dock()
        self._build_decoder_dock()
        self._apply_device_profile()
        self._sync_playback_ui()
        saved = self.settings.radios.get(self.current_device_key())
        if saved is not None:
            # Rate, zoom and levels came in with the last snapshot; the gains did not.
            self.apply_radio_hardware(saved)
        self._apply_geometry()
        self.spectrum.set_center_marker(source.center_freq)

        # Debounced, so dragging a spinbox does not rewrite the file on every step.
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)
        self._save_timer.timeout.connect(self._save_state)

        # Repeater and tone settings, before audio starts so its tone squelch is right.
        self._apply_fm_snapshot(self.settings.last or Snapshot(freq_hz=source.center_freq))
        if self._initial_mode != "off":
            self.set_mode(self._initial_mode)
        self._sync_zerobeat_enabled()
        self._sync_pitch_visible()
        self._sync_mode_extras()
        self._update_passband()

        self._zerobeat_timer = QtCore.QTimer(self)
        self._zerobeat_timer.timeout.connect(self._zerobeat_step)

        self._space_filter = _SpaceTogglesTransmit(self)
        QtWidgets.QApplication.instance().installEventFilter(self._space_filter)
        self._sync_tx_enabled()

        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._on_frame)
        self._timer.start(int(1000 / self.fps))
        self._sync_ppm()
        if self._auto_calibrate:
            QtCore.QTimer.singleShot(1500, self._calibrate_if_new)

    # -- SDR selection -----------------------------------------------------

    def current_device_key(self) -> str:
        profile = getattr(self.source, "profile", None) or profile_for(self.source.caps.driver)
        return profile.key if profile else self.source.caps.driver

    def _detect_radios(self) -> list:
        self._detected = list(self._availability_fn())
        return self._detected

    def detected_radios(self) -> set[str]:
        """Radios known to be there: attached ones, and the one in use. The WiFi IC-705
        cannot be seen without logging in, so it counts only while in use."""
        found = {a.profile.key for a in self._detected
                 if a.connected and a.profile.key != "icom705net"}
        if not self.playing_back:
            found.add(self.current_device_key())
        return found

    def radio_covers(self, key: str, hz: float) -> bool:
        """Whether radio `key` can tune `hz`: by what it reported when last opened, or
        failing that by its profile."""
        if not self.playing_back and key == self.current_device_key():
            return self.source.caps.covers(hz)
        measured = self.settings.radio_ranges.get(key)
        if measured:
            return any(lo <= hz <= hi for lo, hi in measured)
        profile = profile_for(key)
        return bool(profile and profile.covers(hz))

    def radio_for(self, hz: float) -> str:
        """The radio that should tune `hz`: the preferred one if it reaches, else the
        current one if it does, else the first other radio there that does (in list
        order). The current one if none can."""
        current = self.current_device_key()
        preferred = self.settings.preferred_device or current
        there = self.detected_radios()
        if preferred in there and self.radio_covers(preferred, hz):
            return preferred
        if self.radio_covers(current, hz):
            return current
        order = [a.profile.key for a in self._detected]
        return next((k for k in order if k in there and self.radio_covers(k, hz)), current)

    def _change_radio_for(self, hz: float) -> bool:
        """Hand `hz` to another radio if `radio_for` says so; True if that happened (the
        new radio is then already there). Looks for newly attached radios once if none
        known reaches it."""
        if self.settings.preferred_device is None:
            # Never chosen from the list: the radio in use until now is first choice.
            self.settings.preferred_device = self.current_device_key()
        key = self.radio_for(hz)
        if key == self.current_device_key() and not self.radio_covers(key, hz):
            self._detect_radios()
            key = self.radio_for(hz)
        current = self.current_device_key()
        if key == current:
            if not self.radio_covers(current, hz):
                self._status.showMessage(
                    f"no radio here reaches {hz / 1e6:.4f} MHz", 6000)
            return False
        old = profile_for(current)
        if not self.switch_device(key, freq_hz=hz, chosen=False):
            return False
        new = profile_for(key)
        why = ("back to the first choice" if key == self.settings.preferred_device
               else f"{hz / 1e6:.4f} MHz is beyond the {old.label if old else current}")
        self._status.showMessage(f"now using {new.label if new else key}: {why}", 8000)
        self._freq_spin.blockSignals(True)
        self._freq_spin.setValue(self.source.center_freq / 1e6)
        self._freq_spin.blockSignals(False)
        return True

    def _refresh_device_list(self) -> None:
        """List every supported radio, saying which are connected, and select ours.
        Radios detected now are shown in yellow."""
        entries = self._detect_radios()
        current = self.current_device_key()
        there = self.detected_radios()
        combo = self._device_combo
        combo.blockSignals(True)
        combo.clear()
        if self.playing_back:
            # Choosing any radio from the list ends playback.
            combo.addItem(f"\u25b6 {self.source.path.name}", PLAYBACK_DRIVER)
        for entry in entries:
            profile = entry.profile
            label = profile.label
            if profile.key != current and not entry.connected:
                label += f"  \u2014 {entry.status}"
            combo.addItem(label, profile.key)
            tip = [f"{profile.label}: {profile.describe_ranges()}"]
            if profile.notes:
                tip.append(profile.notes)
            if not entry.installed:
                tip.append(f"Install: {profile.install}")
            if profile.key in there:
                tip.append("Detected now")
                row = combo.count() - 1
                combo.setItemData(row, QtGui.QColor(DETECTED_COLOUR),
                                  QtCore.Qt.ItemDataRole.BackgroundRole)
                combo.setItemData(row, QtGui.QColor("black"),
                                  QtCore.Qt.ItemDataRole.ForegroundRole)
            combo.setItemData(combo.count() - 1, "\n".join(tip),
                              QtCore.Qt.ItemDataRole.ToolTipRole)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _refresh_freq_range(self) -> None:
        """The frequency box takes anything a radio that is there can tune: going beyond
        this one's reach changes radio (`radio_for`)."""
        caps = self.source.caps
        if not self.playing_back:
            self.settings.radio_ranges[self.current_device_key()] = [
                (r.min_hz, r.max_hz) for r in caps.freq_ranges]
        spans = [(r.min_hz, r.max_hz) for r in caps.freq_ranges]
        if not self.playing_back:
            for key in self.detected_radios():
                spans += self.settings.radio_ranges.get(key) or [
                    (r.min_hz, r.max_hz) for r in getattr(profile_for(key), "freq_ranges", ())]
        lo = min((a for a, _ in spans), default=0.0) / 1e6
        hi = max((b for _, b in spans), default=6000e6) / 1e6
        self._freq_spin.blockSignals(True)
        self._freq_spin.setRange(lo, hi)
        self._freq_spin.setValue(self.source.center_freq / 1e6)
        self._freq_spin.blockSignals(False)
        self._freq_spin.setToolTip(f"Tunable: {caps.describe_ranges()}")

    def _refresh_rates(self) -> None:
        rates = sorted(self.source.caps.sample_rates, reverse=True)
        combo = self._rate_combo
        combo.blockSignals(True)
        combo.clear()
        for rate in rates:
            label = f"{rate / 1e3:g} kS/s" if rate < 1e6 else f"{rate / 1e6:g} MS/s"
            combo.addItem(label, rate)
        index = combo.findData(self.source.sample_rate)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)
        visible = len(rates) > 1
        self._rate_label.setVisible(visible)
        combo.setVisible(visible)

    def _refresh_hw_bandwidths(self) -> None:
        widths = sorted(self.source.caps.bandwidths)
        actual = getattr(self.source, "bandwidth", 0.0) or 0.0
        if widths and actual > 0 and all(abs(w - actual) > 0.01 * actual for w in widths):
            # The radio can be set to a width it does not list: the Pluto opens at 18 MHz
            # with 10 MHz its widest option. Show that, not the nearest option.
            widths = sorted(widths + [actual])
        combo = self._bw_combo
        combo.blockSignals(True)
        combo.clear()
        for bw in widths:
            combo.addItem(f"{bw / 1e3:g} kHz", bw)
        # Show what the radio is really using. Drivers such as the HackRF's set it from
        # the sample rate, so the first entry would be a lie (1.75 MHz shown, 3.5 in use).
        if widths and actual > 0:
            combo.setCurrentIndex(min(range(len(widths)), key=lambda i: abs(widths[i] - actual)))
        self._if_bw_chosen = False
        combo.blockSignals(False)
        self._hw_bw_label.setVisible(bool(widths))
        combo.setVisible(bool(widths))
        self._sync_radio_row()

    def _on_hw_bandwidth_changed(self) -> None:
        width = self._bw_combo.currentData()
        if width is not None:
            self.source.set_bandwidth(width)
            self._if_bw_chosen = True
            self._schedule_save()

    def _rebuild_device_controls(self) -> None:
        layout = self._device_slot_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        box = self._build_device_controls()
        layout.addWidget(box)
        caps = self.source.caps
        has_any = bool(caps.has_agc or caps.gain_elements or getattr(caps, "settings", ())
                       or getattr(self.source, "controls", ()))
        self._device_slot.setVisible(has_any)
        self._sync_radio_row()

    def _apply_device_profile(self) -> None:
        """Make the window match the radio: everything device-specific in one place."""
        caps = self.source.caps
        self.setWindowTitle(f"{APP_TITLE} \u2014 {caps.label or caps.driver}")
        self._refresh_device_list()
        self._refresh_freq_range()
        self._refresh_rates()
        self._refresh_hw_bandwidths()
        self._rebuild_device_controls()
        self._reset_tx_gains()
        if hasattr(self, "_tx_button"):
            self._sync_tx_enabled()
        if hasattr(self, "scanner_panel"):
            self.scanner_panel.set_coverage(caps.freq_ranges)
        self._sync_transceiver_ui()

    @property
    def is_transceiver(self) -> bool:
        """A radio that demodulates itself (the IC-705): its scope, not our FFT."""
        return getattr(self.source, "kind", "sdr") == "transceiver"

    def _sync_transceiver_ui(self) -> None:
        """Hide what only an IQ radio can do; the rest works on a transceiver as is."""
        sdr = not self.is_transceiver
        for widget in (self._zoom_label, self._zoom_combo, self._fft_label, self._fft_combo,
                       self._scan_button, self._decode_button, self._classify_button):
            widget.setVisible(sdr)
        for widget in (self._span_label, self._span_combo):
            widget.setVisible(not sdr)
        # The radio has its own S meter on the display panel; the SDR one is not needed.
        # Nor are the listening offset and recording: both work on the app's own IQ and
        # demodulated audio, which a transceiver does not give it.
        self.smeter.setVisible(sdr)
        for widget in (self._offset_label, self._offset_spin, self._rec_title,
                       self._rec_audio_button, self._rec_iq_button, self._rec_label):
            widget.setVisible(sdr)
        if hasattr(self, "_controls_layout"):
            self._compact_rows(not sdr)
        if hasattr(self, "_radio_panel"):
            self._radio_panel.setVisible(not sdr)
            if not sdr:
                self.function_panel.build(self.source)
        if not sdr:
            if self._scan_button.isChecked():
                self._scan_button.setChecked(False)
            if self.decimator.factor != 1:
                self.set_decimation(1)
                self._sync_zoom_combo()
        # Recording from the IC-705 comes with its audio, later (PLANNING.md P7).
        for widget in (self._rec_audio_button, self._rec_iq_button):
            widget.setEnabled(sdr)
        # The radio demodulates, so the app's listening offset does not apply, and its
        # squelch is the radio's own (SQL, Radio row) -- hidden here rather than greyed
        # out, so there is one squelch control, not a dead one beside it.
        self._offset_spin.setEnabled(sdr)
        self._squelch_check.setVisible(sdr)
        self._squelch_spin.setVisible(sdr)
        if sdr:
            self._sync_squelch_enabled()
        if sdr:
            self._stop_radio_audio()
        else:
            self._start_radio_audio()
        self._fill_mode_combo()
        # A transceiver's mode is the radio's, so it needs no audio device to change.
        self._mode_combo.setEnabled(self._audio_ok or not sdr)
        self._mode_combo.setToolTip("" if sdr else "The IC-705's mode")
        self._refresh_bandwidths()
        self._update_passband()
        self.spectrum.setLabel("left", "power" if sdr else "scope level",
                               units="dBFS" if sdr else "dB")
        self.waterfall.resize_bins(self._sdr_waterfall_bins if sdr else SCOPE_POINTS)
        self._scope_geometry = None
        self._apply_geometry()

    def _on_device_chosen(self, index: int) -> None:
        key = self._device_combo.itemData(index)
        if key and key != PLAYBACK_DRIVER:
            # Choosing a radio makes it first choice wherever it reaches.
            self.settings.preferred_device = key
            self._schedule_save()
        if key == "icom705net" and not self._ask_network_login():
            self._refresh_device_list()
            return
        if key == "icom705net" and key == self.current_device_key():
            self.switch_device("icom705net")                     # log in again
        elif key and key != self.current_device_key():
            self.switch_device(key)

    def _ask_network_login(self) -> bool:
        """Where the IC-705 is on the network and how to log in, saved for next time.
        False if cancelled."""
        from ..device.icom_net import load_login, save_login
        from .network_login import NetworkLoginDialog

        dialog = NetworkLoginDialog(load_login(), self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        save_login(dialog.login())
        return True

    def switch_device(self, key: str, freq_hz: float | None = None,
                      chosen: bool = True) -> bool:
        """Change to another radio (or reopen this one), at `freq_hz` if given and it
        reaches it. A `chosen` radio becomes first choice (`radio_for`); one changed to
        automatically or by a memory does not. Returns False, leaving the current one,
        on failure."""
        profile = profile_for(key)
        if profile is None:
            return False
        entry = {a.profile.key: a for a in self._availability_fn()}.get(key)
        if entry is None or not entry.connected:
            if entry is not None and entry.installed:
                reason = "is not connected. Plug it in and choose it again."
            else:
                reason = f"needs its driver installed first: {profile.install}."
            self._status.showMessage(f"{profile.label} {reason}", 10000)
            self._refresh_device_list()      # put the selector back on the live radio
            return False

        previous_mode = self._release_source()
        old = self.source
        old_driver, old_freq = old.caps.driver, old.center_freq
        if self.playing_back:
            # Back from a recording: reopen the radio where it was left, not at the
            # recording's frequency.
            old_driver = profile.driver
            old_freq = self._radio_before_playback[1] if self._radio_before_playback else old_freq
        wanted = old_freq if freq_hz is None else float(freq_hz)
        centre = (wanted if self.radio_covers(key, wanted)
                  else starting_frequency(profile, wanted))
        try:
            old.close()
        except Exception:
            pass

        switched = True
        try:
            new = self._source_factory(profile.driver, centre)
            new.start()
        except Exception as exc:
            switched = False
            self._status.showMessage(f"could not open {profile.label}: {exc}", 10000)
            try:
                # Better the radio we had than a window with nothing behind it.
                new = self._source_factory(old_driver, old_freq)
                new.start()
            except Exception:
                new = old

        self._radio_before_playback = None
        self._adopt_source(new, previous_mode, restore_settings=switched)
        if switched:
            self.settings.device = profile.key
            if chosen:
                self.settings.preferred_device = profile.key
            self._save_state()
            self._status.showMessage(
                f"now using {profile.label} at {self.source.center_freq / 1e6:.4f} MHz", 5000
            )
        return switched

    def _release_source(self) -> str:
        """Stop everything that uses the current source, ready to replace it. Returns
        the demodulator mode, to restore on the new one."""
        previous_mode = self.mode
        self._tx_button.setChecked(False)
        self._stop_radio_audio()
        if self.scanner is not None:
            self.stop_scan()
        self._stop_zerobeat()
        self.stop_audio_recording("changing radio")
        self.stop_iq_recording("changing radio")
        self.set_mode("off")
        self._stop_decoder()
        self._timer.stop()
        if not self.playing_back:
            self.settings.radios[self.current_device_key()] = self.current_radio_settings()
        return previous_mode

    def _adopt_source(self, new: IQSource, previous_mode: str,
                      restore_settings: bool = True) -> None:
        """Make `new` the source and the window match it."""
        self.source = new
        self.spectrum.reset()
        self.waterfall.clear_history()
        self._rows_pushed = 0
        # Different radios sit at very different levels: take the new radio's own, or
        # refit the colours if it has none saved.
        self._levels_explicit = False
        self._auto_pending = True
        self._ppm_measured = None
        self._apply_device_profile()
        saved = (self.settings.radios.get(self.current_device_key())
                 if restore_settings and not self.playing_back else None)
        if saved is not None:
            self.apply_radio_settings(saved)
        self._apply_geometry()
        self.spectrum.set_center_marker(self.source.center_freq)
        self._freq_spin.blockSignals(True)
        self._freq_spin.setValue(self.source.center_freq / 1e6)
        self._freq_spin.blockSignals(False)
        self._update_offset_range()
        self._sync_playback_ui()
        self._timer.start(int(1000 / self.fps))
        if previous_mode != "off":
            self.set_mode(previous_mode)
        self._start_decoder(self.decoder_panel.decoder)
        self._sync_ppm()
        if self._auto_calibrate:
            QtCore.QTimer.singleShot(500, self._calibrate_if_new)

    # -- IQ playback ---------------------------------------------------------

    @property
    def playing_back(self) -> bool:
        return getattr(self.source.caps, "driver", "") == PLAYBACK_DRIVER

    def open_recording(self, path) -> bool:
        """Play an IQ recording in place of the radio. False, leaving the radio, if the
        file cannot be played."""
        from ..device.playback import FileIQSource

        try:
            new = FileIQSource(path)
        except (OSError, ValueError) as exc:
            self._status.showMessage(f"cannot play {Path(path).name}: {exc}", 10000)
            return False
        if not self.playing_back:
            self._radio_before_playback = (self.current_device_key(), self.source.center_freq)
            self._save_state()                     # the radio's state, before it goes
        previous_mode = self._release_source()
        try:
            self.source.close()
        except Exception:
            pass
        new.start()
        self._adopt_source(new, previous_mode)
        self._status.showMessage(
            f"playing {new.path.name}: {new.duration_s:.1f} s at "
            f"{new.sample_rate / 1e3:g} kS/s, {new.recorded_center / 1e6:.4f} MHz", 8000)
        return True

    def stop_playback(self) -> bool:
        """Back to the radio that was in use before the recording was opened."""
        if not self.playing_back:
            return False
        key = self._radio_before_playback[0] if self._radio_before_playback else None
        key = key or self.settings.device or "airspyhf"
        return self.switch_device(key)

    def _on_play_clicked(self) -> None:
        start = self.recordings_dir if self.recordings_dir.exists() else Path.home()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Play an IQ recording", str(start), "IQ recordings (*.cf32)")
        if path:
            self.open_recording(path)

    def _on_pause_toggled(self, paused: bool) -> None:
        if self.playing_back:
            self.source.set_paused(paused)

    def _sync_playback_ui(self) -> None:
        playing = self.playing_back
        for widget in (self._pause_button, self._stop_play_button, self._play_label):
            widget.setVisible(playing)
        self._pause_button.setChecked(False)
        # Recording a recording only copies it.
        self._rec_iq_button.setEnabled(not playing and not self.is_transceiver)
        self._update_play_label()

    def _update_play_label(self) -> None:
        if not self.playing_back:
            self._play_label.setText("")
            return
        src = self.source
        self._play_label.setText(f"{src.path.name}  {src.position_s:5.1f} / "
                                 f"{src.duration_s:.1f} s")

    # -- per-radio settings ------------------------------------------------

    def current_radio_settings(self) -> RadioSettings:
        """What is set on this radio now, as opposed to what station is tuned."""
        caps = self.source.caps
        gains: dict[str, float] = {}
        getter = getattr(self.source, "get_gain", None)
        for element in caps.gain_elements if getter is not None else ():
            try:
                gains[element.name] = float(getter(element.name))
            except Exception:
                pass
        flags: dict[str, bool] = {}
        reader = getattr(self.source, "read_setting", None)
        for setting in getattr(caps, "settings", ()) if reader is not None else ():
            try:
                flags[setting.key] = bool(reader(setting.key))
            except Exception:
                pass
        explicit = self._levels_explicit
        return RadioSettings(
            sample_rate=self.source.sample_rate,
            decimation=self.decimator.factor,
            gains=gains,
            agc=self._agc_check.isChecked() if self._agc_check is not None else None,
            # Only a width the user picked: otherwise the driver's automatic choice,
            # which follows the rate, is the right one to come back to.
            if_bandwidth_hz=(self._bw_combo.currentData()
                             if self._if_bw_chosen and self._bw_combo.count() else None),
            driver_settings=flags,
            min_db=self._levels[0] if explicit else None,
            max_db=self._levels[1] if explicit else None,
            tx_gains=dict(self._tx_gains),
            tx_audio_level=self._tx_audio_level if self.is_transceiver else None,
            tx_mic_gain_db=(self._tx_mic_gain_db
                            if not self.is_transceiver and self.source.caps.tx else None),
            ppm=self._ppm_measured,
        )

    def apply_radio_hardware(self, radio: RadioSettings) -> None:
        """Gains, AGC, IF bandwidth and driver switches -- what the radio itself holds."""
        caps = self.source.caps
        stages = {g.name: g for g in caps.gain_elements}
        for name, db in radio.gains.items():
            stage = stages.get(name)
            if stage is not None:
                self.source.set_gain(name, min(max(db, stage.min_db), stage.max_db))
        if radio.agc is not None and caps.has_agc:
            self.source.set_agc(radio.agc)
        writer = getattr(self.source, "write_setting", None)
        known = {setting.key for setting in getattr(caps, "settings", ())}
        for key, enabled in radio.driver_settings.items():
            if writer is not None and key in known:
                writer(key, enabled)
        if radio.if_bandwidth_hz is not None:
            index = self._bw_combo.findData(radio.if_bandwidth_hz)
            if index >= 0:
                self._bw_combo.setCurrentIndex(index)
                self.source.set_bandwidth(radio.if_bandwidth_hz)
                self._if_bw_chosen = True
        if radio.tx_audio_level is not None:
            self._tx_audio_level = min(1.0, max(0.01, radio.tx_audio_level))
        if radio.ppm is not None and self.can_correct_ppm:
            self._set_ppm(radio.ppm)
        if radio.tx_mic_gain_db is not None:
            from ..dsp.modulate import MIC_GAIN_RANGE_DB

            low, high = MIC_GAIN_RANGE_DB
            self._tx_mic_gain_db = min(high, max(low, radio.tx_mic_gain_db))
        self._rebuild_device_controls()       # show what was just set
        tx = caps.tx
        if tx is not None:
            stages = {g.name: g for g in tx.gain_elements}
            for name, db in radio.tx_gains.items():
                if name in stages:
                    self._tx_gains[name] = min(max(db, stages[name].min_db), stages[name].max_db)
            self._rebuild_tx_controls()

    def apply_radio_settings(self, radio: RadioSettings) -> None:
        """Everything saved for this radio: rate, zoom and colour levels as well."""
        index = self._rate_combo.findData(radio.sample_rate) if radio.sample_rate else -1
        if index >= 0 and radio.sample_rate != self.source.sample_rate:
            self._rate_combo.setCurrentIndex(index)       # restarts the stream
        factor = radio.decimation if radio.decimation in ZOOM_FACTORS else 1
        if factor != self.decimator.factor:
            self.set_decimation(factor)
            self._sync_zoom_combo()
        if radio.min_db is not None and radio.max_db is not None:
            self._levels_explicit = True
            self._auto_pending = False
            self._set_levels(radio.min_db, radio.max_db)
        self.apply_radio_hardware(radio)

    def _radio_settings_for_new_station(self, snap: Snapshot) -> RadioSettings:
        """Settings for a station never saved on this radio.

        This radio's own last-used settings, with the zoom chosen to give about the
        span the memory had on the radio it was saved on -- the rate itself may not
        exist here.
        """
        base = self.settings.radios.get(self.current_device_key()) or RadioSettings()
        rates = self.source.caps.sample_rates
        rate = base.sample_rate if base.sample_rate in rates else self.source.sample_rate
        span = snap.sample_rate / max(1, snap.decimation)
        factor = min(ZOOM_FACTORS, key=lambda f: abs(np.log((rate / f) / span)))
        return dataclasses.replace(base, sample_rate=rate, decimation=factor)

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
        # Hidden until asked for, with the Scan button; the button and the dock's own
        # close box stay in step either way round.
        self._scanner_dock.hide()
        self._scan_button.setChecked(False)
        self._scan_button.toggled.connect(self._scanner_dock.setVisible)
        self._scanner_dock.visibilityChanged.connect(self._on_scanner_visibility)

    def _on_scanner_visibility(self, visible: bool) -> None:
        # visibilityChanged also fires when the window is minimised or the dock is
        # tabbed away; only a real close should untick the button.
        shown = not self._scanner_dock.isHidden()
        if self._scan_button.isChecked() != shown:
            self._scan_button.blockSignals(True)
            self._scan_button.setChecked(shown)
            self._scan_button.blockSignals(False)

    def _scan_config(self) -> ScanConfig:
        panel = self.scanner_panel
        return ScanConfig(
            start_hz=panel.start_hz,
            end_hz=panel.end_hz,
            step_hz=panel.step_hz,
            threshold_db=panel.threshold_db,
            stop_on_signal=panel.stop_on_signal,
            min_sightings=panel.min_sightings,
            dc_guard_hz=5e3 if getattr(self.source.profile, "dc_offset", False) else 1.5e3,
            dc_spike_offset_hz=self.source.dc_spike_offset_hz,
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
        self._retune(freq_hz, allow_snap=False)

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

    def _build_decoder_dock(self) -> None:
        self.decoder_panel = DecoderPanel()
        self.decoder_panel.decoderChanged.connect(self._on_decoder_changed)
        self.decoder_panel.mapRequested.connect(self.show_map)
        self._decoder_dock = QtWidgets.QDockWidget("Decode", self)
        self._decoder_dock.setObjectName("decoderDock")
        self._decoder_dock.setWidget(self.decoder_panel)
        # Across the top, full width: ACARS and ADS-B lines are long, and in a window
        # that is not maximised a side dock left them unreadable (VK3RQ, 2026-10-05).
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, self._decoder_dock)
        self._decoder_dock.hide()
        self._decode_button.toggled.connect(self._show_decoder_dock)
        self._decoder_dock.visibilityChanged.connect(self._on_decoder_visibility)

    def _show_decoder_dock(self, show: bool) -> None:
        """Show or hide the Decode panel; on showing, give it a fifth of the screen."""
        was_hidden = self._decoder_dock.isHidden()
        self._decoder_dock.setVisible(show)
        if show and was_hidden and not self._decoder_dock.isFloating():
            screen = self.screen() or QtWidgets.QApplication.primaryScreen()
            height = max(140, screen.availableGeometry().height() // 5)
            self.resizeDocks([self._decoder_dock], [height], QtCore.Qt.Orientation.Vertical)

    def _on_decoder_visibility(self, visible: bool) -> None:
        shown = not self._decoder_dock.isHidden()
        if self._decode_button.isChecked() != shown:
            self._decode_button.blockSignals(True)
            self._decode_button.setChecked(shown)
            self._decode_button.blockSignals(False)

    def _apply_decoder(self, key: str) -> None:
        """Choose decoder `key` ("" for none), as a recalled memory asks, showing the
        Decode panel when there is one to watch."""
        from ..decoding import DECODERS

        key = key if key in DECODERS else ""
        combo = self.decoder_panel.combo
        if combo.currentData() != key:
            # The memory brings its own mode: the decoder must not replace it.
            self._decoder_from_memory = True
            try:
                combo.setCurrentIndex(max(0, combo.findData(key)))   # starts or stops it
            finally:
                self._decoder_from_memory = False
        if key and not self.is_transceiver:
            self._decode_button.setChecked(True)

    def _mode_for_decoder(self, key: str) -> None:
        """Listen in the mode a chosen decoder's signal is sent in (VK3RQ, 2026-10-06):
        NBFM for the FM data modes, AM for ACARS. ADS-B's pulses have no audio worth a
        mode, so it leaves the mode alone, as does a memory, which has its own."""
        from ..decoding import DECODERS

        spec = DECODERS.get(key)
        if (spec is None or spec.mode not in MODES or self.is_transceiver
                or getattr(self, "_decoder_from_memory", False) or self.mode == spec.mode):
            return
        index = self._mode_combo.findData(spec.mode)
        if index >= 0:
            self._mode_combo.setCurrentIndex(index)

    def _on_decoder_changed(self, key: str) -> None:
        self._mode_for_decoder(key)
        self._start_decoder(key)
        if key in ("ais", "acars", "adsb"):
            self.show_map()                 # ships and aircraft are best seen on a map

    def classify_signal(self) -> None:
        """Capture the tuned frequency for a few seconds and name what is there, on the
        waterfall (`classify.py`). Runs on its own thread; `_poll_classify` collects it."""
        from ..classify import CAPTURE_S, ClassifyJob

        if self.is_transceiver or getattr(self, "_classify_job", None) is not None:
            return
        self.waterfall.clear_label()
        self._classify_job = ClassifyJob(self.source, self.listen_freq)
        self._classify_button.setEnabled(False)
        self._classify_button.setText("\u2026")
        self._status.showMessage(
            f"listening to {self.listen_freq / 1e6:.4f} MHz for {CAPTURE_S:g} s to "
            "classify it...", int(CAPTURE_S * 1000) + 2000)
        QtCore.QTimer.singleShot(100, self._poll_classify)

    def _cancel_classify(self) -> None:
        job = getattr(self, "_classify_job", None)
        if job is not None:
            job.cancel()

    def _poll_classify(self) -> None:
        job = getattr(self, "_classify_job", None)
        if job is None:
            return
        if not job.done:
            QtCore.QTimer.singleShot(100, self._poll_classify)
            return
        self._classify_job = None
        self._classify_button.setEnabled(True)
        self._classify_button.setText("?")
        if job._cancel.is_set() or job.centre_hz != self.source.center_freq:
            return                                 # retuned meanwhile: it is stale
        if job.error:
            self._status.showMessage(f"could not classify: {job.error}", 8000)
        elif job.result is None:
            self._status.showMessage(
                f"no signal at {job.listen_hz / 1e6:.4f} MHz to classify", 8000)
        else:
            text = job.result.text()
            self.waterfall.show_label(job.result.freq_hz, text)
            self._status.showMessage(text, 15000)

    # -- frequency correction ------------------------------------------------------

    @property
    def can_correct_ppm(self) -> bool:
        return (not self.is_transceiver and not self.playing_back
                and hasattr(self.source, "set_ppm"))

    def _sync_ppm(self) -> None:
        """Show the radio's correction, or hide the control for one that has none (a
        transceiver tunes itself; a recording is what it is)."""
        usable = self.can_correct_ppm
        self._ppm_slot.setVisible(usable)
        self._ppm_spin.blockSignals(True)
        self._ppm_spin.setValue(getattr(self.source, "ppm", 0.0) if usable else 0.0)
        self._ppm_spin.blockSignals(False)
        self._sync_radio_row()

    def _set_ppm(self, ppm: float) -> None:
        self.source.set_ppm(ppm)
        self._ppm_measured = float(ppm)
        self._ppm_spin.blockSignals(True)
        self._ppm_spin.setValue(ppm)
        self._ppm_spin.blockSignals(False)

    def _on_ppm_changed(self, ppm: float) -> None:
        if self.can_correct_ppm:
            self._set_ppm(ppm)
            self._schedule_save()

    def calibrate(self, references=None) -> None:
        """Measure this radio's error from a known carrier, on a thread: at `references`
        (Hz or (Hz, name) pairs; the Cal button passes the listening frequency), else the
        known local ones (calibrate.REFERENCES). The radio is retuned while it measures
        and put back; `_poll_calibration` applies the result."""
        from ..calibrate import REFERENCES, CalibrateJob

        if not self.can_correct_ppm or getattr(self, "_calibration", None) is not None:
            return
        if references is None:
            references = REFERENCES
        elif not isinstance(references, (list, tuple)):
            references = [float(references)]
        self._calibration = CalibrateJob(self.source, references)
        self._cal_button.setEnabled(False)
        self._status.showMessage("measuring this radio's frequency error...", 15000)
        QtCore.QTimer.singleShot(200, self._poll_calibration)

    def _cancel_calibration(self) -> None:
        job = getattr(self, "_calibration", None)
        if job is not None:
            job.cancel()

    def _poll_calibration(self) -> None:
        job = getattr(self, "_calibration", None)
        if job is None:
            return
        if not job.done:
            if job.stage:
                self._status.showMessage(job.stage + "...", 3000)
            QtCore.QTimer.singleShot(200, self._poll_calibration)
            return
        self._calibration = None
        self._cal_button.setEnabled(True)
        if job.cancelled or job.source is not self.source:
            return
        self.waterfall.clear_history()            # rows from the reference frequency
        label = self.source.caps.label or self.current_device_key()
        if job.result is None:
            self._status.showMessage(
                f"{label} not calibrated: {job.error}. Tune exactly to a known carrier "
                "and press Cal.", 15000)
            return
        self._set_ppm(job.result.ppm)
        self._schedule_save()
        self._status.showMessage(job.result.describe(label), 15000)

    def _calibrate_if_new(self) -> None:
        """A radio never calibrated is measured against the known references when it is
        first connected (VK3RQ: "set up once for each new radio"). Once a session: a
        radio that heard none of them is not retuned again and again."""
        key = self.current_device_key()
        saved = self.settings.radios.get(key)
        if (not self.can_correct_ppm or key in self._calibration_tried
                or self._ppm_measured is not None
                or (saved is not None and saved.ppm is not None)):
            return
        self._calibration_tried.add(key)
        self.calibrate()

    def show_map(self) -> None:
        if self.map_window is None:
            self.map_window = MapWindow(self.targets, parent=self)
        self.map_window.show()
        self.map_window.raise_()
        self.map_window.refresh()

    def _start_decoder(self, key: str) -> None:
        """Run decoder `key` on the current source, or none for "". Replaces any other."""
        self._stop_decoder()
        if not key or self.is_transceiver:
            self.decoder_panel.set_status("")
            return
        worker = DecodeWorker(self.source, key, self._offset_spin.value() * 1e3)
        worker.start()
        self.decode_worker = worker
        self._update_decoder_status()

    def _update_decoder_status(self) -> None:
        worker = self.decode_worker
        if worker is None:
            return
        label = worker.spec.label
        if worker.problem:
            self.decoder_panel.set_status(f"{label}: {worker.problem}")
            return
        if worker.fixed_channels:
            channels = worker.channels_in_view()
            seen = [f"{n} {hz / 1e6:.3f}".strip() for n, hz, ok in channels if ok]
            missed = [f"{n} {hz / 1e6:.3f}".strip() for n, hz, ok in channels if not ok]
            text = f"{label} on {', '.join(seen) if seen else 'no channel'} MHz"
            if missed:
                mid = sum(hz for _, hz, _ in channels) / len(channels)
                text += (f" -- {', '.join(missed)} MHz out of view: tune near "
                         f"{mid / 1e6:.3f} MHz for all of them")
        else:
            listening = self.source.center_freq + self._offset_spin.value() * 1e3
            text = (f"{label} on {listening / 1e6:.4f} MHz, "
                    f"{worker.spec.bandwidth_hz / 1e3:g} kHz channel")
        self.decoder_panel.set_status(text)

    def _stop_decoder(self) -> None:
        if self.decode_worker is not None:
            self.decode_worker.stop()
            self.decode_worker = None

    def _collect_decoded(self) -> None:
        worker = self.decode_worker
        if worker is not None:
            messages = worker.take()
            self.decoder_panel.add(messages)
            self.targets.update(messages)
        if self.map_window is not None and self.map_window.isVisible():
            self.map_window.refresh()

    def _build_controls(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(bar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        self._controls_layout = outer
        outer.addWidget(self._build_tuning_row())
        outer.addWidget(self._build_radio_row())
        self._display_row = self._build_display_row()
        outer.addWidget(self._display_row)
        self._audio_row = self._build_audio_row()
        outer.addWidget(self._audio_row)
        outer.addWidget(self._build_fm_row())
        self._memory_row = self._build_memory_row()
        outer.addWidget(self._memory_row)
        return bar

    def _compact_rows(self, compact: bool) -> None:
        """For a transceiver, fold five control lines into three: the display settings
        onto the end of the Radio line, and Memory/Record into the room the SDR S meter
        leaves on the Audio line (VK3RQ, 2026-09-30). Back to their own lines for an SDR."""
        outer = self._controls_layout
        radio = self._radio_row.layout()
        audio = self._audio_row.layout()
        if compact:
            # Before each row's closing stretch.
            radio.insertWidget(radio.count() - 1, self._display_row)
            audio.insertWidget(audio.count() - 1, self._memory_row)
            outer.setSpacing(2)
        else:
            outer.insertWidget(outer.indexOf(self._radio_row) + 1, self._display_row)
            outer.addWidget(self._memory_row)
            outer.setSpacing(4)

    def _build_radio_row(self) -> QtWidgets.QWidget:
        """The radio's own settings: IF bandwidth, gain stages, bias-tee.

        A row of its own because a HackRF's three gains and bias-tee pushed the tuning
        row off the side of the screen. Hidden entirely for a radio with nothing to set.
        """
        self._radio_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self._radio_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("Radio"))

        self._hw_bw_label = QtWidgets.QLabel("IF BW")
        row.addWidget(self._hw_bw_label)
        self._bw_combo = QtWidgets.QComboBox()
        self._bw_combo.currentIndexChanged.connect(self._on_hw_bandwidth_changed)
        row.addWidget(self._bw_combo)

        # Gain and driver settings, rebuilt whenever the radio changes.
        self._device_slot = QtWidgets.QWidget()
        self._device_slot_layout = QtWidgets.QHBoxLayout(self._device_slot)
        self._device_slot_layout.setContentsMargins(12, 0, 0, 0)
        row.addWidget(self._device_slot)

        # Frequency correction (VK3RQ, 2026-10-06): the radio's error in ppm, and Cal to
        # measure it from a known carrier at the listening frequency.
        self._ppm_slot = QtWidgets.QWidget()
        ppm_row = QtWidgets.QHBoxLayout(self._ppm_slot)
        ppm_row.setContentsMargins(12, 0, 0, 0)
        ppm_row.addWidget(QtWidgets.QLabel("PPM"))
        self._ppm_spin = QtWidgets.QDoubleSpinBox()
        self._ppm_spin.setRange(-200.0, 200.0)
        self._ppm_spin.setDecimals(2)
        self._ppm_spin.setSingleStep(0.1)
        self._ppm_spin.setKeyboardTracking(False)
        self._ppm_spin.setToolTip(
            "This radio's frequency error in parts per million (+ when it reads high),\n"
            "corrected in tuning so the display shows true frequency. Saved per radio.\n"
            "Measured automatically when a radio is first connected, from a known\n"
            "carrier (Essendon ATIS 119.8 MHz, or the 144.650 MHz beacon).")
        self._ppm_spin.valueChanged.connect(self._on_ppm_changed)
        ppm_row.addWidget(self._ppm_spin)
        self._cal_button = QtWidgets.QPushButton("Cal")
        self._cal_button.setToolTip(
            "Measure the error from a carrier at the listening frequency: tune exactly\n"
            "to a signal whose frequency you know (a CW beacon, an AM carrier) first.")
        self._cal_button.clicked.connect(lambda: self.calibrate(self.listen_freq))
        ppm_row.addWidget(self._cal_button)
        row.addWidget(self._ppm_slot)

        # Transmit gains, for radios that transmit.
        self._tx_slot = QtWidgets.QWidget()
        self._tx_slot_layout = QtWidgets.QHBoxLayout(self._tx_slot)
        self._tx_slot_layout.setContentsMargins(24, 0, 0, 0)
        row.addWidget(self._tx_slot)
        row.addStretch(1)
        return self._radio_row

    def _sync_radio_row(self) -> None:
        self._radio_row.setVisible(
            not self._bw_combo.isHidden() or not self._device_slot.isHidden()
            or not self._tx_slot.isHidden() or not self._ppm_slot.isHidden()
        )

    def _build_fm_row(self) -> QtWidgets.QWidget:
        """NBFM only: repeater split for transmit, and CTCSS/DCS signalling."""
        self._fm_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self._fm_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("NBFM"))

        # The split only matters to a radio that transmits, so it hides on others.
        self._rpt_box = QtWidgets.QWidget()
        rpt = QtWidgets.QHBoxLayout(self._rpt_box)
        rpt.setContentsMargins(0, 0, 0, 0)
        self._shift_combo = QtWidgets.QComboBox()
        for label, key in (("Simplex", SIMPLEX), ("Duplex +", PLUS), ("Duplex \u2212", MINUS)):
            self._shift_combo.addItem(label, key)
        self._shift_combo.setToolTip("Transmit on the listening frequency, or above or below "
                                     "it by the offset to work a repeater")
        self._shift_combo.currentIndexChanged.connect(self._on_fm_changed)
        rpt.addWidget(self._shift_combo)
        # "Rpt offset", not "Offset": the audio row already has a listening offset.
        rpt.addWidget(QtWidgets.QLabel("Rpt offset"))
        self._rpt_offset_spin = QtWidgets.QDoubleSpinBox()
        self._rpt_offset_spin.setRange(0.0, 100_000.0)
        self._rpt_offset_spin.setDecimals(1)
        self._rpt_offset_spin.setSingleStep(100.0)
        self._rpt_offset_spin.setSuffix(" kHz")
        self._rpt_offset_spin.setToolTip("Repeater split. Australia: 600 kHz on 2 m, "
                                         "5 MHz on 70 cm (7 MHz on some older repeaters)")
        self._rpt_offset_spin.valueChanged.connect(self._on_rpt_offset_edited)
        rpt.addWidget(self._rpt_offset_spin)
        self._tx_freq_label = QtWidgets.QLabel("")
        self._tx_freq_label.setStyleSheet("color: #ff4d4d;")
        rpt.addWidget(self._tx_freq_label)
        row.addWidget(self._rpt_box)

        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Tone"))
        self._tone_combo = QtWidgets.QComboBox()
        for label, key in (("Off", "off"), ("Tone (CTCSS TX)", "tone"),
                           ("TSQL (CTCSS TX+RX)", "tsql"), ("DCS (TX+RX)", "dcs")):
            self._tone_combo.addItem(label, key)
        self._tone_combo.setToolTip(
            "Tone: CTCSS sent on transmit.\n"
            "TSQL: CTCSS sent, and receive stays muted unless it is heard.\n"
            "DCS: the DCS code sent, and receive muted unless it is heard.")
        self._tone_combo.currentIndexChanged.connect(self._on_tone_mode_changed)
        row.addWidget(self._tone_combo)
        self._tone_value_combo = QtWidgets.QComboBox()
        self._tone_value_combo.currentIndexChanged.connect(self._on_tone_value_changed)
        row.addWidget(self._tone_value_combo)
        self._tone_state = QtWidgets.QLabel("")
        row.addWidget(self._tone_state)
        row.addStretch(1)

        # Per-band offsets edited this session; the band's standard otherwise.
        self._rpt_offsets: dict[str | None, float] = {}
        self._rpt_band = band_for(self.source.center_freq)
        self._ctcss_hz = 88.5
        self._dcs_code = "023"
        self._fm_loading = False
        return self._fm_row

    # -- NBFM: repeater and tones ------------------------------------------------

    @property
    def repeater_shift(self) -> str:
        return self._shift_combo.currentData() or SIMPLEX

    @property
    def tone_mode(self) -> str:
        return self._tone_combo.currentData() or "off"

    @property
    def tx_freq(self) -> float:
        """Where TX goes: the listening frequency, shifted for a repeater in NBFM."""
        if self.mode != "nbfm" or self.source.caps.tx is None:
            return self.listen_freq
        return tx_frequency(self.listen_freq, self.repeater_shift,
                            self._rpt_offset_spin.value() * 1e3)

    def tx_tone(self) -> tuple[str, object] | None:
        if self.mode != "nbfm":
            return None
        if self.tone_mode in ("tone", "tsql"):
            return ("ctcss", self._ctcss_hz)
        if self.tone_mode == "dcs":
            return ("dcs", self._dcs_code)
        return None

    def rx_tone(self) -> tuple[str, object] | None:
        if self.mode != "nbfm":
            return None
        if self.tone_mode == "tsql":
            return ("ctcss", self._ctcss_hz)
        if self.tone_mode == "dcs":
            return ("dcs", self._dcs_code)
        return None

    def _fill_tone_values(self) -> None:
        combo = self._tone_value_combo
        combo.blockSignals(True)
        combo.clear()
        if self.tone_mode == "dcs":
            for code in DCS_CODES:
                combo.addItem(f"D{code}N", code)
            combo.setCurrentIndex(max(0, combo.findData(self._dcs_code)))
            combo.setToolTip("DCS code, normal polarity")
        elif self.tone_mode in ("tone", "tsql"):
            for hz in CTCSS_TONES:
                combo.addItem(f"{hz:.1f} Hz", hz)
            combo.setCurrentIndex(max(0, combo.findData(self._ctcss_hz)))
            combo.setToolTip("CTCSS tone frequency")
        combo.blockSignals(False)
        combo.setVisible(self.tone_mode != "off")

    def _on_tone_mode_changed(self) -> None:
        self._fill_tone_values()
        self._apply_rx_tone()
        self._on_fm_changed()

    def _on_tone_value_changed(self) -> None:
        value = self._tone_value_combo.currentData()
        if value is None:
            return
        if self.tone_mode == "dcs":
            self._dcs_code = str(value)
        else:
            self._ctcss_hz = float(value)
        self._apply_rx_tone()
        self._on_fm_changed()

    def _apply_rx_tone(self) -> None:
        if self.audio is not None:
            self.audio.set_tone_squelch(self.rx_tone())
        if self.rx_tone() is None:
            self._tone_state.setText("")

    def _on_rpt_offset_edited(self, khz: float) -> None:
        if not self._fm_loading:
            self._rpt_offsets[self._rpt_band.name if self._rpt_band else None] = khz * 1e3
        self._on_fm_changed()

    def _on_fm_changed(self) -> None:
        self._sync_fm_row()
        if not self._fm_loading:
            self._schedule_save()

    def _sync_fm_row(self) -> None:
        """Follow the mode (NBFM only), the radio, and the band for the offset."""
        if not hasattr(self, "_fm_row"):
            return
        nbfm = self.mode == "nbfm"
        self._fm_row.setVisible(nbfm)
        self._rpt_box.setVisible(self.source.caps.tx is not None)
        band = band_for(self.listen_freq)
        if band != self._rpt_band:
            self._rpt_band = band
            key = band.name if band else None
            standard = band.offset_hz if band else self._rpt_offset_spin.value() * 1e3
            self._fm_loading, loading = True, self._fm_loading
            self._rpt_offset_spin.setValue(self._rpt_offsets.get(key, standard) / 1e3)
            self._fm_loading = loading
        shifted = self.repeater_shift != SIMPLEX
        self._rpt_offset_spin.setEnabled(shifted and not self.transmitting)
        self._tx_freq_label.setText(f"TX {self.tx_freq / 1e6:.4f} MHz" if shifted else "")

    def _update_tone_state(self) -> None:
        if self.rx_tone() is None or self.audio is None:
            return
        state = getattr(self.audio, "tone_open", None)
        if state is None:
            return
        self._tone_state.setText("\u25cf open" if state else "\u25cb closed")
        self._tone_state.setStyleSheet("color: #8ee6a0;" if state else "color: #888;")

    def _apply_fm_snapshot(self, snap: Snapshot) -> None:
        self._fm_loading = True
        try:
            self._ctcss_hz = float(snap.ctcss_hz) if snap.ctcss_hz in CTCSS_TONES else 88.5
            self._dcs_code = snap.dcs_code if snap.dcs_code in DCS_CODES else "023"
            self._shift_combo.setCurrentIndex(max(0, self._shift_combo.findData(
                snap.repeater_shift)))
            self._tone_combo.setCurrentIndex(max(0, self._tone_combo.findData(snap.tone_mode)))
            self._fill_tone_values()
            band = band_for(snap.freq_hz)
            self._rpt_band = band
            offset = (snap.repeater_offset_hz if snap.repeater_offset_hz is not None
                      else band.offset_hz if band else 600e3)
            self._rpt_offsets[band.name if band else None] = offset
            self._rpt_offset_spin.setValue(offset / 1e3)
        finally:
            self._fm_loading = False
        self._apply_rx_tone()
        self._sync_fm_row()

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

        self._pitch_label = QtWidgets.QLabel("Pitch")
        row.addWidget(self._pitch_label)
        self._pitch_combo = QtWidgets.QComboBox()
        for pitch in CW_PITCHES:
            self._pitch_combo.addItem(f"{pitch:.0f} Hz", pitch)
        index = self._pitch_combo.findData(self._initial_pitch)
        self._pitch_combo.setCurrentIndex(index if index >= 0 else
                                          self._pitch_combo.findData(500.0))
        self._pitch_combo.setToolTip("CW beat-note pitch, and what Zero beat tunes to")
        self._pitch_combo.currentIndexChanged.connect(self._on_pitch_changed)
        row.addWidget(self._pitch_combo)

        self._stereo_check = QtWidgets.QCheckBox("Stereo")
        self._stereo_check.setChecked(self._initial_stereo)
        self._stereo_check.setToolTip(
            "Decode stereo when the station sends it.\n"
            "Untick for mono, which is quieter on a weak station."
        )
        self._stereo_check.toggled.connect(self._on_stereo_toggled)
        row.addWidget(self._stereo_check)

        row.addWidget(QtWidgets.QLabel("Vol"))
        self._volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(int(self._initial_volume * 100))
        self._volume_slider.setFixedWidth(110)
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        row.addWidget(self._volume_slider)

        self._mute_button = QtWidgets.QPushButton("Mute")
        self._mute_button.setCheckable(True)
        self._mute_button.setFixedWidth(60)
        self._mute_button.setToolTip("Silence the output without losing the volume setting")
        self._mute_button.toggled.connect(self._on_mute_toggled)
        row.addWidget(self._mute_button)

        self._tx_button = QtWidgets.QPushButton("TX")
        self._tx_button.setCheckable(True)
        self._tx_button.setFixedWidth(60)
        self._tx_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._tx_button.setStyleSheet(
            "QPushButton { color: #ff4d4d; border: 2px solid #d62828; border-radius: 6px;"
            " font-weight: bold; padding: 3px; }"
            "QPushButton:checked { color: white; background-color: #d62828; }"
            "QPushButton:disabled { color: #7a4a4a; border-color: #5a3030; }"
        )
        self._tx_button.toggled.connect(self._on_tx_toggled)
        row.addWidget(self._tx_button)

        self._offset_label = QtWidgets.QLabel("Offset")
        row.addWidget(self._offset_label)
        self._offset_spin = QtWidgets.QDoubleSpinBox()
        self._offset_spin.setDecimals(2)
        self._offset_spin.setSuffix(" kHz")
        self._offset_spin.setSingleStep(1.0)
        self._offset_spin.setToolTip(
            "Listen this far from the tuned centre, without moving the radio.\n"
            "Keeps a wide span on screen while you hear one signal inside it,\n"
            "and keeps the listened-to signal clear of the centre (DC) spike."
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

        # First on the top line, so it is never off the edge of a narrow window.
        self._map_button = QtWidgets.QPushButton("Map")
        self._map_button.setToolTip("Show the map of ships and aircraft (AIS, ACARS, ADS-B)")
        self._map_button.setStyleSheet("QPushButton { font-weight: bold; padding: 2px 12px; }")
        self._map_button.clicked.connect(self.show_map)
        row.addWidget(self._map_button)
        # What is this signal? Yellow, top left beside Map (VK3RQ, 2026-10-06).
        self._classify_button = QtWidgets.QPushButton("?")
        self._classify_button.setToolTip(
            "What is this signal? Listens to the tuned frequency for a few seconds and\n"
            "names it -- P25, DMR, POCSAG, AIS, ACARS, ADS-B, packet, or by its\n"
            "modulation (WBFM, NBFM, AM, USB, LSB, CW) -- on the waterfall. A name with\n"
            "'?' is a judgement from the signal's shape; without, a decoder confirmed it.")
        self._classify_button.setStyleSheet(
            f"QPushButton {{ background: {DETECTED_COLOUR}; color: black; "
            "font-weight: bold; padding: 2px 10px; }"
            "QPushButton:disabled { background: #776f3a; color: #333; }")
        self._classify_button.clicked.connect(self.classify_signal)
        row.addWidget(self._classify_button)

        row.addSpacing(8)

        row.addWidget(QtWidgets.QLabel("SDR"))
        self._device_combo = DeviceCombo()
        # Sized to the radio names, not to "ADALM-Pluto -- driver not installed": the
        # status text is for the open list, and would otherwise widen the whole row.
        self._device_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._device_combo.setMinimumContentsLength(16)
        self._device_combo.view().setMinimumWidth(320)
        # The macOS style's own popup ignores item colours; a plain delegate draws the
        # yellow of detected radios.
        self._device_combo.setItemDelegate(QtWidgets.QStyledItemDelegate(self._device_combo))
        self._device_combo.aboutToShow.connect(self._refresh_device_list)
        self._device_combo.activated.connect(self._on_device_chosen)
        row.addWidget(self._device_combo)

        row.addWidget(QtWidgets.QLabel("Freq"))
        self._freq_spin = FrequencyDisplay()
        self._freq_spin.setDecimals(6)
        self._freq_spin.setSuffix(" MHz")
        self._freq_spin.setKeyboardTracking(False)
        self._freq_spin.valueChanged.connect(
            lambda mhz: self._retune(mhz * 1e6, from_spin=True, auto_radio=True)
        )
        # A dragged digit is an exact frequency: the step grid must not round it away.
        self._freq_spin.digitDragged.connect(
            lambda mhz: self._retune(mhz * 1e6, from_spin=True, allow_snap=False,
                                     auto_radio=True)
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

        self._snap_check = QtWidgets.QCheckBox("Snap")
        self._snap_check.setChecked(self._initial_snap)
        self._snap_check.setToolTip(
            "Round tuning to a multiple of the step, for channelised bands.\n"
            "Recalled memories and scanner hits are left exactly where they are."
        )
        self._snap_check.toggled.connect(self._on_snap_changed)
        row.addWidget(self._snap_check)
        self._on_step_changed()

        # Rate and hardware bandwidth always exist but are hidden when the radio has
        # nothing to choose between; switching radios repopulates them.
        self._rate_label = QtWidgets.QLabel("Rate")
        row.addWidget(self._rate_label)
        self._rate_combo = QtWidgets.QComboBox()
        self._rate_combo.currentIndexChanged.connect(self._on_rate_changed)
        row.addWidget(self._rate_combo)

        self._zoom_label = QtWidgets.QLabel("Zoom")
        row.addWidget(self._zoom_label)
        self._zoom_combo = QtWidgets.QComboBox()
        for factor in ZOOM_FACTORS:
            self._zoom_combo.addItem(f"{factor}x", factor)
        self._zoom_combo.setCurrentText(f"{self.decimator.factor}x")
        self._zoom_combo.setToolTip(
            "Decimate the IQ stream: narrower span, proportionally finer resolution"
        )
        self._zoom_combo.currentIndexChanged.connect(self._on_zoom_changed)
        row.addWidget(self._zoom_combo)

        self._scan_button = QtWidgets.QPushButton("Scan")
        self._scan_button.setCheckable(True)
        self._scan_button.setToolTip("Show or hide the scanner")
        row.addWidget(self._scan_button)
        self._decode_button = QtWidgets.QPushButton("Decode")
        self._decode_button.setCheckable(True)
        self._decode_button.setToolTip("Show or hide the data decoders (POCSAG, APRS, AIS, ACARS, ADS-B, P25, DMR)")
        row.addWidget(self._decode_button)

        # A transceiver's scope span, in place of Zoom: the radio's to set, from here too.
        self._span_label = QtWidgets.QLabel("Span")
        row.addWidget(self._span_label)
        self._span_combo = QtWidgets.QComboBox()
        for half in civ.SCOPE_SPANS_HZ:
            self._span_combo.addItem(f"\u00b1{half / 1e3:g} kHz", half)
        self._span_combo.setToolTip("The radio's scope span (centre mode)")
        self._span_combo.activated.connect(self._on_span_chosen)
        row.addWidget(self._span_combo)
        self._span_label.hide()
        self._span_combo.hide()

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
        self._rec_title = QtWidgets.QLabel("Record")
        row.addWidget(self._rec_title)
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

        row.addSpacing(20)
        self._play_button = QtWidgets.QPushButton("Play\u2026")
        self._play_button.setToolTip("Play an IQ recording in place of the radio")
        self._play_button.clicked.connect(self._on_play_clicked)
        row.addWidget(self._play_button)
        self._pause_button = QtWidgets.QPushButton("Pause")
        self._pause_button.setCheckable(True)
        self._pause_button.toggled.connect(self._on_pause_toggled)
        row.addWidget(self._pause_button)
        self._stop_play_button = QtWidgets.QPushButton("Stop")
        self._stop_play_button.setToolTip("Stop playing and go back to the radio")
        self._stop_play_button.clicked.connect(self.stop_playback)
        row.addWidget(self._stop_play_button)
        self._play_label = QtWidgets.QLabel("")
        row.addWidget(self._play_label)
        for widget in (self._pause_button, self._stop_play_button, self._play_label):
            widget.hide()

        row.addStretch(1)
        self._refresh_memories()
        return box

    def _build_display_row(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)

        self._fft_label = QtWidgets.QLabel("FFT")
        row.addWidget(self._fft_label)
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

        # The info line: decoded Morse in CW, stereo status and RDS in broadcast FM.
        # Beside Peak hold, where the row has room to spare. Right-aligned, so decoded CW
        # sits against the edge and grows leftwards like a ticker.
        row.addSpacing(12)
        self._info_label = QtWidgets.QLabel("")
        mono = QtGui.QFont("Menlo")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        mono.setPointSizeF(max(10.0, self._info_label.font().pointSizeF()))
        self._info_label.setFont(mono)
        self._info_label.setMinimumWidth(430)
        self._info_label.setMaximumWidth(620)
        self._info_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self._info_label.setStyleSheet("color: #8ee6a0;")
        self._info_label.setToolTip("Decoded CW, or stereo status and RDS for broadcast FM")
        row.addWidget(self._info_label)

        self._zerobeat_button = QtWidgets.QPushButton("Zero beat")
        self._zerobeat_button.setToolTip(
            "Hold to tune a nearby CW carrier onto the selected beat-note pitch.\n"
            f"Searches {ZEROBEAT_SEARCH_HZ:.0f} Hz either side. CW mode only."
        )
        self._zerobeat_button.pressed.connect(self._start_zerobeat)
        self._zerobeat_button.released.connect(self._stop_zerobeat)
        row.addWidget(self._zerobeat_button)

        row.addStretch(1)
        return bar

    def _build_device_controls(self) -> QtWidgets.QWidget:
        """Gain UI derived from probed capabilities."""
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        caps = self.source.caps
        self._radio_widgets = {}
        self._agc_check = None                 # set below for an SDR that has one
        if self.is_transceiver:
            self._build_transceiver_controls(row)
            return box

        self._agc_check = None
        if caps.has_agc:
            self._agc_check = QtWidgets.QCheckBox("AGC")
            self._agc_check.setChecked(getattr(self.source, "get_agc", lambda: False)())
            self._agc_check.toggled.connect(self._on_agc_toggled)
            row.addWidget(self._agc_check)

        for element in caps.gain_elements:
            if element.step_db and element.max_db - element.min_db == element.step_db:
                # Two values only -- the HackRF's AMP is 0 or 14 dB -- so it is a switch.
                check = QtWidgets.QCheckBox(element.name)
                check.setToolTip(f"{element.name}: +{element.max_db - element.min_db:g} dB")
                check.setChecked(self.source.get_gain(element.name) > element.min_db)
                check.toggled.connect(
                    lambda on, e=element: self.source.set_gain(
                        e.name, e.max_db if on else e.min_db)
                )
                check.toggled.connect(self._schedule_save)
                row.addWidget(check)
                continue
            row.addWidget(QtWidgets.QLabel(element.name))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(element.min_db, element.max_db)
            spin.setSingleStep(element.step_db or 1.0)
            spin.setSuffix(" dB")
            spin.setValue(self.source.get_gain(element.name))
            spin.valueChanged.connect(
                lambda value, name=element.name: self.source.set_gain(name, value)
            )
            spin.valueChanged.connect(self._schedule_save)
            row.addWidget(spin)

        # Boolean driver settings -- bias-tee and the like -- as the driver names them.
        for setting in getattr(caps, "settings", ()):
            check = QtWidgets.QCheckBox(setting.name)
            check.setToolTip(setting.description or setting.key)
            reader = getattr(self.source, "read_setting", None)
            check.setChecked(bool(reader(setting.key)) if reader else setting.default)
            writer = getattr(self.source, "write_setting", None)
            if writer is not None:
                check.toggled.connect(lambda on, key=setting.key: writer(key, on))
                check.toggled.connect(self._schedule_save)
            row.addWidget(check)

        # Nothing to control: say nothing. A "no gain controls" label only takes space
        # from things that matter.
        box.setVisible(row.count() > 0)
        return box

    def _build_transceiver_controls(self, row: QtWidgets.QHBoxLayout) -> None:
        """The radio's own settings (IC-705), as the source describes them."""
        state = self.source.state
        for control in self.source.controls:
            if getattr(control, "placement", "row") != "row":
                continue                        # function keys and their levels: below
            value = state.get(control.key)
            if control.kind == "switch":
                widget = QtWidgets.QCheckBox(control.label)
                widget.setChecked(bool(value))
                widget.toggled.connect(
                    lambda on, k=control.key: self.source.set_control(k, 1 if on else 0))
            elif control.kind == "choice":
                row.addWidget(QtWidgets.QLabel(control.label))
                widget = QtWidgets.QComboBox()
                for label, code in control.choices:
                    widget.addItem(label, code)
                if value is not None:
                    widget.setCurrentIndex(max(0, widget.findData(value)))
                widget.activated.connect(lambda _i, k=control.key, w=widget:
                                         self.source.set_control(k, w.currentData()))
            else:
                # Levels are 0-255 on the radio and 0-100% on its screen.
                row.addWidget(QtWidgets.QLabel(control.label))
                widget = QtWidgets.QSpinBox()
                widget.setRange(0, 100)
                widget.setSuffix(" %")
                widget.setKeyboardTracking(False)
                if value is not None:
                    widget.setValue(round(value / 255 * 100))
                widget.valueChanged.connect(lambda pct, k=control.key:
                                            self.source.set_control(k, round(pct / 100 * 255)))
            widget.setToolTip(control.tooltip)
            self._radio_widgets[control.key] = widget
            row.addWidget(widget)

        # The app's own: how loud the Mac microphone is sent to the radio. Not a radio
        # setting, so not in `_radio_widgets` (which follow the radio's state).
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Mic"))
        mic = QtWidgets.QSpinBox()
        mic.setRange(1, 100)
        mic.setSuffix(" %")
        mic.setKeyboardTracking(False)
        mic.setValue(round(self._tx_audio_level * 100))
        mic.setToolTip("Mac microphone level sent to the radio for TX. Lower it if the\n"
                       "audio is reported distorted or over-compressed; it can be\n"
                       "changed while transmitting.")
        mic.valueChanged.connect(self._on_tx_audio_level)
        self._mic_level_spin = mic
        row.addWidget(mic)
    def _on_tx_audio_level(self, percent: int) -> None:
        self._tx_audio_level = percent / 100.0
        audio_out = getattr(self.transmitter, "audio_out", None)
        if audio_out is not None:
            audio_out.level = self._tx_audio_level          # live, while transmitting
        self._schedule_save()

    def _on_span_chosen(self) -> None:
        half = self._span_combo.currentData()
        if half and self.is_transceiver:
            self.source.set_span(half)

    def _sync_span_combo(self) -> None:
        src = self.source
        half = getattr(src, "half_span", None)
        fixed = not getattr(src, "centre_mode", True)
        self._span_combo.setEnabled(not fixed)
        self._span_combo.setToolTip("The radio's scope span (centre mode)" if not fixed else
                                    "The radio's scope is in fixed mode: its edges set the span")
        if half is None or self._span_combo.view().isVisible():
            return
        index = self._span_combo.findData(half)
        if index >= 0 and index != self._span_combo.currentIndex():
            self._span_combo.blockSignals(True)
            self._span_combo.setCurrentIndex(index)
            self._span_combo.blockSignals(False)

    def _sync_transceiver_state(self) -> None:
        """Follow the radio: its knobs, mode, filter and meter, and any refusal."""
        src = self.source
        if hasattr(self, "radio_display"):
            self.radio_display.update_from(src)
            self.function_panel.sync()
        for key, widget in self._radio_widgets.items():
            value = src.state.get(key)
            if value is None or widget.hasFocus():
                continue                       # leave alone what is being edited
            widget.blockSignals(True)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentIndex(max(0, widget.findData(value)))
            else:
                widget.setValue(round(value / 255 * 100))
            widget.blockSignals(False)
        if src.mode is not None and src.mode != self.mode:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(max(0, self._mode_combo.findData(src.mode)))
            self._mode_combo.blockSignals(False)
            self._sync_mode_extras()
        if src.filter is not None and self._bw_audio_combo.currentData() != src.filter:
            self._bw_audio_combo.blockSignals(True)
            self._bw_audio_combo.setCurrentIndex(max(0, self._bw_audio_combo.findData(src.filter)))
            self._bw_audio_combo.blockSignals(False)
        if self.radio_audio is not None:
            # The 705 sends its audio unsquelched; silence it while its squelch is shut.
            self.radio_audio.squelch_open = getattr(src, "squelch_open", None) is not False
        if getattr(src, "transmitting", False):
            if src.po is not None:
                text = f"Po {power_percent(src.po):.0f}%"
                if src.swr is not None:
                    text += f"   SWR {swr_value(src.swr):.1f}"
                self.smeter.set_s_reading(power_percent(src.po) / 100, text)
        elif src.smeter is not None:
            self.smeter.set_s_reading(src.smeter / 255, s_meter_text(src.smeter))
        if src.refused:
            what, src.refused = src.refused, None
            # Kept in the status line itself: a one-off message is overwritten by the
            # next scope line's status update within a quarter of a second.
            self._radio_note = (f"IC-705 refused the {what} change in "
                                f"{(src.mode or '').upper()}", time.monotonic() + 5.0)

    def _start_radio_audio(self) -> None:
        """Play the transceiver's received audio on the Mac."""
        if self.radio_audio is not None or not self._audio_ok:
            return
        factory = self._radio_audio_factory
        link = getattr(self.source, "link", None)
        if factory is None and link is not None:
            from ..audio import NetworkRadioAudio

            def factory(volume: float):
                return NetworkRadioAudio(link, volume=volume)
        elif factory is None:
            from ..audio import RadioAudio

            factory = RadioAudio
        audio = factory(volume=self._volume_slider.value() / 100.0)
        audio.set_muted(self.muted)
        try:
            audio.start()
        except Exception as exc:
            self._audio_problem = f"radio audio: {exc}"
            return
        self.radio_audio = audio

    def _stop_radio_audio(self) -> None:
        audio, self.radio_audio = self.radio_audio, None
        if audio is not None:
            audio.stop()

    def _fill_mode_combo(self) -> None:
        """The app's demodulators for an SDR; the radio's own modes for a transceiver."""
        combo = self._mode_combo
        current = self.mode
        combo.blockSignals(True)
        combo.clear()
        if self.is_transceiver:
            for name in TRANSCEIVER_MODES:
                combo.addItem(name.upper(), name)
            wanted = getattr(self.source, "mode", None) or _TO_RADIO_MODE.get(current, current)
        else:
            combo.addItem("Off", "off")
            for name in MODES:
                combo.addItem(name.upper(), name)
            wanted = _TO_SDR_MODE.get(current, current)
            wanted = wanted if wanted in MODES else "off"
        combo.setCurrentIndex(max(0, combo.findData(wanted)))
        combo.blockSignals(False)

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

    def _apply_geometry(self, preserve_span: bool = False) -> None:
        """Re-place the display on the axes.

        `preserve_span` is used when only the tuning moved. Changing the sample rate or
        the decimation changes the span itself, which the user asked for explicitly, so
        those reset the view.
        """
        lines_per_s = SCOPE_LINES_PER_S if self.is_transceiver else float(self.fps)
        history_s = self.waterfall.buffer.rows / lines_per_s
        # A transceiver's scope has its own centre: in fixed mode, not the dial.
        centre = getattr(self.source, "display_center_freq", None) or self.source.center_freq
        self.waterfall.set_geometry(
            centre, self.effective_rate, history_s,
            preserve_span=preserve_span,
        )

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
        self._retune(target, auto_radio=True)
        return self.source.center_freq

    def fine_tune_limit(self) -> float:
        """Below this much movement, a tune is an adjustment rather than a change.

        One display bin wide. Smaller than that and the waterfall would shift by under a
        pixel, so its history is still honest; it is also small against any channel
        filter, so the demodulator's state remains valid.
        """
        return self.effective_rate / max(1, self.waterfall.buffer.cols)

    def _retune(
        self,
        hz: float,
        from_spin: bool = False,
        from_scan: bool = False,
        allow_snap: bool = True,
        auto_radio: bool = False,
    ) -> None:
        """Tune, dropping whatever described the old frequency.

        With `auto_radio` (a frequency the user chose), a radio that is there and
        reaches it better takes over first: the preferred radio wherever it reaches, else
        any that does (`radio_for`).

        A large move invalidates everything: the ring, the waterfall history, the
        spectrum smoothing and the audio in flight all describe the previous tuning, and
        keeping any of it smears a stale signal across the new span.

        A *fine* move does not. Nudging 100 Hz to pitch an SSB voice would otherwise
        clear the display and interrupt the audio on every step, which defeats the point
        of tuning by ear -- so below `fine_tune_limit` the history and the audio are left
        running and only the axis moves.
        """
        if self.transmitter is not None:
            # On a half-duplex radio the receiver and transmitter share one synthesizer:
            # retuning now would move the transmission.
            self._status.showMessage("stop TX before retuning", 3000)
            self._freq_spin.blockSignals(True)
            self._freq_spin.setValue(self.source.center_freq / 1e6)
            self._freq_spin.blockSignals(False)
            return
        if not from_scan and self.scanner is not None:
            # A manual tune means the user wants to stay here, so stop sweeping rather
            # than fighting them for the dial.
            self.stop_scan()
        if auto_radio and not self.playing_back and self._change_radio_for(float(hz)):
            return

        previous = self.source.center_freq
        requested = float(hz)
        if allow_snap and not from_scan:
            # Scanner hits are already on their own grid, and a recalled memory is an
            # exact frequency someone chose; neither should be moved.
            hz = self.snap_frequency(hz)
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
            self.waterfall.clear_label()          # it named what was here before
            self._cancel_classify()
            self._cancel_calibration()
            if self.audio is not None:
                self.audio.reset()
            if self.decode_worker is not None:
                self.decode_worker.reset()
        if self.decode_worker is not None:
            self.decode_worker.follow_tuning()       # fixed channels stay put
            self._update_decoder_status()

        self.spectrum.set_center_marker(actual)
        self._apply_geometry(preserve_span=True)
        self._update_passband()

        if not from_spin or abs(actual - requested) > 1.0:
            # Compared against what was *asked for*, not the snapped value: otherwise a
            # typed frequency stays in the box while the radio sits on the nearest
            # channel, and the display quietly disagrees with the hardware.
            self._freq_spin.blockSignals(True)
            self._freq_spin.setValue(actual / 1e6)
            self._freq_spin.blockSignals(False)
        if not from_scan:
            # Sweeping rewrites the frequency many times a second; that is not a
            # preference worth persisting.
            self._schedule_save()

    def _on_rate_changed(self) -> None:
        if self._rate_combo.currentData() is None:
            return
        self.source.set_sample_rate(self._rate_combo.currentData())
        self._refresh_hw_bandwidths()        # the driver may have moved it with the rate
        self.spectrum.reset()
        self.waterfall.clear_history()
        self._apply_geometry()
        self._update_offset_range()
        if self.audio is not None:
            # The chain's decimation and audio rate both derive from the sample rate.
            self.audio.restart()
        if self.decode_worker is not None:
            self._start_decoder(self.decode_worker.name)
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
        self._sync_fm_row()          # the band, hence the repeater offset, may have changed
        if self.is_transceiver:
            # The radio's filter widths are its own; nothing here knows them.
            self.spectrum.clear_passband()
            return
        mode = self.mode
        if mode == "off" or mode not in MODE_SPECS:
            self.spectrum.clear_passband()
            return
        width = self._channel_bandwidth()
        centre = self.source.center_freq + self._offset_spin.value() * 1e3
        if mode == "cw":
            # Centred on the tuned frequency: the BFO is folded into the mixer, so a
            # carrier *there* is what comes out at the pitch. (It was once drawn at
            # centre + pitch, 500 Hz right of what was actually heard.)
            self.spectrum.set_passband(centre, width)
            return
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
        self._audio_problem = ""
        if self.is_transceiver:
            radio_mode = _TO_RADIO_MODE.get(mode, mode)
            if radio_mode in civ.MODE_CODES and radio_mode != getattr(self.source, "mode", None):
                self.source.set_mode(radio_mode)
            self._refresh_bandwidths()
            self._sync_mode_extras()
            self._update_passband()
            return
        if self.transmitter is not None and mode != self.transmitter.mode:
            self._tx_button.setChecked(False)      # a different mode means stop sending
        if mode == "off":
            if self.audio is not None:
                self.audio.stop()
                self.audio = None
        elif not self._audio_ok:
            self._audio_problem = "no audio device available"
        elif self.audio is None:
            sink = AudioSink(
                self.source, mode=mode,
                offset_hz=self._offset_spin.value() * 1e3,
                volume=self._volume_slider.value() / 100.0,
                squelch_dbfs=self._squelch_value(),
                bandwidth_hz=self.bandwidth_hz(),
                pitch_hz=self.pitch_hz,
            )
            sink.set_force_mono(not self._stereo_check.isChecked())
            try:
                sink.set_tone_squelch(self.rx_tone())
                sink.start()
                sink.set_muted(self.muted)
                self.audio = sink
            except Exception as exc:
                self._audio_problem = f"could not start audio: {exc}"
        else:
            try:
                self.audio.set_mode(mode)
                self.audio.set_tone_squelch(self.rx_tone())
            except Exception as exc:
                self._audio_problem = f"could not switch mode: {exc}"

        if self._audio_problem:
            self._status.showMessage(self._audio_problem, 4000)   # and kept in the line
        self._sync_squelch_enabled()
        self._refresh_bandwidths()
        self._sync_zerobeat_enabled()
        self._sync_pitch_visible()
        self._sync_mode_extras()
        self._update_passband()

    def _on_mode_changed(self) -> None:
        self.set_mode(self.mode)
        self._schedule_save()

    @property
    def pitch_hz(self) -> float:
        return float(self._pitch_combo.currentData() or 500.0)

    def _sync_mode_extras(self) -> None:
        """Show only what the current mode can use.

        Pitch, zero beat and decoded Morse belong to CW; the stereo switch belongs to
        broadcast FM; the info line serves both.
        """
        mode = self.mode
        self._sync_fm_row()
        self._zerobeat_button.setVisible(mode == "cw")
        self._stereo_check.setVisible(mode == "wbfm")
        # Decoding is the app's own, so none for a transceiver: the radio demodulates.
        show = mode in ("cw", "wbfm") and not self.is_transceiver
        self._info_label.setVisible(show)
        if not show:
            self._info_label.setText("")
            self._info_label.setToolTip("")

    def _update_info_line(self) -> None:
        if self.audio is None:
            return
        if self.mode == "cw":
            text = self.audio.cw_text
            wpm = self.audio.cw_wpm
            self._info_label.setText(f"{text}   [{wpm:.0f} wpm]" if text else "")
        elif self.mode == "wbfm":
            self._show_broadcast_info()

    def _show_broadcast_info(self) -> None:
        """STEREO or MONO, then the RDS station name, programme type and radio text."""
        rds = self.audio.rds
        parts = ["STEREO" if self.audio.stereo else "MONO"]
        detail = []
        if rds is not None:
            if rds.ps_name:
                parts.append(rds.ps_name)
            if rds.pty_name:
                parts.append(rds.pty_name)
            if rds.pi is not None:
                detail.append(f"PI {rds.pi:04X}")
            if rds.radio_text:
                detail.append(rds.radio_text)
        text = "  \u00b7  ".join(parts)
        if rds is not None and rds.radio_text:
            text += "   " + rds.radio_text
        # Elided rather than allowed to widen the row: radio text runs to 64 characters.
        metrics = QtGui.QFontMetrics(self._info_label.font())
        width = max(100, self._info_label.maximumWidth() - 8)
        self._info_label.setText(
            metrics.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, width)
        )
        self._info_label.setToolTip("\n".join(["  \u00b7  ".join(parts)] + detail))

    # Kept for callers and tests written before the line served FM as well.
    _sync_cw_label = _sync_mode_extras
    _update_cw_text = _update_info_line

    def _sync_pitch_visible(self) -> None:
        """Only CW has a beat note, so only CW shows the control."""
        show = self.mode == "cw"
        self._pitch_label.setVisible(show)
        self._pitch_combo.setVisible(show)

    def _on_stereo_toggled(self, enabled: bool) -> None:
        if self.audio is not None:
            self.audio.set_force_mono(not enabled)
        self._schedule_save()

    def _on_pitch_changed(self) -> None:
        if self.audio is not None:
            self.audio.set_pitch(self.pitch_hz)
        self._update_passband()
        self._schedule_save()

    @property
    def snap_enabled(self) -> bool:
        return self._snap_check.isChecked()

    def _on_snap_changed(self, enabled: bool) -> None:
        self._schedule_save()
        if enabled:
            # Apply at once, so ticking it visibly does something.
            self._retune(self.source.center_freq)

    def snap_frequency(self, hz: float) -> float:
        """Round to the nearest multiple of the step, when snapping is on."""
        if not self.snap_enabled:
            return float(hz)
        step = self.step_hz
        if step <= 0:
            return float(hz)
        return round(float(hz) / step) * step

    @property
    def muted(self) -> bool:
        return self._mute_button.isChecked()

    # -- transmit ------------------------------------------------------------

    def _sync_tx_enabled(self) -> None:
        tx = self.source.caps.tx
        self._tx_button.setEnabled(tx is not None)
        if tx is None:
            self._tx_button.setToolTip(f"{self.source.caps.label or 'This radio'} cannot transmit")
        else:
            self._tx_button.setToolTip(
                "Transmit on the listening frequency (space bar or click to toggle).\n"
                "Microphone audio, TX gains as set on the Radio row.\n"
                f"Modes: {', '.join(m.upper() for m in TX_MODES)}. "
                f"Stops itself after {TX_TIMEOUT_S / 60:.0f} minutes.")

    @property
    def transmitting(self) -> bool:
        return self.transmitter is not None

    @property
    def listen_freq(self) -> float:
        """Where the receiver is listening, which is where TX goes."""
        return self.source.center_freq + self._offset_spin.value() * 1e3

    def _make_transmitter(self, mode: str) -> Transmitter:
        """Microphone -> modulator -> this radio's transmitter, on the listening frequency.

        A source with no transmit path (a test stand-in) gives a dry run instead.
        """
        if self.is_transceiver:
            # The radio modulates: the app keys it and feeds it the microphone.
            from ..audio import CodecOutput, NetworkAudioOutput

            link = getattr(self.source, "link", None)
            audio_out = (NetworkAudioOutput(link, level=self._tx_audio_level) if link is not None
                         else CodecOutput(level=self._tx_audio_level))
            return RadioTransmitter(self.source, _TO_RADIO_MODE.get(mode, mode),
                                    audio_out=audio_out)
        opener = getattr(self.source, "open_tx_sink", None)
        sink = opener(self.tx_freq, TX_IQ_RATE, dict(self._tx_gains)) if opener else None
        return Transmitter(mode, sink=sink, tone=self.tx_tone(),
                           mic_gain_db=self._tx_mic_gain_db)

    def _set_tx_mic_gain(self, db: float) -> None:
        self._tx_mic_gain_db = float(db)
        modulator = getattr(self.transmitter, "modulator", None)
        if modulator is not None:
            modulator.set_mic_gain_db(self._tx_mic_gain_db)     # live, while transmitting
        self._schedule_save()

    def _reset_tx_gains(self) -> None:
        from ..dsp.modulate import MIC_GAIN_DB

        self._tx_mic_gain_db = MIC_GAIN_DB
        tx = self.source.caps.tx
        self._tx_gains = {g.name: g.min_db for g in tx.gain_elements} if tx else {}
        self._rebuild_tx_controls()

    def _set_tx_gain(self, name: str, db: float) -> None:
        self._tx_gains[name] = float(db)
        sink = getattr(self.transmitter, "sink", None)
        if sink is not None:
            sink.set_gain(name, float(db))      # live, while transmitting
        self._schedule_save()

    def _rebuild_tx_controls(self) -> None:
        layout = self._tx_slot_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        tx = self.source.caps.tx
        if tx is not None and tx.gain_elements:
            title = QtWidgets.QLabel("TX")
            title.setStyleSheet("color: #ff4d4d; font-weight: bold;")
            layout.addWidget(title)
            for element in tx.gain_elements:
                value = self._tx_gains.get(element.name, element.min_db)
                if element.step_db and element.max_db - element.min_db == element.step_db:
                    check = QtWidgets.QCheckBox(f"TX {element.name}")
                    check.setToolTip(f"Transmit {element.name}: +{element.max_db:g} dB")
                    check.setChecked(value > element.min_db)
                    check.toggled.connect(lambda on, e=element: self._set_tx_gain(
                        e.name, e.max_db if on else e.min_db))
                    layout.addWidget(check)
                    continue
                layout.addWidget(QtWidgets.QLabel(f"TX {element.name}"))
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(element.min_db, element.max_db)
                spin.setSingleStep(element.step_db or 1.0)
                spin.setSuffix(" dB")
                spin.setValue(value)
                spin.setToolTip(f"Transmit {element.name} gain")
                spin.valueChanged.connect(lambda db, n=element.name: self._set_tx_gain(n, db))
                layout.addWidget(spin)
            from ..dsp.modulate import MIC_GAIN_RANGE_DB

            layout.addWidget(QtWidgets.QLabel("Mic"))
            mic = QtWidgets.QDoubleSpinBox()
            mic.setRange(*MIC_GAIN_RANGE_DB)
            mic.setSingleStep(1.0)
            mic.setDecimals(0)
            mic.setSuffix(" dB")
            mic.setValue(self._tx_mic_gain_db)
            mic.setToolTip("Microphone gain before the modulator. A limiter holds peaks at\n"
                           "full deviation, so raise it until the audio is loud enough;\n"
                           "\"drive\" in the status line shows how close peaks come.\n"
                           "Can be changed while transmitting.")
            mic.valueChanged.connect(self._set_tx_mic_gain)
            self._tx_mic_spin = mic
            layout.addWidget(mic)
        self._tx_slot.setVisible(tx is not None and bool(tx.gain_elements))
        self._sync_radio_row()

    def _set_tx_lock(self, locked: bool) -> None:
        """While keyed, freeze what would move or disturb the transmission."""
        for widget in (self._freq_spin, self._rate_combo, self._device_combo,
                       self._offset_spin, self._memory_combo, self._device_slot,
                       self._bw_combo, self._scan_button, self._shift_combo,
                       self._tone_combo, self._tone_value_combo):
            widget.setEnabled(not locked)
        self._sync_fm_row()

    def _on_tx_toggled(self, on: bool) -> None:
        if on:
            self._start_transmit()
        else:
            self._stop_transmit()

    def _start_transmit(self) -> None:
        def refuse(message: str) -> None:
            self._tx_button.blockSignals(True)
            self._tx_button.setChecked(False)
            self._tx_button.blockSignals(False)
            self._status.showMessage(f"cannot transmit: {message}", 6000)

        if self.source.caps.tx is None:
            refuse(f"{self.source.caps.label or 'this radio'} has no transmitter")
            return
        try:
            tx = self._transmitter_factory(self.mode)
            tx.start()
        except Exception as exc:
            refuse(str(exc))
            return
        self.transmitter = tx
        self._set_tx_lock(True)
        if self.scanner is not None:
            self.stop_scan()
        # Half duplex: the receiver goes quiet while transmitting. Muted rather than
        # stopped, so it comes straight back when TX ends -- and a transceiver's audio
        # cannot reach the microphone through the speakers.
        if self.audio is not None:
            self.audio.set_muted(True)
        if self.radio_audio is not None:
            self.radio_audio.set_muted(True)

    def _stop_transmit(self) -> None:
        tx, self.transmitter = self.transmitter, None
        if tx is not None:
            tx.stop()
            self._set_tx_lock(False)
        if self.audio is not None:
            self.audio.set_muted(self.muted)
        if self.radio_audio is not None:
            self.radio_audio.set_muted(self.muted)

    def _on_mute_toggled(self, muted: bool) -> None:
        if self.audio is not None:
            self.audio.set_muted(muted or self.transmitting)
        if self.radio_audio is not None:
            self.radio_audio.set_muted(muted or self.transmitting)
        self._mute_button.setText("Muted" if muted else "Mute")

    def _on_volume_changed(self, value: int) -> None:
        if self.audio is not None:
            self.audio.set_volume(value / 100.0)
        if self.radio_audio is not None:
            self.radio_audio.set_volume(value / 100.0)
        self._schedule_save()

    def _on_offset_changed(self, khz: float) -> None:
        if self.audio is not None:
            self.audio.set_offset(khz * 1e3)
            self.audio.reset()
        if self.decode_worker is not None:
            self.decode_worker.set_offset(khz * 1e3)
            self._update_decoder_status()
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
        if self.is_transceiver:
            # The radio's own three filters; their widths are set on the radio.
            for number in (1, 2, 3):
                self._bw_audio_combo.addItem(f"FIL{number}", number)
            current = getattr(self.source, "filter", None) or 1
            self._bw_audio_combo.setCurrentIndex(self._bw_audio_combo.findData(current))
            self._bw_audio_combo.blockSignals(False)
            self._bw_audio_combo.setEnabled(True)
            return
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
        if self.is_transceiver:
            return None                        # FIL1-3 are filter numbers, not widths
        data = self._bw_audio_combo.currentData()
        return float(data) if data is not None else None

    def _on_audio_bandwidth_changed(self) -> None:
        if self.is_transceiver:
            number = self._bw_audio_combo.currentData()
            if number:
                self.source.set_mode(_TO_RADIO_MODE.get(self.mode, self.mode), number)
            return
        width = self.bandwidth_hz()
        if width is None:
            return
        if self.audio is not None:
            self.audio.set_bandwidth(width)
        self._update_passband()
        self._schedule_save()

    # -- CW zero beat ------------------------------------------------------

    def _sync_zerobeat_enabled(self) -> None:
        """Only meaningful in CW: every other mode has no beat note to centre, so the
        button is hidden rather than greyed out (see _sync_mode_extras)."""
        self._zerobeat_button.setVisible(self.mode == "cw")

    def _start_zerobeat(self) -> None:
        if self.mode != "cw":
            return
        self._zerobeat_timer.start(ZEROBEAT_INTERVAL_MS)
        self._zerobeat_step()          # act at once rather than after the first interval

    def _stop_zerobeat(self) -> None:
        self._zerobeat_timer.stop()

    def zerobeat_once(self) -> float | None:
        """Measure the nearby carrier and correct the tuning once.

        Returns the correction applied in Hz, or None when there was nothing to tune to.
        The correction is signed, so the radio moves up or down as needed; it is applied
        in one go because the measurement is accurate to a couple of hertz, and hunting
        in fixed steps would only be slower.
        """
        if self.mode != "cw":
            return None
        listen_hz = self._offset_spin.value() * 1e3
        iq = self.source.read_latest(ZEROBEAT_FFT)
        if iq.size < ZEROBEAT_FFT:
            return None
        found = measure_carrier(
            iq,
            self.source.sample_rate,
            listen_hz=listen_hz,
            search_hz=ZEROBEAT_SEARCH_HZ,
        )
        if found is None:
            self._status.showMessage("zero beat: no signal within 500 Hz", 2000)
            return None
        if abs(found.error_hz) <= ZEROBEAT_DEADBAND_HZ:
            self._status.showMessage(
                f"zero beat: on tune ({found.snr_db:.0f} dB S/N)", 2000
            )
            return 0.0
        # Bounded by the search width, so a bad measurement cannot throw the radio.
        correction = float(np.clip(found.error_hz, -ZEROBEAT_SEARCH_HZ, ZEROBEAT_SEARCH_HZ))
        # Snapping is skipped deliberately: a channel grid is exactly what zero-beating
        # has to ignore.
        self._retune(self.source.center_freq + correction, allow_snap=False)
        self._status.showMessage(
            f"zero beat: {correction:+.0f} Hz ({found.snr_db:.0f} dB S/N)", 2000
        )
        return correction

    def _zerobeat_step(self) -> None:
        self.zerobeat_once()

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
        recorder = AudioRecorder(self.recordings_dir / name, self.audio.audio_rate,
                                 channels=self.audio.channels)
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
            snap=self.snap_enabled,
            pitch_hz=self.pitch_hz,
            stereo=self._stereo_check.isChecked(),
            repeater_shift=self.repeater_shift,
            repeater_offset_hz=self._rpt_offset_spin.value() * 1e3,
            tone_mode=self.tone_mode,
            ctcss_hz=self._ctcss_hz,
            dcs_code=self._dcs_code,
            decoder=self.decoder_panel.decoder,
        )

    def apply_snapshot(self, snap: Snapshot) -> None:
        """Put the receiver back into a stored state.

        Rate first: changing it restarts the stream, which would otherwise undo the
        frequency and zoom set afterwards.
        """
        if self._rate_combo.count() > 1 and snap.sample_rate != self.source.sample_rate:
            self.source.set_sample_rate(snap.sample_rate)
            self._rate_combo.blockSignals(True)
            index = self._rate_combo.findData(self.source.sample_rate)
            if index >= 0:
                self._rate_combo.setCurrentIndex(index)
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

        self._retune(snap.freq_hz, allow_snap=False)

        self._initial_bandwidth = snap.bandwidth_hz
        self._snap_check.blockSignals(True)
        self._snap_check.setChecked(snap.snap)
        self._snap_check.blockSignals(False)
        self._stereo_check.blockSignals(True)
        self._stereo_check.setChecked(snap.stereo)
        self._stereo_check.blockSignals(False)
        pitch_index = self._pitch_combo.findData(snap.pitch_hz)
        if pitch_index >= 0:
            self._pitch_combo.blockSignals(True)
            self._pitch_combo.setCurrentIndex(pitch_index)
            self._pitch_combo.blockSignals(False)
        index = self._step_combo.findData(snap.step_hz)
        if index >= 0:
            self._step_combo.blockSignals(True)
            self._step_combo.setCurrentIndex(index)
            self._step_combo.blockSignals(False)
            self._on_step_changed()
        self._apply_fm_snapshot(snap)
        if self.is_transceiver:
            wanted = _TO_RADIO_MODE.get(snap.mode, snap.mode)
        else:
            wanted = _TO_SDR_MODE.get(snap.mode, snap.mode)
            wanted = wanted if wanted in MODES else "off"
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

    def tune_radio_memory(self, freq_hz: float, radio_mode: str) -> None:
        """Tune to one of the transceiver's own memory channels, as its window asks."""
        # The mode list holds the radio's own names for a transceiver, the app's for an SDR.
        mode = radio_mode if self.is_transceiver else _TO_SDR_MODE.get(radio_mode, radio_mode)
        index = self._mode_combo.findData(mode)
        if index >= 0:
            self._mode_combo.setCurrentIndex(index)
        else:
            self.set_mode(mode)
        self._offset_spin.setValue(0.0)
        self._retune(freq_hz, allow_snap=False)

    def save_radio_memory(self, name: str, freq_hz: float, radio_mode: str) -> None:
        """Copy a transceiver memory channel into the app's memories: tune it, then save
        the app's settings there under `name`."""
        self.tune_radio_memory(freq_hz, radio_mode)
        self.save_memory(name)

    def save_memory(self, name: str) -> bool:
        """Store the current settings. Returns True if an existing name was replaced."""
        replaced = self.settings.add_memory(
            name, self.current_snapshot(), self.current_device_key(),
            self.current_radio_settings(),
        )
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
        wanted = memory.snapshot.freq_hz
        if not self.source.caps.covers(wanted) and not self.playing_back:
            # Out of this radio's reach: change to the first connected radio the memory
            # was set up for that can tune it -- "ADS-B 1090" brings the Pluto in.
            connected = {a.profile.key for a in self._availability_fn() if a.connected}
            for radio_key in memory.radios:
                profile = profile_for(radio_key)
                if (radio_key in connected and profile is not None
                        and profile.covers(wanted) and radio_key != self.current_device_key()):
                    if self.switch_device(radio_key, chosen=False):
                        break
        key = self.current_device_key()
        radio = memory.radios.get(key)
        note = ""
        if radio is None:
            radio = self._radio_settings_for_new_station(memory.snapshot)
            note = " -- first time on this radio, using its own settings"
        snap = dataclasses.replace(
            memory.snapshot,
            sample_rate=radio.sample_rate or self.source.sample_rate,
            decimation=radio.decimation,
            min_db=radio.min_db,
            max_db=radio.max_db,
            agc=bool(radio.agc) if radio.agc is not None else memory.snapshot.agc,
        )
        self.apply_snapshot(snap)
        self.apply_radio_hardware(radio)
        self._apply_decoder(memory.snapshot.decoder)
        if not self.source.caps.covers(memory.snapshot.freq_hz):
            note = (f" -- {memory.snapshot.freq_hz / 1e6:.4f} MHz is outside this "
                    f"radio's range ({self.source.caps.describe_ranges()})")
            meant = [profile_for(k).label for k in memory.radios
                     if profile_for(k) is not None and profile_for(k).covers(wanted)]
            if meant:
                note += f"; it is set up for the {' or '.join(meant)}: connect one"
        self._status.showMessage(f"recalled {memory.name}{note}", 6000)
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
        # A recording is not a radio: saving would make the next launch open the file's
        # frequency on the real radio, or try to open "file" as a driver.
        if not self._persist or self.playing_back:
            return
        self.settings.last = self.current_snapshot()
        self.settings.device = self.current_device_key()     # the radio to reopen
        self.settings.radios[self.current_device_key()] = self.current_radio_settings()
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
        if self.transmitter is not None and self.transmitter.expired():
            self._tx_button.setChecked(False)
            self._status.showMessage(
                f"TX stopped: {TX_TIMEOUT_S / 60:.0f} minute transmit timeout", 8000)
        if self.is_transceiver:
            self._scope_frame()
            return
        iq = self.source.read_latest(self._frame_request())
        if iq.size < self.decimator.input_for_output(self.analyzer.fft_size):
            # A half-duplex radio receives nothing while it transmits.
            self._status.showMessage(self._tx_status().lstrip(" |") if self.transmitter
                                     else "waiting for samples...")
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
        self._update_info_line()
        self._update_tone_state()
        self._collect_decoded()
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
            self._update_play_label()
            if self._recording_active():
                self._update_recording_label()

    def _scope_frame(self) -> None:
        """A transceiver's display: its own scope lines, drawn as they arrive (~4/s)."""
        self._sync_transceiver_state()
        line = self.source.take_scope_line()
        if line is None:
            return
        n = line.amplitudes.size
        # Pixel centres across the scope's span.
        freqs = line.low_hz + (np.arange(n) + 0.5) * (line.span_hz / n)
        dbfs = scope_level_db(line.amplitudes)
        geometry = (line.centre_hz, line.span_hz)
        if geometry != self._scope_geometry:
            # The radio's dial or span moved: re-place the display on the new axis. A new
            # span is a new picture -- the old rows were drawn at another scale, and a
            # zoom kept from the old span would hide most of the new one.
            span_changed = (self._scope_geometry is None
                            or line.span_hz != self._scope_geometry[1])
            self._scope_geometry = geometry
            if self.waterfall.buffer.cols != n:
                self.waterfall.resize_bins(n)
            if span_changed:
                self.waterfall.clear_history()
                self.spectrum.reset()
            self._apply_geometry(preserve_span=not span_changed)
            self._sync_span_combo()
            self.spectrum.set_center_marker(self.source.center_freq)
            self._freq_spin.blockSignals(True)
            self._freq_spin.setValue(self.source.center_freq / 1e6)
            self._freq_spin.blockSignals(False)
            self._update_passband()
        self.spectrum.update_spectrum(freqs, dbfs)
        self.waterfall.push(dbfs)
        self._rows_pushed += 1
        if self._auto_pending and self._rows_pushed >= 8:
            self._auto_pending = False
            self._on_auto_levels()
        self._update_status(dbfs)

    def _status_free(self) -> bool:
        """Whether the frame's own line may replace what the status bar shows. A notice
        ("now using ...", "stop TX before retuning") was overwritten within a frame,
        25 times a second, so none was ever seen: one is kept until its timeout clears
        it, or for NOTICE_HOLD_S if it has none."""
        shown = self._status.currentMessage()
        if not shown or shown == getattr(self, "_frame_status", None):
            return True
        if shown != getattr(self, "_notice", None):
            self._notice, self._notice_since = shown, time.monotonic()
        return time.monotonic() - self._notice_since > NOTICE_HOLD_S

    def _update_status(self, dbfs) -> None:
        if not self._status_free():
            return
        stats = getattr(self.source, "stats", {})
        span = self.effective_rate
        self._frame_status = (
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
        self._status.showMessage(self._frame_status)

    def _tx_status(self) -> str:
        tx = self.transmitter
        if tx is None:
            return ""
        level = getattr(tx.mic, "level_dbfs", -200.0)
        elapsed = int(tx.elapsed_s)
        what = ("TX dry run, no RF" if tx.dry_run
                else f"TX {tx.tx_freq / 1e6:.4f} MHz")
        drive = getattr(getattr(tx, "modulator", None), "drive", None)
        shown = f"  drive {drive * 100:.0f}%" if drive is not None else ""
        return (f"  |  {what}  {tx.mode.upper()}  mic {level:.0f} dBFS{shown}"
                f"  {elapsed // 60}:{elapsed % 60:02d}")

    def _audio_status(self) -> str:
        if self.transmitter is not None:
            return self._tx_status()
        # The status line is rewritten every frame, so a one-off message would vanish
        # before it could be read: why there is no sound has to live here instead.
        if self.audio is None and self.is_transceiver:
            text = f"  |  IC-705 scope, {self.source.stats.get('lines', 0)} lines"
            if self.radio_audio is not None:
                a = self.radio_audio.stats
                text += f"  |  audio {a['audio_rate'] / 1e3:.0f} kHz  ur {int(a['underrun_samples'])}"
            elif self._audio_problem:
                text += f"  |  {self._audio_problem}"
            note = getattr(self, "_radio_note", None)
            if note is not None and time.monotonic() < note[1]:
                text += f"  |  {note[0]}"
            link = getattr(self.source, "link", None)
            if link is not None and link.error:
                text += f"  |  WiFi: {link.error} -- choose the radio again to reconnect"
            elif link is not None:
                lost = link.stats
                if lost["lost_civ"] or lost["lost_audio"]:
                    text += f"  |  WiFi lost {lost['lost_civ']} CI-V, {lost['lost_audio']} audio"
            return text
        if self.audio is None:
            if self.mode == "off":
                return "  |  audio off (choose a Mode)"
            return f"  |  no audio: {self._audio_problem or 'not running'}"
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
        self._tx_button.setChecked(False)
        self._stop_radio_audio()
        QtWidgets.QApplication.instance().removeEventFilter(self._space_filter)
        self._timer.stop()
        self._zerobeat_timer.stop()
        self._save_timer.stop()
        self._save_state()   # immediately, not debounced: there is no later
        if self.scanner is not None:
            self.scanner.stop()
            self.scanner = None
        # Drop the shared-axis link before tearing down. pyqtgraph keeps a registry of
        # linked views, and leaving a closed one in it is a dangling reference.
        try:
            self.waterfall.setXLink(None)
        except Exception:
            pass
        self.stop_audio_recording()
        self.stop_iq_recording()
        self._stop_decoder()
        if self.map_window is not None:
            self.map_window.close()
        if self.audio is not None:
            self.audio.stop()
            self.audio = None
        self.source.close()
        super().closeEvent(event)


def run(source: IQSource, debug_gestures: bool = False, **kwargs) -> int:
    """Start the Qt app against an already-configured source."""
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Conventional Qt app identity. Note this does *not* rename the macOS Dock tile: the
    # desktop shortcut execs a system interpreter, so macOS attributes the running process
    # to Python.framework and the Dock says "Python". Fixing that needs the interpreter
    # bundled inside the .app, which is more than a launcher shortcut warrants.
    app.setApplicationName("RGC SDR")
    app.setApplicationDisplayName(APP_TITLE)
    icon_path = Path(__file__).resolve().parents[3] / "assets" / "icon.png"
    if icon_path.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    if debug_gestures:
        from .gesture_debug import install

        install(app)
    source.start()
    kwargs.setdefault("auto_calibrate", True)
    window = MainWindow(source, **kwargs)
    window.resize(1280, 800)
    window.show()
    try:
        return app.exec()
    finally:
        # The window may have switched radios, so close the one it ended up with.
        window.source.close()
