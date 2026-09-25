"""UI tests, run windowless via QT_QPA_PLATFORM=offscreen (set in conftest).

`StubSource` is test scaffolding for the widgets -- a test double for the IQSource
protocol, not a simulated device mode in the application (PLANNING.md section 1). It lets
the UI be regression-tested without the radio attached.
"""

import gc

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from src.rgc_sdr.device.source import (  # noqa: E402
    DeviceCaps,
    FreqRange,
    GainElement,
    IQSource,
    SequentialReader,
    _Ring,
)
from src.rgc_sdr.dsp.spectrum import SpectrumAnalyzer  # noqa: E402
from src.rgc_sdr.settings import Settings, Snapshot  # noqa: E402
from src.rgc_sdr.ui.main_window import MainWindow  # noqa: E402


class StubSource(IQSource):
    """A test double for IQSource: one tone over a noise floor.

    The noise matters. A noiseless tone leaves most bins at numerical zero (~-294 dBFS),
    which no real receiver produces and which skews anything that fits a range to the
    data. Seeded, so tests stay deterministic.
    """

    def __init__(self, caps, rate=768e3, center=7.1e6, tone_hz=96e3, noise=1e-4):
        self._caps, self._rate, self._center, self._tone = caps, rate, center, tone_hz
        self._noise = noise
        self._rng = np.random.default_rng(1234)
        # Generated samples are mirrored into a ring so gapless consumers (audio, IQ
        # recording) see the same stream the display does, as on a real source.
        self._ring = _Ring(400_000)

    @property
    def caps(self):
        return self._caps

    @property
    def sample_rate(self):
        return self._rate

    @property
    def center_freq(self):
        return self._center

    @property
    def stats(self):
        return {"samples": 1, "overflows": 0, "timeouts": 0, "errors": 0}

    def start(self):
        pass

    def stop(self):
        pass

    def set_agc(self, enabled):
        self.agc = enabled

    def set_gain(self, name, db):
        self.gain = (name, db)

    def get_gain(self, name):
        return 0.0

    def set_center_freq(self, hz, flush=True):
        self._center = self._caps.clamp_freq(hz)
        self.last_flush = flush
        if flush:
            self._ring.clear()
        return self._center

    def set_sample_rate(self, hz):
        self._rate = self._caps.nearest_sample_rate(hz)
        return self._rate

    def set_bandwidth(self, hz):
        self.bandwidth_set = hz
        return hz

    def sequential_reader(self):
        return SequentialReader(self._ring)

    def read_latest(self, n):
        t = np.arange(n) / self._rate
        tone = 0.5 * np.exp(2j * np.pi * self._tone * t)
        noise = self._noise * (self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n))
        block = (tone + noise).astype(np.complex64)
        self._ring.write(block)
        return block


def _caps(**kw):
    base = dict(
        driver="airspyhf", label="AirSpy HF+", serial="x",
        sample_rates=(768e3,), freq_ranges=(FreqRange(9e3, 31e6),),
        gain_elements=(), has_agc=True, formats=("CF32",),
    )
    base.update(kw)
    return DeviceCaps(**base)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


#: Windows created by a test, destroyed by the autouse fixture below.
_OPEN_WINDOWS = []


def window_for(src, **kw):
    """MainWindow with audio off: tests must not open a real output device."""
    kw.setdefault("enable_audio", False)
    win = MainWindow(src, **kw)
    _OPEN_WINDOWS.append(win)
    return win


@pytest.fixture(autouse=True)
def _destroy_windows(qapp):
    """Destroy each test's windows rather than leaking them.

    close() only hides a window; the C++ object survives until garbage collection. With
    a hundred-odd windows accumulating in one process, pyqtgraph's registry of
    axis-linked views ends up holding closed ones, and following a stale link segfaults.
    Leaked widgets also made failures depend on test order.
    """
    yield
    while _OPEN_WINDOWS:
        win = _OPEN_WINDOWS.pop()
        try:
            win.close()
            win.setParent(None)
            win.deleteLater()
        except RuntimeError:
            pass          # already deleted by Qt
    qapp.processEvents()
    gc.collect()
    qapp.processEvents()


def _pump(app, window, frames):
    """Render exactly `frames` frames.

    The window's own 25 FPS timer is stopped first: processEvents() would otherwise let it
    fire extra frames, which made row counts depend on how long a frame took to compute.
    """
    window._timer.stop()
    for _ in range(frames):
        window._on_frame()
        app.processEvents()


def test_window_renders_frames(qapp):
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25)
    _pump(qapp, win, 5)
    assert win._rows_pushed == 5
    assert win.waterfall.buffer.image.shape == (512, 1024)
    freqs, values = win.spectrum._curve.getData()
    assert freqs.size == 1024 and values.size == 1024
    win.close()


def test_tone_appears_at_correct_frequency_in_waterfall(qapp):
    """End-to-end: a +96 kHz tone must land right of centre in the newest row."""
    win = MainWindow(StubSource(_caps(), tone_hz=96e3), fft_size=4096, fps=25)
    _pump(qapp, win, 3)
    row = win.waterfall.buffer.image[0]
    assert int(np.argmax(row)) > row.size // 2
    win.close()


def test_startup_auto_range_brings_data_into_view(qapp):
    """The fixed -115..-40 default rendered blank on a weak antenna; auto-fit must not."""
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25, levels=None)
    assert win._auto_pending
    _pump(qapp, win, 25)
    assert not win._auto_pending
    lo, hi = win._levels
    buf = win.waterfall.buffer
    written = buf.image[: buf.written_rows]
    inside = np.mean((written >= lo) & (written <= hi))
    assert inside > 0.9, f"auto-range {lo:.1f}..{hi:.1f} covers only {inside:.0%} of data"
    win.close()


def test_explicit_levels_are_not_overridden(qapp):
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25, levels=(-100.0, -20.0))
    assert not win._auto_pending
    _pump(qapp, win, 25)
    assert win._levels == (-100.0, -20.0)
    win.close()


def test_agc_only_for_airspyhf_no_gain_sliders(qapp):
    """Capability-driven UI: airspyhf reports no gain elements, so none may be built."""
    win = MainWindow(StubSource(_caps()), fft_size=1024)
    box = win._build_device_controls()
    assert any(isinstance(c, QtWidgets.QCheckBox) for c in box.children())
    assert not any(isinstance(c, QtWidgets.QDoubleSpinBox) for c in box.children())
    win.close()


def test_gain_sliders_built_for_radio_that_has_them(qapp):
    caps = _caps(driver="hackrf", gain_elements=(GainElement("LNA", 0.0, 40.0, 8.0),))
    win = MainWindow(StubSource(caps), fft_size=1024)
    box = win._build_device_controls()
    assert any(isinstance(c, QtWidgets.QDoubleSpinBox) for c in box.children())
    win.close()


def test_fft_size_change_resets_history(qapp):
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25)
    _pump(qapp, win, 4)
    win._fft_combo.setCurrentText("4096")
    assert win.analyzer.fft_size == 4096
    assert win._rows_pushed == 0
    _pump(qapp, win, 2)
    assert win.spectrum._curve.getData()[0].size == 4096
    win.close()


def test_frame_with_no_samples_is_survivable(qapp):
    class Empty(StubSource):
        def read_latest(self, n):
            return np.empty(0, dtype=np.complex64)

    win = MainWindow(Empty(_caps()), fft_size=1024)
    win._on_frame()  # must not raise
    assert win._rows_pushed == 0
    win.close()


def test_waterfall_geometry_follows_sample_rate(qapp):
    win = MainWindow(StubSource(_caps(), rate=768e3, center=7.1e6), fft_size=1024, fps=25)
    (x0, x1), _ = win.waterfall.getViewBox().viewRange()
    assert x0 == pytest.approx(7.1e6 - 384e3, abs=1e3)
    assert x1 == pytest.approx(7.1e6 + 384e3, abs=1e3)
    win.close()


def test_analyzer_requests_enough_samples_for_widest_fft(qapp):
    for size in (1024, 4096, 16384):
        sa = SpectrumAnalyzer(fft_size=size)
        assert sa.samples_wanted() >= size


def test_waterfall_image_maps_onto_axes_when_geometry_set_before_first_push(qapp):
    """Regression: ImageItem.setRect scales from the image size at call time.

    Setting geometry while `image is None` made pyqtgraph scale as if the image were
    1x1, stretching it to 98 MHz x 1280 s so the pane rendered blank.
    """
    from src.rgc_sdr.ui.waterfall import WaterfallView

    wf = WaterfallView(rows=64, cols=128)
    wf.set_geometry(0.909e6, 768e3, 20.0)  # before any push, as MainWindow does
    for _ in range(3):
        wf.push(np.full(128, -100.0, dtype=np.float32))

    rect = wf._img.mapRectToView(wf._img.boundingRect())
    assert rect.width() == pytest.approx(768e3, abs=1.0)
    assert rect.height() == pytest.approx(20.0, abs=0.01)
    assert rect.left() == pytest.approx(0.909e6 - 384e3, abs=1.0)


def test_waterfall_image_stays_mapped_after_bin_count_change(qapp):
    from src.rgc_sdr.ui.waterfall import WaterfallView

    wf = WaterfallView(rows=32, cols=64)
    wf.set_geometry(7.1e6, 768e3, 10.0)
    wf.push(np.full(64, -100.0, dtype=np.float32))
    wf.resize_bins(256)
    wf.push(np.full(256, -100.0, dtype=np.float32))
    rect = wf._img.mapRectToView(wf._img.boundingRect())
    assert rect.width() == pytest.approx(768e3, abs=1.0)
    assert rect.height() == pytest.approx(10.0, abs=0.01)


def test_window_waterfall_is_actually_on_screen(qapp):
    """The rendered image must overlap the visible view, not sit off in the distance."""
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25)
    _pump(qapp, win, 3)
    img = win.waterfall._img
    drawn = img.mapRectToView(img.boundingRect())
    (x0, x1), (y0, y1) = win.waterfall.getViewBox().viewRange()
    view = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
    overlap = drawn.intersected(view)
    assert overlap.width() / view.width() > 0.95
    assert overlap.height() / view.height() > 0.95
    win.close()


# -- P2: tuning UI -----------------------------------------------------------

def test_click_to_tune_retunes_to_clicked_frequency(qapp):
    src = StubSource(_caps())
    win = MainWindow(src, fft_size=1024, fps=25)
    _pump(qapp, win, 3)
    win.waterfall.frequencySelected.emit(7.3e6)
    assert src.center_freq == pytest.approx(7.3e6)
    assert win._freq_spin.value() == pytest.approx(7.3, abs=1e-6)
    win.close()


def test_retune_clears_stale_history_and_smoothing(qapp):
    """History and peak hold describe the old frequency and must not survive a retune."""
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25)
    _pump(qapp, win, 6)
    assert win.waterfall.buffer.written_rows == 6
    assert win.waterfall.buffer.image[0].max() > -50.0  # the tone is in there
    win.spectrum.frequencySelected.emit(10e6)
    assert win.waterfall.buffer.written_rows == 0
    assert np.all(win.waterfall.buffer.image == -140.0)
    assert win.spectrum._smoothed is None
    assert win.spectrum._peak is None
    win.close()


def test_retune_moves_the_frequency_axis(qapp):
    win = MainWindow(StubSource(_caps(), rate=768e3, center=7.1e6), fft_size=1024, fps=25)
    win.waterfall.frequencySelected.emit(20e6)
    (x0, x1), _ = win.waterfall.getViewBox().viewRange()
    assert x0 == pytest.approx(20e6 - 384e3, abs=1e3)
    assert x1 == pytest.approx(20e6 + 384e3, abs=1e3)
    win.close()


def test_clicking_into_the_untunable_gap_snaps_and_shows_the_truth(qapp):
    """Clamping must be reflected back into the spinbox, not silently diverge."""
    src = StubSource(_caps(freq_ranges=(FreqRange(9e3, 31e6), FreqRange(60e6, 260e6))))
    win = MainWindow(src, fft_size=1024, fps=25)
    win.waterfall.frequencySelected.emit(45e6)
    assert src.center_freq == pytest.approx(31e6)
    assert win._freq_spin.value() == pytest.approx(31.0, abs=1e-6)
    win.close()


def test_spinbox_change_retunes_the_source(qapp):
    src = StubSource(_caps())
    win = MainWindow(src, fft_size=1024, fps=25)
    win._freq_spin.setValue(14.2)
    assert src.center_freq == pytest.approx(14.2e6)
    win.close()


def test_step_size_drives_the_spinbox_increment(qapp):
    win = MainWindow(StubSource(_caps()), fft_size=1024)
    win._step_combo.setCurrentText("1 MHz")
    assert win._freq_spin.singleStep() == pytest.approx(1.0)
    win._step_combo.setCurrentText("9 kHz")
    assert win._freq_spin.singleStep() == pytest.approx(0.009)
    win.close()


def test_rate_combo_offers_every_supported_rate_and_applies_it(qapp):
    rates = (912e3, 768e3, 650e3, 456e3, 384e3, 228e3, 192e3)
    src = StubSource(_caps(sample_rates=rates))
    win = MainWindow(src, fft_size=1024, fps=25)
    assert win._rate_combo is not None
    assert win._rate_combo.count() == len(rates)
    win._rate_combo.setCurrentText("456 kS/s")
    assert src.sample_rate == pytest.approx(456e3)
    (x0, x1), _ = win.waterfall.getViewBox().viewRange()
    assert (x1 - x0) == pytest.approx(456e3, abs=1e3)
    win.close()


def test_rate_selector_is_hidden_when_there_is_only_one_rate(qapp):
    """Nothing to choose, so it takes no space -- but it still exists, because switching
    to a radio with several rates repopulates it."""
    win = window_for(StubSource(_caps(sample_rates=(768e3,))), fft_size=1024)
    assert win._rate_combo.isHidden()
    assert win._rate_label.isHidden()
    win.close()

def test_no_bandwidth_control_for_airspyhf(qapp):
    """Capability-driven: this driver reports no bandwidth options, so none is shown."""
    win = window_for(StubSource(_caps(bandwidths=())), fft_size=1024)
    assert win._bw_combo.isHidden()
    assert win._hw_bw_label.isHidden()
    win.close()

def test_bandwidth_control_appears_when_the_driver_offers_options(qapp):
    src = StubSource(_caps(driver="hackrf", bandwidths=(1.75e6, 2.5e6, 3.5e6)))
    win = MainWindow(src, fft_size=1024)
    assert not win._bw_combo.isHidden() and win._bw_combo.count() == 3
    win._bw_combo.setCurrentText("2500 kHz")
    assert src.bandwidth_set == pytest.approx(2.5e6)
    win.close()


def test_centre_marker_follows_tuning(qapp):
    win = MainWindow(StubSource(_caps(), center=7.1e6), fft_size=1024, fps=25)
    assert win.spectrum._center_line.value() == pytest.approx(7.1e6)
    win.waterfall.frequencySelected.emit(12e6)
    assert win.spectrum._center_line.value() == pytest.approx(12e6)
    win.close()


def test_right_click_does_not_tune(qapp):
    """Only left-click tunes; right-click is reserved for the view's own handling."""
    src = StubSource(_caps())
    win = MainWindow(src, fft_size=1024)
    before = src.center_freq
    ev = type("E", (), {
        "button": lambda self: QtCore.Qt.MouseButton.RightButton,
        "scenePos": lambda self: QtCore.QPointF(0, 0),
        "accept": lambda self: None,
    })()
    win.waterfall._on_click(ev)
    assert src.center_freq == before
    win.close()


# -- zoom / decimation -------------------------------------------------------

def test_zoom_narrows_the_span_and_sharpens_resolution(qapp):
    win = MainWindow(StubSource(_caps(), rate=768e3, center=7.1e6), fft_size=1024, fps=25)
    assert win.effective_rate == pytest.approx(768e3)
    win._zoom_combo.setCurrentText("8x")
    assert win.decimator.factor == 8
    assert win.effective_rate == pytest.approx(96e3)
    (x0, x1), _ = win.waterfall.getViewBox().viewRange()
    assert (x1 - x0) == pytest.approx(96e3, abs=1e2)
    assert x0 == pytest.approx(7.1e6 - 48e3, abs=1e2)
    win.close()


def test_zoom_clears_history_since_the_span_changed(qapp):
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25)
    _pump(qapp, win, 5)
    assert win.waterfall.buffer.written_rows == 5
    win._zoom_combo.setCurrentText("4x")
    assert win.waterfall.buffer.written_rows == 0
    assert win.spectrum._smoothed is None
    win.close()


def test_zoomed_frames_still_render(qapp):
    """The whole decimated path, end to end, at every offered factor."""
    for factor in (1, 2, 4, 8, 16, 32):
        win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25, decimation=factor)
        _pump(qapp, win, 3)
        assert win.waterfall.buffer.written_rows == 3, f"no frames at {factor}x"
        assert win.spectrum._curve.getData()[0].size == 1024
        win.close()


def test_zoomed_tone_lands_at_the_same_absolute_frequency(qapp):
    """Zoom must not move a signal: 20 kHz offset stays at 20 kHz offset."""
    src = StubSource(_caps(), rate=768e3, center=7.1e6, tone_hz=20e3, noise=1e-5)
    win = MainWindow(src, fft_size=4096, fps=25, decimation=8)
    _pump(qapp, win, 2)
    freqs, values = win.spectrum._curve.getData()
    peak_hz = freqs[int(np.argmax(values))]
    assert peak_hz == pytest.approx(7.1e6 + 20e3, abs=200.0)
    win.close()


def test_deep_zoom_requests_more_input_than_it_returns(qapp):
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25, decimation=16)
    assert win._frame_request() > 1024 * 16
    win.close()


# -- memories ----------------------------------------------------------------

def test_save_and_recall_a_memory(qapp, tmp_path):
    settings = Settings(tmp_path / "s.json")
    src = StubSource(_caps(sample_rates=(768e3, 192e3)))
    win = MainWindow(src, fft_size=1024, fps=25, settings=settings)

    win._freq_spin.setValue(9.6)
    win._zoom_combo.setCurrentText("4x")
    assert win.save_memory("31m broadcast") is False

    win._freq_spin.setValue(14.2)
    win._zoom_combo.setCurrentText("1x")
    assert src.center_freq == pytest.approx(14.2e6)

    assert win.recall_memory("31m broadcast") is True
    assert src.center_freq == pytest.approx(9.6e6)
    assert win.decimator.factor == 4
    assert win._zoom_combo.currentText() == "4x"
    assert win._freq_spin.value() == pytest.approx(9.6, abs=1e-6)
    win.close()


def test_memory_survives_a_restart(qapp, tmp_path):
    """Saved memories must come back in a fresh window from the file on disk."""
    path = tmp_path / "s.json"
    first = MainWindow(StubSource(_caps()), fft_size=1024, settings=Settings(path))
    first._freq_spin.setValue(0.198)
    first.save_memory("Radio 4 LW")
    first.close()

    second = MainWindow(StubSource(_caps()), fft_size=1024, settings=Settings.load(path))
    assert "Radio 4 LW" in second.settings.names()
    assert second.recall_memory("Radio 4 LW") is True
    assert second.source.center_freq == pytest.approx(0.198e6)
    second.close()


def test_recalling_an_unknown_memory_is_harmless(qapp, tmp_path):
    win = MainWindow(StubSource(_caps()), fft_size=1024, settings=Settings(tmp_path / "s.json"))
    assert win.recall_memory("nope") is False
    win.close()


def test_saving_the_same_name_replaces_it(qapp, tmp_path):
    win = MainWindow(StubSource(_caps()), fft_size=1024, settings=Settings(tmp_path / "s.json"))
    win._freq_spin.setValue(7.1)
    win.save_memory("spot")
    win._freq_spin.setValue(7.2)
    assert win.save_memory("spot") is True
    assert len(win.settings.memories) == 1
    assert win.settings.get_memory("spot").snapshot.freq_hz == pytest.approx(7.2e6)
    win.close()


def test_delete_memory_updates_the_combo(qapp, tmp_path):
    win = MainWindow(StubSource(_caps()), fft_size=1024, settings=Settings(tmp_path / "s.json"))
    win.save_memory("a")
    win.save_memory("b")
    assert win._memory_combo.count() == 3        # placeholder + two
    assert win.delete_memory("a") is True
    assert win._memory_combo.count() == 2
    assert win.delete_memory("a") is False
    win.close()


def test_memory_controls_disabled_when_there_are_none(qapp, tmp_path):
    win = MainWindow(StubSource(_caps()), fft_size=1024, settings=Settings(tmp_path / "s.json"))
    assert not win._memory_combo.isEnabled()
    assert not win._delete_button.isEnabled()
    win.save_memory("one")
    assert win._memory_combo.isEnabled()
    assert win._delete_button.isEnabled()
    win.close()


def test_no_settings_store_means_no_disk_writes(qapp, tmp_path):
    """Windows built without a store must not touch the real settings file."""
    win = MainWindow(StubSource(_caps()), fft_size=1024)
    assert win._persist is False
    win._save_state()
    win.close()
    assert not list(tmp_path.iterdir())


# -- last-state restore ------------------------------------------------------

def test_closing_saves_the_current_state(qapp, tmp_path):
    path = tmp_path / "s.json"
    win = MainWindow(StubSource(_caps()), fft_size=1024, settings=Settings(path))
    win._freq_spin.setValue(21.05)
    win._zoom_combo.setCurrentText("16x")
    win.close()

    again = Settings.load(path)
    assert again.last is not None
    assert again.last.freq_hz == pytest.approx(21.05e6)
    assert again.last.decimation == 16
    assert again.last.fft_size == 1024


def test_apply_snapshot_restores_everything(qapp):
    src = StubSource(_caps(sample_rates=(768e3, 192e3)))
    win = MainWindow(src, fft_size=1024, fps=25)
    win.apply_snapshot(Snapshot(
        freq_hz=5.0e6, sample_rate=192e3, decimation=8, fft_size=4096,
        colormap="viridis", min_db=-118.0, max_db=-72.0, agc=False, peak_hold=False,
    ))
    assert src.center_freq == pytest.approx(5.0e6)
    assert src.sample_rate == pytest.approx(192e3)
    assert win.decimator.factor == 8
    assert win.analyzer.fft_size == 4096
    assert win._cmap_combo.currentText() == "viridis"
    assert win._levels == (-118.0, -72.0)
    assert win._peak_check.isChecked() is False
    assert win._agc_check.isChecked() is False
    win.close()


def test_auto_fitted_levels_are_not_persisted_as_a_preference(qapp, tmp_path):
    """A range fitted to today's antenna should re-fit next launch, not be restored."""
    path = tmp_path / "s.json"
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25, settings=Settings(path))
    _pump(qapp, win, 25)                     # triggers the startup auto-fit
    assert not win._auto_pending
    win.close()
    assert Settings.load(path).last.min_db is None


def test_explicitly_chosen_levels_are_persisted(qapp, tmp_path):
    path = tmp_path / "s.json"
    win = MainWindow(StubSource(_caps()), fft_size=1024, fps=25, settings=Settings(path))
    win._min_spin.setValue(-125.0)
    win._max_spin.setValue(-75.0)
    win.close()
    saved = Settings.load(path).last
    assert saved.min_db == pytest.approx(-125.0)
    assert saved.max_db == pytest.approx(-75.0)


def test_snapshot_round_trip_through_the_window(qapp):
    win = MainWindow(StubSource(_caps()), fft_size=2048, fps=25, decimation=4)
    snap = win.current_snapshot()
    assert snap.decimation == 4
    assert snap.fft_size == 2048
    assert snap.freq_hz == pytest.approx(win.source.center_freq)
    win.close()


# -- audio controls ----------------------------------------------------------

def test_mode_defaults_to_off_and_starts_no_audio(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    assert win.mode == "off"
    assert win.audio is None
    win.close()


def test_mode_combo_offers_off_plus_every_mode(qapp):
    from src.rgc_sdr.dsp.demod import MODES as DEMOD_MODES

    win = window_for(StubSource(_caps()), fft_size=1024)
    assert win._mode_combo.count() == len(DEMOD_MODES) + 1
    assert win._mode_combo.itemData(0) == "off"
    win.close()


def test_mode_is_refused_without_an_audio_device(qapp):
    """No device must degrade gracefully, not raise or half-start a sink."""
    win = window_for(StubSource(_caps()), fft_size=1024, enable_audio=False)
    win.set_mode("am")
    assert win.audio is None
    assert "audio" in win._status.currentMessage().lower()
    win.close()


def test_passband_appears_for_am_and_hides_when_off(qapp):
    win = window_for(StubSource(_caps(), center=7.1e6), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    win._update_passband()
    assert win.spectrum._passband.isVisible()
    lo, hi = win.spectrum._passband.getRegion()
    assert (hi - lo) == pytest.approx(9e3, abs=1.0)
    assert (lo + hi) / 2 == pytest.approx(7.1e6, abs=1.0)

    win._mode_combo.setCurrentIndex(win._mode_combo.findData("off"))
    win._update_passband()
    assert not win.spectrum._passband.isVisible()
    win.close()


def test_passband_follows_the_offset(qapp):
    win = window_for(StubSource(_caps(), center=7.1e6), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    win._offset_spin.setValue(20.0)          # kHz
    lo, hi = win.spectrum._passband.getRegion()
    assert (lo + hi) / 2 == pytest.approx(7.1e6 + 20e3, abs=1.0)
    win.close()


def test_ssb_passband_is_one_sided(qapp):
    """USB sits above the carrier and LSB below; a symmetric band would mislead."""
    win = window_for(StubSource(_caps(), center=7.1e6), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("usb"))
    lo, hi = win.spectrum._passband.getRegion()
    assert lo == pytest.approx(7.1e6, abs=1.0)
    assert hi > lo

    win._mode_combo.setCurrentIndex(win._mode_combo.findData("lsb"))
    lo, hi = win.spectrum._passband.getRegion()
    assert hi == pytest.approx(7.1e6, abs=1.0)
    assert lo < hi
    win.close()


def test_offset_range_tracks_the_visible_span(qapp):
    win = window_for(StubSource(_caps(), rate=768e3), fft_size=1024)
    assert win._offset_spin.maximum() == pytest.approx(384.0)   # kHz
    win._zoom_combo.setCurrentText("8x")
    assert win._offset_spin.maximum() == pytest.approx(48.0)
    win.close()


def test_squelch_only_enabled_for_modes_that_support_it(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("usb"))
    assert not win._squelch_check.isEnabled()
    for mode in ("am", "nbfm", "wbfm"):
        win._mode_combo.setCurrentIndex(win._mode_combo.findData(mode))
        assert win._squelch_check.isEnabled(), mode
    win.close()


def test_squelch_value_is_none_when_unchecked(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("nbfm"))
    win._squelch_check.setChecked(False)
    assert win._squelch_value() is None
    win._squelch_check.setChecked(True)
    win._squelch_spin.setValue(-95.0)
    assert win._squelch_value() == pytest.approx(-95.0)
    win.close()


def test_audio_settings_round_trip_through_a_snapshot(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("lsb"))
    win._volume_slider.setValue(65)
    win._offset_spin.setValue(-12.5)
    snap = win.current_snapshot()
    assert snap.mode == "lsb"
    assert snap.volume == pytest.approx(0.65)
    assert snap.offset_hz == pytest.approx(-12.5e3)
    win.close()


def test_audio_settings_are_restored_from_a_memory(qapp, tmp_path):
    settings = Settings(tmp_path / "s.json")
    win = window_for(StubSource(_caps()), fft_size=1024, settings=settings)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("usb"))
    win._volume_slider.setValue(30)
    win._offset_spin.setValue(3.5)
    win.save_memory("cw spot")

    win._mode_combo.setCurrentIndex(win._mode_combo.findData("off"))
    win._volume_slider.setValue(90)
    win._offset_spin.setValue(0.0)

    assert win.recall_memory("cw spot") is True
    assert win.mode == "usb"
    assert win._volume_slider.value() == 30
    assert win._offset_spin.value() == pytest.approx(3.5)
    win.close()


def test_mode_persists_across_a_restart(qapp, tmp_path):
    path = tmp_path / "s.json"
    first = window_for(StubSource(_caps()), fft_size=1024, settings=Settings(path))
    first._mode_combo.setCurrentIndex(first._mode_combo.findData("nbfm"))
    first._squelch_check.setChecked(True)
    first._squelch_spin.setValue(-88.0)
    first.close()

    saved = Settings.load(path).last
    assert saved.mode == "nbfm"
    assert saved.squelch_dbfs == pytest.approx(-88.0)


def test_retune_resets_nothing_when_audio_is_off(qapp):
    """Guard: the retune path must not assume a sink exists."""
    win = window_for(StubSource(_caps()), fft_size=1024, fps=25)
    _pump(qapp, win, 2)
    win.waterfall.frequencySelected.emit(12e6)        # must not raise
    assert win.source.center_freq == pytest.approx(12e6)
    win.close()


# -- P4: bandwidth, S-meter, recording ---------------------------------------

def test_bandwidth_presets_follow_the_mode(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    assert win.bandwidth_hz() == pytest.approx(9e3)         # the AM default
    am_widths = [win._bw_audio_combo.itemData(i) for i in range(win._bw_audio_combo.count())]

    win._mode_combo.setCurrentIndex(win._mode_combo.findData("usb"))
    assert win.bandwidth_hz() == pytest.approx(2.7e3)
    ssb_widths = [win._bw_audio_combo.itemData(i) for i in range(win._bw_audio_combo.count())]
    assert am_widths != ssb_widths
    assert max(ssb_widths) < min(am_widths) * 2
    win.close()


def test_bandwidth_choice_narrows_the_passband(qapp):
    win = window_for(StubSource(_caps(), center=7.1e6), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    win._bw_audio_combo.setCurrentIndex(win._bw_audio_combo.findData(3e3))
    lo, hi = win.spectrum._passband.getRegion()
    assert (hi - lo) == pytest.approx(3e3, abs=1.0)
    win.close()


def test_bandwidth_round_trips_through_a_snapshot(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    win._bw_audio_combo.setCurrentIndex(win._bw_audio_combo.findData(16e3))
    assert win.current_snapshot().bandwidth_hz == pytest.approx(16e3)

    win.apply_snapshot(Snapshot(freq_hz=7.1e6, mode="am", bandwidth_hz=4.5e3))
    assert win.bandwidth_hz() == pytest.approx(4.5e3)
    win.close()


def test_smeter_reads_a_signal_above_the_noise(qapp):
    """A tone inside the channel must give a level well above the noise floor."""
    src = StubSource(_caps(), rate=768e3, center=7.1e6, tone_hz=1e3, noise=1e-4)
    win = window_for(src, fft_size=4096, fps=25)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    _pump(qapp, win, 3)
    assert win.smeter.level_dbfs is not None
    assert win.smeter.level_dbfs > -60.0
    win.close()


def test_smeter_level_drops_when_the_signal_leaves_the_channel(qapp):
    """Tuning away from a signal must be visible on the meter."""
    src = StubSource(_caps(), rate=768e3, center=7.1e6, tone_hz=1e3, noise=1e-4)
    win = window_for(src, fft_size=4096, fps=25)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    _pump(qapp, win, 3)
    on_signal = win.smeter.level_dbfs

    win._offset_spin.setValue(200.0)      # 200 kHz away from the tone
    _pump(qapp, win, 3)
    assert win.smeter.level_dbfs < on_signal - 20.0
    win.close()


def test_smeter_peak_holds_then_decays(qapp):
    from src.rgc_sdr.ui.smeter import SMeter

    meter = SMeter(peak_decay_db=1.0)
    meter.set_level(-40.0)
    assert meter.peak_dbfs == pytest.approx(-40.0)
    meter.set_level(-90.0)
    assert meter.peak_dbfs == pytest.approx(-41.0)   # decayed by one step
    for _ in range(100):
        meter.set_level(-90.0)
    assert meter.peak_dbfs == pytest.approx(-90.0)   # settles to the level


def test_smeter_reset_clears(qapp):
    from src.rgc_sdr.ui.smeter import SMeter

    meter = SMeter()
    meter.set_level(-50.0)
    meter.reset()
    assert meter.level_dbfs is None and meter.peak_dbfs is None


def test_audio_recording_needs_a_mode(qapp, tmp_path):
    """Recording audio with no demodulator running is refused, not silently empty."""
    win = window_for(StubSource(_caps()), fft_size=1024, recordings_dir=tmp_path)
    assert win.start_audio_recording() is None
    assert "mode" in win._status.currentMessage().lower()
    assert list(tmp_path.iterdir()) == []
    win.close()


def test_iq_recording_writes_a_file_and_a_sidecar(qapp, tmp_path):
    import time

    src = StubSource(_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25, recordings_dir=tmp_path)
    path = win.start_iq_recording()
    assert path is not None and path.suffix == ".cf32"
    assert win.iq_recorder is not None
    time.sleep(0.3)
    win.stop_iq_recording()
    assert win.iq_recorder is None
    assert path.is_file()
    assert path.with_suffix(".cf32.json").is_file()
    win.close()


def test_iq_recording_filename_carries_the_frequency(qapp, tmp_path):
    src = StubSource(_caps(), center=0.684e6)
    win = window_for(src, fft_size=1024, recordings_dir=tmp_path)
    path = win.start_iq_recording()
    assert "0.6840MHz" in path.name
    win.stop_iq_recording()
    win.close()


def test_retuning_stops_an_iq_recording(qapp, tmp_path):
    """The sidecar states one centre frequency, so a retune would make the file a lie."""
    src = StubSource(_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25, recordings_dir=tmp_path)
    win._rec_iq_button.setChecked(True)
    win.start_iq_recording()
    assert win.iq_recorder is not None

    win.waterfall.frequencySelected.emit(10e6)
    assert win.iq_recorder is None, "recording continued across a retune"
    assert not win._rec_iq_button.isChecked()
    assert "frequency changed" in win._status.currentMessage()
    win.close()


def test_closing_stops_recordings(qapp, tmp_path):
    src = StubSource(_caps())
    win = window_for(src, fft_size=1024, recordings_dir=tmp_path)
    path = win.start_iq_recording()
    win.close()
    assert win.iq_recorder is None
    assert path.is_file()


def test_record_buttons_untoggle_when_refused(qapp, tmp_path):
    win = window_for(StubSource(_caps()), fft_size=1024, recordings_dir=tmp_path)
    win._rec_audio_button.setChecked(True)
    win._on_record_audio(True)               # no audio mode, so refused
    assert not win._rec_audio_button.isChecked()
    win.close()


def test_recordings_default_under_documents(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    assert win.recordings_dir.name == "RGC_SDR"
    assert "Documents" in str(win.recordings_dir)
    win.close()


# -- P5: scanner -------------------------------------------------------------

def _airband_caps():
    return _caps(freq_ranges=(FreqRange(9e3, 31e6), FreqRange(60e6, 260e6)))


def _scan_window(win, carriers=()):
    """A spectrum for the window the scanner is currently on."""
    centre = win.source.center_freq
    bins = win.analyzer.fft_size
    freqs = centre + (np.arange(bins) - bins // 2) * (win.effective_rate / bins)
    rng = np.random.default_rng(11)
    dbfs = -118.0 + rng.normal(0, 0.7, bins)
    for freq, excess in carriers:
        idx = int(np.argmin(np.abs(freqs - freq)))
        for off, share in ((0, 1.0), (-1, 0.6), (1, 0.6)):
            if 0 <= idx + off < bins:
                dbfs[idx + off] = -118.0 + excess * share
    return freqs, dbfs


def test_scan_starts_and_tunes_into_the_range(qapp, tmp_path):
    src = StubSource(_airband_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, settings=Settings(tmp_path / "s.json"))
    win.scanner_panel.apply_config(118e6, 137e6, 25e3, 10.0, True)
    assert win.start_scan() is True
    assert win.scanner is not None
    assert 118e6 <= src.center_freq <= 137e6
    win.stop_scan()
    assert win.scanner is None
    win.close()


def test_scan_refuses_a_range_the_receiver_cannot_reach(qapp, tmp_path):
    """The Airspy has a 31-60 MHz hole and stops at 260 MHz."""
    win = window_for(StubSource(_airband_caps()), fft_size=1024,
                     settings=Settings(tmp_path / "s.json"))
    win.scanner_panel.apply_config(400e6, 450e6, 25e3, 10.0, True)
    assert win.start_scan() is False
    assert win.scanner is None
    assert "outside this receiver" in win.scanner_panel._status.text()
    win.close()


def test_scan_refuses_a_backwards_range(qapp, tmp_path):
    win = window_for(StubSource(_airband_caps()), fft_size=1024,
                     settings=Settings(tmp_path / "s.json"))
    win.scanner_panel.apply_config(137e6, 118e6, 25e3, 10.0, True)
    assert win.start_scan() is False
    win.close()


def test_scan_records_hits_into_found_not_memories(qapp, tmp_path):
    settings = Settings(tmp_path / "s.json")
    settings.add_memory("hand saved", Snapshot(freq_hz=7.1e6))
    src = StubSource(_airband_caps(), center=7.1e6)
    win = window_for(src, fft_size=4096, fps=25, settings=settings)
    # Confirmation off, so one sighting is reported immediately.
    win.scanner_panel.apply_config(118e6, 137e6, 25e3, 10.0, False, 1)
    win.start_scan()

    target = win.source.center_freq + 150e3
    freqs, dbfs = _scan_window(win, carriers=[(target, 30)])
    win.scanner._state = __import__(
        "src.rgc_sdr.scanner", fromlist=["ScanState"]
    ).ScanState.SEARCHING
    win._scan_frame(freqs, dbfs)

    assert len(settings.found) == 1
    assert settings.names() == ["hand saved"], "a scan hit leaked into the memories"
    win.stop_scan()
    win.close()


def test_found_channels_survive_a_restart(qapp, tmp_path):
    path = tmp_path / "s.json"
    first = window_for(StubSource(_airband_caps()), fft_size=1024, settings=Settings(path))
    first.settings.record_found(118.325e6, -95.0, 22.0)
    first.close()

    second = window_for(StubSource(_airband_caps()), fft_size=1024,
                        settings=Settings.load(path))
    assert [c.freq_hz for c in second.settings.found] == [pytest.approx(118.325e6)]
    assert second.scanner_panel._found_list.count() == 1
    second.close()


def test_lock_out_persists_and_shows_in_the_panel(qapp, tmp_path):
    path = tmp_path / "s.json"
    win = window_for(StubSource(_airband_caps()), fft_size=1024, settings=Settings(path))
    win.settings.record_found(121.5e6, -90.0, 25.0)
    win._refresh_scan_lists()
    assert win.scanner_panel._found_list.count() == 1

    win.lock_out(121.5e6)
    assert 121.5e6 in win.settings.lockout
    assert win.scanner_panel._found_list.count() == 0, "locked channel still listed as found"
    assert win.scanner_panel._lock_list.count() == 1
    win.close()

    assert 121.5e6 in Settings.load(path).lockout


def test_unlock_removes_it_from_the_lockout_list(qapp, tmp_path):
    win = window_for(StubSource(_airband_caps()), fft_size=1024,
                     settings=Settings(tmp_path / "s.json"))
    win.lock_out(121.5e6)
    win.unlock(121.5e6)
    assert win.settings.lockout == set()
    assert win.scanner_panel._lock_list.count() == 0
    win.close()


def test_locked_channel_is_not_found_again_while_scanning(qapp, tmp_path):
    """The point of lockout: subsequent passes must not stop there."""
    from src.rgc_sdr.scanner import ScanState

    settings = Settings(tmp_path / "s.json")
    win = window_for(StubSource(_airband_caps()), fft_size=4096, fps=25, settings=settings)
    win.scanner_panel.apply_config(118e6, 137e6, 25e3, 10.0, False)
    win.start_scan()
    target = round((win.source.center_freq + 150e3) / 25e3) * 25e3
    win.lock_out(target)

    freqs, dbfs = _scan_window(win, carriers=[(target, 30)])
    win.scanner._state = ScanState.SEARCHING
    win._scan_frame(freqs, dbfs)
    assert settings.found == [], "a locked-out channel was reported"
    win.stop_scan()
    win.close()


def test_dwelling_uses_the_audio_offset_and_does_not_retune(qapp, tmp_path):
    from src.rgc_sdr.scanner import ScanState

    src = StubSource(_airband_caps(), center=7.1e6)
    win = window_for(src, fft_size=4096, fps=25, settings=Settings(tmp_path / "s.json"))
    win.scanner_panel.apply_config(118e6, 137e6, 25e3, 10.0, True)
    win.start_scan()
    window_centre = src.center_freq

    target = window_centre + 150e3
    freqs, dbfs = _scan_window(win, carriers=[(target, 30)])
    win.scanner._state = ScanState.SEARCHING
    win._scan_frame(freqs, dbfs)

    assert win.scanner.dwell_hz is not None
    assert src.center_freq == pytest.approx(window_centre), "the radio moved to dwell"
    win.stop_scan()
    win.close()


def test_manual_tuning_stops_the_scan(qapp, tmp_path):
    src = StubSource(_airband_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25, settings=Settings(tmp_path / "s.json"))
    win.scanner_panel.apply_config(118e6, 137e6, 25e3, 10.0, True)
    win.start_scan()
    assert win.scanner is not None

    win.waterfall.frequencySelected.emit(125e6)      # user clicks the waterfall
    assert win.scanner is None, "the scan kept fighting the user for the dial"
    assert not win.scanner_panel._scan_button.isChecked()
    win.close()


def test_tuning_a_found_channel_stops_the_scan_and_goes_there(qapp, tmp_path):
    settings = Settings(tmp_path / "s.json")
    settings.record_found(121.5e6, -90.0, 25.0)
    src = StubSource(_airband_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25, settings=settings)
    win._refresh_scan_lists()
    win.scanner_panel.apply_config(118e6, 137e6, 25e3, 10.0, True)
    win.start_scan()

    win.scanner_panel.channelActivated.emit(121.5e6)
    assert win.scanner is None
    assert src.center_freq == pytest.approx(121.5e6)
    assert win._offset_spin.value() == pytest.approx(0.0)
    win.close()


def test_scan_range_is_saved(qapp, tmp_path):
    path = tmp_path / "s.json"
    win = window_for(StubSource(_airband_caps()), fft_size=1024, settings=Settings(path))
    win.scanner_panel.apply_config(156e6, 163e6, 12.5e3, 15.0, False)
    win._on_scan_config_changed()
    win.close()

    scan = Settings.load(path).scan
    assert scan.start_hz == pytest.approx(156e6)
    assert scan.step_hz == pytest.approx(12.5e3)
    assert scan.stop_on_signal is False


def test_clear_found_leaves_lockout_and_memories_alone(qapp, tmp_path):
    settings = Settings(tmp_path / "s.json")
    settings.add_memory("keep me", Snapshot(freq_hz=7.1e6))
    settings.record_found(118.325e6, -95.0, 20.0)
    settings.add_lockout(121.5e6)
    win = window_for(StubSource(_airband_caps()), fft_size=1024, settings=settings)
    win.clear_found()
    assert settings.found == []
    assert settings.lockout == {121.5e6}
    assert settings.names() == ["keep me"]
    win.close()


def test_band_presets_include_airband(qapp):
    from src.rgc_sdr.ui.scanner_panel import BAND_PRESETS

    names = [p[0] for p in BAND_PRESETS]
    assert any("Airband" in n for n in names)
    for _, start, end, step in BAND_PRESETS:
        assert end > start and step > 0


def test_smeter_snr_is_zero_not_absurd_when_there_is_no_signal(qapp):
    """A dead channel must read 0 dB S/N, not 10*log10(epsilon)."""
    src = StubSource(_caps(), rate=768e3, center=7.1e6, tone_hz=300e3, noise=1e-4)
    win = window_for(src, fft_size=4096, fps=25)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("usb"))
    win._offset_spin.setValue(-200.0)      # listen far from the tone: nothing there
    _pump(qapp, win, 3)
    snr = win.smeter._snr
    assert snr is not None
    assert 0.0 <= snr < 100.0, f"implausible S/N reading {snr}"
    win.close()


def test_smeter_snr_is_bounded_for_a_strong_signal(qapp):
    src = StubSource(_caps(), rate=768e3, center=7.1e6, tone_hz=1e3, noise=1e-6)
    win = window_for(src, fft_size=4096, fps=25)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    _pump(qapp, win, 3)
    assert 0.0 <= win.smeter._snr <= 100.0
    win.close()


# -- fine steps and swipe tuning ---------------------------------------------

def test_step_options_include_ssb_sized_increments(qapp):
    """1 kHz is far too coarse for SSB; a few hundred hertz changes intelligibility."""
    win = window_for(StubSource(_caps()), fft_size=1024)
    offered = [win._step_combo.itemData(i) for i in range(win._step_combo.count())]
    assert 100.0 in offered
    assert 10.0 in offered
    assert min(offered) <= 100.0
    win.close()


def test_selecting_a_100_hz_step_sets_the_spinbox_increment(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._step_combo.setCurrentText("100 Hz")
    assert win.step_hz == pytest.approx(100.0)
    # The spin box works in MHz, so 100 Hz is 0.0001 and must survive its precision.
    assert win._freq_spin.singleStep() == pytest.approx(1e-4)
    assert win._freq_spin.decimals() >= 4
    win.close()


def test_nudge_moves_by_exactly_one_step(qapp):
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("100 Hz")
    win.nudge_frequency(1)
    assert src.center_freq == pytest.approx(14.2e6 + 100.0)
    win.nudge_frequency(-3)
    assert src.center_freq == pytest.approx(14.2e6 - 200.0)
    win.close()


def test_nudge_respects_the_chosen_step(qapp):
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("1 kHz")
    win.nudge_frequency(2)
    assert src.center_freq == pytest.approx(14.2e6 + 2000.0)
    win.close()


def test_nudge_of_zero_does_nothing(qapp):
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024)
    assert win.nudge_frequency(0) == pytest.approx(14.2e6)
    win.close()


def test_step_choice_is_remembered(qapp, tmp_path):
    path = tmp_path / "s.json"
    first = window_for(StubSource(_caps()), fft_size=1024, settings=Settings(path))
    first._step_combo.setCurrentText("100 Hz")
    first.close()
    assert Settings.load(path).last.step_hz == pytest.approx(100.0)

    second = window_for(StubSource(_caps()), fft_size=1024,
                        step_hz=Settings.load(path).last.step_hz)
    assert second.step_hz == pytest.approx(100.0)
    second.close()


def _wheel(view, dx, dy, px=0, py=0, shift=False):
    """A wheel event of the kind a trackpad or mouse produces."""
    return QtGui.QWheelEvent(
        QtCore.QPointF(view.width() / 2, view.height() / 2),
        QtCore.QPointF(
            view.mapToGlobal(QtCore.QPoint(int(view.width() / 2), int(view.height() / 2)))
        ),
        QtCore.QPoint(int(px), int(py)),
        QtCore.QPoint(int(dx), int(dy)),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier if shift
        else QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.ScrollUpdate,
        False,
    )


def test_the_waterfall_does_not_tune_on_a_sideways_swipe(qapp):
    """Tuning lives on the FFT display only; the waterfall is for reading history."""
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win._step_combo.setCurrentText("100 Hz")
    assert not hasattr(win.waterfall, "frequencyNudged")
    win.waterfall.wheelEvent(_wheel(win.waterfall, 120, 0))
    assert src.center_freq == pytest.approx(14.2e6)
    win.close()


def test_sideways_swipe_tunes_with_pixel_deltas_only(qapp):
    """macOS trackpads populate pixelDelta and may leave angleDelta empty."""
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win._step_combo.setCurrentText("100 Hz")
    win.spectrum.wheelEvent(_wheel(win.spectrum, 0, 0, px=40, py=0))
    assert src.center_freq == pytest.approx(14.2e6 + 100.0)
    win.close()


def test_sideways_swipe_tunes_on_the_spectrum(qapp):
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win._step_combo.setCurrentText("100 Hz")
    win.spectrum.wheelEvent(_wheel(win.spectrum, -120, 0))
    assert src.center_freq == pytest.approx(14.2e6 - 100.0)
    win.close()


def test_vertical_swipe_still_zooms_and_does_not_tune(qapp):
    """The existing gesture must keep working untouched."""
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    before = src.center_freq
    (x0, x1), _ = win.spectrum.getViewBox().viewRange()
    win.spectrum.wheelEvent(_wheel(win.spectrum, 0, 120))
    assert src.center_freq == pytest.approx(before), "a vertical swipe tuned the radio"
    (z0, z1), _ = win.spectrum.getViewBox().viewRange()
    assert (z1 - z0) != pytest.approx(x1 - x0), "a vertical swipe no longer zooms"
    win.close()


def test_small_sideways_deltas_tune_gradually(qapp):
    """A trackpad sends many tiny deltas; each must not be a whole step."""
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win._step_combo.setCurrentText("100 Hz")
    for _ in range(5):
        win.spectrum.wheelEvent(_wheel(win.spectrum, 20, 0))
    assert src.center_freq == pytest.approx(14.2e6), "tuned before a full step"
    win.spectrum.wheelEvent(_wheel(win.spectrum, 20, 0))
    assert src.center_freq == pytest.approx(14.2e6 + 100.0)
    win.close()


def test_swipe_tuning_stops_a_scan(qapp, tmp_path):
    """Swiping is a manual tune, so it should take the dial back from the scanner."""
    src = StubSource(_airband_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25, settings=Settings(tmp_path / "s.json"))
    win.resize(900, 600)
    win.scanner_panel.apply_config(118e6, 137e6, 25e3, 10.0, True)
    win.start_scan()
    win.spectrum.wheelEvent(_wheel(win.spectrum, 120, 0))
    assert win.scanner is None
    win.close()


def test_a_fine_nudge_keeps_the_waterfall_and_audio_running(qapp):
    """Tuning 100 Hz by ear must not wipe the display or click the audio each step."""
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("100 Hz")
    _pump(qapp, win, 5)
    assert win.waterfall.buffer.written_rows == 5

    win.nudge_frequency(1)
    assert win.waterfall.buffer.written_rows == 5, "history cleared by a 100 Hz nudge"
    assert win.spectrum._smoothed is not None, "smoothing reset by a 100 Hz nudge"
    win.close()


def test_a_coarse_retune_still_clears_everything(qapp):
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    _pump(qapp, win, 5)
    win.waterfall.frequencySelected.emit(20e6)
    assert win.waterfall.buffer.written_rows == 0
    assert win.spectrum._smoothed is None
    win.close()


def test_the_fine_limit_is_one_display_bin(qapp):
    win = window_for(StubSource(_caps(), rate=768e3), fft_size=1024, waterfall_bins=1024)
    assert win.fine_tune_limit() == pytest.approx(750.0)
    # A 500 Hz step is fine, 1 kHz is not.
    win._step_combo.setCurrentText("500 Hz")
    assert win.step_hz < win.fine_tune_limit()
    win._step_combo.setCurrentText("1 kHz")
    assert win.step_hz > win.fine_tune_limit()
    win.close()


def test_the_fine_limit_shrinks_with_zoom(qapp):
    """Zoomed in, a smaller move already shifts the display by a whole bin."""
    win = window_for(StubSource(_caps(), rate=768e3), fft_size=1024, waterfall_bins=1024)
    wide = win.fine_tune_limit()
    win._zoom_combo.setCurrentText("8x")
    assert win.fine_tune_limit() == pytest.approx(wide / 8)
    win.close()


def test_a_fine_nudge_does_not_stop_an_iq_recording(qapp, tmp_path):
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25, recordings_dir=tmp_path)
    win._step_combo.setCurrentText("100 Hz")
    win.start_iq_recording()
    win.nudge_frequency(1)
    assert win.iq_recorder is not None, "a 100 Hz nudge killed the capture"
    win.nudge_frequency(100)          # 10 kHz: a real retune
    assert win.iq_recorder is None
    win.close()


def test_a_fine_nudge_does_not_flush_the_buffer(qapp):
    """Flushing on every 100 Hz step cost 3.7 s of audio silence, measured on hardware."""
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("100 Hz")
    win.nudge_frequency(1)
    assert src.last_flush is False

    win._step_combo.setCurrentText("10 kHz")
    win.nudge_frequency(1)
    assert src.last_flush is True


def test_shift_plus_vertical_swipe_tunes(qapp):
    """A guaranteed route: macOS may claim horizontal swipes, but never vertical ones."""
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win._step_combo.setCurrentText("100 Hz")
    win.spectrum.wheelEvent(_wheel(win.spectrum, 0, 120, shift=True))
    assert src.center_freq == pytest.approx(14.2e6 + 100.0)
    win.spectrum.wheelEvent(_wheel(win.spectrum, 0, -120, shift=True))
    assert src.center_freq == pytest.approx(14.2e6)
    win.close()


def test_shift_plus_vertical_does_not_zoom(qapp):
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    (x0, x1), _ = win.spectrum.getViewBox().viewRange()
    win.spectrum.wheelEvent(_wheel(win.spectrum, 0, 120, shift=True))
    (z0, z1), _ = win.spectrum.getViewBox().viewRange()
    assert (z1 - z0) == pytest.approx(x1 - x0, rel=1e-6), "shift+swipe zoomed as well"
    win.close()


def test_wheel_events_reach_the_handler_through_the_viewport(qapp):
    """Regression: Qt delivers wheel events to the viewport, not the view.

    Calling wheelEvent() directly passes even when the real delivery path is broken, so
    this drives it the way Qt actually does.
    """
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win._step_combo.setCurrentText("100 Hz")
    qapp.sendEvent(win.spectrum.viewport(), _wheel(win.spectrum, 120, 0))
    assert src.center_freq == pytest.approx(14.2e6 + 100.0), "not reached via the viewport"
    win.close()


def test_shift_vertical_reaches_the_handler_through_the_viewport(qapp):
    src = StubSource(_caps(), center=14.2e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win._step_combo.setCurrentText("100 Hz")
    qapp.sendEvent(win.spectrum.viewport(), _wheel(win.spectrum, 0, 120, shift=True))
    assert src.center_freq == pytest.approx(14.2e6 + 100.0)
    win.close()


# -- mute, CW, snap ----------------------------------------------------------

def test_mute_button_starts_unmuted_and_toggles(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    assert win.muted is False
    assert win._mute_button.text() == "Mute"
    win._mute_button.setChecked(True)
    assert win.muted is True
    assert win._mute_button.text() == "Muted"
    win.close()


def test_mute_does_not_change_the_volume_setting(qapp):
    """The point of a mute button: your level is still there when you come back."""
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._volume_slider.setValue(65)
    win._mute_button.setChecked(True)
    assert win._volume_slider.value() == 65
    win._mute_button.setChecked(False)
    assert win._volume_slider.value() == 65
    assert win.current_snapshot().volume == pytest.approx(0.65)
    win.close()


def test_mute_is_not_persisted(qapp, tmp_path):
    """Coming back muted with no explanation would look like a broken radio."""
    path = tmp_path / "s.json"
    win = window_for(StubSource(_caps()), fft_size=1024, settings=Settings(path))
    win._mute_button.setChecked(True)
    win.close()
    saved = Settings.load(path).last
    assert not hasattr(saved, "muted")


def test_cw_is_offered_as_a_mode(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    assert win._mode_combo.findData("cw") >= 0
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    assert win.mode == "cw"
    win.close()


def test_cw_offers_narrow_bandwidths(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    widths = [win._bw_audio_combo.itemData(i) for i in range(win._bw_audio_combo.count())]
    assert min(widths) <= 250.0
    assert max(widths) <= 1.5e3, "CW filters should all be narrow"
    assert win.bandwidth_hz() == pytest.approx(500.0)
    win.close()


def test_cw_passband_sits_at_the_pitch(qapp):
    """The tone appears above where you are listening, so shade it there."""
    from src.rgc_sdr.dsp.demod import MODE_SPECS as SPECS

    win = window_for(StubSource(_caps(), center=14.05e6), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    lo, hi = win.spectrum._passband.getRegion()
    assert (lo + hi) / 2 == pytest.approx(14.05e6 + SPECS["cw"].pitch_hz, abs=1.0)
    assert (hi - lo) == pytest.approx(500.0, abs=1.0)
    win.close()


def test_snap_rounds_tuning_to_the_step(qapp):
    src = StubSource(_airband_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("25 kHz")
    win._snap_check.setChecked(True)
    # 118.262 is 12 kHz above the 118.250 channel and 13 kHz below 118.275, so it
    # rounds down; 118.263 crosses the midpoint and rounds up.
    win.waterfall.frequencySelected.emit(118_262_000)
    assert src.center_freq == pytest.approx(118_250_000)
    win.waterfall.frequencySelected.emit(118_263_000)
    assert src.center_freq == pytest.approx(118_275_000)
    win.close()


def test_snap_off_tunes_exactly_where_asked(qapp):
    src = StubSource(_airband_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("25 kHz")
    win._snap_check.setChecked(False)
    win.waterfall.frequencySelected.emit(118_262_000)
    assert src.center_freq == pytest.approx(118_262_000)
    win.close()


def test_snap_uses_whatever_step_is_selected(qapp):
    src = StubSource(_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._snap_check.setChecked(True)
    win._step_combo.setCurrentText("100 Hz")
    win.spectrum.frequencySelected.emit(7_100_037.0)
    assert src.center_freq == pytest.approx(7_100_000.0)
    win._step_combo.setCurrentText("9 kHz")
    win.spectrum.frequencySelected.emit(1_000_000.0)
    assert src.center_freq == pytest.approx(999_000.0)   # nearest multiple of 9 kHz
    win.close()


def test_snap_applies_to_the_frequency_box(qapp):
    src = StubSource(_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("500 Hz")
    win._snap_check.setChecked(True)
    win._freq_spin.setValue(7.100_137)
    assert src.center_freq == pytest.approx(7_100_000.0)
    win.close()


def test_ticking_snap_realigns_immediately(qapp):
    """Enabling it should visibly do something rather than wait for the next tune."""
    src = StubSource(_caps(), center=7_100_037.0)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("100 Hz")
    win._snap_check.setChecked(True)
    assert src.center_freq == pytest.approx(7_100_000.0)
    win.close()


def test_snap_leaves_a_recalled_memory_exactly_where_it_was(qapp, tmp_path):
    """A saved frequency is a deliberate choice; snapping would quietly move it."""
    settings = Settings(tmp_path / "s.json")
    settings.add_memory("odd spot", Snapshot(freq_hz=7_100_037.0))
    src = StubSource(_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25, settings=settings)
    win._step_combo.setCurrentText("25 kHz")
    win._snap_check.setChecked(True)
    win.recall_memory("odd spot")
    assert src.center_freq == pytest.approx(7_100_037.0)
    win.close()


def test_snap_leaves_a_scanner_hit_alone(qapp, tmp_path):
    settings = Settings(tmp_path / "s.json")
    settings.record_found(118_262_500.0, -90.0, 20.0)
    src = StubSource(_airband_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25, settings=settings)
    win._refresh_scan_lists()
    win._step_combo.setCurrentText("25 kHz")
    win._snap_check.setChecked(True)
    win.scanner_panel.channelActivated.emit(118_262_500.0)
    assert src.center_freq == pytest.approx(118_262_500.0)
    win.close()


def test_snap_is_remembered(qapp, tmp_path):
    path = tmp_path / "s.json"
    first = window_for(StubSource(_caps()), fft_size=1024, settings=Settings(path))
    first._snap_check.setChecked(True)
    first.close()
    assert Settings.load(path).last.snap is True

    second = window_for(StubSource(_caps()), fft_size=1024, snap=True)
    assert second.snap_enabled is True
    second.close()


def test_snap_keeps_nudges_on_the_grid(qapp):
    src = StubSource(_caps(), center=7_100_037.0)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("100 Hz")
    win._snap_check.setChecked(True)          # realigns to 7_100_000
    win.nudge_frequency(1)
    assert src.center_freq == pytest.approx(7_100_100.0)
    win.close()


def test_no_gain_controls_shown_when_agc_is_not_honoured(qapp):
    """The Airspy claims AGC and ignores it, so there is nothing to show -- and nothing
    is shown, rather than a "no gain controls" label taking up the row."""
    win = window_for(StubSource(_caps(has_agc=False)), fft_size=1024)
    assert win._agc_check is None
    assert win._device_slot.isHidden()
    labels = [w.text() for w in win.findChildren(QtWidgets.QLabel)]
    assert not any("no gain" in t for t in labels)
    win.close()

def test_snapping_a_typed_frequency_updates_the_box(qapp):
    """The box must not keep showing what was typed while the radio sits elsewhere."""
    src = StubSource(_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._step_combo.setCurrentText("100 Hz")
    win._snap_check.setChecked(True)
    win._freq_spin.setValue(7.100137)
    assert src.center_freq == pytest.approx(7_100_100.0)
    assert win._freq_spin.value() == pytest.approx(7.1001, abs=1e-6)
    win.close()


def test_typing_an_exact_frequency_is_left_alone_with_snap_off(qapp):
    src = StubSource(_caps(), center=7.1e6)
    win = window_for(src, fft_size=1024, fps=25)
    win._snap_check.setChecked(False)
    win._freq_spin.setValue(7.100137)
    assert src.center_freq == pytest.approx(7_100_137.0)
    assert win._freq_spin.value() == pytest.approx(7.100137, abs=1e-6)
    win.close()


# -- CW zero beat ------------------------------------------------------------

class AbsoluteToneSource(StubSource):
    """A carrier at a fixed *absolute* frequency, so retuning really moves it.

    StubSource puts its tone at a constant offset from centre, which can never converge:
    the signal follows the radio. Zero-beating only means something against a signal
    that stays put.
    """

    def __init__(self, caps, centre, carrier_hz, amplitude=0.3, noise=1e-4):
        super().__init__(caps, center=centre, noise=noise)
        self.carrier_hz = float(carrier_hz)
        self._amplitude = amplitude

    def read_latest(self, n):
        offset = self.carrier_hz - self._center
        t = np.arange(n) / self._rate
        signal = self._amplitude * np.exp(2j * np.pi * offset * t)
        hiss = self._noise * (
            self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)
        )
        block = (signal + hiss).astype(np.complex64)
        self._ring.write(block)
        return block


def test_zero_beat_is_only_shown_in_cw(qapp):
    """Hidden, not greyed out: it means nothing in any other mode."""
    win = window_for(StubSource(_caps()), fft_size=1024)
    assert win._zerobeat_button.isHidden()
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    assert not win._zerobeat_button.isHidden()
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("usb"))
    assert win._zerobeat_button.isHidden()
    win.close()


def test_zero_beat_does_nothing_outside_cw(qapp):
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 150.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("usb"))
    assert win.zerobeat_once() is None
    assert src.center_freq == pytest.approx(14.05e6)
    win.close()


def test_zero_beat_corrects_upward(qapp):
    """A carrier above the tuning must pull the radio up, not down."""
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 180.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    correction = win.zerobeat_once()
    assert correction == pytest.approx(180.0, abs=6.0)
    assert src.center_freq == pytest.approx(14.05e6 + 180.0, abs=6.0)
    win.close()


def test_zero_beat_corrects_downward(qapp):
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 - 240.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    correction = win.zerobeat_once()
    assert correction == pytest.approx(-240.0, abs=6.0)
    assert src.center_freq == pytest.approx(14.05e6 - 240.0, abs=6.0)
    win.close()


def test_zero_beat_converges_and_then_holds(qapp):
    """Repeated presses must settle, not oscillate around the target."""
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 300.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    for _ in range(4):
        win.zerobeat_once()
    assert src.center_freq == pytest.approx(14.05e6 + 300.0, abs=5.0)
    # Already on tune: reports zero rather than jittering.
    assert win.zerobeat_once() == pytest.approx(0.0)
    win.close()


def test_zero_beat_ignores_a_signal_beyond_the_search_limit(qapp):
    """Bounded to 500 Hz, so it cannot wander off onto something else."""
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 2000.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    assert win.zerobeat_once() is None
    assert src.center_freq == pytest.approx(14.05e6)
    assert "no signal" in win._status.currentMessage()
    win.close()


def test_zero_beat_finds_a_carrier_at_the_edge_of_the_search(qapp):
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 450.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    assert win.zerobeat_once() == pytest.approx(450.0, abs=8.0)
    win.close()


def test_zero_beat_does_nothing_with_only_noise(qapp):
    src = StubSource(_caps(), center=14.05e6, tone_hz=96e3, noise=1e-3)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    before = src.center_freq
    win.zerobeat_once()
    assert src.center_freq == pytest.approx(before)
    win.close()


def test_zero_beat_follows_the_listening_offset(qapp):
    """It centres the carrier you are listening to, not whatever is nearest DC."""
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 20_100.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    win._offset_spin.setValue(20.0)                # listening 20 kHz up
    assert win.zerobeat_once() == pytest.approx(100.0, abs=6.0)
    win.close()


def test_zero_beat_ignores_snap(qapp):
    """A channel grid is exactly what zero-beating has to disregard."""
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 137.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    win._step_combo.setCurrentText("1 kHz")
    win._snap_check.setChecked(True)
    win.zerobeat_once()
    assert src.center_freq == pytest.approx(14.05e6 + 137.0, abs=6.0)
    win.close()


def test_zero_beat_only_runs_while_the_button_is_held(qapp):
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 150.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    assert not win._zerobeat_timer.isActive()
    win._zerobeat_button.pressed.emit()
    assert win._zerobeat_timer.isActive()
    win._zerobeat_button.released.emit()
    assert not win._zerobeat_timer.isActive()
    win.close()


def test_pressing_zero_beat_acts_immediately(qapp):
    """Holding it should respond at once, not after the first timer interval."""
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 200.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    win._zerobeat_button.pressed.emit()
    assert src.center_freq == pytest.approx(14.05e6 + 200.0, abs=8.0)
    win._zerobeat_button.released.emit()
    win.close()


def test_zero_beat_stops_on_close(qapp):
    src = AbsoluteToneSource(_caps(), 14.05e6, 14.05e6 + 150.0)
    win = window_for(src, fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    win._zerobeat_button.pressed.emit()
    win.close()
    assert not win._zerobeat_timer.isActive()


def test_pitch_selector_only_shows_in_cw(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win.show()
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    assert not win._pitch_combo.isVisible()
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    assert win._pitch_combo.isVisible()
    win.close()


def test_pitch_defaults_to_500(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    assert win.pitch_hz == pytest.approx(500.0)
    win.close()


def test_pitch_moves_the_passband(qapp):
    """The shaded band must follow the pitch, since that is where the tone appears."""
    win = window_for(StubSource(_caps(), center=7.015e6), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    win._pitch_combo.setCurrentText("400 Hz")
    lo, hi = win.spectrum._passband.getRegion()
    assert (lo + hi) / 2 == pytest.approx(7.015e6 + 400.0, abs=1.0)
    win._pitch_combo.setCurrentText("800 Hz")
    lo, hi = win.spectrum._passband.getRegion()
    assert (lo + hi) / 2 == pytest.approx(7.015e6 + 800.0, abs=1.0)
    win.close()


def test_pitch_is_remembered(qapp, tmp_path):
    path = tmp_path / "s.json"
    first = window_for(StubSource(_caps()), fft_size=1024, settings=Settings(path))
    first._mode_combo.setCurrentIndex(first._mode_combo.findData("cw"))
    first._pitch_combo.setCurrentText("600 Hz")
    first.close()
    assert Settings.load(path).last.pitch_hz == pytest.approx(600.0)

    second = window_for(StubSource(_caps()), fft_size=1024, pitch_hz=600.0)
    assert second.pitch_hz == pytest.approx(600.0)
    second.close()


def test_zero_beat_targets_whatever_pitch_is_selected(qapp):
    """Zero beat centres the carrier; the chain then renders it at the chosen pitch, so
    changing the pitch must not change what zero beat does."""
    for pitch in ("400 Hz", "700 Hz"):
        src = AbsoluteToneSource(_caps(), 7.015e6, 7.015e6 + 160.0)
        win = window_for(src, fft_size=1024)
        win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
        win._pitch_combo.setCurrentText(pitch)
        assert win.zerobeat_once() == pytest.approx(160.0, abs=6.0)
        win.close()


# -- zoom survives tuning ----------------------------------------------------

def _view_span(win):
    (lo, hi), _ = win.waterfall.getViewBox().viewRange()
    return lo, hi, hi - lo


def test_pinch_zoom_survives_a_retune(qapp):
    """Zoom in, click the station next door, and the zoom must still be there."""
    src = StubSource(_caps(), rate=768e3, center=7.100e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    _pump(qapp, win, 2)

    # Zoom to a 60 kHz window, as a two-finger swipe would.
    win.waterfall.getViewBox().setXRange(7.100e6 - 30e3, 7.100e6 + 30e3, padding=0)
    _, _, before = _view_span(win)
    assert before == pytest.approx(60e3, rel=0.02)

    win.waterfall.frequencySelected.emit(7.120e6)
    lo, hi, after = _view_span(win)
    assert after == pytest.approx(before, rel=0.02), "zoom was reset by tuning"
    # And the station just tuned is in view, near the middle.
    assert lo < 7.120e6 < hi
    assert (lo + hi) / 2 == pytest.approx(7.120e6, abs=before * 0.05)
    win.close()


def test_zoom_survives_a_fine_nudge(qapp):
    src = StubSource(_caps(), rate=768e3, center=7.100e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win._step_combo.setCurrentText("100 Hz")
    win.waterfall.getViewBox().setXRange(7.100e6 - 5e3, 7.100e6 + 5e3, padding=0)
    _, _, before = _view_span(win)
    win.nudge_frequency(3)
    _, _, after = _view_span(win)
    assert after == pytest.approx(before, rel=0.02)
    win.close()


def test_a_full_span_view_is_left_alone(qapp):
    """Nothing to preserve when not zoomed; the view should track the new span."""
    src = StubSource(_caps(), rate=768e3, center=7.100e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    _pump(qapp, win, 2)
    win.waterfall.frequencySelected.emit(7.300e6)
    lo, hi, span = _view_span(win)
    assert span == pytest.approx(768e3, rel=0.02)
    assert (lo + hi) / 2 == pytest.approx(7.300e6, abs=1e3)
    win.close()


def test_changing_the_decimation_resets_the_view(qapp):
    """The user changed the span deliberately, so show the new one."""
    src = StubSource(_caps(), rate=768e3, center=7.100e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win.waterfall.getViewBox().setXRange(7.100e6 - 30e3, 7.100e6 + 30e3, padding=0)
    win._zoom_combo.setCurrentText("8x")
    _, _, span = _view_span(win)
    assert span == pytest.approx(96e3, rel=0.02)
    win.close()


def test_changing_the_sample_rate_resets_the_view(qapp):
    src = StubSource(_caps(sample_rates=(768e3, 192e3)), rate=768e3, center=7.100e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win.waterfall.getViewBox().setXRange(7.100e6 - 30e3, 7.100e6 + 30e3, padding=0)
    win._rate_combo.setCurrentText("192 kS/s")
    _, _, span = _view_span(win)
    assert span == pytest.approx(192e3, rel=0.02)
    win.close()


def test_a_preserved_zoom_is_clamped_inside_the_span(qapp):
    """Tuning near the edge must not scroll the view off the available spectrum."""
    src = StubSource(_caps(), rate=768e3, center=7.100e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win.waterfall.getViewBox().setXRange(7.100e6 - 100e3, 7.100e6 + 100e3, padding=0)
    win.waterfall.frequencySelected.emit(7.100e6)
    lo, hi, span = _view_span(win)
    full_lo = src.center_freq - 768e3 / 2
    full_hi = src.center_freq + 768e3 / 2
    assert lo >= full_lo - 1.0 and hi <= full_hi + 1.0
    assert span == pytest.approx(200e3, rel=0.02)
    win.close()


def test_the_spectrum_follows_the_preserved_zoom(qapp):
    """Both plots share one frequency axis, so they must stay in step."""
    src = StubSource(_caps(), rate=768e3, center=7.100e6)
    win = window_for(src, fft_size=1024, fps=25)
    win.resize(900, 600)
    win.waterfall.getViewBox().setXRange(7.100e6 - 20e3, 7.100e6 + 20e3, padding=0)
    win.waterfall.frequencySelected.emit(7.130e6)
    (wlo, whi), _ = win.waterfall.getViewBox().viewRange()
    (slo, shi), _ = win.spectrum.getViewBox().viewRange()
    assert slo == pytest.approx(wlo, abs=500.0)
    assert shi == pytest.approx(whi, abs=500.0)
    win.close()


# -- decoded CW line ---------------------------------------------------------

def test_cw_line_only_appears_in_cw_mode(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win.show()
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("usb"))
    assert not win._info_label.isVisible()
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    assert win._info_label.isVisible()
    win.close()


def test_cw_line_is_cleared_when_leaving_cw(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))
    win._info_label.setText("CQ CQ DE G4ABC")
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    assert win._info_label.text() == ""
    win.close()


def test_cw_line_shows_decoded_text_and_speed(qapp):
    """The label renders whatever the sink has decoded."""
    class Decoding(StubSource):
        pass

    win = window_for(Decoding(_caps()), fft_size=1024, fps=25)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))

    class FakeSink:
        cw_text = "CQ CQ DE G4ABC K"
        cw_wpm = 22.0

    win.audio = FakeSink()
    try:
        win._update_cw_text()
        assert "CQ CQ DE G4ABC K" in win._info_label.text()
        assert "22 wpm" in win._info_label.text()
    finally:
        win.audio = None
    win.close()


def test_cw_line_is_blank_when_nothing_has_been_decoded(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024, fps=25)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("cw"))

    class Silent:
        cw_text = ""
        cw_wpm = 0.0

    win.audio = Silent()
    try:
        win._update_cw_text()
        assert win._info_label.text() == ""
    finally:
        win.audio = None
    win.close()


def test_cw_line_is_right_aligned_so_it_grows_leftwards(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    assert win._info_label.alignment() & QtCore.Qt.AlignmentFlag.AlignRight
    win.close()


# -- choosing the SDR --------------------------------------------------------

from src.rgc_sdr.device.profiles import (  # noqa: E402
    Availability,
    PROFILES,
    caps_from_profile,
    profile_for,
)
from src.rgc_sdr.device.source import SettingInfo  # noqa: E402


class ProfiledStub(StubSource):
    """A stand-in for any supported radio, built from its profile."""

    def __init__(self, key, centre, settings=()):
        profile = profile_for(key)
        caps = caps_from_profile(profile)
        if settings:
            from dataclasses import replace
            caps = replace(caps, settings=tuple(settings))
        super().__init__(caps, rate=profile.default_rate, center=centre)
        self.profile = profile
        self.closed = False
        self.started = False
        self.written = {}

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def read_setting(self, key):
        return self.written.get(key, False)

    def write_setting(self, key, enabled):
        self.written[key] = enabled


def fleet(connected=("airspyhf", "hackrf", "rtlsdr", "plutosdr", "airspy")):
    """Availability as if these radios were plugged in."""
    def lister():
        return [Availability(p, p.key in connected, p.key in connected) for p in PROFILES]
    return lister


def switching_window(connected=("airspyhf", "hackrf", "rtlsdr", "plutosdr", "airspy"),
                     start="airspyhf", settings=None, fail=()):
    opened = []

    def factory(driver, centre):
        if driver in fail:
            raise RuntimeError(f"{driver} would not open")
        src = ProfiledStub(driver, centre,
                           settings=(SettingInfo("biastee", "Bias-T", "Antenna power"),)
                           if driver in ("hackrf", "rtlsdr") else ())
        opened.append(src)
        return src

    first = ProfiledStub(start, profile_for(start).default_freq)
    win = window_for(first, fft_size=1024, fps=25, source_factory=factory,
                     availability_fn=fleet(connected), settings=settings)
    return win, first, opened


def test_sdr_selector_lists_every_supported_radio(qapp):
    win, _, _ = switching_window()
    keys = [win._device_combo.itemData(i) for i in range(win._device_combo.count())]
    assert set(keys) == {p.key for p in PROFILES}
    assert win._device_combo.currentData() == "airspyhf"
    win.close()


def test_selector_says_which_radios_are_unavailable(qapp):
    win, _, _ = switching_window(connected=("airspyhf",))
    labels = {win._device_combo.itemData(i): win._device_combo.itemText(i)
              for i in range(win._device_combo.count())}
    assert "not connected" in labels["hackrf"] or "not installed" in labels["hackrf"]
    assert labels["airspyhf"] == "Airspy HF+"
    win.close()


def test_airspy_hf_shows_no_gain_section_at_all(qapp):
    win, _, _ = switching_window()
    assert win._device_slot.isHidden()
    assert not any("no gain" in w.text() for w in win.findChildren(QtWidgets.QLabel))
    win.close()


def test_switching_to_hackrf_brings_up_its_gain_stages(qapp):
    win, first, opened = switching_window()
    assert win.switch_device("hackrf") is True
    assert win.current_device_key() == "hackrf"
    assert not win._device_slot.isHidden()
    names = {w.text() for w in win._device_slot.findChildren(QtWidgets.QLabel)}
    assert {"LNA", "VGA"} <= names
    spins = win._device_slot.findChildren(QtWidgets.QDoubleSpinBox)
    assert len(spins) == 2
    # AMP is 0 or 14 dB and nothing between, so it is a switch rather than a spinbox.
    checks = {c.text(): c for c in win._device_slot.findChildren(QtWidgets.QCheckBox)}
    assert "AMP" in checks
    assert "HackRF" in win.windowTitle()
    win.close()


def test_radio_settings_have_their_own_row(qapp):
    """A HackRF's gains pushed the tuning row off the screen; they live below it now."""
    win, _, _ = switching_window()
    assert win._radio_row.isHidden()                 # the HF+ has nothing to set
    win.switch_device("hackrf")
    assert not win._radio_row.isHidden()
    assert win._device_slot.parentWidget() is win._radio_row
    assert win._device_slot.parentWidget() is not win._freq_spin.parentWidget()
    win.switch_device("airspyhf")
    assert win._radio_row.isHidden()
    win.close()


def test_amp_switch_sets_zero_or_full_gain(qapp):
    win, _, opened = switching_window()
    win.switch_device("hackrf")
    amp = {c.text(): c for c in win._device_slot.findChildren(QtWidgets.QCheckBox)}["AMP"]
    amp.setChecked(True)
    assert win.source.gain == ("AMP", 14.0)
    amp.setChecked(False)
    assert win.source.gain == ("AMP", 0.0)
    win.close()


def test_switching_offers_the_new_radios_rates(qapp):
    win, _, _ = switching_window()
    win.switch_device("hackrf")
    rates = [win._rate_combo.itemData(i) for i in range(win._rate_combo.count())]
    assert set(rates) == set(profile_for("hackrf").sample_rates)
    assert not win._rate_combo.isHidden()
    assert any("MS/s" in win._rate_combo.itemText(i) for i in range(win._rate_combo.count()))
    win.close()


def test_switching_back_to_the_hf_plus_hides_the_gain_section_again(qapp):
    win, _, _ = switching_window()
    win.switch_device("hackrf")
    assert not win._device_slot.isHidden()
    win.switch_device("airspyhf")
    assert win._device_slot.isHidden()
    win.close()


def test_driver_settings_become_checkboxes(qapp):
    """A bias-tee, or whatever the driver advertises, appears without per-radio code."""
    win, _, opened = switching_window()
    win.switch_device("rtlsdr")
    checks = {c.text(): c for c in win._device_slot.findChildren(QtWidgets.QCheckBox)}
    assert "Bias-T" in checks
    checks["Bias-T"].setChecked(True)
    assert opened[-1].written["biastee"] is True
    win.close()


def test_frequency_is_kept_when_the_new_radio_can_tune_it(qapp):
    win, first, _ = switching_window()
    win._freq_spin.setValue(7.1)
    win.switch_device("hackrf")                 # 1 MHz - 6 GHz covers 7.1 MHz
    assert win.source.center_freq == pytest.approx(7.1e6)
    win.close()


def test_frequency_moves_somewhere_useful_when_it_cannot(qapp):
    win, _, _ = switching_window()
    win._freq_spin.setValue(7.1)
    win.switch_device("rtlsdr")                 # starts at 24 MHz
    assert win.source.center_freq == pytest.approx(profile_for("rtlsdr").default_freq)
    assert win._freq_spin.minimum() >= 24.0 - 1e-6
    win.close()


def test_the_old_radio_is_released(qapp):
    """Otherwise switching back to it fails with 'Unable to open'."""
    win, first, _ = switching_window()
    win.switch_device("hackrf")
    assert first.closed
    win.close()


def test_an_unconnected_radio_is_refused_and_the_current_one_kept(qapp):
    win, first, opened = switching_window(connected=("airspyhf",))
    assert win.switch_device("hackrf") is False
    assert win.source is first and not first.closed
    assert opened == []
    assert win._device_combo.currentData() == "airspyhf"
    assert "hackrf" not in win._status.currentMessage().lower() or \
        "connected" in win._status.currentMessage() or "install" in win._status.currentMessage()
    win.close()


def test_a_missing_driver_explains_how_to_install_it(qapp):
    win, _, _ = switching_window(connected=("airspyhf",))
    win.switch_device("rtlsdr")
    assert "soapyrtlsdr" in win._status.currentMessage()
    win.close()


def test_a_radio_that_fails_to_open_falls_back_to_the_previous_one(qapp):
    win, first, opened = switching_window(fail=("hackrf",))
    assert win.switch_device("hackrf") is False
    assert win.source.caps.driver == "airspyhf"
    assert "could not open" in win._status.currentMessage()
    win.close()


def test_audio_mode_survives_a_switch(qapp):
    win, _, _ = switching_window()
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    win.switch_device("hackrf")
    assert win.mode == "am"
    win.close()


def test_a_scan_is_stopped_by_a_switch(qapp, tmp_path):
    win, _, _ = switching_window(settings=Settings(tmp_path / "s.json"))
    win.scanner_panel.apply_config(118e6, 137e6, 25e3, 10.0, True)
    win.start_scan()
    assert win.scanner is not None
    win.switch_device("hackrf")
    assert win.scanner is None
    win.close()


def test_the_chosen_radio_is_remembered(qapp, tmp_path):
    path = tmp_path / "s.json"
    win, _, _ = switching_window(settings=Settings(path))
    win.switch_device("hackrf")
    win.close()
    assert Settings.load(path).device == "hackrf"


def test_scanner_presets_outside_the_radio_are_greyed_out(qapp):
    """The RTL-SDR starts at 24 MHz, so medium wave and 40 m are not offered."""
    win, _, _ = switching_window()
    win.switch_device("rtlsdr")
    combo = win.scanner_panel._preset_combo
    model = combo.model()
    enabled = {combo.itemText(i): model.item(i).isEnabled() for i in range(1, combo.count())}
    assert enabled["MW broadcast"] is False
    assert enabled["40 m amateur"] is False
    assert enabled["Airband (VHF AM)"] is True
    win.close()


def test_scanner_steps_further_round_dc_on_spiky_radios(qapp, tmp_path):
    win, _, _ = switching_window(settings=Settings(tmp_path / "s.json"))
    assert win._scan_config().dc_guard_hz == pytest.approx(1.5e3)   # HF+: no spike
    win.switch_device("hackrf")
    assert win._scan_config().dc_guard_hz == pytest.approx(5e3)
    win.close()


def test_choosing_the_current_radio_again_does_nothing(qapp):
    win, first, opened = switching_window()
    win._on_device_chosen(win._device_combo.findData("airspyhf"))
    assert opened == [] and not first.closed
    win.close()


def test_sdr_selector_does_not_widen_the_row_with_status_text(qapp):
    win, _, _ = switching_window(connected=("airspyhf",))
    win.resize(1400, 800)
    win.show()
    qapp.processEvents()
    assert win._device_combo.width() < 260
    win.close()


# -- mode-specific controls and the broadcast FM info line -------------------

from src.rgc_sdr.dsp.rds import StationInfo  # noqa: E402


def _visible_in(win, mode):
    win._mode_combo.setCurrentIndex(win._mode_combo.findData(mode))
    return {
        "pitch": not win._pitch_combo.isHidden(),
        "zerobeat": not win._zerobeat_button.isHidden(),
        "info": not win._info_label.isHidden(),
        "stereo": not win._stereo_check.isHidden(),
    }


@pytest.mark.parametrize("mode", ["off", "am", "nbfm", "usb", "lsb"])
def test_cw_and_fm_extras_are_hidden_in_other_modes(qapp, mode):
    win = window_for(StubSource(_caps()), fft_size=1024)
    shown = _visible_in(win, mode)
    assert not any(shown.values()), f"{mode} shows {[k for k, v in shown.items() if v]}"
    win.close()


def test_cw_shows_pitch_zero_beat_and_decoded_text_only(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    assert _visible_in(win, "cw") == {"pitch": True, "zerobeat": True,
                                       "info": True, "stereo": False}
    win.close()


def test_wbfm_shows_stereo_and_the_info_line_only(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    assert _visible_in(win, "wbfm") == {"pitch": False, "zerobeat": False,
                                         "info": True, "stereo": True}
    win.close()


class FakeFmSink:
    def __init__(self, stereo=True, ps="BBC R2", text="", pty=10, pi=0xC202):
        self.stereo = stereo
        info = StationInfo()
        if ps:
            info.ps = list(ps.ljust(8))
            info.ps_seen = {0, 1, 2, 3}
        info.rt_complete = text
        info.pty = pty
        info.pi = pi
        self.rds = info


def test_info_line_shows_stereo_and_station_name(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("wbfm"))
    win.audio = FakeFmSink(stereo=True, ps="BBC R2", text="Now playing: Test")
    try:
        win._update_info_line()
        text = win._info_label.text()
        assert text.startswith("STEREO")
        assert "BBC R2" in text and "Pop Music" in text
        assert "Now playing" in text
        assert "PI C202" in win._info_label.toolTip()
    finally:
        win.audio = None
    win.close()


def test_info_line_says_mono_without_a_pilot(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("wbfm"))
    win.audio = FakeFmSink(stereo=False, ps="")
    try:
        win._update_info_line()
        assert win._info_label.text().startswith("MONO")
    finally:
        win.audio = None
    win.close()


def test_long_radio_text_is_elided_not_allowed_to_widen_the_row(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("wbfm"))
    win.audio = FakeFmSink(text="X" * 64)
    try:
        win._update_info_line()
        metrics = QtGui.QFontMetrics(win._info_label.font())
        assert metrics.horizontalAdvance(win._info_label.text()) <= win._info_label.maximumWidth()
    finally:
        win.audio = None
    win.close()


def test_stereo_choice_is_remembered(qapp, tmp_path):
    path = tmp_path / "s.json"
    win = window_for(StubSource(_caps()), fft_size=1024, settings=Settings(path))
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("wbfm"))
    win._stereo_check.setChecked(False)
    win.close()
    assert Settings.load(path).last.stereo is False


# -- why there is no sound ---------------------------------------------------
# The status line is redrawn every frame, so a one-off message is gone before it can
# be read. The reason for silence has to be part of the line itself.

def test_status_says_audio_is_off_when_no_mode_is_chosen(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("off"))
    assert "audio off (choose a Mode)" in win._audio_status()
    win.close()


def test_status_keeps_saying_there_is_no_audio_device(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)       # audio disabled
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("wbfm"))
    assert "no audio: no audio device available" in win._audio_status()
    win.close()


def test_status_keeps_the_reason_audio_failed_to_start(qapp, monkeypatch):
    import src.rgc_sdr.ui.main_window as mw

    class BrokenSink:
        def __init__(self, *args, **kwargs):
            pass

        def set_force_mono(self, mono):
            pass

        def start(self):
            raise RuntimeError("output device busy")

    monkeypatch.setattr(mw, "AudioSink", BrokenSink)
    win = window_for(StubSource(_caps()), fft_size=1024)
    win._audio_ok = True
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("wbfm"))
    assert win.audio is None
    assert "could not start audio: output device busy" in win._audio_status()
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("off"))
    assert "audio off" in win._audio_status()
    win.close()



# -- layout, scanner button, and what a memory keeps -------------------------

def test_info_line_sits_beside_peak_hold(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    row = win._peak_check.parentWidget().layout()
    widgets = [row.itemAt(i).widget() for i in range(row.count())]
    widgets = [w for w in widgets if w is not None]          # drop spacers
    assert widgets.index(win._info_label) == widgets.index(win._peak_check) + 1
    win.close()


def test_scan_button_sits_right_of_zoom_and_shows_the_scanner(qapp):
    win = window_for(StubSource(_caps()), fft_size=1024)
    row = win._zoom_combo.parentWidget().layout()
    widgets = [row.itemAt(i).widget() for i in range(row.count())]
    assert widgets.index(win._scan_button) == widgets.index(win._zoom_combo) + 1
    assert win._scanner_dock.isHidden() and not win._scan_button.isChecked()
    win._scan_button.click()
    assert not win._scanner_dock.isHidden()
    win._scanner_dock.close()                     # the dock's own close box
    assert not win._scan_button.isChecked()
    win.close()


def test_memory_keeps_bandwidth_snap_zoom_rate_step_and_squelch(qapp, tmp_path):
    settings = Settings(tmp_path / "s.json")
    src = StubSource(_caps(sample_rates=(768e3, 192e3)))
    win = window_for(src, fft_size=1024, settings=settings)
    win._rate_combo.setCurrentIndex(win._rate_combo.findData(192e3))
    win._zoom_combo.setCurrentText("4x")
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    win._bw_audio_combo.setCurrentIndex(win._bw_audio_combo.count() - 1)
    wanted_bw = win.bandwidth_hz()
    win._step_combo.setCurrentIndex(win._step_combo.findData(9e3))
    win._snap_check.setChecked(False)
    win._squelch_check.setChecked(True)
    win._squelch_spin.setValue(-77.0)
    win.save_memory("everything")

    win._rate_combo.setCurrentIndex(win._rate_combo.findData(768e3))
    win._zoom_combo.setCurrentText("1x")
    win._bw_audio_combo.setCurrentIndex(0)
    win._step_combo.setCurrentIndex(win._step_combo.findData(1e3))
    win._snap_check.setChecked(True)
    win._squelch_check.setChecked(False)

    assert win.recall_memory("everything")
    assert src.sample_rate == 192e3
    assert win.decimator.factor == 4
    assert win.bandwidth_hz() == wanted_bw
    assert win.step_hz == 9e3
    assert not win._snap_check.isChecked()
    assert win._squelch_value() == pytest.approx(-77.0)
    win.close()
