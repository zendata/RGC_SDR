"""UI tests, run windowless via QT_QPA_PLATFORM=offscreen (set in conftest).

`StubSource` is test scaffolding for the widgets -- a test double for the IQSource
protocol, not a simulated device mode in the application (PLANNING.md section 1). It lets
the UI be regression-tested without the radio attached.
"""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6 import QtCore, QtWidgets  # noqa: E402

from src.rgc_sdr.device.source import (  # noqa: E402
    DeviceCaps,
    FreqRange,
    GainElement,
    IQSource,
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

    def set_center_freq(self, hz):
        self._center = self._caps.clamp_freq(hz)
        return self._center

    def set_sample_rate(self, hz):
        self._rate = self._caps.nearest_sample_rate(hz)
        return self._rate

    def set_bandwidth(self, hz):
        self.bandwidth_set = hz
        return hz

    def read_latest(self, n):
        t = np.arange(n) / self._rate
        tone = 0.5 * np.exp(2j * np.pi * self._tone * t)
        noise = self._noise * (self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n))
        return (tone + noise).astype(np.complex64)


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


def window_for(src, **kw):
    """MainWindow with audio off: tests must not open a real output device."""
    kw.setdefault("enable_audio", False)
    return MainWindow(src, **kw)


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


def test_no_rate_combo_when_only_one_rate(qapp):
    win = MainWindow(StubSource(_caps(sample_rates=(768e3,))), fft_size=1024)
    assert win._rate_combo is None
    win.close()


def test_no_bandwidth_control_for_airspyhf(qapp):
    """Capability-driven: this driver reports no bandwidth options, so none is shown."""
    win = MainWindow(StubSource(_caps(bandwidths=())), fft_size=1024)
    assert win._bw_combo is None
    win.close()


def test_bandwidth_control_appears_when_the_driver_offers_options(qapp):
    src = StubSource(_caps(driver="hackrf", bandwidths=(1.75e6, 2.5e6, 3.5e6)))
    win = MainWindow(src, fft_size=1024)
    assert win._bw_combo is not None and win._bw_combo.count() == 3
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
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("am"))
    assert not win._squelch_check.isEnabled()
    win._mode_combo.setCurrentIndex(win._mode_combo.findData("nbfm"))
    assert win._squelch_check.isEnabled()
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
