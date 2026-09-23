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


def _pump(app, window, frames):
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
