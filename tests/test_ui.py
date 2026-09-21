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

from src.rgc_sdr.device.source import DeviceCaps, FreqRange, GainElement, IQSource  # noqa: E402
from src.rgc_sdr.dsp.spectrum import SpectrumAnalyzer  # noqa: E402
from src.rgc_sdr.ui.main_window import MainWindow  # noqa: E402


class StubSource(IQSource):
    def __init__(self, caps, rate=768e3, center=7.1e6, tone_hz=96e3):
        self._caps, self._rate, self._center, self._tone = caps, rate, center, tone_hz

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

    def read_latest(self, n):
        t = np.arange(n) / self._rate
        return (0.5 * np.exp(2j * np.pi * self._tone * t)).astype(np.complex64)


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
    written = win.waterfall.buffer.image[win.waterfall.buffer.image > -140.0]
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
