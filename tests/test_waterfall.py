"""Waterfall buffer and bin reduction tests: no Qt, no radio."""

import numpy as np
import pytest

from src.rgc_sdr.dsp.waterfall import FILL_DBFS, WaterfallBuffer, reduce_max


def test_newest_row_is_first_and_history_scrolls():
    wf = WaterfallBuffer(rows=4, cols=8)
    for i in range(3):
        wf.push(np.full(8, float(i), dtype=np.float32))
    assert wf.image[0][0] == 2.0
    assert wf.image[1][0] == 1.0
    assert wf.image[2][0] == 0.0
    assert wf.image[3][0] == FILL_DBFS  # not yet written


def test_history_correct_after_wraparound():
    rows = 8
    wf = WaterfallBuffer(rows=rows, cols=4)
    for i in range(100):
        wf.push(np.full(4, float(i), dtype=np.float32))
    expected = [99.0 - k for k in range(rows)]
    assert [wf.image[k][0] for k in range(rows)] == expected


def test_push_reduces_oversized_row():
    wf = WaterfallBuffer(rows=2, cols=16)
    wf.push(np.arange(4096, dtype=np.float32))
    assert wf.image[0].size == 16
    assert wf.image[0][-1] == 4095.0  # max-pooled, so the bucket maximum survives


def test_clear_and_resize():
    wf = WaterfallBuffer(rows=3, cols=8)
    wf.push(np.zeros(8, dtype=np.float32))
    wf.clear()
    assert np.all(wf.image == FILL_DBFS)
    wf.resize_cols(32)
    assert wf.cols == 32 and wf.image.shape == (3, 32)


def test_reduce_max_preserves_single_bin_carrier():
    """The reason max-pooling is used: averaging buries a one-bin carrier."""
    n, width = 4096, 512
    x = np.full(n, -110.0, dtype=np.float32)
    x[1234] = -40.0
    pooled = reduce_max(x, width)
    assert pooled.max() == pytest.approx(-40.0)
    averaged = x.reshape(width, n // width).mean(axis=1)
    assert averaged.max() < -95.0  # carrier diluted away


def test_reduce_max_handles_non_divisible_width():
    """reshape-based pooling would turn trailing buckets into padding here."""
    x = np.arange(100, dtype=np.float32)
    pooled = reduce_max(x, 30)
    assert pooled.size == 30
    assert pooled[-1] == 99.0
    assert np.all(np.diff(pooled) > 0)
    assert np.all(np.isfinite(pooled))


def test_reduce_max_covers_every_input_sample():
    n, width = 997, 64
    x = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    assert reduce_max(x, width).max() == pytest.approx(x.max())


def test_reduce_max_passthrough_when_width_exceeds_bins():
    x = np.arange(10, dtype=np.float32)
    assert np.array_equal(reduce_max(x, 10), x)
    assert np.array_equal(reduce_max(x, 50), x)


def test_percentile_levels_ignore_unwritten_rows():
    wf = WaterfallBuffer(rows=64, cols=32)
    rng = np.random.default_rng(2)
    for _ in range(8):
        wf.push((rng.standard_normal(32) * 2 - 100).astype(np.float32))
    lo, hi = wf.percentile_levels()
    assert -120.0 < lo < -90.0 and lo < hi
    assert hi < -80.0  # the -140 fill rows must not drag the range down


def test_percentile_levels_on_empty_history():
    lo, hi = WaterfallBuffer(rows=4, cols=4).percentile_levels()
    assert (lo, hi) == (-115.0, -40.0)


def test_written_rows_tracks_pushes_and_saturates():
    wf = WaterfallBuffer(rows=4, cols=8)
    assert wf.written_rows == 0
    for expected in (1, 2, 3, 4, 4, 4):
        wf.push(np.zeros(8, dtype=np.float32))
        assert wf.written_rows == expected


def test_written_rows_resets_on_clear_and_resize():
    wf = WaterfallBuffer(rows=4, cols=8)
    wf.push(np.zeros(8, dtype=np.float32))
    wf.clear()
    assert wf.written_rows == 0
    wf.push(np.zeros(8, dtype=np.float32))
    wf.resize_cols(16)
    assert wf.written_rows == 0


def test_percentile_levels_use_very_quiet_rows_not_mistaken_for_fill():
    """A real bin below the -140 fill sentinel must still count as data."""
    wf = WaterfallBuffer(rows=8, cols=16)
    wf.push(np.full(16, -294.0, dtype=np.float32))
    lo, hi = wf.percentile_levels()
    assert lo < -290.0, f"quiet row ignored: got {lo:.1f}..{hi:.1f}"
