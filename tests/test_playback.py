"""IQ playback: a recording replayed through the same ring a radio fills."""

import json
import time

import numpy as np
import pytest

from src.rgc_sdr.device.playback import FileIQSource, read_sidecar
from src.rgc_sdr.device.source import _Ring
from src.rgc_sdr.recorder import IQRecorder

RATE = 48_000.0
CENTRE = 145.0e6


def _write(tmp_path, samples, rate=RATE, centre=CENTRE, **meta):
    path = tmp_path / "rec.cf32"
    samples.astype(np.complex64).tofile(path)
    side = {"format": "complex64", "byte_order": "little", "sample_rate_hz": rate,
            "center_freq_hz": centre, **meta}
    (tmp_path / "rec.cf32.json").write_text(json.dumps(side))
    return path


def _tone(offset_hz, n, rate=RATE):
    return np.exp(2j * np.pi * offset_hz * np.arange(n) / rate)


def _peak_hz(x, rate=RATE):
    spec = np.abs(np.fft.fftshift(np.fft.fft(x)))
    return (np.argmax(spec) - x.size / 2) * rate / x.size


def test_caps_describe_the_recording(tmp_path):
    src = FileIQSource(_write(tmp_path, _tone(0, 4800)))
    assert src.sample_rate == RATE
    assert src.center_freq == CENTRE
    assert src.caps.sample_rates == (RATE,)
    assert src.caps.covers(CENTRE + RATE / 2 - 1) and not src.caps.covers(CENTRE + RATE)
    assert src.caps.tx is None and src.caps.gain_elements == ()
    assert src.duration_s == pytest.approx(0.1)


def test_blocks_come_out_in_order(tmp_path):
    data = (np.arange(4800) + 0j)
    src = FileIQSource(_write(tmp_path, data))
    first, second = src.next_block(), src.next_block()
    assert np.array_equal(np.concatenate([first, second]), data[:first.size + second.size])
    assert np.array_equal(src.read_latest(second.size), second)


def test_tuning_moves_a_signal_to_centre(tmp_path):
    """Tuned 6 kHz up, a tone recorded at +6 kHz must come out at 0 Hz."""
    src = FileIQSource(_write(tmp_path, _tone(6e3, 48_000)))
    assert src.set_center_freq(CENTRE + 6e3) == CENTRE + 6e3
    src.next_block()                       # NCO settles nothing; any block will do
    block = np.concatenate([src.next_block() for _ in range(5)])
    assert abs(_peak_hz(block)) < 100
    assert src.dc_spike_offset_hz == pytest.approx(-6e3)


def test_tuning_is_clamped_to_the_recorded_span(tmp_path):
    src = FileIQSource(_write(tmp_path, _tone(0, 4800)))
    assert src.set_center_freq(CENTRE + 1e6) == pytest.approx(CENTRE + RATE / 2)


def test_loops_at_the_end_or_stops(tmp_path):
    path = _write(tmp_path, np.arange(2000) + 0j)
    looping = FileIQSource(path)
    blocks = [looping.next_block() for _ in range(5)]
    assert all(b is not None for b in blocks) and looping.stats["loops"] >= 1
    once = FileIQSource(path, loop=False)
    while once.next_block() is not None:
        pass
    assert once.finished


def test_seek_clears_what_was_buffered(tmp_path):
    src = FileIQSource(_write(tmp_path, np.arange(48_000) + 0j))
    src.next_block()
    src.seek(0.5)
    assert src.read_latest(10).size == 0
    assert src.position_s == pytest.approx(0.5)
    assert src.next_block()[0] == 24_000


def test_plays_in_real_time(tmp_path):
    src = FileIQSource(_write(tmp_path, _tone(1e3, 480_000)))
    src.start()
    try:
        time.sleep(0.5)
        played = src.stats["samples"]
    finally:
        src.stop()
    assert 0.3 * RATE < played < 0.8 * RATE


def test_pause_stops_the_clock(tmp_path):
    src = FileIQSource(_write(tmp_path, _tone(1e3, 480_000)))
    src.set_paused(True)
    src.start()
    try:
        time.sleep(0.2)
        assert src.stats["samples"] == 0
    finally:
        src.stop()


@pytest.mark.parametrize("meta, message", [
    ({"format": "int16"}, "unsupported format"),
    ({"sample_rate_hz": 0}, "sample_rate_hz"),
])
def test_unplayable_sidecars_are_refused(tmp_path, meta, message):
    path = _write(tmp_path, _tone(0, 100), **meta)
    with pytest.raises(ValueError, match=message):
        read_sidecar(path)


def test_missing_sidecar_is_refused(tmp_path):
    path = tmp_path / "bare.cf32"
    np.zeros(10, np.complex64).tofile(path)
    with pytest.raises(ValueError, match="no sidecar"):
        FileIQSource(path)


class _RingSource:
    """Just enough of an IQSource for the recorder: a ring and its description."""

    def __init__(self):
        self.ring = _Ring(400_000)
        self.sample_rate = RATE
        self.center_freq = CENTRE

        class caps:
            driver = "airspyhf"
        self.caps = caps

    def sequential_reader(self):
        from src.rgc_sdr.device.source import SequentialReader
        return SequentialReader(self.ring)


def test_plays_back_what_the_recorder_wrote(tmp_path):
    """The point of the sidecar: the recorder's own files must play."""
    src = _RingSource()
    rec = IQRecorder(tmp_path / "r.cf32", src, block_samples=4800)
    rec.start()
    data = _tone(3e3, 48_000).astype(np.complex64)
    src.ring.write(data)
    deadline = time.time() + 3
    while rec.samples_written < data.size and time.time() < deadline:
        time.sleep(0.02)
    rec.stop()
    play = FileIQSource(tmp_path / "r.cf32")
    assert play.center_freq == CENTRE and play.sample_rate == RATE
    out = np.concatenate([play.next_block() for _ in range(10)])
    assert np.allclose(out, data[:out.size])
