"""Recording tests: WAV output, raw IQ output, and the safety behaviours."""

import json
import wave

import numpy as np
import pytest

from src.rgc_sdr.device.source import DeviceCaps, FreqRange, IQSource, SequentialReader, _Ring
from src.rgc_sdr.recorder import AudioRecorder, IQRecorder, timestamp_name


def wait_for(predicate, timeout=3.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class FeedSource(IQSource):
    """A source whose ring we fill by hand, standing in for the radio."""

    def __init__(self, rate=768e3, centre=7.1e6, capacity=1_200_000):
        self._ring = _Ring(capacity)
        self._rate = rate
        self._centre = centre
        self._n = 0

    @property
    def caps(self):
        return DeviceCaps("airspyhf", "stub", "", (self._rate,),
                          (FreqRange(9e3, 31e6),), (), False, ("CF32",))

    @property
    def sample_rate(self):
        return self._rate

    @property
    def center_freq(self):
        return self._centre

    def start(self):
        pass

    def stop(self):
        pass

    def read_latest(self, n):
        return self._ring.read_latest(n)

    def sequential_reader(self):
        return SequentialReader(self._ring)

    def feed(self, count):
        block = (np.arange(self._n, self._n + count) * (1 + 1j)).astype(np.complex64)
        self._n += count
        self._ring.write(block)


# -- filenames ---------------------------------------------------------------

def test_timestamp_name_is_descriptive_and_sortable():
    name = timestamp_name(684_000.0, ".wav", "am")
    assert name.endswith("_0.6840MHz_am.wav")
    assert name[:4].isdigit()


def test_timestamp_name_without_a_mode():
    assert timestamp_name(7.1e6, ".cf32").endswith("_7.1000MHz.cf32")


# -- audio WAV ---------------------------------------------------------------

def test_audio_recording_is_a_playable_wav(tmp_path):
    path = tmp_path / "a.wav"
    rec = AudioRecorder(path, 48000)
    rec.start()
    tone = np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000).astype(np.float32)
    for _ in range(4):
        rec.submit(tone)
    assert wait_for(lambda: rec.bytes_written >= 4 * tone.size * 2)
    rec.stop()

    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 48000
        assert w.getnframes() == 4 * tone.size
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    # The tone must survive the round trip through 16-bit PCM.
    assert np.max(np.abs(data)) > 30000
    spectrum = np.abs(np.fft.rfft(data.astype(np.float64)))
    freqs = np.fft.rfftfreq(data.size, 1 / 48000)
    assert freqs[int(np.argmax(spectrum))] == pytest.approx(1000.0, abs=5.0)


def test_audio_recording_clips_rather_than_wrapping(tmp_path):
    """Out-of-range input must saturate, not alias round to the opposite sign."""
    path = tmp_path / "loud.wav"
    rec = AudioRecorder(path, 48000)
    rec.start()
    rec.submit(np.full(1000, 5.0, dtype=np.float32))
    rec.submit(np.full(1000, -5.0, dtype=np.float32))
    assert wait_for(lambda: rec.bytes_written >= 4000)
    rec.stop()
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert data[:1000].min() > 32000
    assert data[1000:].max() < -32000


def test_audio_recording_reports_seconds(tmp_path):
    rec = AudioRecorder(tmp_path / "a.wav", 48000)
    rec.start()
    rec.submit(np.zeros(48000, dtype=np.float32))
    assert wait_for(lambda: rec.seconds_recorded >= 0.99)
    rec.stop()
    assert rec.seconds_recorded == pytest.approx(1.0, abs=0.01)


def test_audio_recorder_creates_missing_directories(tmp_path):
    rec = AudioRecorder(tmp_path / "deep" / "nested" / "a.wav", 48000)
    rec.start()
    rec.stop()
    assert (tmp_path / "deep" / "nested" / "a.wav").is_file()


def test_submitting_after_stop_is_ignored(tmp_path):
    rec = AudioRecorder(tmp_path / "a.wav", 48000)
    rec.start()
    rec.stop()
    rec.submit(np.ones(100, dtype=np.float32))     # must not raise
    assert rec.bytes_written == 0


def test_full_queue_drops_rather_than_blocking(tmp_path):
    """The audio path must never be throttled by the disk."""
    rec = AudioRecorder(tmp_path / "a.wav", 48000, queue_blocks=2)
    rec._running.set()          # queue accepts, but nothing is draining it
    for _ in range(50):
        rec.submit(np.ones(1024, dtype=np.float32))
    assert rec.dropped_blocks > 0


# -- IQ ----------------------------------------------------------------------

def test_iq_recording_writes_complex64_and_a_sidecar(tmp_path):
    src = FeedSource()
    src.feed(200_000)
    rec = IQRecorder(tmp_path / "c.cf32", src, block_samples=16384)
    rec.start()
    src.feed(200_000)
    assert wait_for(lambda: rec.samples_written >= 100_000, timeout=5.0)
    rec.stop()

    raw = np.fromfile(tmp_path / "c.cf32", dtype=np.complex64)
    assert raw.size == rec.samples_written
    # Written in order: the stub's I component increments by one each sample.
    assert np.all(np.diff(raw.real[:1000]) == 1.0)

    meta = json.loads((tmp_path / "c.cf32.json").read_text())
    assert meta["format"] == "complex64"
    assert meta["sample_rate_hz"] == pytest.approx(768e3)
    assert meta["center_freq_hz"] == pytest.approx(7.1e6)
    assert meta["samples"] == raw.size
    assert meta["driver"] == "airspyhf"


def test_iq_sidecar_exists_even_if_the_capture_is_interrupted(tmp_path):
    """A killed capture must still leave a file you can interpret."""
    src = FeedSource()
    rec = IQRecorder(tmp_path / "c.cf32", src)
    rec.start()
    assert (tmp_path / "c.cf32.json").is_file()
    meta = json.loads((tmp_path / "c.cf32.json").read_text())
    assert meta["sample_rate_hz"] == pytest.approx(768e3)
    rec.stop()


def test_iq_recording_stops_itself_at_the_size_limit(tmp_path):
    src = FeedSource()
    src.feed(400_000)
    limit = 16384 * 8
    rec = IQRecorder(tmp_path / "c.cf32", src, block_samples=4096, max_bytes=limit)
    rec.start()
    for _ in range(10):
        src.feed(100_000)
        if wait_for(lambda: not rec.running, timeout=0.5):
            break
    rec.stop()
    assert rec.stopped_reason is not None and "limit" in rec.stopped_reason
    # Accurate to within one block, because the limit counts submitted bytes rather than
    # written ones -- a queue's worth of overshoot would be megabytes.
    one_block = 4096 * np.dtype(np.complex64).itemsize
    assert rec.bytes_written <= limit + one_block


def test_iq_recording_reports_duration(tmp_path):
    src = FeedSource(rate=768e3)
    src.feed(200_000)
    rec = IQRecorder(tmp_path / "c.cf32", src, block_samples=8192)
    rec.start()
    src.feed(120_000)          # comfortably more than 76800, in whole 8192 blocks
    assert wait_for(lambda: rec.samples_written >= 76800, timeout=5.0)
    rec.stop()
    assert rec.seconds_recorded == pytest.approx(rec.samples_written / 768e3, rel=1e-6)


def test_iq_recorder_is_independent_of_display_reads(tmp_path):
    """A lossy display read must not put a hole in the recording."""
    src = FeedSource()
    src.feed(100_000)
    rec = IQRecorder(tmp_path / "c.cf32", src, block_samples=8192)
    rec.start()
    for _ in range(6):
        src.feed(50_000)
        src.read_latest(30_000)        # the display peeking
    assert wait_for(lambda: rec.samples_written >= 200_000, timeout=5.0)
    rec.stop()
    raw = np.fromfile(tmp_path / "c.cf32", dtype=np.complex64)
    assert np.all(np.diff(raw.real) == 1.0), "recording has a discontinuity"


def test_stereo_audio_records_as_a_two_channel_wav(tmp_path):
    path = tmp_path / "s.wav"
    rec = AudioRecorder(path, 48000, channels=2)
    rec.start()
    frames = np.column_stack([np.full(4800, 0.5), np.full(4800, -0.5)]).astype(np.float32)
    rec.submit(frames)
    assert wait_for(lambda: rec.bytes_written >= frames.size * 2)
    rec.stop()
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 2 and w.getnframes() == 4800
        data = np.frombuffer(w.readframes(4800), dtype="<i2").reshape(-1, 2)
    assert data[:, 0].min() > 16000 and data[:, 1].max() < -16000
    assert rec.seconds_recorded == pytest.approx(0.1, abs=0.001)
