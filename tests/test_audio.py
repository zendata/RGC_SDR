"""Audio plumbing: the FIFO, the sequential reader, and the sink's lifecycle.

The FIFO and reader are tested directly; the sink needs a sound device, so those tests
are marked `audio` and skip when none is usable.
"""

import numpy as np
import pytest

from src.rgc_sdr.audio import AudioFifo, audio_available
from src.rgc_sdr.device.source import SequentialReader, _Ring


# -- FIFO --------------------------------------------------------------------

def test_fifo_pull_returns_exactly_what_was_asked():
    fifo = AudioFifo(1000)
    fifo.push(np.arange(10, dtype=np.float32))
    out = fifo.pull(4)
    assert out.size == 4 and np.array_equal(out, [0, 1, 2, 3])
    assert len(fifo) == 6


def test_fifo_spans_block_boundaries():
    fifo = AudioFifo(1000)
    fifo.push(np.array([1, 2], dtype=np.float32))
    fifo.push(np.array([3, 4, 5], dtype=np.float32))
    assert np.array_equal(fifo.pull(5), [1, 2, 3, 4, 5])
    assert len(fifo) == 0


def test_fifo_pads_with_silence_and_counts_the_underrun():
    """A starved callback must still get a full buffer, and the gap must be recorded."""
    fifo = AudioFifo(1000)
    fifo.push(np.ones(3, dtype=np.float32))
    out = fifo.pull(8)
    assert out.size == 8
    assert np.array_equal(out[3:], np.zeros(5, dtype=np.float32))
    assert fifo.underrun_samples == 5


def test_fifo_drops_oldest_when_over_capacity():
    """Bounded: if the sink stops draining, audio is discarded rather than accumulating."""
    fifo = AudioFifo(10)
    for _ in range(5):
        fifo.push(np.ones(4, dtype=np.float32))
    assert len(fifo) <= 10
    assert fifo.dropped_samples > 0


def test_fifo_keeps_the_newest_audio_when_dropping():
    fifo = AudioFifo(8)
    fifo.push(np.full(8, 1.0, dtype=np.float32))
    fifo.push(np.full(8, 2.0, dtype=np.float32))
    assert np.allclose(fifo.pull(8), 2.0)


def test_fifo_clear():
    fifo = AudioFifo(100)
    fifo.push(np.ones(10, dtype=np.float32))
    fifo.clear()
    assert len(fifo) == 0


def test_fifo_ignores_empty_pushes():
    fifo = AudioFifo(100)
    fifo.push(np.zeros(0, dtype=np.float32))
    assert len(fifo) == 0


# -- sequential reader -------------------------------------------------------

def test_sequential_reader_is_gapless():
    ring = _Ring(1000)
    reader = SequentialReader(ring)
    ring.write(np.arange(30, dtype=np.complex64))
    assert np.array_equal(reader.read(10).real, np.arange(0, 10))
    assert np.array_equal(reader.read(10).real, np.arange(10, 20))
    assert np.array_equal(reader.read(10).real, np.arange(20, 30))


def test_sequential_reader_waits_rather_than_returning_partial_data():
    ring = _Ring(1000)
    reader = SequentialReader(ring)
    ring.write(np.arange(5, dtype=np.complex64))
    assert reader.read(10).size == 0      # not enough yet
    ring.write(np.arange(5, 10, dtype=np.complex64))
    assert np.array_equal(reader.read(10).real, np.arange(10))


def test_sequential_reader_starts_at_the_present_not_the_past():
    ring = _Ring(1000)
    ring.write(np.arange(100, dtype=np.complex64))
    reader = SequentialReader(ring)
    assert reader.available() == 0        # history before it existed is not replayed
    ring.write(np.arange(100, 110, dtype=np.complex64))
    assert reader.available() == 10


def test_sequential_reader_survives_wraparound():
    ring = _Ring(64)
    reader = SequentialReader(ring)
    expected = 0
    for start in range(0, 200, 16):
        ring.write(np.arange(start, start + 16, dtype=np.complex64))
        got = reader.read(16)
        assert np.array_equal(got.real, np.arange(expected, expected + 16))
        expected += 16
    assert reader.lost == 0


def test_sequential_reader_resyncs_and_counts_loss_when_lapped():
    """If the consumer falls behind the writer, it must skip ahead, not read torn data."""
    ring = _Ring(32)
    reader = SequentialReader(ring)
    ring.write(np.arange(200, dtype=np.complex64))   # laps the ring many times
    out = reader.read(16)
    assert out.size == 16
    assert reader.lost > 0
    # Whatever it returns must be real contiguous data, not a mix of old and new.
    assert np.all(np.diff(out.real) == 1)


def test_sequential_reader_resyncs_after_a_clear():
    ring = _Ring(128)
    reader = SequentialReader(ring)
    ring.write(np.arange(64, dtype=np.complex64))
    ring.clear()                                      # what a retune does
    ring.write(np.arange(1000, 1064, dtype=np.complex64))
    out = reader.read(64)
    assert out.size == 64
    assert out.real[0] == 1000.0, "stale pre-retune samples were served"


def test_skip_to_latest_discards_backlog():
    ring = _Ring(1000)
    reader = SequentialReader(ring)
    ring.write(np.arange(100, dtype=np.complex64))
    reader.skip_to_latest()
    assert reader.available() == 0


def test_display_and_audio_readers_are_independent():
    """The lossy display read must not disturb the audio cursor."""
    ring = _Ring(1000)
    reader = SequentialReader(ring)
    ring.write(np.arange(50, dtype=np.complex64))
    ring.read_latest(20)                              # display peeks at the newest
    assert np.array_equal(reader.read(50).real, np.arange(50))


# -- sink lifecycle ----------------------------------------------------------

audio_only = pytest.mark.skipif(not audio_available(), reason="no usable audio device")


@audio_only
def test_sink_plays_a_synthetic_am_signal():
    """End to end through a real output stream, without a radio."""
    import time

    from src.rgc_sdr.audio import AudioSink
    from src.rgc_sdr.device.source import DeviceCaps, FreqRange, IQSource

    class ToneSource(IQSource):
        """A modulated carrier on tap, standing in for the radio."""

        def __init__(self):
            self._ring = _Ring(1_200_000)
            self._n = 0
            self._caps = DeviceCaps(
                driver="stub", label="stub", serial="", sample_rates=(768e3,),
                freq_ranges=(FreqRange(1e3, 30e6),), gain_elements=(),
                has_agc=False, formats=("CF32",),
            )

        @property
        def caps(self):
            return self._caps

        @property
        def sample_rate(self):
            return 768e3

        @property
        def center_freq(self):
            return 7.1e6

        def start(self):
            pass

        def stop(self):
            pass

        def read_latest(self, n):
            return self._ring.read_latest(n)

        def sequential_reader(self):
            return SequentialReader(self._ring)

        def fill(self, count):
            t = (self._n + np.arange(count)) / 768e3
            self._n += count
            env = 1.0 + 0.5 * np.cos(2 * np.pi * 1000.0 * t)
            self._ring.write(env.astype(np.complex64))

    src = ToneSource()
    src.fill(200_000)
    sink = AudioSink(src, mode="am", volume=0.2, blocksize=512)
    sink.start()
    try:
        assert sink.running
        assert sink.audio_rate == pytest.approx(48e3)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            src.fill(40_000)
            time.sleep(0.04)
            if sink.stats["queued"] > 0:
                break
        assert sink.stats["queued"] > 0, "worker produced no audio"
        assert sink.stats["chain_errors"] == 0
    finally:
        sink.stop()
    assert not sink.running


@audio_only
def test_sink_rejects_an_unknown_mode():
    from src.rgc_sdr.audio import AudioSink
    from src.rgc_sdr.device.source import DeviceCaps, FreqRange, IQSource

    class Dummy(IQSource):
        @property
        def caps(self):
            return DeviceCaps("s", "s", "", (768e3,), (FreqRange(1e3, 30e6),), (), False, ("CF32",))

        @property
        def sample_rate(self):
            return 768e3

        @property
        def center_freq(self):
            return 7.1e6

        def start(self):
            pass

        def stop(self):
            pass

        def read_latest(self, n):
            return np.zeros(0, dtype=np.complex64)

    with pytest.raises(ValueError):
        AudioSink(Dummy()).set_mode("fmx")


# -- mute --------------------------------------------------------------------

def test_fifo_silence_still_counts_as_data():
    """Muting pushes zeros rather than nothing, so the stream stays fed."""
    fifo = AudioFifo(100)
    fifo.push(np.zeros(10, dtype=np.float32))
    assert len(fifo) == 10
    assert fifo.underrun_samples == 0


@audio_only
def test_mute_silences_output_but_not_the_recording(tmp_path):
    """Muting is a choice about the room; a silent recording would be a nasty surprise."""
    import time

    from src.rgc_sdr.audio import AudioSink
    from src.rgc_sdr.device.source import DeviceCaps, FreqRange, IQSource

    class ToneSource(IQSource):
        def __init__(self):
            self._ring = _Ring(1_200_000)
            self._n = 0

        @property
        def caps(self):
            return DeviceCaps("stub", "stub", "", (768e3,), (FreqRange(1e3, 30e6),),
                              (), False, ("CF32",))

        @property
        def sample_rate(self):
            return 768e3

        @property
        def center_freq(self):
            return 7.1e6

        def start(self):
            pass

        def stop(self):
            pass

        def read_latest(self, n):
            return self._ring.read_latest(n)

        def sequential_reader(self):
            return SequentialReader(self._ring)

        def fill(self, count):
            t = (self._n + np.arange(count)) / 768e3
            self._n += count
            env = 1.0 + 0.5 * np.cos(2 * np.pi * 1000.0 * t)
            self._ring.write(env.astype(np.complex64))

    src = ToneSource()
    src.fill(300_000)
    recorded = []
    sink = AudioSink(src, mode="am", volume=0.5, blocksize=512)
    sink.on_audio = recorded.append
    sink.start()
    try:
        sink.set_muted(True)
        assert sink.muted is True
        deadline = time.time() + 3.0
        while time.time() < deadline and len(recorded) < 3:
            src.fill(40_000)
            time.sleep(0.05)
        assert recorded, "worker produced nothing"
        # The tap saw real audio even though the output was muted.
        assert max(float(np.max(np.abs(b))) for b in recorded) > 1e-4
        assert sink.stats["muted"] is True
    finally:
        sink.stop()


@audio_only
def test_unmute_restores_output():
    from src.rgc_sdr.audio import AudioSink
    from src.rgc_sdr.device.source import DeviceCaps, FreqRange, IQSource

    class Quiet(IQSource):
        @property
        def caps(self):
            return DeviceCaps("s", "s", "", (768e3,), (FreqRange(1e3, 30e6),),
                              (), False, ("CF32",))

        @property
        def sample_rate(self):
            return 768e3

        @property
        def center_freq(self):
            return 7.1e6

        def start(self):
            pass

        def stop(self):
            pass

        def read_latest(self, n):
            return np.zeros(0, dtype=np.complex64)

        def sequential_reader(self):
            return SequentialReader(_Ring(1000))

    sink = AudioSink(Quiet(), mode="am")
    sink.set_muted(True)
    assert sink.muted is True
    sink.set_muted(False)
    assert sink.muted is False


# -- CW decoding in the sink -------------------------------------------------

@audio_only
def test_sink_decodes_cw_it_receives():
    """End to end: a keyed carrier through the real sink comes out as text."""
    import time

    from src.rgc_sdr.audio import AudioSink
    from src.rgc_sdr.device.source import DeviceCaps, FreqRange, IQSource
    from src.rgc_sdr.dsp.morse import text_to_morse

    class MorseSource(IQSource):
        """Keys a carrier with a repeating message."""

        def __init__(self, text="CQ TEST", wpm=20.0, fs=768e3):
            self._ring = _Ring(2_000_000)
            self._fs = fs
            self._phase = 0
            dot = int(1.2 / wpm * fs)
            plan = [(False, dot * 6)]
            for symbol in text_to_morse(text):
                if symbol == ".":
                    plan += [(True, dot), (False, dot)]
                elif symbol == "-":
                    plan += [(True, 3 * dot), (False, dot)]
                else:
                    plan += [(False, 2 * dot)]
            plan.append((False, dot * 8))
            self._plan = plan
            self._index = 0

        @property
        def caps(self):
            return DeviceCaps("stub", "stub", "", (768e3,), (FreqRange(1e3, 30e6),),
                              (), False, ("CF32",))

        @property
        def sample_rate(self):
            return self._fs

        @property
        def center_freq(self):
            return 7.02e6

        def start(self):
            pass

        def stop(self):
            pass

        def read_latest(self, n):
            return self._ring.read_latest(n)

        def sequential_reader(self):
            return SequentialReader(self._ring)

        @property
        def finished(self):
            return self._index >= len(self._plan)

        def pump(self):
            """Write one keying element and return how long it represents.

            The caller paces itself by that duration. Writing faster than real time
            overruns the ring and the sink's reader gets lapped, which garbles the very
            timing the decoder depends on.
            """
            on, count = self._plan[self._index]
            self._index += 1
            t = (self._phase + np.arange(count)) / self._fs
            self._phase += count
            block = (0.3 * np.exp(2j * np.pi * 0.0 * t)) if on else np.zeros(count, complex)
            self._ring.write(block.astype(np.complex64))
            return count / self._fs

    # VVV first, as an operator tunes up: the decoder starts mid-stream and its very
    # first element is clipped while the threshold primes, so the steady state is what
    # matters here. Decoding itself is covered thoroughly in test_morse.py.
    src = MorseSource(text="VVV CQ TEST")
    # Prime a little so the stream is never starved at startup.
    lead = sum(src.pump() for _ in range(2))
    sink = AudioSink(src, mode="cw", volume=0.05, blocksize=512)
    sink.start()
    try:
        deadline = time.time() + 20.0
        while not src.finished and time.time() < deadline:
            time.sleep(src.pump())          # paced at real time
        # Let the tail work through the chain.
        end = time.time() + 2.0
        while time.time() < end and "CQ TEST" not in sink.cw_text:
            time.sleep(0.05)
        assert "CQ TEST" in sink.cw_text, f"decoded {sink.cw_text!r}"
        assert sink.cw_wpm > 5.0
        assert sink.stats["chain_errors"] == 0
    finally:
        sink.stop()


@audio_only
def test_sink_does_not_decode_outside_cw():
    from src.rgc_sdr.audio import AudioSink
    from src.rgc_sdr.device.source import DeviceCaps, FreqRange, IQSource

    class Quiet(IQSource):
        @property
        def caps(self):
            return DeviceCaps("s", "s", "", (768e3,), (FreqRange(1e3, 30e6),),
                              (), False, ("CF32",))

        @property
        def sample_rate(self):
            return 768e3

        @property
        def center_freq(self):
            return 7.02e6

        def start(self):
            pass

        def stop(self):
            pass

        def read_latest(self, n):
            return np.zeros(0, dtype=np.complex64)

        def sequential_reader(self):
            return SequentialReader(_Ring(1000))

    sink = AudioSink(Quiet(), mode="am")
    sink.start()
    try:
        assert sink.cw_text == ""
        assert sink.cw_wpm == 0.0
    finally:
        sink.stop()
