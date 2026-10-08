"""P12: SigMF recordings, playback of other tools' formats, and the replay buffer."""

import json
import time

import numpy as np
import pytest

from src.rgc_sdr.device.playback import FileIQSource, overview
from src.rgc_sdr.device.sigmf import read_recording, sigmf_meta, write_sigmf
from src.rgc_sdr.device.source import DeviceCaps, FreqRange, IQSource, SequentialReader, _Ring
from src.rgc_sdr.recorder import IQRecorder


def _tone(n, rate=1e6, f=100e3, amp=0.5):
    return (amp * np.exp(2j * np.pi * f * np.arange(n) / rate)).astype(np.complex64)


def test_write_then_read_a_sigmf_recording(tmp_path):
    iq = _tone(10_000)
    path = write_sigmf(tmp_path / "a.sigmf-data", iq, 1e6, 145e6, hw="Test radio",
                       extras={"note": "x"})
    meta = json.loads((tmp_path / "a.sigmf-meta").read_text())
    assert meta["global"]["core:datatype"] == "cf32_le"
    assert meta["global"]["core:sample_rate"] == 1e6
    assert meta["global"]["core:version"] == "1.0.0"
    assert meta["global"]["core:hw"] == "Test radio"
    assert meta["global"]["rgc_sdr:note"] == "x"
    assert meta["captures"][0]["core:frequency"] == 145e6
    assert meta["captures"][0]["core:datetime"].endswith("Z")
    for either in (path, tmp_path / "a.sigmf-meta"):
        rec = read_recording(either)
        assert (rec.sample_rate, rec.center_freq, rec.samples) == (1e6, 145e6, 10_000)
        assert np.array_equal(rec.to_complex(rec.open()), iq)


@pytest.mark.parametrize("datatype,dtype,scale,offset", [
    ("ci16_le", "<i2", 32767, 0.0), ("ci8", "i1", 127, 0.0), ("cu8", "u1", 127, 127.5)])
def test_integer_recordings_from_other_tools_play(tmp_path, datatype, dtype, scale, offset):
    iq = _tone(4096)
    pairs = np.empty(2 * iq.size)
    pairs[0::2], pairs[1::2] = iq.real, iq.imag
    (pairs * scale + offset).round().astype(dtype).tofile(tmp_path / "b.sigmf-data")
    meta = sigmf_meta(1e6, 100e6)
    meta["global"]["core:datatype"] = datatype
    (tmp_path / "b.sigmf-meta").write_text(json.dumps(meta))
    rec = read_recording(tmp_path / "b.sigmf-data")
    back = rec.to_complex(rec.open())
    assert back.dtype == np.complex64
    assert np.abs(back - iq).max() < 0.02                  # within the format's resolution


def test_unplayable_recordings_say_why(tmp_path):
    with pytest.raises(ValueError, match="no x.sigmf-meta"):
        read_recording(tmp_path / "x.sigmf-data")
    meta = sigmf_meta(1e6, 100e6)
    meta["global"]["core:datatype"] = "rf32_le"
    (tmp_path / "y.sigmf-meta").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unsupported SigMF datatype"):
        read_recording(tmp_path / "y.sigmf-meta")
    (tmp_path / "z.sigmf-meta").write_text(json.dumps(sigmf_meta(1e6, 100e6)))
    with pytest.raises(ValueError, match="no z.sigmf-data"):
        read_recording(tmp_path / "z.sigmf-meta")


def test_playback_plays_sigmf_and_keeps_its_speed(tmp_path):
    write_sigmf(tmp_path / "c.sigmf-data", _tone(200_000), 1e6, 100e6)
    src = FileIQSource(tmp_path / "c.sigmf-data", loop=False)
    assert src.duration_s == pytest.approx(0.2)
    block = src.next_block()
    assert block.dtype == np.complex64 and block.size == 20_000
    src.seek(0.1)
    assert src.position_s == pytest.approx(0.1)
    src.set_speed(4.0)
    src.start()
    time.sleep(0.15)
    src.stop()
    assert src.finished                                   # 0.1 s left, at 4x
    picture = overview(src.recording, columns=50, bins=64)
    assert picture.shape == (64, 50)
    assert np.argmax(picture.mean(axis=1)) == pytest.approx(32 + 64 * 100e3 / 1e6, abs=1)


class _Stub(IQSource):
    def __init__(self, rate=1e6):
        self._rate, self._ring = rate, _Ring(200_000)
        self._caps = DeviceCaps("stub", "Stub", "", (rate,), (FreqRange(1e6, 2e9),), (),
                                False, ("CF32",))

    caps = property(lambda self: self._caps)
    sample_rate = property(lambda self: self._rate)
    center_freq = property(lambda self: 100e6)

    def start(self):
        pass

    def stop(self):
        pass

    def read_latest(self, n):
        return self._ring.read_latest(n)

    def sequential_reader(self):
        return SequentialReader(self._ring)

    def push(self, block):                                # as a reader thread does
        self._ring.write(block)
        self._remember(block)


def test_the_replay_buffer_keeps_the_last_seconds():
    src = _Stub()
    assert src.history_seconds == 0.0 and src.history(1).size == 0
    assert src.set_history_seconds(2.0) == pytest.approx(2.0)
    for i in range(30):                                   # 3 s, of which 2 are kept
        src.push(np.full(100_000, i, np.complex64))
    assert src.history_filled_s == pytest.approx(2.0)
    last = src.history(0.5)
    assert last.size == 500_000
    assert last[-1] == 29 and last[0] == 25
    assert src.history().size == 2_000_000


def test_the_replay_buffer_is_capped():
    from src.rgc_sdr.device.source import REPLAY_MAX_BYTES

    src = _Stub(rate=10e6)
    kept = src.set_history_seconds(600)
    assert kept == pytest.approx(REPLAY_MAX_BYTES / 8 / 10e6)
    assert src.set_history_seconds(0) == 0.0


def test_the_recorder_writes_sigmf_metadata(tmp_path):
    src = _Stub()
    rec = IQRecorder(tmp_path / "d.sigmf-data", src, block_samples=4096)
    rec.start()
    for _ in range(5):
        src.push(_tone(10_000))
        time.sleep(0.02)
    rec.stop()
    meta = json.loads((tmp_path / "d.sigmf-meta").read_text())
    assert meta["global"]["core:datatype"] == "cf32_le"
    assert meta["global"]["rgc_sdr:driver"] == "stub"
    assert meta["global"]["rgc_sdr:samples"] == rec.samples_written > 0
    assert read_recording(tmp_path / "d.sigmf-data").samples == rec.samples_written
