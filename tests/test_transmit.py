"""The transmit engine, with a stand-in microphone and sink: nothing is radiated."""

import numpy as np
import pytest

from src.rgc_sdr.transmit import TX_IQ_RATE, Transmitter


class FakeMic:
    def __init__(self, level=0.3):
        self.level = level
        self.level_dbfs = 20 * np.log10(level)
        self.started = self.stopped = False
        self.t = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def available(self):
        return 4096

    def read(self, n):
        t = (self.t + np.arange(n)) / 48e3
        self.t += n
        return self.level * np.sin(2 * np.pi * 1000 * t)


class FakeSink:
    def __init__(self, log):
        self.log = log
        self.written = 0

    def start(self):
        self.log.append("key")

    def write(self, iq):
        self.written += iq.size
        return iq.size

    def stop(self):
        self.log.append("unkey")


def test_dry_run_modulates_the_microphone_without_a_sink():
    tx = Transmitter("usb", mic=FakeMic())
    assert tx.dry_run
    produced = tx.pump_once()
    assert produced == 1024 * TX_IQ_RATE / 48e3
    assert 0 < tx.iq_peak <= 1.0


def test_iq_rate_suits_every_mode_and_the_hackrf():
    assert TX_IQ_RATE >= 2e6
    for mode in ("am", "nbfm", "wbfm", "usb", "lsb"):
        Transmitter(mode, mic=FakeMic())          # would raise on an unusable rate


def test_a_sink_gets_everything_and_is_unkeyed_before_the_mic_stops():
    log = []
    mic = FakeMic()
    sink = FakeSink(log)
    tx = Transmitter("nbfm", mic=mic, sink=sink)
    assert not tx.dry_run
    tx.start()
    tx.stop()
    assert log[0] == "key" and log[-1] == "unkey"
    assert mic.stopped and sink.written > 0


def test_cw_is_refused():
    with pytest.raises(ValueError, match="CW"):
        Transmitter("cw", mic=FakeMic())


def test_off_is_refused_with_a_hint():
    with pytest.raises(ValueError, match="choose a mode"):
        Transmitter("off", mic=FakeMic())


def test_timeout_expires_on_the_clock():
    now = [100.0]
    tx = Transmitter("am", mic=FakeMic(), timeout_s=180.0, clock=lambda: now[0])
    tx.start()
    now[0] += 179.0
    assert not tx.expired()
    now[0] += 2.0
    assert tx.expired()
    tx.stop()
    assert not tx.expired()


# -- the Soapy transmit sink, against a recording stand-in device ---------------------

from src.rgc_sdr.device.sink import SoapyIQSink  # noqa: E402
from src.rgc_sdr.device.source import LO_OFFSET_HZ, SOAPY_TX  # noqa: E402
from src.rgc_sdr.device.profiles import profile_for, caps_from_profile  # noqa: E402


class _Result:
    def __init__(self, ret):
        self.ret = ret


class RecordingDevice:
    """Records Soapy calls; accepts writes in chunks, as a real driver does."""

    def __init__(self, chunk=10_000):
        self.calls = []
        self.chunk = chunk
        self.sent = []

    def setSampleRate(self, d, ch, rate):
        self.calls.append(("rate", d, rate))

    def setFrequency(self, d, ch, hz):
        self.calls.append(("freq", d, hz))

    def setGain(self, d, ch, name, db):
        self.calls.append(("gain", d, name, db))

    def setupStream(self, d, fmt):
        self.calls.append(("setup", d, fmt))
        return "tx-stream"

    def activateStream(self, s):
        self.calls.append(("activate", s))

    def deactivateStream(self, s):
        self.calls.append(("deactivate", s))

    def closeStream(self, s):
        self.calls.append(("close", s))

    def writeStream(self, s, bufs, n, timeoutUs=0):
        take = min(n, self.chunk)
        self.sent.append(np.array(bufs[0][:take]))
        return _Result(take)


class FakeSource:
    def __init__(self, key="hackrf"):
        self.profile = profile_for(key)
        self.caps = caps_from_profile(self.profile)
        self.soapy_device = RecordingDevice()
        self.events = []

    def stop(self):
        self.events.append("rx stop")

    def start(self):
        self.events.append("rx start")


def test_half_duplex_pauses_the_receiver_around_transmit():
    src = FakeSource()
    sink = SoapyIQSink(src, 146.5e6, 2.4e6, gains={"VGA": 10.0})
    sink.start()
    assert src.events == ["rx stop"]
    sink.stop()
    assert src.events == ["rx stop", "rx start"]
    kinds = [c[0] for c in src.soapy_device.calls]
    assert kinds.index("setup") < kinds.index("activate") < kinds.index("deactivate")


def test_transmits_from_the_lo_offset_and_shifts_the_signal_onto_frequency():
    """HackRF: LO 200 kHz above, signal shifted down, so it lands on the wanted
    frequency and the LO leakage 200 kHz away."""
    src = FakeSource()
    sink = SoapyIQSink(src, 146.5e6, 2.4e6)
    sink.start()
    assert ("freq", SOAPY_TX, 146.5e6 + LO_OFFSET_HZ) in src.soapy_device.calls
    sink.write(np.ones(24_000, np.complex64))               # a carrier at baseband 0
    sent = np.concatenate(src.soapy_device.sent)
    spectrum = np.abs(np.fft.fft(sent))
    freqs = np.fft.fftfreq(sent.size, 1 / 2.4e6)
    assert freqs[np.argmax(spectrum)] == pytest.approx(-LO_OFFSET_HZ, abs=200.0)
    sink.stop()


def test_every_sample_is_written_even_when_the_driver_takes_it_in_pieces():
    src = FakeSource()
    sink = SoapyIQSink(src, 146.5e6, 2.4e6)
    sink.start()
    assert sink.write(np.ones(48_000, np.complex64)) == 48_000
    assert sum(b.size for b in src.soapy_device.sent) == 48_000
    sink.stop()


def test_tx_gains_are_applied_to_the_transmit_side():
    src = FakeSource()
    sink = SoapyIQSink(src, 146.5e6, 2.4e6, gains={"VGA": 12.0, "AMP": 0.0})
    sink.start()
    assert ("gain", SOAPY_TX, "VGA", 12.0) in src.soapy_device.calls
    sink.stop()


def test_a_receive_only_radio_cannot_make_a_sink():
    with pytest.raises(RuntimeError, match="cannot transmit"):
        SoapyIQSink(FakeSource("airspyhf"), 7.1e6, 2.4e6)
