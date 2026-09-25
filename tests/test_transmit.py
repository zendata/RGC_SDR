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
