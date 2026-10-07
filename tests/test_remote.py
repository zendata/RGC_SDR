"""The network radio server and its client (P9a, PLANNING.md 7q).

The server is run on localhost with a stand-in radio injected through its source factory
(tests only, as the UI tests inject stub sources): what is tested is the wire, the
decimation, the generations and the client's bookkeeping, not a driver.
"""

import threading
import time

import numpy as np
import pytest

from src.rgc_sdr import netserver
from src.rgc_sdr.device.profiles import profile_for
from src.rgc_sdr.device.remote import (
    RemoteIQSource, ServerAddress, list_radios, load_servers, parse_servers, remote_key,
    remote_profile, save_servers,
)
from src.rgc_sdr.device.remote_protocol import (
    MAX_LINK_RATE, MIN_LINK_RATE, caps_from_dict, caps_to_dict, decode_iq, encode_iq,
    link_plan,
)
from src.rgc_sdr.device.source import (
    DeviceCaps, FreqRange, GainElement, SequentialReader, SettingInfo, _Ring,
)


# -- the wire ------------------------------------------------------------------------


def _payload(frame):
    return frame[5:]


def test_iq_round_trip_keeps_a_quiet_block_precise():
    rng = np.random.default_rng(1)
    iq = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
    iq *= 1e-6                                           # about -120 dBFS
    generation, back = decode_iq(_payload(encode_iq(iq, 7)))
    assert generation == 7 and back.dtype == np.complex64
    error = np.abs(back - iq).max() / np.abs(iq).max()
    assert error < 1e-4                                  # 16 bits of the block's own peak


def test_an_empty_or_silent_block_survives():
    for iq in (np.zeros(0, np.complex64), np.zeros(10, np.complex64)):
        _, back = decode_iq(_payload(encode_iq(iq, 0)))
        assert back.size == iq.size and not np.any(back)


def test_caps_round_trip_without_the_transmitter():
    caps = DeviceCaps("hackrf", "HackRF One", "abc", (2e6, 4e6), (FreqRange(1e6, 6e9),),
                      (GainElement("LNA", 0, 40, 8),), True, ("CS8",), (1.75e6,),
                      (SettingInfo("bias_tx", "Antenna Bias"),))
    back = caps_from_dict(caps_to_dict(caps))
    assert back == caps and back.tx is None


def test_link_plan_halves_from_the_rate_nearest_the_default():
    plan = link_plan((912e3, 768e3, 456e3, 384e3, 256e3, 192e3), 768e3)
    assert plan[384e3] == (768e3, 2) and plan[768e3] == (768e3, 1)
    assert plan[456e3] == (912e3, 2)
    hackrf = link_plan((2e6, 4e6, 8e6, 10e6, 20e6), 4e6)
    assert hackrf[1e6] == (4e6, 4) and hackrf[500e3] == (4e6, 8)
    assert hackrf[625e3] == (10e6, 16)
    assert all(MIN_LINK_RATE <= r <= MAX_LINK_RATE for p in (plan, hackrf) for r in p)
    assert link_plan((20e6,), 20e6) == {}                # above what the Pi is asked to do


# -- servers and remote profiles ----------------------------------------------------------


def test_servers_are_parsed_saved_and_named(tmp_path):
    servers = parse_servers("radiopi, other.example.ts.net:55134,,")
    assert servers == [ServerAddress("radiopi"), ServerAddress("other.example.ts.net", 55134)]
    assert [s.name for s in servers] == ["radiopi", "other"]
    path = tmp_path / "servers.json"
    save_servers(servers, path)
    assert load_servers(path) == servers
    assert load_servers(tmp_path / "missing.json") == []


def test_a_remote_profile_is_its_own_radio():
    profile = remote_profile(profile_for("rtlsdr"), ServerAddress("radiopi"))
    assert profile.key == "rtlsdr@radiopi" == remote_key("rtlsdr", ServerAddress("radiopi"))
    assert profile.remote == ("radiopi", 55133, "rtlsdr") and profile.tx is None
    assert max(profile.sample_rates) <= MAX_LINK_RATE
    assert profile.covers(144e6) and "radiopi" in profile.label
    assert profile_for("rtlsdr@radiopi").remote is not None     # resolvable from its key
    assert profile_for("nonsense@radiopi") is None


def test_an_unreachable_server_lists_nothing():
    assert list_radios(ServerAddress("127.0.0.1", 9), use_cache=False) == []


# -- server and client together ----------------------------------------------------------


class StandInRadio:
    """What the server needs of SoapyIQSource: a tone `tone_hz` from centre, written to a
    ring by a thread at the radio rate."""

    instances = []

    def __init__(self, driver, serial=None, center_freq=100e6, tone_hz=20e3):
        self.caps = DeviceCaps(driver, "Stand-in", "", (2e6, 1e6), (FreqRange(24e6, 1.7e9),),
                               (GainElement("TUNER", 0, 49, 1),), True, ("CF32",))
        self.sample_rate = 2e6
        self.center_freq = float(center_freq)
        self.tone_hz = tone_hz
        self.gains = {"TUNER": 10.0}
        self.agc = False
        self.ppm = 0.0
        self.dc_spike_offset_hz = 0.0
        self.bandwidth = 0.0
        self._ring = _Ring(2_000_000)
        self._running = threading.Event()
        self._thread = None
        self._n = 0
        self.closed = False
        StandInRadio.instances.append(self)

    def set_sample_rate(self, hz):
        self.sample_rate = float(hz)
        self._ring = _Ring(2_000_000)
        return self.sample_rate

    def set_center_freq(self, hz, flush=True):
        self.center_freq = float(hz)
        if flush:
            self._ring.clear()
        return self.center_freq

    def sequential_reader(self):
        return SequentialReader(self._ring)

    def start(self):
        if self._thread is None:
            self._running.set()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        block = 20_000
        while self._running.is_set():
            n = np.arange(self._n, self._n + block)
            self._n += block
            tone = 0.01 * np.exp(2j * np.pi * self.tone_hz * n / self.sample_rate)
            self._ring.write(tone.astype(np.complex64))
            time.sleep(block / self.sample_rate)

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def close(self):
        self.stop()
        self.closed = True

    def get_gain(self, name):
        return self.gains[name]

    def set_gain(self, name, db):
        self.gains[name] = db

    def get_agc(self):
        return self.agc

    def set_agc(self, on):
        self.agc = on

    def read_setting(self, key):
        return False

    def set_ppm(self, ppm):
        self.ppm = ppm


@pytest.fixture
def server(monkeypatch):
    StandInRadio.instances.clear()
    monkeypatch.setattr(netserver, "list_radios",
                        lambda: [{"driver": "rtlsdr", "serial": "1", "label": "Stand-in"}])
    srv = netserver.Server("127.0.0.1", 0, source_factory=StandInRadio)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield ServerAddress("127.0.0.1", srv.address[1])
    srv.close()


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _tone_hz(iq, rate):
    spectrum = np.abs(np.fft.fft(iq * np.hanning(iq.size)))
    return np.fft.fftfreq(iq.size, 1 / rate)[np.argmax(spectrum)]


def test_a_remote_radio_streams_decimated_iq(server):
    profile = remote_profile(profile_for("rtlsdr"), server)
    src = RemoteIQSource(server, "rtlsdr", sample_rate=250e3, center_freq=145e6,
                         profile=profile)
    try:
        assert src.sample_rate == 250e3 and src.center_freq == 145e6
        assert src.caps.driver == profile.key and src.caps.tx is None
        assert set(src.caps.sample_rates) == {1e6, 500e3, 250e3, 125e3, 62.5e3}
        src.start()
        assert _wait(lambda: len(src.read_latest(16384)) == 16384)
        assert _tone_hz(src.read_latest(16384), src.sample_rate) == pytest.approx(20e3, abs=50)
        radio = StandInRadio.instances[-1]
        assert radio.sample_rate == 2e6                    # served from 2 MS/s by 8
    finally:
        src.close()
    assert _wait(lambda: StandInRadio.instances[-1].closed)


def test_a_flushing_retune_drops_what_came_before(server):
    src = RemoteIQSource(server, "rtlsdr", sample_rate=250e3, center_freq=145e6)
    try:
        src.start()
        assert _wait(lambda: src.stats["samples"] > 50_000)
        reader = src.sequential_reader()
        assert src.set_center_freq(146e6) == 146e6          # predicted at once
        assert src.set_center_freq(5e9) == 1.7e9            # clamped to the radio
        radio = StandInRadio.instances[-1]
        assert _wait(lambda: radio.center_freq == 1.7e9)
        assert _wait(lambda: reader.available() > 10_000)
        assert src.center_freq == 1.7e9
    finally:
        src.close()


def test_gains_and_agc_reach_the_radio(server):
    src = RemoteIQSource(server, "rtlsdr", center_freq=145e6)
    try:
        assert src.get_gain("TUNER") == 10.0
        src.set_gain("TUNER", 30.0)
        src.set_agc(True)
        src.set_ppm(-11.7)
        radio = StandInRadio.instances[-1]
        assert _wait(lambda: radio.gains["TUNER"] == 30.0 and radio.agc and radio.ppm == -11.7)
        assert src.get_gain("TUNER") == 30.0 and src.get_agc()
    finally:
        src.close()


def test_a_rate_change_is_served_and_the_ring_rebuilt(server):
    src = RemoteIQSource(server, "rtlsdr", sample_rate=250e3, center_freq=145e6)
    try:
        src.start()
        assert src.set_sample_rate(1e6) == 1e6
        assert _wait(lambda: len(src.read_latest(32768)) == 32768)
        assert _tone_hz(src.read_latest(32768), 1e6) == pytest.approx(20e3, abs=100)
    finally:
        src.close()


def test_listing_does_not_disturb_the_radio_in_use_but_opening_replaces_it(server):
    first = RemoteIQSource(server, "rtlsdr", center_freq=145e6)
    try:
        first.start()
        assert list_radios(server, use_cache=False) == [
            {"driver": "rtlsdr", "serial": "1", "label": "Stand-in"}]
        assert not StandInRadio.instances[0].closed
        second = RemoteIQSource(server, "rtlsdr", center_freq=146e6)
        try:
            assert StandInRadio.instances[0].closed          # the first client's radio
            assert second.center_freq == 146e6
        finally:
            second.close()
    finally:
        first._closed = True                                # do not reconnect
        first.close()
