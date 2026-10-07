"""The network radio server and its client (P9, PLANNING.md 7q).

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
    MAX_IQ_RATE, caps_from_dict, caps_to_dict, decode_iq, decode_spectrum, encode_iq,
    encode_spectrum, iq_rate_for, needs_recentre,
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


def test_the_iq_window_is_the_radio_rate_halved_to_at_most_400k():
    assert iq_rate_for(2.048e6) == (256e3, 8)
    assert iq_rate_for(768e3) == (384e3, 2)
    assert iq_rate_for(4e6) == (250e3, 16)
    assert iq_rate_for(192e3) == (192e3, 1)
    assert all(iq_rate_for(r)[0] <= MAX_IQ_RATE for r in (912e3, 2.4e6, 6e6, 10e6))


def test_tuning_moves_the_radio_only_near_the_edge_of_its_span():
    # 2.048 MS/s, 256 kS/s window: reach is 1.024M - 128k - 102.4k = 793.6 kHz.
    assert not needs_recentre(100e6, 2.048e6, 256e3, 100.7e6)
    assert needs_recentre(100e6, 2.048e6, 256e3, 100.8e6)
    assert needs_recentre(100e6, 2.048e6, 256e3, 99.1e6)
    assert needs_recentre(7e6, 256e3, 256e3, 7.001e6)      # window = span: always


def test_a_spectrum_line_round_trips_within_its_step():
    dbfs = np.linspace(-130, -40, 4096).astype(np.float32)
    centre, span, back = decode_spectrum(encode_spectrum(145e6, 2.048e6, dbfs)[5:])
    assert (centre, span) == (145e6, 2.048e6) and back.size == 4096
    assert np.abs(back - dbfs).max() <= 90 / 255 / 2 + 1e-3
    flat = np.full(16, -100.0, np.float32)
    assert np.allclose(decode_spectrum(encode_spectrum(1e6, 1e6, flat)[5:])[2], -100.0)


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
    assert 2.048e6 in profile.sample_rates and profile.default_rate == 2.048e6
    hackrf = remote_profile(profile_for("hackrf"), ServerAddress("radiopi"))
    assert max(hackrf.sample_rates) <= 10e6
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
        self.retunes = 0
        StandInRadio.instances.append(self)

    def set_sample_rate(self, hz):
        self.sample_rate = float(hz)
        self._ring = _Ring(2_000_000)
        return self.sample_rate

    def set_center_freq(self, hz, flush=True):
        self.center_freq = self.caps.clamp_freq(float(hz))
        self.retunes += 1
        if flush:
            self._ring.clear()
        return self.center_freq

    def read_latest(self, n):
        return self._ring.read_latest(n)

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


def _lines(src, timeout=5.0):
    """Spectrum lines, waiting until some arrive."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lines = src.take_spectrum_lines()
        if lines:
            return lines
        time.sleep(0.02)
    return []


def _peak_hz(line):
    n = line.dbfs.size
    return line.centre_hz + (int(np.argmax(line.dbfs)) - n // 2) * line.span_hz / n


def test_a_remote_radio_sends_its_whole_span_and_an_iq_window(server):
    profile = remote_profile(profile_for("rtlsdr"), server)
    src = RemoteIQSource(server, "rtlsdr", sample_rate=2e6, center_freq=145e6,
                         profile=profile)
    try:
        assert src.span_rate == 2e6 and src.sample_rate == 250e3     # 2 MS/s by 8
        assert src.center_freq == 145e6
        assert src.caps.driver == profile.key and src.caps.tx is None
        assert set(src.caps.sample_rates) == {2e6, 1e6}
        src.set_display(1024, 25, 1)
        src.start()
        line = _lines(src)[-1]
        assert (line.centre_hz, line.span_hz, line.dbfs.size) == (145e6, 2e6, 1024)
        assert _peak_hz(line) == pytest.approx(145.02e6, abs=2e6 / 1024)
        assert _wait(lambda: len(src.read_latest(16384)) == 16384)
        assert _tone_hz(src.read_latest(16384), src.sample_rate) == pytest.approx(20e3, abs=50)
    finally:
        src.close()
    assert _wait(lambda: StandInRadio.instances[-1].closed)


def test_tuning_inside_the_span_moves_only_the_window(server):
    src = RemoteIQSource(server, "rtlsdr", sample_rate=2e6, center_freq=145e6)
    try:
        src.start()
        radio = StandInRadio.instances[-1]
        retunes = radio.retunes
        reader = src.sequential_reader()
        assert src.set_center_freq(145.02e6) == 145.02e6       # onto the tone
        assert _wait(lambda: src._state.get("freq") == 145.02e6)
        assert radio.center_freq == 145e6 and radio.retunes == retunes
        reader.skip_to_latest()
        assert _wait(lambda: reader.available() >= 16384)
        iq = reader.read(16384)
        assert abs(_tone_hz(iq, src.sample_rate)) < 50              # the tone at the centre
        line = _lines(src)[-1]
        assert line.centre_hz == 145e6                              # the picture stays
    finally:
        src.close()


def test_tuning_past_the_edge_moves_the_radio(server):
    src = RemoteIQSource(server, "rtlsdr", sample_rate=2e6, center_freq=145e6)
    try:
        src.start()
        radio = StandInRadio.instances[-1]
        src.set_center_freq(146.5e6)
        assert _wait(lambda: radio.center_freq == 146.5e6)
        assert _wait(lambda: any(l.centre_hz == 146.5e6 for l in src.take_spectrum_lines()))
        assert src.set_center_freq(5e9) == 1.7e9                    # clamped to the radio
        assert _wait(lambda: radio.center_freq == 1.7e9)
    finally:
        src.close()


def test_zoom_centres_the_lines_on_the_tuned_frequency(server):
    src = RemoteIQSource(server, "rtlsdr", sample_rate=2e6, center_freq=145e6)
    try:
        src.start()
        src.set_center_freq(145.1e6)
        src.set_display(1024, 25, 4)
        assert _wait(lambda: any(l.span_hz == 500e3 and l.centre_hz == 145.1e6
                                 for l in src.take_spectrum_lines()))
        line = _lines(src)[-1]
        assert _peak_hz(line) == pytest.approx(145.02e6, abs=500e3 / 1024 * 2)
        assert src.display_span == 500e3
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


def test_a_rate_change_moves_the_span_and_the_window(server):
    src = RemoteIQSource(server, "rtlsdr", sample_rate=2e6, center_freq=145e6)
    try:
        src.start()
        assert src.set_sample_rate(1e6) == 1e6
        assert src.span_rate == 1e6 and src.sample_rate == 250e3    # 1 MS/s by 4
        assert _wait(lambda: len(src.read_latest(32768)) == 32768)
        assert _tone_hz(src.read_latest(32768), 250e3) == pytest.approx(20e3, abs=100)
        assert _wait(lambda: any(l.span_hz == 1e6 for l in src.take_spectrum_lines()))
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
